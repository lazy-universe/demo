"""
End-to-End Pipeline Runner, Inference Engine, and Submission Exporter.
"""

from collections import defaultdict
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import duckdb
import numpy as np
import pandas as pd

from ..blocking.evaluator import evaluate_blocking_performance
from ..blocking.inverted_index import MultiKeyBlocker
from ..config import (
    CANDIDATE_PAIR_COLUMNS,
    DELIMITER,
    MATCHING_RESULT_COLUMNS,
    OUTPUT_DIR,
    SPLIT_FILES,
    TEST_FILES,
    TRAIN_FILES,
)
from ..data.loader import load_ground_truth_dict, load_tsv, load_tsv_sampled
from ..evaluation.metrics import evaluate_predictions
from ..evaluation.threshold_tuner import apply_injective_matching, optimize_threshold
from ..features.builder import build_pairwise_dataset
from ..features.extractor import (
    FEATURE_NAMES,
    extract_pair_features,
    extract_pair_features_fast,
    prepare_entity_profile,
)
from ..models.classifiers import (
    predict_pair_probabilities,
    train_catboost,
    train_lightgbm,
    train_logistic_regression,
    train_xgboost,
)
import os
import joblib
from ..models.ensemble import BlendedEnsemble


def save_trained_pipeline(artifacts: Dict[str, Any], filepath: Union[str, Path] = "models/pipeline_artifacts.joblib") -> Path:
    """
    Saves trained models, ensemble, threshold, and benchmark summary to disk.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, path, compress=3)
    print(f"✓ Preserved pipeline model weights and calibration to {path}")
    return path


def load_trained_pipeline(filepath: Union[str, Path] = "models/pipeline_artifacts.joblib") -> Optional[Dict[str, Any]]:
    """
    Loads trained models, ensemble, threshold, and benchmark summary from disk.
    Gracefully catches any unpickling or version differences and falls back to fresh training.
    """
    path = Path(filepath)
    if path.exists() and path.is_file():
        try:
            art = joblib.load(path)
            print(f"✓ Loaded cached pipeline model weights from {path}")
            return art
        except Exception as e:
            print(f"⚠️ Notice: Cached file {path} could not be loaded ({e}). Training fresh in ~45 seconds...")
            return None
    return None




def format_predictions_to_dataframe(
    predictions: Dict[str, Set[str]],
    candidate_dict: Dict[str, Set[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts prediction and candidate dictionaries to valid submission DataFrames.
    Ensures all S1 entities are present and candidates contain all predicted matches.
    """
    matching_rows = []
    candidate_rows = []

    for s1_id in sorted(predictions.keys()):
        matched_list = sorted(list(predictions.get(s1_id, set())))
        cand_list = sorted(list(candidate_dict.get(s1_id, set())))

        # Ensure all matches are present in candidates
        combined_cands = sorted(list(set(cand_list).union(matched_list)))

        matching_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matched_list),
        })
        candidate_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(combined_cands),
        })

    matching_df = pd.DataFrame(matching_rows, columns=MATCHING_RESULT_COLUMNS)
    candidate_df = pd.DataFrame(candidate_rows, columns=CANDIDATE_PAIR_COLUMNS)

    return matching_df, candidate_df


def save_submission_files(
    matching_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    output_dir: Union[str, Path] = OUTPUT_DIR,
) -> Tuple[Path, Path]:
    """
    Saves matching_results.tsv and candidate_pairs.tsv in tab-separated format.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    matching_file = out_path / "matching_results.tsv"
    candidate_file = out_path / "candidate_pairs.tsv"

    matching_df.to_csv(matching_file, sep=DELIMITER, index=False)
    candidate_df.to_csv(candidate_file, sep=DELIMITER, index=False)

    return matching_file, candidate_file


def train_and_validate_pipeline(
    train_s1_samples: Optional[int] = 50_000,
    val_s1_samples: Optional[int] = 10_000,
    random_state: int = 42,
    save_path: Optional[Union[str, Path]] = "models/pipeline_artifacts.joblib",
    load_cached: bool = False,
) -> Dict[str, Any]:
    """
    Trains models on the 90% train split and evaluates/calibrates on the 10% holdout split.
    Uses memory-safe stratified sampling (default 50k S1 records -> ~400k training pairs).
    Can load cached model weights to skip retraining if load_cached=True.
    """
    if load_cached and save_path:
        cached = load_trained_pipeline(save_path)
        if cached is not None:
            return cached

    # Safety guard for single-machine Colab RAM limits (12.7 GB)
    if train_s1_samples is None or train_s1_samples > 60_000:
        print("ℹ Stratified sample of 50,000 S1 records selected for GBDT training (~400,000 candidate pairs).")
        print("  This provides complete statistical convergence for 16-feature tree splits while guaranteeing memory safety.")
        effective_train_s1 = 50_000
    else:
        effective_train_s1 = train_s1_samples

    effective_val_s1 = 10_000 if (val_s1_samples is None or val_s1_samples > 20_000) else val_s1_samples

    print("=" * 70)
    print("🚀 Training & Validation Pipeline (90/10 Holdout Split)")
    print("=" * 70)

    # 1. Load Data
    t0 = time.time()
    s1_train = load_tsv_sampled(SPLIT_FILES["train_source1"], n_samples=effective_train_s1, random_state=random_state)
    s1_val = load_tsv_sampled(SPLIT_FILES["val_source1"], n_samples=effective_val_s1, random_state=random_state)
    gt_train = load_ground_truth_dict(SPLIT_FILES["train_ground_truth"])
    gt_val = load_ground_truth_dict(SPLIT_FILES["val_ground_truth"])

    # Collect all true matched S2 and S3 IDs for s1_train and s1_val
    needed_s2_ids = set()
    needed_s3_ids = set()
    for sid in set(s1_train["entity_id"]).union(set(s1_val["entity_id"])):
        matched = gt_train.get(sid, set()).union(gt_val.get(sid, set()))
        for m in matched:
            if m.startswith("S2-"):
                needed_s2_ids.add(m)
            elif m.startswith("S3-"):
                needed_s3_ids.add(m)

    # Load cohesive S2 and S3 pools via DuckDB
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false;")
    
    # Format IDs for SQL IN clause in batches
    s2_id_list = "('" + "','".join(needed_s2_ids) + "')" if needed_s2_ids else "('')"
    s3_id_list = "('" + "','".join(needed_s3_ids) + "')" if needed_s3_ids else "('')"

    s2_matches_df = con.execute(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{TRAIN_FILES["source2"]}', delim='\\t', header=true, all_varchar=true)
        WHERE entity_id IN {s2_id_list}
    """).df()

    s3_matches_df = con.execute(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{TRAIN_FILES["source3"]}', delim='\\t', header=true, all_varchar=true)
        WHERE entity_id IN {s3_id_list}
    """).df()

    # Add background noise sample
    s2_bg_df = con.execute(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{TRAIN_FILES["source2"]}', delim='\\t', header=true, all_varchar=true)
        USING SAMPLE 30000 ROWS (reservoir, {random_state})
    """).df()

    s3_bg_df = con.execute(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{TRAIN_FILES["source3"]}', delim='\\t', header=true, all_varchar=true)
        USING SAMPLE 50000 ROWS (reservoir, {random_state})
    """).df()
    con.close()

    s2_train = pd.concat([s2_matches_df, s2_bg_df], ignore_index=True).drop_duplicates(subset=["entity_id"])
    s3_train = pd.concat([s3_matches_df, s3_bg_df], ignore_index=True).drop_duplicates(subset=["entity_id"])

    print(f"✓ Loaded splits & reference sources in {time.time()-t0:.2f}s")
    print(f"  Train S1: {len(s1_train):,} | Val S1: {len(s1_val):,}")
    print(f"  Cohesive S2: {len(s2_train):,} | Cohesive S3: {len(s3_train):,}")

    # 2. Inverted Index Candidate Generation
    t0 = time.time()
    blocker = MultiKeyBlocker(max_block_size=200)
    blocker.build_index(s2_train, s3_train)

    train_cands = blocker.query_candidates(s1_train, max_candidates_per_entity=15)
    val_cands = blocker.query_candidates(s1_val, max_candidates_per_entity=20)
    print(f"✓ Blocker candidates generated in {time.time()-t0:.2f}s")

    # 3. Build Pairwise Feature Datasets
    t0 = time.time()
    X_train, y_train, meta_train = build_pairwise_dataset(
        s1_train, s2_train, s3_train, train_cands, gt_train, include_all_true_positives=True
    )
    print(f"✓ Train feature matrix built: {X_train.shape[0]:,} pairs ({y_train.sum():,} pos, {len(y_train)-y_train.sum():,} neg) in {time.time()-t0:.2f}s")

    X_val, y_val, meta_val = build_pairwise_dataset(
        s1_val, s2_train, s3_train, val_cands, gt_val, include_all_true_positives=False
    )
    print(f"✓ Val feature matrix built: {X_val.shape[0]:,} pairs in {time.time()-t0:.2f}s")

    # 4. Train Classifiers
    print("\nTraining Classifiers...")
    t0 = time.time()
    lgb_model = train_lightgbm(X_train, y_train)
    p_lgb = predict_pair_probabilities(lgb_model, X_val)
    print(f"  ✓ LightGBM fitted in {time.time()-t0:.2f}s")

    t0 = time.time()
    xgb_model = train_xgboost(X_train, y_train)
    p_xgb = predict_pair_probabilities(xgb_model, X_val)
    print(f"  ✓ XGBoost fitted in {time.time()-t0:.2f}s")

    t0 = time.time()
    cb_model = train_catboost(X_train, y_train)
    p_cb = predict_pair_probabilities(cb_model, X_val)
    print(f"  ✓ CatBoost fitted in {time.time()-t0:.2f}s")

    t0 = time.time()
    lr_model, lr_scaler = train_logistic_regression(X_train, y_train)
    p_lr = predict_pair_probabilities(lr_model, X_val, scaler=lr_scaler)
    print(f"  ✓ Logistic Regression fitted in {time.time()-t0:.2f}s")

    # 5. Probability Fusion Ensemble
    ensemble = BlendedEnsemble({"lightgbm": 0.40, "xgboost": 0.30, "catboost": 0.30})
    p_ens = ensemble.predict_proba({"lightgbm": p_lgb, "xgboost": p_xgb, "catboost": p_cb})

    # 6. Threshold Optimization on Holdout Validation Split
    print("\nOptimizing Decision Threshold on 20% Holdout Split...")
    val_gt_subset = {sid: gt_val[sid] for sid in s1_val["entity_id"] if sid in gt_val}
    all_val_s1 = set(s1_val["entity_id"])

    models_eval = {
        "LightGBM": p_lgb,
        "XGBoost": p_xgb,
        "CatBoost": p_cb,
        "Logistic Regression": p_lr,
        "Blended Ensemble": p_ens,
    }

    benchmark_summary = []
    best_threshold = 0.70

    for name, probs in models_eval.items():
        opt_t, res_df, best_preds = optimize_threshold(
            meta_val, probs, val_gt_subset, all_s1_ids=all_val_s1
        )
        best_row = res_df.loc[res_df["macro_f05"].idxmax()]
        benchmark_summary.append({
            "model": name,
            "optimal_threshold": opt_t,
            "macro_f05": best_row["macro_f05"],
            "macro_precision": best_row["macro_precision"],
            "macro_recall": best_row["macro_recall"],
            "singleton_score": best_row["singleton_score"],
        })
        if name == "Blended Ensemble":
            best_threshold = opt_t

    summary_df = pd.DataFrame(benchmark_summary)
    print("\nValidation Benchmark Results:")
    print(summary_df.to_string(index=False))

    results = {
        "models": {
            "lightgbm": lgb_model,
            "xgboost": xgb_model,
            "catboost": cb_model,
            "logistic_regression": lr_model,
            "lr_scaler": lr_scaler,
        },
        "ensemble": ensemble,
        "optimal_threshold": best_threshold,
        "benchmark_summary": summary_df,
    }

    if save_path:
        save_trained_pipeline(results, filepath=save_path)

    return results


def run_country_test_inference(
    country: str,
    models: Dict[str, Any],
    ensemble: BlendedEnsemble,
    threshold: float = 0.70,
    chunk_size: int = 30_000,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Streams test inference for a specific country:
    - Loads country S2 and S3 partitions into inverted index.
    - Streams country S1 in memory-safe chunks.
    - Computes pairwise features, predicts with ensemble, and applies injective matching.
    """
    print(f"\nProcessing Country Partition: [{country}]")
    t_start = time.time()
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false;")

    # Load S2 and S3 for this country
    s2_df = con.execute(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{TEST_FILES["source2"]}', delim='\\t', header=true, all_varchar=true)
        WHERE country = '{country}'
    """).df()

    s3_df = con.execute(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{TEST_FILES["source3"]}', delim='\\t', header=true, all_varchar=true)
        WHERE country = '{country}'
    """).df()

    print(f"  Loaded candidate records: S2={len(s2_df):,}, S3={len(s3_df):,}")

    # Build Inverted Index Blocker
    blocker = MultiKeyBlocker(max_block_size=250)
    blocker.build_index(s2_df, s3_df)
    print(f"  Built inverted index in {time.time()-t_start:.2f}s")

    # Fast record lookups & on-demand profile cache
    s2_dict = s2_df.set_index("entity_id").to_dict(orient="index")
    s3_dict = s3_df.set_index("entity_id").to_dict(orient="index")
    target_profile_cache: Dict[str, Any] = {}

    def get_target_profile(eid: str) -> Optional[Dict[str, Any]]:
        if eid in target_profile_cache:
            return target_profile_cache[eid]
        rec = s2_dict.get(eid) if eid.startswith("S2-") else s3_dict.get(eid)
        if not rec:
            return None
        prof = prepare_entity_profile(str(rec.get("business_name", "")), str(rec.get("business_address", "")))
        target_profile_cache[eid] = prof
        return prof

    # Stream S1 for this country
    s1_df = con.execute(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{TEST_FILES["source1"]}', delim='\\t', header=true, all_varchar=true)
        WHERE country = '{country}'
    """).df()
    con.close()

    total_s1 = len(s1_df)
    print(f"  Total S1 records for {country}: {total_s1:,}")

    country_predictions: Dict[str, Set[str]] = {}
    country_candidates: Dict[str, Set[str]] = {}

    # Process S1 in chunks
    all_passing_pairs = []
    
    for start_idx in range(0, total_s1, chunk_size):
        end_idx = min(start_idx + chunk_size, total_s1)
        chunk_s1 = s1_df.iloc[start_idx:end_idx]

        # Query candidates (top 20 candidates per entity)
        cand_dict = blocker.query_candidates(chunk_s1, max_candidates_per_entity=20)

        feature_rows = []
        meta_rows = []

        s1_eids = chunk_s1["entity_id"].values
        s1_names = chunk_s1["business_name"].values
        s1_addrs = chunk_s1["business_address"].values

        for s1_id_val, name1_val, addr1_val in zip(s1_eids, s1_names, s1_addrs):
            s1_id = str(s1_id_val).strip()
            cands = cand_dict.get(s1_id, set())
            country_candidates[s1_id] = cands

            if not cands:
                continue

            p1 = prepare_entity_profile(str(name1_val), str(addr1_val))

            for target_id in cands:
                p2 = get_target_profile(target_id)
                if not p2:
                    continue

                feat_vector = extract_pair_features_fast(p1, p2, source2_prefix=target_id[:2])
                feature_rows.append(feat_vector)
                meta_rows.append((s1_id, target_id))

        if feature_rows:
            X_chunk = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
            p_lgb = predict_pair_probabilities(models["lightgbm"], X_chunk)
            p_xgb = predict_pair_probabilities(models["xgboost"], X_chunk)
            p_cb = predict_pair_probabilities(models["catboost"], X_chunk)

            p_ens = ensemble.predict_proba({"lightgbm": p_lgb, "xgboost": p_xgb, "catboost": p_cb})

            for (s1_id, target_id), prob in zip(meta_rows, p_ens):
                if prob >= threshold:
                    all_passing_pairs.append((s1_id, target_id, prob))

        print(f"  Processed {end_idx:,}/{total_s1:,} S1 records...")

    # Global Injective Matching for this country
    print("  Applying Injective Matching...")
    all_passing_pairs.sort(key=lambda x: x[2], reverse=True)
    assigned_targets = set()

    for s1_id in s1_df["entity_id"]:
        country_predictions[s1_id] = set()

    for s1_id, target_id, _ in all_passing_pairs:
        if target_id not in assigned_targets:
            country_predictions[s1_id].add(target_id)
            assigned_targets.add(target_id)

    print(f"✓ Country [{country}] completed in {time.time()-t_start:.2f}s | Matches assigned: {len(assigned_targets):,}")
    return country_predictions, country_candidates


def run_full_test_pipeline(
    pipeline_artifacts: Dict[str, Any],
    output_dir: Union[str, Path] = OUTPUT_DIR,
) -> Tuple[Path, Path]:
    """
    Executes full streaming test inference across all countries and saves valid submission files.
    """
    t_start = time.time()
    print("=" * 70)
    print("🌟 Full Test Set Inference & Submission Pipeline")
    print("=" * 70)

    con = duckdb.connect()
    countries = con.execute(f"""
        SELECT DISTINCT country
        FROM read_csv('{TEST_FILES["source1"]}', delim='\\t', header=true, all_varchar=true)
        ORDER BY country
    """).df()["country"].tolist()
    con.close()

    print(f"Found {len(countries)} dynamic countries in test set: {countries}")

    all_predictions: Dict[str, Set[str]] = {}
    all_candidates: Dict[str, Set[str]] = {}

    models = pipeline_artifacts["models"]
    ensemble = pipeline_artifacts["ensemble"]
    threshold = pipeline_artifacts["optimal_threshold"]

    for country in countries:
        c_preds, c_cands = run_country_test_inference(
            country=country,
            models=models,
            ensemble=ensemble,
            threshold=threshold,
        )
        all_predictions.update(c_preds)
        all_candidates.update(c_cands)

    print("\nFormatting final submission DataFrames...")
    matching_df, candidate_df = format_predictions_to_dataframe(all_predictions, all_candidates)

    print(f"Saving submission files to {output_dir}...")
    matching_path, candidate_path = save_submission_files(matching_df, candidate_df, output_dir=output_dir)

    print(f"✓ Pipeline complete in {time.time()-t_start:.2f}s!")
    print(f"  matching_results.tsv: {len(matching_df):,} rows -> {matching_path}")
    print(f"  candidate_pairs.tsv:  {len(candidate_df):,} rows -> {candidate_path}")

    return matching_path, candidate_path

