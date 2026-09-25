"""
Production Pipeline & Official Submission Generator (Multi-Match Enabled).
Memory-Safe LightGBM Engine (1.5M Segment Size | 50k Query Chunks | Pre-Profiled C++ RapidFuzz).
Generates multi-match output/matching_results.tsv and output/candidate_pairs.tsv for all 1,732,544 test entities.
"""

import os
import sys
import time
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import duckdb
import joblib
import numpy as np
import pandas as pd

# Add workspace root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import src
from src.config import (
    TRAIN_FILES, TEST_FILES, SPLIT_FILES, OUTPUT_DIR, DELIMITER,
    MATCHING_RESULT_COLUMNS, CANDIDATE_PAIR_COLUMNS
)
from src.blocking.inverted_index import MultiKeyBlocker
from src.features.extractor import FEATURE_NAMES, extract_pair_features_fast, prepare_entity_profile

# High-Speed Constants for Colab (1.5M Targets per Segment)
TARGET_SEGMENT_SIZE = 1_500_000   # 1.5M records per segment
QUERY_CHUNK_SIZE = 50_000         # 50k query entities per batch
MAX_CANDIDATES_PER_ENTITY = 6     # Top 6 high-precision candidates per source


def main():
    total_start = time.time()
    print("=" * 75)
    print("🚀 AMAZON ML CHALLENGE 2026: MASTER SUBMISSION GENERATOR")
    print("   Multi-Match Enabled LightGBM Engine (1.5M Segments | Pre-Profiled)")
    print("=" * 75)

    # -------------------------------------------------------------
    # Step 1: Load Trained Pipeline Artifacts
    # -------------------------------------------------------------
    print("\n--- Step 1: Loading Trained Model & Calibrated Threshold ---")
    t0 = time.time()
    model_path = Path("models/pipeline_artifacts.joblib")
    if not model_path.exists():
        model_path = Path("help-me/pipeline_artifacts.joblib")

    if model_path.exists():
        artifacts = joblib.load(model_path)
        print(f"✓ Loaded cached pipeline model weights from {model_path}")
    else:
        artifacts = src.train_and_validate_pipeline(
            train_s1_samples=50000,
            val_s1_samples=10000,
            random_state=42,
            save_path="models/pipeline_artifacts.joblib",
            load_cached=True
        )
    
    lgb_model = artifacts["models"]["lightgbm"]
    summary_df = artifacts.get("benchmark_summary")
    
    if summary_df is not None and isinstance(summary_df, pd.DataFrame):
        lgb_row = summary_df[summary_df["model"] == "LightGBM"]
        threshold = float(lgb_row["optimal_threshold"].values[0]) if not lgb_row.empty else 0.72
    else:
        threshold = float(artifacts.get("optimal_threshold", 0.72))
        
    print(f"✓ LightGBM model ready (Optimal Threshold τ* = {threshold:.2f}) in {time.time()-t0:.2f}s")

    # -------------------------------------------------------------
    # Step 2: Prepare Submission Output Files
    # -------------------------------------------------------------
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    matching_file = out_dir / "matching_results.tsv"
    candidate_file = out_dir / "candidate_pairs.tsv"

    # Reset / Overwrite files with headers
    with open(matching_file, "w", encoding="utf-8") as f_match, \
         open(candidate_file, "w", encoding="utf-8") as f_cand:
        f_match.write("\t".join(MATCHING_RESULT_COLUMNS) + "\n")
        f_cand.write("\t".join(CANDIDATE_PAIR_COLUMNS) + "\n")

    # -------------------------------------------------------------
    # Step 3: Stream Test Inference Country-by-Country
    # -------------------------------------------------------------
    print("\n--- Step 2: Streaming Test Inference Across Country Partitions ---")
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false;")
    countries = con.execute(f"""
        SELECT DISTINCT country
        FROM read_csv('{TEST_FILES["source1"]}', delim='\\t', header=true, all_varchar=true)
        ORDER BY country
    """).df()["country"].tolist()
    con.close()

    print(f"Discovered {len(countries)} dynamic test country partitions: {countries}")

    total_entities_processed = 0
    total_matches_assigned = 0

    for country in countries:
        t_country = time.time()
        print(f"\n[{country.upper()}] Starting partition...")

        con = duckdb.connect()
        con.execute("SET enable_progress_bar=false;")

        # Load S1 for this country
        s1_df = con.execute(f"""
            SELECT entity_id, business_name, business_address, country
            FROM read_csv('{TEST_FILES["source1"]}', delim='\\t', header=true, all_varchar=true)
            WHERE country = '{country}'
        """).df()

        # Count total targets in S2 and S3 for this country
        s2_total = con.execute(f"""
            SELECT COUNT(*) FROM read_csv('{TEST_FILES["source2"]}', delim='\\t', header=true, all_varchar=true)
            WHERE country = '{country}'
        """).fetchone()[0]

        s3_total = con.execute(f"""
            SELECT COUNT(*) FROM read_csv('{TEST_FILES["source3"]}', delim='\\t', header=true, all_varchar=true)
            WHERE country = '{country}'
        """).fetchone()[0]
        con.close()

        total_s1 = len(s1_df)
        print(f"  Records: S1={total_s1:,}, S2={s2_total:,}, S3={s3_total:,} (Total Targets: {s2_total+s3_total:,})")

        # Pre-profile S1 entities for ultra-fast C++ matching
        print("  Pre-profiling S1 entities...")
        t_prof = time.time()
        s1_profile_map: Dict[str, Dict[str, Any]] = {
            str(eid): prepare_entity_profile(str(name), str(addr))
            for eid, name, addr in zip(s1_df["entity_id"].values, s1_df["business_name"].values, s1_df["business_address"].values)
        }
        print(f"  ✓ Pre-profiled {total_s1:,} S1 entities in {time.time()-t_prof:.2f}s")

        # Accumulators
        candidate_map: Dict[str, Set[str]] = {str(eid): set() for eid in s1_df["entity_id"].values}
        all_passing_pairs: List[Tuple[str, str, float]] = []

        # Universal 1.5M Segmented Streaming across S2 and S3
        for src_label, src_file, src_total in [("S2", TEST_FILES["source2"], s2_total), ("S3", TEST_FILES["source3"], s3_total)]:
            if src_total == 0:
                continue

            total_segs = ((src_total - 1) // TARGET_SEGMENT_SIZE) + 1
            for offset in range(0, src_total, TARGET_SEGMENT_SIZE):
                t_seg = time.time()
                seg_end = min(offset + TARGET_SEGMENT_SIZE, src_total)
                seg_num = (offset // TARGET_SEGMENT_SIZE) + 1
                print(f"  --> Processing {src_label} Segment {seg_num}/{total_segs} ({offset:,} to {seg_end:,})...")

                con = duckdb.connect()
                con.execute("SET enable_progress_bar=false;")
                target_df = con.execute(f"""
                    SELECT entity_id, business_name, business_address, country
                    FROM read_csv('{src_file}', delim='\\t', header=true, all_varchar=true)
                    WHERE country = '{country}'
                    LIMIT {TARGET_SEGMENT_SIZE} OFFSET {offset}
                """).df()
                con.close()

                # Build Segment Inverted Index
                blocker = MultiKeyBlocker(max_block_size=200)
                blocker.build_index_df(target_df)

                # Pre-profile target records once for this segment
                target_profiles: Dict[str, Dict[str, Any]] = {
                    str(eid): prepare_entity_profile(str(name), str(addr))
                    for eid, name, addr in zip(target_df["entity_id"].values, target_df["business_name"].values, target_df["business_address"].values)
                }
                del target_df
                gc.collect()

                # Stream S1 queries against this segment
                for s1_start in range(0, total_s1, QUERY_CHUNK_SIZE):
                    s1_end = min(s1_start + QUERY_CHUNK_SIZE, total_s1)
                    chunk_s1 = s1_df.iloc[s1_start:s1_end]

                    cand_dict = blocker.query_candidates(chunk_s1, max_candidates_per_entity=MAX_CANDIDATES_PER_ENTITY)

                    feature_rows = []
                    meta_rows = []

                    for s1_id_val in chunk_s1["entity_id"].values:
                        s1_id = str(s1_id_val).strip()
                        cands = cand_dict.get(s1_id)
                        if not cands:
                            continue

                        # Accumulate candidate IDs
                        candidate_map[s1_id].update(cands)
                        p1 = s1_profile_map[s1_id]

                        for target_id in cands:
                            p2 = target_profiles.get(target_id)
                            if not p2:
                                continue

                            feat_vector = extract_pair_features_fast(p1, p2, source2_prefix=src_label)
                            feature_rows.append(feat_vector)
                            meta_rows.append((s1_id, target_id))

                    if feature_rows:
                        X_chunk = pd.DataFrame(feature_rows, columns=FEATURE_NAMES, dtype=np.float32)
                        probs = lgb_model.predict_proba(X_chunk)[:, 1]

                        for (s1_id, target_id), prob in zip(meta_rows, probs):
                            if prob >= threshold:
                                all_passing_pairs.append((s1_id, target_id, float(prob)))

                print(f"      Segment {seg_num}/{total_segs} processed in {time.time()-t_seg:.2f}s")
                del blocker, target_profiles
                gc.collect()

        # Apply Global Multi-Match Injective Assignment for this country
        # (Allows S1 entities to match MULTIPLE targets from S2 and S3 while preventing target collision)
        print(f"  Applying global multi-match injective assignment on {len(all_passing_pairs):,} passing pairs...")
        all_passing_pairs.sort(key=lambda x: x[2], reverse=True)
        assigned_targets = set()
        country_predictions: Dict[str, Set[str]] = {eid: set() for eid in s1_df["entity_id"].values}

        for s1_id, target_id, prob in all_passing_pairs:
            if target_id not in assigned_targets:
                country_predictions[s1_id].add(target_id)
                assigned_targets.add(target_id)

        # Count multi-matches
        total_matched_entities = sum(1 for m in country_predictions.values() if m)
        multi_matched_entities = sum(1 for m in country_predictions.values() if len(m) > 1)
        print(f"  Matched S1 entities: {total_matched_entities:,} (Singletons: {total_s1-total_matched_entities:,} | Multi-matched: {multi_matched_entities:,})")

        # Append Country Results to Submission Files on Disk
        print(f"  Writing {total_s1:,} rows to submission files...")
        with open(matching_file, "a", encoding="utf-8") as f_match, \
             open(candidate_file, "a", encoding="utf-8") as f_cand:
            for s1_id_val in s1_df["entity_id"].values:
                s1_id = str(s1_id_val).strip()
                matched_set = country_predictions.get(s1_id, set())
                cands_set = candidate_map.get(s1_id, set())

                # Ensure candidate list includes all matched entities
                if matched_set:
                    cands_set.update(matched_set)

                matched_str = ",".join(sorted(list(matched_set)))
                cand_str = ",".join(sorted(list(cands_set)))

                f_match.write(f"{s1_id}\t{matched_str}\n")
                f_cand.write(f"{s1_id}\t{cand_str}\n")

        total_entities_processed += total_s1
        total_matches_assigned += len(assigned_targets)
        print(f"✓ [{country.upper()}] completed in {time.time()-t_country:.2f}s | Matches assigned: {len(assigned_targets):,}")

        del s1_df, s1_profile_map, candidate_map, all_passing_pairs, country_predictions
        gc.collect()

    print("\n" + "=" * 75)
    print("🏆 ALL TEST PARTITIONS COMPLETED!")
    print(f"Total S1 Entities Processed: {total_entities_processed:,}")
    print(f"Total Targets Assigned:      {total_matches_assigned:,}")
    print(f"Total Time Taken:            {time.time()-total_start:.2f}s")
    print(f"Matching Results:            {matching_file}")
    print(f"Candidate Pairs:             {candidate_file}")
    print("=" * 75)

    # -------------------------------------------------------------
    # Step 4: Run Official Submission Validator
    # -------------------------------------------------------------
    print("\n--- Step 3: Running Official Submission Validator ---")
    val_cmd = (
        f"python3 {PROJECT_ROOT}/dataset/student_resource/utils/validate_submission.py "
        f"--matching {matching_file} "
        f"--candidate {candidate_file} "
        f"--test-dir {TEST_FILES['source1'].parent}"
    )
    os.system(val_cmd)


if __name__ == "__main__":
    main()
