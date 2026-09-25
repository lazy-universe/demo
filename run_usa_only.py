"""
Run USA Test Inference Only.
Loads model from pipeline_artifacts.joblib, runs US partition (663,106 records),
and outputs:
  - output/matching_results_usa.tsv
  - output/candidate_pairs_usa.tsv
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

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DELIMITER, MATCHING_RESULT_COLUMNS, CANDIDATE_PAIR_COLUMNS,
    TEST_FILES, OUTPUT_DIR
)
from src.blocking.inverted_index import MultiKeyBlocker
from src.features.extractor import FEATURE_NAMES, extract_pair_features_fast, prepare_entity_profile

TARGET_SEGMENT_SIZE = 1_500_000   # 1.5M records per segment
QUERY_CHUNK_SIZE = 50_000         # 50k query entities per batch
MAX_CANDIDATES_PER_ENTITY = 6     # Top 6 high-precision candidates


def main():
    total_start = time.time()
    print("=" * 75)
    print("🇺🇸 USA PARTITION TEST INFERENCE ENGINE")
    print("=" * 75)

    # 1. Load Model Weights & Calibrated Threshold
    print("\n--- Step 1: Loading Trained Model & Calibrated Threshold ---")
    model_path = Path("models/pipeline_artifacts.joblib")
    if not model_path.exists():
        model_path = Path("help-me/pipeline_artifacts.joblib")
    
    if not model_path.exists():
        print(f"❌ Error: Could not find pipeline_artifacts.joblib in models/ or help-me/")
        sys.exit(1)

    artifacts = joblib.load(model_path)
    lgb_model = artifacts["models"]["lightgbm"]
    threshold = float(artifacts["optimal_threshold"])
    print(f"✓ LightGBM model loaded from {model_path} (Optimal Threshold τ* = {threshold:.2f})")

    # 2. Output Paths
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    usa_matching_file = out_dir / "matching_results_usa.tsv"
    usa_candidate_file = out_dir / "candidate_pairs_usa.tsv"

    # 3. Load US Test Records
    print("\n--- Step 2: Loading US Test Source Data ---")
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false;")

    s1_us_df = con.execute(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{TEST_FILES["source1"]}', delim='\\t', header=true, all_varchar=true)
        WHERE country = 'US'
    """).df()

    s2_us_total = con.execute(f"""
        SELECT COUNT(*) FROM read_csv('{TEST_FILES["source2"]}', delim='\\t', header=true, all_varchar=true)
        WHERE country = 'US'
    """).fetchone()[0]

    s3_us_total = con.execute(f"""
        SELECT COUNT(*) FROM read_csv('{TEST_FILES["source3"]}', delim='\\t', header=true, all_varchar=true)
        WHERE country = 'US'
    """).fetchone()[0]
    con.close()

    total_us_s1 = len(s1_us_df)
    print(f"  US Records: S1={total_us_s1:,}, S2={s2_us_total:,}, S3={s3_us_total:,} (Total Targets: {s2_us_total+s3_us_total:,})")

    # 4. Pre-Profile US S1 Entities (High Speed C++ Optimization)
    print("  Pre-profiling US S1 entities...")
    t_prof = time.time()
    s1_us_profiles: Dict[str, Dict[str, Any]] = {
        str(eid): prepare_entity_profile(str(name), str(addr))
        for eid, name, addr in zip(s1_us_df["entity_id"].values, s1_us_df["business_name"].values, s1_us_df["business_address"].values)
    }
    print(f"  ✓ Pre-profiled {total_us_s1:,} US S1 entities in {time.time()-t_prof:.2f}s")

    us_candidates: Dict[str, Set[str]] = {str(eid): set() for eid in s1_us_df["entity_id"].values}
    us_best_matches: Dict[str, Tuple[str, float]] = {}

    # 5. Universal Segmented Streaming across US S2 and S3
    for src_label, src_file, src_total in [("S2", TEST_FILES["source2"], s2_us_total), ("S3", TEST_FILES["source3"], s3_us_total)]:
        if src_total == 0:
            continue

        total_segs = ((src_total - 1) // TARGET_SEGMENT_SIZE) + 1
        for offset in range(0, src_total, TARGET_SEGMENT_SIZE):
            t_seg = time.time()
            seg_end = min(offset + TARGET_SEGMENT_SIZE, src_total)
            seg_num = (offset // TARGET_SEGMENT_SIZE) + 1
            print(f"  --> Processing US {src_label} Segment {seg_num}/{total_segs} ({offset:,} to {seg_end:,})...")

            con = duckdb.connect()
            con.execute("SET enable_progress_bar=false;")
            target_df = con.execute(f"""
                SELECT entity_id, business_name, business_address, country
                FROM read_csv('{src_file}', delim='\\t', header=true, all_varchar=true)
                WHERE country = 'US'
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

            # Stream US S1 queries against this segment
            for s1_start in range(0, total_us_s1, QUERY_CHUNK_SIZE):
                s1_end = min(s1_start + QUERY_CHUNK_SIZE, total_us_s1)
                chunk_s1 = s1_us_df.iloc[s1_start:s1_end]

                cand_dict = blocker.query_candidates(chunk_s1, max_candidates_per_entity=MAX_CANDIDATES_PER_ENTITY)

                feature_rows = []
                meta_rows = []

                for s1_id_val in chunk_s1["entity_id"].values:
                    s1_id = str(s1_id_val).strip()
                    cands = cand_dict.get(s1_id)
                    if not cands:
                        continue

                    us_candidates[s1_id].update(cands)
                    p1 = s1_us_profiles[s1_id]

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
                            if s1_id not in us_best_matches or prob > us_best_matches[s1_id][1]:
                                us_best_matches[s1_id] = (target_id, float(prob))

            print(f"      Segment {seg_num}/{total_segs} processed in {time.time()-t_seg:.2f}s")
            del blocker, target_profiles
            gc.collect()

    # 6. Apply Global Injective Matching for US
    print(f"\n--- Step 3: Applying Injective Matching on {len(us_best_matches):,} Candidate Matches ---")
    sorted_pairs = sorted(us_best_matches.items(), key=lambda x: x[1][1], reverse=True)
    assigned_targets = set()
    us_predictions: Dict[str, str] = {}

    for s1_id, (target_id, prob) in sorted_pairs:
        if target_id not in assigned_targets:
            us_predictions[s1_id] = target_id
            assigned_targets.add(target_id)

    # 7. Write US Results to Disk
    print(f"\n--- Step 4: Writing {total_us_s1:,} US Rows to Disk ---")
    with open(usa_matching_file, "w", encoding="utf-8") as f_match, \
         open(usa_candidate_file, "w", encoding="utf-8") as f_cand:
        
        f_match.write("\t".join(MATCHING_RESULT_COLUMNS) + "\n")
        f_cand.write("\t".join(CANDIDATE_PAIR_COLUMNS) + "\n")

        for s1_id_val in s1_us_df["entity_id"].values:
            s1_id = str(s1_id_val).strip()
            matched_id = us_predictions.get(s1_id, "")
            cands_set = us_candidates.get(s1_id, set())

            if matched_id:
                cands_set.add(matched_id)

            cand_str = ",".join(sorted(list(cands_set)))
            f_match.write(f"{s1_id}\t{matched_id}\n")
            f_cand.write(f"{s1_id}\t{cand_str}\n")

    print("=" * 75)
    print("🏆 USA PARTITION INFERENCE COMPLETE!")
    print(f"Total US S1 Entities:     {total_us_s1:,}")
    print(f"Total Matches Assigned:   {len(assigned_targets):,}")
    print(f"Time Taken:               {time.time()-total_start:.2f}s")
    print(f"USA Matching File:        {usa_matching_file}")
    print(f"USA Candidate File:       {usa_candidate_file}")
    print("=" * 75)


if __name__ == "__main__":
    main()
