"""
SOTA 2026 Deep Learning Submission Generator.
End-to-End Qwen3-Embedding (FAISS-GPU) + Qwen3-Reranker (Cross-Attention) + Hungarian Bipartite Solver.
Outputs valid multi-match output/matching_results.tsv and output/candidate_pairs.tsv for all 1,732,544 test entities.
"""

import os
import sys
import time
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import duckdb
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DELIMITER, MATCHING_RESULT_COLUMNS, CANDIDATE_PAIR_COLUMNS,
    TEST_FILES, OUTPUT_DIR
)
from src.blocking.hybrid_blocker import TriHybridBlocker
from src.models.transformer_reranker import QwenTransformerReranker
from src.evaluation.bipartite_matcher import MultiMatchBipartiteSolver
from src.preprocessing.text import clean_address, clean_business_name

TARGET_SEGMENT_SIZE = 1_000_000   # 1.0M records per segment on GPU
QUERY_CHUNK_SIZE = 100_000        # 100k query entities per batch
DENSE_TOP_K = 6                   # Top 6 dense candidates from Qwen3
EXACT_TOP_K = 4                   # Top 4 exact lexical candidates


def main():
    total_start = time.time()
    print("=" * 75)
    print("🚀 SOTA 2026 PURE DEEP LEARNING SUBMISSION ENGINE")
    print("   Qwen3-Embedding (FAISS-GPU) + Qwen3-Reranker + Hungarian Solver")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Active Hardware Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    # 1. Initialize SOTA DL Components
    print("\n--- Step 1: Initializing Foundation Models ---")
    t0 = time.time()
    hybrid_blocker = TriHybridBlocker(
        dense_model_name="Qwen/Qwen3-Embedding-0.6B",
        dense_dim=512,
        use_learned_sparse=False,
        device=device,
    )
    reranker = QwenTransformerReranker(
        model_name="Qwen/Qwen3-Reranker-0.6B",
        fallback_model_name="BAAI/bge-reranker-v2-m3",
        max_length=256,
        batch_size=256,
        device=device,
    )
    matcher = MultiMatchBipartiteSolver(default_threshold=0.68)
    print(f"✓ Models and Hungarian solver initialized in {time.time()-t0:.2f}s")

    # 2. Output Paths
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    matching_file = out_dir / "matching_results.tsv"
    candidate_file = out_dir / "candidate_pairs.tsv"

    with open(matching_file, "w", encoding="utf-8") as fm, \
         open(candidate_file, "w", encoding="utf-8") as fc:
        fm.write("\t".join(MATCHING_RESULT_COLUMNS) + "\n")
        fc.write("\t".join(CANDIDATE_PAIR_COLUMNS) + "\n")

    # 3. Stream Test Inference Across Country Partitions
    print("\n--- Step 2: Streaming DL Inference Across Country Partitions ---")
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false;")
    countries = con.execute(f"""
        SELECT DISTINCT country
        FROM read_csv('{TEST_FILES["source1"]}', delim='\\t', header=true, all_varchar=true)
        ORDER BY country
    """).df()["country"].tolist()
    con.close()

    total_entities_processed = 0
    total_matches_assigned = 0

    for country in countries:
        t_country = time.time()
        print(f"\n[{country.upper()}] Starting partition...")

        con = duckdb.connect()
        con.execute("SET enable_progress_bar=false;")

        s1_df = con.execute(f"""
            SELECT entity_id, business_name, business_address, country
            FROM read_csv('{TEST_FILES["source1"]}', delim='\\t', header=true, all_varchar=true)
            WHERE country = '{country}'
        """).df()

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

        candidate_map: Dict[str, Set[str]] = {str(eid): set() for eid in s1_df["entity_id"].values}
        all_passing_pairs: List[Tuple[str, str, float]] = []

        # Universal Segmented Streaming across S2 and S3
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

                # Build Hybrid Blocker Index (Inverted Index + Dense Embeddings)
                hybrid_blocker.build_index(target_df)

                target_raw: Dict[str, Tuple[str, str]] = {
                    str(eid): (str(name), str(addr))
                    for eid, name, addr in zip(target_df["entity_id"].values, target_df["business_name"].values, target_df["business_address"].values)
                }
                del target_df
                gc.collect()

                # Stream S1 Queries in Chunks
                for s1_start in range(0, total_s1, QUERY_CHUNK_SIZE):
                    s1_end = min(s1_start + QUERY_CHUNK_SIZE, total_s1)
                    chunk_s1 = s1_df.iloc[s1_start:s1_end]

                    # 1. Query Hybrid Blocker (Dense FAISS-GPU + Exact Lexical)
                    cand_dict = hybrid_blocker.query_candidates(
                        chunk_s1,
                        target_df=pd.DataFrame([{"entity_id": k, "business_name": v[0], "business_address": v[1], "country": country} for k, v in target_raw.items()]),
                        top_k_dense=DENSE_TOP_K,
                        top_k_exact=EXACT_TOP_K,
                    )

                    pair_texts = []
                    pair_meta = []

                    for eid, name, addr in zip(chunk_s1["entity_id"].values, chunk_s1["business_name"].values, chunk_s1["business_address"].values):
                        s1_id = str(eid).strip()
                        cands = cand_dict.get(s1_id, set())
                        if not cands:
                            continue

                        candidate_map[s1_id].update(cands)
                        s1_txt = f"Business: {clean_business_name(str(name))} | Address: {clean_address(str(addr))}"

                        for tid in cands:
                            trec = target_raw.get(tid)
                            if not trec:
                                continue
                            t_txt = f"Business: {clean_business_name(trec[0])} | Address: {clean_address(trec[1])}"
                            pair_texts.append((s1_txt, t_txt))
                            pair_meta.append((s1_id, tid))

                    # 2. Score with Qwen3-Reranker Cross-Encoder
                    if pair_texts:
                        probs = reranker.predict_pair_probabilities(pair_texts)
                        for (s1_id, tid), prob in zip(pair_meta, probs):
                            if prob >= 0.68:
                                all_passing_pairs.append((s1_id, tid, float(prob)))

                print(f"      Segment {seg_num}/{total_segs} processed in {time.time()-t_seg:.2f}s")
                del target_raw
                gc.collect()

        # 4. Multi-Match Hungarian / Bipartite Assignment
        print(f"  Applying Multi-Match Hungarian assignment on {len(all_passing_pairs):,} candidate pairs...")
        all_s1_set = set(s1_df["entity_id"].values)
        predictions = matcher.solve_multi_match_assignment(all_passing_pairs, all_s1_set, threshold=0.68)

        total_matched = sum(1 for m in predictions.values() if m)
        multi_matched = sum(1 for m in predictions.values() if len(m) > 1)
        print(f"  Matched S1 entities: {total_matched:,} (Singletons: {total_s1-total_matched:,} | Multi-matched: {multi_matched:,})")

        # 5. Write to Master Submission Files
        print(f"  Writing {total_s1:,} rows to disk...")
        with open(matching_file, "a", encoding="utf-8") as fm, \
             open(candidate_file, "a", encoding="utf-8") as fc:
            for s1_id_val in s1_df["entity_id"].values:
                s1_id = str(s1_id_val).strip()
                matched_set = predictions.get(s1_id, set())
                cands_set = candidate_map.get(s1_id, set())

                if matched_set:
                    cands_set.update(matched_set)

                matched_str = ",".join(sorted(list(matched_set)))
                cand_str = ",".join(sorted(list(cands_set)))

                fm.write(f"{s1_id}\t{matched_str}\n")
                fc.write(f"{s1_id}\t{cand_str}\n")

        total_entities_processed += total_s1
        total_matches_assigned += sum(len(m) for m in predictions.values())
        print(f"✓ [{country.upper()}] completed in {time.time()-t_country:.2f}s | Matches assigned: {total_matches_assigned:,}")

        del s1_df, candidate_map, all_passing_pairs, predictions
        gc.collect()

    print("\n" + "=" * 75)
    print("🏆 SOTA DEEP LEARNING INFERENCE COMPLETE!")
    print(f"Total S1 Entities:     {total_entities_processed:,}")
    print(f"Total Targets Matched: {total_matches_assigned:,}")
    print(f"Total Time:            {time.time()-total_start:.2f}s")
    print(f"Matching Results:      {matching_file}")
    print(f"Candidate Pairs:       {candidate_file}")
    print("=" * 75)

    # 6. Run Official Validator
    print("\n--- Running Official Validator ---")
    val_cmd = (
        f"python3 {PROJECT_ROOT}/dataset/student_resource/utils/validate_submission.py "
        f"--matching {matching_file} "
        f"--candidate {candidate_file} "
        f"--test-dir {TEST_FILES['source1'].parent}"
    )
    os.system(val_cmd)


if __name__ == "__main__":
    main()
