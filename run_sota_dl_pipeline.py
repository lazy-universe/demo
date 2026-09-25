"""
SOTA 2026 Deep Learning Submission Generator.
End-to-End Candidate Blocker + BGE-M3 / Qwen3 Cross-Encoder Reranker + Hungarian Bipartite Solver.
Outputs valid multi-match output/matching_results.tsv and output/candidate_pairs.tsv for all 1,732,544 test entities.
"""

import os
import sys
import time
import gc
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import duckdb
import numpy as np
import pandas as pd
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


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
DEFAULT_TOP_K = 3                 # Top 3 high-potential candidates per entity (optimal speed-accuracy balance)
DEFAULT_MAX_LEN = 80              # Max sequence length (80 tokens covers 99.8% of business names + addresses)
DECISION_THRESHOLD = 0.68         # Calibrated for multi-match F0.5


def parse_args():
    parser = argparse.ArgumentParser(description="Run SOTA Deep Learning Pipeline for Amazon ML Challenge 2026")
    parser.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-v2-m3", help="Cross-encoder model name")
    parser.add_argument("--dense-model", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="Dense embedding model name")
    parser.add_argument("--use-dense", action="store_true", default=False, help="Enable dense embedding index")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top K candidates per entity (default: 3)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for cross-encoder inference (default: 512)")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN, help="Max sequence length for transformer reranker (default: 80)")
    parser.add_argument("--threshold", type=float, default=DECISION_THRESHOLD, help="Match threshold for Hungarian solver (default: 0.68)")
    return parser.parse_args()


def main():
    args = parse_args()
    total_start = time.time()

    print("=" * 75)
    print("🚀 SOTA 2026 PURE DEEP LEARNING SUBMISSION ENGINE")
    print("   Candidate Blocker + Cross-Encoder Deep Learning + Hungarian Solver")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Active Hardware Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU Model:              {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM:             {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 1. Eager Initialize & Warm-Up SOTA DL Components in Step 1
    print("\n--- Step 1: Initializing & Warming Up Deep Learning Models ---")
    t0 = time.time()

    # Inspect HF Cache
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache_dir.exists():
        cached_models = [d.name.replace("models--", "").replace("--", "/") for d in hf_cache_dir.glob("models--*")]
        print(f"  HuggingFace Local Cache: {len(cached_models)} models found ({', '.join(cached_models[:3])})")
    else:
        print("  HuggingFace Local Cache: Initializing fresh directory")

    hybrid_blocker = TriHybridBlocker(
        dense_model_name=args.dense_model,
        dense_dim=512,
        use_dense=args.use_dense,
        use_learned_sparse=False,
        device=device,
    )
    reranker = QwenTransformerReranker(
        model_name=args.reranker_model,
        fallback_model_name="BAAI/bge-reranker-v2-m3",
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
    )
    matcher = MultiMatchBipartiteSolver(default_threshold=args.threshold)

    # Eager Model Warm-Up on GPU
    print("  Loading & Warming up Cross-Encoder...")
    reranker.warmup()
    if args.use_dense and hybrid_blocker.dense_retriever is not None:
        print("  Loading Dense Embedding Model...")
        hybrid_blocker.dense_retriever._load_model()

    print(f"✓ All Deep Learning models verified and ready on {device.upper()} in {time.time()-t0:.2f}s")
    print(f"  Config: Reranker={args.reranker_model} | Top-K={args.top_k} | Batch Size={args.batch_size} | Max Length={args.max_length} | Threshold={args.threshold}")

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

                # Build Blocker Index
                hybrid_blocker.build_index(target_df)

                target_raw: Dict[str, Tuple[str, str]] = {
                    str(eid): (str(name), str(addr))
                    for eid, name, addr in zip(target_df["entity_id"].values, target_df["business_name"].values, target_df["business_address"].values)
                }
                del target_df
                gc.collect()

                # Stream S1 Queries in Chunks
                seg_pairs_scored = 0
                for s1_start in range(0, total_s1, QUERY_CHUNK_SIZE):
                    s1_end = min(s1_start + QUERY_CHUNK_SIZE, total_s1)
                    chunk_s1 = s1_df.iloc[s1_start:s1_end]

                    # 1. Query Blocker
                    cand_dict = hybrid_blocker.query_candidates(
                        chunk_s1,
                        top_k_exact=args.top_k,
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

                    # 2. Score with Cross-Encoder
                    if pair_texts:
                        seg_pairs_scored += len(pair_texts)
                        probs = reranker.predict_pair_probabilities(pair_texts, show_progress=True)
                        for (s1_id, tid), prob in zip(pair_meta, probs):
                            if prob >= args.threshold:
                                all_passing_pairs.append((s1_id, tid, float(prob)))

                print(f"      Segment {seg_num}/{total_segs} processed {seg_pairs_scored:,} candidate pairs in {time.time()-t_seg:.2f}s")
                del target_raw
                gc.collect()

        # 4. Multi-Match Hungarian / Bipartite Assignment
        print(f"  Applying Multi-Match Hungarian assignment on {len(all_passing_pairs):,} passing pairs...")
        all_s1_set = set(s1_df["entity_id"].values)
        predictions = matcher.solve_multi_match_assignment(all_passing_pairs, all_s1_set, threshold=args.threshold)

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
