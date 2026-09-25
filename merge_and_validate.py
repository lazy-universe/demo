"""
Merges France + India (from help-me/ or output/) with USA (from output/matching_results_usa.tsv)
and validates the final 1,732,544 submission files with the official competition validator.
"""

import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, Set
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MATCHING_RESULT_COLUMNS, CANDIDATE_PAIR_COLUMNS, TEST_FILES, OUTPUT_DIR


def main():
    print("=" * 75)
    print("🔗 MASTER SUBMISSION MERGER & OFFICIAL VALIDATOR")
    print("=" * 75)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_matching = out_dir / "matching_results.tsv"
    final_candidate = out_dir / "candidate_pairs.tsv"

    # 1. Check sources for France + India
    fr_in_match = Path("help-me/matching_results.tsv")
    if not fr_in_match.exists():
        fr_in_match = out_dir / "matching_results_fr_in.tsv"

    # 2. Check source for USA
    usa_match = out_dir / "matching_results_usa.tsv"
    if not usa_match.exists():
        usa_match = Path("help-me/matching_results_usa.tsv")

    usa_cand = out_dir / "candidate_pairs_usa.tsv"
    if not usa_cand.exists():
        usa_cand = Path("help-me/candidate_pairs_usa.tsv")

    if not fr_in_match.exists():
        print(f"❌ Error: Missing France + India matching file: {fr_in_match}")
        sys.exit(1)

    if not usa_match.exists():
        print(f"❌ Error: Missing USA matching file: {usa_match}")
        sys.exit(1)

    print(f"✓ Found France + India matches: {fr_in_match}")
    print(f"✓ Found USA matches:            {usa_match}")

    # Load All Matches
    all_matches: Dict[str, str] = {}
    with open(fr_in_match, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                all_matches[parts[0]] = parts[1] if len(parts) > 1 else ""

    with open(usa_match, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                all_matches[parts[0]] = parts[1] if len(parts) > 1 else ""

    print(f"✓ Loaded matches for {len(all_matches):,} total S1 entities")

    # Load Candidates
    all_candidates: Dict[str, Set[str]] = {}
    for cand_file in [Path("help-me/candidate_pairs.tsv"), usa_cand]:
        if cand_file.exists():
            with open(cand_file, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.strip().split("\t")
                    if parts and parts[0]:
                        cands = set(parts[1].split(",")) if len(parts) > 1 and parts[1].strip() else set()
                        all_candidates[parts[0]] = cands

    # Read all required S1 test IDs in official order
    con = duckdb.connect()
    s1_test_ids = con.execute(f"""
        SELECT entity_id
        FROM read_csv('{TEST_FILES["source1"]}', delim='\\t', header=true, all_varchar=true)
    """).df()["entity_id"].tolist()
    con.close()

    total_required = len(s1_test_ids)
    print(f"Writing {total_required:,} rows to master submission files...")

    written = 0
    with open(final_matching, "w", encoding="utf-8") as fm, \
         open(final_candidate, "w", encoding="utf-8") as fc:
        
        fm.write("\t".join(MATCHING_RESULT_COLUMNS) + "\n")
        fc.write("\t".join(CANDIDATE_PAIR_COLUMNS) + "\n")

        for s1_id in s1_test_ids:
            s1_id = str(s1_id).strip()
            matched_id = all_matches.get(s1_id, "")
            cands_set = set(all_candidates.get(s1_id, set()))

            if matched_id:
                cands_set.add(matched_id)

            cand_str = ",".join(sorted(list(cands_set)))
            fm.write(f"{s1_id}\t{matched_id}\n")
            fc.write(f"{s1_id}\t{cand_str}\n")
            written += 1

    print(f"✓ Successfully wrote {written:,} rows to:")
    print(f"  {final_matching}")
    print(f"  {final_candidate}")

    # Run Official Validator
    print("\n--- Running Official Validator ---")
    val_cmd = (
        f"python3 {PROJECT_ROOT}/dataset/student_resource/utils/validate_submission.py "
        f"--matching {final_matching} "
        f"--candidate {final_candidate} "
        f"--test-dir {TEST_FILES['source1'].parent}"
    )
    exit_code = os.system(val_cmd)

    if exit_code == 0:
        zip_path = "submission_package.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(final_matching, arcname="output/matching_results.tsv")
            zipf.write(final_candidate, arcname="output/candidate_pairs.tsv")
            if Path("DOCUMENTATION.md").exists():
                zipf.write("DOCUMENTATION.md", arcname="DOCUMENTATION.md")
            for root, dirs, filenames in os.walk("src"):
                for filename in filenames:
                    if filename.endswith((".py", ".md")):
                        filepath = os.path.join(root, filename)
                        zipf.write(filepath, arcname=filepath)
        print(f"\n🎉 Package created: {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
