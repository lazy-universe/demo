"""
Single-Command End-to-End Master Pipeline Runner for Google Colab / CLI.
Executes dataset download, split creation, model training/loading, test inference, and validation in one shot.
"""

import os
import sys
import time
import zipfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import src
from src.data.validation_split import ValidationSplitManager
from src.config import TEST_FILES, OUTPUT_DIR

DATASET_ZIP_URL = "https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip"


def step_1_download_dataset():
    print("=" * 70)
    print("Step 1: Checking Dataset Files")
    print("=" * 70)
    test_s1 = PROJECT_ROOT / "dataset/student_resource/dataset/test/test_source1.tsv"
    if not test_s1.exists():
        print(f"Downloading dataset from {DATASET_ZIP_URL}...")
        os.system(f"wget -q --show-progress '{DATASET_ZIP_URL}' -O dataset.zip")
        print("Extracting dataset...")
        os.makedirs(PROJECT_ROOT / "dataset/student_resource", exist_ok=True)
        os.system(f"unzip -q -o dataset.zip -d {PROJECT_ROOT}/dataset/student_resource/")
        print("✓ Dataset extracted successfully!")
    else:
        print("✓ Dataset already exists on disk.")


def step_2_train_and_validate():
    print("\n" + "=" * 70)
    print("Step 2: Split Creation & Model Benchmark")
    print("=" * 70)
    split_mgr = ValidationSplitManager(train_ratio=0.80, random_seed=42)
    split_mgr.create_splits()

    artifacts = src.train_and_validate_pipeline(
        train_s1_samples=10000,
        val_s1_samples=2500,
        random_state=42,
        save_path="models/pipeline_artifacts.joblib",
        load_cached=True
    )
    print("\nBenchmark Summary:")
    print(artifacts["benchmark_summary"].to_string(index=False))
    return artifacts


def step_3_generate_submission():
    print("\n" + "=" * 70)
    print("Step 3: Generating Full Test Submission")
    print("=" * 70)
    os.system(f"python3 {PROJECT_ROOT}/generate_final_submission.py")


def step_4_package_submission():
    print("\n" + "=" * 70)
    print("Step 4: Packaging Submission Zip")
    print("=" * 70)
    submission_zip = PROJECT_ROOT / "submission_package.zip"
    matching_file = PROJECT_ROOT / "output/matching_results.tsv"
    candidate_file = PROJECT_ROOT / "output/candidate_pairs.tsv"
    doc_file = PROJECT_ROOT / "DOCUMENTATION.md"

    with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        if matching_file.exists():
            zipf.write(matching_file, arcname="output/matching_results.tsv")
        if candidate_file.exists():
            zipf.write(candidate_file, arcname="output/candidate_pairs.tsv")
        if doc_file.exists():
            zipf.write(doc_file, arcname="DOCUMENTATION.md")
        for root, dirs, filenames in os.walk(PROJECT_ROOT / "src"):
            for filename in filenames:
                if filename.endswith((".py", ".md")):
                    filepath = os.path.join(root, filename)
                    relpath = os.path.relpath(filepath, PROJECT_ROOT)
                    zipf.write(filepath, arcname=relpath)

    print(f"✓ Created {submission_zip} ({os.path.getsize(submission_zip)/1e6:.2f} MB)")


def main():
    start_time = time.time()
    step_1_download_dataset()
    step_2_train_and_validate()
    step_3_generate_submission()
    step_4_package_submission()
    print(f"\n🎉 ALL STEPS COMPLETED IN {time.time()-start_time:.2f}s!")


if __name__ == "__main__":
    main()
