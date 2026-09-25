"""
Centralized configuration settings, dataset paths, column definitions, and global constants
for the Amazon ML Challenge 2026 Business Entity Resolution project.
"""

import os
from pathlib import Path

# Base Directory Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "student_resource" / "dataset"
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"
UTILS_DIR = PROJECT_ROOT / "dataset" / "student_resource" / "utils"
OUTPUT_DIR = PROJECT_ROOT / "output"
SPLITS_DIR = PROJECT_ROOT / "dataset" / "splits"

# Dataset Files
TRAIN_FILES = {
    "source1": TRAIN_DIR / "train_source1.tsv",
    "source2": TRAIN_DIR / "train_source2.tsv",
    "source3": TRAIN_DIR / "train_source3.tsv",
    "ground_truth": TRAIN_DIR / "train_ground_truth.tsv",
}

TEST_FILES = {
    "source1": TEST_DIR / "test_source1.tsv",
    "source2": TEST_DIR / "test_source2.tsv",
    "source3": TEST_DIR / "test_source3.tsv",
}

SPLIT_FILES = {
    "train_source1": SPLITS_DIR / "split_train_source1.tsv",
    "val_source1": SPLITS_DIR / "split_val_source1.tsv",
    "train_ground_truth": SPLITS_DIR / "split_train_ground_truth.tsv",
    "val_ground_truth": SPLITS_DIR / "split_val_ground_truth.tsv",
}


# Column Schemas
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
MATCHING_RESULT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_PAIR_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]

# Parsing Encodings & Delimiters
DELIMITER = "\t"
ENCODING = "utf-8"

# Evaluation Metric Parameters
F_BETA = 0.5
BETA_SQ = F_BETA ** 2  # 0.25

# Performance & Streaming Tuning Constants
DEFAULT_CHUNK_SIZE = 100_000
MAX_BLOCK_SIZE = 300
DEFAULT_RANDOM_SEED = 42
