# Amazon ML Challenge 2026: Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Metric: Macro F0.5](https://img.shields.io/badge/Metric-Macro%20F0.5-brightgreen.svg)]()
[![Validation: Passed](https://img.shields.io/badge/Validation-Pass-success.svg)]()

A modular, reproducible, and memory-efficient machine learning solution for **Business Entity Resolution** across three heterogeneous, noisy data sources.

---

## 📌 Project Overview

When business records arrive from multiple disparate providers, they often lack shared unique keys. This project implements a high-recall **candidate generation (blocking)** engine coupled with a precision-optimized **multi-model matching pipeline** to link external records from **Source 2** and **Source 3** back to reference records in **Source 1**.

### Core Dataset Highlights:
- **Total Records:** 26.4+ Million rows across 7 `.tsv` files.
- **Geography:** Training set covers **US (59.98%)** and **India (40.02%)**. Test set introduces unseen **France (14.40%)**.
- **Country Boundary:** 100% intra-country matches (zero cross-border linkages; dynamic open-set country partitioning).
- **Singleton Rate:** **5.58%** of Source 1 entities have 0 matches (crucial for scoring 1.0 on Macro $F_{0.5}$).
- **Evaluation Metric:** **Macro $F_{0.5}$** (Precision weighted $2\times$ over recall).

---

## 📁 Repository Structure

```text
.
├── 01_dataset_exploration_and_eda.ipynb              # Comprehensive Exploratory Data Analysis & baseline walkthrough
├── 02_validation_split_and_multimodel_benchmark.ipynb # 80/20 train/val holdout split, 4 ML models, threshold optimization
├── WALKTHROUGH.md                                     # Structured chronological guide to all documentation & code
├── README.md                                          # Repository overview & quick start
├── docs/                                              # In-depth architectural & conceptual guide
│   ├── 01_PROBLEM_AND_METRIC.md                       # Entity resolution mechanics & Macro F0.5 properties
│   ├── 02_DATASET_AND_ARCHITECTURE.md                 # Complete schema, file sizes, and streaming design
│   ├── 03_CODEBASE_AND_SRC_MODULES.md                 # Modular code specification & pipeline breakdown
│   ├── 04_NOTEBOOK_BLOCK_BY_BLOCK.md                  # Block-by-block explanation of notebooks
│   └── 05_EMPIRICAL_FINDINGS_AND_STRATEGY.md          # Noise patterns, country dynamics & ML strategy
├── dataset/
│   ├── README.md                                      # Dataset directory guide & noise patterns
│   ├── splits/                                        # Isolated 80/20 train/validation holdout splits
│   └── student_resource/                              # Official dataset files, train/test, and validation tool
├── output/
│   ├── matching_results.tsv                           # Official submission matching predictions
│   └── candidate_pairs.tsv                            # Official submission candidate pairs audit
└── src/                                               # Modular Python sub-packages
    ├── __init__.py                                    # Central package re-exports
    ├── config.py                                      # Paths, schemas, chunk sizes, F0.5 parameters
    ├── data/                                          # Data loading, DuckDB sampling & 80/20 splitter
    │   ├── eda.py
    │   ├── loader.py
    │   └── validation_split.py
    ├── preprocessing/                                 # Unicode NFD normalization, tokenizers & regex
    │   ├── text.py
    │   └── tokenizer.py
    ├── blocking/                                      # Dynamic inverted index & recall evaluator
    │   ├── evaluator.py
    │   └── inverted_index.py
    ├── features/                                      # 16 similarity features & pairwise dataset builder
    │   ├── builder.py
    │   └── extractor.py
    ├── models/                                        # LightGBM, XGBoost, CatBoost, LogReg & Ensemble
    │   ├── baseline.py
    │   ├── classifiers.py
    │   └── ensemble.py
    ├── evaluation/                                    # Exact Macro F0.5 engine & threshold tuner
    │   ├── metrics.py
    │   └── threshold_tuner.py
    └── pipeline/                                      # TSV exporters & submission formatters
        └── runner.py
```

---

## 🚀 Quick Start & Reproduction

### 1. Install Dependencies
```bash
pip install pandas numpy polars duckdb scikit-learn matplotlib seaborn rapidfuzz tabulate lightgbm xgboost catboost ipykernel
```

### 2. Run Modular Python Pipeline
```python
from src.data import load_tsv, load_ground_truth_dict
from src.blocking import MultiKeyBlocker
from src.features import build_pairwise_dataset
from src.models import train_lightgbm, predict_pair_probabilities
from src.evaluation import optimize_threshold
from src.pipeline import format_predictions_to_dataframe, save_submission_files

# 1. Load data
s1 = load_tsv("dataset/splits/split_train_source1.tsv", nrows=10000)
s2 = load_tsv("dataset/student_resource/dataset/train/train_source2.tsv", nrows=50000)
s3 = load_tsv("dataset/student_resource/dataset/train/train_source3.tsv", nrows=50000)

# 2. Block candidates
blocker = MultiKeyBlocker(max_block_size=200)
blocker.index_sources(s2, s3)
candidates = blocker.query_candidates(s1, max_candidates_per_entity=50)

# 3. Build features & train classifier
X, y, meta = build_pairwise_dataset(s1, s2, s3, candidates)
model = train_lightgbm(X, y)
```

### 3. Validate Submission Locally
```bash
python3 dataset/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/student_resource/dataset/test
```
