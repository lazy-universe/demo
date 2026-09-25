# 📍 Project Checkpoint & Milestone Tracker
**Amazon ML Challenge 2026: Business Entity Resolution**  
*Last Updated: September 25, 2026*

---

## 🚦 Executive Status Summary

| Phase | Description | Status | Key Deliverables / Artifacts |
| :--- | :--- | :---: | :--- |
| **Phase 1** | Dataset Audit & Exploratory Data Analysis | ✅ **Completed** | [`01_dataset_exploration_and_eda.ipynb`](file:///workspaces/amazon-ml-challenge/01_dataset_exploration_and_eda.ipynb), [`docs/01_PROBLEM_AND_METRIC.md`](file:///workspaces/amazon-ml-challenge/docs/01_PROBLEM_AND_METRIC.md), [`docs/02_DATASET_AND_ARCHITECTURE.md`](file:///workspaces/amazon-ml-challenge/docs/02_DATASET_AND_ARCHITECTURE.md) |
| **Phase 2** | Preprocessing & Multi-Key Inverted Index Blocking | ✅ **Completed** | [`src/preprocessing/`](file:///workspaces/amazon-ml-challenge/src/preprocessing/), [`src/blocking/`](file:///workspaces/amazon-ml-challenge/src/blocking/) (Dynamic open-set country prefix partitioning) |
| **Phase 3** | Pairwise Feature Engineering (16 Tabular Signals) | ✅ **Completed** | [`src/features/extractor.py`](file:///workspaces/amazon-ml-challenge/src/features/extractor.py), [`src/features/builder.py`](file:///workspaces/amazon-ml-challenge/src/features/builder.py) |
| **Phase 4** | Zero-Leakage 80/20 Holdout Split | ✅ **Completed** | [`dataset/splits/`](file:///workspaces/amazon-ml-challenge/dataset/splits/) (1.76M Train S1 / 441k Val S1) |
| **Phase 5** | Multi-Model Benchmark & Threshold Calibration | ✅ **Completed** | [`02_validation_split_and_multimodel_benchmark.ipynb`](file:///workspaces/amazon-ml-challenge/02_validation_split_and_multimodel_benchmark.ipynb), [`src/models/`](file:///workspaces/amazon-ml-challenge/src/models/), [`src/evaluation/`](file:///workspaces/amazon-ml-challenge/src/evaluation/) |
| **Phase 6** | Full-Scale Test Set Inference Pipeline | 🔄 **In Progress** | [`src/pipeline/runner.py`](file:///workspaces/amazon-ml-challenge/src/pipeline/runner.py), Full test batch processing |
| **Phase 7** | Official Submission Export & Validation | ⏳ **Next Step** | [`output/matching_results.tsv`](file:///workspaces/amazon-ml-challenge/output/matching_results.tsv), [`output/candidate_pairs.tsv`](file:///workspaces/amazon-ml-challenge/output/candidate_pairs.tsv) |
| **Phase 8** | Final Documentation & Packaging | ⏳ **Next Step** | [`Documentation_template.md`](file:///workspaces/amazon-ml-challenge/dataset/student_resource/Documentation_template.md) |

---

## 📊 Benchmark & Validation Progress

### 1. Multi-Model Benchmark on 20% Holdout Validation Split

| Rank | Model Architecture | Optimal Threshold ($\tau^*$) | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Singleton Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Blended Ensemble** (LGBM 40% + XGB 30% + CatBoost 30%) | **0.70** | **0.9899** | **0.9918** | **0.9823** | **1.0000** |
| 🥈 | **LightGBM Classifier** | **0.70** | **0.9896** | **0.9913** | **0.9827** | **1.0000** |
| 🥉 | **CatBoost Classifier** | **0.70** | **0.9885** | **0.9910** | **0.9785** | **1.0000** |
| 4 | **XGBoost Classifier** | **0.70** | **0.9877** | **0.9904** | **0.9772** | **1.0000** |
| 5 | **Regularized Logistic Regression** | **0.60** | **0.9734** | **0.9744** | **0.9696** | **1.0000** |

### 2. Candidate Generation (Blocking) Efficiency
- **Pair Reduction Ratio:** **> 99.998%** (pruned from $10^{13}$ Cartesian space down to $< 35$ candidates per entity).
- **Pair Recall on Validation Links:** **98.6%+**.
- **Dynamic Country Partitioning:** 100% intra-country isolation via `f"{country}:np:{prefix}"`, preventing impossible cross-country comparisons.

### 3. Metric Optimization Properties
- Macro $F_{0.5}$ penalizes False Positives with weight $\beta^2 = 0.25$ ($2\times$ heavier penalty than false negatives).
- Singletons (5.58% in dataset) achieve exact **1.0000** when predicted as empty lists.
- Post-processing enforces **Injective Matching** (no external S2 or S3 entity is mapped to multiple S1 records).

---

## 🛠️ Detailed Component Checklist

### A. Data & Preprocessing
- [x] Ingest all 7 TSVs via low-memory streaming and DuckDB zero-copy sampling (`src/data/loader.py`).
- [x] Unicode NFD diacritic normalizer stripping accents across French, Spanish, German (`src/preprocessing/text.py`).
- [x] Multi-lingual corporate legal suffix remover (`LLC`, `Pvt Ltd`, `SAS`, `GmbH`, `SA`, `SARL`).
- [x] Universal 5/6 digit postal code regex extractor (`src/preprocessing/tokenizer.py`).

### B. Blocking & Feature Engineering
- [x] Multi-key inverted index blocker with block-size capping (`src/blocking/inverted_index.py`).
- [x] 16 Tabular similarity features (`RapidFuzz` sort/set/partial, 3-gram Jaccard, token Jaccard, address number overlap) (`src/features/extractor.py`).
- [x] Pairwise training dataset builder combining true links and blocker hard negatives (`src/features/builder.py`).

### C. Validation & Modeling
- [x] Isolated 80/20 train/validation holdout split created on disk (`dataset/splits/`).
- [x] 4 ML Classifiers implemented + probability fusion ensemble (`src/models/`).
- [x] Fast grid threshold tuner optimizing for Macro $F_{0.5}$ (`src/evaluation/threshold_tuner.py`).
- [x] Injective 1-to-1 conflict resolver (`apply_injective_matching`).

### D. Full Pipeline & Final Submission (Current Focus)
- [ ] End-to-end streamed test set inference runner across all 1,732,544 test S1 entities.
- [ ] Memory-safe batch candidate generation & prediction (< 1.5 GB RAM footprint).
- [ ] Export `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- [ ] Validate submission with official `validate_submission.py` (Verify 0 errors, 1,732,544 rows).
- [ ] Fill official `Documentation_template.md` with methodology, ablation scores, and architecture.

---

## 🧭 Immediate Next Action
Execute the full-scale streaming test inference pipeline to produce the official submission files for all 1,732,544 test records and verify with the official submission validator.
