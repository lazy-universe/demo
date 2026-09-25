# 🚀 Google Colab Master Execution Guide (90/10 Split & High-Speed Engine)

This document contains each clean code block ready to copy and paste directly into your Google Colab notebook cells.

---

### **[Cell 1] Hardware & GPU Specs Check**
```python
import torch
import os

print("=== Hardware Specifications ===")
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "Running in CPU/High-RAM mode"
!free -h

has_gpu = torch.cuda.is_available()
print(f"\nPyTorch CUDA Available: {has_gpu}")
if has_gpu:
    print(f"Active GPU Device: {torch.cuda.get_device_name(0)}")
```

---

### **[Cell 2] Clone Repository & Enter Directory**
```python
import os

REPO_URL = "https://github.com/lazy-universe/demo.git"
REPO_NAME = "demo"

if not os.path.exists(REPO_NAME):
    !git clone {REPO_URL}
    %cd {REPO_NAME}
else:
    %cd {REPO_NAME}
    !git pull

!pwd
```

---

### **[Cell 3] Install Dependencies**
```python
!pip install -q -r requirements.txt
print("✓ All required libraries installed successfully!")
```

---

### **[Cell 4] Download & Extract Official Dataset**
```python
DATASET_ZIP_URL = "https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip"

dataset_test_s1 = "dataset/student_resource/dataset/test/test_source1.tsv"

if not os.path.exists(dataset_test_s1):
    os.makedirs("dataset/student_resource", exist_ok=True)
    print("Downloading official dataset zip...")
    !wget -q --show-progress "{DATASET_ZIP_URL}" -O dataset.zip
    print("Extracting dataset files...")
    !unzip -q -o dataset.zip -d dataset/
    print("✓ Dataset ready!")
else:
    print("✓ Dataset already extracted and ready!")
```

---

### **[Cell 5] Train & Calibrate Models (90/10 Split - 1.98M Train Entities)**
```python
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import src
from src.data.validation_split import ValidationSplitManager

# 1. Ensure isolated 90/10 split exists (1.98M Train / 220k Val)
split_mgr = ValidationSplitManager(train_ratio=0.90, random_seed=42)
split_mgr.create_splits()

# 2. Train and tune threshold (with GPU acceleration if available)
# Using 50,000 stratified training S1 entities (~400,000 candidate pairs with hard negatives)
artifacts = src.train_and_validate_pipeline(
    train_s1_samples=50000,
    val_s1_samples=10000,
    random_state=42,
    save_path="models/pipeline_artifacts.joblib",
    load_cached=True
)

print("\n=== Benchmark Performance on 10% Holdout Split ===")
display(artifacts["benchmark_summary"])
```

---

### **[Cell 6] Full Streaming Test Inference (1,732,544 Entities | 1.5M Segment Size)**
```python
# Runs high-speed streaming test inference across France, India, and US
!python3 generate_final_submission.py
```

---

### **[Cell 7] Official Submission Validation**
```python
!python3 dataset/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/student_resource/dataset/test
```

---

### **[Cell 8] Create and Auto-Download Submission Package**
```python
import zipfile
from google.colab import files

submission_zip = "submission_package.zip"

with zipfile.ZipFile(submission_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    if os.path.exists("output/matching_results.tsv"):
        zipf.write("output/matching_results.tsv", arcname="output/matching_results.tsv")
    if os.path.exists("output/candidate_pairs.tsv"):
        zipf.write("output/candidate_pairs.tsv", arcname="output/candidate_pairs.tsv")
    if os.path.exists("DOCUMENTATION.md"):
        zipf.write("DOCUMENTATION.md", arcname="DOCUMENTATION.md")
    for root, dirs, filenames in os.walk("src"):
        for filename in filenames:
            if filename.endswith((".py", ".md")):
                filepath = os.path.join(root, filename)
                zipf.write(filepath, arcname=filepath)

zip_size_mb = os.path.getsize(submission_zip) / (1024 * 1024)
print(f"✓ Submission package created: {submission_zip} ({zip_size_mb:.2f} MB)")

# Trigger automatic browser download
files.download(submission_zip)
```
