# Reproducibility Guide

This document provides step-by-step instructions to reproduce all results
from the paper **"Formalizing the Accuracy–Fairness–Efficiency Trilemma
in Mobile Image Classification"**.

---

## 1. Environment Setup

```bash
git clone https://gitlab.com/hamanhtoan/autism-dlm.git
cd autism-dlm
pip install -e ".[3d_aug,mobile,dev]"
```

Verified with: Python 3.9, PyTorch 1.13.1, CUDA 11.8, Ubuntu 20.04.

---

## 2. Dataset Preparation

### 2.1 Kaggle dataset
Download from: https://www.kaggle.com/datasets/gpiosenka/autistic-children-data-set-traintestvalidate

```bash
mkdir -p data/raw/kaggle/asd data/raw/kaggle/non_asd
# Copy images to respective folders
```

### 2.2 Vietnamese dataset
699 images collected via the VN-ASDetect mobile application.
Available upon request (contact authors for data access agreement).

```bash
mkdir -p data/raw/vietnamese/asd data/raw/vietnamese/non_asd
```

### 2.3 Preprocessing (4-stage pipeline)

```bash
python scripts/preprocess.py \
    --data_dir data/raw \
    --output_dir data/processed \
    --create_splits
```

Expected output:
```
[Preprocessing] Found 2,936 candidate images
[Preprocessing] Done. Retention: 2821/2936 = 96.1%
[Splits] train=2258, val=281, test=282
```

---

## 3. Training All 11 Configurations

```bash
# Run all configs with seed=42 (paper default)
bash scripts/run_all_configs.sh
```

Or train a single configuration:

```bash
# C2: M7 + PFP (Pareto knee point)
python scripts/train.py \
    --config configs/C2_m7_pfp.yaml \
    --seed 42 \
    --output_dir outputs/C2
```

**Training time estimates** (RTX 3080 Ti):

| Config | Augmentation | Compression | ~Time |
|--------|-------------|-------------|-------|
| A1     | 2D basic    | None        | 2h    |
| A2     | 3D-aware    | None        | 6h    |
| A3     | 2D basic    | Mag.Prune   | 2.5h  |
| B1     | 2D basic    | PFP         | 3h    |
| B2     | 3D-aware    | INT8        | 6.5h  |
| B3     | 3D-aware    | KD          | 8h    |
| B4     | 2D basic    | PFP+INT8    | 3.5h  |
| C1     | 3D-aware    | Mag.Prune   | 6.5h  |
| C2     | 3D-aware    | PFP         | 7h    |
| C3     | 3D-aware    | INT8        | 6.5h  |
| C4     | 2D basic    | PFP         | 3h    |

---

## 4. Evaluation

```bash
# Evaluate all configurations
python scripts/evaluate.py --all --results_dir outputs/

# Pareto analysis + Table 3-5 reproduction
python scripts/pareto_analysis.py \
    --results_dir outputs/ \
    --bootstrap_n 1000
```

---

## 5. Mobile Deployment (E3 Latency)

```bash
# Export to TFLite
python scripts/export_tflite.py \
    --checkpoint outputs/C2/C2_final.pt \
    --output outputs/C2/C2.tflite

# Benchmark on E3 (Redmi Note 11, Helio G88)
adb push outputs/C2/C2.tflite /data/local/tmp/
adb shell "benchmark_model \
    --graph=/data/local/tmp/C2.tflite \
    --num_threads=4 \
    --num_runs=500 \
    --warmup_runs=20"
```

---

## 6. Expected Results (Paper Table 3)

After full training and evaluation, `outputs/pareto_analysis.json` should
contain:

```json
{
  "knee_point": "C2",
  "knee_metrics": {
    "f1": 0.934,
    "eod_gender": 0.020,
    "eod_eth": 0.037,
    "size_mb": 6.3,
    "latency_ms": 187
  },
  "dfz_qualified": ["B1","B2","B3","B4","C2","C3","C4"],
  "bootstrap_stability": {"C2": 0.942, "C3": 0.058}
}
```

---

## 7. Running Unit Tests

```bash
pytest tests/ -v --cov=src
```

Key test assertions:
- `test_pfp.py::TestPareto::test_knee_point_is_c2` — verifies C2 is
  the ‖L‖₂-minimizing configuration among DFZ-qualified configs
- `test_metrics.py::TestDFZQualification::test_per_attribute_not_aggregate`
  — verifies per-attribute EOD constraint logic (Eq. 8b)
- `test_pfp.py::TestATWS::test_weights_sum_to_one` — verifies ℓ₁
  normalization after every ATWS step

---

## 8. Notebook Walkthroughs

```bash
cd notebooks
jupyter lab
```

- `01_dataset_eda.ipynb` — Table 1, imbalance heatmap
- `02_pareto_visualization.ipynb` — 3D Pareto frontier, ternary sensitivity
- `03_fairness_analysis.ipynb` — Table 4, EOD bar chart, ATWS ablation

---

## Key Design Decisions

### Why is the test set accessed only once?

The test partition (n=282) is fixed before any augmentation
(paper Section 4.1) and accessed **exactly once per configuration**
after CV-based model selection. This prevents:
1. Multiple testing inflation (test set "peeking")
2. Bootstrap CI computation uses only test set data — never influences
   model selection

### Why shared hyperparameters across 11 configs?

Section 4.3: "Configuration-specific hyperparameter tuning was
deliberately withheld to provide a controlled comparison of optimization
strategies rather than a comparison of hyperparameter tuning effectiveness."

B3 (knowledge distillation) may be most affected — default T=4, λ=0.7
from Hinton et al. are not tuned to this dataset (Section 6.3).

### Why per-attribute EOD constraint (not aggregate L_fair)?

Section 3.3: A config with EOD_gender=0.02 and EOD_ethnicity=0.18
would have L_fair=0.10 (acceptable aggregate) but violates Eq. (8b)
(per-attribute). Per-attribute constraints prevent one group from
being systematically disadvantaged even when aggregate fairness
appears acceptable.

---

## Citation

```bibtex
@article{thanh2025trilemma,
  title={Formalizing the Accuracy--Fairness--Efficiency Trilemma in Mobile
         Image Classification: A Pareto Benchmark for Demographic-Constrained Deployment},
  author={Tran Van Thanh and Nguyen Van An and Nguyen Van Anh},
  journal={arXiv preprint arXiv:2025.XXXXX},
  year={2025}
}
```
