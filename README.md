# Accuracy–Fairness–Efficiency Trilemma in Mobile Image Classification

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.13.1-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Official implementation of:

> **Formalizing the Accuracy–Fairness–Efficiency Trilemma in Mobile Image Classification:
> A Pareto Benchmark for Demographic-Constrained Deployment**  
> Tran Van Thanh · Nguyen Van An · Nguyen Van Anh  
> Faculty of Information Technology, Lac Hong University, Bien Hoa, Vietnam

---

## Overview

This repository provides a complete, reproducible implementation of:

- **DFZ (Deployment-Feasible Zone)** — feasibility-constrained Pareto frontier concept
- **PFP (Protected Fairness Pruning)** — fairness-aware model compression
- **3D-aware augmentation pipeline** — DECA/FLAME-based minority-stratified oversampling
- **ATWS (Adaptive Trilemma Weight Scheduler)** — phased gradient-conflict-aware training
- **Full Pareto benchmark** — 11 optimization configurations (A1–A3, B1–B4, C1–C4)

All 11 configurations share a fixed DenseNet-121 backbone and identical training
hyperparameters to isolate the contribution of each optimization strategy.

---

## Project Structure

```
trilemma/
├── configs/               # YAML configs for all 11 configurations
│   ├── base.yaml          # Shared hyperparameters
│   ├── A1_basic_aug.yaml
│   ├── A2_3d_aug.yaml
│   ├── A3_std_pruning.yaml
│   ├── B1_pfp.yaml
│   ├── B2_int8_quant.yaml
│   ├── B3_kd_mobilenet.yaml
│   ├── B4_pfp_int8.yaml
│   ├── C1_m7_std_prune.yaml
│   ├── C2_m7_pfp.yaml
│   ├── C3_m7_int8.yaml
│   └── C4_m1_pfp.yaml
├── src/
│   ├── data/
│   │   ├── dataset.py     # ASD facial image dataset loader
│   │   ├── preprocessing.py   # MediaPipe face detection + quality filter
│   │   └── splits.py     # Stratified 80/10/10 partitioning
│   ├── models/
│   │   ├── backbone.py    # DenseNet-121 with classification head
│   │   └── student.py     # MobileNetV3-Small student model
│   ├── training/
│   │   ├── trainer.py     # Base trainer with ATWS support
│   │   ├── atws.py        # Adaptive Trilemma Weight Scheduler
│   │   └── losses.py      # Multi-component trilemma loss
│   ├── augmentation/
│   │   ├── standard_aug.py    # M1: basic 2D augmentation
│   │   └── aug_3d.py          # M7: 3D-aware DECA/FLAME pipeline
│   ├── compression/
│   │   ├── pfp.py             # Protected Fairness Pruning
│   │   ├── quantization.py    # INT8 post-training quantization (TFLite)
│   │   └── distillation.py    # Knowledge distillation to MobileNetV3
│   └── evaluation/
│       ├── metrics.py         # F1, EOD, DPD, SPG, sensitivity, specificity
│       ├── fairness.py        # Per-attribute fairness evaluation
│       ├── efficiency.py      # Size, FLOPs, latency measurement
│       └── pareto.py          # Pareto frontier + DFZ analysis
├── scripts/
│   ├── run_all_configs.sh     # Run all 11 configurations sequentially
│   ├── train.py               # Single config training entry point
│   ├── evaluate.py            # Evaluation + metrics reporting
│   ├── export_tflite.py       # TFLite export for mobile benchmarking
│   └── pareto_analysis.py     # Generate Pareto frontier plots
├── notebooks/
│   ├── 01_dataset_eda.ipynb
│   ├── 02_pareto_visualization.ipynb
│   └── 03_fairness_analysis.ipynb
├── tests/
│   ├── test_pfp.py
│   ├── test_atws.py
│   ├── test_metrics.py
│   └── test_augmentation.py
├── requirements.txt
├── setup.py
└── .gitlab-ci.yml
```

---

## Installation

```bash
git clone https://gitlab.com/your-username/trilemma.git
cd trilemma
pip install -e .
```

### Requirements

```
torch==1.13.1
torchvision==0.14.1
pytorch-lightning==1.9.0
deca-pytorch>=0.1.0        # For 3D augmentation (M7)
mediapipe>=0.9.0           # Face detection/preprocessing
onnx>=1.13.0
tensorflow>=2.11.0         # TFLite export
ptflops>=0.6.9             # FLOPs measurement
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
pyyaml>=6.0
tqdm>=4.64.0
```

---

## Dataset

The benchmark uses a 2,821-image facial classification dataset:

| Source | Images | Demographics |
|--------|--------|-------------|
| Kaggle Autism Facial Expression | 2,122 | 78.2% White-presenting |
| Vietnamese children (VN-ASDetect app) | 699 | Southeast Asian |
| **Total** | **2,821** | **24 intersectional subgroups** |

Partition (fixed, performed **before** augmentation):

| Split | Images | % |
|-------|--------|---|
| Train | 2,258 | 80% |
| Val | 281 | 10% |
| Test | 282 | 10% |

The test partition is fixed and accessed **once per configuration** after CV-based model selection.

Place dataset at `data/raw/` with structure:
```
data/raw/
├── kaggle/
│   ├── asd/
│   └── non_asd/
└── vietnamese/
    ├── asd/
    └── non_asd/
```

---

## Reproducing the Benchmark

### 1. Preprocess data

```bash
python scripts/preprocess.py --data_dir data/raw --output_dir data/processed
```

### 2. Run a single configuration

```bash
# Example: C2 (M7 + PFP) — the Pareto knee point
python scripts/train.py --config configs/C2_m7_pfp.yaml --seed 42
```

### 3. Run all 11 configurations

```bash
bash scripts/run_all_configs.sh
```

### 4. Evaluate and generate Pareto analysis

```bash
python scripts/evaluate.py --results_dir outputs/
python scripts/pareto_analysis.py --results_dir outputs/
```

---

## Key Results

| Config | F1 | EOD_gender | EOD_eth | Size (MB) | Lat E3 (ms) | DFZ |
|--------|----|-----------|---------|-----------|-------------|-----|
| A1 | 0.506 | 12.1% | 16.2% | 27.8 | 218 | ✗ |
| A2 | 0.946 | 2.1% | 3.9% | 27.8 | 214 | ✗* |
| A3 | 0.874 | 8.3% | 11.6% | 6.3 | 187 | ✗ |
| B1 | 0.904 | 2.1% | 3.9% | 6.3 | 187 | ✓ |
| B2 | 0.919 | 2.4% | 4.2% | 7.1 | 142 | ✓ |
| B3 | 0.891 | 3.1% | 5.0% | 4.2 | 96 | ✓ |
| B4 | 0.887 | 2.3% | 4.0% | 3.1 | 91 | ✓ |
| C1 | 0.921 | 7.9% | 10.8% | 6.3 | 187 | ✗ |
| **C2** | **0.934** | **2.0%** | **3.7%** | **6.3** | **187** | **✓ (knee)** |
| C3 | 0.938 | 2.2% | 3.8% | 7.1 | 142 | ✓ |
| C4 | 0.871 | 2.4% | 4.1% | 6.3 | 187 | ✓ |

*A2: Pareto-optimal but excluded from DFZ (size > 10 MB)

DFZ thresholds: F1 ≥ 0.85, EOD < 10% per attribute, size ≤ 10 MB, latency ≤ 300 ms (E3)

---

## Citation

```bibtex
@article{thanh2025trilemma,
  title={Formalizing the Accuracy--Fairness--Efficiency Trilemma in Mobile
         Image Classification: A Pareto Benchmark for Demographic-Constrained Deployment},
  author={Tran Van Thanh and Nguyen Van An and Nguyen Van Anh},
  journal={arXiv preprint},
  year={2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
