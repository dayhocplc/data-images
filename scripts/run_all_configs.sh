#!/usr/bin/env bash
# scripts/run_all_configs.sh
# Run all 11 benchmark configurations sequentially.
# Paper Section 4.4: "All configurations share DenseNet-121 backbone
# and identical training hyperparameters."
#
# Usage:
#   bash scripts/run_all_configs.sh
#   bash scripts/run_all_configs.sh --gpu 0   # specify GPU
#   bash scripts/run_all_configs.sh --dry_run  # print commands only

set -euo pipefail

GPU=${GPU:-0}
SEED=${SEED:-42}
DRY_RUN=${DRY_RUN:-false}
OUTPUT_ROOT=${OUTPUT_ROOT:-"outputs"}
PYTHON=${PYTHON:-"python"}

CONFIGS=(
    "configs/A1_basic_aug.yaml"
    "configs/A2_3d_aug.yaml"
    "configs/A3_std_pruning.yaml"
    "configs/B1_pfp.yaml"
    "configs/B2_int8_quant.yaml"
    "configs/B3_kd_mobilenet.yaml"
    "configs/B4_pfp_int8.yaml"
    "configs/C1_m7_std_prune.yaml"
    "configs/C2_m7_pfp.yaml"
    "configs/C3_m7_int8.yaml"
    "configs/C4_m1_pfp.yaml"
)

echo "============================================================"
echo "Trilemma Benchmark — 11 Configurations"
echo "Seed: ${SEED} | GPU: ${GPU} | Output: ${OUTPUT_ROOT}"
echo "============================================================"

FAILED=()

for config in "${CONFIGS[@]}"; do
    config_id=$(python -c "
import yaml, sys
with open('${config}') as f:
    c = yaml.safe_load(f)
print(c.get('config_id', '${config}'))
" 2>/dev/null || echo "${config}")

    echo ""
    echo "── Config: ${config_id} ──────────────────────────"
    echo "   File: ${config}"

    CMD="CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} scripts/train.py \
        --config ${config} \
        --seed ${SEED} \
        --output_dir ${OUTPUT_ROOT}/${config_id}"

    if [ "${DRY_RUN}" = "true" ]; then
        echo "   [DRY RUN] ${CMD}"
        continue
    fi

    if eval "${CMD}"; then
        echo "   ✓ ${config_id} completed"
    else
        echo "   ✗ ${config_id} FAILED"
        FAILED+=("${config_id}")
    fi
done

echo ""
echo "============================================================"
echo "Benchmark complete."
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED configurations: ${FAILED[*]}"
    exit 1
else
    echo "All 11 configurations completed successfully."
fi

echo ""
echo "Running Pareto analysis..."
${PYTHON} scripts/pareto_analysis.py \
    --results_dir "${OUTPUT_ROOT}" \
    --output "${OUTPUT_ROOT}/pareto_results.json"

echo "Results saved to ${OUTPUT_ROOT}/pareto_results.json"
