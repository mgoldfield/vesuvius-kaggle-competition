#!/bin/bash
# Evaluate baseline (swa_70pre_30margin_dist_ep5) on val scroll only
# This gives a fair comparison baseline for the external data checkpoints
set -e

EVAL_CMD="/workspace/venv/bin/python3 scripts/eval_transunet.py --n-eval 20 --t-low 0.70 --t-high 0.90"

echo "=== Baseline eval (val scroll 26002 only, 20 volumes) ==="
echo "=== Model: swa_70pre_30margin_dist_ep5 (current best) ==="
echo "=== Started at $(date) ==="

$EVAL_CMD --weights checkpoints/swa_topo/swa_70pre_30margin_dist_ep5.weights.h5 2>&1 | tee logs/eval_baseline_val_only.log

echo ""
echo "=== Also re-running ep1 (OOM'd on first attempt) ==="
$EVAL_CMD --weights checkpoints/transunet_external_data_frozen/transunet_ep1.weights.h5 2>&1 | tee logs/eval_external_data_ep1.log

echo ""
echo "=== Baseline + ep1 re-run complete at $(date) ==="
