#!/bin/bash
# Quick evaluation sweep: compare pretrained vs 3 fine-tuned models
# ~10-12 min per model × 4 = ~45 min total
set -e

PYTHON="/workspace/venv/bin/python"
EVAL="scripts/eval_transunet.py"
N=20

echo "============================================"
echo "Model Comparison Sweep (${N} volumes each, no TTA, T_low=0.70)"
echo "Started: $(date)"
echo "============================================"

echo ""
echo "=== Model 1/4: Pretrained (comboloss) ==="
$PYTHON $EVAL --weights pretrained_weights/transunet/transunet.seresnext50.160px.comboloss.weights.h5 \
    --n-eval $N --t-low 0.70 2>&1 | tee logs/eval_pretrained.log

echo ""
echo "=== Model 2/4: Run 1 — baseline (SkelRecall=0.75, FP=0.50) ==="
$PYTHON $EVAL --weights checkpoints/transunet/transunet_best.weights.h5 \
    --n-eval $N --t-low 0.70 2>&1 | tee logs/eval_baseline.log

echo ""
echo "=== Model 3/4: Run 2 — thin_fp (FP=1.5) ==="
$PYTHON $EVAL --weights checkpoints/transunet_thin_fp/transunet_best.weights.h5 \
    --n-eval $N --t-low 0.70 2>&1 | tee logs/eval_thin_fp.log

echo ""
echo "=== Model 4/4: Run 3 — thin_dist (FP=1.5, Dist=1.0) ==="
$PYTHON $EVAL --weights checkpoints/transunet_thin_dist/transunet_best.weights.h5 \
    --n-eval $N --t-low 0.70 2>&1 | tee logs/eval_thin_dist.log

echo ""
echo "============================================"
echo "Sweep Complete: $(date)"
echo "============================================"
