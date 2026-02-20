#!/bin/bash
# Quick eval of early checkpoints to check for catastrophic forgetting
set -e

PYTHON="/workspace/venv/bin/python"
EVAL="scripts/eval_transunet.py"
N=10

echo "============================================"
echo "Early Checkpoint Sweep ($N volumes each)"
echo "Started: $(date)"
echo "============================================"

for CKPT in \
    "checkpoints/transunet/transunet_ep1.weights.h5" \
    "checkpoints/transunet/transunet_ep5.weights.h5" \
    "checkpoints/transunet_thin_fp/transunet_ep5.weights.h5" \
    "checkpoints/transunet_thin_dist/transunet_ep5.weights.h5" \
; do
    NAME=$(echo "$CKPT" | sed 's|checkpoints/transunet_\?||;s|/transunet_||;s|\.weights\.h5||')
    echo ""
    echo "=== $NAME ==="
    $PYTHON $EVAL --weights "$CKPT" --n-eval $N --t-low 0.70 2>&1
    echo ""
done

echo "============================================"
echo "Early Checkpoint Sweep Complete: $(date)"
echo "============================================"
