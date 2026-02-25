#!/bin/bash
# Cross-scroll eval (old way) for ep4 and ep10 - 24 vols
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EVAL_CMD="/workspace/venv/bin/python3 scripts/eval_transunet.py --cross-scroll --max-per-scroll 4 --t-low 0.70 --t-high 0.90"
CKPT_DIR="checkpoints/transunet_external_data_frozen"

echo "=== Cross-scroll eval (24 vols, old method) at $(date) ==="

echo ""
echo "=== Epoch 4 (cross-scroll) ==="
$EVAL_CMD --weights "$CKPT_DIR/transunet_ep4.weights.h5" 2>&1 | tee "logs/eval_external_data_ep4_cross.log"

echo ""
echo "=== Epoch 10 (cross-scroll) ==="
$EVAL_CMD --weights "$CKPT_DIR/transunet_ep10.weights.h5" 2>&1 | tee "logs/eval_external_data_ep10_cross.log"

echo ""
echo "=== CROSS-SCROLL SUMMARY ==="
for ep in 4 10; do
    LOG="logs/eval_external_data_ep${ep}_cross.log"
    if [ -f "$LOG" ]; then
        COMP=$(grep "comp_score:" "$LOG" | tail -1 | awk '{print $2}')
        SDICE=$(grep "sdice:" "$LOG" | tail -1 | awk '{print $2}')
        TOPO=$(grep "topo:" "$LOG" | tail -1 | awk '{print $2}')
        VOI=$(grep "voi:" "$LOG" | tail -1 | awk '{print $2}')
        echo "ep${ep}: comp=$COMP sdice=$SDICE topo=$TOPO voi=$VOI"
    fi
done
echo "(Baseline cross-scroll: comp=0.5551 sdice=0.8299 topo=0.2477)"

echo ""
echo "=== Cross-scroll eval complete at $(date) ==="
