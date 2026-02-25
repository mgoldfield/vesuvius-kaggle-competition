#!/bin/bash
# Pass 1: ep7, ep10, ep13, ep15 + baseline eval, then STOP
# ep4 already done (comp=0.5285)
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EVAL_CMD="/workspace/venv/bin/python3 scripts/eval_transunet.py --n-eval 20 --t-low 0.70 --t-high 0.90"
CKPT_DIR="checkpoints/transunet_external_data_frozen"

echo "=== Pass 1 + baseline eval started at $(date) ==="
echo "=== PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF ==="

# Pass 1 remaining: ep7, ep10, ep13, ep15
for ep in 7 10 13 15; do
    WEIGHTS="$CKPT_DIR/transunet_ep${ep}.weights.h5"
    if [ -f "$WEIGHTS" ]; then
        echo ""
        echo "=== [Pass 1] Epoch $ep ==="
        $EVAL_CMD --weights "$WEIGHTS" 2>&1 | tee "logs/eval_external_data_ep${ep}.log"
    fi
done

# Baseline eval
echo ""
echo "=== BASELINE: swa_70pre_30margin_dist_ep5 (current best, val scroll only) ==="
$EVAL_CMD --weights checkpoints/swa_topo/swa_70pre_30margin_dist_ep5.weights.h5 2>&1 | tee "logs/eval_baseline_val_only.log"

# Summary
echo ""
echo "=== PASS 1 SUMMARY (val scroll 26002 only, 20 vols) ==="
for ep in 4 7 10 13 15; do
    LOG="logs/eval_external_data_ep${ep}.log"
    if [ -f "$LOG" ]; then
        COMP=$(grep "comp_score:" "$LOG" | tail -1 | awk '{print $2}')
        SDICE=$(grep "sdice:" "$LOG" | tail -1 | awk '{print $2}')
        TOPO=$(grep "topo:" "$LOG" | tail -1 | awk '{print $2}')
        echo "ep${ep}: comp=$COMP sdice=$SDICE topo=$TOPO"
    fi
done
LOG="logs/eval_baseline_val_only.log"
if [ -f "$LOG" ]; then
    COMP=$(grep "comp_score:" "$LOG" | tail -1 | awk '{print $2}')
    SDICE=$(grep "sdice:" "$LOG" | tail -1 | awk '{print $2}')
    TOPO=$(grep "topo:" "$LOG" | tail -1 | awk '{print $2}')
    echo "BASELINE: comp=$COMP sdice=$SDICE topo=$TOPO"
fi

echo ""
echo "=== Pass 1 + baseline complete at $(date) ==="
echo "=== PAUSING - review results before continuing to Pass 2 ==="
