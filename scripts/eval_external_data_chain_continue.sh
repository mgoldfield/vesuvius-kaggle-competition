#!/bin/bash
# Continue eval chain from ep7 with PYTORCH_CUDA_ALLOC_CONF to prevent OOM
# ep1: OOM'd (will re-run at end)
# ep4: DONE (comp=0.5285)
set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EVAL_CMD="/workspace/venv/bin/python3 scripts/eval_transunet.py --n-eval 20 --t-low 0.70 --t-high 0.90"
CKPT_DIR="checkpoints/transunet_external_data_frozen"

echo "=== Eval chain CONTINUED at $(date) ==="
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

echo ""
echo "=== PASS 1 SUMMARY (val scroll only) ==="
for ep in 1 4 7 10 13 15; do
    LOG="logs/eval_external_data_ep${ep}.log"
    if [ -f "$LOG" ]; then
        COMP=$(grep "comp_score:" "$LOG" | tail -1 | awk '{print $2}')
        SDICE=$(grep "sdice:" "$LOG" | tail -1 | awk '{print $2}')
        TOPO=$(grep "topo:" "$LOG" | tail -1 | awk '{print $2}')
        echo "ep${ep}: comp=$COMP sdice=$SDICE topo=$TOPO"
    fi
done

# Pass 2: fill in remaining epochs
for ep in 2 3 5 6 8 9 11 12 14; do
    WEIGHTS="$CKPT_DIR/transunet_ep${ep}.weights.h5"
    if [ -f "$WEIGHTS" ]; then
        echo ""
        echo "=== [Pass 2] Epoch $ep ==="
        $EVAL_CMD --weights "$WEIGHTS" 2>&1 | tee "logs/eval_external_data_ep${ep}.log"
    fi
done

# Re-run ep1 (OOM'd earlier without ALLOC_CONF)
WEIGHTS="$CKPT_DIR/transunet_ep1.weights.h5"
if [ -f "$WEIGHTS" ]; then
    echo ""
    echo "=== [Re-run] Epoch 1 ==="
    $EVAL_CMD --weights "$WEIGHTS" 2>&1 | tee "logs/eval_external_data_ep1.log"
fi

# Baseline eval for comparison
echo ""
echo "=== BASELINE: swa_70pre_30margin_dist_ep5 (current best, val scroll only) ==="
$EVAL_CMD --weights checkpoints/swa_topo/swa_70pre_30margin_dist_ep5.weights.h5 2>&1 | tee "logs/eval_baseline_val_only.log"

echo ""
echo "=== FULL SUMMARY (all epochs + baseline, val scroll only) ==="
echo "--- Baseline ---"
LOG="logs/eval_baseline_val_only.log"
if [ -f "$LOG" ]; then
    COMP=$(grep "comp_score:" "$LOG" | tail -1 | awk '{print $2}')
    SDICE=$(grep "sdice:" "$LOG" | tail -1 | awk '{print $2}')
    TOPO=$(grep "topo:" "$LOG" | tail -1 | awk '{print $2}')
    echo "BASELINE: comp=$COMP sdice=$SDICE topo=$TOPO"
fi
echo "--- External data frozen ---"
for ep in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    LOG="logs/eval_external_data_ep${ep}.log"
    if [ -f "$LOG" ]; then
        COMP=$(grep "comp_score:" "$LOG" | tail -1 | awk '{print $2}')
        SDICE=$(grep "sdice:" "$LOG" | tail -1 | awk '{print $2}')
        TOPO=$(grep "topo:" "$LOG" | tail -1 | awk '{print $2}')
        echo "ep${ep}: comp=$COMP sdice=$SDICE topo=$TOPO"
    fi
done

echo ""
echo "=== Eval chain complete at $(date) ==="
