#!/bin/bash
# Eval external_data_frozen checkpoints — val scroll (26002) only
# Pass 1: every 3rd epoch for quick breadth, then fill in
set -e

EVAL_CMD="/workspace/venv/bin/python3 scripts/eval_transunet.py --n-eval 20 --t-low 0.70 --t-high 0.90"
CKPT_DIR="checkpoints/transunet_external_data_frozen"

echo "=== External data eval chain started at $(date) ==="
echo "=== Evaluating on scroll 26002 (val) only, 20 volumes ==="

# Pass 1: every 3rd epoch for quick breadth
for ep in 1 4 7 10 13 15; do
    WEIGHTS="$CKPT_DIR/transunet_ep${ep}.weights.h5"
    if [ -f "$WEIGHTS" ]; then
        echo ""
        echo "=== [Pass 1] Epoch $ep ==="
        $EVAL_CMD --weights "$WEIGHTS" 2>&1 | tee "logs/eval_external_data_ep${ep}.log"
    fi
done

echo ""
echo "=== PASS 1 SUMMARY (every 3rd epoch, val scroll only) ==="
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

echo ""
echo "=== FULL SUMMARY (all epochs, val scroll only) ==="
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
