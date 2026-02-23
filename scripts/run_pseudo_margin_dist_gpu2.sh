#!/bin/bash
# GPU2: Pseudo-label training with margin distance loss
# Combines margin dist (best topo loss) with pseudo-labels (expanded training signal)
set -uo pipefail
cd /workspace/vesuvius-kaggle-competition
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VENV=/workspace/venv/bin/python3
LOG="logs/pseudo_margin_dist_gpu2.log"

WEIGHTS="checkpoints/swa_topo/swa_70pre_30topo_ep5.weights.h5"
LABEL_DIR="data/pseudo_labels"
RUN_NAME="pseudo_frozen_margin_dist"

echo "" >> "$LOG"
echo "=== Pseudo-Label + Margin Distance Training ===" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

# ── Training ──
$VENV scripts/train_transunet.py \
    --run-name "${RUN_NAME}" \
    --weights "$WEIGHTS" \
    --label-dir "$LABEL_DIR" \
    --freeze-encoder \
    --lr 5e-5 \
    --epochs 25 \
    --save-every 5 \
    --skel-weight 0.75 \
    --fp-weight 1.5 \
    --dist-weight 0.02 \
    --dist-power 2.0 \
    --dist-margin 3.0 \
    --boundary-weight 0.3 \
    --cldice-weight 0.0 \
    2>&1 | tee -a "$LOG"

TRAIN_EXIT=$?
if [[ $TRAIN_EXIT -ne 0 ]]; then
    echo "!!! Training failed (exit=$TRAIN_EXIT) !!!" | tee -a "$LOG"
fi
echo "--- Training COMPLETE at $(date) ---" | tee -a "$LOG"
echo "=== PIPELINE DONE at $(date) ===" | tee -a "$LOG"
