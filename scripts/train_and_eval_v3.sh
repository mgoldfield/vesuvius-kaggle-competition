#!/bin/bash
# Train baseline_v3 (z-score normalization fix) and evaluate all checkpoints.
#
# Usage:
#   bash scripts/train_and_eval_v3.sh           # full run
#   bash scripts/train_and_eval_v3.sh --dry-run  # quick test
#
set -euo pipefail

cd /workspace/vesuvius-kaggle-competition
VENV=/workspace/venv/bin/python3
LOG=logs/train_eval_v3.log
mkdir -p logs checkpoints

DRY_RUN=""
EPOCHS=25
SAVE_EVERY=5
N_EVAL=20  # volumes for comp score eval
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    EPOCHS=2
    SAVE_EVERY=1
    N_EVAL=2
    echo "=== DRY RUN MODE ==="
fi

echo "=== PHASE 1: Training baseline_v3 ===" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

$VENV scripts/train_transunet.py \
    --run-name baseline_v3 \
    --epochs $EPOCHS \
    --save-every $SAVE_EVERY \
    --skel-weight 0.75 \
    --fp-weight 0.50 \
    $DRY_RUN \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== PHASE 2: Evaluating checkpoints ===" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

CKPT_DIR=checkpoints/transunet_baseline_v3
PRETRAINED=pretrained_weights/transunet/transunet.seresnext50.160px.comboloss.weights.h5
EVAL_LOG=logs/eval_v3_results.csv

# Header for results CSV
echo "model,comp_score,topo,sdice,voi,prob_max,n_volumes" > "$EVAL_LOG"

eval_model() {
    local name="$1"
    local weights="$2"
    echo "" | tee -a "$LOG"
    echo "--- Evaluating: $name ---" | tee -a "$LOG"
    echo "Weights: $weights" | tee -a "$LOG"

    # Run eval, capture summary line
    OUTPUT=$($VENV scripts/eval_transunet.py \
        --weights "$weights" \
        --cross-scroll \
        --max-per-scroll 4 \
        --t-low 0.70 \
        --t-high 0.90 \
        2>&1)
    echo "$OUTPUT" | tail -20 | tee -a "$LOG"

    # Parse scores from output
    COMP=$(echo "$OUTPUT" | grep "comp_score:" | awk '{print $2}')
    TOPO=$(echo "$OUTPUT" | grep "topo:" | awk '{print $2}')
    SDICE=$(echo "$OUTPUT" | grep "sdice:" | awk '{print $2}')
    VOI=$(echo "$OUTPUT" | grep "voi:" | awk '{print $2}')
    PMAX=$(echo "$OUTPUT" | grep "prob_max:" | awk '{print $2}')
    NVOL=$(echo "$OUTPUT" | grep "Overall" | grep -o 'n=[0-9]*' | grep -o '[0-9]*')

    echo "$name,$COMP,$TOPO,$SDICE,$VOI,$PMAX,$NVOL" >> "$EVAL_LOG"
    echo "  -> comp=$COMP" | tee -a "$LOG"
}

# Evaluate pretrained baseline first
eval_model "pretrained" "$PRETRAINED"

# Evaluate periodic checkpoints
for ep in $(seq $SAVE_EVERY $SAVE_EVERY $EPOCHS); do
    CKPT="$CKPT_DIR/transunet_ep${ep}.weights.h5"
    if [[ -f "$CKPT" ]]; then
        eval_model "baseline_v3_ep${ep}" "$CKPT"
    fi
done

# Evaluate best checkpoint (if different from last epoch)
BEST="$CKPT_DIR/transunet_best.weights.h5"
if [[ -f "$BEST" ]]; then
    eval_model "baseline_v3_best" "$BEST"
fi

echo "" | tee -a "$LOG"
echo "=== RESULTS SUMMARY ===" | tee -a "$LOG"
echo "$(date)" | tee -a "$LOG"
column -t -s',' "$EVAL_LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Full results: $EVAL_LOG" | tee -a "$LOG"
echo "Training log: $LOG" | tee -a "$LOG"
echo "=== DONE ===" | tee -a "$LOG"
