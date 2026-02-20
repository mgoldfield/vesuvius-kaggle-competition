#!/bin/bash
# Train a single TransUNet variant and evaluate all checkpoints.
#
# Usage:
#   # GPU 1: High FP_Volume (thinner predictions)
#   bash scripts/train_and_eval_variant.sh thin_fp 0.75 1.5 0.0 1.0
#
#   # GPU 2: DistFromSkeleton
#   bash scripts/train_and_eval_variant.sh dist_skel 0.75 1.5 1.0 1.0
#
#   # GPU 3: Squared DistFromSkeleton
#   bash scripts/train_and_eval_variant.sh dist_sq 0.75 1.5 2.0 2.0
#
# Args: RUN_NAME SKEL_WEIGHT FP_WEIGHT DIST_WEIGHT DIST_POWER [LR]
#
set -euo pipefail

RUN_NAME="${1:?Usage: $0 RUN_NAME SKEL_WEIGHT FP_WEIGHT DIST_WEIGHT DIST_POWER [LR]}"
SKEL_W="${2:-0.75}"
FP_W="${3:-0.50}"
DIST_W="${4:-0.0}"
DIST_P="${5:-1.0}"
LR="${6:-5e-5}"

cd /workspace/vesuvius-kaggle-competition
VENV=/workspace/venv/bin/python3
LOG="logs/train_eval_${RUN_NAME}.log"
mkdir -p logs checkpoints

echo "=== Training variant: ${RUN_NAME} ===" | tee "$LOG"
echo "  skel=${SKEL_W} fp=${FP_W} dist=${DIST_W} dist_power=${DIST_P} lr=${LR}" | tee -a "$LOG"
echo "  Start: $(date)" | tee -a "$LOG"

# Phase 1: Dry run
echo "" | tee -a "$LOG"
echo "--- Dry run ---" | tee -a "$LOG"
$VENV scripts/train_transunet.py \
    --run-name "${RUN_NAME}" \
    --skel-weight "$SKEL_W" \
    --fp-weight "$FP_W" \
    --dist-weight "$DIST_W" \
    --dist-power "$DIST_P" \
    --lr "$LR" \
    --dry-run \
    2>&1 | tee -a "$LOG"

# Clean up dry run checkpoint
rm -f "checkpoints/transunet_${RUN_NAME}/transunet_ep1.weights.h5"
rm -f "checkpoints/transunet_${RUN_NAME}/transunet_best.weights.h5"

echo "--- Dry run PASSED ---" | tee -a "$LOG"

# Phase 2: Full training
echo "" | tee -a "$LOG"
echo "=== PHASE 1: Training ${RUN_NAME} ===" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

$VENV scripts/train_transunet.py \
    --run-name "${RUN_NAME}" \
    --epochs 25 \
    --save-every 5 \
    --skel-weight "$SKEL_W" \
    --fp-weight "$FP_W" \
    --dist-weight "$DIST_W" \
    --dist-power "$DIST_P" \
    --lr "$LR" \
    2>&1 | tee -a "$LOG"

# Phase 3: Evaluate checkpoints
echo "" | tee -a "$LOG"
echo "=== PHASE 2: Evaluating checkpoints ===" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

CKPT_DIR="checkpoints/transunet_${RUN_NAME}"
PRETRAINED="pretrained_weights/transunet/transunet.seresnext50.160px.comboloss.weights.h5"
EVAL_LOG="logs/eval_${RUN_NAME}_results.csv"

echo "model,comp_score,topo,sdice,voi,prob_max,n_volumes" > "$EVAL_LOG"

eval_model() {
    local name="$1"
    local weights="$2"
    echo "" | tee -a "$LOG"
    echo "--- Evaluating: $name ---" | tee -a "$LOG"

    OUTPUT=$($VENV scripts/eval_transunet.py \
        --weights "$weights" \
        --cross-scroll \
        --max-per-scroll 4 \
        --t-low 0.70 \
        --t-high 0.90 \
        2>&1)
    echo "$OUTPUT" | tail -20 | tee -a "$LOG"

    COMP=$(echo "$OUTPUT" | grep "comp_score:" | awk '{print $2}')
    TOPO=$(echo "$OUTPUT" | grep "topo:" | awk '{print $2}')
    SDICE=$(echo "$OUTPUT" | grep "sdice:" | awk '{print $2}')
    VOI=$(echo "$OUTPUT" | grep "voi:" | awk '{print $2}')
    PMAX=$(echo "$OUTPUT" | grep "prob_max:" | awk '{print $2}')
    NVOL=$(echo "$OUTPUT" | grep "Overall" | grep -o 'n=[0-9]*' | grep -o '[0-9]*')

    echo "$name,$COMP,$TOPO,$SDICE,$VOI,$PMAX,$NVOL" >> "$EVAL_LOG"
    echo "  -> comp=$COMP" | tee -a "$LOG"
}

# Evaluate pretrained baseline
eval_model "pretrained" "$PRETRAINED"

# Evaluate periodic checkpoints
for ep in 5 10 15 20 25; do
    CKPT="$CKPT_DIR/transunet_ep${ep}.weights.h5"
    if [[ -f "$CKPT" ]]; then
        eval_model "${RUN_NAME}_ep${ep}" "$CKPT"
    fi
done

# Evaluate best
BEST="$CKPT_DIR/transunet_best.weights.h5"
if [[ -f "$BEST" ]]; then
    eval_model "${RUN_NAME}_best" "$BEST"
fi

echo "" | tee -a "$LOG"
echo "=== RESULTS SUMMARY ===" | tee -a "$LOG"
echo "$(date)" | tee -a "$LOG"
column -t -s',' "$EVAL_LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "=== DONE ===" | tee -a "$LOG"
