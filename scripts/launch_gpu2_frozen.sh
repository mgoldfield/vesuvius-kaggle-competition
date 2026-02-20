#!/bin/bash
# GPU2 restart experiment: Frozen encoder + dist_sq loss
# Only trains decoder + segmentation head. Encoder (SEResNeXt50 + ViT) is frozen.
# dist_sq loss: skel=0.75, fp=1.5, dist=2.0, power=2.0
#
# Run on gpu2 after restart with more storage:
#   nohup bash scripts/launch_gpu2_frozen.sh > logs/train_eval_frozen_dist_sq.log 2>&1 &
#
set -euo pipefail
cd /workspace/vesuvius-kaggle-competition

VENV=/workspace/venv/bin/python3
RUN_NAME="frozen_dist_sq"
LOG="logs/train_eval_${RUN_NAME}.log"
mkdir -p logs checkpoints

echo "=== Frozen Encoder Training ===" | tee "$LOG"
echo "  Start: $(date)" | tee "$LOG"

# Phase 1: Dry run
echo "--- Dry run ---" | tee -a "$LOG"
$VENV scripts/train_transunet.py \
    --run-name "${RUN_NAME}" \
    --freeze-encoder \
    --lr 5e-5 \
    --skel-weight 0.75 \
    --fp-weight 1.5 \
    --dist-weight 2.0 \
    --dist-power 2.0 \
    --dry-run \
    2>&1 | tee -a "$LOG"

rm -f "checkpoints/transunet_${RUN_NAME}/transunet_ep1.weights.h5"
rm -f "checkpoints/transunet_${RUN_NAME}/transunet_best.weights.h5"
echo "--- Dry run PASSED ---" | tee -a "$LOG"

# Phase 2: Full training
echo "" | tee -a "$LOG"
echo "=== PHASE 1: Training ${RUN_NAME} ===" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

$VENV scripts/train_transunet.py \
    --run-name "${RUN_NAME}" \
    --freeze-encoder \
    --lr 5e-5 \
    --epochs 25 \
    --save-every 5 \
    --skel-weight 0.75 \
    --fp-weight 1.5 \
    --dist-weight 2.0 \
    --dist-power 2.0 \
    2>&1 | tee -a "$LOG"

# Phase 3: Evaluate
echo "" | tee -a "$LOG"
echo "=== PHASE 2: Evaluating checkpoints ===" | tee -a "$LOG"

CKPT_DIR="checkpoints/transunet_${RUN_NAME}"
PRETRAINED="pretrained_weights/transunet/transunet.seresnext50.160px.comboloss.weights.h5"
EVAL_LOG="logs/eval_${RUN_NAME}_results.csv"

echo "model,comp_score,topo,sdice,voi,prob_max,n_volumes" > "$EVAL_LOG"

eval_model() {
    local name="$1"
    local weights="$2"
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

eval_model "pretrained" "$PRETRAINED"
for ep in 5 10 15 20 25; do
    CKPT="$CKPT_DIR/transunet_ep${ep}.weights.h5"
    [[ -f "$CKPT" ]] && eval_model "${RUN_NAME}_ep${ep}" "$CKPT"
done
BEST="$CKPT_DIR/transunet_best.weights.h5"
[[ -f "$BEST" ]] && eval_model "${RUN_NAME}_best" "$BEST"

echo "" | tee -a "$LOG"
echo "=== RESULTS ===" | tee -a "$LOG"
column -t -s',' "$EVAL_LOG" | tee -a "$LOG"
echo "=== DONE at $(date) ===" | tee -a "$LOG"
