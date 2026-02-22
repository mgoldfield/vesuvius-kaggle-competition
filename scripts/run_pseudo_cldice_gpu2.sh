#!/bin/bash
# GPU2: Pseudo-label training WITH clDice (48GB GPU, RTX 6000 Ada)
# Key differences from run_pseudo_stage3_4.sh:
#   - RUN_NAME=pseudo_frozen_cldice
#   - --cldice-weight 0.5 (was 0.0)
#   - --cldice-iters 3 (reduced from default 10 to fit in 48GB)
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (prevent fragmentation OOM)
#   - LOG=logs/pseudo_cldice_gpu2.log
set -uo pipefail
cd /workspace/vesuvius-kaggle-competition
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VENV=/workspace/venv/bin/python3
LOG="logs/pseudo_cldice_gpu2.log"

WEIGHTS="checkpoints/swa_topo/swa_70pre_30topo_ep5.weights.h5"
LABEL_DIR="data/pseudo_labels"
RUN_NAME="pseudo_frozen_cldice"

echo "" >> "$LOG"
echo "=== GPU2: Pseudo-Label Training WITH clDice ===" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

# ── Stage 3: Training with pseudo-labels + clDice ──
echo "" | tee -a "$LOG"
echo "=== STAGE 3: Training with Pseudo-Labels + clDice (iters=3) ===" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

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
    --dist-weight 2.0 \
    --dist-power 2.0 \
    --boundary-weight 0.3 \
    --cldice-weight 0.5 \
    --cldice-iters 3 \
    2>&1 | tee -a "$LOG"

TRAIN_EXIT=$?
if [[ $TRAIN_EXIT -ne 0 ]]; then
    echo "!!! Stage 3 training failed (exit=$TRAIN_EXIT) — continuing to eval !!!" | tee -a "$LOG"
fi
echo "--- Stage 3 COMPLETE at $(date) ---" | tee -a "$LOG"

# ── Stage 4: Evaluate checkpoints ──
echo "" | tee -a "$LOG"
echo "=== STAGE 4: Evaluating Checkpoints ===" | tee -a "$LOG"

CKPT_DIR="checkpoints/transunet_${RUN_NAME}"
EVAL_CSV="logs/eval_${RUN_NAME}_results.csv"
echo "model,comp_score,topo,sdice,voi,prob_max,n_volumes" > "$EVAL_CSV"

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
        2>&1) || {
        echo "!!! Eval FAILED for $name !!!" | tee -a "$LOG"
        return 1
    }
    echo "$OUTPUT" | tail -20 | tee -a "$LOG"
    COMP=$(echo "$OUTPUT" | grep "comp_score:" | awk '{print $2}')
    TOPO=$(echo "$OUTPUT" | grep "topo:" | awk '{print $2}')
    SDICE=$(echo "$OUTPUT" | grep "sdice:" | awk '{print $2}')
    VOI=$(echo "$OUTPUT" | grep "voi:" | awk '{print $2}')
    PMAX=$(echo "$OUTPUT" | grep "prob_max:" | awk '{print $2}')
    NVOL=$(echo "$OUTPUT" | grep "Overall" | grep -o 'n=[0-9]*' | grep -o '[0-9]*')
    echo "$name,$COMP,$TOPO,$SDICE,$VOI,$PMAX,$NVOL" >> "$EVAL_CSV"
    echo "  -> comp=$COMP topo=$TOPO sdice=$SDICE voi=$VOI" | tee -a "$LOG"
}

# Evaluate pretrained baseline for reference
eval_model "swa_70pre_30topo_ep5" "$WEIGHTS"

# Evaluate each checkpoint
for ep in 5 10 15 20 25; do
    CKPT="$CKPT_DIR/transunet_ep${ep}.weights.h5"
    [[ -f "$CKPT" ]] && eval_model "${RUN_NAME}_ep${ep}" "$CKPT"
done
BEST="$CKPT_DIR/transunet_best.weights.h5"
[[ -f "$BEST" ]] && eval_model "${RUN_NAME}_best" "$BEST"

echo "" | tee -a "$LOG"
echo "=== RESULTS ===" | tee -a "$LOG"
column -t -s',' "$EVAL_CSV" | tee -a "$LOG"
echo "=== PIPELINE DONE at $(date) ===" | tee -a "$LOG"
