#!/bin/bash
# Pseudo-labeling pipeline: probmap generation → pseudo-label creation → training → eval
#
# Run on gpu0:
#   tmux new-session -d -s pseudo 'cd /workspace/vesuvius-kaggle-competition && bash scripts/run_pseudo_pipeline.sh > logs/pseudo_pipeline.log 2>&1'
#
set -uo pipefail
cd /workspace/vesuvius-kaggle-competition
export CUDA_VISIBLE_DEVICES=0

VENV=/workspace/venv/bin/python3
LOG="logs/pseudo_pipeline.log"
mkdir -p logs data/pseudo_probmaps data/pseudo_labels

WEIGHTS="checkpoints/swa_topo/swa_70pre_30topo_ep5.weights.h5"
PROBMAP_DIR="data/pseudo_probmaps"
LABEL_DIR="data/pseudo_labels"
RUN_NAME="pseudo_frozen_boundary"

# Helper: run a stage command and check exit code
run_stage() {
    local stage_name="$1"
    shift
    "$@" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        echo "!!! STAGE FAILED: $stage_name (exit code $rc) at $(date) !!!" | tee -a "$LOG"
        return $rc
    fi
    return 0
}

echo "=== Pseudo-Labeling Pipeline ===" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
echo "Weights: $WEIGHTS" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── Stage 1: Generate probmaps ──
echo "=== STAGE 1: Probmap Generation ===" | tee -a "$LOG"

# Dry run first
echo "--- Stage 1 dry run ---" | tee -a "$LOG"
if ! run_stage "Stage 1 dry run" $VENV scripts/generate_probmaps.py \
    --weights "$WEIGHTS" \
    --output-dir "$PROBMAP_DIR" \
    --dry-run; then
    echo "FATAL: Stage 1 dry run failed. Aborting pipeline." | tee -a "$LOG"
    exit 1
fi
echo "--- Stage 1 dry run PASSED ---" | tee -a "$LOG"

# Full run (skip existing for resumability)
# Note: generate_probmaps.py has try/except for OOM — it will skip
# problematic volumes and report them at the end rather than crashing.
echo "" | tee -a "$LOG"
echo "--- Stage 1 full run ---" | tee -a "$LOG"
run_stage "Stage 1 full run" $VENV scripts/generate_probmaps.py \
    --weights "$WEIGHTS" \
    --output-dir "$PROBMAP_DIR" \
    --skip-existing
echo "--- Stage 1 COMPLETE at $(date) ---" | tee -a "$LOG"

# ── Stage 2: Generate pseudo-labels ──
echo "" | tee -a "$LOG"
echo "=== STAGE 2: Pseudo-Label Generation ===" | tee -a "$LOG"

# Dry run first
echo "--- Stage 2 dry run ---" | tee -a "$LOG"
if ! run_stage "Stage 2 dry run" $VENV scripts/generate_pseudo_labels.py \
    --probmap-dir "$PROBMAP_DIR" \
    --output-dir "$LABEL_DIR" \
    --fg-threshold 0.85 \
    --bg-threshold 0.15 \
    --dry-run; then
    echo "FATAL: Stage 2 dry run failed. Aborting pipeline." | tee -a "$LOG"
    exit 1
fi
echo "--- Stage 2 dry run PASSED ---" | tee -a "$LOG"

# Full run
echo "" | tee -a "$LOG"
echo "--- Stage 2 full run ---" | tee -a "$LOG"
if ! run_stage "Stage 2 full run" $VENV scripts/generate_pseudo_labels.py \
    --probmap-dir "$PROBMAP_DIR" \
    --output-dir "$LABEL_DIR" \
    --fg-threshold 0.85 \
    --bg-threshold 0.15; then
    echo "FATAL: Stage 2 failed. Aborting pipeline." | tee -a "$LOG"
    exit 1
fi
echo "--- Stage 2 COMPLETE at $(date) ---" | tee -a "$LOG"

# ── Stage 3: Training with pseudo-labels ──
echo "" | tee -a "$LOG"
echo "=== STAGE 3: Training with Pseudo-Labels ===" | tee -a "$LOG"

# Dry run first
echo "--- Stage 3 dry run ---" | tee -a "$LOG"
if ! run_stage "Stage 3 dry run" $VENV scripts/train_transunet.py \
    --run-name "${RUN_NAME}" \
    --weights "$WEIGHTS" \
    --label-dir "$LABEL_DIR" \
    --freeze-encoder \
    --lr 5e-5 \
    --skel-weight 0.75 \
    --fp-weight 1.5 \
    --dist-weight 2.0 \
    --dist-power 2.0 \
    --boundary-weight 0.3 \
    --cldice-weight 0.3 \
    --dry-run; then
    echo "FATAL: Stage 3 dry run failed. Aborting pipeline." | tee -a "$LOG"
    exit 1
fi

rm -f "checkpoints/transunet_${RUN_NAME}/transunet_ep1.weights.h5"
rm -f "checkpoints/transunet_${RUN_NAME}/transunet_best.weights.h5"
echo "--- Stage 3 dry run PASSED ---" | tee -a "$LOG"

# Full training
echo "" | tee -a "$LOG"
echo "--- Stage 3 full training ---" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

if ! run_stage "Stage 3 training" $VENV scripts/train_transunet.py \
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
    --cldice-weight 0.3; then
    echo "!!! Stage 3 training failed — continuing to eval whatever checkpoints exist !!!" | tee -a "$LOG"
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
