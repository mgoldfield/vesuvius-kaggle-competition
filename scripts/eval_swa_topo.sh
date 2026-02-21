#!/bin/bash
# SWA blending focused on best-topo checkpoints
# Strategy: blend pretrained (best SDice/VOI) with checkpoints that have
# the highest topo scores, at various ratios.
#
# Run on gpu2:
#   tmux new-session -d -s swa 'cd /workspace/vesuvius-kaggle-competition && bash scripts/eval_swa_topo.sh > logs/eval_swa_topo.log 2>&1'
#
set -euo pipefail
cd /workspace/vesuvius-kaggle-competition

VENV=/workspace/venv/bin/python3
LOG="logs/eval_swa_topo.log"
EVAL_CSV="logs/eval_swa_topo_results.csv"
PRETRAINED="pretrained_weights/transunet/transunet.seresnext50.160px.comboloss.weights.h5"
SWA_DIR="checkpoints/swa_topo"
mkdir -p "$SWA_DIR" logs

echo "=== SWA Topo-Focused Blending ===" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
echo "model,comp_score,topo,sdice,voi,prob_max,n_volumes" > "$EVAL_CSV"

# --- Generate blends ---
echo "" | tee -a "$LOG"
echo "=== Generating SWA blends ===" | tee -a "$LOG"

# Best topo checkpoint: frozen_boundary_ep10 (topo=0.2642)
BEST_TOPO="checkpoints/transunet_frozen_boundary/transunet_ep10.weights.h5"

# Blend 1: 90/10 pretrained + best-topo (very conservative)
$VENV scripts/swa_average.py \
    --checkpoints "$PRETRAINED" "$BEST_TOPO" \
    --weights 0.90 0.10 \
    --output "$SWA_DIR/swa_90pre_10topo.weights.h5" \
    2>&1 | tail -3 | tee -a "$LOG"

# Blend 2: 80/20 pretrained + best-topo
$VENV scripts/swa_average.py \
    --checkpoints "$PRETRAINED" "$BEST_TOPO" \
    --weights 0.80 0.20 \
    --output "$SWA_DIR/swa_80pre_20topo.weights.h5" \
    2>&1 | tail -3 | tee -a "$LOG"

# Blend 3: 70/30 pretrained + best-topo
$VENV scripts/swa_average.py \
    --checkpoints "$PRETRAINED" "$BEST_TOPO" \
    --weights 0.70 0.30 \
    --output "$SWA_DIR/swa_70pre_30topo.weights.h5" \
    2>&1 | tail -3 | tee -a "$LOG"

# Blend 4: 60/40 pretrained + best-topo (aggressive)
$VENV scripts/swa_average.py \
    --checkpoints "$PRETRAINED" "$BEST_TOPO" \
    --weights 0.60 0.40 \
    --output "$SWA_DIR/swa_60pre_40topo.weights.h5" \
    2>&1 | tail -3 | tee -a "$LOG"

# Blend 5: 70/30 pretrained + frozen_boundary_ep5 (second best topo=0.2619)
SECOND_TOPO="checkpoints/transunet_frozen_boundary/transunet_ep5.weights.h5"
$VENV scripts/swa_average.py \
    --checkpoints "$PRETRAINED" "$SECOND_TOPO" \
    --weights 0.70 0.30 \
    --output "$SWA_DIR/swa_70pre_30topo_ep5.weights.h5" \
    2>&1 | tail -3 | tee -a "$LOG"

# Blend 6: 70/30 pretrained + frozen_boundary_ep15 (best SDice among frozen=0.7871)
BEST_SDICE="checkpoints/transunet_frozen_boundary/transunet_ep15.weights.h5"
$VENV scripts/swa_average.py \
    --checkpoints "$PRETRAINED" "$BEST_SDICE" \
    --weights 0.70 0.30 \
    --output "$SWA_DIR/swa_70pre_30sdice_ep15.weights.h5" \
    2>&1 | tail -3 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Evaluating all blends ===" | tee -a "$LOG"

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
    echo "$name,$COMP,$TOPO,$SDICE,$VOI,$PMAX,$NVOL" >> "$EVAL_CSV"
    echo "  -> comp=$COMP topo=$TOPO sdice=$SDICE voi=$VOI" | tee -a "$LOG"
}

eval_model "swa_90pre_10topo" "$SWA_DIR/swa_90pre_10topo.weights.h5"
eval_model "swa_80pre_20topo" "$SWA_DIR/swa_80pre_20topo.weights.h5"
eval_model "swa_70pre_30topo" "$SWA_DIR/swa_70pre_30topo.weights.h5"
eval_model "swa_60pre_40topo" "$SWA_DIR/swa_60pre_40topo.weights.h5"
eval_model "swa_70pre_30topo_ep5" "$SWA_DIR/swa_70pre_30topo_ep5.weights.h5"
eval_model "swa_70pre_30sdice_ep15" "$SWA_DIR/swa_70pre_30sdice_ep15.weights.h5"

echo "" | tee -a "$LOG"
echo "=== RESULTS ===" | tee -a "$LOG"
column -t -s',' "$EVAL_CSV" | tee -a "$LOG"
echo "=== DONE at $(date) ===" | tee -a "$LOG"
