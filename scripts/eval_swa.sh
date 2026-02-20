#!/bin/bash
# Evaluate all SWA weight blends
set -euo pipefail
cd /workspace/vesuvius-kaggle-competition

VENV=/workspace/venv/bin/python3
LOG="logs/eval_swa.log"
EVAL_CSV="logs/eval_swa_results.csv"

echo "=== SWA Evaluation ===" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
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

eval_model "swa_70pre_30lowlr_ep5" "checkpoints/swa/swa_70pre_30lowlr_ep5.weights.h5"
eval_model "swa_50pre_50lowlr_ep5" "checkpoints/swa/swa_50pre_50lowlr_ep5.weights.h5"
eval_model "swa_70pre_30distsq_ep5" "checkpoints/swa/swa_70pre_30distsq_ep5.weights.h5"
eval_model "swa_lowlr_late_avg" "checkpoints/swa/swa_lowlr_late_avg.weights.h5"

echo "" | tee -a "$LOG"
echo "=== RESULTS ===" | tee -a "$LOG"
column -t -s',' "$EVAL_CSV" | tee -a "$LOG"
echo "=== DONE at $(date) ===" | tee -a "$LOG"
