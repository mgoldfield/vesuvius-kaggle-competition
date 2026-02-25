#!/bin/bash
# Monitor overnight pipeline every 10 minutes
# Outputs status, detects failures, shows ETA
cd /workspace/vesuvius-kaggle-competition

LOG="logs/overnight_final_push.log"
TRAIN_LOG="logs/train_all_data_frozen.log"

echo "=== Pipeline Monitor ($(date)) ==="

# Check if tmux session is alive
if ! tmux has-session -t overnight 2>/dev/null; then
    echo "WARNING: tmux session 'overnight' is DEAD"
    echo "Last 10 lines of log:"
    tail -10 "$LOG"
    exit 1
fi

echo "tmux session: alive"

# Determine current phase from log
if grep -q "PIPELINE COMPLETE" "$LOG" 2>/dev/null; then
    echo "STATUS: PIPELINE COMPLETE"
    tail -20 "$LOG"
elif grep -q "FATAL" "$LOG" 2>/dev/null; then
    echo "STATUS: PIPELINE FAILED"
    grep "FATAL" "$LOG"
    tail -10 "$LOG"
elif grep -q "PHASE 6" "$LOG" 2>/dev/null; then
    echo "STATUS: Phase 6 - Submitting to Kaggle"
    tail -5 "$LOG"
elif grep -q "PHASE 5" "$LOG" 2>/dev/null; then
    echo "STATUS: Phase 5 - Uploading weights"
    tail -5 "$LOG"
elif grep -q "PHASE 4" "$LOG" 2>/dev/null; then
    echo "STATUS: Phase 4 - SWA blending train-all"
    tail -5 "$LOG"
elif grep -q "PHASE 3" "$LOG" 2>/dev/null; then
    echo "STATUS: Phase 3 - Training on all 786 volumes"
    if [ -f "$TRAIN_LOG" ]; then
        LAST_EPOCH=$(grep "^Epoch" "$TRAIN_LOG" | tail -1)
        LAST_STEP=$(grep "step" "$TRAIN_LOG" | tail -1)
        echo "  Last epoch: $LAST_EPOCH"
        echo "  Last step: $LAST_STEP"
    fi
elif grep -q "PHASE 2" "$LOG" 2>/dev/null; then
    echo "STATUS: Phase 2 - Dry-run train-all"
    tail -3 "$LOG"
elif grep -q "PHASE 1" "$LOG" 2>/dev/null; then
    echo "STATUS: Phase 1 - SWA blend evals"
    tail -3 "$LOG"
else
    echo "STATUS: Starting up"
    tail -3 "$LOG"
fi

# GPU status
echo ""
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1
