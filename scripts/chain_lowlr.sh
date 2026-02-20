#!/bin/bash
# Chain script: wait for current training to finish, then launch low-LR follow-up.
#
# Usage:
#   nohup bash scripts/chain_lowlr.sh CURRENT_RUN SKEL FP DIST POWER > logs/chain_CURRENT_RUN.log 2>&1 &
#
# This script:
#   1. Waits for the current train_and_eval_variant.sh to finish
#   2. Launches a low-LR (5e-6) version with "_lowlr" suffix
#   3. The low-LR run includes its own dry run + training + eval
#
set -euo pipefail

CURRENT_RUN="${1:?Usage: $0 CURRENT_RUN SKEL FP DIST POWER}"
SKEL_W="${2:-0.75}"
FP_W="${3:-0.50}"
DIST_W="${4:-0.0}"
DIST_P="${5:-1.0}"

LOWLR_RUN="${CURRENT_RUN}_lowlr"

echo "=== Chain script started: $(date) ==="
echo "  Waiting for '${CURRENT_RUN}' to finish..."
echo "  Will then launch '${LOWLR_RUN}' with LR=5e-6"

# Wait for any train_and_eval_variant.sh process to finish
# Check every 60 seconds
while pgrep -f "train_and_eval_variant.sh ${CURRENT_RUN}" > /dev/null 2>&1; do
    sleep 60
done

echo ""
echo "=== Current run '${CURRENT_RUN}' finished at $(date) ==="
echo "  Results:"
cat "/workspace/vesuvius-kaggle-competition/logs/eval_${CURRENT_RUN}_results.csv" 2>/dev/null || echo "  (no results file found)"
echo ""
echo "=== Launching low-LR follow-up: ${LOWLR_RUN} ==="

bash /workspace/vesuvius-kaggle-competition/scripts/train_and_eval_variant.sh \
    "${LOWLR_RUN}" "$SKEL_W" "$FP_W" "$DIST_W" "$DIST_P" "5e-6"

echo ""
echo "=== Chain complete at $(date) ==="
