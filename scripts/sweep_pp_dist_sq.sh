#!/bin/bash
# Post-processing sweep for the dist_sq model.
#
# Phase 1: Generate probmaps using dist_sq weights (scroll 26002 val set)
# Phase 2: Sweep PP parameters on those probmaps
#
# The dist_sq model produces thinner predictions — this sweep tests whether
# different PP params (lower closing, higher t_low) can recover SDice while
# keeping the VOI improvement.
#
# Usage:
#   bash scripts/sweep_pp_dist_sq.sh           # full run
#   bash scripts/sweep_pp_dist_sq.sh --dry-run # quick test (2 volumes, 3 configs)
#
set -euo pipefail
cd /workspace/vesuvius-kaggle-competition

VENV=/workspace/venv/bin/python3
LOG="logs/sweep_pp_dist_sq.log"
DIST_SQ_WEIGHTS="checkpoints/transunet_dist_sq/transunet_ep5.weights.h5"
PROBMAP_DIR="data/dist_sq_probmaps"

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "*** DRY RUN MODE ***"
fi

echo "=== Post-Processing Sweep: dist_sq model ===" | tee "$LOG"
echo "  Start: $(date)" | tee -a "$LOG"
echo "  Weights: $DIST_SQ_WEIGHTS" | tee -a "$LOG"
echo "  Probmap dir: $PROBMAP_DIR" | tee -a "$LOG"

# Phase 1: Generate probmaps
echo "" | tee -a "$LOG"
echo "=== PHASE 1: Generate dist_sq probmaps ===" | tee -a "$LOG"

if [[ -n "$DRY_RUN" ]]; then
    N_EVAL="--n-eval 2"
else
    N_EVAL=""
fi

$VENV scripts/eval_transunet.py \
    --weights "$DIST_SQ_WEIGHTS" \
    --save-probmaps \
    --probmap-dir "$PROBMAP_DIR" \
    --t-low 0.70 \
    --t-high 0.90 \
    $N_EVAL \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== PHASE 2: Post-processing sweep ===" | tee -a "$LOG"

$VENV scripts/sweep_postprocessing.py \
    --probmap-dir "$PROBMAP_DIR" \
    $DRY_RUN \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== DONE at $(date) ===" | tee -a "$LOG"
