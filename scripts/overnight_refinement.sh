#!/bin/bash
# Overnight pipeline: Generate probmaps → Train refinement model
# Run inside tmux. Logs to logs/refinement_overnight.log
#
# Usage:
#   bash scripts/overnight_refinement.sh 2>&1 | tee logs/refinement_overnight.log

set -e

VENV="/workspace/venv/bin"
ROOT="/workspace/vesuvius-kaggle-competition"
TRACED_MODEL="$ROOT/kaggle/kaggle_weights_download/best_segresnet_v9_traced.pt"
PROBMAP_DIR="$ROOT/data/refinement_data/probmaps"

echo "============================================"
echo "Overnight Refinement Pipeline"
echo "Started: $(date)"
echo "============================================"
echo ""

# ── Step 1: Generate probmaps ────────────────────────────
echo "=== Step 1: Generate probmaps (786 volumes, TTA) ==="
echo "Model: $TRACED_MODEL"
echo "Output: $PROBMAP_DIR"
echo ""

$VENV/python "$ROOT/scripts/generate_refinement_data.py" \
    --traced "$TRACED_MODEL" --tta

echo ""
echo "=== Step 1 complete: $(date) ==="
echo "Probmaps: $(ls $PROBMAP_DIR/*.npy 2>/dev/null | wc -l) files"
echo ""

# ── Step 2: Run refinement notebook ──────────────────────
echo "=== Step 2: Train refinement model (Phase 1 + Phase 2) ==="
echo ""

# Create checkpoints dir
mkdir -p "$ROOT/checkpoints/models"

$VENV/jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output vesuvius_train_refinement_executed.ipynb \
    "$ROOT/notebooks/refinement/vesuvius_train_refinement.ipynb"

echo ""
echo "=== Step 2 complete: $(date) ==="
echo ""

echo "============================================"
echo "Pipeline complete: $(date)"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Check training curves in notebooks/refinement/vesuvius_train_refinement_executed.ipynb"
echo "  2. Check head-to-head results (refinement vs hand-tuned)"
echo "  3. If refinement wins: upload traced model + update Kaggle inference"
