#!/bin/bash
# Full overnight pipeline: Refinement model → v12 training → eval → submission
# Run inside tmux. Logs to logs/overnight_full.log
#
# Usage:
#   bash scripts/overnight_full.sh 2>&1 | tee logs/overnight_full.log

set -e

VENV="/workspace/venv/bin"
ROOT="/workspace/vesuvius-kaggle-competition"
TRACED_V9="$ROOT/kaggle/kaggle_weights_download/best_segresnet_v9_traced.pt"
PROBMAP_DIR="$ROOT/data/refinement_data/probmaps"

echo "============================================"
echo "Full Overnight Pipeline"
echo "Started: $(date)"
echo "============================================"
echo ""

# ── Phase 1: Generate probmaps for refinement model ──────────
echo "=== Phase 1: Generate probmaps (786 volumes, TTA) ==="
echo "Model: $TRACED_V9"
echo "Output: $PROBMAP_DIR"
echo ""

$VENV/python "$ROOT/scripts/generate_refinement_data.py" \
    --traced "$TRACED_V9" --tta

echo ""
echo "=== Phase 1 complete: $(date) ==="
echo "Probmaps: $(ls $PROBMAP_DIR/*.npy 2>/dev/null | wc -l) files"
echo ""

# ── Phase 2: Train refinement model ──────────────────────────
echo "=== Phase 2: Train refinement model (Phase 1 + Phase 2) ==="
echo ""

mkdir -p "$ROOT/checkpoints/models"

$VENV/jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output vesuvius_train_refinement_executed.ipynb \
    "$ROOT/notebooks/refinement/vesuvius_train_refinement.ipynb"

echo ""
echo "=== Phase 2 complete: $(date) ==="
echo ""

# ── Phase 3: Train v12 (flat_cos + 50 epochs) ────────────────
echo "=== Phase 3: Train v12 (SegResNet, flat_cos, 50 epochs) ==="
echo ""

$VENV/jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output vesuvius_train_v12_executed.ipynb \
    "$ROOT/notebooks/vesuvius_train_v12.ipynb"

echo ""
echo "=== Phase 3 complete: $(date) ==="
echo ""

# ── Phase 4: Compare v12 vs v9 ───────────────────────────────
echo "=== Phase 4: Summary ==="
echo ""
echo "Checkpoints saved:"
ls -lh "$ROOT/checkpoints/models/"*v12* 2>/dev/null || echo "  (none found)"
ls -lh "$ROOT/checkpoints/models/"*refinement* 2>/dev/null || echo "  (no refinement checkpoints)"
echo ""
echo "Traced models:"
ls -lh "$ROOT/kaggle/kaggle_weights_download/"*v12* 2>/dev/null || echo "  (none found)"
echo ""

echo "============================================"
echo "Pipeline complete: $(date)"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Check v12 training curves in notebooks/vesuvius_train_v12_executed.ipynb"
echo "  2. Check refinement head-to-head in notebooks/refinement/vesuvius_train_refinement_executed.ipynb"
echo "  3. Compare v12 comp_score vs v9 (0.570)"
echo "  4. If v12 wins: update Kaggle inference to use v12 traced model"
echo "  5. If refinement wins: add refinement model to Kaggle inference pipeline"
echo "  6. If both win: use v12 + refinement (best case)"
