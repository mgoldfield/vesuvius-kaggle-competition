#!/bin/bash
# Overnight training pipeline: 3 sequential fine-tuning runs
# Total estimated time: ~9 hours (3 runs × 3 hours each)
#
# Run 1 (baseline): Already running separately — SkeletonRecall + 0.50 FP_Volume
# Run 2 (thin_fp): Higher FP_Volume (1.5) for thinner predictions
# Run 3 (thin_dist): Distance-from-skeleton penalty for even thinner predictions

set -e

SCRIPT="scripts/train_transunet.py"
PYTHON="/workspace/venv/bin/python"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "Overnight Training Pipeline"
echo "Started: $(date)"
echo "============================================"

# ── Dry run verification ──
echo ""
echo "=== Dry run: verifying Run 2 (thin_fp) ==="
$PYTHON $SCRIPT --dry-run --run-name dry_test_fp --fp-weight 1.5 2>&1 | tail -5
echo "Dry run 2 passed."

echo ""
echo "=== Dry run: verifying Run 3 (thin_dist) ==="
$PYTHON $SCRIPT --dry-run --run-name dry_test_dist --fp-weight 1.5 --dist-weight 1.0 2>&1 | tail -5
echo "Dry run 3 passed."

# Clean up dry run checkpoints
rm -rf checkpoints/transunet_dry_test_fp checkpoints/transunet_dry_test_dist

echo ""
echo "=== Both dry runs passed. Starting full training. ==="

# ── Run 2: High FP_Volume ──
echo ""
echo "============================================"
echo "Run 2: thin_fp (FP_Volume=1.5)"
echo "Started: $(date)"
echo "============================================"
$PYTHON $SCRIPT \
    --run-name thin_fp \
    --epochs 25 \
    --save-every 5 \
    --fp-weight 1.5 \
    2>&1 | tee "$LOG_DIR/train_thin_fp.log"

echo ""
echo "Run 2 complete: $(date)"

# ── Run 3: Distance-from-skeleton ──
echo ""
echo "============================================"
echo "Run 3: thin_dist (FP_Volume=1.5 + DistFromSkel=1.0)"
echo "Started: $(date)"
echo "============================================"
$PYTHON $SCRIPT \
    --run-name thin_dist \
    --epochs 25 \
    --save-every 5 \
    --fp-weight 1.5 \
    --dist-weight 1.0 \
    2>&1 | tee "$LOG_DIR/train_thin_dist.log"

echo ""
echo "============================================"
echo "Overnight Training Pipeline Complete"
echo "Finished: $(date)"
echo ""
echo "Checkpoints:"
echo "  Run 1 (baseline): checkpoints/transunet/"
echo "  Run 2 (thin_fp):  checkpoints/transunet_thin_fp/"
echo "  Run 3 (thin_dist): checkpoints/transunet_thin_dist/"
echo "============================================"
