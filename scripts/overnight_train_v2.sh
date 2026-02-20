#!/bin/bash
# Overnight training pipeline v2 — sequential training runs
# Fixes from v1: float32 (no mixed precision), correct intensity shift
#
# All runs start from pretrained comboloss weights.
# All logs go to logs/train_<run_name>.log
# All checkpoints go to checkpoints/transunet_<run_name>/
#
# Usage:
#   nohup bash scripts/overnight_train_v2.sh > logs/overnight_v2.log 2>&1 &

set -e

PYTHON="/workspace/venv/bin/python"
SCRIPT="scripts/train_transunet.py"
LOG_DIR="logs"
PRETRAINED="pretrained_weights/transunet/transunet.seresnext50.160px.comboloss.weights.h5"
mkdir -p "$LOG_DIR"

# ── Helper function ──
run_training() {
    local NAME="$1"
    shift
    echo ""
    echo "============================================"
    echo "Training: $NAME"
    echo "Args: $@"
    echo "Started: $(date)"
    echo "============================================"
    $PYTHON $SCRIPT --run-name "$NAME" --weights "$PRETRAINED" "$@" 2>&1 | tee "$LOG_DIR/train_${NAME}.log"
    echo ""
    echo "$NAME complete: $(date)"
    echo ""
}

echo "============================================"
echo "Overnight Training Pipeline v2"
echo "Started: $(date)"
echo "Fixes: float32, correct intensity shift"
echo "============================================"

# ── Dry-run all configs first ──
echo ""
echo "=== DRY RUN VERIFICATION ==="

echo "Dry run A (baseline_v2)..."
$PYTHON $SCRIPT --run-name dry_a --dry-run --weights "$PRETRAINED" --epochs 25 --save-every 5 --fp-weight 0.50 --skel-weight 0.75 2>&1
echo "PASS"

echo "Dry run B (thin_fp_v2)..."
$PYTHON $SCRIPT --run-name dry_b --dry-run --weights "$PRETRAINED" --epochs 25 --save-every 5 --fp-weight 1.5 --skel-weight 0.75 2>&1
echo "PASS"

echo "Dry run C (thin_dist_v2)..."
$PYTHON $SCRIPT --run-name dry_c --dry-run --weights "$PRETRAINED" --epochs 25 --save-every 5 --fp-weight 1.5 --skel-weight 0.75 --dist-weight 1.0 2>&1
echo "PASS"

echo "Dry run D (dist_sq_v2)..."
$PYTHON $SCRIPT --run-name dry_d --dry-run --weights "$PRETRAINED" --epochs 25 --save-every 5 --fp-weight 1.5 --skel-weight 0.75 --dist-weight 2.0 --dist-power 2.0 2>&1
echo "PASS"

# Clean up dry run checkpoints
rm -rf checkpoints/transunet_dry_a checkpoints/transunet_dry_b checkpoints/transunet_dry_c checkpoints/transunet_dry_d

echo ""
echo "=== ALL DRY RUNS PASSED ==="
echo ""

### FULL TRAINING RUNS ###
### All start from pretrained weights ###

# Run A: Baseline (same config as competitor) — control to verify training works
run_training "baseline_v2" \
    --epochs 25 --save-every 5 \
    --fp-weight 0.50 --skel-weight 0.75

# Run B: Higher FP_Volume for thinner predictions
run_training "thin_fp_v2" \
    --epochs 25 --save-every 5 \
    --fp-weight 1.5 --skel-weight 0.75

# Run C: Distance-from-skeleton penalty (linear)
run_training "thin_dist_v2" \
    --epochs 25 --save-every 5 \
    --fp-weight 1.5 --skel-weight 0.75 --dist-weight 1.0

# Run D: Squared distance variant (quadratic penalty)
run_training "dist_sq_v2" \
    --epochs 25 --save-every 5 \
    --fp-weight 1.5 --skel-weight 0.75 \
    --dist-weight 2.0 --dist-power 2.0

echo "============================================"
echo "Overnight Training Pipeline v2 Complete"
echo "Finished: $(date)"
echo ""
echo "Checkpoints:"
ls -d checkpoints/transunet_*/
echo "============================================"
