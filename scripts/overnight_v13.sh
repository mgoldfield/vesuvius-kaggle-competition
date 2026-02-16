#!/bin/bash
# Overnight pipeline for GPU 2 (remote): wait for v13 training, then sweep checkpoints
# Run on remote GPU with:
#   nohup bash /workspace/vesuvius-kaggle-competition/scripts/overnight_v13.sh \
#     > /workspace/vesuvius-kaggle-competition/logs/overnight_v13.log 2>&1 &
# Dry run:  bash /workspace/vesuvius-kaggle-competition/scripts/overnight_v13.sh --dry-run

set -e
cd /workspace/vesuvius-kaggle-competition
source /workspace/venv/bin/activate

DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

echo "=== Overnight v13 pipeline started at $(date) ==="

# Step 1: Wait for v13 training to finish
if $DRY_RUN; then
    if pgrep -f "vesuvius_train_v13.ipynb" > /dev/null 2>&1; then
        echo "[DRY RUN] v13 training is still running — would wait here"
    else
        echo "[DRY RUN] v13 training is NOT running — would proceed immediately"
    fi
else
    echo "[$(date)] Waiting for v13 training (nbconvert) to complete..."
    while pgrep -f "vesuvius_train_v13.ipynb" > /dev/null 2>&1; do
        sleep 120
    done
    echo "[$(date)] v13 training completed!"
fi

# Step 2: Check if checkpoints exist
echo "[$(date)] Checking for v13 checkpoints..."
ls -la checkpoints/models/segresnet_v13_ep*.pth 2>/dev/null || echo "  No periodic checkpoints found (expected if training hasn't finished)"
ls -la checkpoints/models/best_segresnet_v13.pth 2>/dev/null || echo "  No best checkpoint found (expected if training hasn't finished)"

# Step 3: Run checkpoint sweep
if $DRY_RUN; then
    echo "[DRY RUN] Would run: python scripts/eval_checkpoint_sweep.py --version v13 --three-class --n-volumes 10"
    echo "[DRY RUN] Verifying sweep script imports..."
    python -c "
from monai.networks.nets import SegResNet
from topometrics.leaderboard import compute_leaderboard_score
print('  All imports OK')
"
else
    echo "[$(date)] Running checkpoint sweep (SWI only, 10 volumes, 3-class)..."
    python scripts/eval_checkpoint_sweep.py --version v13 --three-class --n-volumes 10 2>&1
fi

echo ""
echo "=== Overnight v13 pipeline completed at $(date) ==="
echo "Check logs/checkpoint_sweep_v13.csv for detailed results."
