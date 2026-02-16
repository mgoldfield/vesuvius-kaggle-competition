#!/bin/bash
# Overnight pipeline for GPU 1 (local): wait for v12 training, then sweep checkpoints
# Run with: nohup bash scripts/overnight_v12.sh > logs/overnight_v12.log 2>&1 &
# Dry run:  bash scripts/overnight_v12.sh --dry-run

set -e
cd /workspace/vesuvius-kaggle-competition
source /workspace/venv/bin/activate

DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

echo "=== Overnight v12 pipeline started at $(date) ==="

# Step 1: Wait for v12 training to finish
if $DRY_RUN; then
    if pgrep -f "vesuvius_train_v12.ipynb" > /dev/null 2>&1; then
        echo "[DRY RUN] v12 training is still running — would wait here"
    else
        echo "[DRY RUN] v12 training is NOT running — would proceed immediately"
    fi
else
    echo "[$(date)] Waiting for v12 training (nbconvert) to complete..."
    while pgrep -f "vesuvius_train_v12.ipynb" > /dev/null 2>&1; do
        sleep 120
    done
    echo "[$(date)] v12 training completed!"
fi

# Step 2: Check if checkpoints exist
echo "[$(date)] Checking for v12 checkpoints..."
ls -la checkpoints/models/segresnet_v12_ep*.pth 2>/dev/null || echo "  No periodic checkpoints found (expected if training hasn't finished)"
ls -la checkpoints/models/best_segresnet_v12.pth 2>/dev/null || echo "  No best checkpoint found (expected if training hasn't finished)"

# Step 3: Run checkpoint sweep
if $DRY_RUN; then
    echo "[DRY RUN] Would run: python scripts/eval_checkpoint_sweep.py --version v12 --n-volumes 10"
    echo "[DRY RUN] Verifying sweep script imports..."
    python -c "import scripts.eval_checkpoint_sweep; print('  eval_checkpoint_sweep.py imports OK')" 2>/dev/null || \
    python scripts/eval_checkpoint_sweep.py --version v12 --n-volumes 0 2>&1 | head -5 || \
    echo "  (sweep script will be tested when checkpoints exist)"
else
    echo "[$(date)] Running checkpoint sweep (SWI only, 10 volumes)..."
    python scripts/eval_checkpoint_sweep.py --version v12 --n-volumes 10 2>&1
fi

echo ""
echo "=== Overnight v12 pipeline completed at $(date) ==="
echo "Check logs/checkpoint_sweep_v12.csv for detailed results."
