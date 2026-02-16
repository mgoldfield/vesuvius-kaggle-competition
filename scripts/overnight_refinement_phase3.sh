#!/bin/bash
# Overnight pipeline for GPU 3: train refinement Phase 3, then evaluate
# Run on GPU 3 with:
#   nohup bash /workspace/vesuvius-kaggle-competition/scripts/overnight_refinement_phase3.sh \
#     > /workspace/vesuvius-kaggle-competition/logs/refinement_phase3_training.log 2>&1 &
# Dry run:  bash /workspace/vesuvius-kaggle-competition/scripts/overnight_refinement_phase3.sh --dry-run

set -e
cd /workspace/vesuvius-kaggle-competition
source /workspace/venv/bin/activate

DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

echo "=== Refinement Phase 3 overnight pipeline started at $(date) ==="

# Step 1: Verify data is available
echo "[$(date)] Checking data availability..."
PROBMAP_COUNT=$(ls data/refinement_data/probmaps/*.npy 2>/dev/null | wc -l)
LABEL_COUNT=$(ls data/train_labels/*.tif 2>/dev/null | wc -l)
echo "  Probmaps: $PROBMAP_COUNT, Labels: $LABEL_COUNT"

if [ "$PROBMAP_COUNT" -lt 700 ]; then
    echo "ERROR: Only $PROBMAP_COUNT probmaps found. Expected ~786. Aborting."
    exit 1
fi

echo "  Phase 2 checkpoint:"
ls -la checkpoints/models/best_refinement_phase2.pth 2>/dev/null || {
    echo "ERROR: Phase 2 checkpoint not found!"
    exit 1
}

echo "  Notebook:"
ls -la notebooks/refinement/vesuvius_train_refinement_phase3.ipynb || {
    echo "ERROR: Phase 3 notebook not found!"
    exit 1
}

# Step 2: Verify imports
echo "[$(date)] Verifying Python imports..."
python -c "
import torch, numpy, pandas, tifffile, fastai, monai, scipy
from topometrics.leaderboard import compute_leaderboard_score
print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print('  All imports OK')
"

# Step 3: Train Phase 3
if $DRY_RUN; then
    echo "[DRY RUN] Would run: jupyter nbconvert --execute ... vesuvius_train_refinement_phase3.ipynb"
    echo "[DRY RUN] Training would take ~2-3 hours (30 epochs, 786 volumes)"
else
    echo "[$(date)] Starting Phase 3 training..."
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.kernel_name=venv \
        --output vesuvius_train_refinement_phase3_executed.ipynb \
        notebooks/refinement/vesuvius_train_refinement_phase3.ipynb 2>&1
    echo "[$(date)] Phase 3 training completed!"
fi

# Step 4: Check checkpoint exists
echo "[$(date)] Checking for Phase 3 checkpoint..."
ls -la checkpoints/models/best_refinement_phase3.pth 2>/dev/null || {
    if $DRY_RUN; then
        echo "  [DRY RUN] No checkpoint yet (expected — training hasn't run)"
    else
        echo "ERROR: No Phase 3 checkpoint found!"
        exit 1
    fi
}

# Step 5: Run head-to-head evaluation
if $DRY_RUN; then
    echo "[DRY RUN] Would run: python scripts/eval_refinement.py --phase 3 --n-volumes 20"
    echo "[DRY RUN] Verifying eval script..."
    python scripts/eval_refinement.py --help 2>&1 | head -3
else
    echo "[$(date)] Running head-to-head evaluation (20 val volumes)..."
    python scripts/eval_refinement.py --phase 3 --n-volumes 20 2>&1
fi

echo ""
echo "=== Refinement Phase 3 pipeline completed at $(date) ==="
echo "Check logs/refinement_eval.csv for detailed results."
