#!/bin/bash
# Overnight queue for GPU 1: Option B evals + Phase 3 artifact pull + retest
# Run with: nohup bash scripts/overnight_gpu1_queue.sh > logs/overnight_gpu1_queue.log 2>&1 &
# Dry run:  bash scripts/overnight_gpu1_queue.sh --dry-run

set -e
cd /workspace/vesuvius-kaggle-competition
source /workspace/venv/bin/activate

KEY="/root/.ssh/remote-gpu"
GPU3="root@REDACTED"
GPU3_PORT=REDACTED
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30"

DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

# In dry-run, use 1 volume; in real mode, use 20
if $DRY_RUN; then N_VOLS=1; else N_VOLS=20; fi

echo "=== GPU 1 overnight queue started at $(date) ==="

# ── Step 0: Verify all checkpoints and imports ────────────────────────
echo ""
echo "[$(date)] Verifying checkpoints..."
for f in checkpoints/models/segresnet_v12_ep20.pth \
         checkpoints/models/segresnet_v12_ep45.pth \
         checkpoints/models/segresnet_v13_ep15.pth; do
    ls -la "$f" 2>/dev/null || { echo "ERROR: $f not found!"; exit 1; }
done

echo "[$(date)] Verifying imports..."
python -c "
import torch
from monai.networks.nets import SegResNet
from topometrics.leaderboard import compute_leaderboard_score
print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print('  All imports OK')
"

# ── Step 1: Option B — v12 ep20 with TTA ─────────────────────────────
echo ""
echo "[$(date)] Step 1: v12 ep20 with TTA ($N_VOLS volumes)..."
python scripts/eval_checkpoint_sweep.py \
    --checkpoint checkpoints/models/segresnet_v12_ep20.pth \
    --tta --n-volumes $N_VOLS 2>&1
echo "[$(date)] Step 1 complete."

# ── Step 2: v12 ep45 with TTA ────────────────────────────────────────
# Later epochs have better topo — check if TTA changes the ranking
echo ""
echo "[$(date)] Step 2: v12 ep45 with TTA ($N_VOLS volumes)..."
python scripts/eval_checkpoint_sweep.py \
    --checkpoint checkpoints/models/segresnet_v12_ep45.pth \
    --tta --n-volumes $N_VOLS 2>&1
echo "[$(date)] Step 2 complete."

# ── Step 3: v13 ep15 with TTA — best v13 checkpoint ──────────────────
echo ""
echo "[$(date)] Step 3: v13 ep15 (best 3-class) with TTA ($N_VOLS volumes)..."
python scripts/eval_checkpoint_sweep.py \
    --checkpoint checkpoints/models/segresnet_v13_ep15.pth \
    --three-class --tta --n-volumes $N_VOLS 2>&1
echo "[$(date)] Step 3 complete."
echo "(v9 baseline reference: ~0.570 comp_score from prior evals)"

# ── Step 4: Wait for Phase 3 on GPU 3, pull artifacts ────────────────
echo ""
echo "[$(date)] Step 4: Waiting for Phase 3 to complete on GPU 3..."

if $DRY_RUN; then
    # In dry-run, just check connectivity
    RUNNING=$(ssh $SSH_OPTS -i $KEY -p $GPU3_PORT $GPU3 \
        "pgrep -f 'overnight_refinement_phase3' > /dev/null 2>&1 && echo 'yes' || echo 'no'" 2>/dev/null || echo "unreachable")
    echo "  [DRY RUN] GPU 3 status: $RUNNING"
    echo "  [DRY RUN] Would wait for Phase 3 to finish, then rsync artifacts"
else
    MAX_WAIT=14400  # 4 hours max
    WAITED=0
    while [ $WAITED -lt $MAX_WAIT ]; do
        RUNNING=$(ssh $SSH_OPTS -i $KEY -p $GPU3_PORT $GPU3 \
            "pgrep -f 'overnight_refinement_phase3' > /dev/null 2>&1 && echo 'yes' || echo 'no'" 2>/dev/null || echo "unreachable")

        if [ "$RUNNING" = "no" ]; then
            echo "[$(date)] Phase 3 pipeline finished on GPU 3."
            break
        elif [ "$RUNNING" = "unreachable" ]; then
            echo "[$(date)] GPU 3 unreachable. Skipping Phase 3 pull."
            break
        fi

        echo "[$(date)] Phase 3 still running on GPU 3... (waited ${WAITED}s)"
        sleep 120
        WAITED=$((WAITED + 120))
    done

    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "[$(date)] Timed out waiting for Phase 3 (${MAX_WAIT}s). Skipping."
    fi

    # Pull Phase 3 checkpoint
    echo "[$(date)] Pulling Phase 3 artifacts from GPU 3..."
    rsync -avz -e "ssh $SSH_OPTS -i $KEY -p $GPU3_PORT" \
        $GPU3:/workspace/vesuvius-kaggle-competition/checkpoints/models/best_refinement_phase3.pth \
        checkpoints/models/ 2>&1 || echo "  Failed to pull Phase 3 checkpoint"

    rsync -avz -e "ssh $SSH_OPTS -i $KEY -p $GPU3_PORT" \
        $GPU3:/workspace/vesuvius-kaggle-competition/logs/refinement_phase3_training.log \
        logs/ 2>&1 || echo "  Failed to pull Phase 3 log"
fi

# ── Step 5: Option A with Phase 3 weights ────────────────────────────
if [ -f checkpoints/models/best_refinement_phase3.pth ]; then
    echo ""
    echo "[$(date)] Step 5: Option A with Phase 3 weights ($N_VOLS volumes)..."
    python scripts/eval_refinement.py --phase 3 --n-volumes $N_VOLS 2>&1
    echo "[$(date)] Step 5 complete."
else
    echo "[$(date)] No Phase 3 checkpoint available. Skipping Step 5."
fi

echo ""
echo "=== GPU 1 overnight queue completed at $(date) ==="
echo "Check logs/overnight_gpu1_queue.log for all results."
