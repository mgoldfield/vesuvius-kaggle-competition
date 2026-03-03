#!/bin/bash
# Pull training artifacts from remote GPUs back to primary /workspace/ volume.
# Run from GPU 1 (local) after remote training completes.
#
# Usage: bash scripts/pull_artifacts.sh

set -e
cd /workspace/vesuvius-kaggle-competition

KEY="/root/.ssh/remote-gpu"
GPU2="root@REDACTED"
GPU2_PORT=REDACTED
GPU3="root@REDACTED"
GPU3_PORT=REDACTED

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
BASE="/workspace/vesuvius-kaggle-competition"

echo "=== Pulling artifacts from remote GPUs ==="

# GPU 2: v13 checkpoints + logs
echo ""
echo "--- GPU 2 (v13) ---"
rsync -avz -e "ssh $SSH_OPTS -i $KEY -p $GPU2_PORT" \
    "$GPU2:$BASE/checkpoints/models/segresnet_v13_ep*.pth" \
    "$GPU2:$BASE/checkpoints/models/best_segresnet_v13.pth" \
    checkpoints/models/ 2>/dev/null || echo "  No v13 checkpoints found on GPU 2"

rsync -avz -e "ssh $SSH_OPTS -i $KEY -p $GPU2_PORT" \
    "$GPU2:$BASE/logs/overnight_v13.log" \
    "$GPU2:$BASE/logs/checkpoint_sweep_v13.csv" \
    "$GPU2:$BASE/logs/v13_training.log" \
    logs/ 2>/dev/null || echo "  Some v13 logs not found on GPU 2"

echo "  GPU 2 artifacts pulled."

# GPU 3: refinement Phase 3 checkpoints + logs
echo ""
echo "--- GPU 3 (refinement Phase 3) ---"
rsync -avz -e "ssh $SSH_OPTS -i $KEY -p $GPU3_PORT" \
    "$GPU3:$BASE/checkpoints/models/best_refinement_phase3.pth" \
    checkpoints/models/ 2>/dev/null || echo "  No Phase 3 checkpoint found on GPU 3"

rsync -avz -e "ssh $SSH_OPTS -i $KEY -p $GPU3_PORT" \
    "$GPU3:$BASE/logs/refinement_phase3_eval.csv" \
    "$GPU3:$BASE/logs/refinement_phase3_training.log" \
    logs/ 2>/dev/null || echo "  Some Phase 3 logs not found on GPU 3"

# Also grab the traced model if it exists
rsync -avz -e "ssh $SSH_OPTS -i $KEY -p $GPU3_PORT" \
    "$GPU3:$BASE/kaggle/kaggle_weights/refinement_phase3_traced.pt" \
    kaggle/kaggle_weights/ 2>/dev/null || echo "  No traced Phase 3 model found on GPU 3"

echo "  GPU 3 artifacts pulled."

echo ""
echo "=== Done! Artifacts pulled at $(date) ==="
echo "Check:"
echo "  ls -la checkpoints/models/segresnet_v13_ep*.pth"
echo "  ls -la checkpoints/models/best_refinement_phase3.pth"
echo "  cat logs/overnight_v13.log"
echo "  cat logs/refinement_phase3_eval.csv"
