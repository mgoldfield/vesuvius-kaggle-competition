#!/bin/bash
# Wait for probmap transfer to complete, then launch Phase 3 training on GPU 3.
# Run from GPU 1 (local):
#   nohup bash scripts/wait_and_launch_phase3.sh > logs/gpu3_launch.log 2>&1 &

set -e
cd /workspace/vesuvius-kaggle-competition

KEY="/root/.ssh/remote-gpu"
GPU3="root@103.196.86.227"
GPU3_PORT=13963
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30"

echo "=== Waiting for probmap transfer to GPU 3 ==="

# Wait for rsync to finish (check if rsync process is still running)
while pgrep -f "rsync.*103.196.86.227.*probmaps" > /dev/null 2>&1; do
    TRANSFERRED=$(ssh $SSH_OPTS -i $KEY -p $GPU3_PORT $GPU3 \
        "ls /workspace/vesuvius-kaggle-competition/data/refinement_data/probmaps/*.npy 2>/dev/null | wc -l" 2>/dev/null || echo "?")
    echo "[$(date)] Transfer in progress: $TRANSFERRED/786 probmaps"
    sleep 120
done

# Verify transfer completed
FINAL_COUNT=$(ssh $SSH_OPTS -i $KEY -p $GPU3_PORT $GPU3 \
    "ls /workspace/vesuvius-kaggle-competition/data/refinement_data/probmaps/*.npy 2>/dev/null | wc -l" 2>/dev/null)
echo "[$(date)] Transfer complete: $FINAL_COUNT/786 probmaps on GPU 3"

if [ "$FINAL_COUNT" -lt 700 ]; then
    echo "ERROR: Only $FINAL_COUNT probmaps transferred. Expected ~786. Aborting."
    exit 1
fi

# Launch Phase 3 training on GPU 3
echo "[$(date)] Launching Phase 3 overnight pipeline on GPU 3..."
ssh $SSH_OPTS -i $KEY -p $GPU3_PORT $GPU3 << 'REMOTE_EOF'
source /workspace/venv/bin/activate
nohup bash /workspace/vesuvius-kaggle-competition/scripts/overnight_refinement_phase3.sh \
    > /workspace/vesuvius-kaggle-competition/logs/refinement_phase3_training.log 2>&1 &
echo "Launched with PID: $!"
REMOTE_EOF

echo "[$(date)] Phase 3 launched on GPU 3!"
echo "=== Done ==="
