#!/bin/bash
# Chain script: wait for discrim_dist_sq eval to finish, then launch discrim_boundary.
# Run under tmux on gpu1:
#   tmux new-session -d -s chain 'cd /workspace/vesuvius-kaggle-competition && bash scripts/chain_gpu1_discrim_boundary.sh > logs/chain_gpu1.log 2>&1'
#
set -euo pipefail
cd /workspace/vesuvius-kaggle-competition

echo "=== Chain: waiting for discrim_dist_sq to finish ==="
echo "  Started: $(date)"

# Wait for any train/eval processes to finish
while pgrep -f 'train_transunet|eval_transunet' > /dev/null 2>&1; do
    LAST=$(grep -E '^Epoch|comp=|DONE|Evaluating' logs/train_eval_discrim_dist_sq.log 2>/dev/null | tail -1)
    echo "[$(date)] Waiting... Last: $LAST"
    sleep 120
done

echo "[$(date)] gpu1 is free. Launching discrim_boundary..."
echo ""

# Launch the next experiment
bash scripts/launch_gpu1_discrim_boundary.sh
