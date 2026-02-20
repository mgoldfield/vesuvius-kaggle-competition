#!/bin/bash
# Wait for Run 2 to finish, then launch Run 3
# Usage: nohup bash scripts/launch_run3.sh <run2_pid> > logs/launch_run3.log 2>&1 &

RUN2_PID=$1
echo "Waiting for Run 2 (PID $RUN2_PID) to finish..."

while kill -0 $RUN2_PID 2>/dev/null; do
    sleep 60
done

echo "Run 2 finished at $(date). Launching Run 3..."
/workspace/venv/bin/python scripts/train_transunet.py \
    --run-name thin_dist \
    --epochs 25 \
    --save-every 5 \
    --fp-weight 1.5 \
    --dist-weight 1.0 \
    > logs/train_thin_dist.log 2>&1

echo "Run 3 finished at $(date)."
