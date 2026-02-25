#!/bin/bash
# Wait for eval chain to finish, then run baseline + ep1 re-eval
echo "[$(date)] Waiting for eval chain (PID 4057482) to finish..."

while kill -0 4057482 2>/dev/null; do
    sleep 30
done

echo "[$(date)] Eval chain finished. Starting baseline + ep1 re-eval..."
sleep 10  # Let GPU memory fully free

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/eval_baseline_val_only.sh
