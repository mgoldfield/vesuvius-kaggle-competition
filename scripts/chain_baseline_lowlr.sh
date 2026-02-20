#!/bin/bash
# Chain script: after thin_fp_lowlr finishes on GPU1, run baseline_lowlr.
# baseline_lowlr = same loss as pretrained (skel=0.75, fp=0.50) but LR=5e-6
#
set -euo pipefail

echo "=== Baseline low-LR chain started: $(date) ==="
echo "  Waiting for thin_fp_lowlr to finish..."

# Wait for the thin_fp_lowlr chain to finish
while pgrep -f "chain_lowlr.sh thin_fp" > /dev/null 2>&1; do
    sleep 60
done

echo ""
echo "=== thin_fp_lowlr chain finished at $(date) ==="
echo "=== Launching baseline_lowlr (skel=0.75, fp=0.50, lr=5e-6) ==="

bash /workspace/vesuvius-kaggle-competition/scripts/train_and_eval_variant.sh \
    baseline_lowlr 0.75 0.50 0.0 1.0 "5e-6"

echo ""
echo "=== Baseline low-LR complete at $(date) ==="
