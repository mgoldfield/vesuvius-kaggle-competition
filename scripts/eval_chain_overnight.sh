#!/bin/bash
# Overnight eval chain — 4 remaining SWA blends
# Launched Feb 24 ~evening EST
set -e

EVAL_CMD="/workspace/venv/bin/python3 scripts/eval_transunet.py --cross-scroll --max-per-scroll 4 --t-low 0.70 --t-high 0.90"

echo "=== Eval chain started at $(date) ==="

echo ""
echo "=== [1/4] ViT balanced ep9 ==="
$EVAL_CMD \
  --weights checkpoints/swa_topo/swa_70pre_30unfreeze_vit_balanced_ep9.weights.h5 \
  2>&1 | tee logs/eval_swa_unfreeze_vit_balanced_ep9.log

echo ""
echo "=== [2/4] ViT high-LR ep1 ==="
$EVAL_CMD \
  --weights checkpoints/swa_topo/swa_70pre_30unfreeze_vit_highLR_ep1.weights.h5 \
  2>&1 | tee logs/eval_swa_unfreeze_vit_highLR_ep1.log

echo ""
echo "=== [3/4] Decoder balanced ep5 ==="
$EVAL_CMD \
  --weights checkpoints/swa_topo/swa_70pre_30unfreeze_decoder_balanced_ep5.weights.h5 \
  2>&1 | tee logs/eval_swa_unfreeze_decoder_balanced_ep5.log

echo ""
echo "=== [4/4] Decoder balanced ep10 ==="
$EVAL_CMD \
  --weights checkpoints/swa_topo/swa_70pre_30unfreeze_decoder_balanced_ep10.weights.h5 \
  2>&1 | tee logs/eval_swa_unfreeze_decoder_balanced_ep10.log

echo ""
echo "=== Eval chain complete at $(date) ==="
echo "Results in logs/transunet_eval.csv"
