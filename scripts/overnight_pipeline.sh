#!/bin/bash
# Overnight automation pipeline:
# 1. Wait for v10 training to complete
# 2. Trace v10 model, upload to Kaggle, push inference notebook
# 3. Run v11 training (3-fold CV)
# 4. (v11 submission will be done manually after reviewing results)
#
# Usage: nohup bash scripts/overnight_pipeline.sh > logs/overnight.log 2>&1 &

set -e

ROOT="/home/mongomatt/Projects/vesuvius"
VENV="$ROOT/vesuvius/bin"
PYTHON="$VENV/python"
JUPYTER="$VENV/jupyter"
KAGGLE="$VENV/kaggle"
SCRIPTS="$ROOT/scripts"
CKPT="$ROOT/checkpoints/models"
KAGGLE_WEIGHTS="$ROOT/kaggle/kaggle_weights"
KAGGLE_NOTEBOOK="$ROOT/kaggle/kaggle_notebook"
LOGS="$ROOT/logs"

V10_PID=8527

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "$(timestamp) === OVERNIGHT PIPELINE STARTED ==="
echo "$(timestamp) V10 training PID: $V10_PID"

# ── Step 1: Wait for v10 training ────────────────────────
echo "$(timestamp) [1/4] Waiting for v10 training to complete..."
while kill -0 $V10_PID 2>/dev/null; do
    sleep 300  # check every 5 minutes
    echo "$(timestamp)   v10 still running (PID $V10_PID)..."
done
echo "$(timestamp) [1/4] v10 training complete!"

# Verify checkpoint exists
if [ ! -f "$CKPT/best_segresnet_v10.pth" ]; then
    echo "$(timestamp) ERROR: v10 checkpoint not found at $CKPT/best_segresnet_v10.pth"
    echo "$(timestamp) Check $LOGS/v10_execution.log for errors"
    exit 1
fi
echo "$(timestamp) v10 checkpoint found: $(ls -lh $CKPT/best_segresnet_v10.pth)"

# ── Step 2: Trace, upload, and submit v10 ────────────────
echo "$(timestamp) [2/4] Tracing v10 model..."
$PYTHON "$SCRIPTS/trace_model.py" \
    --checkpoint "$CKPT/best_segresnet_v10.pth" \
    --output "$KAGGLE_WEIGHTS/best_segresnet_v10_traced.pt"

echo "$(timestamp) Uploading v10 weights to Kaggle..."
# Update the inference script to point to v10 weights
cp "$KAGGLE_NOTEBOOK/vesuvius-inference.py" "$KAGGLE_NOTEBOOK/vesuvius-inference.py.bak"
sed -i 's/best_segresnet_v9_traced\.pt/best_segresnet_v10_traced.pt/g' "$KAGGLE_NOTEBOOK/vesuvius-inference.py"

# Upload weights
$KAGGLE datasets version -p "$KAGGLE_WEIGHTS/" -m "v10 weights (deep supervision + attention gates)" --quiet

# Wait for dataset to be ready
echo "$(timestamp) Waiting for dataset to be ready..."
sleep 60

# Push inference notebook
echo "$(timestamp) Pushing v10 inference notebook..."
$KAGGLE kernels push -p "$KAGGLE_NOTEBOOK/"

echo "$(timestamp) [2/4] v10 submitted! Monitor at:"
echo "  https://www.kaggle.com/code/mgoldfield/vesuvius-surface-detection-inference"

# ── Step 3: Evaluate improved inference pipeline ─────────
echo "$(timestamp) [3/4] Running inference pipeline comparison (old vs new)..."
echo "$(timestamp)   Comparing: uniform SWI + prob TTA vs Gaussian SWI + logit TTA"
echo "$(timestamp)   With threshold sweep on val set"

# Use v9 checkpoint first (known good), then v10 if available
EVAL_CKPT="$CKPT/best_segresnet_v9.pth"
if [ -f "$CKPT/best_segresnet_v10.pth" ]; then
    EVAL_CKPT="$CKPT/best_segresnet_v10.pth"
    echo "$(timestamp)   Using v10 checkpoint for eval"
else
    echo "$(timestamp)   v10 checkpoint not found, using v9 for eval"
fi

$PYTHON "$SCRIPTS/eval_inference.py" \
    --checkpoint "$EVAL_CKPT" \
    --n-eval 5 \
    --sweep \
    --split \
    > "$LOGS/eval_inference.log" 2>&1

echo "$(timestamp) [3/4] Inference eval complete. Results in logs/eval_inference.log"

# Also run with v9 if we used v10 above (compare both models)
if [ "$EVAL_CKPT" != "$CKPT/best_segresnet_v9.pth" ] && [ -f "$CKPT/best_segresnet_v9.pth" ]; then
    echo "$(timestamp) [3b/4] Also evaluating v9 checkpoint for comparison..."
    $PYTHON "$SCRIPTS/eval_inference.py" \
        --checkpoint "$CKPT/best_segresnet_v9.pth" \
        --n-eval 5 \
        --sweep \
        --split \
        > "$LOGS/eval_inference_v9.log" 2>&1
    echo "$(timestamp) [3b/4] v9 eval complete. Results in logs/eval_inference_v9.log"
fi

echo ""
echo "$(timestamp) === OVERNIGHT PIPELINE COMPLETE ==="
echo "$(timestamp) Results to check in the morning:"
echo "  1. v10 Kaggle submission: kaggle kernels status mgoldfield/vesuvius-surface-detection-inference"
echo "  2. Inference eval (new vs old): cat logs/eval_inference.log"
echo "  3. Threshold sweep results: tail -20 logs/eval_inference.log"
echo "  4. If v9 also evaluated: cat logs/eval_inference_v9.log"
