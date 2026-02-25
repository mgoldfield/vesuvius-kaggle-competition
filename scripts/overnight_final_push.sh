#!/bin/bash
# Overnight Final Push Pipeline
# Runs: eval SWA blends → train on all 786 → SWA blend → upload → submit
# Started: Feb 25 ~12:50 AM EST
# Expected completion: ~8-9 AM EST
set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/vesuvius-kaggle-competition

VENV="/workspace/venv/bin/python"
LOG_DIR="logs"
CKPT_DIR="checkpoints"
SWA_DIR="$CKPT_DIR/swa_topo"
PRETRAINED="pretrained_weights/transunet/transunet.seresnext50.160px.comboloss.weights.h5"
KAGGLE_WEIGHTS="kaggle/kaggle_weights_download"

echo "=============================================="
echo "OVERNIGHT FINAL PUSH PIPELINE"
echo "Started: $(date)"
echo "=============================================="

# ──────────────────────────────────────────────────
# PHASE 1: Wait for ep10 eval + run ep4 eval (~20 min)
# ──────────────────────────────────────────────────
echo ""
echo "=== PHASE 1: SWA Blend Evals ==="

# Wait for ep10 eval (already running)
echo "[$(date +%H:%M)] Waiting for ep10 cross-scroll eval to finish..."
while pgrep -f "eval_transunet.*swa_70pre_30external_data_ep10" > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date +%H:%M)] ep10 eval done."

# Extract ep10 result
EP10_COMP=$(grep "comp_score:" "$LOG_DIR/eval_swa_external_ep10_cross.log" | tail -1 | grep -oP '[\d.]+' | head -1)
echo "[$(date +%H:%M)] SWA 70/30 external_data ep10 cross-scroll comp: $EP10_COMP"

# Run ep4 eval
echo "[$(date +%H:%M)] Running ep4 SWA blend cross-scroll eval..."
$VENV scripts/eval_transunet.py \
    --cross-scroll --max-per-scroll 4 --t-low 0.70 --t-high 0.90 \
    --weights "$SWA_DIR/swa_70pre_30external_data_ep4.weights.h5" \
    2>&1 | tee "$LOG_DIR/eval_swa_external_ep4_cross.log"

EP4_COMP=$(grep "comp_score:" "$LOG_DIR/eval_swa_external_ep4_cross.log" | tail -1 | grep -oP '[\d.]+' | head -1)
echo "[$(date +%H:%M)] SWA 70/30 external_data ep4 cross-scroll comp: $EP4_COMP"

# Pick best init weights for train-all
# Compare ep4, ep10 blends, and current best (0.5551)
echo ""
echo "=== PHASE 1 RESULTS ==="
echo "  SWA ep4:  $EP4_COMP"
echo "  SWA ep10: $EP10_COMP"
echo "  Current best (margin_dist_ep5): 0.5551"

# Always use margin_dist_ep5 as the base for train-all (proven best)
# But record the external data results for ensemble selection
TRAIN_ALL_INIT="$SWA_DIR/swa_70pre_30margin_dist_ep5.weights.h5"
echo "  Using $TRAIN_ALL_INIT as train-all init weights"

# ──────────────────────────────────────────────────
# PHASE 2: Dry-run train-all (~5 min)
# ──────────────────────────────────────────────────
echo ""
echo "=== PHASE 2: Dry-run train-all ==="
echo "[$(date +%H:%M)] Running dry-run..."

$VENV scripts/train_transunet.py \
    --run-name all_data_frozen \
    --weights "$TRAIN_ALL_INIT" \
    --freeze-encoder \
    --train-all \
    --epochs 15 \
    --lr 5e-5 \
    --grad-accum 4 \
    --dist-weight 0.1 --dist-margin 3 \
    --boundary-weight 0.3 \
    --skel-weight 0.75 \
    --fp-weight 0.5 \
    --save-every 5 \
    --dry-run \
    2>&1 | tee "$LOG_DIR/train_all_dry_run.log"

if [ $? -ne 0 ]; then
    echo "FATAL: Dry-run failed! Aborting pipeline."
    exit 1
fi
echo "[$(date +%H:%M)] Dry-run passed."

# ──────────────────────────────────────────────────
# PHASE 3: Full train-all (~5 hours)
# ──────────────────────────────────────────────────
echo ""
echo "=== PHASE 3: Train on all 786 volumes ==="
echo "[$(date +%H:%M)] Launching full training (15 epochs, ~5 hours)..."

$VENV scripts/train_transunet.py \
    --run-name all_data_frozen \
    --weights "$TRAIN_ALL_INIT" \
    --freeze-encoder \
    --train-all \
    --epochs 15 \
    --lr 5e-5 \
    --grad-accum 4 \
    --dist-weight 0.1 --dist-margin 3 \
    --boundary-weight 0.3 \
    --skel-weight 0.75 \
    --fp-weight 0.5 \
    --save-every 5 \
    2>&1 | tee "$LOG_DIR/train_all_data_frozen.log"

if [ $? -ne 0 ]; then
    echo "FATAL: Training failed! Skipping to submission with existing best."
    # Fall through to submit existing best model
    TRAIN_ALL_FAILED=1
fi

echo "[$(date +%H:%M)] Training complete."

# ──────────────────────────────────────────────────
# PHASE 4: SWA blend train-all checkpoints (~2 min)
# ──────────────────────────────────────────────────
echo ""
echo "=== PHASE 4: SWA Blend train-all ==="

if [ -z "$TRAIN_ALL_FAILED" ]; then
    # Create SWA blends for ep5, ep10, ep15
    for EP in 5 10 15; do
        CKPT="$CKPT_DIR/transunet_all_data_frozen/transunet_ep${EP}.weights.h5"
        if [ -f "$CKPT" ]; then
            OUT="$SWA_DIR/swa_70pre_30all_data_ep${EP}.weights.h5"
            echo "[$(date +%H:%M)] Blending ep${EP}..."
            $VENV scripts/swa_average.py \
                --checkpoints "$PRETRAINED" "$CKPT" \
                --weights 0.7 0.3 \
                --output "$OUT" 2>&1
            echo "  Created: $OUT"
        fi
    done
fi

# ──────────────────────────────────────────────────
# PHASE 5: Upload weights + submit to Kaggle
# ──────────────────────────────────────────────────
echo ""
echo "=== PHASE 5: Upload + Submit ==="

# Delete stale upload tokens
rm -f /tmp/.kaggle/uploads/*.json 2>/dev/null

# Copy weights to Kaggle dataset dir
echo "[$(date +%H:%M)] Copying weights to Kaggle dataset..."

# Always include current best
# cp "$SWA_DIR/swa_70pre_30margin_dist_ep5.weights.h5" "$KAGGLE_WEIGHTS/" (already there)

# Add external data blend if it improved
if [ -f "$SWA_DIR/swa_70pre_30external_data_ep10.weights.h5" ]; then
    cp "$SWA_DIR/swa_70pre_30external_data_ep10.weights.h5" "$KAGGLE_WEIGHTS/"
    echo "  Copied swa_70pre_30external_data_ep10"
fi

# Add train-all blends
if [ -z "$TRAIN_ALL_FAILED" ]; then
    for EP in 5 10 15; do
        SWA_FILE="$SWA_DIR/swa_70pre_30all_data_ep${EP}.weights.h5"
        if [ -f "$SWA_FILE" ]; then
            cp "$SWA_FILE" "$KAGGLE_WEIGHTS/"
            echo "  Copied swa_70pre_30all_data_ep${EP}"
        fi
    done
fi

# Upload dataset
echo "[$(date +%H:%M)] Uploading Kaggle dataset (may hang at 0% for ~8 min, normal)..."
cd "$KAGGLE_WEIGHTS"
kaggle datasets version -m "Final push: train-all blends + external data blends" --dir-mode zip 2>&1
cd /workspace/vesuvius-kaggle-competition
echo "[$(date +%H:%M)] Upload complete."

# ──────────────────────────────────────────────────
# PHASE 6: Submit notebook (single model first, safest)
# ──────────────────────────────────────────────────
echo ""
echo "=== PHASE 6: Submit Kaggle Notebook ==="

# Submit with current best weights (no changes needed to notebook — already configured)
echo "[$(date +%H:%M)] Pushing notebook to Kaggle..."
kaggle kernels push -p kaggle/kaggle_notebook/ 2>&1
echo "[$(date +%H:%M)] Notebook pushed."

# Wait for it to start
sleep 60
echo "[$(date +%H:%M)] Checking notebook status..."
kaggle kernels status mgoldfield/vesuvius-surface-detection-inference 2>&1

# ──────────────────────────────────────────────────
# PHASE 7: Summary
# ──────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "OVERNIGHT PIPELINE COMPLETE"
echo "Finished: $(date)"
echo "=============================================="
echo ""
echo "Results summary:"
echo "  SWA ep4 cross-scroll:  $EP4_COMP"
echo "  SWA ep10 cross-scroll: $EP10_COMP"
echo "  Current best:          0.5551"
if [ -z "$TRAIN_ALL_FAILED" ]; then
    echo "  Train-all:             completed (check logs/train_all_data_frozen.log)"
    echo "  SWA blends:            ep5/ep10/ep15 created"
else
    echo "  Train-all:             FAILED (submitted existing best)"
fi
echo ""
echo "Next steps (morning):"
echo "  1. Check Kaggle status:  kaggle kernels status mgoldfield/vesuvius-surface-detection-inference"
echo "  2. Check score:          kaggle competitions submissions -c vesuvius-challenge-surface-detection"
echo "  3. If train-all produced good blends, update notebook weights and resubmit"
echo "  4. Optionally submit ensemble variant (uncomment 2nd model in vesuvius-inference.py)"
echo ""
echo "Train-all checkpoints: $CKPT_DIR/transunet_all_data_frozen/"
echo "SWA blends: $SWA_DIR/swa_70pre_30all_data_ep*.weights.h5"
echo "Training log: $LOG_DIR/train_all_data_frozen.log"
