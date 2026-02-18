#!/bin/bash
# TransUNet overnight pipeline: benchmark → inference → sweep → exploration
#
# Usage:
#   bash scripts/overnight_transunet.sh 2>&1 | tee logs/overnight_transunet.log
#
# Dry run:
#   bash scripts/overnight_transunet.sh --dry-run 2>&1 | tee logs/overnight_transunet_dry.log

set -e
set -o pipefail

VENV="/workspace/venv/bin"
ROOT="/workspace/vesuvius-kaggle-competition"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# Wait for GPU memory to be (mostly) free before starting a phase.
# Prevents OOM from stale processes of crashed prior phases.
wait_for_gpu() {
    local max_wait=60  # seconds
    local interval=5
    local elapsed=0
    while (( elapsed < max_wait )); do
        local used
        used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
        if (( used < 500 )); then
            return 0
        fi
        echo "  GPU in use (${used} MiB), waiting ${interval}s..."
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    echo "  WARNING: GPU still in use after ${max_wait}s (${used} MiB), proceeding anyway"
}

DRY_RUN=""
N_EVAL=""
N_CROSS=""
N_TTA=""

if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    N_EVAL="--n-eval 2"
    N_TTA="--n-eval 2"
    N_CROSS="--max-per-scroll 1"
    echo "*** DRY RUN MODE ***"
else
    N_TTA="--n-eval 5"
    N_CROSS="--max-per-scroll 5"
fi

echo "============================================"
echo "TransUNet Overnight Pipeline"
echo "Started: $(date)"
echo "============================================"
echo ""

# ── Phase 1: Timing benchmark ──────────────────────────
echo "=== Phase 1: Timing Benchmark ==="
echo ""

# Benchmark is informational — don't let it kill the pipeline
$VENV/python "$ROOT/scripts/benchmark_transunet.py" $DRY_RUN \
    2>&1 | tee "$LOG_DIR/benchmark_transunet.log" || echo "WARNING: Benchmark failed, continuing..."

echo ""
echo "=== Phase 1 complete: $(date) ==="
echo ""

# ── Phase 2: Full validation inference ──────────────────
echo "=== Phase 2: TransUNet Validation Inference ==="
echo "Runs SWI on all val volumes (scroll 26002), saves probmaps"
echo ""
wait_for_gpu

$VENV/python "$ROOT/scripts/eval_transunet.py" \
    $N_EVAL \
    --save-probmaps \
    --downsample 1 \
    2>&1 | tee "$LOG_DIR/eval_transunet.log"

echo ""
echo "=== Phase 2 complete: $(date) ==="
echo ""

# ── Phase 3: Also run with TTA on subset ────────────────
echo "=== Phase 3: TransUNet with TTA (5 volumes) ==="
echo ""
wait_for_gpu

$VENV/python "$ROOT/scripts/eval_transunet.py" \
    $N_TTA \
    --tta \
    --downsample 1 \
    2>&1 | tee "$LOG_DIR/eval_transunet_tta.log"

echo ""
echo "=== Phase 3 complete: $(date) ==="
echo ""

# ── Phase 4: Cross-scroll evaluation ───────────────────
echo "=== Phase 4: Cross-scroll evaluation (5 per scroll) ==="
echo ""
wait_for_gpu

$VENV/python "$ROOT/scripts/eval_transunet.py" \
    --cross-scroll \
    $N_CROSS \
    --save-probmaps \
    --downsample 1 \
    2>&1 | tee "$LOG_DIR/eval_transunet_cross_scroll.log"

echo ""
echo "=== Phase 4 complete: $(date) ==="
echo ""

# ── Phase 5: Post-processing sweep ────────────────────
echo "=== Phase 5: Post-processing parameter sweep ==="
echo ""

$VENV/python "$ROOT/scripts/sweep_postprocessing.py" \
    --downsample 1 \
    $DRY_RUN \
    2>&1 | tee "$LOG_DIR/postprocessing_sweep.log"

echo ""
echo "=== Phase 5 complete: $(date) ==="
echo ""

# ── Phase 6: Data exploration notebook ─────────────────
echo "=== Phase 6: Data Exploration Notebook ==="
echo ""

# Pass DRY_RUN env var to notebook (set to "1" if --dry-run flag was passed)
if [[ -n "$DRY_RUN" ]]; then
    export DRY_RUN=1
else
    export DRY_RUN=0
fi

jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=venv \
    --output transunet_exploration_executed.ipynb \
    "$ROOT/notebooks/analysis/transunet_exploration.ipynb" \
    2>&1 | tee "$LOG_DIR/transunet_exploration.log"

echo ""
echo "=== Phase 6 complete: $(date) ==="
echo ""

# ── Summary ───────────────────────────────────────────
echo "============================================"
echo "Pipeline complete: $(date)"
echo "============================================"
echo ""
echo "Results:"
echo "  Benchmark:    $LOG_DIR/benchmark_transunet.log"
echo "  Val scores:   $LOG_DIR/eval_transunet.log"
echo "  Val+TTA:      $LOG_DIR/eval_transunet_tta.log"
echo "  Cross-scroll: $LOG_DIR/eval_transunet_cross_scroll.log"
echo "  PP sweep:     $LOG_DIR/postprocessing_sweep.log"
echo "  Exploration:  $ROOT/notebooks/analysis/transunet_exploration_executed.ipynb"
echo "  Plots:        $ROOT/plots/transunet_exploration/"
echo "  Probmaps:     $ROOT/data/transunet_probmaps/"
echo ""
echo "CSV outputs:"
echo "  $ROOT/logs/transunet_eval.csv"
echo "  $ROOT/logs/postprocessing_sweep.csv"
echo ""
echo "Next steps:"
echo "  1. Review exploration notebook plots"
echo "  2. Check best post-processing params"
echo "  3. Start fine-tuning: python scripts/train_transunet.py --epochs 10"
echo "  4. Submit to Kaggle"
