#!/bin/bash
# Generate probmaps for top-topo models, then run connectivity PP sweep on each
set -uo pipefail
cd /workspace/vesuvius-kaggle-competition
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
VENV=/workspace/venv/bin/python3
LOG=logs/gen_topo_model_probmaps.log

echo "=== Generating probmaps for top-topo models ===" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

# 1. frozen_margin_dist_ep5 (topo=0.2679)
echo "" | tee -a "$LOG"
echo "--- frozen_margin_dist_ep5 ---" | tee -a "$LOG"
$VENV scripts/generate_probmaps.py \
    --weights checkpoints/transunet_frozen_margin_dist/transunet_ep5.weights.h5 \
    --scroll-ids 26002 \
    --include-val \
    --output-dir data/frozen_margin_dist_ep5_probmaps \
    2>&1 | tee -a "$LOG"
echo "Done: $(date)" | tee -a "$LOG"

# 2. frozen_boundary_ep10 (topo=0.2642)
echo "" | tee -a "$LOG"
echo "--- frozen_boundary_ep10 ---" | tee -a "$LOG"
$VENV scripts/generate_probmaps.py \
    --weights checkpoints/transunet_frozen_boundary/transunet_ep10.weights.h5 \
    --scroll-ids 26002 \
    --include-val \
    --output-dir data/frozen_boundary_ep10_probmaps \
    --skip-existing \
    2>&1 | tee -a "$LOG"
echo "Done: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== ALL PROBMAPS DONE at $(date) ===" | tee -a "$LOG"

# 3. Wait for the current PP sweep pipeline to finish, then run PP on these
echo "Waiting for current PP sweep to finish..." | tee -a "$LOG"
while pgrep -f "sweep_connectivity_pp" > /dev/null; do sleep 60; done
echo "PP sweep done. Starting topo-model PP sweeps..." | tee -a "$LOG"

# 4. PP sweep on frozen_margin_dist_ep5 probmaps
$VENV scripts/sweep_connectivity_pp.py \
    --probmap-dir data/frozen_margin_dist_ep5_probmaps --n-eval 20 \
    > logs/connectivity_pp_frozen_margin_dist_ep5_sweep.log 2>&1
echo "=== MARGIN DIST PP SWEEP DONE ===" >> logs/connectivity_pp_frozen_margin_dist_ep5_sweep.log

# 5. PP sweep on frozen_boundary_ep10 probmaps
$VENV scripts/sweep_connectivity_pp.py \
    --probmap-dir data/frozen_boundary_ep10_probmaps --n-eval 20 \
    > logs/connectivity_pp_frozen_boundary_ep10_sweep.log 2>&1
echo "=== FROZEN BOUNDARY PP SWEEP DONE ===" >> logs/connectivity_pp_frozen_boundary_ep10_sweep.log

echo "=== ALL TOPO-MODEL PP SWEEPS DONE at $(date) ===" | tee -a "$LOG"
