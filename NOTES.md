# Vesuvius Surface Detection — Project Notes

Active working document. Historical run details and design rationale are in **HISTORY.md**
(blog source material). **Move completed work to HISTORY.md** when it's no longer needed
for active decision-making — keep this file lean.

## Competition
- **Goal:** Detect papyrus surfaces in 3D CT scans of Herculaneum scrolls
- **Metric:** `0.30*TopoScore + 0.35*SurfaceDice@tau=2 + 0.35*VOI_score`
- **Deadline:** Feb 27, 2026 (5 days remaining)
- **Submission:** Code competition — Kaggle notebook, GPU, ≤9hr, no internet
- **Leaderboard:** 1,334 teams. Top score 0.607. Our best public: 0.504 (v20, TransUNet)
- **Public test: only 1 volume** (ID 1407735). Scores are high-variance/unreliable.

## Data
- 786 training volumes, 320^3 uint8. Labels: 0=bg, 1=fg (sparse ~2-8%), 2=unlabeled (ignore)
- ~120 hidden test volumes. Val: scroll 26002 holdout (82 volumes)
- 6 scroll_ids: 34117 (382), 35360 (176), 26010 (130), 26002 (88), 44430 (17), 53997 (13)

## Hardware

| Name | GPU | SSH | Role |
|---|---|---|---|
| **gpu0** (this machine) | RTX 5090 32GB | local | Primary control. Pseudo-label training (no clDice). |
| **gpu2** | RTX 6000 Ada 48GB | `ssh -i ~/.ssh/remote-gpu root@195.26.233.87 -p 25763` | Pseudo-label training WITH clDice (needs 48GB). |

- gpu1 (RTX 5090): **decommissioned** Feb 22. All data synced to gpu0.
- Venv: `/workspace/venv/`, bootstrap: `bash /workspace/start.sh`
- See INSTALLATION.md for full reinstall instructions.

## Best Models

| Model | Val Comp | Notes |
|-------|----------|-------|
| **swa_70pre_30topo_ep5** | **0.5549** (24-vol cross-scroll) | 70% pretrained + 30% frozen_boundary_ep5. Current best. |
| pretrained (comboloss) | 0.5526 (24-vol) | Baseline. All fine-tuned models degrade SDice vs this. |
| SegResNet v9 | 0.570 (ds=4) / ~0.41 (ds=1) | Legacy. Not competitive at full resolution. |

Weights: `checkpoints/swa_topo/swa_70pre_30topo_ep5.weights.h5`

## Competition Scores
| Run | Val Comp | Public Score | Notes |
|-----|----------|-------------|-------|
| Run 9 (SegResNet) | 0.570 (ds=4) | 0.398 | v15: fixed thresholds |
| TransUNet pretrained | 0.5526 (ds=1) | **0.504** | v20: dual-stream + 7-TTA |
| v22 (SWA best) | 0.5549 (ds=1) | PENDING | Manually submitted Feb 22 |

## Key Insights
1. **Dice and comp_score are uncorrelated.** Don't use dice for model selection.
2. **Public leaderboard is unreliable.** Only 1 test volume. Trust local validation.
3. **Cross-scroll evaluation essential.** Single-scroll validation risks overfitting.
4. **Training and eval normalization MUST match.** /255 vs z-score mismatch caused all early
   fine-tuning runs to appear catastrophically degraded.
5. **Val loss can decrease while comp score collapses** if train/eval normalization differs.
6. **No fine-tuned model beats pretrained alone.** All degrade SDice. SWA blending (70/30
   pretrained + fine-tuned) is the only approach that improves on pretrained.
7. **Frozen encoder > discriminative LR** for preserving pretrained features while fine-tuning.
8. **Predictions are 3-5x too thick.** Model-level issue, not PP. Ridge thinning destroys topo.
   Training must learn thinness; PP should reconnect fragments.
9. **T_low is the only meaningful PP parameter.** Optimal value is model-dependent (pretrained=0.70,
   dist_sq=0.30-0.40). Must re-sweep after any model change.

## Current Status (Feb 22)

### Pseudo-labeling pipeline (active)

Uses best model's high-confidence predictions to convert unlabeled voxels (label=2,
~52% of each volume) into training signal. 80.4% of unlabeled voxels converted at
0.85/0.15 thresholds, FG nearly doubles.

**Stages 1-2 COMPLETE:** 704 probmaps (42 GB) + 704 pseudo-labels (21 GB) generated.

**Stage 3 (training) — two parallel experiments:**

| GPU | Run | clDice | Status | ETA |
|-----|-----|--------|--------|-----|
| gpu0 | pseudo_frozen_boundary | OFF (OOM on 32GB) | Epoch 1/25, 704 steps/ep | ~26 hrs |
| gpu2 | pseudo_frozen_cldice | ON (iters=10, 48GB GPU) | Setting up | ~26 hrs after setup |

Both use: frozen encoder, lr=5e-5, skel=0.75, fp=1.5, dist=2.0, power=2.0, boundary=0.3.

Scripts: `scripts/generate_probmaps.py`, `scripts/generate_pseudo_labels.py`,
`scripts/run_pseudo_pipeline.sh`, `scripts/run_pseudo_stage3_4.sh`
Training: `--label-dir data/pseudo_labels` and `--cldice-weight` flags added to `train_transunet.py`
Log: `logs/pseudo_pipeline.log`

### Kaggle v22 (submitted Feb 22)

Updated inference to use SWA best weights (swa_70pre_30topo_ep5, local val 0.5549).
Manually submitted — score pending (takes several hours).
Notebook: https://www.kaggle.com/code/mgoldfield/vesuvius-surface-detection-inference

### Disk usage (gpu0)

~170 GB / 350 GB used (probmaps 42 GB + pseudo-labels 21 GB).

## SWA Weight Averaging — current best approach

Blending pretrained + fine-tuned weights at 70/30 ratio is the only approach that beats pretrained.
This is our primary model improvement strategy. After pseudo-label training completes, we should
SWA blend the best pseudo-label checkpoint with pretrained (same 70/30 recipe).

**Topo-focused blends (frozen_boundary source):**

| Model | Comp | Topo | SDice | VOI | n |
|-------|------|------|-------|-----|---|
| pretrained (baseline) | 0.5526 | 0.2354 | 0.8255 | 0.5517 | 24 |
| swa_90pre_10topo (ep10) | 0.5544 | 0.2403 | 0.8285 | 0.5496 | 24 |
| swa_70pre_30topo (ep10) | 0.5545 | 0.2490 | 0.8301 | 0.5407 | 24 |
| **swa_70pre_30topo_ep5** | **0.5549** | 0.2499 | 0.8291 | 0.5420 | 24 |
| swa_70pre_30sdice_ep15 | 0.5548 | 0.2489 | 0.8301 | 0.5418 | 24 |

**Key findings:**
- 70/30 ratio is the sweet spot (consistent across dist_sq and frozen_boundary blends)
- Topo improvement is substantial (+0.014) with SDice maintained or improved
- ep5 slightly outperforms ep10 in the blend despite ep10 having better individual topo
- Pure fine-tuned weights are much worse — pretrained carries most value

Results: `logs/eval_swa_topo_results.csv`, `logs/eval_swa_results.csv`
Weights: `checkpoints/swa_topo/`
Script: `scripts/swa_blend.py`

## Fine-Tuning Experiments (completed, inform future work)

All fine-tuned models degrade SDice vs pretrained. Frozen encoder consistently better than
discriminative LR. These results inform pseudo-label training strategy (frozen encoder, boundary loss).

| Model | Best Comp | Best Topo | Best SDice | Strategy |
|-------|-----------|-----------|------------|----------|
| frozen_boundary (gpu2) | 0.5408 (ep10) | **0.2642** (ep10) | 0.7871 (ep15) | Frozen encoder |
| frozen_dist_sq (gpu2) | 0.5402 (ep25) | 0.2634 (ep25) | 0.7885 (ep10) | Frozen encoder |
| discrim_boundary (gpu1) | 0.5286 (ep15) | 0.2342 (ep15) | 0.7836 (ep15) | Discriminative LR |
| discrim_dist_sq (gpu1) | 0.5269 (ep25) | 0.2292 (ep15/25) | 0.7841 (ep25) | Discriminative LR |

Results: `logs/eval_frozen_boundary_results.csv`, `logs/eval_discrim_boundary_results.csv`, etc.

## PP Sweep Findings (inform future PP tuning)

26 configs on pretrained probmaps (82 vols) + 26 configs on dist_sq probmaps.

**Key findings:**
- **T_low is the only meaningful PP parameter.** Closing, dust removal, confidence filtering = noise (±0.001).
- **Optimal T_low depends on the model.** Pretrained optimal T_low=0.70, dist_sq optimal T_low=0.30-0.40.
  Thinner predictions need lower threshold to preserve connectivity.
- **PP barely helps fine-tuned models.** Best fine-tuned PP config = 0.506 vs pretrained's 0.553. Gap is in the model.
- **After pseudo-label training, re-sweep T_low** on the new model — optimal value will likely shift.

Results: `logs/postprocessing_sweep.csv`, `logs/sweep_pp_dist_sq_results.csv`

## Prediction Thickness (core problem, ongoing)

**Problem:** Model predicts 15-30% foreground per volume vs GT's 2-8%. Surfaces are 3-5x too thick.
Confirmed from exploration notebook: probmaps themselves are thick (model-level, not PP artifact).

**Impact on metrics:**
- **SDice (35%):** Thick slabs create two boundary surfaces; one aligns with GT, other is penalized.
- **VOI (35%):** Excess voxels increase conditional entropy; thickness merges nearby surfaces.
- **Topo (30%):** Merged surfaces change component count and create false tunnels.

**What we've tried:**
- dist_sq loss (quadratic penalty far from skeleton) — partial improvement, best topo=0.2642
- Frozen encoder + boundary loss — best at preserving topo while thinning
- SWA blending — 30% fine-tuned dose thins slightly without destroying SDice
- Ridge thinning PP — **destroys topology** (topo 0.29→0.005). PP can't thin safely.
- clDice — needs 48GB VRAM. Testing on gpu2 with pseudo-labels.

**Remaining approaches:**
- **Pseudo-labeling** (active) — expanded training signal may help model learn sharper boundaries
- **clDice + pseudo-labels** (active, gpu2) — soft-skeletonization loss directly measures thin alignment
- **Higher dist_sq power** (power=3 or 4) — more aggressive thinning of outer voxels
- **Connectivity PP** — instead of thinning, reconnect fragments after aggressive thresholding:
  probmap-guided gap filling, dilate-merge-erode, two-pass hysteresis

Visual analysis: `notebooks/analysis/multi_model_comparison.ipynb`

## Strategy (Feb 22 → Feb 27)

### Phase 1: Pseudo-labeling (Feb 22-24, in progress)
- Two parallel training experiments (gpu0 no clDice, gpu2 with clDice)
- Eval sweep of all checkpoints after training
- SWA blend best pseudo-label checkpoint with pretrained (if it improves)

### Phase 2: Tuning and hardening (Feb 24-27)
- PP parameter tuning with best model
- Adaptive TTA timer (safety-critical for 9hr Kaggle limit on 120 volumes)
- Kaggle notebook hardening (memory, timing, error handling)
- Ensemble best checkpoints if time allows
- Final submissions

### Ideas on deck (if time allows)
- [ ] **Component-level refinement** — Novel learned post-processing targeting VOI
- [ ] **Multi-scale inference fusion** — 128^3 + 160^3 averaged
- [ ] **Ensemble** — Fine-tuned + pretrained TransUNet logit averaging
- [ ] **Connectivity PP** — Probmap-guided gap filling, dilate-merge-erode

## Kaggle Submission Quick Reference
```bash
# Upload weights
kaggle datasets version -p kaggle/kaggle_weights/ -m "message" --dir-mode zip
# Push inference notebook
kaggle kernels push -p kaggle/kaggle_notebook/
# Check status
kaggle kernels status mgoldfield/vesuvius-surface-detection-inference
# Check score
kaggle competitions submissions -c vesuvius-challenge-surface-detection
```
**Gotchas:** No internet on Kaggle. Use `--dir-mode zip` for subdirectories.
Uploads hang at 0% for ~8 min (normal). Delete stale tokens: `rm /tmp/.kaggle/uploads/*.json`.

## Kaggle Notebook Versions
| Version | Weights | Date | Status | Notes |
|---------|---------|------|--------|-------|
| v15 | v9 traced | Feb 15 | 0.398 | Fixed thresholds |
| v20 | TransUNet comboloss | Feb 17 | **0.504** | Dual-stream + 7-fold TTA + seeded hysteresis |
| v21 | TransUNet comboloss | Feb 18 | 0.504 | T_low=0.70 (same — public test is 1 volume) |
| v22 | SWA best | Feb 22 | PENDING | swa_70pre_30topo_ep5 (local val 0.5549) |

## File Structure
```
/workspace/vesuvius-kaggle-competition/
├── notebooks/                     # Training notebooks (v1-v13 + refinement)
│   └── analysis/                  #   Multi-model comparison, exploration
├── data/                          # Competition data (not in git)
│   ├── train_images/              #   786 .tif volumes
│   ├── train_labels/              #   786 .tif labels
│   ├── pseudo_labels/             #   704 pseudo-labeled .tif files
│   └── train.csv, test.csv
├── pretrained_weights/transunet/  # TransUNet SEResNeXt50 weights from Kaggle
├── checkpoints/                   # Model checkpoints
│   ├── swa_topo/                  #   SWA blends (current best)
│   └── transunet_pseudo_*/        #   Pseudo-label training checkpoints
├── kaggle/                        # Submission artifacts
│   ├── kaggle_notebook/           #   Inference script + kernel metadata
│   └── kaggle_weights_download/   #   Weight datasets for upload
├── scripts/                       # Training, eval, pipeline scripts
├── libs/topological-metrics-kaggle/  # topometrics library
├── logs/                          # Pipeline logs + eval CSVs
├── NOTES.md                       # This file (active)
├── HISTORY.md                     # Run history & blog source
├── INSTALLATION.md                # Dependency reinstall guide
├── TRANSUNET_SETUP.md             # TransUNet installation guide
├── COMPETITOR_ANALYSIS.md         # Competitor notebook analysis
└── CLAUDE.md                      # Claude Code instructions
```
