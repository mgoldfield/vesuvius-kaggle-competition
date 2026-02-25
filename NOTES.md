# Vesuvius Surface Detection — Project Notes

Active working document. Historical run details and design rationale are in **HISTORY.md**
(blog source material). **Move completed work to HISTORY.md** when it's no longer needed
for active decision-making — keep this file lean.

## Competition
- **Goal:** Detect papyrus surfaces in 3D CT scans of Herculaneum scrolls
- **Metric:** `0.30*TopoScore + 0.35*SurfaceDice@tau=2 + 0.35*VOI_score`
- **Deadline:** Feb 27, 2026 (2 days remaining)
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
| **gpu0** (this machine) | RTX 5090 32GB | local | Primary control. Probmap generation + PP sweep. |
| **gpu1** (new) | RTX 6000 Ada 48GB | `ssh -i ~/.ssh/remote-gpu root@195.26.233.34 -p 39479` | ViT high-LR unfreeze (9/15 ep done, idle after restart). |
| **gpu2** | RTX 6000 Ada 48GB | `ssh -i ~/.ssh/remote-gpu root@195.26.233.87 -p 25763` | Round-2 pseudo-labeling pipeline. Restarted Feb 23. |
| **gpu3** (new) | RTX 6000 Ada 48GB | `ssh -i ~/.ssh/remote-gpu root@195.26.233.74 -p 32327` | ViT balanced done (7/7 ep). IDLE. |
| **gpu4** (new) | RTX 6000 Ada 48GB | `ssh -i ~/.ssh/remote-gpu root@195.26.233.43 -p 53200` | Decoder balanced exp 2. Restarted Feb 23. |
| **data-gpu** (new) | ? | `ssh -i ~/.ssh/remote-gpu root@195.26.233.98 -p 31182` | External data download + pseudo-labeling. |

- Old gpu1 (RTX 5090): **decommissioned** Feb 22. All data synced to gpu0.
- Venv: `/workspace/venv/`, bootstrap: `bash /workspace/start.sh`
- See INSTALLATION.md for full reinstall instructions.

## Best Models

| Model | Val Comp | Notes |
|-------|----------|-------|
| **swa_70pre_30margin_dist_ep5** | **0.5551** (24-vol cross-scroll) | 70% pretrained + 30% margin_dist_ep5. **NEW BEST.** |
| swa_70pre_30topo_ep5 | 0.5549 (24-vol cross-scroll) | 70% pretrained + 30% frozen_boundary_ep5. Previous best. |
| pretrained (comboloss) | 0.5526 (24-vol) | Baseline. All fine-tuned models degrade SDice vs this. |
| SegResNet v9 | 0.570 (ds=4) / ~0.41 (ds=1) | Legacy. Not competitive at full resolution. |

Weights: `checkpoints/swa_topo/swa_70pre_30margin_dist_ep5.weights.h5`

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

## What We Know Works (high confidence)

1. **Frozen encoder training** — consistently outperforms discriminative LR for preserving pretrained features
2. **SWA 70/30 blending** — every fine-tuned model improves when blended with pretrained at 70/30
3. **Pseudo-labels + clDice** — strong SDice gains (0.8304, our best). Extra training data is extremely beneficial.
4. **Lower T_low + close_erode PP** — potentially huge topo gains (0.5595 comp on 2-vol dry-run, needs confirmation)
5. **close_erode PP is model-independent** — if confirmed, it benefits ANY model. Model improvement and PP improvement are multiplicative.

## Tunable Dimensions (what we can still optimize)

### Training / Loss
- **Margin distance parameter:** margin=2 (gpu1) vs margin=3 (gpu2). Tighter margin = thinner surfaces. Could push to margin=1.
- **clDice weight:** 0.3 (gpu1) vs 0.5 (gpu2 clDice, gpu2 round-2). Higher = more emphasis on thin structure alignment.
- **clDice iterations:** 5 (current, reduced from 10 for memory). Higher = finer skeletonization but more VRAM.
- **Boundary weight:** 0.3 (current). Could try 0.5-1.0 for more edge-squeezing.
- **Loss combinations:** clDice + margin_dist + boundary (gpu1 testing). Could also try clDice alone, margin_dist alone, different ratios.
- **Epoch selection for SWA:** ep5 often beats ep10/15/20 in blends despite worse individual scores. Early stopping matters for blend quality.

### Selective Component Unfreezing — NEW IDEA

Currently `--freeze-encoder` freezes stem + CNN stages + ViT (~48.5M), trains only decoder +
upsampler (~6M). But different model components control different failure modes:

| Component | Params | Controls | Best loss to target it |
|-----------|--------|----------|----------------------|
| Stem + CNN stages | ~23.5M | Local texture/edge features | **Keep frozen** — pretrained excels here |
| ViT (12 layers) | ~25M | Global context at 5^3 | Connectivity/topo losses (clDice) — ONLY place distant surface fragments communicate |
| Query decoder | ~4M | Region assignment, which voxels each query claims | Thinning losses (margin dist, boundary) |
| U-Net upsampler + head | ~2M | Final spatial sharpness | Boundary/sharpness losses |

**Proposed experiments (can run on new GPUs tonight, doesn't interfere with current pipeline):**

1. **Unfreeze ViT only + clDice** (48GB GPU) — teach transformer to preserve surface connectivity.
   The 12 attention layers can learn to propagate surface-membership signals across the whole 5^3
   volume. This targets our topo weakness at the source. ~15 min/epoch, 15 epochs = ~4 hrs.

2. **Unfreeze decoder+queries only + margin dist (margin=1)** (32GB GPU) — aggressive thinning
   on just the decoder. 100 queries learn to claim thinner regions. Only ~4M trainable params,
   very fast. ~10-12 min/epoch, 15 epochs = ~3 hrs.

**Why this could be big:** Current frozen training trains ~6M params to fix problems that
originate in the 25M-param ViT (connectivity) and 4M-param decoder (thickness). By selectively
unfreezing the RIGHT component with the RIGHT loss, we target each failure mode directly.

**Code change needed:** Add `--unfreeze` flag to `train_transunet.py` that accepts component
names (vit, decoder, queries, head) instead of current all-or-nothing `--freeze-encoder`.

**GPU requirements:**
- ViT unfreezing + clDice: 48GB (clDice is memory-hungry)
- Decoder-only unfreezing without clDice: 32GB is fine
- Both: very fast training since CNN encoder stays frozen (no gradients through 23.5M params)

### SWA Blending
- **Blend ratio:** 70/30 is current best. Could try 60/40 or 80/20 with stronger fine-tuned models.
- **Which fine-tuned model to blend:** margin_dist_ep5 currently best (0.5551). Round-2 and gpu1 results pending.
- **Multi-model blending:** Could blend pretrained + multiple fine-tuned models (e.g., 60% pretrained + 20% clDice + 20% margin_dist).

### Post-Processing
- **T_low:** Currently 0.40 in close_erode config. Sweep testing 0.30-0.70.
- **Closing iterations:** 1 (face-connected) currently. Could try 2, or larger structuring elements.
- **Erosion iterations:** 1 currently. Could try 2, or skip for different T_low values.
- **Structuring element:** face-connected (6-neighbor) vs full (26-neighbor). Face-connected used in sweep.
- **Dust removal threshold:** 100 voxels currently. Could tune up or down.
- **Method combinations:** close_erode, erode-only, gap-fill, DME, two-pass hysteresis — all tested in sweep.

### Inference
- **Ensemble:** Average logits from multiple models at inference. Have 4-5 distinct models. Cheap to implement.
- **TTA level:** 7-fold (current), could add more augmentations or weight them differently.
- **Overlap:** 0.42/0.43/0.60 (dual-stream). Higher overlap = better quality but slower.

## Priorities (Feb 24-27, revised)

### Priority 1: Train on expanded pseudo-labeled data (Feb 24-25) — MAIN BET
data-gpu is processing external scroll data from scrollprize.org → pseudo-labels via best model.
Once complete, train on the expanded dataset (competition + external pseudo-labels).
- This is our highest-leverage remaining experiment — more data is the most reliable way to improve.
- After training, SWA blend with pretrained (70/30) and re-sweep T_low on the new model.

### Priority 2: Ensemble at inference (Feb 25-26)
- Average logits from 2-3 diverse models (e.g., pretrained + margin_dist blend + cldice blend).
- Orthogonal to training improvements, cheap to implement, typically +0.01-0.02.
- Can be built in parallel with Priority 1.

### Priority 3: Train on all 786 volumes (Feb 25-26) — FINAL SUBMISSION
- Final model: train on ALL data (including 82 val volumes) with best config.
- ~10% more data + no val holdout penalty. Can't validate locally after this.
- Must finalize hyperparams and PP config first.

### Priority 4: Kaggle hardening + final submissions (Feb 26-27)
- Kaggle notebook hardening (memory, timing, error handling)
- Final submissions with buffer for Kaggle queue (deadline Feb 27)

### Completed / abandoned priorities
- ~~Confirm close_erode PP~~ — DONE. erode_tl0.40_e1 best (+0.001-0.002 comp). Marginal.
- ~~Selective component unfreezing~~ — DEAD END. ViT unfreeze blends at 0.5534, decoder at 0.5498.
  Aggressive losses (dist_weight=1.0) don't learn at all. Pretrained encoder is hard to improve.
- ~~Multi-model SWA~~ — DEAD END for comp. Pushes SDice but doesn't beat single 70/30 blend.
- ~~Iterative pseudo-labeling (round 2 on gpu2)~~ — gpu2 abandoned (disk full).

## Current Status (Feb 25, ~11:40 AM EST / 16:40 UTC)

### FINAL PUSH — SUBMISSIONS RUNNING

Best local comp: **0.5551** (swa_70pre_30margin_dist_ep5, 24-vol cross-scroll).
Public LB: 0.504 (all submissions score the same — only 1 public test volume).

**Train-all completed:** 15/15 epochs on all 786 volumes, 6.1 hours. Loss: 0.9637→0.9630.
Checkpoints: ep5, ep10, ep15 at `checkpoints/transunet_all_data_frozen/`.
Cannot validate locally (trained on all data including val holdout).

**SWA blend created:** 70% pretrained + 30% train-all ep5 → `swa_70pre_30all_data_ep5.weights.h5`

| Submission | Version | Config | Status |
|------------|---------|--------|--------|
| **v25** (ensemble) | Kaggle v25 | 2-model: margin_dist + all_data blends, averaged logits | **RUNNING** |
| **v26** (safety net) | Kaggle v26 | Single model: margin_dist only (validated best 0.5551) | **RUNNING** |

**Kaggle weights dataset:** Cleaned up to 543MB (2 SWA blends + wheels only). Old SegResNet and base pretrained weights removed.

**Kaggle API auth:** Old key expired. New token format: `export KAGGLE_API_TOKEN=KGAT_...` (env var, not kaggle.json).

### What's left
- Wait for v25 + v26 scores (few hours each)
- If time allows, can try additional variants (different blend ratios, ep10/ep15 blends)
- Deadline: Feb 27

### External data training results

| Epoch | Comp (standalone val-only) | Comp (standalone cross-scroll) | SWA 70/30 cross-scroll | Notes |
|-------|---------------------------|-------------------------------|----------------------|-------|
| ep4 | 0.5285 | 0.5530 | OOM'd (~0.537 on 2 vols) | Won't beat ep10's 0.5536 |
| ep7 | 0.5265 | — | — | |
| ep10 | 0.5277 | 0.5539 | **0.5536** | Below best (0.5551) |
| ep13 | 0.5264 | — | — | |
| ep15 | 0.5266 | — | — | |

**External data blending verdict:** SWA 70/30 of external_data_ep10 gives 0.5536 — below current best (0.5551).
The external data model didn't improve enough over pretrained to beat the margin_dist blend.
**Current best remains: swa_70pre_30margin_dist_ep5 at 0.5551.**

### GPU Fleet Status

| GPU | Status | Task |
|-----|--------|------|
| gpu0 | **ACTIVE** | SWA blending + eval, then train-all |
| gpu1 | SHUT DOWN | All checkpoints pulled |
| gpu2 | ABANDONED | Disk full |
| gpu3 | SHUT DOWN | All checkpoints pulled |
| gpu4 | SHUT DOWN | All checkpoints pulled |
| data-gpu | SHUT DOWN | External data + pseudo-labels complete |

### Completed experiments (details in HISTORY.md)
- Margin distance training, clDice pseudo-label training, SWA connectivity PP sweep
- All SWA blend evaluations (Feb 23) — best: swa_70pre_30margin_dist_ep5 (0.5551)
- Selective unfreezing (DEAD END), Multi-model SWA (DEAD END for comp)
- T_low PP sweeps (erode_tl0.40_e1 best, +0.001-0.002)
- gpu2 pseudo_frozen_margin_dist eval
- Pseudo-labeling pipeline (704 volumes)
- Fine-tuning experiments (frozen encoder > discriminative LR)
- PP sweep findings, connectivity PP (disappointing)

## Strategy — FINAL PUSH (Feb 25-27)

### Step 1: SWA Blend External Data ep4+ep10 (~30 min) — IN PROGRESS
Create 70/30 blend of pretrained + external_data ep10 (and ep4), eval cross-scroll.
External ep10 standalone was 0.5539, ep4 was 0.5530 — SWA blend should push higher.

### Step 2: Build Ensemble Inference (~2 hours coding + eval) — PENDING
Modify Kaggle notebook to support 2-model ensemble (average logits before thresholding).
Timing: with 2 models + adaptive TTA, should still fit in 9hr Kaggle limit.
Test locally with eval_transunet.py modified to average logits from 2 models.

### Step 3: Train on All 786 Volumes (~5 hours) — PENDING
Train with NO val holdout — all 786 competition volumes. Best config:
- Frozen encoder, SWA init (swa_70pre_30margin_dist_ep5)
- dist_weight=0.1, dist_margin=3, boundary=0.3, skel=0.75, fp=0.5
- 15 epochs, LR=5e-5, grad-accum=4
- New `--train-all` flag skips val split
After training: SWA blend best epoch with pretrained at 70/30.
**Cannot validate locally after this — flying blind.**

### Step 4: Upload + Submit to Kaggle (~2-3 hours) — PENDING
1. Copy final weights to `kaggle/kaggle_weights_download/`
2. Upload dataset, push notebook, monitor
3. Submit multiple variants if time allows (single model vs ensemble)

### Completed phases (prior sessions)
- [x] Phase 1: All training + eval experiments complete
- [x] Phase 2: External data pseudo-labeling + training (15 epochs)
- [x] T_low PP sweeps — erode_tl0.40_e1 best (+0.001-0.002)
- [x] Selective unfreezing — DEAD END
- [x] Multi-model SWA — DEAD END for comp
- [x] Adaptive TTA timer (v23)

### External data from scrollprize.org — DONE
External scroll data downloaded and processed into pseudo-labeled training volumes.
Training completed (15 epochs). See external data training results above.

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
