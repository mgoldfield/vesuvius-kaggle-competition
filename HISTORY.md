# Vesuvius Surface Detection — Project History

Full narrative of iterations, decisions, and findings. Source material for a blog post
after the competition. See **NOTES.md** for the active working document.

## Blog Post Plan
Planning to write a blog post about this project after the competition. These notes
capture the full story of iterations — what we tried, why, what worked, what didn't, and
the key insights along the way (e.g. the inference pipeline discovery, the lr_find struggles,
the disconnect between val metrics and public scores).

---

## Training Run Details (SegResNet era, Feb 5-16)

### Run 1: Baseline 3D U-Net
- **Architecture:** 3D U-Net with BatchNorm, 22.6M params
- **Channels:** 1 → 32 → 64 → 128 → 256 → 512 (bottleneck)
- **Patch size:** 128^3, batch_size=2, mixed precision (fp16)
- **LR:** 1.9e-2 (from lr_find), one-cycle policy, 30 epochs
- **Loss:** 0.5*MaskedBCE + 0.5*MaskedDice
- **Val set:** scroll 26002 (82 volumes)
- **Results:** Best dice **0.121**. Public score **0.331**.
- **Notebook:** `notebooks/vesuvius_train.ipynb`

### Run 2: GroupNorm + Dropout
- Switch BatchNorm → GroupNorm (better for small batch sizes)
- Add dropout at bottleneck (0.2)
- lr_find suggested 3.63e-3 (lower than Run 1's 1.9e-2)
- Everything else same as Run 1
- **Results:** Best dice **0.172** at epoch 9 (vs Run 1's 0.121 — 42% improvement)
- Dice unstable across epochs (bounces 0.05–0.17), loss plateau around 0.55
- Public score **0.290** — worse than Run 1 despite better dice!

### Run 3: SuPreM SegResNet + clDice
- **Architecture:** MONAI SegResNet, 4.7M params (vs 22.6M UNet)
- **Pre-trained:** SuPreM supervised weights (2,100 CT volumes, 25 organ classes)
- **Channels:** 16 → 32 → 64 → 128, residual blocks, GroupNorm
- **Weights:** 79/83 params loaded from checkpoint, final conv randomly init
- **Loss:** 0.3*BCE + 0.3*Dice + 0.4*clDice (all masked for label=2)
  - clDice uses soft skeletons via iterative morphological erosion (5 iters)
  - Skeleton computed on 2x-downsampled patches (64^3) to save VRAM
  - Target skeleton computed with `torch.no_grad()` — only pred needs gradients
- **LR:** lr_find suggested 1.74e-3, trained with 1e-3
- **Results:** Best dice **0.281** at epoch 8 (63% improvement over Run 2).
  Public score **0.348**.
- **Checkpoint:** 18 MB (vs 87 MB UNet)
- **Notebook:** `notebooks/vesuvius_train_v3.ipynb`

### Run 4: Boundary Loss + Post-Processing
- **Loss:** 0.2*BCE + 0.2*Dice + 0.3*clDice + 0.3*Boundary
  - Boundary loss uses signed distance transform from GT surface
  - Directly targets SurfaceDice@τ=2 (35% of competition score)
- **Results:** Best dice **0.232** — regressed from Run 3's 0.281
- **Notebook:** `notebooks/vesuvius_train_v4.ipynb`

### Run 5: Competition Metric Monitoring
- Added exact competition metric evaluation each epoch via `topometrics` library
- `SaveModelCallback` monitors `comp_score` instead of `dice`
- **Notebook:** `notebooks/vesuvius_train_v5.ipynb`

### Run 6: Discriminative LR + Fast Metric
- 3 param groups via `segresnet_splitter`: encoder (LR/100), decoder (LR/10), head (LR)
- Downsampled competition metric: 4x downsample (320^3 → 80^3) for fast Betti matching
- **Confirmed:** dice and comp_score are uncorrelated — dropped dice as a metric
- **Notebook:** `notebooks/vesuvius_train_v6.ipynb`

### Run 7: TTA + Hysteresis + 160^3 Patches
- 160^3 patches (was 128^3), stride=80
- TTA (7-fold) at inference: original + 3 axis flips + 3 HW-plane rotations
- Hysteresis thresholding: strong seeds (prob >= T_HIGH), propagate into weak region (prob >= T_LOW)
- Anisotropic closing: z-heavy structuring element for layered papyrus
- Dust removal: remove connected components < 100 voxels
- **Notebook:** `notebooks/vesuvius_train_v7.ipynb`

### Run 8: Foreground-Biased Patch Sampling
- FG_BIAS=0.5: 50% of crops centered on random foreground voxel
- **LR:** lr_find suggested 5.75e-4
- **Results:** Best comp_score **0.562** at epoch 1 — peaked immediately (LR too high)
- Public score **0.423**
- **Issue:** lr_find auto-selected 5.75e-4 but the usable LR is ~1e-5.
- **Notebook:** `notebooks/vesuvius_train_v8.ipynb`

### Run 9: Lower Learning Rate (best SegResNet model)
- **LR:** Hardcoded 1e-5. Discriminative: encoder=1e-7, decoder=1e-6, head=1e-5
- Everything else same as Run 8
- **Results:** Best comp_score **0.570** at epoch 8 (vs Run 8's 0.562 at epoch 1)
  - Loss steadily decreased: train 0.523→0.480, valid 0.543→0.474
  - Threshold sweep: T_low=0.40 won (dice=0.278)
  - Verification comp_score (3 volumes): 0.573, 0.546, 0.601 → mean **0.573**
- **Notebook:** `notebooks/vesuvius_train_v9.ipynb`

### Run 10: Deep Supervision + Attention Gates
- `SegResNetDSAttn` — subclasses SegResNet with attention gates on skip connections
  + auxiliary heads at intermediate decoder levels
- **Results:** Best comp_score **0.584** at epoch 0 — peaked immediately, never improved
  - **Issue:** New modules (attn gates, DS heads) are randomly initialized but shared the
    same low LR (1e-5). They needed higher LR to learn. Pretrained modules dominated.
- **Notebook:** `notebooks/vesuvius_train_v10.ipynb`

### Run 10b-v1: Per-Module lr_find Valley
- 4 parameter groups with independent lr_find for each
- Valley LRs: encoder=9.12e-6, decoder=9.12e-5, new_modules=2.40e-3, head=5.50e-3
- **Results:** Best comp_score **0.567** at epoch 1 — peaked early again
  - **Issue:** `lr_find().valley` consistently picks ~100x too aggressively.
- **Notebook:** `notebooks/vesuvius_train_v10b.ipynb`

### Run 10b-v2: Hardcoded LRs from lr_find Plots
- Hardcoded LRs ~100x lower than valley: encoder=1e-7, decoder=1e-6, new_modules=1e-5, head=1e-4
- **Results:** Best comp_score **0.581** at epoch 2. Loss monotonically decreased all 30 epochs.
  Comp_score improved epochs 0-2, dipped during one_cycle warmup, then **started recovering
  after epoch 11** as LR annealed down. Recovery never captured because SaveModelCallback
  only saved epoch 2.
- **LR Schedule Insight (critical):** `fit_one_cycle` warmup is harmful for pretrained models
  with new randomly-init modules. The productive learning happened during annealing (epochs 11-30).
  Fix: use `fit_flat_cos` (no warmup). Train longer (50+ epochs). Save periodic checkpoints.
- **Notebook:** `notebooks/vesuvius_train_v10b.ipynb`

### Run 11: 3-Fold Cross-Validation Ensemble
- 3-fold scroll-grouped CV with probability averaging ensemble
- **Results:** Completed but unreliable — used same SegResNetDSAttn + LR as v10, so all folds
  suffer from the same LR issue. The technique (CV + ensembling) is unverified.
- **Notebook:** `notebooks/vesuvius_train_v11.ipynb`

---

## Inference Pipeline Evaluation (Feb 12)

Ran `eval_inference.py` with v9 checkpoint. Results (5 val volumes, no surface splitting):

| Volume | Old (uniform SWI + prob TTA) | New (Gaussian SWI + logit TTA) | Delta |
|--------|------------------------------|-------------------------------|-------|
| 26894125 | 0.271 | 0.573 | +0.302 |
| 105796630 | 0.548 | 0.548 | +0.001 |
| 327851248 | 0.241 | 0.241 | +0.000 |
| 418613908 | 0.241 | 0.576 | +0.335 |
| 477109023 | 0.235 | 0.552 | +0.317 |
| **Mean** | **0.307** | **0.498** | **+0.191** |

Gaussian SWI + logit TTA gives **+0.19 comp_score** with the same weights (no retraining).
Surface splitting was impractical (~80 min/volume for Dijkstra3D).

**Threshold sweep (new pipeline):**
| T_low | T_high | comp_score |
|-------|--------|-----------|
| 0.35 | 0.80 | **0.570** |
| 0.30 | 0.80 | 0.568 |
| 0.40 | 0.80 | 0.568 |
| 0.45 | 0.80 | 0.566 |
| 0.35 | 0.85 | 0.503 |
| 0.30 | 0.90 | 0.235 |

Best: **T_low=0.35, T_high=0.80 → 0.570**. T_high=0.85 was significantly worse.

---

## Overnight Pipeline (Feb 12)

The overnight automation script (`scripts/overnight_pipeline.sh`) ran:
1. Wait for v10 training to complete — done ~02:12
2. Trace v10 model → upload weights → push Kaggle inference (version 9) — done ~02:32
3. Run v11 training (3-fold CV via nbconvert) — done ~03:31

**Issues discovered:**
- V10 submission used bad weights (peaked at epoch 0 due to LR issue)
- Eval inference never ran (script added after pipeline started)
- Wheels not uploaded (missing `--dir-mode zip`)
- V11 results unreliable (same LR issue as v10)

---

## Kaggle Threshold Bug (discovered Feb 15)

The v13 submission scored 0.200 despite v9 having 0.570 local comp_score. Root cause:
the Kaggle inference notebook had **T_HIGH=0.85** (too aggressive for v9's output range)
and **USE_SURFACE_SPLIT=True**. These caused **0% foreground** in predictions.
Fixed in v15: T_LOW=0.35, T_HIGH=0.80, splitting OFF → fg=28.0%.

---

## Design Decisions & Rationale

### Validation split: scroll-level holdout vs random %
We split by scroll_id rather than randomly. Volumes from the same scroll share similar
texture/density, so random splits would inflate validation scores. Holding out entire
scroll 26002 (82 volumes, ~10%) gives honest generalization signal.

### GroupNorm over BatchNorm
BatchNorm is poor at batch_size=2 — statistics are too noisy. GroupNorm (8 groups) is
batch-size independent and well-established for 3D medical segmentation.

### Loss function vs competition metric gap
Training loss (BCE + Dice) optimizes voxel overlap. Competition metric rewards topology
(TopoScore), surface accuracy (SurfaceDice@τ=2), and instance separation (VOI). Added
clDice (Run 3) and Boundary loss (Run 4) to close the gap. Post-processing pipeline
(Run 7) addresses remaining gap.

### Model selection is broken
Training loop uses simplified inference (no TTA, basic threshold). Full pipeline (Gaussian
SWI + logit TTA + hysteresis) gives 2x better comp_score. The model that looks best during
training may not be best after full pipeline. Solution: periodic checkpoints + post-training
eval sweep with full pipeline.

---

## VRAM Measurements
| Config | VRAM |
|--------|------|
| Train 128^3 bs=2 (fwd+bwd) | 11.0 GB |
| Train 160^3 bs=2 (fwd+bwd) | 21.4 GB |
| Train 320^3 bs=1 | OOM (needs ~53 GB) |
| Inference 128^3 sliding window | 8.1 GB |

---

## Original UNet3D Architecture (Run 1-2)
4 encoder blocks, bottleneck, 4 decoder blocks with skip connections.
Channel progression: 1 → 32 → 64 → 128 → 256 → 512 (bottleneck). 22.6M parameters.
Random 128^3 patches from 320^3 volumes. Sliding window inference stride 96.

---

## External Data Sources

Competition rules allow freely & publicly available external data including pre-trained models.

| Source | Value | Effort |
|--------|-------|--------|
| Pre-trained 3D encoder (SuPreM) | High | Low — **done in Run 3** |
| nnUNet framework | High | Medium |
| scrollprize.org segments → extra labels | High | High |
| Raw scroll CTs | Medium | Very high |

- **scrollprize.org** has 5+ full scrolls at `dl.ash2txt.org/full-scrolls/` (multi-TB each)
  with OBJ surface meshes that could be rasterized into labels. No competition-format labels.
- **nnUNet preprocessed** (91.8 GB) available on Kaggle, indicates nnUNet is popular among competitors.

---

## Issues Encountered
- **fastprogress 1.1.3 bug:** `NBMasterBar` missing `out` attribute. Fix: pin to 1.0.5.
- **torch downgrade:** pip downgraded torch when installing fastai. No issues observed.
- **Missing volume files:** 20 IDs in train.csv lack .tif files. Code filters these out.
- **Full volume OOM:** 320^3 needs ~53 GB for activations. Solved with 128^3/160^3 patches.
- **clDice OOM:** Soft skeleton builds deep autograd graph. Fixed: 2x downsample + no_grad + fewer iters.
- **Stale Jupyter kernels eating GPU:** Kill stale python processes via nvidia-smi.
- **Kaggle upload hangs:** Stale resume tokens in `/tmp/.kaggle/uploads/*.json`. Delete and retry.
  Uploads sit at 0% for ~8-9 min before starting (normal GCS behavior).
- **Folders skipped without `--dir-mode zip`:** Must pass flag for subdirectories.
- **PCIe instability (home machine):** RTX 5090 PCIe link crashed under sustained transfers.
  Mitigated with compact dtypes, no prefetching, smaller patches.
- **imagecodecs needed:** Training TIFFs use LZW compression; tifffile requires imagecodecs.

---

## TransUNet Pivot (Feb 17-18)

### Setup
- Installed Keras 3 + medicai (from GitHub source — pip version is WRONG)
- Downloaded 3 pretrained weight sets from Kaggle (see `TRANSUNET_SETUP.md`)
- Verified: model loads, forward pass works, produces correct (1,160,160,160,3) output
- Using `KERAS_BACKEND=torch` locally for PyTorch backend

### TensorFlow GPU memory bug
Keras 3 imports TensorFlow even with `KERAS_BACKEND=torch`. TF grabbed ~15 GiB GPU
by default, causing OOM alongside PyTorch. Fix: `tf.config.set_visible_devices([], 'GPU')`
at top of all scripts. Peak VRAM: 16.27 GB (fwd), 21.26 GB (fwd+bwd).

### Metric downsample discovery
METRIC_DOWNSAMPLE=4 inflates local scores by +0.16:

| Downsample | Mean CompScore (5 vol) | Difference |
|-----------|----------------------|-----------|
| **ds=1 (full res, what Kaggle uses)** | **0.4113** | — |
| ds=2 | 0.5039 | +0.093 |
| ds=4 (what we'd been using) | 0.5700 | **+0.159** |

Our "0.57 local val" was actually ~0.41 at full resolution. All future eval uses ds=1.

### Kaggle TransUNet v20 (scored 0.504)
Built dual-stream inference notebook based on Tony Li's 0.552 approach.
**Public LB: 0.504** — +17% over previous best (0.431).

Features: JAX backend, Keras 3, medicai offline wheels, dual-stream inference
(public stream overlap=0.42 for argmax labels, private stream overlap=0.43/0.60),
7-fold TTA, binary logit (logsumexp(L1,L2)-L0), seeded hysteresis, anisotropic closing.

### Overnight 6-phase pipeline (completed Feb 18)
82-vol validation, TTA, cross-scroll (30 vols), PP sweep (26 configs), exploration notebook.
- Mean composite: 0.510 (82 vols, ds=1)
- Cross-scroll: 0.544 (30 vols across 6 scrolls)
- PP sweep: T_low=0.6 best (+0.0076), T_low is the only meaningful PP parameter
- Results: `logs/overnight_transunet.log`, `logs/postprocessing_sweep.csv`

### Adaptive T_low analysis (Feb 18)
Swept 11 T_low values on 20 cross-scroll volumes.
- Best fixed T_low=0.70 (comp=0.5324), NOT 0.60 from 82-vol val
- Adaptive T_low not worthwhile (correlations too weak, best r=-0.539)
- v21 uses T_low=0.70. Scored 0.504 (same as v20 — public test is 1 volume).

---

## Training Bug Investigation (Feb 19)

### Critical: Normalization mismatch

All previous fine-tuning runs degraded the model from comp=0.535 → 0.264-0.343. Found two bugs:

**Bug 1 — Normalization mismatch (HIGH impact):**
Training used `/255` normalization (range [0,1]), but eval/Kaggle use medicai's
`NormalizeIntensity(nonzero=True)` — z-score normalization (range ~[-4, +7], mean=0).
Head-to-head on baseline_v2: z-score eval comp=0.2828, /255 eval comp=0.4506.

**Bug 2 — Intensity shift effectively zero (HIGH impact):**
Shift of ±0.15 absolute on 0-255 data = ±0.0006 after /255 normalization = no augmentation.
Fixed: ±0.10 in z-score space (matching competitor TPU notebook).

**Fix (applied in baseline_v3):** Training now uses z-score normalization matching eval.

### Pretrained model training chain
The comboloss weights were trained in two stages by different people:
1. **Stage 1 (TPU, Innat):** ImageNet encoder → 200 epochs at 128^3 with z-score + DiceCE + clDice
2. **Stage 2 (GPU, 0537 notebook):** Fine-tuned for 25 epochs at 160^3 with /255 + DiceCE + SkeletonRecall + FP_Volume

Model has seen both normalizations. Eval uses z-score (Stage 1), so training should too.

### Training pipeline comparison vs competitor
| Difference | Risk | Status |
|---|---|---|
| Normalization: /255 vs z-score | **HIGH** | **FIXED** |
| Intensity shift: ~zero vs ±0.10 | **HIGH** | **FIXED** |
| Validation: patch loss vs full-vol SWI | MEDIUM | Mitigated by checkpoint eval sweep |
| FG-biased sampling: 50% vs none | MEDIUM | Keeping |
| Effective batch: 4 vs 8 | MEDIUM | Keeping |
| Loss, LR, optimizer, CutOut, mixed precision | OK | Matches competitor |

### Disk cleanup (Feb 19)
Deleted ~135 GB of broken checkpoints (all trained with wrong normalization):
`data/refinement_data/` (94 GB), `checkpoints/transunet/` (13 GB),
`transunet_thin_fp/` (11 GB), `transunet_thin_dist/` (11 GB),
`transunet_baseline_v2/` (5.3 GB).

---

## TransUNet Fine-Tuning Campaign (Feb 19-22)

### baseline_v3 (Feb 19) — normalization fix verified
Training completed (25 epochs, ~4.5h). No catastrophic degradation — normalization fix worked.
But none beat pretrained (0.5526).

| Model | Comp | Topo | SDice | VOI |
|-------|------|------|-------|-----|
| pretrained | **0.5526** | **0.2353** | **0.8254** | 0.5517 |
| ep5 | 0.5361 | 0.2148 | 0.7945 | 0.5532 |
| ep10 | 0.5269 | 0.1961 | 0.7811 | **0.5562** |
| ep15 | 0.5240 | 0.1901 | 0.7856 | 0.5487 |
| ep20 | 0.5300 | 0.2057 | 0.7915 | 0.5463 |
| ep25 | 0.5296 | 0.2045 | 0.7879 | 0.5499 |

Results: `logs/eval_v3_results.csv`

### Overnight 3-GPU sweep (Feb 19-20)
All use z-score normalization, 25 epochs, checkpoints every 5, auto-eval.

| GPU | Variant | Loss Config |
|-----|---------|-------------|
| gpu0 | dist_sq | skel=0.75, fp=1.5, dist=2.0, power=2.0 |
| gpu1 | thin_fp | skel=0.75, fp=1.5, dist=0.0, power=1.0 |
| gpu2 | dist_skel | skel=0.75, fp=1.5, dist=1.0, power=1.0 |

Plus low-LR (5e-6) follow-ups auto-chained after each.

### Phase 3: Discriminative LR + Frozen Encoder (Feb 20)

| GPU | Experiment | Key Idea |
|-----|-----------|----------|
| gpu1 | Discriminative LR | enc=lr/100, vit=lr/10, dec=lr/10, head=lr |
| gpu2 | Frozen encoder | Freeze SEResNeXt50+ViT, only train decoder+head |

### All completed experiments summary (Feb 20-22)

| Model | Best Comp | Best Topo | Best SDice | Notes |
|-------|-----------|-----------|------------|-------|
| frozen_boundary (gpu2) | 0.5408 (ep10) | **0.2642** (ep10) | 0.7871 (ep15) | Best individual topo |
| frozen_dist_sq (gpu2) | 0.5402 (ep25) | 0.2634 (ep25) | 0.7885 (ep10) | Similar to frozen_boundary |
| discrim_dist_sq (gpu1) | 0.5269 (ep25) | 0.2292 (ep15/25) | 0.7841 (ep25) | Worse than frozen |
| discrim_boundary (gpu1) | 0.5286 (ep15) | 0.2342 (ep15) | 0.7836 (ep15) | Worse than frozen |

**Key finding:** No fine-tuned model beats pretrained alone. All degrade SDice.
Frozen encoder is consistently better than discriminative LR. SWA blending is the bridge.

Results: `logs/eval_frozen_boundary_results.csv`, `logs/eval_frozen_dist_sq_results.csv`,
`logs/eval_discrim_dist_sq_results.csv`, `logs/eval_discrim_boundary_results.csv`

---

## SWA Weight Averaging (Feb 20-21)

### First SWA results (Feb 20)
First approach to beat pretrained! Blending pretrained + fine-tuned weights.

| Model | Comp | Topo | SDice | VOI |
|-------|------|------|-------|-----|
| pretrained (baseline) | 0.5526 | 0.2354 | 0.8255 | 0.5517 |
| **swa_70pre_30distsq_ep5** | **0.5530** | **0.2462** | 0.8289 | 0.5401 |
| swa_70pre_30lowlr_ep5 | 0.5518 | 0.2367 | 0.8296 | 0.5440 |
| swa_50pre_50lowlr_ep5 | 0.5518 | 0.2467 | 0.8272 | 0.5378 |
| swa_lowlr_late_avg | 0.5243 | 0.2276 | 0.7785 | 0.5244 |

Key insight: fine-tuning learns useful signal, but only a small dose (30%) improves
on pretrained without degrading SDice. Pure fine-tuned average is much worse.

Results: `logs/eval_swa_results.csv`, Weights: `checkpoints/swa/`

### Topo-focused blending (Feb 21) — New best model
Blending pretrained with frozen_boundary checkpoints (best individual topo scores).

| Model | Comp | Topo | SDice | VOI |
|-------|------|------|-------|-----|
| pretrained (baseline) | 0.5526 | 0.2354 | 0.8255 | 0.5517 |
| swa_90pre_10topo (ep10) | 0.5544 | 0.2403 | 0.8285 | 0.5496 |
| swa_80pre_20topo (ep10) | 0.5531 | 0.2396 | 0.8300 | 0.5450 |
| swa_70pre_30topo (ep10) | 0.5545 | 0.2490 | 0.8301 | 0.5407 |
| swa_60pre_40topo (ep10) | 0.5541 | 0.2507 | 0.8287 | 0.5395 |
| **swa_70pre_30topo_ep5** | **0.5549** | 0.2499 | 0.8291 | 0.5420 |
| swa_70pre_30sdice_ep15 | 0.5548 | 0.2489 | 0.8301 | 0.5418 |

**New overall best: swa_70pre_30topo_ep5 at comp=0.5549** (+0.0023 over pretrained).
All 6 blends beat pretrained. 70/30 ratio is the sweet spot.

Results: `logs/eval_swa_topo_results.csv`, Weights: `checkpoints/swa_topo/`

---

## PP Sweep on dist_sq Probmaps (Feb 20)

26 configs on dist_sq_ep5 probmaps. Key findings:
- T_low is the only meaningful PP parameter. Everything else is noise (±0.001).
- dist_sq optimal T_low=0.30-0.40 (vs pretrained's 0.70) — thinner predictions need lower threshold
- PP barely helps fine-tuned models (best config 0.506 vs pretrained's 0.553 — gap is in the model)
- Results: `logs/postprocessing_sweep.csv`

---

## Multi-Model Comparison Notebook (Feb 21)

Visual exploration comparing top models: pretrained, swa_70pre_30distsq, frozen_boundary_ep10, dist_sq_ep5.
Sections: score summary, cross-sections, probability distributions, thickness analysis, SDice deep dive,
connected components, PP sensitivity analysis.
- Notebook: `notebooks/analysis/multi_model_comparison.ipynb`
- Fixed uint8 bitwise NOT bug in SDice analysis (`~uint8(0)=255`, not `False` — breaks EDT)

---

## Prediction Thickness Investigation (Feb 18-20)

**Problem:** Model predicts 15-30% foreground per volume vs GT's 2-8%.
Surfaces are 3-5x too thick. This is the single biggest scoring problem.

**Impact on each metric:**
- **SDice (35%):** Thick predictions create two boundary surfaces (top+bottom). One aligns with GT, the other is penalized.
- **VOI (35%):** Excess voxels increase conditional entropy. Thickness merges nearby surfaces.
- **Topo (30%):** Merged surfaces change component count and create false tunnels.

**Root cause:** The probmaps themselves are too thick (model-level, not PP artifact).

**Training approaches tried:**
- dist_sq loss (quadratic penalty far from skeleton) — partial improvement
- Frozen encoder + dist_sq — best individual topo but degrades SDice
- clDice — needs 48GB VRAM for soft-skeletonization

**PP thinning verdict:** Ridge extraction destroys topology (topo 0.29→0.005).
Post-processing should reconnect fragments, not thin. Model must learn thinness.

---

## Refinement Model (ABANDONED)

Approached abandoned in favor of TransUNet pivot.
- Phase 2 result: delta -0.0298 vs baseline. Improves topo+sdice but destroys VOI.
- Per-voxel refinement fragments predictions — need component-level approach instead.

---

## TransUNet Fine-Tuning + SWA Era (Feb 20-25)

### All SWA blend evaluation results (Feb 23)

| Model | Comp | SDice | Notes |
|-------|------|-------|-------|
| **swa_70pre_30margin_dist_ep5** | **0.5551** | 0.8299 | **BEST** (original labels) |
| swa_70pre_30pseudo_margin_dist_ep5 | 0.5551 | 0.8306 | Ties best (pseudo-labels), better SDice |
| swa_70pre_30topo_ep5 | 0.5549 | 0.8291 | Previous best (frozen_boundary) |
| pseudo_margin2_cldice_ep5 (standalone) | 0.5547 | 0.8300 | gpu1, not SWA blended |
| pseudo_frozen_cldice_ep20 (standalone) | 0.5546 | 0.8304 | Best SDice standalone |
| swa_70pre_30cldice_ep20 | 0.5543 | 0.8308 | |
| swa_70pre_30pseudo_margin_dist_ep15 | 0.5542 | — | |
| swa_70pre_30unfreeze_vit_cldice_ep5 | 0.5534 | 0.8277 | ViT unfreeze (gpu3) |
| swa_70pre_30unfreeze_vit_balanced_ep5 | 0.5534 | 0.8277 | ViT unfreeze balanced (gpu3) |
| swa_70pre_30unfreeze_decoder_margin1_ep5 | 0.5498 | 0.8126 | Decoder unfreeze (gpu4) — degraded |
| swa_70pre_30pseudo_margin2_cldice_ep10 | 0.5545 | 0.8304 | gpu1 pseudo labels, ep10 |
| swa_70pre_30pseudo_margin2_cldice_ep15 | 0.5548 | 0.8308 | gpu1 pseudo labels, ep15 |
| multi 50/25/25 (margin+cldice) | 0.5549 | **0.8314** | Best SDice ever |
| multi 5-model (50/15/15/10/10) | 0.5545 | 0.8311 | |
| multi 60/20/20 (margin+cldice) | 0.5543 | 0.8311 | |

**Key insight:** SWA 70/30 blends consistently land at 0.553-0.555 regardless of fine-tuned model.
Pretrained weights dominate. Multi-model SWA pushes SDice but doesn't improve comp.

### Multi-model SWA blend results — DEAD END

| Blend | Pretrained | Fine-tuned models | Comp | SDice | Notes |
|-------|-----------|-------------------|------|-------|-------|
| **Best single** | 70% | 30% margin_dist_ep5 | **0.5551** | 0.8306 | Current best |
| Multi D | 50% | 25% margin_dist + 25% cldice | 0.5549 | **0.8314** | Best SDice ever |
| Multi B (5-model) | 50% | 15/15/10/10 spread | 0.5545 | 0.8311 | |
| Multi A | 60% | 20% margin_dist + 20% cldice | 0.5543 | 0.8311 | |

**Conclusion:** Multi-model SWA doesn't beat single 70/30 blend on comp (0.5549 vs 0.5551).

### Selective unfreezing results — DEAD END

| Experiment | Component | Loss | Comp (SWA 70/30) | Status |
|------------|-----------|------|-------------------|--------|
| gpu3 exp 1 | ViT | pure clDice | 0.5534 | Done (15 ep) |
| gpu3 exp 2 | ViT | balanced clDice | 0.5534 | Done (15 ep, resumed) |
| gpu4 exp 1 | Decoder+head | margin dist | 0.5498 | Done (15 ep) |
| gpu4 exp 2 | Decoder+head | balanced | pending eval | Done (15 ep) |
| gpu1 highLR | ViT | balanced, LR=1e-4 | pending eval | 9/15 ep done |
| gpu3 NEW | ViT+Decoder | balanced clDice | — | Training |
| gpu4 NEW | ViT | dist focus (dist=1.0) | — | Training |

All underperformed. Pretrained encoder is very hard to improve with these loss functions.

### gpu2 pseudo_frozen_margin_dist eval results

| Epoch | Comp (standalone) | SDice | Notes |
|-------|------------------|-------|-------|
| ep10 | 0.5543 | 0.8294 | |
| **ep15** | **0.5559** | **0.8308** | Best standalone |
| ep20 | 0.5550 | 0.8304 | |
| ep25 | 0.5552 | 0.8307 | |
| SWA 70/30 ep15 | 0.5542 | — | Below current best |
| SWA 70/30 ep5 | 0.5551 | 0.8306 | Ties best, better SDice |

### T_low PP sweeps (20-vol) — FINAL RESULTS

| Probmaps | Best Config | Comp | vs base_tl0.70 |
|----------|-------------|------|----------------|
| SWA val | close_erode_tl0.40_c2_e1 | 0.5368 | +0.0018 |
| Margin dist blend | erode_tl0.40_e1 | 0.5370 | +0.0014 |

**Conclusion:** erode_tl0.40_e1 best PP (+0.001-0.002 comp). 2-vol dry-run was misleading.

### T_low PP sweep — PRELIMINARY (2-vol, misleading)

| Config | Comp | Topo | SDice | VOI | FG% |
|--------|------|------|-------|-----|-----|
| **close_erode_tl0.40_c1_e1** | **0.5595** | **0.3357** | 0.7800 | 0.5307 | 10.2% |
| B_dme_tl0.4_r1 | 0.5346 | 0.3285 | 0.7049 | 0.5410 | 14.8% |
| base_tl0.40 | 0.5251 | 0.2998 | 0.7021 | 0.5413 | 14.6% |
| *baseline_t70 (current)* | *0.5350* | *0.2277* | *0.7907* | *0.5427* | — |

### Margin distance training results

| Model | Comp | Topo | SDice | VOI | n |
|-------|------|------|-------|-----|---|
| swa_70pre_30topo_ep5 (baseline) | 0.5526 | 0.2354 | 0.8255 | 0.5517 | 24 |
| **frozen_margin_dist_ep5** | **0.5500** | **0.2679** | 0.8069 | 0.5350 | 24 |
| frozen_margin_dist_ep15 | 0.5482 | 0.2616 | 0.8059 | 0.5361 | 24 |
| frozen_margin_dist_best | 0.5470 | 0.2622 | 0.8022 | 0.5358 | 24 |
| frozen_margin_dist_ep10 | 0.5463 | 0.2650 | 0.7997 | 0.5341 | 24 |

### SWA connectivity PP sweep results — DISAPPOINTING

No connectivity method beat simple baselines on SWA probmaps:

| Config | Comp | Topo | SDice | VOI |
|--------|------|------|-------|-----|
| baseline_t70 | **0.5350** | 0.2277 | 0.7907 | 0.5427 |
| B_dme_tl0.7_r1 | 0.5345 | 0.2273 | 0.7909 | 0.5414 |
| baseline_t50 | 0.5328 | 0.2365 | 0.7616 | 0.5579 |
| D_combo_bd3_i2 | 0.5270 | 0.2156 | 0.7846 | 0.5364 |

### SWA Weight Averaging — proven approach

70/30 pretrained/fine-tuned ratio is the sweet spot.

**Topo-focused blends (frozen_boundary source):**

| Model | Comp | Topo | SDice | VOI | n |
|-------|------|------|-------|-----|---|
| pretrained (baseline) | 0.5526 | 0.2354 | 0.8255 | 0.5517 | 24 |
| swa_90pre_10topo (ep10) | 0.5544 | 0.2403 | 0.8285 | 0.5496 | 24 |
| swa_70pre_30topo (ep10) | 0.5545 | 0.2490 | 0.8301 | 0.5407 | 24 |
| **swa_70pre_30topo_ep5** | **0.5549** | 0.2499 | 0.8291 | 0.5420 | 24 |
| swa_70pre_30sdice_ep15 | 0.5548 | 0.2489 | 0.8301 | 0.5418 | 24 |

### Fine-Tuning Experiments

All fine-tuned models degrade SDice vs pretrained. Frozen encoder > discriminative LR.

| Model | Best Comp | Best Topo | Best SDice | Strategy |
|-------|-----------|-----------|------------|----------|
| frozen_boundary (gpu2) | 0.5408 (ep10) | **0.2642** (ep10) | 0.7871 (ep15) | Frozen encoder |
| frozen_dist_sq (gpu2) | 0.5402 (ep25) | 0.2634 (ep25) | 0.7885 (ep10) | Frozen encoder |
| discrim_boundary (gpu1) | 0.5286 (ep15) | 0.2342 (ep15) | 0.7836 (ep15) | Discriminative LR |
| discrim_dist_sq (gpu1) | 0.5269 (ep25) | 0.2292 (ep15/25) | 0.7841 (ep25) | Discriminative LR |

### PP Sweep Findings

- **T_low is the only meaningful PP parameter.** Closing, dust removal, confidence filtering = noise (±0.001).
- **Optimal T_low depends on the model.** Pretrained optimal T_low=0.70, dist_sq optimal T_low=0.30-0.40.
- **PP barely helps fine-tuned models.** Best fine-tuned PP config = 0.506 vs pretrained's 0.553.

### Prediction Thickness (core problem)

Model predicts 15-30% FG per volume vs GT's 2-8%. Surfaces are 3-5x too thick.
dist_sq partial improvement (best topo=0.2642), SWA blending thins slightly,
ridge thinning destroys topology. Margin distance loss implemented Feb 22.

### Pseudo-labeling pipeline

80.4% of unlabeled voxels converted at 0.85/0.15 thresholds.
704 probmaps (42 GB) + 704 pseudo-labels (21 GB) generated.

### clDice pseudo-label training

25 epochs on gpu2. Loss plateaued ~1.052 from ep11.
Config: frozen encoder, SWA weights, pseudo-labels, cldice=0.5, iters=5.

### Margin distance loss (implemented Feb 22)

Replaces dist_sq with margin-based variant: `penalty = max(0, dist - margin)^power`.
Voxels within margin of skeleton get zero penalty. With margin=3, surfaces up to ~6 voxels thick
incur no penalty. Distance normalization changed to raw voxel distances capped at 10.

### GPU Fleet History

| Name | GPU | Role | Final Status |
|---|---|---|---|
| gpu0 | RTX 5090 32GB | Primary control, eval, training | Active |
| gpu1 | RTX 6000 Ada 48GB | ViT high-LR unfreeze, pseudo_margin2_cldice | Shut down, checkpoints on gpu0 |
| gpu2 | RTX 6000 Ada 48GB | Round-2 pseudo-labeling | Abandoned (disk full) |
| gpu3 | RTX 6000 Ada 48GB | ViT+decoder balanced unfreeze | Shut down, checkpoints on gpu0 |
| gpu4 | RTX 6000 Ada 48GB | Decoder balanced | Shut down, checkpoints on gpu0 |
| data-gpu | RTX 6000 Ada 48GB | External data download + pseudo-labeling | Shut down, data transferred |

### Connectivity PP Approaches Tested

4 methods in `scripts/sweep_connectivity_pp.py`:
(A) probmap-guided gap filling, (B) dilate-merge-erode, (C) two-pass hysteresis,
(D) combined C→A→bridge cleanup. ~30 configs total.
Methods B and D reduce FG% but no method beat baselines on SWA probmaps.
