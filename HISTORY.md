# Vesuvius Surface Detection — Project History

Full narrative of iterations, decisions, and findings. Source material for a blog post
after the competition. See **NOTES.md** for the active working document.

## Blog Post Plan
Planning to write a blog post about this project after the competition. These notes
capture the full story of iterations — what we tried, why, what worked, what didn't, and
the key insights along the way (e.g. the inference pipeline discovery, the lr_find struggles,
the disconnect between val metrics and public scores).

---

## Training Run Details

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

### Run 9: Lower Learning Rate (our current best model)
- **LR:** Hardcoded 1e-5. Discriminative: encoder=1e-7, decoder=1e-6, head=1e-5
- Everything else same as Run 8
- **Results:** Best comp_score **0.570** at epoch 8 (vs Run 8's 0.562 at epoch 1)
  - Loss steadily decreased: train 0.523→0.480, valid 0.543→0.474
  - Model improved continuously through epoch 8, then plateaued
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

## Inference Pipeline Evaluation (Feb 12, 2026)

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

## Overnight Pipeline (Feb 12, 2026)

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
