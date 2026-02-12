# Vesuvius Surface Detection — Project Notes

## Blog Post
Planning to write a blog post about this project after the competition. These notes should
capture the full story of iterations — what we tried, why, what worked, what didn't, and
the key insights along the way (e.g. the inference pipeline discovery, the lr_find struggles,
the disconnect between val metrics and public scores). Keep recording decision rationale,
surprising findings, and dead ends — they're all part of the narrative.

## Competition
- **Goal:** Detect papyrus surfaces in 3D CT scans of Herculaneum scrolls
- **Metric:** `0.30*TopoScore + 0.35*SurfaceDice@tau=2 + 0.35*VOI_score`
- **Deadline:** Feb 27, 2026
- **Submission:** Code competition — inference runs in a Kaggle notebook (GPU, ≤9hr, no internet)

## Data
- 786 training volumes, each `(320, 320, 320)` uint8
- Labels: 0=background, 1=foreground (papyrus), 2=unlabeled (ignore in loss)
- Foreground is sparse (~2-8% of voxels); sample volume showed bg=37.3%, fg=4.9%, unlabeled=57.8%
- ~120 hidden test volumes at submission time
- 6 scroll_ids: 34117 (382), 35360 (176), 26010 (130), 26002 (88), 44430 (17), 53997 (13)
- train.csv has 806 rows but only 786 have files on disk (some IDs in CSV lack .tif files)
- After filtering to available files: 704 train, 82 val

## Hardware
- RTX 5090 (32GB VRAM), 125GB RAM, 32 CPU cores
- Venv: `/home/mongomatt/Projects/vesuvius/vesuvius/` (Python 3.12.3)
- Installed: torch 2.9.1+cu130, fastai 2.8.6, fastprogress 1.0.5, ipywidgets, tifffile, scipy, scikit-image

## Training Runs

### Run 1: Baseline 3D U-Net (in progress)
- **Architecture:** 3D U-Net with BatchNorm, 22.6M params
- **Channels:** 1 → 32 → 64 → 128 → 256 → 512 (bottleneck)
- **Patch size:** 128^3, batch_size=2, mixed precision (fp16)
- **LR:** 1.9e-2 (from lr_find), one-cycle policy, 30 epochs
- **Loss:** 0.5*MaskedBCE + 0.5*MaskedDice
- **Val set:** scroll 26002 (82 volumes)
- **Status:** Training in progress
- **Results:** TBD

### Run 2: GroupNorm + Dropout (complete)
- Switch BatchNorm → GroupNorm (better for small batch sizes)
- Add dropout at bottleneck (0.2)
- lr_find suggested 3.63e-3 (lower than Run 1's 1.9e-2)
- Everything else same as Run 1
- **Results:** Best dice **0.172** at epoch 9 (vs Run 1's 0.121 — 42% improvement)
- Dice unstable across epochs (bounces 0.05–0.17), loss plateau around 0.55
- See "Queued Changes" section below for details

### Run 3: SuPreM SegResNet + clDice (complete)
- **Architecture:** MONAI SegResNet, 4.7M params (vs 22.6M UNet)
- **Pre-trained:** SuPreM supervised weights (2,100 CT volumes, 25 organ classes)
- **Channels:** 16 → 32 → 64 → 128, residual blocks, GroupNorm
- **Weights:** 79/83 params loaded from checkpoint, final conv randomly init
- **Loss:** 0.3*BCE + 0.3*Dice + 0.4*clDice (all masked for label=2)
  - clDice uses soft skeletons via iterative morphological erosion (5 iters)
  - Skeleton computed on 2x-downsampled patches (64^3) to save VRAM
  - Target skeleton computed with `torch.no_grad()` — only pred needs gradients
  - Directly targets competition's TopoScore component
- **LR:** lr_find suggested 1.74e-3, trained with 1e-3
- **Results:** Best dice **0.281** at epoch 8 (vs Run 2's 0.172 — **63% improvement**)
  - Dice still unstable across epochs (0.13–0.28)
  - Foreground prediction: 8.6% (vs Run 2's ~1.9%, closer to true ~4-8%)
  - Loss decreased 0.67 → 0.62 over 30 epochs
- **Checkpoint:** 18 MB (vs 87 MB UNet)
- **Notebook:** `notebooks/vesuvius_train_v3.ipynb`
- **Kaggle submission:** Version 5 of inference notebook (uses MONAI SegResNet)
- See `PRETRAINED_MODELS.md` for full research

## What's Been Built

### `notebooks/vesuvius_train.ipynb`
Full training notebook with 8 sections, ready to run cell-by-cell in Jupyter Lab.

**Architecture:** 3D U-Net
- 4 encoder blocks, bottleneck, 4 decoder blocks with skip connections
- Channel progression: 1 → 32 → 64 → 128 → 256 → 512 (bottleneck)
- 22.6M parameters
- Final layer: 1x1x1 Conv3d → 1 channel (raw logits, sigmoid applied in loss/inference)

**Training setup:**
- Random 128^3 patches cropped from full 320^3 volumes (full volumes OOM even on 5090)
- Training uses ~11 GB VRAM at batch_size=2
- Mixed precision (fp16) via fastai MixedPrecision callback
- Augmentations: random flips (each axis independently), random 90-degree rotations in random planes
- Loss: 0.5*MaskedBCE + 0.5*MaskedDice (both ignore label=2 voxels)
- Validation: scroll 26002 held out (82 volumes after filtering)
- fastai one-cycle policy, 30 epochs, lr_find() suggested 1.9e-2
- Best model saved by Dice metric to `checkpoints/models/best_unet3d.pth`

**Inference:**
- Sliding window: 128^3 patches with stride 96 (32-voxel overlap)
- 27 patches per 320^3 volume, overlapping regions averaged
- Uses ~8 GB VRAM
- Output thresholded at 0.5 → uint8 → .tif → submission.zip

## VRAM Measurements
| Config | VRAM |
|--------|------|
| Train 128^3 bs=2 (fwd+bwd) | 11.0 GB |
| Train 160^3 bs=2 (fwd+bwd) | 21.4 GB |
| Train 320^3 bs=1 | OOM (needs ~53 GB) |
| Inference 128^3 sliding window | 8.1 GB |

## Design Decisions & Rationale

### Validation split: scroll-level holdout vs random %
We split by scroll_id rather than randomly. If we randomly split individual volumes, volumes
from the same scroll would appear in both train and val. Since volumes from the same scroll
share similar texture/density/structure, the model could "cheat" — validation Dice would be
inflated and wouldn't reflect real test performance. Holding out entire scroll 26002 (82 volumes,
~10%) gives an honest signal for generalization to unseen scrolls.

### BatchNorm (Run 1) — and why we're switching to GroupNorm (Run 2)
Run 1 uses BatchNorm because it's the standard default. However, **BatchNorm is a poor choice
at batch_size=2** — it computes mean/variance over the batch dimension, so with only 2 samples
the statistics are very noisy. This hurts training stability.

Better alternatives for small batch sizes:
- **GroupNorm** (recommended) — splits channels into groups (e.g., 8) and normalizes within each
  group. Batch-size independent. Designed specifically for small-batch regime.
- **InstanceNorm** — normalizes per-sample, per-channel. Also batch-size independent. Common in
  style transfer, also works in segmentation.
- **LayerNorm** — normalizes across all channels. Less common in CNNs.

GroupNorm is the best fit here: it's a drop-in replacement and well-established for 3D medical
image segmentation with small batches.

### Dropout
Not used in Run 1. Adding dropout=0.2 at the bottleneck for Run 2. With 700+ training volumes
we may not be severely overfitting, but it's cheap regularization insurance. Dropout at the
bottleneck (the narrowest part of the U-Net) is the standard placement — it regularizes the
most compressed representation without disrupting skip connections.

### Loss function vs competition metric gap
Our training loss (0.5*BCE + 0.5*Dice) optimizes for voxel-level overlap. The competition
metric is `0.30*TopoScore + 0.35*SurfaceDice@τ=2 + 0.35*VOI_score`, which rewards:
- **TopoScore** (30%) — topological correctness (no holes, mergers, breaks)
- **SurfaceDice@τ=2** (35%) — boundary accuracy with 2-voxel tolerance
- **VOI score** (35%) — variation of information (instance separation quality)

Standard Dice is a reasonable proxy to start, but the gap will become the bottleneck as the
base model improves. Options to close the gap (in order of practicality):

**Learnable (built into model/loss):**
1. **clDice (centerline Dice)** — computes soft skeletons and measures Dice on them. Designed
   for thin, continuous structures. Encourages connectivity, penalizes breaks. Directly targets
   TopoScore. Differentiable, drop-in loss term. Best first choice.
2. **Boundary/surface loss** — reformulates loss as distance between predicted and ground truth
   contours (using distance transform). Directly targets SurfaceDice@τ=2. Combinable with Dice.
3. **Distance transform regression** — predict distance-to-nearest-surface instead of binary
   mask. Naturally produces smoother, more continuous surfaces. Threshold to get final mask.
4. **Persistent homology loss** — directly penalizes incorrect Betti numbers (connected
   components, holes, tunnels). Most direct TopoScore optimizer but expensive and complex.
5. **Multi-task surface normals** — add a head predicting surface normal direction, forces
   model to learn surface geometry, implicitly improves connectivity.

**Post-processing (applied after model):**
6. **Connected component filtering** — remove small disconnected fragments
7. **Morphological operations** — closing (fill small holes), opening (remove noise)
8. **Surface smoothing** — Gaussian smoothing on probability map before thresholding

Recommended approach: add **clDice + boundary loss** to our existing BCE+Dice loss as
additional weighted terms. Then apply light post-processing (connected components +
morphological closing) on top. This attacks both TopoScore and SurfaceDice from both sides.

## Queued Changes (for Run 2)

### 1. GroupNorm replacing BatchNorm
Change `ConvBlock3D` to use `nn.GroupNorm(num_groups=8, num_channels=out_ch)` instead of
`nn.BatchNorm3d(out_ch)`. Using 8 groups (so group size = channels/8). This means:
- 32 channels → 8 groups of 4
- 64 channels → 8 groups of 8
- 128 channels → 8 groups of 16
- 256 channels → 8 groups of 32
- 512 channels → 8 groups of 64

### 2. Bottleneck dropout
Add `nn.Dropout3d(p=0.2)` after the bottleneck block in UNet3D.forward(), before the decoder.

## Improvement Roadmap (prioritized)

### High impact, easy to try
- [x] **GroupNorm + dropout** — queued for Run 2
- [x] **Pre-trained 3D encoder (SuPreM SegResNet)** — added in Run 3. MONAI SegResNet with
  SuPreM supervised weights, 4.7M params. Best dice jump: 0.172 → 0.281 (63% improvement).
- [x] **Foreground-biased patch sampling** — currently patches are uniform random, so most
  patches are mostly background/unlabeled. Bias toward patches containing foreground voxels
  so the model sees more positive examples per epoch. Could use a 50/50 strategy: half the
  patches are guaranteed to contain foreground.
- [x] **Test-time augmentation (TTA)** — added in Run 7. 7-fold: original + 3 flips + 3 rotations.
- [x] **Larger patches** — 160^3 added in Run 7 (was 128^3). Stride=80 for uniform overlap.
- [ ] **Gaussian-weighted SWI blending** — replace simple overlap averaging with Gaussian
  importance weighting (center of each patch weighted higher). All top notebooks use this.
  No retraining needed, just change inference. (from notebook analysis)
  **Status:** Implemented in `scripts/eval_inference.py` but eval never ran overnight.
  Need to run eval to measure impact before adopting in Kaggle inference.
- [ ] **Logit-space TTA averaging** — average raw model outputs before sigmoid, not after.
  More principled (equivalent to geometric mean in probability space). Free improvement.
  (from notebook analysis)
  **Status:** Implemented in `scripts/eval_inference.py` but eval never ran overnight.
  Need to run eval to measure impact before adopting in Kaggle inference.
- [x] **Killer ant surface splitting** — raycasting + Dijkstra3D recursive splitting of
  merged papyrus sheets in post-processing. Model-agnostic, directly targets TopoScore + VOI.
  Uses `cc3d` + `dijkstra3d`. Integrated into both eval script and Kaggle inference.
  Wheels bundled in `kaggle/kaggle_weights/wheels/` for offline Kaggle install.
  **Status:** Code integrated but never evaluated on val set. `eval_inference.py --split`
  will measure impact. Kaggle wheels upload also failed (missing `--dir-mode zip`).

### Inference pipeline & model selection
- [ ] **Periodic checkpoint saving** — save every N epochs during training, not just "best".
  After training, run full inference pipeline (Gaussian SWI + logit TTA + hysteresis + splitting)
  on all saved checkpoints via `eval_inference.py` to find the true best. Lightweight training
  metric is directional only — final selection uses the real pipeline.
- [ ] **Track raw vs post-processed scores** — models with lower raw comp_score may benefit
  MORE from post-processing (messier predictions = more headroom for cleanup). Track both
  raw and full-pipeline scores side by side to understand this relationship. Don't assume
  the best raw model is the best final model.
- [x] **Learned post-processing (refinement network)** — train a lightweight second model that
  takes the main SegResNet's raw probability map (320^3 float) as input and outputs a refined
  binary segmentation. Replaces all hand-tuned post-processing (hysteresis, closing, splitting)
  with a single forward pass.
  - **Phase 1:** Train with GT labels as target, using the main model's predictions as input.
    Generate training pairs by running main model inference on all 786 training volumes.
    Architecture can be very small (e.g. shallow 3D U-Net or even 2D slice-wise) since it's
    refining, not segmenting from scratch.
  - **Phase 2:** Swap in topology-aware loss functions that are too expensive for the main
    training loop — persistent homology loss, direct Betti matching, surface-distance loss,
    etc. The refinement model's only job is to fix topology, so these losses are well-targeted.
    Could push beyond what hand-tuned post-processing achieves.
  - **Inference:** Single lightweight forward pass replaces the entire post-processing pipeline.
    Faster, more generalizable, and differentiable (could eventually be fine-tuned end-to-end
    with the main model).
  - **Kaggle:** Would need to trace this model too and bundle it (~small additional weight file).
  - **Phase 3 (end-to-end fine-tuning):** Once the refinement model works standalone, make
    the full pipeline differentiable: main model → refinement model → topology-aware loss.
    Gradients from the topology loss flow back through both models. This is the ultimate
    goal — the main model learns to produce outputs that are easy for the refinement model
    to clean up, and the refinement model learns to fix whatever the main model gets wrong.

### Refinement Model (in progress — Feb 12)
Training data generation running: v9 model Gaussian SWI inference on all 786 training
volumes, saving float16 .npy probmaps at `/data/refinement_data/probmaps/` (~63 MB each).
Script: `scripts/generate_refinement_data.py`. ~2.8s per volume, ~40 min total.

**Notebook:** `notebooks/refinement/vesuvius_train_refinement.ipynb`

**Architecture:** RefinementUNet3D — shallow 3D U-Net, ~365K params
- Channels: [8, 16, 32, 64], 4 encoder levels
- GroupNorm, ReLU, Dropout(0.2) at bottleneck
- Concat skip connections, trilinear upsampling
- Input: 1ch probmap, Output: 1ch refined logits

**Training setup:**
- 160^3 random crops from 320^3 probmaps (~2.5 GB VRAM at bs=2)
- Full 320^3 at inference/metric eval only (forward only, ~1 GB)
- Augmentations: random flips + 90-degree rotations
- Phase 1: `fit_one_cycle`, LR=1e-3, 50 epochs, loss=0.5*BCE + 0.5*Dice
  (from scratch — warmup helps explore loss landscape)
- Phase 2: `fit_flat_cos`, LR=3.3e-4, 30 epochs, loss=0.2*BCE + 0.2*Dice + 0.3*clDice + 0.3*Boundary
  (continuing from Phase 1 weights — no warmup to avoid disruption)
- Metric: RefinementCompetitionMetric — full 320^3 forward pass on 5 val volumes, no SWI needed

**Acceptance criterion:** Head-to-head vs hand-tuned post-processing (hysteresis T_low=0.35,
T_high=0.80 + anisotropic closing + dust removal) on same probmaps. Must beat it on mean
comp_score across val set.

**Kaggle deployment:** `torch.jit.trace` → ~1.5 MB traced model. Pipeline:
main model → Gaussian SWI + logit TTA → probmap → refinement model → threshold 0.5 → done.
Adds ~0.1s per volume.

### Current plan (Feb 12)
1. **Refinement model** — training data generation nearly complete (486/786 done, ~15 min
   remaining). Notebook ready at `notebooks/refinement/vesuvius_train_refinement.ipynb`.
   Run Phase 1 + Phase 2 training, then head-to-head eval vs hand-tuned post-processing.
2. **Then end-to-end fine-tuning** — connect main model + refinement model, train jointly
   with topology-aware losses. Use `fit_flat_cos` schedule (no warmup) with the LR magnitudes
   from v10b-v2. Train longer (50 epochs) to let the recovery phase play out fully.
3. **Meanwhile:** Kill eval_inference `--split` (surface splitting takes 80 min/volume —
   impractical). Rerun without `--split` to get Gaussian SWI + logit TTA comparison quickly.
   Verify Kaggle inference notebook matches best pipeline.

### Medium impact, moderate effort
- [ ] **Attention gates on skip connections** — added in Run 10 but **unverified** due to LR
  issue (new modules couldn't learn). Re-testing in Run 10b with per-module lr_find.
  AttentionGate3D at each skip connection, ~5.5K params (~0.1%).
- [ ] **nnUNet framework** — self-configuring medical segmentation framework that often
  wins competitions. Someone has already preprocessed our data for nnUNetv2 (91.8 GB
  dataset on Kaggle). Would require learning the framework but could yield strong results
  with minimal manual tuning.
- [ ] **Scroll-level cross-validation** — added in Run 11 but **unverified** — v11 used the
  same SegResNetDSAttn + LR as v10, so results are unreliable. Need to re-run after v10b
  proves out correct LRs. 3-fold scroll-grouped CV with probability averaging ensemble.
- [ ] **Post-processing** — connected component filtering, morphological operations to clean
  up surfaces. Important for TopoScore (rewards continuous surfaces, penalizes holes/mergers).
- [ ] **Deep supervision** — added in Run 10 but **unverified** due to LR issue (DS heads
  couldn't learn). Re-testing in Run 10b with per-module lr_find. Auxiliary 1x1 conv heads
  at decoder levels 0 (64ch @ D/4) and 1 (32ch @ D/2). Upsampled to full res, supervised
  with BCE+Dice (weight=0.3).
- [ ] **Volume caching in RAM** — 786 volumes x 32MB = ~25 GB, fits in 125 GB RAM. Would
  eliminate I/O bottleneck if we're data-loading bound.
- [ ] **Skeleton recall loss** — add skeleton-based recall term to loss. Pre-compute GT
  skeleton, penalize missed skeleton voxels. Directly targets TopoScore. (from notebook analysis)
- [ ] **3D CutOut augmentation** — zero out random cuboid regions during training.
  Strong regularizer used by top notebooks. (from notebook analysis)

### Worth experimenting with
- [ ] **3-class formulation** — predict background/foreground/unlabeled as 3 classes instead
  of binary with mask. The model learns to explicitly predict "unlabeled" regions, which may
  improve boundary predictions where labeled and unlabeled regions meet. All top public
  notebooks use this approach (0.486→0.505 for TransUNet). (confirmed by notebook analysis)
- [ ] **Residual connections** in conv blocks (ResBlock instead of plain ConvBlock)
- [x] **clDice loss** — added in Run 3. Soft skeletons on 2x-downsampled patches.
- [x] **Boundary/surface loss** — added in Run 4. Signed distance transform from GT surface.
- [x] **Post-processing pipeline** — added in Run 7. Hysteresis + anisotropic closing + dust removal.
- [ ] **`fit_flat_cos` schedule** — replace `fit_one_cycle` for pretrained+new module training.
  Confirmed in Run 10b-v2: one_cycle warmup disrupts pretrained features, `flat_cos` avoids this.
  LR magnitudes from v10b-v2 are likely correct (1e-7/1e-6/1e-5/1e-4 per group).
- [ ] **Cosine annealing** or different LR schedules
- [ ] **Different loss weighting** — currently 50/50 BCE/Dice. Could try more Dice weight
  since foreground is sparse.
- [ ] **Deeper/wider model** — (32, 64, 128, 256, 512) with 5 encoder levels if using
  smaller patches.
- [ ] **Extra training data from scrollprize.org segments** — manually mapped surface meshes
  exist for 5 scrolls. Could rasterize meshes to create binary labels, crop into 320^3
  patches, and generate more training data. High effort data engineering but potentially
  large data boost.

### Kaggle submission
- [x] **First submission completed** — see submission process below

## Competition Scores
| Run | Submission | Val Dice | Val Comp | Public Score | Notes |
|-----|-----------|----------|----------|-------------|-------|
| (template) | Dec 11 | — | — | 0.441 | Pre-existing public template, not ours |
| Run 1 | Feb 10 | 0.121 | — | 0.331 | BatchNorm baseline |
| Run 2 | Feb 10 | 0.172 | — | 0.290 | GroupNorm + Dropout — worse score despite better dice! |
| Run 3 | Feb 10 | 0.281 | — | 0.348 | SegResNet + clDice |
| Run 8 | Feb 11 | 0.276 | 0.562 | 0.423 | FG-biased sampling + TTA + hysteresis (LR too high — peaked epoch 1) |
| Run 9 | Feb 12 (v13) | 0.278 | 0.570 | PENDING | Lower LR (1e-5), T_low=0.40 — Kaggle v13 with surface splitting fix |
| Run 10 | Feb 12 (v9) | 0.154 | 0.584 | 0.267 | + DS + attn gates — peaked epoch 0, never learned (LR issue) |
| Run 10b-v1 | — | 0.283 | 0.567 | — | Per-module lr_find valley — still too aggressive, peaked epoch 1 |
| Run 10b-v2 | — | 0.159 | 0.581 | — | Hardcoded LRs — loss monotonically decreased, comp_score dipped then recovered (see LR schedule insight) |

**Key insight:** Higher validation dice does NOT equal better competition score. The metric
(TopoScore + SurfaceDice + VOI) cares about topology and surface quality, not just voxel
overlap. This is why targeting the loss function and post-processing matters more than raw dice.

### Run 4: Boundary Loss + Post-Processing (complete)
- **Architecture:** Same SegResNet with SuPreM weights
- **Loss:** 0.2*BCE + 0.2*Dice + 0.3*clDice + 0.3*Boundary
  - Boundary loss uses signed distance transform from GT surface
  - Directly targets SurfaceDice@τ=2 (35% of competition score)
- **New in inference:** Threshold sweep (0.3–0.7) on validation set + post-processing
  (morphological closing + connected component filtering)
- **Results:** Best dice **0.232** — regressed from Run 3's 0.281
- **Notebook:** `notebooks/vesuvius_train_v4.ipynb`

### Run 5: Competition Metric Monitoring (queued)
- **Architecture:** Same SegResNet with SuPreM weights, same loss as Run 4
- **Key change:** Exact competition metric (`0.30*TopoScore + 0.35*SurfaceDice@τ=2 + 0.35*VOI`)
  computed on 5 validation volumes each epoch via `topometrics` library
- `SaveModelCallback` monitors `comp_score` instead of `dice` for model selection
- LR variable auto-updated by `lr_find` (no manual copy-paste)
- Built `topometrics` library locally (Betti matching C++ module + pybind11)
  from `/tmp/vesuvius_metrics/extracted/topological-metrics-kaggle/`
- **Overhead:** ~20s per epoch for competition metric eval (5 volumes x sliding window + scoring)
- **Notebook:** `notebooks/vesuvius_train_v5.ipynb`

### Run 6: Discriminative LR + Fast Metric (queued)
- **Architecture:** Same SegResNet with SuPreM weights, same loss as Run 5
- **Discriminative LR** — 3 param groups via `segresnet_splitter`:
  - Encoder (`convInit` + `down_layers`): `LR/100` — gentle fine-tuning of SuPreM weights
  - Decoder (`up_layers` + `up_samples`): `LR/10` — moderate adaptation
  - Head (`conv_final`): `LR` — aggressive learning for randomly-init final conv
  - Uses `fit_one_cycle(EPOCHS, lr_max=slice(LR/100, LR))` with fastai's `even_mults`
- **Downsampled competition metric** — 4x downsample (320^3 → 80^3) via nearest-neighbor
  `scipy.ndimage.zoom` before Betti matching. ~1s per volume instead of 20+ minutes.
  Topology is a coarse property, so downsampling preserves the signal.
- **Notebook:** `notebooks/vesuvius_train_v6.ipynb`
- **Future LR experiments:**
  - Cosine annealing with warm restarts (SGDR) — periodically reset LR to escape local minima
  - Flat cosine (no warmup) — skip one_cycle warmup since model is pre-trained

### Run 6: Discriminative LR + Fast Metric (queued)
- **Architecture:** Same SegResNet with SuPreM weights, same loss as Run 5
- **Discriminative LR:** 3 param groups via `segresnet_splitter` (encoder/decoder/head)
  - `fit_one_cycle(lr_max=slice(LR/100, LR))` with 3 groups → geometrically spaced LRs
  - Encoder (SuPreM weights): `LR/100` — gentle fine-tuning
  - Decoder: `LR/10` — moderate adaptation
  - Head (conv_final, random init): `LR` — aggressive learning
- **Fast competition metric:** downsample pred+label 4x (320³→80³) before Betti matching
  - ~1s per volume instead of 20+ minutes at full res
  - Topology is coarse, so downsampling preserves the signal
  - Total overhead: ~20s per epoch (5 volumes × ~4s each)
- **Notebook:** `notebooks/vesuvius_train_v6.ipynb`
- **Future experiments:** cosine warm restarts (SGDR), flat cosine (no warmup)

### Run 7: TTA + Hysteresis + 160^3 Patches (queued)
- **Architecture:** Same SegResNet with SuPreM weights, same loss as Run 6
- **160^3 patches** — larger spatial context (was 128^3), stride=80 for uniform overlap
  - Positions [0, 80, 160] = 27 patches per volume (same count, better overlap uniformity)
  - Fits in VRAM: 21.4 GB at bs=2 (measured in Run 1)
- **TTA (7-fold)** at inference only — original + 3 axis flips + 3 HW-plane rotations
  - ~7x inference time: 120 volumes x 7 x ~3s = ~42 min (within Kaggle 9hr limit)
  - NOT used during training metric eval (too slow per-epoch)
- **Hysteresis thresholding** — dual-threshold seed-and-propagate
  - Strong seeds: prob >= T_HIGH (0.85), weak region: prob >= T_LOW (0.45)
  - `scipy.ndimage.binary_propagation` with 26-connectivity
  - Replaces fixed `prob > threshold` everywhere
- **Anisotropic closing** — z-heavy structuring element (z_radius=2, xy_radius=1)
  - Disk-shaped in XY, extended in Z — better for layered papyrus structure
  - Replaces isotropic `np.ones((5,5,5))`
- **Drop dice metric** — dice and comp_score are uncorrelated (confirmed in Run 6)
  - Only track comp_score during training
- **Dust removal** — remove connected components < 100 voxels
- **Threshold sweep:** T_low sweep [0.35, 0.40, 0.45, 0.50, 0.55] with fixed T_high=0.85
- **Notebook:** `notebooks/vesuvius_train_v7.ipynb`

### Run 8: Foreground-Biased Patch Sampling (complete)
- **Architecture:** Same SegResNet with SuPreM weights, same loss as Run 7
- **Foreground-biased sampling** — with probability `FG_BIAS=0.5`, center the 160^3 crop
  on a random foreground voxel instead of random location
  - `np.argwhere(lbl == 1)` finds all foreground coordinates
  - Pick one at random, center crop on it (clamped to volume bounds)
  - Falls back to random crop if no foreground exists in volume
  - Only applied during training (augment=True), not validation
  - 50/50 split ensures model still sees full distribution of background regions
  - Confirmed working: patches average 7.8% FG (vs ~4-5% before), 10/10 patches >1% FG
- **Motivation:** Foreground is sparse (~4-5% of voxels), so random crops are dominated
  by background/unlabeled. Biasing toward foreground gives more positive examples per epoch.
- **LR:** lr_find suggested 5.75e-4, discriminative: slice(5.75e-6, 5.75e-4)
- **Results:** Best comp_score **0.562** at epoch 1 (of 30) — peaked very early
  - Epoch 0: comp_score=0.504, Epoch 1: comp_score=0.562, no improvement after
  - Threshold sweep: T_low=0.50 won (dice=0.276), up from default 0.45
  - Verification comp_score (3 volumes): 0.574, 0.538, 0.573 → mean **0.562**
- **Issue:** Model peaked at epoch 1/30 — LR too high. lr_find auto-selected 5.75e-4
  but the first valley in the lr_find plot is visually around 1e-5. The high LR caused
  the model to overshoot the optimum in a single epoch.
- **Notebook:** `notebooks/vesuvius_train_v8.ipynb`

### Run 9: Lower Learning Rate (complete)
- **Architecture:** Same SegResNet with SuPreM weights, same loss as Run 8
- **LR:** Hardcoded 1e-5 (was 5.75e-4 from lr_find in Run 8)
  - Run 8 peaked at epoch 1/30, indicating LR was too high
  - The first valley in lr_find is visually around 1e-5
  - Discriminative: encoder=1e-7, decoder=1e-6, head=1e-5
- **Skipped lr_find** — go straight to training with hardcoded LR
- Everything else same as Run 8 (FG-biased sampling, 160^3, TTA, hysteresis)
- **Results:** Best comp_score **0.570** at epoch 8 (vs Run 8's 0.562 — peaked at epoch 1)
  - Loss steadily decreased: train 0.523→0.480, valid 0.543→0.474
  - Model improved continuously through epoch 8, then plateaued ~0.562
  - Confirms LR was the issue in Run 8 — lower LR allows gradual improvement
  - Threshold sweep: T_low=0.40 won (dice=0.278), lower than Run 8's 0.50
  - Verification comp_score (3 volumes): 0.573, 0.546, 0.601 → mean **0.573**
- **Notebook:** `notebooks/vesuvius_train_v9.ipynb`

### Run 10: Deep Supervision + Attention Gates (complete)
- **Architecture:** `SegResNetDSAttn` — subclasses SegResNet with two additions:
  - **Attention gates** on all 3 skip connections (Oktay et al., 2018)
    - Learns spatial attention weights to suppress irrelevant background features
    - Particularly useful for sparse foreground (~4%)
    - Channels at each level: [64, 32, 16], F_int = ch // 2
    - Adds ~5,500 params (~0.1% of model)
  - **Deep supervision** — auxiliary 1x1 conv heads at intermediate decoder levels
    - Head 0: 64ch @ D/4 resolution
    - Head 1: 32ch @ D/2 resolution
    - Auxiliary predictions upsampled to full res via trilinear interpolation
    - Supervised with simple BCE + Dice (not full clDice + boundary — too expensive)
    - Weight: DS_WEIGHT=0.3 per auxiliary head
    - Training mode returns [main, aux_0, aux_1]; eval mode returns single tensor
    - Forces coarse decoder levels to produce reasonable segmentations → better gradient flow
- **Splitter:** encoder / decoder+attn_gates / head+ds_heads (3 param groups)
- **SuPreM weights:** Load into base SegResNet layers unchanged; new modules (attention_gates,
  ds_heads) are randomly initialized
- **LR:** Same as Run 9 (1e-5 hardcoded, discriminative)
- **Loss:** Main output: 0.2*BCE + 0.2*Dice + 0.3*clDice + 0.3*Boundary
  Auxiliary outputs: 0.5*BCE + 0.5*Dice, weight=0.3
- Everything else same as Run 9
- **Results:** Best comp_score **0.584** at epoch 0 — model peaked immediately, never improved
  - Dice dropped to **0.154** (vs Run 9's 0.278)
  - Predicted 0% foreground on test volume
  - T_low=0.35 won threshold sweep (dice=0.154)
  - Verification comp_score (3 volumes): 0.575, 0.475, 0.527 → mean **0.526**
  - **Issue:** New attention gates + DS heads are randomly initialized but shared the same
    low LR (1e-5) as the pretrained modules. They needed much higher LR to learn.
    The pretrained modules dominated, and the new modules couldn't catch up.
- **Notebook:** `notebooks/vesuvius_train_v10.ipynb`

### Run 10b-v1: Per-Module lr_find Valley (complete — LR still too high)
- **Architecture:** Same SegResNetDSAttn as Run 10
- **Key change:** 4 parameter groups with independent lr_find for each:
  - Group 0 (encoder): convInit + down_layers [pretrained] — lr_find range [1e-9, 1e-3]
  - Group 1 (decoder): up_layers + up_samples [pretrained] — lr_find range [1e-8, 1e-2]
  - Group 2 (new_modules): attention_gates + ds_heads [random init] — lr_find range [1e-7, 1e-1]
  - Group 3 (head): conv_final [random init] — lr_find range [1e-7, 1e-1]
- **Selective freeze:** For each lr_find run, freeze all groups except the target group.
  Extract valley LR from each. Reset model weights between runs.
- **Valley LRs:** encoder=9.12e-6, decoder=9.12e-5, new_modules=2.40e-3, head=5.50e-3
- **Results:** Best comp_score **0.567** at epoch 1 — peaked early again (same pattern as Run 8)
  - T_low=0.40 won threshold sweep (dice=0.283)
  - Verification comp_score: 0.576, 0.548, 0.589 → mean **0.571**
  - Test volume predicted **0% foreground** — same failure as v10
  - **Issue:** `lr_find().valley` consistently picks ~100x too aggressively for this model.
    The valley heuristic finds the steepest descent point, but the actual usable LR is
    at the first dip before the steep descent region.
- **Notebook:** `notebooks/vesuvius_train_v10b.ipynb`

### Run 10b-v2: Hardcoded LRs from lr_find Plots (complete)
- **Architecture:** Same SegResNetDSAttn as Run 10
- **Key change:** Hardcoded LRs eyeballed from the first valley in lr_find plots (~100x
  lower than valley method, consistent 10x spacing between groups):
  - Group 0 (encoder): **1e-7** (valley was 9.12e-6)
  - Group 1 (decoder): **1e-6** (valley was 9.12e-5)
  - Group 2 (new_modules): **1e-5** (valley was 2.40e-3)
  - Group 3 (head): **1e-4** (valley was 5.50e-3)
- **Motivation:** lr_find valley heuristic is unreliable for this model — consistently
  picks too high, causing model to peak at epoch 0-1. Same issue seen in Run 8→9 where
  manually lowering LR 50x fixed early peaking.
- Everything else same as v10b-v1
- **Results:** Best comp_score **0.581** at epoch 2. Loss monotonically decreased throughout
  all 30 epochs (first time!). Comp_score improved epochs 0-2, then dipped during the
  one_cycle warmup phase, then **started recovering after epoch 11** as LR annealed down.
  - T_low=0.35 won threshold sweep (dice=0.159)
  - Verification comp_score (3 volumes): 0.516, 0.430, 0.483 → mean **0.476**
  - Test volume predicted **0% foreground** (same issue as v10, v10b-v1)
  - SaveModelCallback only saved epoch 2 — the recovery after epoch 11 was never
    captured because it didn't exceed 0.581
- **LR Schedule Insight (important for future runs):**
  `fit_one_cycle` warmup (epochs 0-9, LR ramping UP) is harmful for pretrained models
  with new randomly-init modules. The ramp destabilizes pretrained features. The productive
  learning happened during the annealing phase (epochs 11-30, LR decreasing). This explains
  the dip-then-recovery pattern.
  - **Fix:** Use `fit_flat_cos` (flat LR for 75%, cosine anneal for 25%) — no warmup.
    Or cosine annealing without warmup. The LR magnitudes are likely correct; the schedule
    is the problem.
  - **Also:** Save periodic checkpoints (every 3-5 epochs) to capture the recovery phase.
    The late-epoch models might be better than the epoch 2 "best" after full pipeline eval.
  - **Also:** Train longer (50 epochs) — the model was still improving when LR hit near-zero.
- **Notebook:** `notebooks/vesuvius_train_v10b.ipynb` (Cell 18 updated with hardcoded LRs)

### Run 11: 3-Fold Cross-Validation Ensemble (complete — but LR issue, results unreliable)
- **Architecture:** Same SegResNetDSAttn as Run 10 (attention gates + deep supervision)
- **Key change:** 3-fold scroll-grouped cross-validation with probability averaging ensemble
  - Fold 0: hold out scrolls 35360 + 53997 (189 val, 597 train) — largest training set first
  - Fold 1: hold out scrolls 26010 + 26002 + 44430 (235 val, 551 train)
  - Fold 2: hold out scroll 34117 (382 val, 404 train)
- **Early stopping:** patience=5 on comp_score per fold (~10-15 epochs expected vs 30 max)
- **Ensemble inference:** All 3 models predict on each volume, sigmoid probabilities averaged
  before hysteresis thresholding. With TTA: 3 models × 7 TTA = 21 forward passes per volume.
- **Kaggle time estimate:** 120 volumes × 21 passes × 27 patches × ~0.03s ≈ 34 min (well within 9hr)
- **Weights:** ~54 MB total (3 × 18 MB traced models)
- Everything else same as Run 10
- **Results:** Completed overnight (3 folds trained, all traced). However, v11 uses the same
  SegResNetDSAttn architecture and same LR=1e-5 as v10, so it suffers from the **same LR issue**
  — the attention gates and DS heads couldn't learn. Results are unreliable and the technique
  (CV + ensembling) is unverified. Needs to be re-run after v10b's per-module lr_find proves
  out the correct LRs.
- **Notebook:** `notebooks/vesuvius_train_v11.ipynb`

## Overnight Pipeline (Feb 12, 2026)

The overnight automation script (`scripts/overnight_pipeline.sh`) ran:
1. Wait for v10 training to complete (nbconvert PID 8527) — done ~02:12
2. Trace v10 model → upload weights → push Kaggle inference notebook (version 9) — done ~02:32
3. Run v11 training (3-fold CV via nbconvert) — done ~03:31
4. Trace v11 ensemble models — done ~03:31

## Kaggle Notebook Version History
| Kaggle Version | Weights | Date | Status | Notes |
|----------------|---------|------|--------|-------|
| v9 | v10 traced | Feb 12 ~02:32 | ERROR | Overnight pipeline — v10 weights (bad, peaked epoch 0) |
| v10-v11 | — | — | — | Unknown / intermediate |
| v12 | v9 traced | Feb 12 | ERROR | Fixed wheels (cp312) but `cc3d.__version__` crash |
| v13 | v9 traced | Feb 12 | SUBMITTED/PENDING | Fixed cc3d version print — first real v9 submission |
| v14 | v9 traced | Feb 12 | PENDING | "finally submit local v9" — two pending submissions |

Weights dataset uploaded Feb 12 with `--dir-mode zip`: v9 traced + v10 traced + v11 fold
traces + wheels (cp310/311/312 for cc3d + dijkstra3d). Inference script points at
`best_segresnet_v9_traced.pt`, uses Gaussian SWI + logit TTA + hysteresis + surface splitting.

**Issues discovered (overnight pipeline):**
- **V10 submission used bad weights** — v10 peaked at epoch 0 due to LR issue. The overnight
  pipeline submitted v10 weights to Kaggle (version 9) which are worse than v9.
- **Eval inference never ran** — the `eval_inference.py` script (Gaussian SWI + logit TTA +
  surface splitting comparison) was not in the version of the pipeline that ran overnight.
  The script was added to the pipeline after the overnight run started. Need to run manually.
- **Wheels not uploaded** — `kaggle datasets version` was called without `--dir-mode zip`,
  so the `wheels/` folder (cc3d + dijkstra3d) was skipped. Surface splitting in Kaggle
  inference would fail.
- **V11 results unreliable** — v11 used the same SegResNetDSAttn + LR as v10, so all 3 folds
  suffer from the same LR issue. CV + ensembling technique is unverified.

### Morning Checklist
1. **Check v10 submission status:**
   ```bash
   vesuvius/bin/kaggle kernels status mgoldfield/vesuvius-surface-detection-inference
   ```
   If complete, check the public score on the competition page.

2. **Check overnight pipeline log:**
   ```bash
   tail -50 logs/overnight.log
   ```

3. **Review v10 training results:**
   Open `notebooks/vesuvius_train_v10.ipynb` — look at training curves and comp_score.

4. **Check v11 training status:**
   - If v11 is still running: `nvidia-smi` and `tail -20 logs/v11_execution.log`
   - If v11 is done: open `notebooks/vesuvius_train_v11.ipynb` — check fold results summary
     at the bottom of the training cell, and threshold sweep results.

5. **Submit v11 (if results look good):**
   ```bash
   # Update T_LOW in inference script if threshold sweep found a different value
   cp kaggle/kaggle_notebook/vesuvius-inference-v11.py kaggle/kaggle_notebook/vesuvius-inference.py
   # Upload ensemble weights
   vesuvius/bin/kaggle datasets version -p kaggle/kaggle_weights/ -m "v11 ensemble (3-fold CV)"
   sleep 60
   # Push inference notebook
   vesuvius/bin/kaggle kernels push -p kaggle/kaggle_notebook/
   ```

6. **If v10 scores worse than v9:** Consider reverting to v9 weights and submitting v11
   ensemble based on v9 architecture instead.

### Key files created overnight:
- `notebooks/vesuvius_train_v10.ipynb` — v10 training results (executed)
- `notebooks/vesuvius_train_v11.ipynb` — v11 3-fold CV training results (executed)
- `checkpoints/models/best_segresnet_v10.pth` — v10 best checkpoint
- `checkpoints/models/best_segresnet_v11_fold{0,1,2}.pth` — v11 fold checkpoints
- `kaggle/kaggle_weights/best_segresnet_v10_traced.pt` — v10 traced for Kaggle
- `kaggle/kaggle_weights/best_segresnet_v11_fold{0,1,2}_traced.pt` — v11 traced for Kaggle
- `kaggle/kaggle_notebook/vesuvius-inference-v11.py` — ensemble inference script
- `scripts/trace_model.py` — model tracing utility
- `scripts/overnight_pipeline.sh` — automation script
- `libs/topological-metrics-kaggle/` — topometrics library (reinstalled from Kaggle dataset)
- `logs/overnight.log` — pipeline log

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

**Key findings:**
- Gaussian SWI + logit TTA gives **+0.19 mean comp_score** with the same model weights (no retraining).
- Impact is bimodal: 3/5 volumes get a massive boost (~+0.32), 1 unchanged, 1 negligible.
- Surface splitting was impractical (~80 min/volume for Dijkstra3D). Killed and excluded.

**Threshold sweep (new pipeline):**
| T_low | T_high | comp_score |
|-------|--------|-----------|
| 0.35 | 0.80 | **0.570** |
| 0.30 | 0.80 | 0.568 |
| 0.40 | 0.80 | 0.568 |
| 0.45 | 0.80 | 0.566 |
| 0.35 | 0.85 | 0.503 |
| 0.30 | 0.90 | 0.235 |

**T_high=0.80 beats T_high=0.85** across all T_low values. We've been using 0.85 everywhere.
Best overall: **T_low=0.35, T_high=0.80 → comp_score 0.570** (vs 0.498 at default thresholds).

### Implication: Model selection is broken
The training loop evaluates models using a simplified inference pipeline (basic sliding window,
no TTA, simple threshold). But the actual Kaggle submission uses the full pipeline. The model
that looks best during training may not be the best after full inference post-processing.

**Current plan:**
- **Periodic checkpoint saving** — save every N epochs, not just "best" by lightweight metric
- **Post-training eval sweep** — after training, run full pipeline (Gaussian SWI + logit TTA +
  hysteresis + surface splitting) on all saved checkpoints via `eval_inference.py`
- **Pick the true best** checkpoint based on full-pipeline comp_score
- This decouples model selection from training — lightweight metric still useful as directional
  signal during training, but final selection uses the real pipeline

### Implication: Kaggle inference pipeline matters more than model architecture
If the inference pipeline accounts for a 2x comp_score difference, ensuring the Kaggle notebook
uses the correct pipeline (Gaussian SWI + logit TTA) is potentially more impactful than any
model change. Need to verify the Kaggle inference script matches the "new" pipeline exactly.

## Leaderboard Snapshot (Feb 11, 2026)

1,334 teams competing. Deadline: Feb 27, 2026.

| Rank | Team | Score |
|------|------|-------|
| 1 | Vesuvius Team | 0.606 |
| 2 | Vibes & Scrolls Trade-off | 0.594 |
| 3 | Starry | 0.593 |
| 4 | Dieter | 0.593 |
| 5 | DECEM | 0.592 |
| ... | ... | ... |
| 20 | ryches | 0.580 |
| **937** | **Matt Goldfield (us)** | **0.441** |

Gap to 1st: 0.165. Top teams are tightly clustered (0.58–0.61).
Our best public score (0.441) is from Run 3's weights — Run 8's 0.423 was lower despite
better val comp_score (0.562), possibly due to threshold/post-processing differences in
the Kaggle inference notebook vs local evaluation.

## Kaggle Submission Process

Step-by-step guide for submitting to the competition from the CLI.

### Prerequisites
- Kaggle CLI installed in venv: `/home/mongomatt/Projects/vesuvius/vesuvius/bin/kaggle`
- Auth configured: `~/.kaggle/kaggle.json` (username: mgoldfield)
- Competition: `vesuvius-challenge-surface-detection`

### Step 1: Upload model weights as a Kaggle dataset

```bash
# Create upload directory with weights + metadata
mkdir -p kaggle/kaggle_weights
cp checkpoints/models/best_MODEL_NAME.pth kaggle/kaggle_weights/

# Create dataset-metadata.json in the same directory:
# {
#   "title": "Vesuvius MODEL_NAME Weights",
#   "id": "mgoldfield/vesuvius-MODEL_NAME-weights",
#   "licenses": [{"name": "CC0-1.0"}]
# }

# Upload (first time creates, subsequent times use `kaggle datasets version`)
kaggle datasets create -p kaggle/kaggle_weights/ -r zip

# Check it's ready
kaggle datasets status mgoldfield/vesuvius-MODEL_NAME-weights
```

For updating existing dataset with new weights:
```bash
kaggle datasets version -p kaggle/kaggle_weights/ -m "Updated weights from Run X"
```

### Step 2: Create inference notebook

The inference script must be self-contained (no internet on Kaggle). Key requirements:
- **Model definition** must be inline (can't import from local files)
- **TIFF reading** must use Pillow (`PIL.Image`), NOT tifffile — Kaggle test TIFFs
  use LZW compression which requires `imagecodecs` (not available offline)
- **TIFF writing** also use Pillow to avoid dependency issues
- **Paths:**
  - Competition data: `/kaggle/input/vesuvius-challenge-surface-detection/`
  - Weights dataset: `/kaggle/input/YOUR-DATASET-NAME/`
  - Output: `/kaggle/working/submission.zip`
- **GPU on Kaggle:** Tesla P100 (16GB VRAM) — much less than our RTX 5090

Files needed in `kaggle/kaggle_notebook/`:
- `vesuvius-inference.py` — the inference script
- `kernel-metadata.json` — notebook config

kernel-metadata.json template:
```json
{
  "id": "mgoldfield/vesuvius-surface-detection-inference",
  "title": "Vesuvius Surface Detection Inference",
  "code_file": "vesuvius-inference.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": false,
  "competition_sources": ["vesuvius-challenge-surface-detection"],
  "dataset_sources": ["mgoldfield/YOUR-WEIGHTS-DATASET"],
  "kernel_sources": []
}
```

### Step 3: Push and submit

```bash
# Push notebook to Kaggle (runs automatically)
kaggle kernels push -p kaggle/kaggle_notebook/

# Monitor execution status
kaggle kernels status mgoldfield/vesuvius-surface-detection-inference

# Once status is "complete", go to the notebook page and click Submit:
# https://www.kaggle.com/code/mgoldfield/vesuvius-surface-detection-inference
```

### Gotchas
- **No internet** during execution — can't pip install anything
- **LZW TIFFs** — must use Pillow, not tifffile (no imagecodecs)
- **fastai checkpoint format** — weights may be nested under a `"model"` key
- **DataParallel prefix** — some checkpoints have `"module."` prefix on keys
- **P100 has 16GB VRAM** — model + inference must fit (our UNet uses ~8 GB,
  SegResNet uses ~3.5 GB)
- **9 hour time limit** — not a concern for us (~5 min for 120 volumes)
- **MONAI not installed on Kaggle** — use `torch.jit.trace` to export models
  that depend on MONAI. The traced `.pt` file loads with just `torch.jit.load`
  and has no external dependencies.
- **Dataset uploads are slow to start** — `kaggle datasets version` uploads via
  GCS resumable upload. Each file sits at 0% for ~8-9 minutes before progress
  begins. This is normal — don't kill the process. With 5 × 18MB files it takes
  ~45-50 minutes total. Run uploads in the background.
- **Stale upload resume tokens** — Kaggle CLI 2.0 caches resumable upload info
  in `/tmp/.kaggle/uploads/*.json`. If a previous upload was interrupted, the
  stale files cause `KaggleObject.from_dict() got an unexpected keyword argument
  'token'` and the upload hangs at 0% indefinitely. **Fix:** delete the stale
  JSON files: `rm /tmp/.kaggle/uploads/*.json` and retry.
- **Folders skipped without `--dir-mode zip`** — `kaggle datasets version` skips
  subdirectories by default. To include folders (e.g. `wheels/`), you must pass
  `--dir-mode zip`. Without it you'll see "Skipping folder: wheels" in the output.

### Current submission artifacts
- Weights dataset: https://www.kaggle.com/datasets/mgoldfield/vesuvius-unet3d-weights
- Inference notebook: https://www.kaggle.com/code/mgoldfield/vesuvius-surface-detection-inference
- Local files: `kaggle/kaggle_weights/`, `kaggle/kaggle_notebook/`

## External Data Sources

Competition rules: **"Freely & publicly available external data is allowed, including pre-trained models."**

### scrollprize.org — Full Scroll CT Scans
5+ full scrolls publicly available at `dl.ash2txt.org/full-scrolls/`:
- Scroll 1 (PHerc. Paris 4): 2 volumes, 7.91µm, ~3.2 TB
- Scroll 2 (PHerc. Paris 3): 3 volumes, 7.91µm, ~5.1 TB
- Scroll 3 (PHerc. 332): 3 volumes, 3.24-7.91µm, ~8.4 TB
- Scroll 4 (PHerc. 1667): 2 volumes, 3.24-7.91µm, ~3.7 TB
- Scroll 5 (PHerc. 172): 1 volume, 7.91µm, ~2.6 TB
- PHerc. 139: 3 volumes, 2.4-9.4µm, ~7.7 TB (OME-Zarr)
- **No surface labels** — raw CT only. Would need to generate labels from segment meshes.

### scrollprize.org — Segments (Surface Meshes)
Manually mapped surface meshes for Scrolls 1-5 at `dl.ash2txt.org/full-scrolls/*/paths/`.
Each segment has OBJ meshes defining where the papyrus surface is in 3D. Could be
rasterized into binary masks to create additional labeled training data. High effort.

### scrollprize.org — Fragments
9 fragment scans (87 GB to 645 GB each) at 3.24-7.91µm. No competition-format labels.
Fragments from ESRF (May 2025) in OME-Zarr format also available.

### Kaggle Community Datasets
- **nnUNet preprocessed** (91.8 GB): Same 806 competition volumes reformatted for nnUNetv2.
  Not additional data, but indicates nnUNet is a strong baseline used by competitors.
  https://www.kaggle.com/datasets/jirkaborovec/vesuvius-surface-nnunet-preprocessed
- **Snapshot dataset** (26 GB): Archived competition data from before Dec 23, 2025 update.
  Not additional data.

### Pre-trained Models
Allowed by competition rules. Options:
- Models Genesis (3D CT pre-training)
- Med3D (3D medical image pre-training)
- Any publicly available 3D segmentation weights
- **Easiest high-impact improvement** — swap random init for pre-trained encoder.

### Assessment
| Source | Value | Effort |
|--------|-------|--------|
| Pre-trained 3D encoder | High | Low |
| nnUNet framework | High | Medium |
| Segments → extra labels | High | High |
| Raw scroll CTs | Medium | Very high |

## Issues Encountered
- **fastprogress 1.1.3 bug:** `NBMasterBar` missing `out` attribute, crashes `ProgressCallback`.
  Fixed by downgrading to fastprogress 1.0.5.
- **torch downgrade:** pip downgraded torch from 2.10.0+cu130 to 2.9.1 when installing fastai.
  Hasn't caused issues.
- **Missing volume files:** 20 IDs in train.csv don't have corresponding .tif files on disk.
  Dataset/DataLoader code filters these out.
- **Full volume OOM:** 320^3 at any batch size OOMs during training (needs ~53 GB for
  activations). Solved with patch-based training at 128^3.
- **clDice OOM (v3):** Soft skeleton computation builds deep autograd graph (~40 max_pool3d
  ops storing intermediates). Fixed with three strategies: (1) compute skeletons on 2x-
  downsampled patches (64^3, 8x less memory), (2) target skeleton in `torch.no_grad()`,
  (3) reduced iterations from 10 to 5.
- **Stale Jupyter kernels eating GPU:** Multiple old kernels can accumulate on the GPU
  without being visible. Check with `nvidia-smi` — kill stale python processes if GPU is
  full. Use Jupyter Lab's "Running Kernels" panel to shut down old kernels.
- **Kaggle upload hangs (stale resume tokens):** Kaggle CLI 2.0 caches upload state in
  `/tmp/.kaggle/uploads/*.json`. After interrupted uploads, stale files cause
  `from_dict() got an unexpected keyword argument 'token'` and the upload hangs forever
  at 0%. Fix: `rm /tmp/.kaggle/uploads/*.json` and retry. Even without stale files,
  uploads sit at 0% for ~8-9 minutes per file before progress starts (GCS resumable
  upload initiation delay). Don't kill the process — it will eventually start.

## File Structure
```
/home/mongomatt/Projects/vesuvius/
├── notebooks/                  # Training notebooks
│   ├── vesuvius_train.ipynb    #   Run 1 (baseline 3D U-Net)
│   ├── vesuvius_train_v2.ipynb #   Run 2 (GroupNorm + Dropout)
│   ├── vesuvius_train_v3.ipynb #   Run 3 (SuPreM SegResNet + clDice)
│   ├── vesuvius_train_v4.ipynb #   Run 4 (+ Boundary loss + threshold tuning)
│   ├── vesuvius_train_v5.ipynb #   Run 5 (+ Competition metric monitoring)
│   ├── vesuvius_train_v6.ipynb #   Run 6 (+ Discriminative LR + fast metric)
│   ├── vesuvius_train_v7.ipynb #   Run 7 (+ TTA + hysteresis + 160^3)
│   ├── vesuvius_train_v8.ipynb #   Run 8 (+ foreground-biased sampling)
│   ├── vesuvius_train_v9.ipynb #   Run 9 (+ lower LR hardcoded)
│   ├── vesuvius_train_v10.ipynb #  Run 10 (+ deep supervision + attention gates)
│   ├── vesuvius_train_v10b.ipynb # Run 10b (per-module lr_find)
│   ├── vesuvius_train_v11.ipynb #  Run 11 (3-fold CV ensemble)
│   └── refinement/             #   Refinement (learned post-processing)
│       └── vesuvius_train_refinement.ipynb  # RefinementUNet3D training
├── data/                       # Competition data (not in git)
│   ├── train_images/           #   786 .tif volumes (320^3 uint8)
│   ├── train_labels/           #   786 .tif labels (320^3 uint8, values 0/1/2)
│   ├── test_images/            #   1 .tif (placeholder; real test has ~120)
│   ├── train.csv               #   id, scroll_id (806 rows)
│   └── test.csv                #   id, scroll_id (1 row placeholder)
├── kaggle/                     # Kaggle submission artifacts
│   ├── kaggle_notebook/        #   Inference script + kernel metadata
│   └── kaggle_weights/         #   Uploaded model weights (not in git)
├── checkpoints/                # Model weights saved during training (not in git)
├── pretrained_weights/         # SuPreM etc. (not in git)
├── logs/                       # Training run logs
├── NOTES.md                    # This file
├── PRETRAINED_MODELS.md        # Pre-trained model research
├── scripts/                    # Automation scripts
│   ├── trace_model.py          #   Trace SegResNetDSAttn for Kaggle (torch.jit)
│   ├── eval_inference.py       #   Compare inference pipelines on val set
│   ├── generate_refinement_data.py  # Generate probmaps for refinement model training
│   └── overnight_pipeline.sh   #   Overnight: v10 submit → eval inference
├── libs/                       # Third-party libraries (not in git)
│   ├── topological-metrics-kaggle/  # topometrics (Betti matching + scoring)
│   └── killer-ant/             # Surface splitting post-processing
│       ├── process_helper.py   #   Original hengck23 code (reference)
│       └── surface_splitter.py #   Clean wrapper module
├── research/                   # Competition research
│   ├── COMPETITION_NOTEBOOK_ANALYSIS.md  # Analysis of 14 public notebooks
│   └── notebooks/             #   Downloaded Kaggle notebooks
├── CLAUDE.md                   # Claude Code instructions
└── .gitignore
```
