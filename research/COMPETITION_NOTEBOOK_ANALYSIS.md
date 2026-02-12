# Competition Notebook Analysis — Vesuvius Surface Detection

Analysis of 14 public Kaggle notebooks for the Vesuvius Challenge Surface Detection competition.
Performed Feb 12, 2026.

---

## Executive Summary

The top public notebooks cluster around **TransUNet with SE-ResNeXt50 encoder** (from the `medicai` library), achieving 0.537–0.552 on the public LB. The dominant architecture is a 3D CNN-Transformer hybrid with a **3-class formulation** (bg/fg/unlabeled), **Gaussian-blended sliding window** inference, **7-fold TTA**, and **hysteresis thresholding**. Our SegResNet approach (val comp_score 0.570) is competitive with these architectures, suggesting our gap to the leaderboard (public 0.441) is primarily in **inference engineering and post-processing**, not model quality.

### Key Takeaways for Our Pipeline

| Technique | Impact | Effort | Status |
|-----------|--------|--------|--------|
| Gaussian-weighted sliding window blending | High | Low | **Not implemented** |
| Logit-space TTA averaging (before sigmoid) | Medium | Low | **Not implemented** |
| 3-class formulation (predict unlabeled) | Medium | Medium | In roadmap |
| Killer ant surface splitting (post-proc) | High | Medium | **New discovery** |
| 3D CutOut augmentation | Medium | Low | **Not implemented** |
| Skeleton recall loss | Medium | Medium | **New discovery** |
| Higher SWI overlap (50%+) | Low | Free | Already at 50% |
| TransUNet / Transformer bottleneck | Medium | High | Not planned |

---

## Notebook-by-Notebook Analysis

### 1. Tony Li — "Vesuvius 0.552" (LB: 0.552, 363 votes)

**Architecture:** TransUNet + SE-ResNeXt50 encoder (3D), Keras/JAX via `medicai`
- Input: 160^3, 3-class softmax output
- Single model (not ensemble)

**Key innovations:**
- **3-class formulation** — predicts bg/fg/unlabeled as 3 classes. Binary foreground probability derived via `logsumexp(L1, L2) - L0` in logit space. This is mathematically cleaner than masking.
- **Dual-stream public-anchored inference** — runs 3 different SWI overlap levels simultaneously:
  - "Public" stream (overlap=0.42) reproduces a known 0.55 LB score
  - "Private" stream (overlap=0.60 for main pass, 0.43 for TTA passes)
  - Public stream's foreground predictions **expand the weak region** in hysteresis, acting as a prior
- **Logit-space TTA averaging** — averages raw logits before softmax (more principled than probability averaging)
- **Hysteresis:** T_high=0.90, T_low=0.50, public-anchor expansion
- **Post-processing:** Anisotropic closing (z=3, xy=2) + dust removal (100 voxels)

**Takeaway:** The score comes from inference engineering, not architecture. Same model as the 0.545 notebooks but with smarter overlap/anchoring.

---

### 2. Ensemble 0.546 (LB: 0.546, 143 votes)

**Architecture:** 2x TransUNet + SE-ResNeXt50, weighted 20%/80%
- Model 1: standard loss, softmax activation (LB 0.505 solo)
- Model 2: combo loss, no activation/raw logits (LB 0.545 solo)
- Ensemble: 0.546 (+0.001 over best single model)

**Key details:**
- Weighted probability averaging (80% to better model)
- 7-fold TTA per model (14 forward passes total per volume)
- Same post-processing as Tony Li: hysteresis (T_low=0.30, T_high=0.80) + closing (z=3, xy=2) + dust (100)
- Score progression: SegFormer 0.486 → TransUNet 128^3 0.500 → 160^3 0.505 → +comboloss 0.545 → +ensemble 0.546

**Takeaway:** Ensemble gave minimal gain (+0.001). The big jumps came from 3-class formulation (+0.005), larger patches (+0.005), and combo loss + TTA + postproc (+0.040).

---

### 3. Innat — Inference (LB: 0.545, 494 votes, most upvoted)

**Architecture:** TransUNet + SE-ResNeXt50, single model, Keras/JAX
- Same model as notebooks #1 and #2 (this is the original author of `medicai`)
- 160^3 input, 3-class, combo loss

**Inference pipeline:**
- SWI: Gaussian blending, 50% overlap
- TTA: 7-fold (3 flips + 3 rotations), logit-space averaging
- Hysteresis: T_low=0.50, T_high=0.90
- Closing: z=1, xy=0 (Z-only!)
- Dust: 100 voxels

**Takeaway:** This is the "template" that most top-scoring public notebooks fork from.

---

### 4. TransUNet Training Baseline (LB: 0.537, 352+198 votes)

**Architecture:** Same TransUNet + SE-ResNeXt50

**Training details (the most informative notebook for training specifics):**
- **Loss:** `SkeletonRecallPlusDiceLoss` = DiceCE + 0.75*SkeletonRecall + 0.5*FPVolumePenalty
  - Skeleton recall: computes morphological skeleton of GT, dilates 1x, penalizes missed skeleton voxels
  - FP volume: penalizes predicted foreground on known-background voxels
- **Optimizer:** AdamW, LR=5e-5 cosine decay to 5e-6, WD=1e-5
- **Epochs:** 25 (fine-tuning from a prior checkpoint)
- **Augmentations:** 3-axis flips, rotate90, intensity shift, **3D cuboid cutout** (up to 6 random cuboids zeroed out)
- **Data:** TFRecords, 160^3 random crops, ~780 train + 1 shard val
- **TPU training:** Keras/JAX, data parallel across 8 TPU cores
- **TTA:** 4x rotation only (not 7-fold)
- **SWI:** 50% overlap, Gaussian

**Takeaway:** The **skeleton recall loss** is novel and directly targets TopoScore. The FP volume penalty prevents over-prediction. 3D cuboid cutout is a cheap, effective regularizer.

---

### 5. LB 0.549 Tuning (109 votes)

Fork of the Innat template with tuned post-processing:
- Same TransUNet + SE-ResNeXt50, same weights
- SWI overlap=0.10 (very low!), 7-fold TTA
- Hysteresis: T_low=0.15, T_high=0.50
- Closing: z=3, xy=1
- Dust: 150 voxels
- Score progression: 0.486 → 0.500 → 0.505 → 0.545 → 0.549

**Takeaway:** Post-processing thresholds matter a lot. Different overlap/threshold combos swing scores by ~0.004.

---

### 6. Jakup — Feb 6 Predictions (93 votes)

Another fork of the Innat template:
- Same model + weights
- Key difference: **SWI overlap=0.51** (vs 0.10 in notebook #5)
- Hysteresis: T_low=0.30, T_high=0.90
- Closing: z=3, xy=2

**Takeaway:** Demonstrates that overlap is a critical tuning knob. 51% overlap vs 10% gives smoother predictions.

---

### 7. Cascaded UNet (votes: 129)

**Architecture:** Two-stage cascade:
- Stage 1: MONAI UNet (16→32→64→128 channels, num_res_units=3)
- Stage 2: MONAI SwinUNETR (2 input channels: image + Stage 1 sigmoid output)

**Key details:**
- **Downsamples to 128^3** (from 320^3) for full-volume inference — loses spatial detail
- Binary output (1 class), not 3-class
- Threshold=0.8, CC filter min 3000 voxels, 5px border zeroing
- No TTA
- Fold-based ensemble supported but configured for 1 fold

**Takeaway:** The cascaded coarse-to-fine approach is interesting but the aggressive downsampling hurts. Our patch-based approach is better.

---

### 8. Pankaj's Inference (LB: ~0.55, 271 votes)

Nearly identical to Tony Li's 0.552 — same dual-stream public-anchored approach:
- Same model (TransUNet + SE-ResNeXt50)
- Same 3-overlap-level SWI (0.42, 0.43, 0.60)
- Same public-anchored hysteresis
- Same post-processing

**Takeaway:** Confirms that Tony Li's approach is reproducible and robust.

---

### 9. Jirka — 3D Segm with GPU Augment (315 votes)

**Architecture:** MONAI SegResNet (same family as ours!)
- init_filters=16, dropout=0.2, 2-class softmax
- Resizes full volumes to 160^3 (not patch-based)

**Training:**
- **Loss:** DiceCE + Tversky(alpha=0.7, beta=0.3) — emphasizes recall
- AdamW, LR=2e-3, cosine annealing, 20 epochs (early stopping patience=10)
- **Gradient accumulation:** 18 steps (effective batch=18)
- GPU-accelerated augmentations via MONAI `on_after_batch_transfer`
- Augmentations: flips, rotation, intensity shift, Gaussian noise

**Takeaway:** Our architecture with higher effective batch size via gradient accumulation. The Tversky loss with alpha=0.7 for recall emphasis is worth noting.

---

### 10. Jirka — nnUNet (221 votes)

**Framework:** nnUNetv2 with `nnUNetPlannerResEncM` (ResNet encoder, medium)
- Configuration: `3d_lowres`
- Auto-configures everything (patch size, architecture, augmentations, LR)
- SGD + Nesterov momentum, polynomial LR decay
- 100 epochs, fold="all" (no CV)
- Custom TIFF I/O (no NIfTI conversion)
- Pre-processed data available as 91.8 GB Kaggle dataset

**Takeaway:** nnUNet is a strong self-configuring baseline. The pre-processed dataset makes it easy to try. Could be a high-impact experiment.

---

### 11. Jirka — Topology-Aware Scoring (85 votes)

**Not a model notebook** — implements the competition scoring metric from scratch:
- SurfaceDice@tau=2 (binary erosion + distance transform)
- VOI score (connected component variation of information)
- TopoScore (Betti numbers via Euler characteristic, F1 per Betti number)
- Includes test cases and volume analysis utilities

**Takeaway:** Useful as a second implementation to cross-validate our `topometrics` library.

---

### 12. Killer Ant Post-Processing (39 votes, by hengck23)

**This is the most strategically valuable notebook.** Not a model — a post-processing algorithm.

**Problem solved:** When multiple papyrus sheets are stuck together in the prediction, they form one connected component instead of separate surfaces. This directly hurts TopoScore (wrong beta_0 count) and VOI (wrong instance labels).

**Algorithm: "Marching Ants" recursive surface splitting:**
1. Threshold probability map at 0.3
2. Find connected components (3D)
3. For each component, check if it contains multiple surfaces via **raycasting** — cast rays through the volume and detect if foreground is hit at two distinct Z-ranges
4. If multi-surface detected: find seed points on each surface, then **split using Dijkstra3D shortest paths** between the seeds
5. Recurse (max depth=50) until each component is a single surface

**Dependencies:** `cc3d` (fast 3D connected components), `dijkstra3d` (3D shortest paths), custom `process_helper.py`

**Takeaway:** This is **model-agnostic** — can be applied on top of ANY model's probability output. Directly fixes topology errors that our model produces. The raycasting heuristic for detecting merged sheets is clever and physically motivated (papyrus layers are roughly Z-aligned). This should be high priority to integrate.

---

### 13. Innat — TPU Training (235 votes)

**Architecture:** SegFormer with MiT-B0 encoder (smallest variant), 3-class softmax
- Keras/JAX, TPU v3-8 data parallel (batch_size=8)
- 128^3 random crops

**Training:**
- **Loss:** DiceCE + clDice (50 iterations! vs our 5)
- AdamW, LR=3e-4 cosine decay with 5% warmup, WD=1e-5
- 200 epochs
- **Augmentations:** 3-axis flips, rotate90, rotate, **3D CutOut** (5 cuboids of 32^3, prob=0.8), intensity shift
- Smart cropping: `min_valid_ratio=0.5` — rejects crops with >50% unlabeled voxels
- Validation SWI every 5 epochs (50% overlap, Gaussian)
- No pre-trained weights (from scratch)

**Notable recipes (documented but not active):**
- Deep supervision with nnUNet-style exponentially decaying weights
- Skeleton Recall Loss (described but not used in this config)

**Takeaway:** **clDice with 50 iterations** produces much better skeletons than our 5 iterations. 3D CutOut is a strong regularizer. The `min_valid_ratio` smart cropping is a nice alternative to our foreground-biased sampling.

---

## Architecture Comparison

| Architecture | Best LB | Params (approx) | Framework |
|-------------|---------|-----------------|-----------|
| TransUNet + SE-ResNeXt50 | **0.552** | ~30-60M | Keras/JAX |
| SegFormer + MiT-B2 | 0.486 | ~25M | Keras/JAX |
| MONAI SegResNet (16 filters) | not reported | ~4.7M | PyTorch |
| nnUNet ResEncM | not reported | auto | PyTorch |
| Cascaded UNet + SwinUNETR | not reported | ~40M+ | PyTorch |
| **Ours: SegResNetDSAttn** | **0.570 val** | ~4.7M | PyTorch/fastai |

Our val comp_score (0.570) would rank well among these, but our public LB (0.441) lags — the gap is in inference engineering and post-processing, not model capability.

---

## Common Patterns Across Top Notebooks

### What everyone does:
- 160^3 patch/input size
- 3-class formulation (bg/fg/unlabeled)
- Gaussian-weighted sliding window inference
- TTA (4-7 fold: flips + rotations)
- Hysteresis thresholding
- Anisotropic morphological closing (Z > XY)
- Dust removal (connected components < 100-150 voxels)

### What differentiates the top scores:
- SWI overlap tuning (10%-60%, major impact on smoothness)
- Logit-space vs probability-space TTA averaging
- Dual-stream public-anchored hysteresis (Tony Li's key insight)
- Post-processing threshold tuning (T_low, T_high, closing radii, dust size)
- Loss function design (skeleton recall, FP penalty)

---

## Prioritized Ideas for Our Pipeline

### Immediate (no retraining needed, apply to v10/v11 outputs):
1. **Gaussian-weighted SWI blending** — replace our simple overlap averaging. Should reduce edge artifacts.
2. **Logit-space TTA averaging** — average raw model outputs before sigmoid, not after.
3. **Post-processing threshold sweep** — systematically try different T_low, T_high, closing radii on val set.

### Short-term (moderate effort):
4. **Killer ant surface splitting** — integrate raycasting + Dijkstra splitting into our post-processing pipeline. Model-agnostic, directly targets TopoScore + VOI.
5. **3D CutOut augmentation** — zero out random cuboid regions during training. Simple to implement.
6. **Skeleton recall loss** — add skeleton-based recall term to our loss function.

### Medium-term (requires retraining):
7. **3-class formulation** — predict bg/fg/unlabeled instead of binary + mask.
8. **Increase clDice iterations** — try 20-50 iterations (currently 5) on downsampled patches.
9. **nnUNet experiment** — use pre-processed 91.8 GB dataset, auto-configured baseline.
10. **Gradient accumulation** — increase effective batch size (currently 2) to 8-16.
