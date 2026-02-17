# Competitor Analysis — Vesuvius Surface Detection

Analysis of 17 public Kaggle notebooks, competition discussions, and related work.
Last updated: Feb 17, 2026.

## Leaderboard Context

- **Top public score:** 0.611 (1,334 teams)
- **Our best public:** 0.431 (v17, single model)
- **Public test set:** Only 1 volume (ID 1407735) — high variance, unreliable
- **Scoring formula:** `0.30*TopoScore + 0.35*SurfaceDice@tau=2 + 0.35*VOI_score`

## The Dominant Approach (0.54–0.61 range)

Every top-scoring public notebook uses essentially the same stack:

| Component | Top Teams | Our Approach |
|-----------|-----------|--------------|
| **Architecture** | TransUNet + SEResNeXt50 (medicai library, Keras/JAX) | MONAI SegResNet, 4.7M params |
| **Formulation** | 3-class (bg/surface/unlabeled) | Binary (v9) or 3-class (v13) |
| **Pretrained** | SEResNeXt50 ImageNet encoder | SuPreM self-supervised |
| **Loss** | SparseDiceCE + 0.75\*SkeletonRecall + 0.50\*FP_Volume | 0.2\*BCE + 0.2\*Dice + 0.3\*clDice + 0.3\*Boundary |
| **Augmentation** | Flips + rotations + intensity shift + 3D CutOut (6 blocks) | Flips + rotations + FG-biased sampling |
| **LR** | AdamW, 5e-5 initial, cosine decay, wd=1e-5 | fit_one_cycle or flat_cos, LR=1e-5 |
| **Patch size** | 160^3 | 160^3 (same) |
| **SWI overlap** | 0.42–0.60 (dual-stream) | 0.50 |
| **TTA** | 7-fold (3 flips + 3 rotations + identity) | 7-fold (same) |
| **Post-processing** | Hysteresis (T_low=0.40–0.50, T_high=0.90), closing (z=3, xy=2), dust (100) | Hysteresis (T_low=0.35, T_high=0.75), closing (z=2, xy=1), dust (64) |
| **Ensemble** | 2+ models, weighted averaging | Single model |

---

## Notebook-by-Notebook Breakdown

### Tier 1: Highest Scorers (0.545+)

#### Tony Li — LB 0.552 (412 votes)
**`competitor_notebooks/tony-li-0552/vesuvius-0-552.ipynb`**

The highest-scoring single-model public notebook. Uses the TransUNet SEResNeXt50 from
the `medicai` library with a **dual-stream inference** approach:

- **Public stream:** SWI at overlap=0.42, averages multiclass logits across 7 TTA, takes
  argmax for class labels
- **Private stream:** SWI at overlap=0.60 for the original orientation only (OV06_MAIN_ONLY),
  overlap=0.43 for TTA augmentations. Converts 3-class logits to binary via
  `logsumexp(L1, L2) - L0`, averages binary logits, applies sigmoid.
- **Seeded hysteresis:** `strong = (private_prob >= 0.90)`, `weak = (private_prob >= 0.50) OR
  (public_argmax != 0)`. The public stream's argmax labels expand the weak region.
- **Post-processing:** Anisotropic closing (z=3, xy=2), dust removal (min_size=100)

**Key insight:** The dual-stream approach uses two different SWI overlaps for different purposes.
The public-anchored weak region is the secret sauce — it captures faint surfaces that the
probability stream alone would miss.

#### LB 0.549 Tuning (111 votes)
**`competitor_notebooks/lb549-tuning/lb-54-9-tuning.ipynb`**

Same TransUNet model but with tuned thresholds for a simpler single-stream path:
- T_low=0.15, T_high=0.50 (much lower than others)
- z_radius=3, xy_radius=1 (smaller xy closing)
- dust_min_size=150 (larger dust removal)
- SWI overlap=0.10 (very low — faster inference)

**Key insight:** With argmax-based predictions, lower thresholds work because the model already
produces fairly binary class labels. Shows the model is robust to SWI overlap.

#### Ensemble — LB 0.546 (144 votes)
**`competitor_notebooks/ensemble-0546/0-546-vesuvius-3d-detection-inference-ensemble.ipynb`**

Ensemble of 2 TransUNet models:
- Model 1: "LB 0.505" with softmax, overlap=0.42, weight=0.20
- Model 2: "LB 0.545" combo loss with raw logits, overlap=0.50, weight=0.80
- Hysteresis: T_low=0.30, T_high=0.80
- Post-processing: closing (z=3, xy=2), dust (100)

**Key insight:** Even 2-model weighted ensemble (20/80) gives solid gains. The better model
gets 80% weight. Different training runs of the same architecture provide useful diversity.

### Tier 2: Strong Baselines (0.50–0.54)

#### Jakup Predictions (99 votes, recent Feb 15)
**`competitor_notebooks/jakup-predictions-6feb/vesuvius-predictions-6feb.ipynb`**

Same TransUNet model with some different choices:
- SWI overlap=0.53
- T_low=0.40, T_high=0.90
- `channel_wise=True` normalization (others use `channel_wise=False`)
- Uses the same dual-stream concept as Tony Li

#### TransUNet Inference — LB 0.537 (354 votes)
**`competitor_notebooks/transunet-inference-0537/inference-baseline-transunet-lb-0-537.ipynb`**

Clean baseline inference with the TransUNet model:
- SWI overlap=0.42, 7-TTA
- Standard post-processing: closing (z=3, xy=2), dust (100)
- Straightforward softmax + argmax path

#### Innat Inference (500 votes — the foundation)
**`competitor_notebooks/innat-inference-3d/inference-vesuvius-surface-3d-detection.ipynb`**

The most-forked notebook. Most high-scoring entries are derivatives of this:
- TransUNet SEResNeXt50 + SegFormer (MiT-B2) dual-model
- 3-class softmax, Gaussian SWI, 7-TTA
- This notebook established the standard post-processing pipeline

### Tier 3: Training Code

#### TransUNet Training — LB 0.537 (200 votes)
**`competitor_notebooks/transunet-train-0537/train-transunet-baseline-lb-0-537.ipynb`**

**This is the most important notebook for understanding the training pipeline.**

- **Framework:** Keras 3 + JAX backend, `medicai` library
- **Architecture:** TransUNet(encoder_name="seresnext50", input_shape=(160,160,160,1), num_classes=3)
- **Data:** TFRecords, 780 training samples, batch_size=1*num_devices
- **Schedule:** Cosine decay, initial LR=5e-5, alpha=0.1 (min LR=5e-6), AdamW wd=1e-5
- **Training:** 25 epochs fine-tuning from checkpoint
- **Loss — SkeletonRecallPlusDiceLoss:**
  - `base_loss = SparseDiceCELoss` (Dice + CrossEntropy, ignoring class 2)
  - `skel_loss = SkeletonRecall` — skeletonize GT, dilate by 1, penalize missed skeleton voxels
  - `fp_loss = FP_Volume` — penalize predictions on background voxels
  - **Final = base_loss + 0.75\*skel_loss + 0.50\*fp_loss**
- **Augmentation:**
  - Random flips (all 3 axes, p=0.5)
  - Random 90-degree rotations (axes 0,1, p=0.4)
  - Random intensity shift (+/-0.15, p=0.5)
  - **3D Random Occlusions (CutOut):** Up to 6 random cuboid blocks (size 2-8 voxels), p=1.0
- **Validation:** SlidingWindowInference callback every 10 epochs

#### Innat Training on TPU (242 votes)
**`competitor_notebooks/innat-train-tpu/train-vesuvius-surface-3d-detection-on-tpu.ipynb`**

TPU-distributed training for the TransUNet model. Shows the data pipeline (TFRecords)
and multi-device setup via `medicai`.

#### Jirka 3D Segmentation (319 votes)
**`competitor_notebooks/jirka-3d-segm-gpu-augment/surface-train-inference-3d-segm-gpu-augment.ipynb`**

MONAI-based training (closer to our stack):
- MONAI UNet, 2-class output
- GPU-accelerated MONAI transforms (RandFlip, RandRotate90, RandGaussianNoise, etc.)
- DiceCELoss + TverskyLoss, LR=2e-3
- Pre-converted .npy volumes for faster I/O

### Tier 4: Post-Processing Techniques

#### Killer Ant — Surface Splitting (41 votes)
**`competitor_notebooks/killer-ant-postprocessing/release-killer-ant-post-processing.ipynb`**

**Directly targets TopoScore (30% of metric).**

When connected components contain multiple merged surfaces:
1. Use raycasting from different seed points to detect multi-surface regions
2. Split using Dijkstra 3D shortest paths
3. Recursively split (up to depth 50)

Dependencies: `dijkstra3d`, `connected-components-3d` (cc3d)

**Relevance:** Could improve our TopoScore by separating fused surfaces. We already have
a placeholder in our Kaggle notebook (`USE_SURFACE_SPLIT=False`).

#### Line Tracing — Hole Filling (46 votes)
**`competitor_notebooks/line-tracing-filling-holes/demo-for-line-tracing-for-filling-holes.ipynb`**

Fills holes in surface predictions:
1. Skeletonize prediction, find endpoints via convolution kernels
2. Hungarian matching (`linear_sum_assignment`) to pair endpoints
3. Draw Gaussian-smoothed lines between matched endpoints

**Relevance:** Complementary to killer-ant — this fills holes while killer-ant splits fused
surfaces. Could improve SurfaceDice (fewer missed voxels).

### Tier 5: Alternative Approaches

#### Cascaded UNet (129 votes)
**`competitor_notebooks/cascaded-unet-inference/pytorch-cascaded-unet-inference.ipynb`**

Two-stage approach:
- Stage 1: MONAI UNet (channels=[16,32,64,128]) → coarse segmentation
- Stage 2: SwinUNETR (2 input channels: image + stage1 sigmoid) → refinement
- Downsamples to 128^3 (no SWI needed), upsamples after inference
- Threshold 0.8, CC filtering (min 3000 voxels)

#### nnUNet (233 votes)
**`competitor_notebooks/nnunet-training-inference/surface-nnunet-training-inference-with-2xt4.ipynb`**

Full nnUNetv2 pipeline with ResEnc planner. Auto-configures everything. Handles ignore
label natively. Strong baseline but less customizable than the TransUNet approach.

#### 2nd Place Patch Segmentation (69 votes)
**`competitor_notebooks/2nd-place-3d-patch/2nd-place-advanced-3d-patch-segmentation.ipynb`**

Simple custom 3D UNet, 64x128x128 patches, Dice+Focal loss, only 3 epochs. Early
competition baseline.

### Reference: Scoring Implementation

#### Topology-Aware Scoring (85 votes)
**`competitor_notebooks/topology-aware-scoring/replicate-lb-score-topology-aware-3d-surface-seg.ipynb`**

Complete reference implementation of the competition metric:
- **TopoScore:** Betti numbers B0 (components), B1 (tunnels), B2 (cavities) with
  weights 0.34/0.33/0.33. Computed in 2x2x2 octant tiles.
- **SurfaceDice:** `surface_distance` library, tolerance=2.0 voxels
- **VOI:** `skimage.metrics.variation_of_information` on 3D CC labelings (26-connectivity).
  `voi_score = 1 / (1 + voi_total)`. Split = over-segmentation, merge = under-segmentation.

---

## Key Techniques We Should Adopt

### 1. Confidence-Based CC Filtering (FREE — post-processing only)

From OverthINKingSegmenter (7th place ink detection):
Instead of only removing components < N voxels, also remove components where the
95th percentile probability < threshold (e.g., 0.8). Directly reduces fragmentation
→ improves VOI score (35% of our metric).

### 2. 3D CutOut Augmentation (EASY — training change)

Up to 6 random cuboid blocks (size 2-8 voxels) zeroed out during training (p=1.0).
Strong regularizer that prevents reliance on local context. Every top training notebook
uses this.

### 3. Skeleton Recall Loss (MODERATE — training change)

Skeletonize 3D GT, dilate by 1, penalize missed skeleton voxels. Targets connectivity
and topology directly. Combined with FP_Volume loss for balance:
`final = SparseDiceCE + 0.75*SkeletonRecall + 0.50*FP_Volume`

### 4. Dual-Stream Inference (MODERATE — inference change)

Run two SWI passes at different overlaps. Use the lower-overlap argmax as a weak-region
anchor for seeded hysteresis. Requires the model to produce high-confidence probabilities
(T_HIGH=0.90 needs probs > 0.90, which our model doesn't achieve).

### 5. Model Ensemble (MODERATE — needs multiple models)

Average 2-3 models' probabilities before post-processing. Even same-architecture models
with different training runs help. Weighted averaging (favor better model) outperforms
equal weights.

### 6. TransUNet Architecture (LARGE — framework change)

The dominant architecture. Requires `medicai` library (Keras/JAX) or a PyTorch reimplementation.
See separate section below.

---

## Architecture Deep Dive: TransUNet + SEResNeXt50

### What is it?

TransUNet combines a convolutional encoder (SEResNeXt50, pretrained on ImageNet) with a
transformer bottleneck and a U-Net-style decoder. The `medicai` library provides the
implementation for Keras 3 with JAX backend.

### Why is it better than SegResNet for this task?

1. **Much larger model** — SEResNeXt50 encoder alone has ~25M params vs our total 4.7M
2. **ImageNet pretraining** — Rich low-level features transfer well to CT texture recognition
3. **Squeeze-and-Excitation blocks** — Channel attention helps select relevant features
4. **Transformer bottleneck** — Global context helps with long-range surface continuity

### The `medicai` library

- Provides TransUNet, SegFormer, and other medical imaging architectures for Keras 3
- Built for JAX/TPU training (many competitors train on Kaggle TPUs)
- Includes SlidingWindowInference, loss functions, augmentation transforms
- Available on Kaggle as a dataset/wheel for offline installation
- **NOT a standard PyPI package** — installed from wheel files

### Adoption options

See detailed analysis in the TransUNet adoption research (separate investigation).
Summary: (a) use their Keras/JAX stack directly, (b) reimplement in PyTorch, or
(c) use the pretrained weights from Kaggle and focus on inference/fine-tuning only.

---

## Score Component Analysis

Understanding what each metric rewards helps prioritize improvements:

| Metric | Weight | What it measures | Our weakness |
|--------|--------|-----------------|-------------|
| **TopoScore** | 30% | Betti number matching (components, tunnels, cavities) | Moderate — our topology is reasonable |
| **SurfaceDice** | 35% | Boundary agreement within 2-voxel tolerance | Our refinement model improves this (+0.033) |
| **VOI** | 35% | Connected component consistency (split/merge entropy) | **Our biggest weakness** — refinement destroys this (-0.131) |

**Priority:** Anything that reduces fragmentation (fewer spurious small components)
directly improves VOI, which is 35% of our score. Confidence-based CC filtering and
TV smoothing are the cheapest paths to VOI improvement.

---

## Score Progression Reference (from competitor notebooks)

- SegFormer MiT-B2 128^3: ~0.486
- TransUNet SEResNeXt50 128^3: ~0.500
- TransUNet 160^3: ~0.505
- + combo loss + TTA + PP: ~0.537–0.545
- + dual-stream + seeded hysteresis: ~0.549–0.552
- + ensemble (2 models): ~0.546
- Top of leaderboard: 0.611 (techniques unknown, likely larger ensemble + private tricks)
