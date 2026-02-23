# Competitor Notebook Research — Feb 22, 2026

Analysis of top-scoring public notebooks (LB >= 0.50) from the Vesuvius Surface Detection competition.

## Key Finding: Everyone uses the same pretrained TransUNet

Every notebook with score >= 0.50 uses the **same** `TransUNet SEResNeXt50 160^3 comboloss` weights — the exact same pretrained model we use. The score differences come entirely from **inference-time techniques** (TTA, PP, ensembling) and a handful of privately-trained weight variants.

---

## Greatest Hits — Novel Techniques Worth Investigating

### 1. Dual-Stream Public-Anchored Hysteresis (Tony Li — LB 0.552)
**Source:** `tonylica/vesuvius-0-552` (451 votes)

The single most impactful inference technique. Runs two parallel inference streams:
- **Public stream** (overlap=0.42): Produces multiclass argmax labels
- **Private stream** (overlap=0.60 for original, 0.43 for TTA views): Produces binary probability via `logsumexp(L1, L2) - L0` (foreground vs background in logit space, then sigmoid)

Hysteresis seeding:
```
strong = (private_prob >= 0.90)
weak   = (private_prob >= 0.50) OR (public_argmax != 0)
mask   = binary_propagation(strong, mask=weak)
```

The `public_argmax != 0` OR-union is the key — argmax catches faint surfaces the probability threshold misses, because argmax only needs one class slightly above others, not an absolute threshold.

**Relevance:** We do NO TTA and no dual-stream currently. This is likely the biggest single gap between our inference and the top competitors.

### 2. Binary Logit Conversion from 3-Class (Tony Li)
Instead of argmax on 3-class output, convert to binary logit:
```python
binary_logit = logsumexp(L1, L2) - L0  # = log(P(fg) / P(bg))
binary_prob = sigmoid(binary_logit)
```
This preserves calibration information that argmax destroys. More principled probability estimation for thresholding/hysteresis.

### 3. 7x TTA (Multiple notebooks — baseline technique for LB >= 0.545)
All top-scoring notebooks use 7-fold TTA:
- Identity
- 3 axis flips (D, H, W separately)
- 3 axial rotations (90°, 180°, 270° in HW plane)

Averaged in **logit space** (not probability space) before softmax. This is standard among top scorers.

### 4. Frangi Filter Post-Processing (Poemcourt — 74 votes)
**Source:** `poemcourt/inference-vesuvius-surface-3d-detection-060959`

Uses a 3D Frangi (Hessian-based) filter to enhance sheet-like structures in the probability map. Multi-scale with sigmas=(1, 3, 5).

Three integration modes explored:
- `enhance_before`: Blend Frangi with probmap before hysteresis: `(1-w)*probs + w*frangi`
- `enhance_after` (active): Run PP first, then enhance binary mask with Frangi
- `replace_hyst`: Replace hysteresis with Frangi-enhanced thresholding

Frangi filters are standard in medical imaging for vessel/sheet enhancement. For papyrus surfaces (locally planar/sheet-like), the Hessian eigenvalue analysis can distinguish sheets from blobs and potentially:
- Enhance faint surface predictions below threshold
- Suppress blob-like false positives
- Bridge small gaps where local Hessian still shows sheet structure

**Relevance:** Novel idea for us. Could be a cheap PP addition. `skimage.filters.frangi` has a proper eigenvalue-based implementation.

### 5. Cascaded Coarse-to-Fine Architecture (Cascaded UNet — 130 votes)
**Source:** `mayukh18/pytorch-cascaded-unet-inference`

Two-stage architecture:
1. **Stage 1:** MONAI UNet → coarse probability map
2. **Stage 2:** MONAI SwinUNETR takes **2 input channels**: original image + Stage 1 output → refined prediction

Full-volume inference via downsampling to 128^3 (no sliding window). No TTA needed.

Post-processing: threshold 0.8 + edge zeroing (outer 5 voxels set to 0) + aggressive CC filtering (min_size=3000, 26-connectivity).

**Relevance:** Architecturally interesting for future work. The idea of using a lightweight refinement model that takes both image + coarse prediction as input is powerful. Edge zeroing is a free cleanup.

### 6. Flat Low Threshold + Aggressive CC Filtering (Vincenzo V13)
**Source:** `vincenzorubino/vesuvius-v13-enhanced-tta-cc`

Instead of hysteresis, uses:
- Single flat threshold: 0.35 (very low)
- 6-connectivity CC filtering with min_size=5000

Philosophy: include everything the model thinks is even slightly foreground, then remove all small fragments. The 6-connectivity (face-only, stricter) finds more separate components, then the large min_size filters aggressively.

**Relevance:** We could test this as a PP variant — our current hysteresis uses T_low=0.70, which is much higher than any competitor.

### 7. UNETR++ Architecture with Topo Loss (Nihil — 80 votes)
**Source:** `jorapro/inference-vesuvius-surface-3d-detection`

Uses **UNETR++** (pure vision transformer) instead of TransUNet. Trained for 100 epochs with topology-aware loss. Very aggressive closing (z=4, xy=4).

**Relevance:** Different architecture = different error patterns = good for ensembling. We don't have time to train a UNETR++ from scratch, but worth noting.

### 8. J17 / TPU Weight Variants (Pankaj, Ibrat)
Multiple notebooks reference privately-trained weight checkpoints:
- `J17/model.weights.h5` (Pankaj — 284 votes)
- `train-vesuvius-surface-3d-detection-on-tpu/model.weights.h5` (TPU variant)
- `colab-a-162v4-gpu-transunet-seresnext101-x160/model.weights.h5` (SEResNeXt101)

These are the same architecture but different training runs, providing diversity for ensembling.

---

## Threshold Landscape Across Competitors

| Notebook | T_low | T_high | Notes |
|----------|-------|--------|-------|
| Tony Li (0.552) | 0.50 | 0.90 | + public argmax union |
| Tao Li (top) | 0.50 | 0.90 | + public argmax union |
| LB:54.9 Tuning | 0.15 | 0.50 | Very aggressive low threshold |
| Pankaj (J17) | 0.50 | 0.90 | Dual-stream |
| HARUKI | 0.30 | 0.80 | Single-stream |
| Hossam | 0.30 | 0.80 | Single-stream |
| Parthenos | 0.30 | 0.80 | 2-model ensemble |
| Ibrat | 0.45 | 0.80 | 2-model ensemble |
| Poemcourt | 0.50 | 0.90 | + Frangi filter |
| Vincenzo V13 | 0.35 (flat) | — | No hysteresis, CC filter |
| **Our current** | **0.70** | **0.90** | **Highest T_low by far** |

Our T_low=0.70 is dramatically higher than all competitors (next highest is 0.50). This suggests we may be too conservative — though our models may have sharper probability distributions from SWA/fine-tuning.

---

## PP Closing Parameters Across Competitors

| Notebook | z_radius | xy_radius | dust_min |
|----------|----------|-----------|----------|
| Tony Li | 3 | 2 | 100 |
| LB:54.9 | 3 | 1 | 150 |
| Parthenos | 3 | 2 | 100 |
| Nihil (UNETR++) | 4 | 4 | 200 |
| Ibrat | 2 | 0 | 300 |
| Poemcourt | 1 | 0 | 100 |
| Innat (author) | 1 | 0 | 100 |
| Cascaded UNet | — | — | 3000 |
| Vincenzo V13 | — | — | 5000 (6-conn) |

---

## Ensemble Strategies Seen

| Notebook | Models | Weights | Fusion |
|----------|--------|---------|--------|
| Tony Li | 1 model, 2 streams | — | Public argmax + private prob |
| Parthenos | 2 TransUNet (standard + comboloss) | 0.20 / 0.80 | Argmax averaging |
| Ibrat | 2 TransUNet (comboloss + TPU) | 0.75 / 0.25 | Probability averaging |

Ibrat's probability-space fusion is more principled than Parthenos's argmax approach.

---

## What Competitors Are NOT Doing (Our Advantages)

None of the public notebooks use:
- **SWA blending** (our best technique, 0.5549)
- **Margin distance loss** (our novel loss design)
- **Pseudo-labeling** for training data expansion
- **clDice / soft-skeletonization loss**
- **Boundary loss**
- **Connectivity-aware post-processing** (gap fill, dilate-merge-erode, two-pass hysteresis)

Our training innovations are genuinely differentiated. The gap is in inference techniques.

---

## Priority Actions

1. **Add 7x TTA to our Kaggle notebook** — free lunch, no training needed
2. **Implement binary logit conversion** (`logsumexp(L1,L2) - L0`) for probability estimation
3. **Test dual-stream inference** with public argmax union for hysteresis
4. **Experiment with Frangi filter** on our probability maps (quick to test)
5. **Sweep lower T_low values** (0.30-0.50) with our SWA model — our 0.70 may be too conservative
6. **Test flat-threshold + aggressive CC filtering** as PP alternative
7. **Consider ensemble of SWA + pretrained** in probability space at inference time
