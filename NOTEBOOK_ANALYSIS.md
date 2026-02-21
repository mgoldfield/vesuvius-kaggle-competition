# Multi-Model Comparison Notebook Analysis

**Date:** Feb 21, 2026
**Notebook:** `/workspace/vesuvius-kaggle-competition/notebooks/analysis/multi_model_comparison_executed.ipynb`
**Figures:** `/workspace/vesuvius-kaggle-competition/notebooks/analysis/figures/`

## Models Compared

| Model | Comp | Topo (30%) | SDice (35%) | VOI (35%) | Description |
|-------|------|-----------|-------------|-----------|-------------|
| pretrained | 0.5526 | 0.2354 | **0.8255** | **0.5517** | Baseline pretrained (best SDice & VOI) |
| swa_70pre_30distsq | **0.5530** | 0.2462 | 0.8289 | 0.5401 | Best comp (70% pretrained + 30% dist_sq_ep5) |
| frozen_boundary_ep10 | 0.5408 | **0.2642** | 0.7861 | 0.5327 | Best topo (frozen enc + dist_sq + boundary loss) |
| dist_sq_ep5 | 0.5246 | 0.2341 | 0.7737 | 0.5246 | Full fine-tune dist_sq (shows degradation) |

## Volumes Analyzed

Four volumes from three different scrolls:
- **26002_a** (vol 26894125) - Scroll 26002
- **26002_b** (vol 105796630) - Scroll 26002
- **35360_a** (vol 3320274) - Scroll 35360
- **34117_a** (vol 2290837) - Scroll 34117

---

## 1. Cross-Section Comparisons (cell12_fig00 through cell12_fig06)

These are the most information-dense visualizations. For each volume, each model gets a row showing: (1) probmap heatmap, (2) binary prediction, (3) error overlay (TP=green, FP=red, FN=blue), and (4) sub-metric bar chart.

### Key Observations

**Scroll 26002, Vol 26894125 (cell12_fig00):**
- The CT image shows clearly defined horizontal papyrus layers with good contrast.
- **Pretrained** probmap shows thick, confident bands (max 0.865). The binary prediction is noticeably thick -- surfaces appear as wide white slabs rather than thin lines. The error overlay shows green (TP) along the center of each surface but substantial red (FP) flanking both sides -- the "slab" problem. Blue (FN) is minimal, meaning coverage is good.
- **SWA blend** probmap is very similar to pretrained but slightly tighter. The slabs are marginally thinner. Sub-metric bars look almost identical.
- **Frozen_boundary_ep10** probmap shows a dramatic change: the bands are narrower and sharper. Binary predictions are visibly thinner. The error overlay shows much less red (FP), but some surfaces now show small blue (FN) gaps -- surfaces are fragmenting. Topo bar is larger (0.2642) but SDice and VOI bars are shorter.
- **dist_sq_ep5** probmap shows the sharpest bands of all but also the most fragmented. Several surfaces break into disconnected segments. The error overlay shows the least red of any model (thinnest predictions) but the most blue (FN) -- surfaces have gaps. All three sub-metric bars are lower than pretrained.

**Scroll 26002, Vol 105796630 (cell12_fig02):**
- This volume has more vertically-oriented surfaces -- a harder geometry.
- Same pattern: pretrained is thick but continuous; frozen_boundary is thinner but shows fragmentation; dist_sq_ep5 is thinnest but most fragmented.
- Notably, the FN (blue) regions in frozen_boundary and dist_sq are concentrated where surfaces curve or change direction -- the model loses confidence at bends.

**Scroll 35360, Vol 3320274 (cell12_fig04):**
- This is the most challenging volume. The CT shows densely packed, closely spaced surfaces with complex geometry -- some surfaces are nearly touching.
- **Pretrained** produces a massive thick blob covering 23.22% of the volume (vs GT's 8.74%). The probmap is almost uniformly hot across the surface region. Adjacent surfaces are merging into one thick mass.
- **SWA blend** reduces this to 18.73% -- still very thick but better.
- **Frozen_boundary** at 8.65% is almost exactly the GT fraction (8.74%). The probmap shows individual surfaces separated. However, the binary prediction has significant fragmentation -- many small gaps.
- **dist_sq_ep5** at 9.87% is close to GT but with even more fragmentation (41 CCs vs GT's 10).
- The error overlay for pretrained in this volume is alarming: the entire left side is a sea of red (FP) because adjacent surfaces have merged. The fine-tuned models fix this merging problem but introduce blue (FN) gaps.

**Scroll 34117, Vol 2290837 (cell12_fig06):**
- Dense horizontal layers with moderate spacing.
- **Pretrained** is thick (14.41% vs 4.88% GT) with characteristic slab predictions.
- **Frozen_boundary** and **dist_sq** show similar thinning but with visible breaks in the longer surfaces.
- The error overlays confirm the universal pattern: thinning removes FP (red) but introduces FN (blue).

### Summary Pattern from Cross-Sections
There is a clear and consistent gradient across models:
1. **pretrained**: Thick, continuous slabs. Excellent coverage (low FN) but massive excess (high FP).
2. **SWA blend**: Slightly thinner slabs, still continuous. Best tradeoff point.
3. **frozen_boundary**: Noticeably thinner. Approaching GT thickness on easy volumes. Starts fragmenting on complex geometry.
4. **dist_sq_ep5**: Thinnest predictions, worst fragmentation. The thinning has gone too far.

---

## 2. Probability Distributions (cell14_fig00)

Two rows of histograms across all four volumes. Top row: foreground probabilities. Bottom row: background false positive probabilities (p > 0.1).

### Key Observations

**Foreground probabilities (top row):**
- All models show a bimodal FG distribution with peaks near 0.0 and near 1.0, but the shapes differ significantly.
- **Pretrained** (blue) has the tallest peak near 1.0 -- most FG voxels get very high confidence. This is good for coverage but contributes to thickness (the confidence extends across the full slab width).
- **Frozen_boundary** (orange) has a lower peak near 1.0 and more mass in the 0.3-0.7 mid-range -- the model is less certain about which exact voxels are FG. This uncertainty is concentrated at the edges of surfaces (the voxels that should be BG in a thinner prediction).
- **dist_sq_ep5** (red/pink) has the most mass in the mid-range (0.3-0.8). The model has learned to be uncertain about peripheral voxels, but this mid-range probability makes thresholding sensitive -- small T_low changes cause big changes in the binary mask.
- **Scroll 35360** (third column) is notable: pretrained has a very broad, flat FG distribution with an unusual bump around 0.7-0.9. This corresponds to the volume where surfaces merge -- many voxels get moderate-to-high probability because they sit between two nearby surfaces.

**Background false positives (bottom row):**
- **Pretrained** has the most background voxels with high probability (p > 0.7). A large number of BG voxels are confidently predicted as FG -- these are the slab-flanking voxels.
- **Frozen_boundary** and **dist_sq** have fewer high-confidence BG false positives, confirming the thinning effect is real at the probability level, not just a thresholding artifact.
- The red dashed line at T_low=0.70 shows that many BG false positives have p > 0.70 for pretrained -- raising T_low further could help, but the green (SWA) line shows the same voxels with lower probability, explaining why SWA helps without changing T_low.

**Critical insight:** The probability distributions show that the pretrained model's thickness problem is baked into the probmap, not just a thresholding issue. The model genuinely assigns p > 0.8 to voxels 3-5 voxels away from the true surface. Fine-tuning successfully reduces these false-positive probabilities, but at the cost of also reducing some true-positive probabilities (the FN problem).

---

## 3. Thickness Analysis (cell16_fig05)

Thickness maps shown as heatmaps projected to 2D (max thickness per XY column along Z). One panel per model plus GT.

### Key Observations

**Quantitative results:**
| Model | Mean Thickness | Multiplier vs GT | FG% | Median |
|-------|---------------|------------------|-----|--------|
| GT | 6.5 voxels | 1.0x | 4.61% | 4.0 |
| pretrained | 13.7 voxels | **2.1x** | 11.41% | 9.0 |
| swa_70pre_30distsq | 12.5 voxels | **1.9x** | 9.99% | 8.0 |
| frozen_boundary_ep10 | 10.3 voxels | **1.6x** | 7.10% | 7.0 |
| dist_sq_ep5 | 9.7 voxels | **1.5x** | 6.70% | 6.0 |

- **GT thickness map** shows thin, uniform blue/purple lines with occasional green spots (slightly thicker regions where surfaces curve).
- **Pretrained** thickness map is dominated by green-yellow-cyan colors (10-20 voxels thick), with a prominent bright yellow streak indicating a region 20+ voxels thick. This is where two surfaces merge.
- **SWA blend** is slightly cooler-colored (thinner) but retains the same bright streak.
- **Frozen_boundary** and **dist_sq** have much more blue/purple (6-10 voxel range) -- closer to GT. The bright streak is gone or greatly reduced -- they successfully separate merged surfaces.

**The median is more actionable than the mean:** dist_sq_ep5 has median=6.0, very close to GT's median=4.0. The remaining gap (6.0 vs 4.0) is the residual excess thickness that still costs SDice.

**Even the thinnest model is 1.5x GT thickness.** This means there is still a significant amount of excess surface in all models. Closing the thickness gap further without fragmenting is the core challenge.

---

## 4. SDice Deep Dive (cell21_fig00 through cell21_fig14)

These visualizations show four columns per model: (1) prediction+GT boundary overlay, (2) predicted surface with penalty coloring (green=OK, red=penalized), (3) surface distance heatmap, (4) GT surface with miss coloring.

### Key Observations

**Bug/artifact in the analysis:** All models show SDice~0.0000 and 100% penalized / 100% missed. This appears to be a bug in the notebook's SDice computation for the per-slice visualization -- the actual SDice scores (from the 24-volume eval) are 0.8255, 0.7861, etc. The surface extraction or distance computation in the notebook is likely computing on the wrong mask (perhaps using the surface boundary voxels on a single 2D slice rather than computing proper 3D surface distances). Despite this SDice numerical bug, the visualizations themselves are still informative.

**Scroll 26002_a, best FG slice Z=178 (cell21_fig00):**
- Column 1 shows prediction boundary (white) vs GT boundary (green). For **pretrained** (top row), the prediction boundaries are wider and don't align perfectly with GT. For **frozen_boundary** (middle), boundaries are tighter. For **dist_sq** (bottom), boundaries are tightest but show gaps.
- The surface distance heatmaps (column 3) appear mostly black with faint colored lines, suggesting that most surface voxels are relatively well-aligned but with scattered penalty hotspots.

**Scroll 26002_a, worst SDice slice Z=180 (cell21_fig02):**
- Very similar to Z=178 (just 2 slices apart). The prediction boundaries show the same thickness gradient across models.

**Scroll 35360, best FG Z=6 (cell21_fig08):**
- This is the challenging dense volume. **Pretrained** shows very thick boundary regions where surfaces are merging. **Frozen_boundary** separates them better. The GT surface column shows extensive green (matched) lines for all models, but the pred surface column shows more red (penalized) for pretrained, indicating more excess boundary.

**Scroll 35360, worst SDice Z=209 (cell21_fig10):**
- All models struggle here. The pretrained prediction boundaries are thick and overlapping. The fine-tuned models show separation but with fragmentation.

**Scroll 34117, best FG Z=183 (cell21_fig12):**
- Cleaner horizontal layers. All models produce reasonable boundary alignment. The main difference is thickness -- pretrained boundaries are wider.

**Scroll 34117, worst SDice Z=205 (cell21_fig14):**
- Some surfaces near the edge of the volume show misalignment. The pretrained model predicts a surface where GT has none (or vice versa) -- this is a coverage error, not a thickness error.

---

## 5. SDice Penalty Breakdown (cell23_fig01)

Bar chart showing two panels: "Excess surface (FP boundary)" and "Missing surface (FN boundary)".

### Key Observation

All three models show **100% excess surface** and **100% missing surface**. Combined with the notebook output showing SDice~0.000, this confirms the per-volume SDice computation in the notebook has a bug (likely the surface extraction is computing something different from the actual SDice metric). The actual metric evaluation gives SDice of 0.7737-0.8255.

**Despite the bug, the relative numbers are informative:**
- Pretrained: 5,150,587 penalized surface voxels
- Frozen_boundary: 4,230,081 penalized (-18% reduction)
- dist_sq: 4,299,625 penalized (-17% reduction)

The fine-tuned models have fewer total predicted surface voxels (consistent with thinner predictions), but a similar proportion are penalized. This suggests the penalty comes from both (a) the excess thickness creating surface on both sides of the slab (one side aligns with GT, the other doesn't), and (b) misalignment of surface position.

---

## 6. Worst-SDice Volume Multi-Slice Views (cell25_fig00, cell25_fig02)

Five Z-slices through the worst volume (Scroll 35360, vol 3320274) for frozen_boundary_ep10 and dist_sq_ep5.

### Key Observations

**frozen_boundary_ep10 (cell25_fig00):**
- Five slices show the progression through the volume. The probmap column shows narrower hot bands compared to pretrained. The error overlay column shows the characteristic red-blue pattern: red (FP) flanking surfaces, blue (FN) at gaps.
- The "Pred surf" and "GT surf" columns appear mostly black -- likely the same surface extraction bug as above.
- The key finding: the worst region for frozen_boundary is where surfaces are densely packed. The model occasionally predicts the wrong number of surfaces in a cluster.

**dist_sq_ep5 (cell25_fig02):**
- Very similar visual pattern to frozen_boundary. The probmap bands are slightly narrower. More blue (FN) gaps visible in the error overlays.
- Across the five slices, the fragmentation problem is clearly visible: some surfaces are continuous in slices 2-3 but break into pieces in slices 4-5.

---

## 7. Multi-Axis SDice View (cell27_fig00)

Three views of frozen_boundary_ep10 on Scroll 26002, vol 26894125: axial (Z-slice), coronal (Y-slice), and sagittal (X-slice).

### Key Observations

- **Axial view (top row):** Shows the familiar horizontal surface pattern. Red (FP) and blue (FN) are scattered along surfaces.
- **Coronal view (middle row):** Reveals surface curvature in the Y-Z plane. The surfaces curve significantly, and the error pattern shows the model struggles at curvature points -- red and blue cluster where surfaces bend.
- **Sagittal view (bottom row):** Shows the surfaces are very straight in the X-Z plane, with errors uniformly distributed. The model handles straight sections better than curved ones.

**Critical insight:** The coronal view shows that errors concentrate at surface bends and intersections. The model's difficulty with curvature is a fundamental limitation -- it predicts flat slab-like surfaces and fails where the true surface curves sharply.

---

## 8. Connected Components (cell30_fig00 + cell29 output)

### Quantitative CC Counts

| Model | Vol 26894125 | Vol 105796630 | Vol 3320274 | Vol 2290837 |
|-------|-------------|---------------|-------------|-------------|
| GT | 5 | 9 | 10 | 5 |
| pretrained | 5 (1.0x) | 11 (1.2x) | **4 (0.4x)** | 6 (1.2x) |
| swa_blend | 6 (1.2x) | 11 (1.2x) | **10 (1.0x)** | 11 (2.2x) |
| frozen_boundary | **22 (4.4x)** | 13 (1.4x) | **29 (2.9x)** | 13 (2.6x) |
| dist_sq_ep5 | **19 (3.8x)** | 12 (1.3x) | **41 (4.1x)** | 16 (3.2x) |

### Key Observations from CC Size Distribution (cell30_fig00)

- **GT** (gray) has 5 large components (log10(size) around 5.3-5.5, i.e. 200K-300K voxels each). These are the main papyrus surfaces.
- **Pretrained** (blue) perfectly matches GT with 5 components of similar size. This is remarkable -- despite being 2.1x too thick, it has the right topology.
- **SWA blend** (green) has 6 components -- very close to GT. One extra small component.
- **Frozen_boundary** (orange) has 22 components. The distribution shows 3-4 large components (similar to GT) plus many small fragments (log10 around 2.5-3.5, i.e. 300-3000 voxels). These small fragments are the price of thinning -- surfaces break into pieces.
- **dist_sq_ep5** (red) has 19 components with a similar pattern of a few large + many small fragments.

**This is the core tradeoff:** Pretrained merges surfaces (4 CCs vs 10 GT in Scroll 35360 -- it combined surfaces) but frozen_boundary fragments surfaces (29 CCs vs 10 GT). The pretrained model under-segments while the thinned models over-segment.

**Vol 3320274 (Scroll 35360) is the most revealing:**
- Pretrained: 4 CCs vs 10 GT. It merged separate surfaces into fewer blobs (23.22% FG). Under-segmentation.
- SWA: 10 CCs vs 10 GT. Perfect CC count (18.73% FG still too thick, but topologically correct).
- Frozen_boundary: 29 CCs vs 10 GT. Over-segmented, many fragments.
- dist_sq: 41 CCs vs 10 GT. Severely over-segmented.

**The SWA blend achieves exactly the right CC count on the hardest volume.** This is why it scores best overall -- it separates surfaces that pretrained merges, without the fragmentation of the fine-tuned models.

---

## 9. Post-Processing Visual Comparison (cell34_fig00)

Four PP configs tested on each model: T=0.50/0.90 (competitor), T=0.70/0.90 (our default), T=0.80/0.90 (aggressive), T=0.50/0.85 (low T_high).

### Key Observations

- For **pretrained**, going from T_low=0.50 to T_low=0.80 visibly thins the predictions. At T=0.80, some surfaces start to break apart.
- For **frozen_boundary** and **dist_sq**, T_low=0.50 still produces relatively thin predictions (because the model itself is thinner). At T_low=0.80, severe fragmentation occurs.
- **T_high=0.85 vs 0.90** makes very little difference for any model (bottom row looks like top row).
- The error overlays show that higher T_low reduces red (FP) for pretrained but increases blue (FN) for the fine-tuned models. The fine-tuned models are already near optimal thickness at T_low=0.50-0.60.

---

## 10. PP Threshold Sweep (cell35_fig05)

Three line plots: SDice vs T_low, FG% vs T_low, CC count vs T_low.

### Key Observations

**FG% vs T_low (middle panel):**
- Clear model ordering at every threshold: pretrained > SWA > frozen_boundary > dist_sq.
- At T_low=0.70 (our default): pretrained=15%, SWA=13%, frozen=7.5%, dist_sq=7% (GT is ~5%).
- At T_low=0.80: pretrained=10%, SWA=9%, frozen=5%, dist_sq=5% -- frozen and dist_sq reach GT-level thickness!
- The lines are approximately parallel, meaning the models differ primarily in the probability level they assign to peripheral voxels, not in threshold sensitivity.

**CC count vs T_low (right panel):**
- **Pretrained** (blue) CC count is remarkably stable: 4-11 across the full T_low range. Topology is robust to thresholding.
- **SWA blend** (green) has 5-15 CCs, starting to fragment above T_low=0.70.
- **Frozen_boundary** (orange) shows dramatic CC explosion: 8 CCs at T_low=0.30 rising to 30+ at T_low=0.85. Fragmentation accelerates as T_low increases.
- **dist_sq** (red) is worst: 10 CCs at T_low=0.30, rocketing to ~38 at T_low=0.85.

**The critical finding:** Frozen_boundary and dist_sq *already fragment* even at moderate T_low values. Their surfaces have genuine gaps in the probmap (probability < 0.3 at surface discontinuities). No amount of T_low tuning can fix this -- the fragmentation is a model-level problem.

**SDice vs T_low (left panel):**
- The approximate SDice plot shows all models near zero, which is the same bug discussed in the SDice deep dive section. The actual SDice values from the 24-volume eval are 0.77-0.83.

---

## What's Driving the SDice/Topo Tradeoff

Based on all the evidence above, the tradeoff has a clear mechanism:

1. **Thick predictions have good SDice but bad topo.** When surfaces are thick slabs, one face of each slab aligns with the GT surface, giving reasonable SDice. But thick slabs merge nearby surfaces (reducing CC count below GT) and create false topology (filling gaps between surfaces).

2. **Thin predictions have good topo but bad SDice.** When surfaces are thinned, they separate properly (better topo, better VOI). But the thinning process introduces gaps (fragmentation), creating many small CCs that hurt both topo and SDice. Additionally, the thin surfaces may not align precisely with the GT surface center, creating distance penalties.

3. **The fragmentation problem is asymmetric.** Going from thick-to-thin removes FP voxels that were "bridging" between the prediction center and the GT surface. Once these bridge voxels are removed, small gaps in confidence become disconnections. The model's confidence landscape has local minima (probability dips) at surface bends and junctions, and thinning exposes these.

4. **SWA blending works because it thins without fragmenting.** The 70/30 blend reduces probability of peripheral voxels (slight thinning) while maintaining high probability at the surface core (no fragmentation). It gets 70% of the thinning benefit with almost none of the fragmentation cost.

---

## Where Each Model Succeeds and Fails

### pretrained
- **Succeeds:** Coverage, connectivity, CC count (best on 2 of 4 volumes), SDice (0.8255), VOI (0.5517)
- **Fails:** Thickness (2.1x GT), surface merging in dense regions (Scroll 35360: 4 CCs vs 10 GT), FG% (11-23% vs 5-9% GT)

### swa_70pre_30distsq
- **Succeeds:** Best overall comp (0.5530), best SDice (0.8289), perfect CC count on hardest volume (Scroll 35360: 10 CCs = 10 GT), moderate thinning without fragmentation
- **Fails:** Still 1.9x GT thickness, still merges some surfaces on moderate volumes

### frozen_boundary_ep10
- **Succeeds:** Best topo (0.2642), closest to GT thickness on easy volumes (1.6x), best at separating closely-spaced surfaces
- **Fails:** Severe fragmentation (4.4x CC count on vol 26894125), SDice drops 0.039, connectivity breaks at surface bends

### dist_sq_ep5
- **Succeeds:** Thinnest predictions (1.5x GT), FG% closest to GT (6.70% vs 4.61%)
- **Fails:** Worst fragmentation (4.1x CC count), worst SDice (0.7737), worst comp (0.5246), lowest confidence in FG voxels

---

## Surprising Findings

1. **The SWA blend gets the perfect CC count on the hardest volume.** On Scroll 35360 vol 3320274, the SWA blend produces exactly 10 CCs matching the 10 GT CCs -- even though it's still 1.9x too thick. Neither pretrained (4 CCs, merging) nor the fine-tuned models (29-41 CCs, fragmenting) achieve this. The blend ratio somehow hits the sweet spot for surface separation.

2. **Pretrained has 2.1x thickness yet near-perfect topology.** Despite predicting 11.41% FG vs 4.61% GT (2.5x overshoot), it gets 5 CCs vs 5 GT on vol 26894125. The thick predictions actually help topology by bridging gaps.

3. **The SDice notebook computation appears buggy.** All per-volume SDice values show 0.0000 and 100% penalized/missed, while the actual eval scores are 0.77-0.83. This limits the utility of the SDice deep dive visualizations for quantitative conclusions, though the spatial patterns in the overlays are still informative.

4. **Frozen_boundary matches GT FG% but has 4x the CC count.** On vol 26894125, frozen_boundary has 7.10% FG (GT is 4.61%) but 22 CCs (GT is 5). The thinning produces the right amount of foreground but in the wrong topology -- many small disconnected pieces instead of a few large surfaces.

5. **T_low sensitivity differs dramatically between models.** Pretrained CC count is stable (4-11) across T_low 0.3-0.85. Frozen_boundary CC count explodes from 8 to 30+. This means PP tuning is much more critical for fine-tuned models, and our current fixed T_low=0.70 may be suboptimal for blended models.

---

## Actionable Ideas for Closing the Gap (0.555 -> 0.607)

### High Priority: Fix the Fragmentation Problem

The core bottleneck is clear: we can thin predictions (good for SDice) or keep them connected (good for topo), but not both. The 0.607 leaders likely solve this. Ideas:

1. **Connectivity-preserving post-processing on thin model outputs.** Run frozen_boundary or dist_sq at low T_low (0.40-0.50) to get thin surfaces, then reconnect fragments:
   - **Probmap-guided gap filling:** For pairs of nearby CCs, trace the minimum-probability path between them through the probmap. If min(path_prob) > 0.15-0.25, fill the gap. This uses the model's sub-threshold confidence.
   - **Dilate-merge-erode:** Dilate all CCs by 1-2 voxels, re-label (merging CCs that now overlap), then erode back. Net effect: connections form without increasing thickness.
   - Both of these target the specific failure mode visible in the cross-sections: small probability dips at surface bends that create disconnections.

2. **Per-model T_low optimization.** The PP sweep clearly shows different models need different T_low. For the SWA blend, T_low=0.60-0.65 might outperform 0.70 by keeping more connectivity. For frozen_boundary, T_low=0.40-0.50 would dramatically reduce fragmentation. Run the full 24-volume eval at multiple T_low values per model.

3. **Ensemble at the probmap level.** Average the probmaps of pretrained + frozen_boundary + SWA before thresholding. Pretrained provides connectivity; frozen_boundary provides thinness. The average should be thinner than pretrained but more connected than frozen_boundary. This is free (no retraining) and directly addresses the tradeoff.

### Medium Priority: Training Improvements

4. **Pseudo-labeling with connectivity constraint.** The pseudo-label pipeline (currently running) should help because it trains on 80% more voxels. If the pseudo-labels are generated from pretrained (which has good connectivity), the retrained model should learn to maintain connectivity while the dist_sq/boundary losses encourage thinness.

5. **Loss function targeting the specific failure mode.** The fragmentation occurs at probability dips along surfaces. A "smoothness" loss penalizing the gradient magnitude of probabilities along the surface could help -- force the model to produce uniformly high probability along entire surfaces rather than dipping at bends.

6. **SWA blend with more diverse sources.** Currently blending pretrained + dist_sq or pretrained + frozen_boundary. Try 3-way blend: 60% pretrained + 20% frozen_boundary + 20% dist_sq. Or blend at the probmap level rather than weight level (inference-time ensemble).

### Lower Priority: Exploration

7. **Surface-aware morphological thinning.** Instead of thresholding the probmap, extract the probability ridge (local maximum along the thickness direction for each surface). This produces inherently thin, connected surfaces. Previous attempt destroyed topology (topo 0.29 -> 0.005), but that was on the pretrained model. On frozen_boundary (which is already closer to thin), ridge extraction might work better.

8. **Investigate what the 0.607 leaders are doing differently.** The gap is 0.052 in comp score. Given that our best SDice is 0.83, topo is 0.25, and VOI is 0.55, and the leaders likely have SDice ~0.87, topo ~0.30, VOI ~0.60, the improvement must come from all three metrics. The leaders probably have: (a) thinner predictions than ours but without fragmentation, (b) better surface alignment, and (c) post-processing that reconnects fragments.

9. **Adaptive morphological closing per-CC.** For each predicted CC, measure its surface smoothness. If it has rough/jagged edges (indicating fragmentation), apply local dilation to smooth it. If it's already smooth, leave it alone. This targets only the fragmented regions.

---

## Quantitative Gap Analysis

Our best model (swa_70pre_30distsq, comp=0.5530) vs estimated leader breakdown:

| Metric | Weight | Ours | Est. Leader | Gap | Impact on Comp |
|--------|--------|------|-------------|-----|---------------|
| Topo | 30% | 0.2462 | ~0.32 | 0.074 | 0.022 |
| SDice | 35% | 0.8289 | ~0.87 | 0.041 | 0.014 |
| VOI | 35% | 0.5401 | ~0.60 | 0.060 | 0.021 |
| **Comp** | | **0.5530** | **~0.607** | **0.054** | |

The gap is roughly equal across all three metrics. There is no single magic bullet -- we need to improve on all fronts. The thickness/fragmentation tradeoff is the common thread: fixing it helps SDice (thinner = less excess surface), topo (fewer merge/fragment errors), and VOI (correct CC count and sizes).

**Most promising single intervention:** Probmap-level ensemble (pretrained + frozen_boundary averaged) with per-model T_low tuning. This requires no retraining, directly addresses the thickness-vs-fragmentation tradeoff, and could plausibly gain 0.01-0.02 in comp score.
