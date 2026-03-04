# Vesuvius Challenge - Surface Detection: Top Solutions Survey

**Competition:** [Vesuvius Challenge - Surface Detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection) | $200K prize pool | 1,329 teams | Ended Feb 13, 2026

**Task:** Detect papyrus surfaces in 3D CT scans (320^3 uint8 volumes). Metric: `0.30*TopoScore + 0.35*SurfaceDice@tau=2 + 0.35*VOI_score`

## The Headline Finding

**Every top team used nnU-Net.** Custom architectures (TransUNet, SwinUNETR, etc.) were not competitive. Post-processing was the primary differentiator — contributing +0.02 to +0.035 improvement, often more impactful than model changes.

## Top Solutions

| Rank | Private Score | Architecture | Epochs | Key Post-Processing |
|------|-------------|-------------|--------|---------------------|
| 1st | **0.627** | nnUNet (128 patch, fine-tuned to 256) | 4000 | Height map interpolation, 1-voxel LUT hole filling |
| 3rd | 0.620 | nnUNet M Cascade 3D | 8000 | Hessian ridge detection + EDT dilation |
| 4th | 0.619 | nnUNet 7-stage ResEnc | 2000 | PCA hole filling, Betti number topology hack |
| 7th | 0.618 | nnUNet M 3d_lowres | 2000 | Frangi sheetness filter + CED diffusion |
| 10th | 0.614 | ResEnc-L + Primus-B | ~400 | 5-stage refinement pipeline, diffeomorphic network |
| 18th | 0.609 | nnUNet x3 Specialist | 500 | Quality-based routing, median filter x7 |

*2nd, 5th, 6th, 8th, 9th place writeups not publicly available as of March 2026.*

## Detailed Breakdowns

### 1st Place (0.627) — nnUNet + Height Map Interpolation

[Writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/1st-place-solution-for-the-vesuvius-challenge-su)

- Base nnUNet trained at 128^3 patch size, then fine-tuned at 256^3
- 4000 epochs (vs. baseline ~1200) — reported +0.06 improvement from extended training alone
- Post-processing: height map interpolation to fill holes, 1-voxel lookup table for gap repair
- Loss: CE + Dice (standard nnUNet defaults)

### 3rd Place (0.620) — nnUNet Cascade + Hessian Ridge

[Writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/quick-preview-of-the-3rd-place)

- nnUNet M variant with cascade 3D configuration (coarse-to-fine)
- 8000 epochs (longest training of any top team)
- Post-processing: Hessian eigenvalue-based ridge detection to identify surface ridgelines, then Euclidean Distance Transform (EDT) dilation to thicken them back to valid surfaces
- Hysteresis thresholding with dual thresholds

### 4th Place (0.619) — 7-Stage ResEnc + Topology Hack

[Writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/4-th-place-solution)

- nnUNet with 7-stage Residual Encoder (deeper than default)
- 2000 epochs
- Post-processing: PCA-based hole filling (project predictions into principal component space to interpolate missing surface patches), plus Betti number topology correction
- The topology hack specifically targeted the TopoScore component of the metric

### 7th Place (0.618) — Frangi Sheetness + Coherence-Enhancing Diffusion

- nnUNet M with 3d_lowres configuration
- 2000 epochs
- Post-processing: Frangi sheetness filter (adapted from vessel detection to detect sheet-like structures), combined with Coherence-Enhancing Diffusion (CED) to smooth noisy predictions while preserving sheet geometry
- Explicitly noted that **pseudo-labeling failed** — "noise too high, performance degradation"

### 10th Place (0.614) — Multi-Model + Diffeomorphic Refinement

[Writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/10th-place-solution)

- Ensemble of ResEnc-L and Primus-B architectures
- Only ~400 epochs (shortest training), suggesting architecture diversity mattered more than extended training
- Post-processing: 5-stage refinement pipeline including a learned diffeomorphic network for smooth surface deformation
- Most architecturally novel approach in the top 10

### 18th Place (0.609) — Specialist Routing

- Three separate nnUNet models, each specialized for different volume quality levels
- Quality-based routing: classify each test volume, then route to the appropriate specialist model
- Post-processing: repeated median filtering (7 passes) for smoothing
- 500 epochs per specialist

## Universal Patterns Across Top Teams

### What worked

- **nnUNet v2** as the backbone — no top team used a non-nnUNet primary model
- **Extended training** (2000-8000 epochs vs. baseline 1200)
- **CE + Dice loss** — standard combination; skeleton recall and clDice provided stability but weren't decisive
- **Sophisticated post-processing** — the main differentiator between teams
- **Hole/gap filling** — every top team had some form of surface completeness repair

### What didn't work

- **Pseudo-labeling** — at least the 7th place team explicitly reported it degraded performance due to noise
- **De-adhesion** (separating touching/merged sheets) — described as unsolved across all teams
- **2D approaches** — significantly underperformed 3D methods
- **Non-nnUNet architectures** — custom models couldn't match nnUNet's auto-configured training

## Implications for Our Approach

Looking back at our competition setup (TransUNet, 0.506 private score):

1. **Architecture gap was real.** We used TransUNet; every top team used nnUNet. The auto-configuration and medical imaging optimizations in nnUNet were decisive for this task.
2. **Post-processing was where most points came from.** Our hysteresis thresholding and close-erode PP were on the right track but simpler than the top teams' approaches (height map interpolation, Hessian ridge detection, PCA hole filling, diffeomorphic refinement).
3. **Training duration mattered.** Top teams trained 2000-8000 epochs; we trained far fewer.
4. **Our SWA blending insight was unusual** — no top team mentioned SWA. They had enough training epochs that they didn't need it.
5. **Pseudo-labeling skepticism validated** — our pseudo-label approach showed mixed results, and the 7th place team confirmed this was a dead end.

## Sources

- [Speaker Deck summary of top solutions](https://speakerdeck.com/sugupoko/vesuvius2-combined)
- [1st place writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/1st-place-solution-for-the-vesuvius-challenge-su)
- [3rd place writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/quick-preview-of-the-3rd-place)
- [4th place writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/4-th-place-solution)
- [10th place writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/10th-place-solution)
- [ScrollPrize winners page](https://scrollprize.org/winners)
- [ScrollPrize prizes page](https://scrollprize.org/prizes)
- [Competition homepage](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection)
