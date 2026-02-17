# Vesuvius Surface Detection — Project Notes

Active working document. Historical run details and design rationale are in **HISTORY.md**
(blog source material). **Move completed work to HISTORY.md** when it's no longer needed
for active decision-making — keep this file lean.

## Competition
- **Goal:** Detect papyrus surfaces in 3D CT scans of Herculaneum scrolls
- **Metric:** `0.30*TopoScore + 0.35*SurfaceDice@tau=2 + 0.35*VOI_score`
- **Deadline:** Feb 27, 2026 (11 days remaining)
- **Submission:** Code competition — Kaggle notebook, GPU, ≤9hr, no internet
- **Leaderboard:** 1,334 teams. Top score 0.607. Our best public: 0.423 (v8)
- **Public test: only 1 volume** (ID 1407735). Scores are high-variance/unreliable.

## Data
- 786 training volumes, 320^3 uint8. Labels: 0=bg, 1=fg (sparse ~2-8%), 2=unlabeled (ignore)
- ~120 hidden test volumes. Val: scroll 26002 holdout (82 volumes)
- 6 scroll_ids: 34117 (382), 35360 (176), 26010 (130), 26002 (88), 44430 (17), 53997 (13)

## Hardware
**Cloud GPU (current):** RTX 5090, persistent `/workspace/` volume.
- Venv: `/workspace/venv/`, bootstrap: `bash /workspace/start.sh`
- See INSTALLATION.md for full reinstall instructions.

## Current Best: Run 9 (v9)
- **Architecture:** MONAI SegResNet, 4.7M params, SuPreM pretrained weights
- **Loss:** 0.2*BCE + 0.2*Dice + 0.3*clDice + 0.3*Boundary
- **Training:** LR=1e-5 discriminative (enc=1e-7, dec=1e-6, head=1e-5), fit_one_cycle, 30 epochs
- **Features:** 160^3 patches, FG-biased sampling (0.5), 7-fold TTA, hysteresis thresholding
- **Val comp_score:** 0.570 (at T_low=0.35, T_high=0.80)
- **Traced model:** `kaggle/kaggle_weights_download/best_segresnet_v9_traced.pt` (19 MB)

## Competition Scores
| Run | Val Comp | Public Score | Notes |
|-----|----------|-------------|-------|
| Run 1 | — | 0.331 | BatchNorm baseline UNet |
| Run 2 | — | 0.290 | GroupNorm — worse score despite better dice |
| Run 3 | — | 0.348 | SuPreM SegResNet + clDice |
| Run 8 | 0.562 | 0.423 | + FG sampling + TTA (LR too high, peaked epoch 1) |
| Run 9 | 0.570 | 0.200 | **Best model** — v13 had wrong thresholds on Kaggle |
| Run 9 | 0.570 | 0.398 | v15: fixed thresholds — public test is 1 volume |
| Run 10 | 0.584 | 0.267 | + attn gates + DS — peaked epoch 0 (LR issue) |
| Run 12 | — | — | TRAINING: flat_cos + 50 epochs |

## Key Insights (from 11 runs — details in HISTORY.md)
1. **Inference pipeline > model architecture.** Gaussian SWI + logit TTA gave +0.19 comp_score
   with the same weights. Ensure Kaggle notebook matches the best pipeline.
2. **lr_find().valley is ~100x too aggressive** for this model. Always eyeball the first dip
   in the lr_find plot, or just hardcode LR=1e-5 (proven across v9/v10b-v2).
3. **fit_one_cycle warmup hurts pretrained models.** Destabilizes SuPreM features. Use
   `fit_flat_cos` (no warmup). LR magnitudes are right; schedule was the problem.
4. **Dice and comp_score are uncorrelated.** Don't use dice for model selection.
5. **T_high=0.75 is optimal.** 0.80 fails on scroll 35360 (max prob ~0.79 → zero seeds).
   0.75 fixes 35360 (+0.223) with zero regression on other scrolls. Overall +0.043.
6. **Early comp_score peaks are noise.** v9 peaked at epoch 8/30, v10 at epoch 0. Save by
   comp_score only from the second half of training (DelayedSaveCallback, epoch 25+).
7. **Model selection is broken.** Training uses simplified inference; full pipeline gives 2x
   better scores. Solution: periodic checkpoints + post-training eval sweep.
8. **Public leaderboard is unreliable.** Only 1 test volume — scores are high-variance.
   v8 (0.423) vs v15 (0.398) is noise, not signal. Trust local validation over public score.
9. **Local validation needs cross-scroll evaluation.** Validating on 5 volumes from 1 scroll
   risks overfitting to that scroll's characteristics. Run `scripts/eval_cross_scroll.py` on
   all 786 probmaps grouped by scroll_id to check generalization.
10. **Adaptive T_HIGH has no benefit over fixed 0.75.** Tested 8 strategies (max*0.95, max-0.05,
    p90/p95/p99 percentile). All score identically at 0.5737. What matters is binary: T_HIGH
    below max prob → good; above → catastrophic. Fixed 0.75 is simplest and most robust.

## Current Status (Feb 17 evening)

### STRATEGIC PIVOT: Adopting TransUNet + medicai framework

After competitor analysis revealed that ALL top teams use TransUNet + SEResNeXt50 (70.1M params
vs our 4.7M SegResNet), we are abandoning our SegResNet approach and adopting the competitor
framework directly. The pretrained TransUNet weights already achieve 0.545 LB — we just need
to fine-tune and apply our post-processing improvements.

**TransUNet setup (COMPLETED):**
- Installed Keras 3 + medicai (from GitHub source — pip version is WRONG)
- Downloaded 3 pretrained weight sets from Kaggle (see `TRANSUNET_SETUP.md`)
- Verified: model loads, forward pass works, produces correct (1,160,160,160,3) output
- Memory test: peak 19.89 GB for bs=1 training on 160^3. Safe on RTX 5090 (33.7 GB).
- Using `KERAS_BACKEND=torch` locally for PyTorch backend

**Refinement approach: ABANDONED** (GPU 3 shut down by user)

**Kaggle v17: 0.431** (best public score)

### Key findings this session

**METRIC_DOWNSAMPLE=4 inflates local scores by +0.16:**

| Downsample | Mean CompScore (5 vol) | Difference |
|-----------|----------------------|-----------|
| **ds=1 (full res, what Kaggle uses)** | **0.4113** | — |
| ds=2 | 0.5039 | +0.093 |
| ds=4 (what we've been using) | 0.5700 | **+0.159** |

Our "0.57 local val" is actually ~0.41 at full resolution. All future eval uses **ds=1**.

**Competitor analysis:** See `COMPETITOR_ANALYSIS.md` for full 17-notebook breakdown.
Top approach: TransUNet SEResNeXt50, 3-class, SparseDiceCE + SkeletonRecall + FP_Volume,
dual-stream inference, seeded hysteresis. Pretrained model gets 0.545 LB out of the box.

### TransUNet Fine-Tuning Memory Budget (RTX 5090, 33.7 GB)

| Config | Peak VRAM | Status |
|--------|-----------|--------|
| bs=1, fp32, 160^3 | 19.89 GB | Safe (13.8 GB headroom) |
| bs=2, fp32, 160^3 | ~33.8 GB | Likely OOM |
| bs=1, fp16, 160^3 | ~10-12 GB (est.) | Safe, enables bs=2 |
| Recommended | bs=1 + grad_accum=4 | Simulates bs=4, 19.89 GB |

### What's next

1. **Full-volume SWI inference** with TransUNet on validation set
2. **Compare scores** to our SegResNet baseline (expect ~0.54 at ds=1 vs our 0.41)
3. **Apply our post-processing** (hysteresis + closing + dust) to TransUNet output
4. **Set up Kaggle submission** with TransUNet (use JAX backend on Kaggle)
5. **Fine-tune** TransUNet with our training data + competitor loss functions

**Hardware:** GPU 1 only (RTX 5090). GPU 2 paused. GPU 3 shut down.

## Refinement Model (ABANDONED)
- Approached abandoned in favor of TransUNet pivot. GPU 3 shut down.
- Phase 2 result: delta -0.0298 vs baseline. Improves topo+sdice but destroys VOI.
- If TransUNet fine-tuning leaves spare GPU time, could revisit refinement on top of
  TransUNet probmaps. Low priority.

## Strategy (REVISED Feb 17 evening)

**Pivoting to TransUNet.** Our SegResNet (4.7M params) cannot compete with TransUNet
(70.1M params). The pretrained TransUNet already scores 0.545 LB vs our 0.431. Rather
than incremental improvements to a weak model, we adopt the winning framework directly.

**Priority order (10 days remaining):**
1. **TransUNet inference pipeline** — Run pretrained model on validation set, apply our
   post-processing, measure full-res scores. Should immediately beat our SegResNet.
2. **Kaggle submission with TransUNet** — Adapt inference notebook for Kaggle (JAX backend,
   offline wheels). Target: match or beat public 0.545.
3. **Fine-tune TransUNet** — Train with competitor loss (SparseDiceCE + SkeletonRecall +
   FP_Volume) + our data augmentation. bs=1 with grad_accum=4 on RTX 5090.
4. **Post-processing optimization** — Dual-stream inference, seeded hysteresis, surface
   splitting (killer ant). These are additive on top of the better base model.
5. **Ensemble** — Fine-tuned TransUNet + pretrained TransUNet (different weights).

## Improvement Roadmap

### Done
- [x] **Cross-scroll evaluation** — Scroll 35360 catastrophic failure → fixed with T_HIGH=0.75.
- [x] **Adaptive thresholds** — No benefit over fixed 0.75 (8 strategies, all equal).
- [x] **Refinement Phase 1-2** — Proof of concept. Improves topo+sdice, needs VOI fix.
- [x] **Metric downsample investigation** — ds=4 inflates by +0.16. Use ds=1 going forward.
- [x] **Competitor notebook analysis** — Downloaded 17 notebooks, identified key techniques.
- [x] **TransUNet setup** — Installed medicai, loaded pretrained weights, verified inference.

### In progress
- [~] **TransUNet full-volume inference** — Set up SWI for validation set evaluation.

### Next up (PRIORITY ORDER)
- [ ] **TransUNet val set evaluation** — Run pretrained TransUNet on 5+ val volumes at ds=1.
  Compare to SegResNet baseline (expected: ~0.54 vs 0.41).
- [ ] **Kaggle submission with TransUNet** — Adapt inference for Kaggle (JAX backend,
  offline wheels, competitor post-processing params).
- [ ] **Fine-tune TransUNet** — Train with SparseDiceCE + SkeletonRecall + FP_Volume.
  bs=1, grad_accum=4, cosine decay from 5e-5. Checkpoint every epoch.
- [ ] **Post-processing optimization** — Dual-stream inference, seeded hysteresis,
  confidence-based CC filtering, surface splitting.
- [ ] **Ensemble** — Average fine-tuned + pretrained TransUNet weights.

### Ideas on deck
- [ ] **Surface splitting (killer ant)** — Port the post-processing from competitors.
- [ ] **TV smoothing of probmaps** — `skimage.restoration.denoise_tv_chambolle` before thresholding.
- [ ] **SegResNet ensemble member** — Our old model as diversity for ensemble (low priority).

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
**Gotchas:** No internet on Kaggle. Use Pillow for TIFFs (not tifffile). Use torch.jit.trace
for MONAI models. P100 has 16GB VRAM. Uploads hang at 0% for ~8 min (normal). Delete stale
tokens: `rm /tmp/.kaggle/uploads/*.json`. Use `--dir-mode zip` for subdirectories.

## Kaggle Notebook Versions
| Version | Weights | Date | Status | Notes |
|---------|---------|------|--------|-------|
| v13 | v9 traced | Feb 12 | 0.200 | Wrong thresholds → 0% fg |
| v14 | v9 traced | Feb 12 | COMPLETE | No public score shown |
| v15 | v9 traced | Feb 15 | 0.398 | Fixed thresholds. Public test = 1 volume (high variance) |
| v16 | v9 traced | Feb 16 | PENDING | T_HIGH=0.75 (cross-scroll fix for scroll 35360) |
| v17 | v9 traced | Feb 16 | PENDING | Adaptive T_HIGH (p95 of probs > 0.3, clamped 0.50-0.90) |

## File Structure
```
/workspace/vesuvius-kaggle-competition/
├── notebooks/                     # Training notebooks (v1-v13 + refinement)
│   ├── vesuvius_train_v9.ipynb    #   Best SegResNet model
│   ├── vesuvius_train_v12.ipynb   #   flat_cos retraining
│   └── vesuvius_train_v13.ipynb   #   3-class formulation
├── data/                          # Competition data (not in git)
│   ├── train_images/              #   786 .tif volumes
│   ├── train_labels/              #   786 .tif labels
│   └── train.csv, test.csv
├── pretrained_weights/            # Model weights (not in git)
│   ├── transunet/                 #   TransUNet SEResNeXt50 weights from Kaggle
│   │   ├── transunet.seresnext50.160px.comboloss.weights.h5  (LB 0.545)
│   │   ├── transunet.seresnext50.160px.weights.h5            (LB 0.505)
│   │   └── transunet.seresnext50.128px.weights.h5            (LB 0.500)
│   └── suprem/                    #   SuPreM weights (for SegResNet)
├── competitor_notebooks/          # Downloaded competitor notebooks (17)
├── kaggle/                        # Submission artifacts
│   ├── kaggle_notebook/           #   Inference script + kernel metadata
│   └── kaggle_weights_download/   #   Traced models (v9, v10, v11, v12)
├── scripts/                       # Automation
├── libs/topological-metrics-kaggle/  # topometrics library
├── logs/                          # Pipeline logs
├── NOTES.md                       # This file (active)
├── HISTORY.md                     # Run history & blog source
├── INSTALLATION.md                # Dependency reinstall guide
├── TRANSUNET_SETUP.md             # TransUNet installation guide
├── COMPETITOR_ANALYSIS.md         # Competitor notebook analysis
└── CLAUDE.md                      # Claude Code instructions
```
