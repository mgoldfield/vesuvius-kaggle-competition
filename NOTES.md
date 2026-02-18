# Vesuvius Surface Detection — Project Notes

Active working document. Historical run details and design rationale are in **HISTORY.md**
(blog source material). **Move completed work to HISTORY.md** when it's no longer needed
for active decision-making — keep this file lean.

## Competition
- **Goal:** Detect papyrus surfaces in 3D CT scans of Herculaneum scrolls
- **Metric:** `0.30*TopoScore + 0.35*SurfaceDice@tau=2 + 0.35*VOI_score`
- **Deadline:** Feb 27, 2026 (11 days remaining)
- **Submission:** Code competition — Kaggle notebook, GPU, ≤9hr, no internet
- **Leaderboard:** 1,334 teams. Top score 0.607. Our best public: 0.504 (v20, TransUNet)
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
| TransUNet pretrained | 0.510 (82v) | **0.504** | Comboloss weights, dual-stream, 7-TTA |

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

## Current Status (Feb 17 overnight → Feb 18 morning)

### Kaggle TransUNet submission: v20 scored 0.504 (up from 0.431)

Built and pushed a TransUNet inference notebook based on Tony Li's 0.552 dual-stream
approach. v18 failed (FileNotFoundError — model_sources path wrong). Fixed by uploading
TransUNet weights to our own dataset (`mgoldfield/vesuvius-unet3d-weights`). v20 completed.

**Public LB: 0.504** — +17% over our previous best (0.431), but below the competitor's
0.545 with same weights. Gap likely due to post-processing params (our sweep is showing
tlow=0.6 > tlow=0.5 locally). Next submission should use optimized PP params.

**Key features of v20 inference:**
- JAX backend + Keras 3 + medicai offline wheels
- Dual-stream inference (Tony Li's 0.552 approach):
  - Public stream: overlap=0.42, softmax argmax → foreground anchor
  - Private stream: overlap=0.43 (TTA) / 0.60 (identity), binary logits
- 7-fold TTA (identity + 3 flips + 3 rotations)
- Binary logit: `logsumexp(L1, L2) - L0` from 3-class
- Seeded hysteresis: strong=(prob>=0.90), weak=(prob>=0.50 OR public_fg)
- Anisotropic closing (z=3, xy=2) + dust removal (100 voxels)
- Using comboloss weights (LB 0.545 as single-stream baseline)

**Files:**
- `kaggle/kaggle_notebook/vesuvius-inference.py` — inference script
- `kaggle/kaggle_notebook/kernel-metadata.json` — kernel metadata

### Overnight pipeline: Phase 5 (PP sweep) running

Launched at 05:39. Phases 1-4 completed successfully. Phase 5 (PP sweep, 26 configs) in progress.

**Phase 1 (Benchmark):** Completed. Single patch: 0.141s fwd, 0.443s fwd+bwd. Peak 19.82 GB.
Full volume SWI (overlap=0.5): 6.2s. 10-epoch training: ~1 hour.

**Phase 2 (Validation Inference):** Completed. 82 volumes, scroll 26002.
Mean composite: 0.510 (all 82 vols, ds=1). Probmaps saved to `data/transunet_probmaps/`.

**Phase 3 (TTA):** Completed. (Results in log — 5 volumes with TTA.)

**Phase 4 (Cross-scroll):** Completed. 30 volumes across 6 scrolls.
Mean composite: 0.544 (30 vols). Per-scroll: 26002=0.476, 26010=0.656, 34117=0.486,
35360=0.507, 44430=0.518, 53997=0.508.

**Phase 5 (PP Sweep):** COMPLETED. 26 configs on 82 full-res volumes. Results:
```
Config                      Comp   Topo  SDice    VOI
tlow_0.6                  0.5179 0.2024 0.7342 0.5722  ← BEST (+0.0076)
close_z2_xy1              0.5110 0.2054 0.7064 0.5776
competitor_default        0.5103 0.1997 0.7044 0.5823  (= tlow_0.5)
our_old_defaults          0.4940 0.1907 0.6551 0.5928
tlow_0.3                  0.4892 0.1901 0.6307 0.6041  ← WORST
```
Key: T_low=0.6 is clearly best. T_high barely matters (0.75-0.95 all ~same).
Closing/dust/confidence filtering had minimal impact.

**Phase 6 (Exploration):** COMPLETED. Notebook executed successfully.

**Pipeline completed at 02:37 Feb 18.** All 6 phases successful.

**Results in:**
- `logs/overnight_transunet.log` — full pipeline output
- `logs/transunet_eval.csv` — per-volume scores (82 vols)
- `logs/postprocessing_sweep.csv` — all 26 configs ranked
- `notebooks/analysis/transunet_exploration_executed.ipynb` — visual analysis
- `data/transunet_probmaps/` — saved probmaps for reuse

### Bug fixed: TensorFlow stealing GPU memory

Keras 3 imports TensorFlow even with `KERAS_BACKEND=torch`. TF was grabbing ~15 GiB GPU
by default, causing OOM alongside PyTorch (~16 GiB for TransUNet). Fix: added
`tf.config.set_visible_devices([], 'GPU')` at the top of all scripts. Peak VRAM dropped
from ~30 GiB (OOM) to 16.27 GiB (forward) / 21.26 GiB (fwd+bwd).

### Dry run results (2 volumes, no TTA)

TransUNet pretrained (comboloss weights) at ds=1:
- Mean comp_score: **0.5257** (with TTA) / **0.5251** (cross-scroll, 6 vols, no TTA)
- Per-scroll: 26002=0.476, 26010=0.656, 34117=0.486, 35360=0.507, 44430=0.518, 53997=0.508
- PP sweep: `close_z1_xy1` marginally better (0.5209 vs 0.5194 default)

Full 82-volume results pending from overnight pipeline.

### TransUNet setup (COMPLETED)
- Installed Keras 3 + medicai (from GitHub source — pip version is WRONG)
- Downloaded 3 pretrained weight sets from Kaggle (see `TRANSUNET_SETUP.md`)
- Verified: model loads, forward pass works, produces correct (1,160,160,160,3) output
- Peak VRAM: 16.27 GB forward, 21.26 GB fwd+bwd (after TF GPU fix)
- Using `KERAS_BACKEND=torch` locally for PyTorch backend

### Key findings

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

### TransUNet Training Memory Budget (RTX 5090, 33.7 GB)

| Config | Peak VRAM | Status |
|--------|-----------|--------|
| bs=1, fp32, 160^3, fwd only | 16.27 GB | Safe |
| bs=1, fp32, 160^3, fwd+bwd | 21.26 GB | Safe (12.4 GB headroom) |
| Recommended | bs=1 + grad_accum=4 | Simulates bs=4, 21.26 GB |
| Est. 10 epochs | ~0.8 hours | Per benchmark |

### What to do when you wake up (Feb 18)

1. **Push v21 with T_low=0.6** — Sweep shows T_low=0.6 is best (0.5179 vs 0.5103 default).
   Update `vesuvius-inference.py` line 75: `T_low=0.60`. Push and submit. Quick win.
2. **Investigate 0.504 vs 0.545 gap** — Our v20 scored 0.504 vs competitor's 0.545 with
   same weights. Possible causes: PP params (T_low fix above), binary mode (try class1),
   or subtle implementation difference in dual-stream averaging.
3. **Try simpler single-stream** — The 0.537 baseline uses single stream + simple threshold.
   Might score better than our dual-stream if the dual-stream implementation has issues.
4. **Fine-tune TransUNet** — `scripts/train_transunet.py` exists but untested.
   bs=1, grad_accum=4, ~0.8 hours for 10 epochs.
5. **Loss functions** for fine-tuning: SparseDiceCE + 0.75*SkeletonRecall + 0.50*FP_Volume.
   VOI is our biggest weakness (35% of metric) — need smooth, unfragmented output.

**Hardware:** GPU 1 only (RTX 5090). GPU 2 paused. GPU 3 shut down.

## Investigations

Open questions and observations to explore. These may feed into novel approaches.

### 1. Prediction thickness — our surfaces are 3-5x too thick

**Observation:** Exploration notebook shows our model predicts 15-30% foreground per
volume vs GT's 2-8%. The predicted surfaces are visibly much thicker than ground truth.
This is likely our single biggest scoring problem.

**How it hurts each metric:**
- **SurfaceDice (35%):** Thick predictions create two boundary surfaces (top+bottom of
  slab). One aligns with GT, the other is in empty space → ~half our surface is penalized.
- **VOI (35%):** Excess voxels increase conditional entropy. Thickness can also merge
  nearby parallel surfaces into one blob → wrong component count.
- **TopoScore (30%):** Merged surfaces change B0 (component count) and can fill cavities
  or create false tunnels.

**Potential remedies — training:**
- FP_Volume loss (already in fine-tuning) directly penalizes predictions on BG voxels
- Boundary-focused loss that penalizes surface thickness (e.g., distance transform penalty)
- Skeletonization loss: penalize deviation from 1-voxel-thick surfaces
- Higher weight on FP_Volume (currently 0.50 — could try 1.0 or 2.0)

**Potential remedies — post-processing:**
- Morphological thinning / skeletonization of predictions
- Erode-then-dilate (opening) to thin without fragmenting
- Higher T_low (already shown to help: 0.6-0.75 better than 0.5)
- Surface-aware thinning: keep only voxels near the probability ridge (local maxima
  along the thickness direction)
- Adaptive erosion: erode thick regions more than thin ones

**Potential remedies — inference:**
- Use argmax class 1 probability directly instead of binary logit (might be thinner)
- Higher overlap SWI might produce sharper boundaries

**Key question:** Is the thickness a property of the pretrained weights, or of our
post-processing? Compare raw probmap thickness to post-processed thickness. If the
probmap itself is thick, we need training fixes. If it's sharp but PP makes it thick,
we need PP fixes.

**Answer (confirmed from exploration notebook):** The probmaps themselves are too thick.
This is a model-level issue in the pretrained weights, not just a PP artifact. FP_Volume
loss in fine-tuning should help, but may need higher weight (try 1.0-2.0). Post-processing
can also help: extract the probability ridge (non-max suppression along thickness axis)
rather than thresholding the whole blob.

## Refinement Model (ABANDONED)
- Approached abandoned in favor of TransUNet pivot. GPU 3 shut down.
- Phase 2 result: delta -0.0298 vs baseline. Improves topo+sdice but destroys VOI.
- If TransUNet fine-tuning leaves spare GPU time, could revisit refinement on top of
  TransUNet probmaps. Low priority.

## Strategy (REVISED Feb 18 morning)

**Two-phase plan for the final 9 days.** Replaying public notebooks caps us at ~0.552.
To reach top 10 (0.57-0.60+), we need novel approaches beyond what's publicly shared.

### Phase 1: Aggressive / swing for the fences (Feb 18-23, 5-6 days)

**Quick wins (Day 1):**
- Push v21 with T_low=0.6 (or adaptive if analysis supports it)
- Adaptive T_low analysis (running, results pending)

**Foundation — Fine-tune TransUNet (Days 1-2):**
- Train with SparseDiceCE + 0.75*SkeletonRecall + 0.50*FP_Volume
- AdamW, lr=5e-5, cosine decay, wd=1e-5, bs=1 + grad_accum=4
- 3D CutOut augmentation (6 blocks, 2-8 voxels)
- ~1 hour per 10 epochs. Checkpoint every epoch.
- This is table-stakes, not our differentiator.

**Novel approaches — our edge (Days 3-6):**

1. **Component-level learned refinement** — Fixes our old refinement's VOI problem.
   Instead of per-voxel refinement (which fragments), operate per-component:
   - Run base model → probmap → threshold → extract connected components
   - Per-component features: size, shape, mean/std prob, boundary sharpness
   - Small classifier: keep, remove, or split
   - For splits: small model predicts WHERE to cut
   - Can't fragment because it operates on whole components
   - Directly targets VOI (our worst metric, 35% of score)
   - Nobody in public notebooks does this

2. **Pseudo-labeling on unlabeled regions** — Dataset has massive label=2 (ignore)
   regions. Use pretrained model's high-confidence predictions as soft labels for
   those regions, then retrain. Effectively multiplies training data. Well-known in
   semi-supervised learning but not used in this competition.

3. **Differentiable VOI-proxy loss** — Everyone uses DiceCE + SkeletonRecall. VOI is
   35% of score but nobody optimizes it directly (not differentiable). Approximations:
   penalize probability variance within neighborhoods (encourages smooth, unfragmented
   predictions), or use soft connected-components approximation.

4. **Multi-scale inference fusion** — Everyone runs at 160^3. Also run at 128^3
   (different field of view, different boundary artifacts) and average probmaps.
   Uncorrelated errors = free ensemble from one model.

### Phase 2: Tuning and hardening (Feb 24-27, last 3-4 days)

- Adaptive TTA timer (safety-critical for 9hr Kaggle limit on 120 volumes)
- PP parameter tuning with final model
- Kaggle notebook hardening (memory, timing, error handling)
- Ensemble best checkpoints
- Final submissions (keep 2-3 slots for last-minute improvements)

## Improvement Roadmap

### Done
- [x] **Cross-scroll evaluation** — Scroll 35360 catastrophic failure → fixed with T_HIGH=0.75.
- [x] **Adaptive T_HIGH thresholds** — No benefit over fixed 0.75 (8 strategies, all equal).
- [x] **Refinement Phase 1-2** — Proof of concept. Improves topo+sdice but destroys VOI (-0.13).
- [x] **Metric downsample investigation** — ds=4 inflates by +0.16. Use ds=1 going forward.
- [x] **Competitor notebook analysis** — Downloaded 17 notebooks, identified key techniques.
- [x] **TransUNet setup** — Installed medicai, loaded pretrained weights, verified inference.
- [x] **PP sweep** — 26 configs on 82 volumes at ds=1. T_low=0.6 best (+0.0076 over default).
- [x] **Overnight pipeline (6 phases)** — All completed. 82-vol val, TTA, cross-scroll, PP sweep.
- [x] **Kaggle TransUNet v20** — 0.504 public LB. Dual-stream + 7-TTA + seeded hysteresis.

### In progress
- [~] **Adaptive T_low analysis** — 20-volume sweep running. Early signal: correlations exist.
- [~] **Fine-tune TransUNet** — Setting up training with SkeletonRecall loss.

### Next up
- [ ] **Push v21** — T_low=0.6 (or adaptive if analysis supports it).
- [ ] **Component-level refinement** — Novel learned post-processing targeting VOI.
- [ ] **Pseudo-labeling** — Retrain with soft labels on unlabeled regions.
- [ ] **Ensemble** — Fine-tuned + pretrained TransUNet logit averaging.

### Ideas on deck
- [ ] **Differentiable VOI-proxy loss** — Approximate VOI for training.
- [ ] **Multi-scale inference fusion** — 128^3 + 160^3 averaged.
- [ ] **Surface splitting (killer ant)** — Port competitor post-processing.
- [ ] **Hole filling (line tracing)** — Fill gaps in surface predictions.
- [ ] **TV smoothing of probmaps** — Before thresholding.
- [ ] **SegResNet as ensemble diversity member** — Different architecture, different failures.

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
| v18 | TransUNet comboloss | Feb 17 | FAILED | FileNotFoundError — model_sources path wrong |
| v19 | TransUNet comboloss | Feb 17 | FAILED | Same issue (dataset not ready) |
| v20 | TransUNet comboloss | Feb 17 | **0.504** | Dual-stream + 7-fold TTA + seeded hysteresis |

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
