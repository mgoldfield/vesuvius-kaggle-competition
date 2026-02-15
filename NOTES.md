# Vesuvius Surface Detection — Project Notes

Active working document. Historical run details and design rationale are in **HISTORY.md**
(blog source material). **Move completed work to HISTORY.md** when it's no longer needed
for active decision-making — keep this file lean.

## Competition
- **Goal:** Detect papyrus surfaces in 3D CT scans of Herculaneum scrolls
- **Metric:** `0.30*TopoScore + 0.35*SurfaceDice@tau=2 + 0.35*VOI_score`
- **Deadline:** Feb 27, 2026 (12 days remaining)
- **Submission:** Code competition — Kaggle notebook, GPU, ≤9hr, no internet
- **Leaderboard:** 1,334 teams. Top score 0.607. Our best public: 0.441 (but v15 pending)

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
| Run 9 | 0.570 | PENDING | v15: fixed thresholds, fg=28.0% |
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
5. **T_high=0.80 beats 0.85** across all T_low values. Best: T_low=0.35, T_high=0.80.
6. **Early comp_score peaks are noise.** v9 peaked at epoch 8/30, v10 at epoch 0. Save by
   comp_score only from the second half of training (DelayedSaveCallback, epoch 25+).
7. **Model selection is broken.** Training uses simplified inference; full pipeline gives 2x
   better scores. Solution: periodic checkpoints + post-training eval sweep.

## Current Plan (overnight Feb 15 → afternoon Feb 16)

**Overnight pipeline running** (`scripts/overnight_full.sh`), log: `logs/overnight_full.log`:

1. **Phase 1: Generate probmaps** (~3hr) — Gaussian SWI + logit TTA on 786 volumes with v9.
   Output: `data/refinement_data/probmaps/`
2. **Phase 2: Train refinement model** (~2-3hr) — RefinementUNet3D (~365K params).
   Phase 1: 50 epochs BCE+Dice (save by valid_loss). Phase 2: 30 epochs + topo losses (save by comp_score).
   Head-to-head eval vs hand-tuned post-processing built into notebook.
3. **Phase 3: Train v12** (~5-6hr) — Plain SegResNet, `fit_flat_cos`, 50 epochs, LR=1e-5.
   DelayedSaveCallback (epoch 25+). Periodic checkpoints every 5 epochs. Auto-traces at end.

**When you return:** check `logs/overnight_full.log`. Compare v12 vs v9 (0.570). Check
refinement head-to-head. Best case: use v12 + refinement together.

## Refinement Model
- **Architecture:** RefinementUNet3D — shallow 3D U-Net, ~365K params, channels [8,16,32,64]
- **Input:** v9 probability map (float16, 320^3). **Output:** refined logits.
- **Phase 1:** fit_one_cycle, LR=1e-3, 50 epochs, BCE+Dice, save by valid_loss
- **Phase 2:** fit_flat_cos, LR=3.3e-4, 30 epochs, BCE+Dice+clDice+Boundary, save by comp_score
- **Kaggle:** torch.jit.trace → ~1.5 MB. Pipeline: main model → SWI+TTA → refinement → threshold 0.5
- **Notebook:** `notebooks/refinement/vesuvius_train_refinement.ipynb`

## Improvement Roadmap

### Ready to try next
- [ ] **Post-training eval sweep** — run full pipeline on all periodic v12 checkpoints
- [ ] **3-class formulation** — predict bg/fg/unlabeled as 3 classes. All top notebooks use this.
- [ ] **CV ensemble** — 3-fold scroll-grouped CV with probability averaging. Needs reliable
  base model first (v12 or v9).
- [ ] **nnUNet framework** — self-configuring, often wins competitions. Pre-processed data exists.

### Proven but unverified at scale
- [ ] **Attention gates + deep supervision** — added in v10/v10b but LR issues prevented proper
  testing. Could revisit with flat_cos + per-module LRs if base model improves.
- [ ] **Skeleton recall loss** — pre-compute GT skeleton, penalize missed voxels.
- [ ] **3D CutOut augmentation** — zero out random cuboids. Strong regularizer.

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
| v15 | v9 traced | Feb 15 | PENDING | Fixed: T_low=0.35, T_high=0.80, no splitting → fg=28.0% |

## File Structure
```
/workspace/vesuvius-kaggle-competition/
├── notebooks/                     # Training notebooks (v1-v12 + refinement)
│   ├── vesuvius_train_v9.ipynb    #   Current best model
│   ├── vesuvius_train_v12.ipynb   #   flat_cos retraining (running)
│   └── refinement/                #   Learned post-processing
├── data/                          # Competition data (not in git)
│   ├── train_images/              #   786 .tif volumes
│   ├── train_labels/              #   786 .tif labels
│   ├── refinement_data/probmaps/  #   v9 probmaps for refinement (generating)
│   └── train.csv, test.csv
├── kaggle/                        # Submission artifacts
│   ├── kaggle_notebook/           #   Inference script + kernel metadata
│   └── kaggle_weights_download/   #   Traced models (v9, v10, v11, v12)
├── scripts/                       # Automation
│   ├── overnight_full.sh          #   Current overnight pipeline
│   ├── overnight_refinement.sh    #   Refinement-only pipeline
│   ├── generate_refinement_data.py
│   ├── eval_inference.py          #   Compare inference pipelines
│   ├── trace_model.py             #   Trace models for Kaggle
│   └── smoke_test.py              #   Pre-flight check for notebooks
├── checkpoints/                   # Model weights (not in git)
├── pretrained_weights/            # SuPreM weights (not in git)
├── libs/topological-metrics-kaggle/  # topometrics library
├── logs/                          # Pipeline logs
├── NOTES.md                       # This file (active)
├── HISTORY.md                     # Run history & blog source
├── INSTALLATION.md                # Dependency reinstall guide
└── CLAUDE.md                      # Claude Code instructions
```
