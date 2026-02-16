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

## Current Status (Feb 16 night)

**Overnight plan (Feb 16–17):**
- **GPU 1 (local):** v12 training → checkpoint sweep → log results
- **GPU 2 (remote):** v13 training → checkpoint sweep → log results
- **GPU 3 (new):** Refinement Phase 3 (+ TV loss) → eval vs baseline → log results
- **Kaggle v17:** Submitted, check score tomorrow
- All results logged automatically; review at 1pm ET Feb 17

**Tomorrow (Feb 17 afternoon):**
- Review v12, v13, and refinement Phase 3 results
- If v12 or v13 beats v9 → generate probmaps from winner → retrain refinement on those
- If refinement Phase 3 improved → also try Option A (baseline post-processing after refinement)
- Begin planning CV ensemble timing (target: end of week)

**SSH to GPU 2:** `ssh root@103.196.86.229 -p 13849 -i /root/.ssh/remote-gpu`
**SSH to GPU 3:** `ssh root@103.196.86.227 -p 13963 -i /root/.ssh/remote-gpu`
All 3 GPUs are RTX 5090 (32GB). Key at `/root/.ssh/remote-gpu` (copied from `/workspace/remote-gpu`).

## Refinement Model
- **Phase 2 result:** Loses to baseline by -0.0298 (2W/18L on 20 val volumes).
  Improves TopoScore (+0.015) and SurfaceDice (+0.033) but badly degrades VOI (-0.131).
- **Diagnosis:** VOI degradation is expected — no smoothness objective in loss function.
  The model optimized what it was trained on (topo via clDice, sdice via Boundary) and
  produced fragmented outputs that hurt VOI.
- **Phase 3 plan:** Fine-tune from Phase 2 weights with TV (Total Variation) loss added.
  TV penalizes spatial discontinuity → directly fights fragmentation that kills VOI.
  Loss = BCE + Dice + clDice + Boundary + TV (modest weight 0.1-0.2).
- **Future:** If v12/v13 beats v9, retrain refinement on those probmaps instead of v9's.
- **Option A (untested):** Apply baseline post-processing (hysteresis + closing + dust removal)
  *after* refinement model output. Zero cost, may recover VOI without retraining. Test tomorrow.
- **Eval:** `scripts/eval_refinement.py`, results in `logs/refinement_eval.csv`

## Strategy
- **Differentiation:** Avoid nnUNet (black box, everyone uses it). Custom refinement model
  is our edge — a learned post-processing step that others aren't doing.
- **CV ensemble:** Plan for end of week (~Feb 21-22). Reliable +2-5% but freezes iteration.
  Need best base model finalized first.
- **Iteration order:** base model (v12/v13) → refinement on best probmaps → ensemble last

## Improvement Roadmap

### Done
- [x] **Cross-scroll evaluation** — Scroll 35360 catastrophic failure → fixed with T_HIGH=0.75.
- [x] **Adaptive thresholds** — No benefit over fixed 0.75 (8 strategies, all equal).
- [x] **Refinement Phase 1-2** — Proof of concept. Improves topo+sdice, needs VOI fix.

### In progress
- [~] **v12 (flat_cos retraining)** — Training on local GPU. flat_cos, 50 epochs, periodic checkpoints.
- [~] **v13 (3-class formulation)** — Training on remote GPU. 3-class CE + Dice/clDice/Boundary.
- [~] **Refinement Phase 3 (TV loss)** — Writing notebook, will train on 3rd GPU tonight.

### Next up
- [ ] **Post-training eval sweep** — Script ready (`scripts/eval_checkpoint_sweep.py`).
  Run on v12/v13 checkpoints when training completes.
- [ ] **Generate probmaps from best model** — If v12/v13 beats v9, generate new probmaps
  for refinement training. ~3 hours without TTA.
- [ ] **CV ensemble** — Target end of week. 3-fold scroll-grouped CV, average probabilities.

### Ideas on deck
- [ ] **Attention gates + deep supervision** — Revisit with flat_cos if base model improves.
- [ ] **Skeleton recall loss** — Pre-compute GT skeleton, penalize missed voxels.
- [ ] **3D CutOut augmentation** — Zero out random cuboids. Strong regularizer.

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
│   ├── vesuvius_train_v9.ipynb    #   Current best model
│   ├── vesuvius_train_v12.ipynb   #   flat_cos retraining (queued)
│   ├── vesuvius_train_v13.ipynb   #   3-class formulation (ready to train)
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
│   ├── eval_cross_scroll.py       #   Cross-scroll generalization analysis
│   ├── eval_checkpoint_sweep.py   #   Post-training checkpoint evaluation
│   ├── eval_refinement.py         #   Refinement vs baseline head-to-head
│   ├── trace_model.py             #   Trace models for Kaggle (plain/dsattn/3-class)
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
