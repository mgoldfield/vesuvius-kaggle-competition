# Vesuvius Surface Detection — Project Notes

Active working document. Historical run details and design rationale are in **HISTORY.md**
(blog source material). **Move completed work to HISTORY.md** when it's no longer needed
for active decision-making — keep this file lean.

## Competition
- **Goal:** Detect papyrus surfaces in 3D CT scans of Herculaneum scrolls
- **Metric:** `0.30*TopoScore + 0.35*SurfaceDice@tau=2 + 0.35*VOI_score`
- **Deadline:** Feb 27, 2026 (4 days remaining)
- **Submission:** Code competition — Kaggle notebook, GPU, ≤9hr, no internet
- **Leaderboard:** 1,334 teams. Top score 0.607. Our best public: 0.504 (v20, TransUNet)
- **Public test: only 1 volume** (ID 1407735). Scores are high-variance/unreliable.

## Data
- 786 training volumes, 320^3 uint8. Labels: 0=bg, 1=fg (sparse ~2-8%), 2=unlabeled (ignore)
- ~120 hidden test volumes. Val: scroll 26002 holdout (82 volumes)
- 6 scroll_ids: 34117 (382), 35360 (176), 26010 (130), 26002 (88), 44430 (17), 53997 (13)

## Hardware

| Name | GPU | SSH | Role |
|---|---|---|---|
| **gpu0** (this machine) | RTX 5090 32GB | local | Primary control. Probmap generation + PP sweep. |
| **gpu1** (new) | RTX 6000 Ada 48GB | `ssh -i ~/.ssh/remote-gpu root@195.26.233.34 -p 39422` | Pseudo-label training WITHOUT clDice (control). |
| **gpu2** | RTX 6000 Ada 48GB | `ssh -i ~/.ssh/remote-gpu root@195.26.233.87 -p 25763` | Round-2 pseudo-labeling pipeline. |
| **gpu3** (new) | RTX 6000 Ada 48GB | `ssh -i ~/.ssh/remote-gpu root@195.26.233.74 -p 54496` | Selective unfreeze: ViT + clDice. |
| **gpu4** (new) | RTX 6000 Ada 48GB | `ssh -i ~/.ssh/remote-gpu root@195.26.233.43 -p 52276` | Selective unfreeze: decoder + margin dist. |

- Old gpu1 (RTX 5090): **decommissioned** Feb 22. All data synced to gpu0.
- Venv: `/workspace/venv/`, bootstrap: `bash /workspace/start.sh`
- See INSTALLATION.md for full reinstall instructions.

## Best Models

| Model | Val Comp | Notes |
|-------|----------|-------|
| **swa_70pre_30margin_dist_ep5** | **0.5551** (24-vol cross-scroll) | 70% pretrained + 30% margin_dist_ep5. **NEW BEST.** |
| swa_70pre_30topo_ep5 | 0.5549 (24-vol cross-scroll) | 70% pretrained + 30% frozen_boundary_ep5. Previous best. |
| pretrained (comboloss) | 0.5526 (24-vol) | Baseline. All fine-tuned models degrade SDice vs this. |
| SegResNet v9 | 0.570 (ds=4) / ~0.41 (ds=1) | Legacy. Not competitive at full resolution. |

Weights: `checkpoints/swa_topo/swa_70pre_30margin_dist_ep5.weights.h5`

## Competition Scores
| Run | Val Comp | Public Score | Notes |
|-----|----------|-------------|-------|
| Run 9 (SegResNet) | 0.570 (ds=4) | 0.398 | v15: fixed thresholds |
| TransUNet pretrained | 0.5526 (ds=1) | **0.504** | v20: dual-stream + 7-TTA |
| v22 (SWA best) | 0.5549 (ds=1) | PENDING | Manually submitted Feb 22 |

## Key Insights
1. **Dice and comp_score are uncorrelated.** Don't use dice for model selection.
2. **Public leaderboard is unreliable.** Only 1 test volume. Trust local validation.
3. **Cross-scroll evaluation essential.** Single-scroll validation risks overfitting.
4. **Training and eval normalization MUST match.** /255 vs z-score mismatch caused all early
   fine-tuning runs to appear catastrophically degraded.
5. **Val loss can decrease while comp score collapses** if train/eval normalization differs.
6. **No fine-tuned model beats pretrained alone.** All degrade SDice. SWA blending (70/30
   pretrained + fine-tuned) is the only approach that improves on pretrained.
7. **Frozen encoder > discriminative LR** for preserving pretrained features while fine-tuning.
8. **Predictions are 3-5x too thick.** Model-level issue, not PP. Ridge thinning destroys topo.
   Training must learn thinness; PP should reconnect fragments.
9. **T_low is the only meaningful PP parameter.** Optimal value is model-dependent (pretrained=0.70,
   dist_sq=0.30-0.40). Must re-sweep after any model change.

## What We Know Works (high confidence)

1. **Frozen encoder training** — consistently outperforms discriminative LR for preserving pretrained features
2. **SWA 70/30 blending** — every fine-tuned model improves when blended with pretrained at 70/30
3. **Pseudo-labels + clDice** — strong SDice gains (0.8304, our best). Extra training data is extremely beneficial.
4. **Lower T_low + close_erode PP** — potentially huge topo gains (0.5595 comp on 2-vol dry-run, needs confirmation)
5. **close_erode PP is model-independent** — if confirmed, it benefits ANY model. Model improvement and PP improvement are multiplicative.

## Tunable Dimensions (what we can still optimize)

### Training / Loss
- **Margin distance parameter:** margin=2 (gpu1) vs margin=3 (gpu2). Tighter margin = thinner surfaces. Could push to margin=1.
- **clDice weight:** 0.3 (gpu1) vs 0.5 (gpu2 clDice, gpu2 round-2). Higher = more emphasis on thin structure alignment.
- **clDice iterations:** 5 (current, reduced from 10 for memory). Higher = finer skeletonization but more VRAM.
- **Boundary weight:** 0.3 (current). Could try 0.5-1.0 for more edge-squeezing.
- **Loss combinations:** clDice + margin_dist + boundary (gpu1 testing). Could also try clDice alone, margin_dist alone, different ratios.
- **Epoch selection for SWA:** ep5 often beats ep10/15/20 in blends despite worse individual scores. Early stopping matters for blend quality.

### Selective Component Unfreezing — NEW IDEA

Currently `--freeze-encoder` freezes stem + CNN stages + ViT (~48.5M), trains only decoder +
upsampler (~6M). But different model components control different failure modes:

| Component | Params | Controls | Best loss to target it |
|-----------|--------|----------|----------------------|
| Stem + CNN stages | ~23.5M | Local texture/edge features | **Keep frozen** — pretrained excels here |
| ViT (12 layers) | ~25M | Global context at 5^3 | Connectivity/topo losses (clDice) — ONLY place distant surface fragments communicate |
| Query decoder | ~4M | Region assignment, which voxels each query claims | Thinning losses (margin dist, boundary) |
| U-Net upsampler + head | ~2M | Final spatial sharpness | Boundary/sharpness losses |

**Proposed experiments (can run on new GPUs tonight, doesn't interfere with current pipeline):**

1. **Unfreeze ViT only + clDice** (48GB GPU) — teach transformer to preserve surface connectivity.
   The 12 attention layers can learn to propagate surface-membership signals across the whole 5^3
   volume. This targets our topo weakness at the source. ~15 min/epoch, 15 epochs = ~4 hrs.

2. **Unfreeze decoder+queries only + margin dist (margin=1)** (32GB GPU) — aggressive thinning
   on just the decoder. 100 queries learn to claim thinner regions. Only ~4M trainable params,
   very fast. ~10-12 min/epoch, 15 epochs = ~3 hrs.

**Why this could be big:** Current frozen training trains ~6M params to fix problems that
originate in the 25M-param ViT (connectivity) and 4M-param decoder (thickness). By selectively
unfreezing the RIGHT component with the RIGHT loss, we target each failure mode directly.

**Code change needed:** Add `--unfreeze` flag to `train_transunet.py` that accepts component
names (vit, decoder, queries, head) instead of current all-or-nothing `--freeze-encoder`.

**GPU requirements:**
- ViT unfreezing + clDice: 48GB (clDice is memory-hungry)
- Decoder-only unfreezing without clDice: 32GB is fine
- Both: very fast training since CNN encoder stays frozen (no gradients through 23.5M params)

### SWA Blending
- **Blend ratio:** 70/30 is current best. Could try 60/40 or 80/20 with stronger fine-tuned models.
- **Which fine-tuned model to blend:** margin_dist_ep5 currently best (0.5551). Round-2 and gpu1 results pending.
- **Multi-model blending:** Could blend pretrained + multiple fine-tuned models (e.g., 60% pretrained + 20% clDice + 20% margin_dist).

### Post-Processing
- **T_low:** Currently 0.40 in close_erode config. Sweep testing 0.30-0.70.
- **Closing iterations:** 1 (face-connected) currently. Could try 2, or larger structuring elements.
- **Erosion iterations:** 1 currently. Could try 2, or skip for different T_low values.
- **Structuring element:** face-connected (6-neighbor) vs full (26-neighbor). Face-connected used in sweep.
- **Dust removal threshold:** 100 voxels currently. Could tune up or down.
- **Method combinations:** close_erode, erode-only, gap-fill, DME, two-pass hysteresis — all tested in sweep.

### Inference
- **Ensemble:** Average logits from multiple models at inference. Have 4-5 distinct models. Cheap to implement.
- **TTA level:** 7-fold (current), could add more augmentations or weight them differently.
- **Overlap:** 0.42/0.43/0.60 (dual-stream). Higher overlap = better quality but slower.

## Priorities (Feb 23-27, in order)

### Priority 1: Confirm close_erode PP (Feb 23)
T_low sweeps running overnight on 20 volumes. If confirmed:
- close_erode PP is locked in for ALL future submissions
- **Re-evaluate ALL existing models with the new PP config** — model rankings may shift.
  A model that's slightly worse at T_low=0.70 could be better at T_low=0.40 if it has
  stronger connectivity. This must be done early before committing to a model.

### Priority 2: Pick best model + submit (Feb 23-24)
- Review gpu2 margin_dist eval + SWA blend
- Review gpu1 pseudo_margin2_cldice results when done
- Re-evaluate top models with confirmed PP config
- Submit best combination

### Priority 3: Selective component unfreezing (Feb 23-24) — NEW
- **Experiment A:** Unfreeze ViT only + clDice → connectivity improvement (48GB GPU)
- **Experiment B:** Unfreeze decoder only + margin dist margin=1 → thinning (32GB GPU)
- Runs on new GPUs, doesn't interfere with existing pipeline
- Results in ~4 hours per experiment → SWA blend → eval

### Priority 4: Iterative pseudo-labeling (Feb 23-25)
- Round-2 running on gpu2 now (clDice ep20 teacher → sharper pseudo-labels → retrain)
- If improvement: consider round-3, or go straight to train-on-all-data
- This is our highest-leverage training experiment

### Priority 5: Train on all 786 volumes (Feb 25-26)
- Final submission model: train on ALL data (including val) with best config
- Must finalize hyperparams and PP config first (Priorities 1-4)

### Priority 5: Ensemble + hardening (Feb 26-27)
- Average logits from best 2-3 models at inference
- Kaggle notebook hardening, timing, error handling
- Final submissions with buffer for Kaggle queue

## Current Status (Feb 23, ~05:15 UTC)

### Monitoring
Background monitor checks all tasks every 5 min. Latest: `cat /tmp/monitor_status.log`
History: `/tmp/monitor_history.log`
Monitor PID: 3376328 | Script: `/tmp/monitor_overnight.sh`
Tracks: gpu0 eval, T_low sweeps, gpu1 training, gpu2 round-2, gpu3/gpu4 unfreeze experiments, Kaggle.

### Overnight Automation & Recovery Notes

**If anything below fails, these notes provide full context to recover.**

#### 1. gpu0 eval chain — SWA blend fix script

**Problem:** The eval chain (`/tmp/eval_gpu2_and_blend.sh`) has a bug — it extracts comp with
`grep -oP 'comp_score=\K'` but the actual format is `comp_score: 0.5543` (colon not equals).
So BEST_EP stays empty and PHASE 2 (SWA blend) gets skipped.

**Fix:** A follow-up script (`/tmp/fix_eval_swa_blend.sh`, PID 3378160) is running in background.
It waits for the eval chain to finish, parses actual comp scores from the log using `awk`,
picks the best checkpoint, then runs SWA blend + eval.

**Partial results so far:**
- ep10: comp=0.5543
- ep15: comp=0.5559
- ep20, ep25: evaluating

**If the fix script fails, manually run:**
```bash
# 1. Find best epoch from log
grep "comp_score:" logs/eval_gpu2_margin_dist.log
# 2. SWA blend (replace EP with best epoch number)
/workspace/venv/bin/python3 scripts/swa_average.py \
  --weights pretrained_weights/transunet/transunet.seresnext50.160px.comboloss.weights.h5 \
  checkpoints/transunet_pseudo_frozen_margin_dist/transunet_epEP.weights.h5 \
  --ratios 0.70 0.30 --output checkpoints/swa_topo/swa_70pre_30pseudo_margin_dist_epEP.weights.h5
# 3. Eval the blend
/workspace/venv/bin/python3 scripts/eval_transunet.py --weights checkpoints/swa_topo/swa_70pre_30pseudo_margin_dist_epEP.weights.h5 --cross-scroll --max-per-scroll 4 --t-low 0.70 --t-high 0.90
```

#### 2. gpu3 — Chained selective unfreeze ViT experiments

**SSH:** `ssh -o StrictHostKeyChecking=no -i ~/.ssh/remote-gpu root@195.26.233.74 -p 54496`
**Tmux:** `train` session

**Chain script:** `/workspace/vesuvius-kaggle-competition/launch_gpu3_chain.sh` runs two experiments back-to-back:

**Exp 1:** `launch_gpu3_unfreeze_vit.sh` — ViT-only, pure clDice
- Unfreeze: vit (~25M params) | LR: 1e-5 | 15 epochs
- Loss: cldice=1.0, cldice-iters=5 | All others=0
- Log: `logs/unfreeze_vit_cldice.log`
- Checkpoints: `checkpoints/transunet_unfreeze_vit_cldice/`

**Exp 2:** `launch_gpu3_unfreeze_vit_balanced.sh` — ViT-only, clDice-heavy balanced
- Unfreeze: vit (~25M params) | LR: 1e-5 | 15 epochs
- Loss: cldice=1.5, skel=0.75, fp=0.50, boundary=0.3
- Log: `logs/unfreeze_vit_balanced.log`
- Checkpoints: `checkpoints/transunet_unfreeze_vit_balanced/`

**If chain fails or only Exp 1 ran:**
```bash
# SSH in, check what happened
ssh -i ~/.ssh/remote-gpu root@195.26.233.74 -p 54496
tmux list-sessions
cat logs/unfreeze_vit_cldice.log | tail -20
cat logs/unfreeze_vit_balanced.log | tail -20
# Re-launch exp 2 if needed
tmux new-session -d -s train 'cd /workspace/vesuvius-kaggle-competition && bash launch_gpu3_unfreeze_vit_balanced.sh'
```

#### 3. gpu4 — Chained selective unfreeze decoder experiments

**SSH:** `ssh -o StrictHostKeyChecking=no -i ~/.ssh/remote-gpu root@195.26.233.43 -p 52276`
**Tmux:** `train` session

**Chain script:** `/workspace/vesuvius-kaggle-competition/launch_gpu4_chain.sh` runs two experiments back-to-back:

**Exp 1:** `launch_gpu4_unfreeze_decoder.sh` — Decoder+head, pure margin dist
- Unfreeze: decoder head (~6M params) | LR: 5e-5 | 15 epochs
- Loss: dist=1.0(margin=1), boundary=0.3 | All others=0
- Log: `logs/unfreeze_decoder_margin1.log`
- Checkpoints: `checkpoints/transunet_unfreeze_decoder_margin1/`

**Exp 2:** `launch_gpu4_unfreeze_decoder_balanced.sh` — Decoder+head, margin-dist-heavy balanced
- Unfreeze: decoder head (~6M params) | LR: 5e-5 | 15 epochs
- Loss: dist=1.5(margin=1), skel=0.75, fp=0.50, boundary=0.3, cldice=0.3
- Log: `logs/unfreeze_decoder_balanced.log`
- Checkpoints: `checkpoints/transunet_unfreeze_decoder_balanced/`

**If chain fails or only Exp 1 ran:**
```bash
ssh -i ~/.ssh/remote-gpu root@195.26.233.43 -p 52276
tmux list-sessions
cat logs/unfreeze_decoder_margin1.log | tail -20
cat logs/unfreeze_decoder_balanced.log | tail -20
# Re-launch exp 2 if needed
tmux new-session -d -s train 'cd /workspace/vesuvius-kaggle-competition && bash launch_gpu4_unfreeze_decoder_balanced.sh'
```

#### 4. Code change: --unfreeze flag

Added to `scripts/train_transunet.py`:
- `--unfreeze` CLI arg (line 376): accepts component names: vit, decoder, queries, head, cnn
- Selective freeze/unfreeze logic (line ~450-484): freezes ALL layers, then unfreezes named components
- Training loop guard fix (line ~606): `if not args.freeze_encoder and not args.unfreeze:` prevents
  resetting trainable state each epoch when using selective unfreezing

#### 5. All scripts on disk

| Script | Purpose | Location |
|--------|---------|----------|
| Eval chain | Eval gpu2 checkpoints | `/tmp/eval_gpu2_and_blend.sh` |
| SWA blend fix | Fix broken comp extraction | `/tmp/fix_eval_swa_blend.sh` |
| gpu3 exp 1 | ViT + pure clDice | `/tmp/launch_gpu3_unfreeze_vit.sh` |
| gpu3 exp 2 | ViT + clDice balanced | `/tmp/launch_gpu3_unfreeze_vit_balanced.sh` |
| gpu3 chain | Runs exp 1 → exp 2 | `/tmp/launch_gpu3_chain.sh` |
| gpu4 exp 1 | Decoder + pure margin dist | `/tmp/launch_gpu4_unfreeze_decoder.sh` |
| gpu4 exp 2 | Decoder + margin dist balanced | `/tmp/launch_gpu4_unfreeze_decoder_balanced.sh` |
| gpu4 chain | Runs exp 1 → exp 2 | `/tmp/launch_gpu4_chain.sh` |
| Chain followup gpu3 | Sync balanced scripts + restart with chain | `/tmp/chain_followup_gpu3.sh` |
| Chain followup gpu4 | Sync balanced scripts + restart with chain | `/tmp/chain_followup_gpu4.sh` |
| Monitor | 5-min status checks | `/tmp/monitor_overnight.sh` |

### Completed
- **Margin distance training** on gpu0 — DONE. Best: ep5 (comp=0.5500, topo=0.2679).
- **clDice pseudo-label training** on gpu2 — DONE (25 epochs). Eval DONE (see results below).
- **SWA connectivity PP sweep** — DONE. No connectivity method beat baseline on SWA probmaps.
- **Competitor research** — See `COMPETITOR_RESEARCH_FEB22.md`.
- **v23 Kaggle submission** — Adaptive TTA timer. Completed on Kaggle.
- **clDice eval + probmap gen** — DONE at 22:02 UTC.
- **gpu1 pseudo_frozen_boundary training** — DONE (25 epochs, loss=0.911).
  Checkpoints pulled to gpu0. Watcher eval ran but may have incomplete results.
- **SWA blends created + evaluated:**
  - `swa_70pre_30margin_dist_ep5`: **comp=0.5551**, topo=0.2477, SDice=0.8299. **NEW BEST MODEL.**
  - `swa_70pre_30cldice_ep20`: comp=0.5543, topo=0.2389, SDice=0.8308. Better SDice but worse topo.
- **gpu2 pseudo_frozen_margin_dist training** — DONE (25 epochs, completed 03:35 UTC).
  Checkpoints pulled to gpu0: ep5, ep10, ep15, ep20, ep25.
- **New best model probmaps generated** — 82 val volumes from swa_70pre_30margin_dist_ep5.
  Dir: `data/swa_70pre_30margin_dist_ep5_probmaps/`
- **Kaggle v24 submitted** — new best model + close_erode PP (T_low=0.40). Running on Kaggle.

### clDice pseudo-label eval results

| Model | Comp | Topo | SDice | VOI | n |
|-------|------|------|-------|-----|---|
| **pseudo_frozen_cldice_ep20** | **0.5546** | 0.2476 | **0.8304** | 0.5419 | 24 |
| pseudo_frozen_cldice_ep25 | 0.5545 | 0.2472 | 0.8300 | 0.5425 | 24 |
| pseudo_frozen_cldice_ep10 | 0.5538 | 0.2459 | 0.8295 | 0.5419 | 24 |
| pseudo_frozen_cldice_ep15 | 0.5538 | 0.2451 | 0.8296 | 0.5427 | 24 |
| *swa_70pre_30topo_ep5 (previous best)* | *0.5549* | *0.2499* | *0.8291* | *0.5420* | *24* |

**Key finding:** clDice ep20 nearly matches our best SWA blend (0.5546 vs 0.5549) and has
**better SDice** (0.8304 vs 0.8291). Strong SWA blend candidate.
ep5 OOM'd during eval. Results: `logs/eval_pseudo_frozen_cldice_results.csv`

### T_low PP sweep — PRELIMINARY (2-vol only, needs confirmation)

| Config | Comp | Topo | SDice | VOI | FG% |
|--------|------|------|-------|-----|-----|
| **close_erode_tl0.40_c1_e1** | **0.5595** | **0.3357** | 0.7800 | 0.5307 | 10.2% |
| B_dme_tl0.4_r1 | 0.5346 | 0.3285 | 0.7049 | 0.5410 | 14.8% |
| base_tl0.40 | 0.5251 | 0.2998 | 0.7021 | 0.5413 | 14.6% |
| *baseline_t70 (current)* | *0.5350* | *0.2277* | *0.7907* | *0.5427* | — |

**WARNING: These results are from only 2 volumes.** Full 20-volume sweeps now running on both
old SWA and new best model probmaps to confirm. Pattern: low T_low → closing bridges gaps →
erosion thins back → connected AND thin.

### gpu0: Eval gpu2 pseudo_frozen_margin_dist — RUNNING (GPU)

Script: `/tmp/eval_gpu2_and_blend.sh` | Log: `logs/eval_gpu2_margin_dist.log`
Evaluating 5 checkpoints (ep5/10/15/20/25), then SWA blend best → eval.
Currently on ep10. ETA ~06:00 UTC for all evals, ~07:00 for blend.

### gpu0: T_low PP sweeps (CPU) — RUNNING

Two parallel sweeps, 59 configs × 20 volumes each (~10-13 hours):
1. **Old SWA probmaps** (`swa_70_30_val_probmaps`): Log: `logs/tlow_pp_swa_val_20vols.log`
2. **New best model probmaps** (`swa_70pre_30margin_dist_ep5_probmaps`): Log: `logs/tlow_pp_margin_dist_blend_20vols.log`

ETA: results by ~14:00-17:00 UTC (9 AM - 12 PM ET)

### gpu0: Old connectivity PP sweep (CPU) — RUNNING

Pretrained sweep still in progress. Log: `logs/connectivity_pp_pretrained_sweep.log`

### gpu1: pseudo_margin2_cldice — TRAINING (ep 2/25)

Relaunched after OOM fix. Ep 2/25 at 03:50 UTC, loss=0.965. Stable at 34.7/49.1 GB VRAM.
Config: frozen encoder, SWA weights, pseudo-labels, margin dist (margin=2, power=2, w=0.02),
clDice=0.3 (iters=5), boundary=0.3. 25 epochs, save every 5.
ETA: ~11:00 UTC (6 AM ET). Pull + eval when done.
**SSH:** `ssh -i ~/.ssh/remote-gpu root@195.26.233.34 -p 39422`

### gpu2: Round-2 pseudo-labeling pipeline — RUNNING

Iterative pseudo-labeling: use clDice ep20 (best SDice) as teacher for sharper pseudo-labels.
Tmux session: `round2` | Log: `logs/pseudo_r2_cldice.log`
1. Generate round-2 probmaps (704 volumes, ~16s/vol) — RUNNING (15/704, ETA ~07:15 UTC)
2. Create round-2 pseudo-labels — queued (~30 min)
3. Train round-2 model (25 epochs, cldice=0.5, boundary=0.3) — queued (ETA start ~08:00 UTC)
Training config: frozen encoder, SWA init, cldice=0.5, boundary=0.3, cldice-iters=5.
**SSH:** `ssh -i ~/.ssh/remote-gpu root@195.26.233.87 -p 25763`

### Competition scores
- v22 (SWA best) scored **0.504** — same as v20/v21. Single public test volume.
- v24 (margin_dist blend + close_erode PP) — RUNNING on Kaggle.
- Top score on LB: 0.607.

### Margin distance training results (COMPLETE)

| Model | Comp | Topo | SDice | VOI | n |
|-------|------|------|-------|-----|---|
| swa_70pre_30topo_ep5 (baseline) | 0.5526 | 0.2354 | 0.8255 | 0.5517 | 24 |
| **frozen_margin_dist_ep5** | **0.5500** | **0.2679** | 0.8069 | 0.5350 | 24 |
| frozen_margin_dist_ep15 | 0.5482 | 0.2616 | 0.8059 | 0.5361 | 24 |
| frozen_margin_dist_best | 0.5470 | 0.2622 | 0.8022 | 0.5358 | 24 |
| frozen_margin_dist_ep10 | 0.5463 | 0.2650 | 0.7997 | 0.5341 | 24 |

**Key finding:** ep5 has the **best topo score of any model** (0.2679 vs frozen_boundary's 0.2642).
Same pattern: early epochs best for SWA blending. Strong SWA blend candidate.
Results: `logs/eval_frozen_margin_dist_results.csv`

### SWA connectivity PP sweep results (COMPLETE — disappointing)

No connectivity method beat simple baselines on SWA probmaps:

| Config | Comp | Topo | SDice | VOI |
|--------|------|------|-------|-----|
| baseline_t70 | **0.5350** | 0.2277 | 0.7907 | 0.5427 |
| B_dme_tl0.7_r1 | 0.5345 | 0.2273 | 0.7909 | 0.5414 |
| baseline_t50 | 0.5328 | 0.2365 | 0.7616 | 0.5579 |
| D_combo_bd3_i2 | 0.5270 | 0.2156 | 0.7846 | 0.5364 |

Results: `logs/connectivity_pp_swa_70_30_val_probmaps.csv`
Pretrained + dist_sq sweeps still running — thinner models may benefit more from connectivity PP.

### clDice pseudo-label training (COMPLETE — eval pending)

25 epochs completed on gpu2. Loss plateaued at ~1.052 from ep11. Checkpoints pulled to gpu0.
Eval running now. Checkpoints: `checkpoints/transunet_pseudo_frozen_cldice/`
Config: frozen encoder, SWA weights, pseudo-labels, cldice=0.5, iters=5.

### Margin distance loss (implemented Feb 22)

Replaces dist_sq with a margin-based variant: `penalty = max(0, dist - margin)^power`.
Voxels within `margin` voxels of the skeleton get zero penalty (free zone). Beyond that,
penalty ramps up. With margin=3, surfaces up to ~6 voxels thick incur no penalty.

**Code changes** (on gpu0 and gpu2, NOT gpu1):
- `_generate_dist_from_skeleton` now returns raw voxel distances capped at 10 (was [0,1] normalized).
- `build_loss` accepts `dist_margin` parameter, applies `keras.ops.relu(dist - margin)`.
- CLI: `--dist-margin` flag (default 0.0 = backward compatible with old dist_sq).
- **Weight scaling:** Old dist-weight=2.0 was calibrated for [0,1] distances. With raw voxels,
  dist-weight ~0.02-0.05 gives similar gradient magnitude. At w=0.02, margin dist contributes
  ~15-20% of total loss.

### Pseudo-labeling pipeline (stages 1-2 complete)

Uses best model's high-confidence predictions to convert unlabeled voxels (label=2,
~52% of each volume) into training signal. 80.4% of unlabeled voxels converted at
0.85/0.15 thresholds, FG nearly doubles.

**Stages 1-2 COMPLETE:** 704 probmaps (42 GB) + 704 pseudo-labels (21 GB) generated.

### Kaggle v22 (submitted Feb 22 03:42 UTC)

Updated inference to use SWA best weights (swa_70pre_30topo_ep5, local val 0.5549).
Score **PENDING** — check with `kaggle competitions submissions -c vesuvius-challenge-surface-detection`

### Disk usage (gpu0)

~175 GB / 350 GB used (adding clDice checkpoints 2.2 GB + new probmap dirs).

## SWA Weight Averaging — current best approach

Blending pretrained + fine-tuned weights at 70/30 ratio is the only approach that beats pretrained.
This is our primary model improvement strategy. After pseudo-label training completes, we should
SWA blend the best pseudo-label checkpoint with pretrained (same 70/30 recipe).

**Topo-focused blends (frozen_boundary source):**

| Model | Comp | Topo | SDice | VOI | n |
|-------|------|------|-------|-----|---|
| pretrained (baseline) | 0.5526 | 0.2354 | 0.8255 | 0.5517 | 24 |
| swa_90pre_10topo (ep10) | 0.5544 | 0.2403 | 0.8285 | 0.5496 | 24 |
| swa_70pre_30topo (ep10) | 0.5545 | 0.2490 | 0.8301 | 0.5407 | 24 |
| **swa_70pre_30topo_ep5** | **0.5549** | 0.2499 | 0.8291 | 0.5420 | 24 |
| swa_70pre_30sdice_ep15 | 0.5548 | 0.2489 | 0.8301 | 0.5418 | 24 |

**Key findings:**
- 70/30 ratio is the sweet spot (consistent across dist_sq and frozen_boundary blends)
- Topo improvement is substantial (+0.014) with SDice maintained or improved
- ep5 slightly outperforms ep10 in the blend despite ep10 having better individual topo
- Pure fine-tuned weights are much worse — pretrained carries most value

Results: `logs/eval_swa_topo_results.csv`, `logs/eval_swa_results.csv`
Weights: `checkpoints/swa_topo/`
Script: `scripts/swa_blend.py`

## Fine-Tuning Experiments (completed, inform future work)

All fine-tuned models degrade SDice vs pretrained. Frozen encoder consistently better than
discriminative LR. These results inform pseudo-label training strategy (frozen encoder, boundary loss).

| Model | Best Comp | Best Topo | Best SDice | Strategy |
|-------|-----------|-----------|------------|----------|
| frozen_boundary (gpu2) | 0.5408 (ep10) | **0.2642** (ep10) | 0.7871 (ep15) | Frozen encoder |
| frozen_dist_sq (gpu2) | 0.5402 (ep25) | 0.2634 (ep25) | 0.7885 (ep10) | Frozen encoder |
| discrim_boundary (gpu1) | 0.5286 (ep15) | 0.2342 (ep15) | 0.7836 (ep15) | Discriminative LR |
| discrim_dist_sq (gpu1) | 0.5269 (ep25) | 0.2292 (ep15/25) | 0.7841 (ep25) | Discriminative LR |

Results: `logs/eval_frozen_boundary_results.csv`, `logs/eval_discrim_boundary_results.csv`, etc.

## PP Sweep Findings (inform future PP tuning)

26 configs on pretrained probmaps (82 vols) + 26 configs on dist_sq probmaps.

**Key findings:**
- **T_low is the only meaningful PP parameter.** Closing, dust removal, confidence filtering = noise (±0.001).
- **Optimal T_low depends on the model.** Pretrained optimal T_low=0.70, dist_sq optimal T_low=0.30-0.40.
  Thinner predictions need lower threshold to preserve connectivity.
- **PP barely helps fine-tuned models.** Best fine-tuned PP config = 0.506 vs pretrained's 0.553. Gap is in the model.
- **After pseudo-label training, re-sweep T_low** on the new model — optimal value will likely shift.

Results: `logs/postprocessing_sweep.csv`, `logs/sweep_pp_dist_sq_results.csv`

## Prediction Thickness (core problem, ongoing)

**Problem:** Model predicts 15-30% foreground per volume vs GT's 2-8%. Surfaces are 3-5x too thick.
Confirmed from exploration notebook: probmaps themselves are thick (model-level, not PP artifact).

**Impact on metrics:**
- **SDice (35%):** Thick slabs create two boundary surfaces; one aligns with GT, other is penalized.
- **VOI (35%):** Excess voxels increase conditional entropy; thickness merges nearby surfaces.
- **Topo (30%):** Merged surfaces change component count and create false tunnels.

**What we've tried:**
- dist_sq loss (quadratic penalty far from skeleton) — partial improvement, best topo=0.2642
- Frozen encoder + boundary loss — best at preserving topo while thinning
- SWA blending — 30% fine-tuned dose thins slightly without destroying SDice
- Ridge thinning PP — **destroys topology** (topo 0.29→0.005). PP can't thin safely.
- clDice — needs 48GB VRAM. Testing on gpu2 with pseudo-labels.

**Remaining approaches — training:**
- **Pseudo-labeling** (active) — expanded training signal may help model learn sharper boundaries
- **clDice + pseudo-labels** (active, gpu2) — soft-skeletonization loss directly measures thin alignment
- **Margin distance loss** (implemented Feb 22) — replaces dist_sq. Free zone of `margin` voxels
  around skeleton (zero penalty), then `(dist - margin)^power` beyond. `--dist-margin 3` allows
  ~6-voxel thick surfaces with no penalty, aggressively penalizes thick tails.
  **NOTE:** Distance normalization changed — `_generate_dist_from_skeleton` now returns raw voxel
  distances (capped at 10) instead of normalized [0,1]. Old `--dist-weight 2.0` was calibrated
  for normalized distances. With raw voxels + margin=3 + power=2, a voxel at dist=7 contributes
  `(7-3)^2 = 16` vs old `(0.7)^2 * w = 0.98`. **Scale `--dist-weight` down ~30x** (e.g., 0.05-0.1)
  to get similar gradient magnitude. Or tune from scratch since this is a different loss shape.
- **Higher boundary loss weight** — currently 0.3. Could try 0.5-1.0 to squeeze predictions
  from the edges more aggressively. Boundary loss is complementary to dist_sq: dist_sq says
  "be near the center", boundary loss says "don't extend past the edges".

**Remaining approaches — post-processing:**
- **Connectivity PP** — implemented in `scripts/sweep_connectivity_pp.py`. Dry-run verified (2 vols).
  4 methods: (A) probmap-guided gap filling, (B) dilate-merge-erode, (C) two-pass hysteresis,
  (D) combined C→A→bridge cleanup. ~30 configs total. Full sweep on pretrained probmaps (82 vols)
  takes ~2 hrs CPU-only. Dry-run results (2 vols, preliminary):
  - D_combo comp=0.5433 (best), B_dme comp=0.5411, baseline_t50 comp=0.5194
  - Methods B and D reduce FG% (10-12% vs baseline 15%) while improving SDice
  - Methods A and C alone thicken too much (20% FG) at t_low_strict=0.70; t_low_strict=0.80 untested

Visual analysis: `notebooks/analysis/multi_model_comparison.ipynb`

## Strategy (Feb 22 → Feb 27)

### Phase 1: Training + eval (Feb 22-23) — MOSTLY COMPLETE
- [x] Generate pseudo-labels (stages 1-2)
- [x] Margin distance training on gpu0 (original labels)
- [x] clDice pseudo-label training on gpu2 — DONE, eval DONE
- [x] Pseudo-label training WITHOUT clDice on gpu1 (`pseudo_frozen_boundary`) — DONE
- [x] clDice eval + probmap gen — DONE
- [x] SWA blends created + evaluated — margin_dist_ep5 is NEW BEST (0.5551)
- [x] gpu2 `pseudo_frozen_margin_dist` — DONE, checkpoints pulled to gpu0
- [x] Generate probmaps from new best model — DONE (82 val volumes)
- [ ] gpu1 `pseudo_margin2_cldice` (margin=2 + clDice) — TRAINING ep 2/25 (ETA 11:00 UTC)
- [ ] Eval gpu2 pseudo_frozen_margin_dist checkpoints — RUNNING on gpu0
- [ ] SWA blend gpu2 best checkpoint → eval — queued

### Phase 1.5: T_low + PP sweep + submit (Feb 23)
- [x] Integrate close_erode PP into Kaggle notebook (T_low=0.40)
- [x] **Submit v24** — new best model + close_erode PP — RUNNING on Kaggle
- [ ] T_low sweep on old SWA probmaps (20 vols, 59 configs) — RUNNING (ETA ~17:00 UTC)
- [ ] T_low sweep on new best model probmaps (20 vols, 59 configs) — RUNNING (ETA ~17:00 UTC)
- [ ] Review sweep results → potentially resubmit with tuned config
- **NOTE:** 2-vol dry-run showed close_erode at 0.5595 but needs 20-vol confirmation

### Phase 2: Confirm PP + re-evaluate models (Feb 23-24)
- [ ] Review T_low sweep results (20 vols) → confirm close_erode PP config
- [ ] **Re-evaluate top models with new PP config** — rankings may shift at T_low=0.40
- [ ] Pick best model + PP combination → submit
- [ ] Review gpu1 pseudo_margin2_cldice → SWA blend → eval with new PP

### Phase 2.5: Iterative pseudo-labeling (Feb 23-25) — RUNNING on gpu2

Round-2 pipeline launched: clDice ep20 (best SDice) → sharper pseudo-labels → retrain.
1. Generate round-2 probmaps (704 vols, ~16s/vol) — RUNNING, ETA ~07:15 UTC
2. Threshold at 0.85/0.15 → round-2 pseudo-labels — queued
3. Train on round-2 pseudo-labels (25 epochs, cldice=0.5, boundary=0.3) — queued
4. SWA blend best round-2 checkpoint with pretrained (70/30)
5. Optionally iterate (round-3) if time allows

**Expected benefit:** Round-1 pseudo-labels from pretrained (SDice=0.8255). Round-2 from
clDice ep20 (SDice=0.8304) → ~5% sharper boundaries. Knowledge distillation with better teacher.

### Phase 3: Train on all data — FINAL SUBMISSION (Feb 25-26)

Train on ALL 786 volumes (including 82 val) with best config. ~10% more data + no val holdout
penalty. Must finalize all hyperparams and PP first (can't validate locally).

**Pipeline:**
1. Finalize best loss config, LR, epochs from earlier experiments
2. Generate pseudo-labels for ALL 786 volumes (not just 704 non-val)
3. Train on all 786 volumes with best loss config, frozen encoder
4. SWA blend with pretrained (70/30)
5. Apply best PP config (from T_low sweep)
6. Submit to Kaggle — this is the FINAL model

**Timing:** Must start by Feb 25 to allow training + Kaggle queue time.

### Phase 4: Ensemble + hardening (Feb 26-27)
- [x] **Adaptive TTA timer** — DONE (v23). Auto-reduces 7→4→1 fold TTA.
- [ ] Ensemble best 2-3 models (logit averaging at inference). Cheap +0.01-0.02 boost.
- [ ] Kaggle notebook hardening (memory, timing, error handling)
- [ ] Final submissions with buffer for Kaggle queue (deadline Feb 27)

### Ideas on deck (if time allows)
- [ ] **Multi-scale inference fusion** — 128^3 + 160^3 averaged
- [ ] **External scroll data** — scrollprize.org (see notes below). High effort, low priority.
- [ ] **Frangi filter PP** — Hessian-based sheet enhancement (from competitor research)
- [ ] **Multi-model SWA** — blend pretrained + 2-3 fine-tuned (e.g., 60/20/20)

### External data: scrollprize.org (research notes)

6 full scrolls available with CT scans + OBJ surface meshes at scrollprize.org.
This is NOT the same format as competition data — would require significant processing:

**What we'd need to do:**
1. Download CT volumes (massive — each scroll is hundreds of GB of raw .tif slices)
2. Download OBJ surface meshes (ground truth surface locations)
3. Chunk CT into 320^3 overlapping patches (like competition format)
4. Convert OBJ mesh → binary voxel labels for each 320^3 patch
5. Quality-check: ensure labels are thin (1-2 voxels) like competition GT

**Can we process a section?** Yes — we could download just one scroll section (a few hundred
slices out of thousands) and generate 50-100 volumes. This would be much faster than the
full dataset.

**Feasibility assessment:**
- Mesh-to-voxel conversion is non-trivial (needs proper coordinate alignment)
- Label thickness would need to match competition (1-2 voxels, not solid)
- Download size is the main bottleneck (even one section is many GB)
- Time estimate: 1-2 days including debugging, which is tight for our deadline
- **Recommendation:** Only attempt if iterative pseudo-labeling (Phase 2.5) doesn't
  give enough improvement. Pseudo-labeling is much faster and lower risk.

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
**Gotchas:** No internet on Kaggle. Use `--dir-mode zip` for subdirectories.
Uploads hang at 0% for ~8 min (normal). Delete stale tokens: `rm /tmp/.kaggle/uploads/*.json`.

## Kaggle Notebook Versions
| Version | Weights | Date | Status | Notes |
|---------|---------|------|--------|-------|
| v15 | v9 traced | Feb 15 | 0.398 | Fixed thresholds |
| v20 | TransUNet comboloss | Feb 17 | **0.504** | Dual-stream + 7-fold TTA + seeded hysteresis |
| v21 | TransUNet comboloss | Feb 18 | 0.504 | T_low=0.70 (same — public test is 1 volume) |
| v22 | SWA best | Feb 22 | PENDING | swa_70pre_30topo_ep5 (local val 0.5549) |

## File Structure
```
/workspace/vesuvius-kaggle-competition/
├── notebooks/                     # Training notebooks (v1-v13 + refinement)
│   └── analysis/                  #   Multi-model comparison, exploration
├── data/                          # Competition data (not in git)
│   ├── train_images/              #   786 .tif volumes
│   ├── train_labels/              #   786 .tif labels
│   ├── pseudo_labels/             #   704 pseudo-labeled .tif files
│   └── train.csv, test.csv
├── pretrained_weights/transunet/  # TransUNet SEResNeXt50 weights from Kaggle
├── checkpoints/                   # Model checkpoints
│   ├── swa_topo/                  #   SWA blends (current best)
│   └── transunet_pseudo_*/        #   Pseudo-label training checkpoints
├── kaggle/                        # Submission artifacts
│   ├── kaggle_notebook/           #   Inference script + kernel metadata
│   └── kaggle_weights_download/   #   Weight datasets for upload
├── scripts/                       # Training, eval, pipeline scripts
├── libs/topological-metrics-kaggle/  # topometrics library
├── logs/                          # Pipeline logs + eval CSVs
├── NOTES.md                       # This file (active)
├── HISTORY.md                     # Run history & blog source
├── INSTALLATION.md                # Dependency reinstall guide
├── TRANSUNET_SETUP.md             # TransUNet installation guide
├── COMPETITOR_ANALYSIS.md         # Competitor notebook analysis
└── CLAUDE.md                      # Claude Code instructions
```
