# Memory — Working Notes

Claude's notes about our collaboration. Updated throughout the project to help
maintain continuity across context window resets.

## About the User

**Technical profile:** Strong ML intuition and first-principles thinker, but not a
day-to-day ML engineer. Asks excellent clarifying questions to build understanding
before acting ("what does SDice measure?", "what is clDice?", "what is an NVMe?").
Has great instincts about model behavior — the thinning hypothesis (model should
learn thin predictions from GT, PP should handle connectivity) was spot-on and
better than my initial approach of post-processing-only thinning.

**Working style:**
- Wants to understand WHY before committing to an approach
- Likes to be consulted on strategy, not just given results
- Prefers novel/principled approaches over brute force
- Will push back when I jump to conclusions — correctly called me out when I
  misdiagnosed gpu2 as stalled based on a single nvidia-smi snapshot
- Trusts me with overnight autonomous work but wants monitoring in place
- Night owl — works late, checks in mid-morning (~10 AM ET)

**Communication preferences:**
- Concise status updates with tables (GPU status, eval results)
- Always wants ETAs and timing estimates
- Appreciates clear explanations of ML concepts without being patronizing
- Likes things written down for future reference (NOTES.md, CLAUDE.md policies)
- Responds well to "here's what I found, here's what I recommend, here are the options"

**Decision-making:**
- Makes quick decisions when given clear options
- Good at prioritizing under deadline pressure (competition ends Feb 27)
- Prefers to kick things off manually when possible ("no need to automatically
  trigger — I'll be watching"), but accepts automation for overnight runs
- Thinks about the competition strategically — not just "run more experiments"
  but "what's the highest-leverage thing we can do?"

## Lessons Learned

- **Don't diagnose remote GPU issues from single snapshots.** nvidia-smi is
  point-in-time. Check file timestamps, process state, and power draw before
  concluding something is stalled.
- **Always use tmux for remote training.** nohup over SSH is unreliable — the
  process can die when SSH disconnects. tmux creates a persistent session.
- **pgrep alternation uses `|` not `\|`.** Cost us a failed launch when the
  chain script used wrong regex syntax.
- **Dry run everything.** The user values this highly. Every overnight pipeline
  must pass a dry run first, no exceptions.
- **Context window management matters.** Remote GPU management should be done
  via background agents to keep the main context clean. The user explicitly
  asked for this policy.
- **NFS-cached logs can be stale.** When checking remote GPU progress, read
  logs on the remote machine itself, not from the shared network volume.
- **Disk tools overreport space.** `df` shows NFS volume capacity (~1.7 PB) not
  rented allocation (350 GB on gpu0). Use `du -sh /workspace/` for real usage.
  User can rent more cheaply — ask when approaching limits.
- **Sync data before shutting down GPUs.** Always SCP checkpoints and logs back
  to gpu0 before turning off remote GPUs. Verify with `ls -lh` after transfer.
- **User values visual data exploration.** When asked to build notebooks, include
  `%matplotlib inline` for rendering. Use the venv kernel (`vesuvius`) for execution.
  Papermill works for background execution.
- **Test bash scripts for arg correctness before deploying.** grep argparse
  definitions to verify flags exist rather than running --help (which imports
  heavy libs and is slow). Cross-check extracted flags from scripts against
  parser definitions.
- **Bash grep patterns in eval scripts break easily.** The eval chain used
  `comp_score=` but actual format was `comp_score:`. Always check the actual
  output format when writing extraction patterns. Build fix scripts to run
  after buggy chains rather than trying to kill/restart mid-run.
- **User prefers balanced loss functions over pure/isolated ones.** When
  proposing ablation experiments, the user's instinct is to keep all existing
  loss components and heavily weight the new one, rather than zeroing everything
  else. This often yields better results since the model retains stability.
- **Give ETAs in EST, not UTC.** User is on US East Coast.

## Project Approach

The user's core theory: the pretrained TransUNet is too thick. Fine-tuning with
dist_sq loss thins predictions (improves topo/VOI) but loses SDice (surface
alignment). The fix is two-pronged:
1. **Training:** dist_sq + boundary loss to thin from both sides (skeleton and edges)
2. **Post-processing:** connectivity-focused PP to reconnect fragments

The user prioritizes understanding the problem deeply over running many experiments.
They'd rather run 3 well-motivated experiments than 10 random ones.

## Current State (Feb 23, ~00:15 AM EST)

**Best model:** swa_70pre_30margin_dist_ep5 (comp=0.5551). SWA blending is the winning
strategy — no single fine-tuned model beats pretrained, but 70/30 blends consistently do.

**Key innovation this session:** Selective component unfreezing — `--unfreeze` flag added
to train_transunet.py. Can now train just the ViT (connectivity) or just the decoder
(boundary precision) with targeted loss functions. User proposed this approach and also
the insight that balanced losses (keeping skel+fp defaults while heavily weighting the
new targeted loss) are better than pure isolation.

**Active experiments (5 GPUs):**
- gpu0: Eval chain for gpu2 checkpoints (ep10=0.5543, ep15=0.5559) + T_low sweeps (20 vol)
- gpu1: pseudo_margin2_cldice training ep 5/25
- gpu2: Round-2 iterative pseudo-labeling (probmaps 205/704)
- gpu3: Setting up — ViT unfreeze chain (pure clDice → balanced clDice)
- gpu4: Setting up — Decoder unfreeze chain (pure margin dist → balanced margin dist)

**Strategic priorities (agreed with user):**
1. Confirm close_erode PP with 20-vol T_low sweep → re-evaluate models
2. Pick best model from eval results + SWA blends → submit
3. Selective component unfreezing experiments (gpu3/gpu4)
4. Iterative pseudo-labeling (round 2, gpu2)
5. Train on all 786 volumes for final submission

**GPU fleet:** gpu0 (RTX 5090) + gpu1-4 (RTX 6000 Ada ×4). All active.
SSH key for remotes: `~/.ssh/remote-gpu`. See NOTES.md hardware table for details.
