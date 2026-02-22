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

## Project Approach

The user's core theory: the pretrained TransUNet is too thick. Fine-tuning with
dist_sq loss thins predictions (improves topo/VOI) but loses SDice (surface
alignment). The fix is two-pronged:
1. **Training:** dist_sq + boundary loss to thin from both sides (skeleton and edges)
2. **Post-processing:** connectivity-focused PP to reconnect fragments

The user prioritizes understanding the problem deeply over running many experiments.
They'd rather run 3 well-motivated experiments than 10 random ones.

## Current State (Feb 22, ~06:00 UTC)

**Best model:** swa_70pre_30topo_ep5 (comp=0.5549). SWA blending is the winning
strategy — no single fine-tuned model beats pretrained, but 70/30 blends consistently do.

**Active experiments:**
- gpu0: Connectivity PP sweep running on SWA probmaps (43 configs, 82 vols). Probmaps done.
  After SWA sweep → pretrained sweep automatically. ~4 hrs total.
- gpu1 (RTX 6000 Ada 48GB): Setting up — installing packages, data transfer 89% done.
  Pseudo-label training WITHOUT clDice (control). Needs user approval to launch.
- gpu2 (RTX 6000 Ada 48GB): Pseudo-label training WITH clDice running (iters=10).
  Ep1 step 400/704, ~20 min/epoch, ETA ~14:00 UTC for all 25 epochs. 46.7/49.1 GB VRAM.
- Kaggle v22 score still PENDING.

**Key context:**
- gpu2 SSH IP changed: now `root@195.26.233.87 -p 25763` (old IP timed out).
- Pseudo-label training OOM'd on gpu0 (32GB). Denser pseudo-labels need >32GB GPU.
- User wants approval before launching any new processes (in CLAUDE.md).
- Connectivity PP sweep is DIFFERENT from the earlier standard PP sweep — tests
  gap filling, dilate-merge-erode, two-pass hysteresis, and combined methods.
- User is a night owl, may go to sleep soon and return ~1-2 PM ET (~18:00-19:00 UTC).

**GPU status:** gpu0 (local) + gpu1 + gpu2, all active. Old gpu1 decommissioned.
