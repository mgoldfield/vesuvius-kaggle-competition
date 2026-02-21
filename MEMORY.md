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

## Project Approach

The user's core theory: the pretrained TransUNet is too thick. Fine-tuning with
dist_sq loss thins predictions (improves topo/VOI) but loses SDice (surface
alignment). The fix is two-pronged:
1. **Training:** dist_sq + boundary loss to thin from both sides (skeleton and edges)
2. **Post-processing:** connectivity-focused PP to reconnect fragments

The user prioritizes understanding the problem deeply over running many experiments.
They'd rather run 3 well-motivated experiments than 10 random ones.
