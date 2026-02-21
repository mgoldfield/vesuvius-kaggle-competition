We are participating in a Kaggle machine learning competition.

**Always start every conversation by reading NOTES.md and MEMORY.md** — NOTES.md contains the full project history, run results, and current status. MEMORY.md contains notes about our working relationship, the user's approach, and lessons learned. Update MEMORY.md when you learn something new about how we work together, mistakes to avoid, or insights about the user's preferences and thinking style.

## Engineering standards

- **Overnight/background pipelines must always have a `--dry-run` flag** that tests each component of the pipeline on a small subset of data (e.g., 1 volume instead of 20). This catches missing dependencies, wrong paths, and import errors before committing to a long unattended run.
- **All background pipelines that will run for more than 30 minutes must be fully dry-run tested before they run.** No exceptions — every phase must pass before launching the full pipeline.
- **Dry runs should be the smallest possible run that exercises all the code that will run.** Use 1-2 volumes, 1 config, etc. The goal is to verify every code path executes without error, not to produce meaningful results.
- **Always checkpoint models every 1-5 epochs** regardless of whether they are the best. This ensures we can inspect intermediate weights, know where training has gotten to, and recover from crashes. Save periodic checkpoints (e.g., `model_ep5.pth`, `model_ep10.pth`) alongside the `best_model.pth`.
- **Disk space on gpu0 is 350 GB total (not what `df` reports).** The disk tools overreport available space because they show the shared NFS volume capacity, not the rented amount. Our actual allocation is 350 GB on `/workspace/`. Check with `du -sh /workspace/` for real usage. If we need more, the user can rent more — ask first.
- **Remote GPU management must be done via background agents.** When setting up, monitoring, or managing remote GPUs (SSH, data transfer, launching training), always spin up a background Task agent per GPU rather than running SSH commands in the main conversation. This keeps the main context window clean and avoids it getting clogged with verbose transfer/install output.
- **Always launch remote training under `tmux`** so that SSH disconnections don't kill the process. Use: `ssh ... "tmux new-session -d -s train 'cd /workspace/vesuvius-kaggle-competition && bash scripts/launch_xxx.sh > logs/xxx.log 2>&1'"`. Verify with `tmux list-sessions` after launching. `nohup` over SSH is unreliable.
