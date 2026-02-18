We are participating in a Kaggle machine learning competition.

**Always start every conversation by reading NOTES.md** — it contains the full project history, run results, and current status.

## Engineering standards

- **Overnight/background pipelines must always have a `--dry-run` flag** that tests each component of the pipeline on a small subset of data (e.g., 1 volume instead of 20). This catches missing dependencies, wrong paths, and import errors before committing to a long unattended run.
- **All background pipelines that will run for more than 30 minutes must be fully dry-run tested before they run.** No exceptions — every phase must pass before launching the full pipeline.
- **Dry runs should be the smallest possible run that exercises all the code that will run.** Use 1-2 volumes, 1 config, etc. The goal is to verify every code path executes without error, not to produce meaningful results.
- **Always checkpoint models every 1-5 epochs** regardless of whether they are the best. This ensures we can inspect intermediate weights, know where training has gotten to, and recover from crashes. Save periodic checkpoints (e.g., `model_ep5.pth`, `model_ep10.pth`) alongside the `best_model.pth`.
