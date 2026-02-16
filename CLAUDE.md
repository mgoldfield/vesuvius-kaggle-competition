We are participating in a Kaggle machine learning competition.

**Always start every conversation by reading NOTES.md** — it contains the full project history, run results, and current status.

## Engineering standards

- **Overnight/background pipelines must always have a `--dry-run` flag** that tests each component of the pipeline on a small subset of data (e.g., 1 volume instead of 20). This catches missing dependencies, wrong paths, and import errors before committing to a long unattended run.
