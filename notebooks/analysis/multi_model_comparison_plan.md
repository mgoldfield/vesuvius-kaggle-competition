# Multi-Model Comparison Notebook Plan

`notebooks/analysis/multi_model_comparison.ipynb`

Modeled after `transunet_exploration.ipynb`. Compares models side-by-side to understand the thinning/SDice tradeoff.

## Models to compare

| Key | Weights | Status |
|-----|---------|--------|
| pretrained | `pretrained_weights/transunet/transunet.seresnext50.160px.comboloss.weights.h5` | Ready |
| dist_sq_ep5 | `checkpoints/transunet_dist_sq/transunet_ep5.weights.h5` | Ready |
| discrim_dist_sq | `checkpoints/transunet_discrim_dist_sq/transunet_best.weights.h5` | Training on gpu1 |
| frozen_dist_sq | `checkpoints/transunet_frozen_dist_sq/transunet_best.weights.h5` | Training on gpu2 |

Skip any model whose weights file doesn't exist.

## Sections

1. **Setup** — Load all available models. Same helpers as exploration notebook (sigmoid, logsumexp, postprocess, make_overlay, savefig).

2. **Inference** — Run SWI on 4-6 volumes (same picks as original: 2x scroll 26002, 2x 35360, 1x 34117). Store probmaps per model per volume. Support DRY_RUN=1 for 2 volumes.

3. **Cross-section comparison** — For each volume, one row per model showing: CT | probmap | prediction | error overlay. Direct visual comparison of thickness differences.

4. **Thickness analysis** — Per-volume: count FG voxels per model, measure mean surface thickness along Z-axis (sum of consecutive FG voxels per column). This directly tests the thinning hypothesis. Bar charts comparing models.

5. **Probability histograms** — Overlay FG/BG probability distributions for all models on the same axes. Shows how confidence shifts between models.

6. **SDice vs VOI tradeoff** — Scatter plot of SDice vs VOI per volume per model, showing the tradeoff frontier. This is the key insight: do fine-tuned models improve VOI at the cost of SDice?

7. **PP sensitivity** — For each model, try 4 PP configs (competitor default, high t_low, minimal closing, aggressive) and show how scores change. Tests whether thinner models benefit from different PP params.

8. **Summary table** — Mean comp/topo/sdice/voi per model. Plus per-scroll breakdown.

## Key questions this notebook answers

- How much thinner are fine-tuned predictions vs pretrained?
- Does discriminative LR preserve surface alignment (SDice) better than full fine-tuning?
- Does frozen encoder preserve encoder features while allowing decoder adaptation?
- Can post-processing recover SDice for thinned models?
- Which model has the best SDice/VOI tradeoff?
