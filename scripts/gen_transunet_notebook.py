#!/usr/bin/env python3
"""
Generate the TransUNet exploration Jupyter notebook with proper JSON formatting.

This script creates a well-formed .ipynb file with all cells properly formatted.
All code cells are validated with ast.parse() before saving.
"""

import json
import ast
import os
import sys
import uuid

# ── Notebook structure ──────────────────────────────────────────────

cells = []


def _cell_id():
    """Generate a unique cell ID (8-char hex)."""
    return uuid.uuid4().hex[:8]


def md(source):
    """Add a markdown cell."""
    cells.append({
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": _split_lines(source),
    })


def code(source):
    """Add a code cell, validating syntax first."""
    try:
        ast.parse(source)
    except SyntaxError as e:
        print(f"SYNTAX ERROR in code cell ({len(cells)}):", e)
        print("--- BEGIN CELL ---")
        for i, line in enumerate(source.splitlines(), 1):
            print(f"  {i:3d}: {line}")
        print("--- END CELL ---")
        sys.exit(1)
    cells.append({
        "cell_type": "code",
        "id": _cell_id(),
        "metadata": {},
        "source": _split_lines(source),
        "execution_count": None,
        "outputs": [],
    })


def _split_lines(source):
    """Split source into list of lines, each ending with newline except the last."""
    lines = source.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            result.append(line)
    return result


# ═══════════════════════════════════════════════════════════════════
# Cell 0: Title
# ═══════════════════════════════════════════════════════════════════
md("""\
# TransUNet Exploration

Visual exploration of the pretrained TransUNet SEResNeXt50 model.

**What this notebook shows:**
1. Cross-section views: CT image, probmap, prediction, ground truth
2. Probability histograms per volume (FG vs BG regions)
3. Error overlays: TP (green), FP (red), FN (blue)
4. Connected component size distributions
5. Comparison across different scrolls (including scroll 35360)
6. Multiple post-processing parameter settings

All figures saved to `plots/transunet_exploration/`.""")

# ═══════════════════════════════════════════════════════════════════
# Cell 1: Setup
# ═══════════════════════════════════════════════════════════════════
code("""\
import os
os.environ.setdefault('KERAS_BACKEND', 'torch')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import time
from pathlib import Path
from scipy.ndimage import (
    binary_closing, generate_binary_structure, binary_propagation,
    label as scipy_label, zoom as scipy_zoom,
)
import warnings
warnings.filterwarnings('ignore')

# Dry run mode: set DRY_RUN=1 env var to use fewer volumes
DRY_RUN = os.environ.get('DRY_RUN', '0') == '1'
if DRY_RUN:
    print('*** DRY RUN MODE -- using 2 volumes only ***')

ROOT = Path('/workspace/vesuvius-kaggle-competition')
TRAIN_IMG = ROOT / 'data' / 'train_images'
TRAIN_LBL = ROOT / 'data' / 'train_labels'
PROBMAP_DIR = ROOT / 'data' / 'transunet_probmaps'
PLOT_DIR = ROOT / 'plots' / 'transunet_exploration'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def savefig(fig, name):
    path = PLOT_DIR / f'{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

print('Setup complete. Plots will be saved to:', PLOT_DIR)""")

# ═══════════════════════════════════════════════════════════════════
# Cell 2: Load model header
# ═══════════════════════════════════════════════════════════════════
md("## 1. Load TransUNet model")

# ═══════════════════════════════════════════════════════════════════
# Cell 3: Load model
# ═══════════════════════════════════════════════════════════════════
code("""\
from medicai.models import TransUNet
from medicai.transforms import Compose, NormalizeIntensity
from medicai.utils.inference import SlidingWindowInference
import torch

WEIGHTS = ROOT / 'pretrained_weights' / 'transunet' / 'transunet.seresnext50.160px.comboloss.weights.h5'

model = TransUNet(
    input_shape=(160, 160, 160, 1),
    encoder_name='seresnext50',
    classifier_activation=None,
    num_classes=3,
)
model.load_weights(str(WEIGHTS))
print(f'Model loaded: {model.count_params() / 1e6:.1f}M params')

normalize = Compose([
    NormalizeIntensity(keys=['image'], nonzero=True, channel_wise=False),
])

swi = SlidingWindowInference(
    model,
    num_classes=3,
    roi_size=(160, 160, 160),
    sw_batch_size=1,
    mode='gaussian',
    overlap=0.50,
)

# Warmup
print('Warming up...')
_ = np.asarray(swi(np.zeros((1, 160, 160, 160, 1), dtype=np.float32)))
print('Ready.')""")

# ═══════════════════════════════════════════════════════════════════
# Cell 4: Helpers header
# ═══════════════════════════════════════════════════════════════════
md("## 2. Inference and post-processing helpers")

# ═══════════════════════════════════════════════════════════════════
# Cell 5: Helpers
# ═══════════════════════════════════════════════════════════════════
code("""\
import gc
import torch

def sigmoid_stable(x):
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

def logsumexp2(a, b):
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m) + 1e-12)

def predict_volume(vol_path):
    \"\"\"Load, normalize, run SWI, return binary probability map.\"\"\"
    img = tifffile.imread(vol_path).astype(np.float32)
    vol_5d = img[None, ..., None]
    vol_5d = normalize({'image': vol_5d})['image']

    with torch.no_grad():
        logits = np.asarray(swi(vol_5d))[0]  # (D,H,W,3)

    L0, L1, L2 = logits[..., 0], logits[..., 1], logits[..., 2]
    binary_logit = logsumexp2(L1, L2) - L0
    prob = sigmoid_stable(binary_logit)

    del logits, vol_5d, L0, L1, L2, binary_logit
    gc.collect()
    torch.cuda.empty_cache()

    return img, prob

def build_anisotropic_struct(z_radius=3, xy_radius=2):
    sz = 2 * z_radius + 1
    sxy = 2 * xy_radius + 1
    struct = np.zeros((sz, sxy, sxy), dtype=bool)
    cy, cx = xy_radius, xy_radius
    for y in range(sxy):
        for x in range(sxy):
            if (y - cy) ** 2 + (x - cx) ** 2 <= xy_radius ** 2:
                struct[:, y, x] = True
    return struct

def postprocess(prob, t_low=0.50, t_high=0.90, z_radius=3, xy_radius=2, dust=100):
    strong = prob >= t_high
    weak = prob >= t_low
    struct = generate_binary_structure(3, 3)
    if not strong.any():
        return np.zeros_like(prob, dtype=np.uint8)
    mask = binary_propagation(strong, structure=struct, mask=weak)
    struct_close = build_anisotropic_struct(z_radius, xy_radius)
    mask = binary_closing(mask, structure=struct_close)
    labeled, n = scipy_label(mask)
    if n > 0:
        sizes = np.bincount(labeled.ravel())
        small = sizes < dust
        small[0] = False
        mask[small[labeled]] = 0
    return mask.astype(np.uint8)

def make_overlay(pred, lbl):
    \"\"\"TP=green, FP=red, FN=blue. Ignores label=2.\"\"\"
    mask = (lbl != 2)
    gt = (lbl == 1)
    pred_b = pred.astype(bool)
    rgb = np.zeros((*pred.shape, 3), dtype=np.float32)
    rgb[pred_b & gt & mask] = [0, 1, 0]    # TP
    rgb[pred_b & ~gt & mask] = [1, 0, 0]   # FP
    rgb[~pred_b & gt & mask] = [0, 0.4, 1] # FN
    return rgb

print('Helpers ready.')""")

# ═══════════════════════════════════════════════════════════════════
# Cell 6: Select volumes header
# ═══════════════════════════════════════════════════════════════════
md("""\
## 3. Select volumes for exploration

We pick volumes from multiple scrolls:
- Scroll 26002 (our val set): 3 volumes
- Scroll 35360 (problematic in earlier testing): 2 volumes
- Scroll 34117 (largest scroll): 1 volume""")

# ═══════════════════════════════════════════════════════════════════
# Cell 7: Select volumes
# ═══════════════════════════════════════════════════════════════════
code("""\
train_df = pd.read_csv(ROOT / 'data' / 'train.csv')

# Select volumes
volume_picks = {}

if DRY_RUN:
    # Dry run: just 2 volumes for quick testing
    val_ids = train_df[train_df.scroll_id == 26002].id.tolist()
    volume_picks['26002_a'] = val_ids[0]
    volume_picks['26002_b'] = val_ids[1]
else:
    # Full run: volumes from multiple scrolls
    # Val scroll (26002)
    val_ids = train_df[train_df.scroll_id == 26002].id.tolist()
    volume_picks['26002_a'] = val_ids[0]  # 1407735 (public test volume!)
    volume_picks['26002_b'] = val_ids[1]  # 26894125
    volume_picks['26002_c'] = val_ids[4]  # 418613908

    # Problematic scroll (35360)
    s35_ids = train_df[train_df.scroll_id == 35360].id.tolist()
    volume_picks['35360_a'] = s35_ids[0]
    volume_picks['35360_b'] = s35_ids[1]

    # Largest scroll (34117)
    s34_ids = train_df[train_df.scroll_id == 34117].id.tolist()
    volume_picks['34117_a'] = s34_ids[0]

print(f'Selected {len(volume_picks)} volumes:')
for key, vid in volume_picks.items():
    scroll_id = key.split('_')[0]
    print(f'  {key}: vol_id={vid} (scroll {scroll_id})')""")

# ═══════════════════════════════════════════════════════════════════
# Cell 8: Run inference header
# ═══════════════════════════════════════════════════════════════════
md("## 4. Run inference on all selected volumes")

# ═══════════════════════════════════════════════════════════════════
# Cell 9: Run inference
# ═══════════════════════════════════════════════════════════════════
code("""\
vol_data = {}

for key, vid in volume_picks.items():
    print(f'\\nProcessing {key} (vol {vid})...')
    t0 = time.time()

    img, prob = predict_volume(TRAIN_IMG / f'{vid}.tif')
    lbl = tifffile.imread(TRAIN_LBL / f'{vid}.tif')

    # Also save probmap for reuse
    PROBMAP_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROBMAP_DIR / f'{vid}.npy', prob.astype(np.float16))

    # Generate predictions with different post-processing settings
    pred_competitor = postprocess(prob, t_low=0.50, t_high=0.90, z_radius=3, xy_radius=2, dust=100)
    pred_our_old = postprocess(prob, t_low=0.35, t_high=0.75, z_radius=2, xy_radius=1, dust=64)
    pred_aggressive = postprocess(prob, t_low=0.40, t_high=0.85, z_radius=3, xy_radius=2, dust=100)

    vol_data[key] = {
        'vid': vid,
        'img': img,
        'lbl': lbl,
        'prob': prob,
        'pred_competitor': pred_competitor,
        'pred_our_old': pred_our_old,
        'pred_aggressive': pred_aggressive,
    }

    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.0f}s. prob range: [{prob.min():.3f}, {prob.max():.3f}], '
          f'fg_competitor={pred_competitor.sum()}, fg_our_old={pred_our_old.sum()}')

    # Free GPU memory between volumes
    gc.collect()
    torch.cuda.empty_cache()""")

# ═══════════════════════════════════════════════════════════════════
# Cell 10: Cross-sections header
# ═══════════════════════════════════════════════════════════════════
md("""\
## 5. Cross-section visualizations

For each volume, show the Z-slice with most foreground:
- Row 1: CT image | probmap (hot colormap) | competitor prediction | our old prediction
- Row 2: ground truth | error overlay (competitor) | error overlay (our old) | error overlay (aggressive)""")

# ═══════════════════════════════════════════════════════════════════
# Cell 11: Cross-sections
# ═══════════════════════════════════════════════════════════════════
code("""\
for key, data in vol_data.items():
    vid = data['vid']
    img = data['img']
    lbl = data['lbl']
    prob = data['prob']

    # Find Z-slice with most foreground
    fg_counts = (lbl == 1).sum(axis=(1, 2))
    z_best = int(np.argmax(fg_counts))

    # Also show a middle slice for context
    z_mid = img.shape[0] // 2

    for z, z_label in [(z_best, 'best_fg'), (z_mid, 'center')]:
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        scroll_id = key.split('_')[0]
        fig.suptitle(
            f'TransUNet -- Scroll {scroll_id}, Vol {vid}, Z={z} ({z_label})',
            fontsize=16,
        )

        # Row 1: CT, probmap, predictions
        axes[0, 0].imshow(img[z], cmap='gray')
        axes[0, 0].set_title('CT Image')
        axes[0, 0].axis('off')

        im = axes[0, 1].imshow(prob[z], cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title(f'TransUNet probmap (max={prob[z].max():.3f})')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], shrink=0.8)

        axes[0, 2].imshow(data['pred_competitor'][z], cmap='gray')
        axes[0, 2].set_title('Pred: Competitor PP\\n(T_low=0.50, T_high=0.90)')
        axes[0, 2].axis('off')

        axes[0, 3].imshow(data['pred_our_old'][z], cmap='gray')
        axes[0, 3].set_title('Pred: Our old PP\\n(T_low=0.35, T_high=0.75)')
        axes[0, 3].axis('off')

        # Row 2: GT, overlays
        gt_vis = np.zeros((*lbl[z].shape, 3), dtype=np.float32)
        gt_vis[lbl[z] == 1] = [0, 1, 0]
        gt_vis[lbl[z] == 2] = [0.5, 0.5, 0.5]
        axes[1, 0].imshow(gt_vis)
        axes[1, 0].set_title('Ground Truth (green=fg, gray=unlabeled)')
        axes[1, 0].axis('off')

        preds = [
            ('Competitor PP', 'pred_competitor'),
            ('Our old PP', 'pred_our_old'),
            ('Aggressive PP', 'pred_aggressive'),
        ]
        for j, (name, pkey) in enumerate(preds):
            overlay = make_overlay(data[pkey][z], lbl[z])
            axes[1, j + 1].imshow(overlay)
            axes[1, j + 1].set_title(f'{name}\\n(G=TP, R=FP, B=FN)')
            axes[1, j + 1].axis('off')

        savefig(fig, f'crosssection_{key}_{z_label}_z{z}')""")

# ═══════════════════════════════════════════════════════════════════
# Cell 12: Multi-axis header
# ═══════════════════════════════════════════════════════════════════
md("""\
## 6. Multi-axis cross-sections

Show Y and X slices to see surface continuity from different angles.""")

# ═══════════════════════════════════════════════════════════════════
# Cell 13: Multi-axis
# ═══════════════════════════════════════════════════════════════════
code("""\
for key in list(vol_data.keys())[:3]:  # First 3 volumes
    data = vol_data[key]
    vid = data['vid']
    img = data['img']
    lbl = data['lbl']
    prob = data['prob']
    pred = data['pred_competitor']
    scroll_id = key.split('_')[0]

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    fig.suptitle(f'TransUNet multi-axis -- Scroll {scroll_id}, Vol {vid}', fontsize=16)

    for row, (axis, axis_name) in enumerate([
        (0, 'Z (axial)'), (1, 'Y (coronal)'), (2, 'X (sagittal)')
    ]):
        # Find slice with most foreground along this axis
        sum_axes = tuple(i for i in range(3) if i != axis)
        fg_counts = (lbl == 1).sum(axis=sum_axes)
        s = int(np.argmax(fg_counts))

        slc = [slice(None)] * 3
        slc[axis] = s
        slc = tuple(slc)

        axes[row, 0].imshow(img[slc], cmap='gray', aspect='auto')
        axes[row, 0].set_title(f'{axis_name} s={s}: CT')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(prob[slc], cmap='hot', vmin=0, vmax=1, aspect='auto')
        axes[row, 1].set_title(f'{axis_name}: Probmap')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(pred[slc], cmap='gray', aspect='auto')
        axes[row, 2].set_title(f'{axis_name}: Prediction')
        axes[row, 2].axis('off')

        overlay = make_overlay(pred[slc], lbl[slc])
        axes[row, 3].imshow(overlay, aspect='auto')
        axes[row, 3].set_title(f'{axis_name}: Error overlay')
        axes[row, 3].axis('off')

    savefig(fig, f'multiaxis_{key}')""")

# ═══════════════════════════════════════════════════════════════════
# Cell 14: Probability histograms header
# ═══════════════════════════════════════════════════════════════════
md("""\
## 7. Probability histograms

Shows the distribution of TransUNet probabilities in foreground vs background regions.
Critical for choosing T_low and T_high thresholds.""")

# ═══════════════════════════════════════════════════════════════════
# Cell 15: Probability histograms
# ═══════════════════════════════════════════════════════════════════
code("""\
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, (key, data) in enumerate(vol_data.items()):
    if i >= 6:
        break
    ax = axes[i]
    lbl = data['lbl']
    prob = data['prob']
    vid = data['vid']
    scroll_id = key.split('_')[0]

    fg_mask = (lbl == 1)
    bg_mask = (lbl == 0)

    fg_probs = prob[fg_mask]
    bg_probs = prob[bg_mask]

    ax.hist(bg_probs.ravel(), bins=100, alpha=0.5,
            label=f'BG (n={bg_mask.sum():,})',
            color='blue', density=True, range=(0, 1))
    ax.hist(fg_probs.ravel(), bins=100, alpha=0.5,
            label=f'FG (n={fg_mask.sum():,})',
            color='green', density=True, range=(0, 1))

    # Threshold lines
    for t, c, ls in [(0.50, 'orange', '--'), (0.90, 'red', '--'), (0.75, 'purple', ':')]:
        ax.axvline(t, color=c, ls=ls, alpha=0.8, label=f'T={t}')

    ax.set_title(f'Scroll {scroll_id} -- Vol {vid}')
    ax.set_xlabel('Probability')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7)

    # Print key stats
    if fg_probs.size > 0:
        print(f'{key}: FG p50={np.median(fg_probs):.3f}, '
              f'p95={np.percentile(fg_probs, 95):.3f}, '
              f'max={fg_probs.max():.3f}, '
              f'pct>0.90={100 * (fg_probs > 0.90).mean():.1f}%, '
              f'pct>0.75={100 * (fg_probs > 0.75).mean():.1f}%')

fig.suptitle('TransUNet probability distributions: FG vs BG', fontsize=16)
plt.tight_layout()
savefig(fig, 'probability_histograms')""")

# ═══════════════════════════════════════════════════════════════════
# Cell 16: CC analysis header
# ═══════════════════════════════════════════════════════════════════
md("""\
## 8. Connected component analysis

VOI score (35% of metric) depends on connected component quality.
Shows component counts and size distributions.""")

# ═══════════════════════════════════════════════════════════════════
# Cell 17: CC analysis table
# ═══════════════════════════════════════════════════════════════════
code("""\
header = (f"{'Key':<12s} {'Vol':>12s} {'GT_CC':>6s} {'Comp_CC':>8s} "
          f"{'Old_CC':>7s} {'Aggr_CC':>8s}  {'GT_fg%':>7s} {'Pred_fg%':>9s}")
print(header)
print('-' * 80)

cc_data = {}
for key, data in vol_data.items():
    lbl = data['lbl']
    gt_binary = (lbl == 1).astype(np.uint8)

    _, n_gt = scipy_label(gt_binary)
    _, n_comp = scipy_label(data['pred_competitor'])
    _, n_old = scipy_label(data['pred_our_old'])
    _, n_aggr = scipy_label(data['pred_aggressive'])

    gt_pct = 100 * gt_binary.sum() / gt_binary.size
    pred_pct = 100 * data['pred_competitor'].sum() / data['pred_competitor'].size

    print(f"{key:<12s} {data['vid']:>12d} {n_gt:>6d} {n_comp:>8d} "
          f"{n_old:>7d} {n_aggr:>8d}  {gt_pct:>6.2f}% {pred_pct:>8.2f}%")

    cc_data[key] = {
        'gt_n': n_gt,
        'comp_n': n_comp,
        'old_n': n_old,
        'aggr_n': n_aggr,
    }

print()
print('Target: prediction CC count should match GT CC count for best VOI.')""")

# ═══════════════════════════════════════════════════════════════════
# Cell 18: CC size distributions
# ═══════════════════════════════════════════════════════════════════
code("""\
# CC size distribution for competitor PP
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, (key, data) in enumerate(vol_data.items()):
    if i >= 6:
        break
    ax = axes[i]
    pred = data['pred_competitor']
    lbl = data['lbl']
    gt_binary = (lbl == 1).astype(np.uint8)
    vid = data['vid']
    scroll_id = key.split('_')[0]

    # GT CC sizes
    gt_labeled, n_gt = scipy_label(gt_binary)
    gt_sizes = np.bincount(gt_labeled.ravel())[1:] if n_gt > 0 else []

    # Pred CC sizes
    pred_labeled, n_pred = scipy_label(pred)
    pred_sizes = np.bincount(pred_labeled.ravel())[1:] if n_pred > 0 else []

    if len(gt_sizes) > 0:
        ax.hist(np.log10(gt_sizes + 1), bins=30, alpha=0.6,
                label=f'GT ({n_gt} CCs)', color='green')
    if len(pred_sizes) > 0:
        ax.hist(np.log10(pred_sizes + 1), bins=30, alpha=0.6,
                label=f'Pred ({n_pred} CCs)', color='blue')

    ax.set_title(f'Scroll {scroll_id} -- Vol {vid}')
    ax.set_xlabel('log10(CC size)')
    ax.set_ylabel('Count')
    ax.legend()

fig.suptitle('Connected component size distributions (competitor PP)', fontsize=16)
plt.tight_layout()
savefig(fig, 'cc_size_distributions')""")

# ═══════════════════════════════════════════════════════════════════
# Cell 19: PP comparison header
# ═══════════════════════════════════════════════════════════════════
md("""\
## 9. Post-processing comparison

Compare different post-processing parameter sets side by side.""")

# ═══════════════════════════════════════════════════════════════════
# Cell 20: PP comparison
# ═══════════════════════════════════════════════════════════════════
code("""\
pp_configs = [
    ('Competitor\\n(T=0.50/0.90, z3xy2)',
     dict(t_low=0.50, t_high=0.90, z_radius=3, xy_radius=2, dust=100)),
    ('Our old\\n(T=0.35/0.75, z2xy1)',
     dict(t_low=0.35, t_high=0.75, z_radius=2, xy_radius=1, dust=64)),
    ('High conf\\n(T=0.60/0.95, z3xy2)',
     dict(t_low=0.60, t_high=0.95, z_radius=3, xy_radius=2, dust=100)),
    ('Low threshold\\n(T=0.30/0.80, z3xy2)',
     dict(t_low=0.30, t_high=0.80, z_radius=3, xy_radius=2, dust=100)),
]

for key in list(vol_data.keys())[:3]:
    data = vol_data[key]
    vid = data['vid']
    lbl = data['lbl']
    prob = data['prob']
    scroll_id = key.split('_')[0]

    # Find best Z slice
    fg_counts = (lbl == 1).sum(axis=(1, 2))
    z = int(np.argmax(fg_counts))

    fig, axes = plt.subplots(2, len(pp_configs), figsize=(6 * len(pp_configs), 12))
    fig.suptitle(
        f'Post-processing comparison -- Scroll {scroll_id}, Vol {vid}, Z={z}',
        fontsize=16,
    )

    for j, (name, params) in enumerate(pp_configs):
        pred = postprocess(prob, **params)

        # Row 1: prediction
        axes[0, j].imshow(pred[z], cmap='gray')
        axes[0, j].set_title(f'{name}\\nfg={pred.sum():,}')
        axes[0, j].axis('off')

        # Row 2: error overlay
        overlay = make_overlay(pred[z], lbl[z])
        axes[1, j].imshow(overlay)
        axes[1, j].set_title('G=TP, R=FP, B=FN')
        axes[1, j].axis('off')

    savefig(fig, f'pp_comparison_{key}')""")

# ═══════════════════════════════════════════════════════════════════
# Cell 21: Cross-scroll comparison header
# ═══════════════════════════════════════════════════════════════════
md("""\
## 10. Cross-scroll probability comparison

Compare probability characteristics across different scrolls.
This helps understand if the pretrained model generalizes well.""")

# ═══════════════════════════════════════════════════════════════════
# Cell 22: Cross-scroll comparison
# ═══════════════════════════════════════════════════════════════════
code("""\
# Summary statistics per scroll
header = (f"{'Key':<12s} {'Scroll':>8s} {'prob_max':>10s} {'fg_p50':>8s} "
          f"{'fg_p95':>8s} {'fg_pct>0.9':>12s} {'bg_pct>0.5':>12s}")
print(header)
print('-' * 75)

for key, data in vol_data.items():
    prob = data['prob']
    lbl = data['lbl']
    scroll_id = key.split('_')[0]

    fg_probs = prob[lbl == 1]
    bg_probs = prob[lbl == 0]

    if fg_probs.size > 0:
        fg_p50 = np.median(fg_probs)
        fg_p95 = np.percentile(fg_probs, 95)
        fg_gt90 = 100 * (fg_probs > 0.90).mean()
    else:
        fg_p50 = fg_p95 = fg_gt90 = 0

    bg_gt50 = 100 * (bg_probs > 0.50).mean() if bg_probs.size > 0 else 0

    print(f"{key:<12s} {scroll_id:>8s} {prob.max():>10.3f} {fg_p50:>8.3f} "
          f"{fg_p95:>8.3f} {fg_gt90:>11.1f}% {bg_gt50:>11.1f}%")

print()
print('Key questions:')
print('  - Is prob_max > 0.90 for all scrolls? (needed for T_high=0.90)')
print('  - Is fg_pct>0.90 high? (higher = more confident predictions)')
print('  - Is bg_pct>0.50 low? (lower = fewer false positives)')""")

# ═══════════════════════════════════════════════════════════════════
# Cell 23: Slice gallery header
# ═══════════════════════════════════════════════════════════════════
md("""\
## 11. Probmap slice gallery

Every 20th Z-slice for one volume, showing how the surface evolves through depth.""")

# ═══════════════════════════════════════════════════════════════════
# Cell 24: Slice gallery
# ═══════════════════════════════════════════════════════════════════
code("""\
# Use first val volume
key = list(vol_data.keys())[0]
data = vol_data[key]
img = data['img']
prob = data['prob']
lbl = data['lbl']
pred = data['pred_competitor']
vid = data['vid']
scroll_id = key.split('_')[0]

D = img.shape[0]
slice_step = max(D // 16, 1)
z_slices = list(range(0, D, slice_step))

n_cols = 4
n_rows = len(z_slices)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
fig.suptitle(
    f'Slice gallery -- Scroll {scroll_id}, Vol {vid} (every {slice_step} slices)',
    fontsize=16,
)

for i, z in enumerate(z_slices):
    axes[i, 0].imshow(img[z], cmap='gray')
    axes[i, 0].set_ylabel(f'Z={z}')
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])

    axes[i, 1].imshow(prob[z], cmap='hot', vmin=0, vmax=1)
    axes[i, 1].axis('off')

    axes[i, 2].imshow(pred[z], cmap='gray')
    axes[i, 2].axis('off')

    gt_vis = np.zeros((*lbl[z].shape, 3), dtype=np.float32)
    gt_vis[lbl[z] == 1] = [0, 1, 0]
    gt_vis[lbl[z] == 2] = [0.5, 0.5, 0.5]
    axes[i, 3].imshow(gt_vis)
    axes[i, 3].axis('off')

axes[0, 0].set_title('CT')
axes[0, 1].set_title('Probmap')
axes[0, 2].set_title('Prediction')
axes[0, 3].set_title('Ground Truth')

savefig(fig, f'slice_gallery_{key}')""")

# ═══════════════════════════════════════════════════════════════════
# Cell 25: Summary header
# ═══════════════════════════════════════════════════════════════════
md("## 12. Summary")

# ═══════════════════════════════════════════════════════════════════
# Cell 26: Summary
# ═══════════════════════════════════════════════════════════════════
code("""\
import glob

print('=' * 60)
print('TRANSUNET EXPLORATION SUMMARY')
print('=' * 60)
print()
print(f'Volumes explored: {len(vol_data)}')
print(f'Plots saved to: {PLOT_DIR}')
print()

# List all saved plots
plots = sorted(glob.glob(str(PLOT_DIR / '*.png')))
print(f'Generated {len(plots)} plot files:')
for p in plots:
    print(f'  {Path(p).name}')
print()
print('Probmaps saved to:', PROBMAP_DIR)
probmap_files = list(PROBMAP_DIR.glob('*.npy'))
print(f'Probmap files: {len(probmap_files)}')""")

# ═══════════════════════════════════════════════════════════════════
# Assemble and write notebook
# ═══════════════════════════════════════════════════════════════════

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

# Final validation: check all code cells parse
print(f"Validating {len(cells)} cells...")
n_code = 0
n_md = 0
for i, cell in enumerate(cells):
    if cell["cell_type"] == "code":
        n_code += 1
        src = "".join(cell["source"])
        try:
            ast.parse(src)
        except SyntaxError as e:
            print(f"FATAL: Cell {i} has syntax error: {e}")
            sys.exit(1)
    else:
        n_md += 1

print(f"  {n_code} code cells - all pass ast.parse()")
print(f"  {n_md} markdown cells")

# Write notebook
out_path = "/workspace/vesuvius-kaggle-competition/notebooks/analysis/transunet_exploration.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"\nNotebook written to: {out_path}")
print(f"File size: {os.path.getsize(out_path):,} bytes")
