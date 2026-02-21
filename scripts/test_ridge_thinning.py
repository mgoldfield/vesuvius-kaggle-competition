#!/usr/bin/env python3
"""
Test probability ridge extraction as a thinning post-processing strategy.

For each (x,y) column in the probmap, finds probability peaks along Z and
keeps only voxels within ±k of each peak. This produces thin predictions
that follow the probability ridge rather than bluntly thresholding.

No GPU needed — operates on pre-computed probmaps.

Usage:
    python scripts/test_ridge_thinning.py                  # 5 volumes
    python scripts/test_ridge_thinning.py --n-eval 20      # more volumes
    python scripts/test_ridge_thinning.py --dry-run         # 2 volumes
"""

import os

try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

import argparse
import time
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from scipy.ndimage import (
    binary_closing, generate_binary_structure, binary_propagation,
    label as scipy_label, zoom as scipy_zoom,
)
from scipy.signal import find_peaks
from topometrics.leaderboard import compute_leaderboard_score

ROOT = Path("/workspace/vesuvius-kaggle-competition")
TRAIN_LBL = ROOT / "data" / "train_labels"
PROBMAP_DIR = ROOT / "data" / "transunet_probmaps"
VAL_SCROLL = 26002


def ridge_extract(prob, k, min_peak_height=0.3, min_peak_distance=5):
    """
    Extract probability ridge: for each (x,y) column, find peaks along Z
    and keep only voxels within ±k of each peak.

    Args:
        prob: (D, H, W) probability map
        k: half-width around each peak (k=1 means ±1, so 3 voxels total)
        min_peak_height: minimum probability to count as a peak
        min_peak_distance: minimum Z distance between peaks (for multi-surface)

    Returns:
        mask: (D, H, W) binary uint8
    """
    D, H, W = prob.shape
    mask = np.zeros_like(prob, dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            col = prob[:, y, x]

            # Find peaks in this column
            peaks, properties = find_peaks(
                col,
                height=min_peak_height,
                distance=min_peak_distance,
            )

            if len(peaks) == 0:
                continue

            # Keep ±k voxels around each peak
            for p in peaks:
                z_lo = max(0, p - k)
                z_hi = min(D, p + k + 1)
                # Only keep voxels that are also above a minimum threshold
                for z in range(z_lo, z_hi):
                    if col[z] >= 0.1:  # very low floor to avoid noise
                        mask[z, y, x] = 1

    return mask


def ridge_extract_fast(prob, k, min_peak_height=0.3):
    """
    Fast vectorized ridge extraction using argmax per column.
    Handles single-surface case (one peak per column). Much faster than
    the per-column find_peaks approach.

    For multi-surface: falls back to per-column peak detection.
    """
    D, H, W = prob.shape
    mask = np.zeros_like(prob, dtype=np.uint8)

    # Find max prob along Z for each (y, x)
    max_prob = prob.max(axis=0)  # (H, W)
    argmax_z = prob.argmax(axis=0)  # (H, W) — Z index of peak

    # Only process columns with significant probability
    active = max_prob >= min_peak_height  # (H, W)

    # For each active column, mark ±k around the peak
    ys, xs = np.where(active)
    peak_zs = argmax_z[ys, xs]

    for i in range(len(ys)):
        y, x, pz = ys[i], xs[i], peak_zs[i]
        z_lo = max(0, pz - k)
        z_hi = min(D, pz + k + 1)
        mask[z_lo:z_hi, y, x] = 1

    # Mask out very low probability voxels
    mask[prob < 0.1] = 0

    return mask


def ridge_extract_multisurface(prob, k, min_peak_height=0.3, min_peak_distance=8):
    """
    Ridge extraction supporting multiple surfaces per column.
    Uses a chunked approach for reasonable speed.
    """
    D, H, W = prob.shape
    mask = np.zeros_like(prob, dtype=np.uint8)

    # Only process columns with significant max probability
    max_prob = prob.max(axis=0)
    active_ys, active_xs = np.where(max_prob >= min_peak_height)

    for i in range(len(active_ys)):
        y, x = active_ys[i], active_xs[i]
        col = prob[:, y, x]

        peaks, _ = find_peaks(col, height=min_peak_height, distance=min_peak_distance)

        if len(peaks) == 0:
            # Fallback: use argmax
            pz = col.argmax()
            if col[pz] >= min_peak_height:
                peaks = [pz]
            else:
                continue

        for pz in peaks:
            z_lo = max(0, pz - k)
            z_hi = min(D, pz + k + 1)
            mask[z_lo:z_hi, y, x] = 1

    # Floor
    mask[prob < 0.1] = 0
    return mask


def standard_postprocess(prob, t_low=0.70, t_high=0.90, z_radius=3, xy_radius=2, dust=100):
    """Standard hysteresis + closing + dust removal."""
    strong = prob >= t_high
    weak = prob >= t_low
    struct = generate_binary_structure(3, 3)
    if not strong.any():
        return np.zeros_like(prob, dtype=np.uint8)
    mask = binary_propagation(strong, structure=struct, mask=weak)

    sz = 2 * z_radius + 1
    sxy = 2 * xy_radius + 1
    struct_close = np.zeros((sz, sxy, sxy), dtype=bool)
    cy, cx = xy_radius, xy_radius
    for y_i in range(sxy):
        for x_i in range(sxy):
            if (y_i - cy) ** 2 + (x_i - cx) ** 2 <= xy_radius ** 2:
                struct_close[:, y_i, x_i] = True
    mask = binary_closing(mask, structure=struct_close)

    labeled, n = scipy_label(mask)
    if n > 0:
        sizes = np.bincount(labeled.ravel())
        small = sizes < dust
        small[0] = False
        mask[small[labeled]] = 0

    return mask.astype(np.uint8)


def score_volume(pred, lbl):
    report = compute_leaderboard_score(
        pred, lbl, ignore_label=2, spacing=(1, 1, 1),
        surface_tolerance=2.0, voi_alpha=0.3,
        combine_weights=(0.3, 0.35, 0.35),
    )
    return {
        'comp_score': report.score,
        'topo': report.topo.toposcore,
        'sdice': report.surface_dice,
        'voi': report.voi.voi_score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-eval', type=int, default=5)
    parser.add_argument('--probmap-dir', type=str, default=str(PROBMAP_DIR))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--fast-only', action='store_true',
                        help='Only test fast single-surface ridge (skip slow multi-surface)')
    args = parser.parse_args()

    if args.dry_run:
        args.n_eval = 2

    probmap_dir = Path(args.probmap_dir)

    # Get val volume IDs with both probmaps and labels
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    probmap_ids = set(int(p.stem) for p in probmap_dir.glob("*.npy"))
    label_ids = set(int(p.stem) for p in TRAIN_LBL.glob("*.tif"))
    available = probmap_ids & label_ids

    val_df = train_df[(train_df.scroll_id == VAL_SCROLL) & train_df.id.isin(available)]
    eval_ids = val_df.id.tolist()[:args.n_eval]

    print(f"Ridge thinning test on {len(eval_ids)} volumes")
    print(f"Probmap dir: {probmap_dir}")

    # Load data
    print("Loading probmaps and labels...")
    data = {}
    for vid in eval_ids:
        prob = np.load(probmap_dir / f"{vid}.npy").astype(np.float32)
        lbl = tifffile.imread(TRAIN_LBL / f"{vid}.tif")
        data[vid] = (prob, lbl)
        gt_pct = 100 * (lbl == 1).sum() / lbl.size
        pred_pct = 100 * (prob > 0.5).sum() / prob.size
        print(f"  vol {vid}: GT {gt_pct:.1f}% FG, pred>0.5 {pred_pct:.1f}% FG, "
              f"thickness ratio ~{pred_pct / max(gt_pct, 0.1):.1f}x")
    print()

    # Define configs to test
    configs = []

    # Baseline: standard post-processing
    configs.append({
        'name': 'baseline_t70_t90',
        'method': 'standard',
        'params': {'t_low': 0.70, 't_high': 0.90},
    })

    # Ridge extraction with different k values (single-surface, fast)
    for k in [0, 1, 2, 3, 5]:
        configs.append({
            'name': f'ridge_fast_k{k}',
            'method': 'ridge_fast',
            'params': {'k': k},
        })

    # Multi-surface ridge extraction (slower but handles parallel sheets)
    if not args.fast_only:
        for k in [1, 2, 3]:
            configs.append({
                'name': f'ridge_multi_k{k}',
                'method': 'ridge_multi',
                'params': {'k': k},
            })

    # Ridge + closing (fill small gaps after thinning)
    configs.append({
        'name': 'ridge_fast_k2_close',
        'method': 'ridge_fast_close',
        'params': {'k': 2, 'z_radius': 1, 'xy_radius': 1},
    })

    # Higher threshold baselines for comparison
    for t_low in [0.80, 0.85, 0.90]:
        configs.append({
            'name': f'baseline_t{int(t_low*100)}',
            'method': 'standard',
            'params': {'t_low': t_low, 't_high': 0.90 if t_low < 0.90 else 0.95},
        })

    print(f"Testing {len(configs)} configurations")
    print(f"{'Config':<25} {'Comp':>7} {'Topo':>7} {'SDice':>7} {'VOI':>7} {'FG%':>6} {'Time':>6}")
    print("-" * 75)

    all_results = []

    for cfg in configs:
        scores_list = []
        fg_pcts = []
        t0 = time.time()

        for vid in eval_ids:
            prob, lbl = data[vid]

            if cfg['method'] == 'standard':
                pred = standard_postprocess(prob, **cfg['params'])
            elif cfg['method'] == 'ridge_fast':
                pred = ridge_extract_fast(prob, **cfg['params'])
            elif cfg['method'] == 'ridge_multi':
                pred = ridge_extract_multisurface(prob, **cfg['params'])
            elif cfg['method'] == 'ridge_fast_close':
                k = cfg['params']['k']
                pred = ridge_extract_fast(prob, k=k)
                # Apply light closing
                zr = cfg['params']['z_radius']
                xyr = cfg['params']['xy_radius']
                sz = 2 * zr + 1
                sxy = 2 * xyr + 1
                struct_close = np.zeros((sz, sxy, sxy), dtype=bool)
                cy, cx = xyr, xyr
                for y_i in range(sxy):
                    for x_i in range(sxy):
                        if (y_i - cy) ** 2 + (x_i - cx) ** 2 <= xyr ** 2:
                            struct_close[:, y_i, x_i] = True
                pred = binary_closing(pred, structure=struct_close).astype(np.uint8)

            fg_pct = 100 * pred.sum() / pred.size
            fg_pcts.append(fg_pct)

            s = score_volume(pred, lbl)
            scores_list.append(s)

        elapsed = time.time() - t0
        mean_scores = {k: np.mean([s[k] for s in scores_list])
                       for k in ['comp_score', 'topo', 'sdice', 'voi']}
        mean_fg = np.mean(fg_pcts)

        print(f"{cfg['name']:<25} {mean_scores['comp_score']:>7.4f} "
              f"{mean_scores['topo']:>7.4f} {mean_scores['sdice']:>7.4f} "
              f"{mean_scores['voi']:>7.4f} {mean_fg:>5.1f}% {elapsed:>5.0f}s")

        all_results.append({
            'config': cfg['name'],
            'method': cfg['method'],
            **mean_scores,
            'fg_pct': mean_fg,
            'time': elapsed,
        })

    # Summary
    results_df = pd.DataFrame(all_results).sort_values('comp_score', ascending=False)

    print(f"\n{'='*75}")
    print("RANKED BY COMP_SCORE")
    print(f"{'='*75}")
    for _, row in results_df.iterrows():
        marker = " ***" if row['comp_score'] > all_results[0]['comp_score'] else ""
        print(f"  {row['config']:<25} comp={row['comp_score']:.4f} "
              f"topo={row['topo']:.4f} sdice={row['sdice']:.4f} "
              f"voi={row['voi']:.4f} fg={row['fg_pct']:.1f}%{marker}")

    out_path = ROOT / "logs" / "ridge_thinning_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
