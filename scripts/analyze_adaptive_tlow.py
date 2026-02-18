#!/usr/bin/env python3
"""
Analyze whether T_low should be adaptive (per-volume) or fixed.

For each volume:
1. Sweep T_low values, find the optimal one
2. Compute probability distribution statistics (potential predictors)
3. Check correlation between stats and optimal T_low

No GPU needed — uses pre-computed probmaps.

Usage:
    python scripts/analyze_adaptive_tlow.py
    python scripts/analyze_adaptive_tlow.py --n-vols 20
    python scripts/analyze_adaptive_tlow.py --dry-run
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
    label as scipy_label,
)
from topometrics.leaderboard import compute_leaderboard_score

ROOT = Path("/workspace/vesuvius-kaggle-competition")
TRAIN_LBL = ROOT / "data" / "train_labels"
PROBMAP_DIR = ROOT / "data" / "transunet_probmaps"


def build_anisotropic_struct(z_radius, xy_radius):
    sz = 2 * z_radius + 1
    sxy = 2 * xy_radius + 1
    struct = np.zeros((sz, sxy, sxy), dtype=bool)
    cy, cx = xy_radius, xy_radius
    for y in range(sxy):
        for x in range(sxy):
            if (y - cy) ** 2 + (x - cx) ** 2 <= xy_radius ** 2:
                struct[:, y, x] = True
    return struct


def postprocess(prob, t_low, t_high=0.90, z_radius=3, xy_radius=2, dust=100):
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


def compute_prob_stats(prob, lbl):
    """Compute per-volume probability statistics — potential T_low predictors."""
    all_probs = prob.ravel()
    fg_probs = prob[lbl == 1] if (lbl == 1).any() else np.array([0.0])
    bg_probs = prob[lbl == 0] if (lbl == 0).any() else np.array([0.0])

    # Overall distribution
    stats = {
        'prob_max': float(prob.max()),
        'prob_mean': float(all_probs.mean()),
        'prob_std': float(all_probs.std()),
        'prob_p25': float(np.percentile(all_probs, 25)),
        'prob_p50': float(np.percentile(all_probs, 50)),
        'prob_p75': float(np.percentile(all_probs, 75)),
        'prob_p90': float(np.percentile(all_probs, 90)),
        'prob_p95': float(np.percentile(all_probs, 95)),
        'pct_above_0.3': float((all_probs > 0.3).mean()),
        'pct_above_0.5': float((all_probs > 0.5).mean()),
        'pct_above_0.7': float((all_probs > 0.7).mean()),
        'pct_above_0.9': float((all_probs > 0.9).mean()),
    }

    # FG distribution (requires labels — for analysis only)
    stats['fg_p50'] = float(np.median(fg_probs))
    stats['fg_p25'] = float(np.percentile(fg_probs, 25))
    stats['fg_p95'] = float(np.percentile(fg_probs, 95))
    stats['fg_mean'] = float(fg_probs.mean())
    stats['fg_pct_above_0.9'] = float((fg_probs > 0.9).mean())

    # BG contamination (how much BG leaks into high prob)
    stats['bg_pct_above_0.3'] = float((bg_probs > 0.3).mean())
    stats['bg_pct_above_0.5'] = float((bg_probs > 0.5).mean())
    stats['bg_pct_above_0.7'] = float((bg_probs > 0.7).mean())

    # Separation measures (label-free proxies)
    above_03 = all_probs[all_probs > 0.3]
    if len(above_03) > 10:
        stats['bimodal_gap'] = float(np.percentile(above_03, 75) - np.percentile(above_03, 25))
    else:
        stats['bimodal_gap'] = 0.0

    # Fraction of volume that's "uncertain" (0.3-0.7)
    stats['uncertain_frac'] = float(((all_probs > 0.3) & (all_probs < 0.7)).mean())

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-vols', type=int, default=20,
                        help='Number of volumes to analyze (0=all)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--cross-scroll', action='store_true', default=True,
                        help='Sample across all scrolls (default: True)')
    args = parser.parse_args()

    if args.dry_run:
        args.n_vols = 3

    # Find volumes with both probmaps and labels
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    probmap_ids = set(int(p.stem) for p in PROBMAP_DIR.glob("*.npy"))
    label_ids = set(int(p.stem) for p in TRAIN_LBL.glob("*.tif"))
    available = probmap_ids & label_ids
    train_df = train_df[train_df.id.isin(available)]

    # Sample across scrolls for diversity
    if args.cross_scroll:
        eval_ids = []
        scroll_counts = train_df.scroll_id.value_counts()
        for sid in scroll_counts.index:
            group = train_df[train_df.scroll_id == sid]
            n_take = max(1, int(args.n_vols * len(group) / len(train_df)))
            # Take evenly spaced samples
            ids = group.id.tolist()
            step = max(1, len(ids) // n_take)
            eval_ids.extend(ids[::step][:n_take])
        eval_ids = eval_ids[:args.n_vols]
    else:
        eval_ids = train_df.id.tolist()[:args.n_vols]

    # Map vol_id -> scroll_id
    vol_scroll = dict(zip(train_df.id, train_df.scroll_id))

    print(f"Adaptive T_low analysis on {len(eval_ids)} volumes")
    scroll_breakdown = {}
    for vid in eval_ids:
        sid = vol_scroll[vid]
        scroll_breakdown[sid] = scroll_breakdown.get(sid, 0) + 1
    for sid, cnt in sorted(scroll_breakdown.items()):
        print(f"  scroll {sid}: {cnt} volumes")

    # T_low values to sweep
    tlow_values = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    print(f"\nSweeping {len(tlow_values)} T_low values per volume...")
    print(f"T_low range: {tlow_values}")
    print()

    all_rows = []
    t_global = time.time()

    for i, vid in enumerate(eval_ids):
        t0 = time.time()
        scroll_id = vol_scroll[vid]

        prob = np.load(PROBMAP_DIR / f"{vid}.npy").astype(np.float32)
        lbl = tifffile.imread(TRAIN_LBL / f"{vid}.tif")

        # Compute probability statistics
        stats = compute_prob_stats(prob, lbl)

        # Sweep T_low
        best_tlow = None
        best_comp = -1
        tlow_scores = {}

        for tlow in tlow_values:
            pred = postprocess(prob, t_low=tlow)
            scores = score_volume(pred, lbl)
            tlow_scores[tlow] = scores

            if scores['comp_score'] > best_comp:
                best_comp = scores['comp_score']
                best_tlow = tlow

        elapsed = time.time() - t0
        eta = (elapsed * (len(eval_ids) - i - 1)) / 60

        row = {
            'vol_id': vid,
            'scroll_id': scroll_id,
            'best_tlow': best_tlow,
            'best_comp': best_comp,
            **{f'comp_tlow_{t}': tlow_scores[t]['comp_score'] for t in tlow_values},
            **{f'topo_tlow_{t}': tlow_scores[t]['topo'] for t in tlow_values},
            **{f'sdice_tlow_{t}': tlow_scores[t]['sdice'] for t in tlow_values},
            **{f'voi_tlow_{t}': tlow_scores[t]['voi'] for t in tlow_values},
            **stats,
        }
        all_rows.append(row)

        print(f"[{i+1}/{len(eval_ids)}] vol={vid} scroll={scroll_id} | "
              f"best_tlow={best_tlow:.2f} (comp={best_comp:.4f}) | "
              f"prob_max={stats['prob_max']:.3f} fg_p50={stats['fg_p50']:.3f} "
              f"bg>0.5={stats['bg_pct_above_0.5']:.3f} | "
              f"{elapsed:.0f}s (ETA {eta:.1f}min)")

    df = pd.DataFrame(all_rows)
    total_time = (time.time() - t_global) / 60

    # === Analysis ===
    print(f"\n{'='*70}")
    print(f"ADAPTIVE T_LOW ANALYSIS (n={len(df)}, {total_time:.1f} min)")
    print(f"{'='*70}")

    # 1. Distribution of optimal T_low
    print("\n1. Distribution of per-volume optimal T_low:")
    tlow_dist = df.best_tlow.value_counts().sort_index()
    for tlow, count in tlow_dist.items():
        bar = '#' * count
        print(f"   T_low={tlow:.2f}: {count:>3d} {bar}")
    print(f"   Mean optimal: {df.best_tlow.mean():.3f} +/- {df.best_tlow.std():.3f}")

    # 2. Per-scroll optimal T_low
    print("\n2. Per-scroll optimal T_low:")
    for sid, group in df.groupby('scroll_id'):
        print(f"   scroll {sid} (n={len(group)}): "
              f"mean_best_tlow={group.best_tlow.mean():.3f} +/- {group.best_tlow.std():.3f}")

    # 3. Fixed vs optimal comparison
    print("\n3. Fixed T_low vs per-volume optimal:")
    for tlow in tlow_values:
        col = f'comp_tlow_{tlow}'
        mean_fixed = df[col].mean()
        delta = mean_fixed - df.best_comp.mean()
        marker = " <-- current best fixed" if tlow == 0.60 else ""
        print(f"   Fixed T_low={tlow:.2f}: mean_comp={mean_fixed:.4f} "
              f"(delta vs optimal: {delta:+.4f}){marker}")
    print(f"   Per-vol optimal:     mean_comp={df.best_comp.mean():.4f}")

    # 4. Correlation analysis
    print("\n4. Correlation between prob stats and optimal T_low:")
    stat_cols = [c for c in stats.keys()]
    correlations = []
    for col in stat_cols:
        if col in df.columns and df[col].std() > 1e-8:
            corr = df['best_tlow'].corr(df[col])
            correlations.append((col, corr))
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, corr in correlations[:15]:
        strength = "STRONG" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
        print(f"   {col:<25s} r={corr:+.3f}  ({strength})")

    # 5. Label-free correlations (usable at test time)
    print("\n5. Label-FREE correlations (usable at inference time):")
    label_free = ['prob_max', 'prob_mean', 'prob_std', 'prob_p25', 'prob_p50',
                  'prob_p75', 'prob_p90', 'prob_p95', 'pct_above_0.3',
                  'pct_above_0.5', 'pct_above_0.7', 'pct_above_0.9',
                  'bimodal_gap', 'uncertain_frac']
    lf_corrs = [(c, df['best_tlow'].corr(df[c])) for c in label_free
                if c in df.columns and df[c].std() > 1e-8]
    lf_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, corr in lf_corrs:
        strength = "STRONG" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
        print(f"   {col:<25s} r={corr:+.3f}  ({strength})")

    # 6. Sensitivity: how much does per-volume T_low matter?
    print("\n6. Sensitivity: comp_score range across T_low values per volume:")
    ranges = []
    for _, row in df.iterrows():
        comps = [row[f'comp_tlow_{t}'] for t in tlow_values]
        r = max(comps) - min(comps)
        ranges.append(r)
        vid = int(row['vol_id'])
        best = row['best_tlow']
        print(f"   vol={vid}: range={r:.4f}, best_tlow={best:.2f}, "
              f"worst_tlow={tlow_values[np.argmin(comps)]:.2f}")
    print(f"   Mean range: {np.mean(ranges):.4f}")
    print(f"   Max range:  {np.max(ranges):.4f}")

    # Save
    out_path = ROOT / "logs" / "adaptive_tlow_analysis.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")


if __name__ == '__main__':
    main()
