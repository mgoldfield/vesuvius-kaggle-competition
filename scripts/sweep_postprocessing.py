#!/usr/bin/env python3
"""
Sweep post-processing parameters on pre-computed TransUNet probmaps.

No GPU needed — just loads .npy probmaps and tests different thresholds,
closing parameters, dust sizes, and confidence-based CC filtering.

Usage:
    python scripts/sweep_postprocessing.py
    python scripts/sweep_postprocessing.py --probmap-dir data/transunet_probmaps --n-eval 5
    python scripts/sweep_postprocessing.py --dry-run
"""

import os

# Force TensorFlow to CPU-only — topometrics imports TF, which grabs GPU by default.
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
from topometrics.leaderboard import compute_leaderboard_score

ROOT = Path("/workspace/vesuvius-kaggle-competition")
TRAIN_LBL = ROOT / "data" / "train_labels"
DEFAULT_PROBMAP_DIR = ROOT / "data" / "transunet_probmaps"
VAL_SCROLL = 26002


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


def postprocess(prob, t_low, t_high, z_radius, xy_radius, dust_min_size,
                conf_cc_filter=False, conf_cc_threshold=0.8):
    """Full post-processing pipeline with optional confidence-based CC filtering."""
    strong = prob >= t_high
    weak = prob >= t_low
    struct_hyst = generate_binary_structure(3, 3)

    if not strong.any():
        return np.zeros_like(prob, dtype=np.uint8)

    mask = binary_propagation(strong, structure=struct_hyst, mask=weak)

    # Closing
    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        mask = binary_closing(mask, structure=struct_close)

    # Dust removal + optional confidence-based CC filtering
    labeled, n = scipy_label(mask)
    if n > 0:
        for cc_id in range(1, n + 1):
            cc_mask = labeled == cc_id
            cc_size = cc_mask.sum()

            # Size-based removal
            if cc_size < dust_min_size:
                mask[cc_mask] = 0
                continue

            # Confidence-based removal
            if conf_cc_filter:
                cc_probs = prob[cc_mask]
                p95 = np.percentile(cc_probs, 95)
                if p95 < conf_cc_threshold:
                    mask[cc_mask] = 0

    return mask.astype(np.uint8)


def score_volume(pred, lbl, downsample=1):
    ds = downsample
    if ds > 1:
        pred = scipy_zoom(pred, 1.0 / ds, order=0).astype(np.uint8)
        lbl = scipy_zoom(lbl, 1.0 / ds, order=0).astype(np.uint8)

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
    parser.add_argument('--probmap-dir', type=str, default=str(DEFAULT_PROBMAP_DIR))
    parser.add_argument('--n-eval', type=int, default=0, help='0=all val volumes')
    parser.add_argument('--downsample', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    probmap_dir = Path(args.probmap_dir)
    if not probmap_dir.exists():
        print(f"ERROR: Probmap dir not found: {probmap_dir}")
        print("Run eval_transunet.py with --save-probmaps first.")
        return

    if args.dry_run:
        args.n_eval = 2
        args.max_configs = 3  # exercise code without running all 26
    else:
        args.max_configs = 0  # 0 = all

    # Get val volume IDs
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    probmap_ids = set(int(p.stem) for p in probmap_dir.glob("*.npy"))
    label_ids = set(int(p.stem) for p in TRAIN_LBL.glob("*.tif"))
    available = probmap_ids & label_ids

    val_df = train_df[(train_df.scroll_id == VAL_SCROLL) & train_df.id.isin(available)]
    eval_ids = val_df.id.tolist()
    if args.n_eval > 0:
        eval_ids = eval_ids[:args.n_eval]

    print(f"Post-processing sweep on {len(eval_ids)} volumes (ds={args.downsample})")
    print(f"Probmap dir: {probmap_dir}")

    # Load all probmaps and labels
    print("Loading probmaps and labels...")
    data = {}
    for vid in eval_ids:
        prob = np.load(probmap_dir / f"{vid}.npy").astype(np.float32)
        lbl = tifffile.imread(TRAIN_LBL / f"{vid}.tif")
        data[vid] = (prob, lbl)
    print(f"Loaded {len(data)} volumes")

    # Define sweep configs
    configs = []

    # Competitor defaults
    configs.append({
        'name': 'competitor_default',
        't_low': 0.50, 't_high': 0.90,
        'z_radius': 3, 'xy_radius': 2,
        'dust': 100, 'conf_cc': False,
    })

    # Our old defaults
    configs.append({
        'name': 'our_old_defaults',
        't_low': 0.35, 't_high': 0.75,
        'z_radius': 2, 'xy_radius': 1,
        'dust': 64, 'conf_cc': False,
    })

    # T_low sweep
    for t_low in [0.30, 0.40, 0.50, 0.60]:
        configs.append({
            'name': f'tlow_{t_low}',
            't_low': t_low, 't_high': 0.90,
            'z_radius': 3, 'xy_radius': 2,
            'dust': 100, 'conf_cc': False,
        })

    # T_high sweep
    for t_high in [0.75, 0.80, 0.85, 0.90, 0.95]:
        configs.append({
            'name': f'thigh_{t_high}',
            't_low': 0.50, 't_high': t_high,
            'z_radius': 3, 'xy_radius': 2,
            'dust': 100, 'conf_cc': False,
        })

    # Closing sweep
    for z, xy in [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 2)]:
        configs.append({
            'name': f'close_z{z}_xy{xy}',
            't_low': 0.50, 't_high': 0.90,
            'z_radius': z, 'xy_radius': xy,
            'dust': 100, 'conf_cc': False,
        })

    # Dust sweep
    for dust in [50, 100, 200, 500]:
        configs.append({
            'name': f'dust_{dust}',
            't_low': 0.50, 't_high': 0.90,
            'z_radius': 3, 'xy_radius': 2,
            'dust': dust, 'conf_cc': False,
        })

    # Confidence CC filtering
    for cc_thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        configs.append({
            'name': f'confcc_{cc_thresh}',
            't_low': 0.50, 't_high': 0.90,
            'z_radius': 3, 'xy_radius': 2,
            'dust': 100, 'conf_cc': True, 'conf_cc_thresh': cc_thresh,
        })

    # Remove duplicates by name
    seen = set()
    unique_configs = []
    for c in configs:
        if c['name'] not in seen:
            seen.add(c['name'])
            unique_configs.append(c)
    configs = unique_configs

    # In dry-run mode, only test a few configs to exercise all code paths
    if args.max_configs > 0:
        # Pick first (competitor_default), one with conf_cc=True, and one closing config
        dry_configs = [c for c in configs if c['name'] == 'competitor_default']
        dry_configs += [c for c in configs if c.get('conf_cc', False)][:1]
        dry_configs += [c for c in configs if 'close_' in c['name']][:1]
        configs = dry_configs

    print(f"\nSweeping {len(configs)} configurations...")
    print(f"{'Config':<25} {'Comp':>6} {'Topo':>6} {'SDice':>6} {'VOI':>6}")
    print("-" * 55)

    all_results = []

    for cfg in configs:
        scores_list = []
        t0 = time.time()

        for vid in eval_ids:
            prob, lbl = data[vid]
            pred = postprocess(
                prob,
                t_low=cfg['t_low'],
                t_high=cfg['t_high'],
                z_radius=cfg['z_radius'],
                xy_radius=cfg['xy_radius'],
                dust_min_size=cfg['dust'],
                conf_cc_filter=cfg.get('conf_cc', False),
                conf_cc_threshold=cfg.get('conf_cc_thresh', 0.8),
            )
            s = score_volume(pred, lbl, downsample=args.downsample)
            scores_list.append(s)

        elapsed = time.time() - t0

        mean_scores = {
            k: np.mean([s[k] for s in scores_list])
            for k in ['comp_score', 'topo', 'sdice', 'voi']
        }

        print(f"{cfg['name']:<25} {mean_scores['comp_score']:>6.4f} "
              f"{mean_scores['topo']:>6.4f} {mean_scores['sdice']:>6.4f} "
              f"{mean_scores['voi']:>6.4f}  ({elapsed:.1f}s)")

        all_results.append({
            'config': cfg['name'],
            **cfg,
            **mean_scores,
        })

    # Sort by comp_score
    results_df = pd.DataFrame(all_results).sort_values('comp_score', ascending=False)

    print(f"\n{'='*55}")
    print("TOP 10 CONFIGURATIONS (sorted by comp_score)")
    print(f"{'='*55}")
    for _, row in results_df.head(10).iterrows():
        print(f"  {row['config']:<25} comp={row['comp_score']:.4f} "
              f"topo={row['topo']:.4f} sdice={row['sdice']:.4f} voi={row['voi']:.4f}")

    # Save
    out_path = ROOT / "logs" / "postprocessing_sweep.csv"
    out_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
