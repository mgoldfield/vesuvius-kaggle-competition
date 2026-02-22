#!/usr/bin/env python3
"""
Sweep connectivity-focused post-processing methods on pre-computed probmaps.

Methods:
  A) Probmap-guided gap filling — iterative dilation into sub-threshold signal
  B) Dilate-merge-erode — classic morphological reconnection
  C) Two-pass hysteresis — strict mask + low-threshold gap-zone bridging
  D) Combined (C → A → bridge-only cleanup)

No GPU needed — loads .npy probmaps, applies PP, scores against GT labels.

Usage:
    python scripts/sweep_connectivity_pp.py --probmap-dir data/transunet_probmaps --dry-run
    python scripts/sweep_connectivity_pp.py --probmap-dir data/transunet_probmaps --n-eval 20
    python scripts/sweep_connectivity_pp.py --probmap-dir data/transunet_probmaps
    python scripts/sweep_connectivity_pp.py --probmap-dir data/swa_70_30_probmaps
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
    binary_closing, binary_dilation, binary_erosion, binary_propagation,
    distance_transform_edt, generate_binary_structure,
    label as scipy_label, zoom as scipy_zoom,
)
from topometrics.leaderboard import compute_leaderboard_score

ROOT = Path("/workspace/vesuvius-kaggle-competition")
TRAIN_LBL = ROOT / "data" / "train_labels"
DEFAULT_PROBMAP_DIR = ROOT / "data" / "transunet_probmaps"
VAL_SCROLL = 26002


# ── Shared utilities ─────────────────────────────────────

def hysteresis(prob, t_low, t_high):
    """Standard hysteresis thresholding (26-connected)."""
    strong = prob >= t_high
    if not strong.any():
        return np.zeros_like(prob, dtype=bool)
    weak = prob >= t_low
    struct = generate_binary_structure(3, 3)  # 26-connected
    return binary_propagation(strong, structure=struct, mask=weak)


def remove_dust(mask, min_size):
    """Remove connected components smaller than min_size (in-place)."""
    labeled, n = scipy_label(mask)
    if n > 0:
        sizes = np.bincount(labeled.ravel())
        small = sizes < min_size
        small[0] = False
        mask[small[labeled]] = 0


def fg_pct(mask):
    """Foreground percentage."""
    return 100.0 * mask.sum() / mask.size


def count_components(mask):
    """Number of connected components."""
    _, n = scipy_label(mask)
    return n


# ── Method A: Probmap-guided gap filling ─────────────────

def gap_fill(prob, t_low_strict=0.80, t_high=0.90,
             gap_prob_thresh=0.30, dilation_iters=5, dust=100):
    """
    Threshold aggressively, then iteratively dilate into sub-threshold
    regions where the probmap still has signal. Fills gaps with
    probability support.
    """
    mask = hysteresis(prob, t_low_strict, t_high)

    struct = generate_binary_structure(3, 1)  # 6-connected
    for _ in range(dilation_iters):
        dilated = binary_dilation(mask, struct, iterations=1)
        bridge = dilated & ~mask & (prob >= gap_prob_thresh)
        if not bridge.any():
            break
        mask |= bridge

    remove_dust(mask, dust)
    return mask.astype(np.uint8)


# ── Method B: Dilate-merge-erode ─────────────────────────

def dilate_merge_erode(prob, t_low=0.80, t_high=0.90,
                       dilation_r=2, erosion_r=2, prob_floor=0.10, dust=100):
    """
    Classic morphological reconnection. Dilate to bridge small gaps,
    then erode back to preserve thickness. Prob floor prevents keeping
    voxels with no signal.
    """
    mask = hysteresis(prob, t_low, t_high)

    struct = generate_binary_structure(3, 1)  # 6-connected
    mask = binary_dilation(mask, struct, iterations=dilation_r)
    mask = binary_erosion(mask, struct, iterations=erosion_r)
    mask &= (prob >= prob_floor)

    remove_dust(mask, dust)
    return mask.astype(np.uint8)


# ── Method C: Two-pass hysteresis ────────────────────────

def two_pass_hysteresis(prob, t_low_strict=0.80, t_high=0.90,
                        t_low_gap=0.30, max_bridge_dist=5, dust=100):
    """
    Pass 1: strict hysteresis. Pass 2: in gap zones (within
    max_bridge_dist of strict mask), use a much lower threshold.
    Bridges gaps without global thickening.
    """
    mask = hysteresis(prob, t_low_strict, t_high)
    if not mask.any():
        return mask.astype(np.uint8)

    # Identify gap zones: within max_bridge_dist of mask, but not in mask
    dist = distance_transform_edt(~mask)
    gap_zone = (dist <= max_bridge_dist) & ~mask

    # In gap zones, accept lower-threshold voxels
    weak_in_gap = gap_zone & (prob >= t_low_gap)
    combined = mask | weak_in_gap

    # Propagate from strict mask through combined region (26-connected)
    struct = generate_binary_structure(3, 3)
    bridged = binary_propagation(mask, structure=struct, mask=combined)

    remove_dust(bridged, dust)
    return bridged.astype(np.uint8)


# ── Method D: Combined (C → A → bridge-only cleanup) ────

def combined_method(prob, t_low_strict=0.80, t_high=0.90,
                    t_low_gap=0.30, max_bridge_dist=5,
                    gap_prob_thresh=0.30, dilation_iters=3, dust=100):
    """
    1. Two-pass hysteresis (Method C)
    2. Probmap-guided gap filling (Method A) on the result
    3. Remove added voxels that didn't bridge two original components
    """
    # Step 1: strict mask (for component labeling later)
    strict = hysteresis(prob, t_low_strict, t_high)
    strict_labeled, n_strict = scipy_label(strict)

    # Step 2: two-pass hysteresis
    mask_c = two_pass_hysteresis(prob, t_low_strict, t_high, t_low_gap,
                                 max_bridge_dist, dust=0)  # no dust yet
    mask = mask_c.astype(bool)

    # Step 3: gap fill on the result
    struct6 = generate_binary_structure(3, 1)
    for _ in range(dilation_iters):
        dilated = binary_dilation(mask, struct6, iterations=1)
        bridge = dilated & ~mask & (prob >= gap_prob_thresh)
        if not bridge.any():
            break
        mask |= bridge

    # Step 4: bridge-only cleanup — keep added voxels only if they bridge
    # two or more original strict components
    added = mask & ~strict
    if added.any() and n_strict > 0 and n_strict <= 500:
        struct26 = generate_binary_structure(3, 3)
        # For each added voxel, count how many distinct strict components
        # are adjacent to it
        adj_labels = set()
        adj_count = np.zeros(strict.shape, dtype=np.int32)
        for cc_id in range(1, n_strict + 1):
            cc_dilated = binary_dilation(strict_labeled == cc_id, struct26,
                                         iterations=1)
            adj_count += (cc_dilated & added).astype(np.int32)
        # Keep only voxels adjacent to 2+ different original components
        bridge_voxels = (adj_count >= 2) & added
        mask = strict | bridge_voxels

    remove_dust(mask, dust)
    return mask.astype(np.uint8)


# ── Baseline: standard hysteresis + closing ──────────────

def baseline_hysteresis(prob, t_low=0.50, t_high=0.90, dust=100):
    """Standard hysteresis + closing + dust (matches eval_transunet defaults)."""
    mask = hysteresis(prob, t_low, t_high)

    # Closing with anisotropic struct (z=3, xy=2) — matches competitor defaults
    z_radius, xy_radius = 3, 2
    sz = 2 * z_radius + 1
    sxy = 2 * xy_radius + 1
    struct_close = np.zeros((sz, sxy, sxy), dtype=bool)
    cy, cx = xy_radius, xy_radius
    for y in range(sxy):
        for x in range(sxy):
            if (y - cy) ** 2 + (x - cx) ** 2 <= xy_radius ** 2:
                struct_close[:, y, x] = True
    mask = binary_closing(mask, structure=struct_close)

    remove_dust(mask, dust)
    return mask.astype(np.uint8)


# ── Scoring ──────────────────────────────────────────────

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


# ── Config builders ──────────────────────────────────────

def build_configs(dry_run=False):
    """Build all PP configurations to sweep."""
    configs = []

    # --- Baselines ---
    for t_low, name in [(0.50, 'baseline_t50'), (0.70, 'baseline_t70'),
                         (0.80, 'baseline_t80')]:
        configs.append({
            'name': name,
            'method': 'baseline',
            'params': {'t_low': t_low, 't_high': 0.90, 'dust': 100},
        })

    # --- Method A: Gap fill ---
    for t_low_strict in [0.70, 0.80]:
        for gap_prob in [0.20, 0.30, 0.40]:
            for iters in [3, 5, 8]:
                # Skip some combos to keep total manageable
                if t_low_strict == 0.70 and iters == 8:
                    continue
                configs.append({
                    'name': f'A_gap_tl{t_low_strict}_gp{gap_prob}_i{iters}',
                    'method': 'gap_fill',
                    'params': {
                        't_low_strict': t_low_strict, 't_high': 0.90,
                        'gap_prob_thresh': gap_prob, 'dilation_iters': iters,
                        'dust': 100,
                    },
                })

    # --- Method B: Dilate-merge-erode ---
    for t_low in [0.70, 0.80]:
        for dil_r in [1, 2, 3]:
            configs.append({
                'name': f'B_dme_tl{t_low}_r{dil_r}',
                'method': 'dilate_merge_erode',
                'params': {
                    't_low': t_low, 't_high': 0.90,
                    'dilation_r': dil_r, 'erosion_r': dil_r,
                    'prob_floor': 0.10, 'dust': 100,
                },
            })

    # --- Method C: Two-pass hysteresis ---
    for t_low_strict in [0.70, 0.80]:
        for t_low_gap in [0.20, 0.30, 0.40]:
            for bridge_dist in [3, 5, 8]:
                # Skip some combos
                if t_low_strict == 0.70 and bridge_dist == 8:
                    continue
                configs.append({
                    'name': f'C_2pass_tl{t_low_strict}_gap{t_low_gap}_bd{bridge_dist}',
                    'method': 'two_pass',
                    'params': {
                        't_low_strict': t_low_strict, 't_high': 0.90,
                        't_low_gap': t_low_gap, 'max_bridge_dist': bridge_dist,
                        'dust': 100,
                    },
                })

    # --- Method D: Combined ---
    for bridge_dist in [3, 5]:
        for dil_iters in [2, 3]:
            configs.append({
                'name': f'D_combo_bd{bridge_dist}_i{dil_iters}',
                'method': 'combined',
                'params': {
                    't_low_strict': 0.80, 't_high': 0.90,
                    't_low_gap': 0.30, 'max_bridge_dist': bridge_dist,
                    'gap_prob_thresh': 0.30, 'dilation_iters': dil_iters,
                    'dust': 100,
                },
            })

    if dry_run:
        # In dry-run: 1 baseline + 1 of each method
        dry = [configs[0]]  # baseline_t50
        for method in ['gap_fill', 'dilate_merge_erode', 'two_pass', 'combined']:
            for c in configs:
                if c['method'] == method:
                    dry.append(c)
                    break
        configs = dry

    return configs


# ── Dispatch ─────────────────────────────────────────────

def apply_pp(method, prob, params):
    """Dispatch to the appropriate PP method."""
    if method == 'baseline':
        return baseline_hysteresis(prob, **params)
    elif method == 'gap_fill':
        return gap_fill(prob, **params)
    elif method == 'dilate_merge_erode':
        return dilate_merge_erode(prob, **params)
    elif method == 'two_pass':
        return two_pass_hysteresis(prob, **params)
    elif method == 'combined':
        return combined_method(prob, **params)
    else:
        raise ValueError(f"Unknown method: {method}")


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sweep connectivity post-processing methods on probmaps")
    parser.add_argument('--probmap-dir', type=str, default=str(DEFAULT_PROBMAP_DIR))
    parser.add_argument('--n-eval', type=int, default=0, help='0=all val volumes')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Metric downsample (1=full res)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Quick test: 2 volumes, 1 config per method')
    args = parser.parse_args()

    probmap_dir = Path(args.probmap_dir)
    if not probmap_dir.exists():
        print(f"ERROR: Probmap dir not found: {probmap_dir}")
        print("Run eval_transunet.py with --save-probmaps first.")
        return

    if args.dry_run:
        args.n_eval = 2

    # Get val volume IDs
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    probmap_ids = set(int(p.stem) for p in probmap_dir.glob("*.npy"))
    label_ids = set(int(p.stem) for p in TRAIN_LBL.glob("*.tif"))
    available = probmap_ids & label_ids

    val_df = train_df[(train_df.scroll_id == VAL_SCROLL) & train_df.id.isin(available)]
    eval_ids = sorted(val_df.id.tolist())
    if args.n_eval > 0:
        eval_ids = eval_ids[:args.n_eval]

    print(f"Connectivity PP sweep on {len(eval_ids)} volumes (ds={args.downsample})")
    print(f"Probmap dir: {probmap_dir}")

    # Load all probmaps and labels
    print("Loading probmaps and labels...")
    data = {}
    for vid in eval_ids:
        prob = np.load(probmap_dir / f"{vid}.npy").astype(np.float32)
        lbl = tifffile.imread(TRAIN_LBL / f"{vid}.tif")
        data[vid] = (prob, lbl)
    print(f"Loaded {len(data)} volumes")

    # Build configs
    configs = build_configs(dry_run=args.dry_run)
    print(f"\nSweeping {len(configs)} configurations...")
    print(f"{'Config':<40} {'Comp':>6} {'Topo':>6} {'SDice':>6} {'VOI':>6} "
          f"{'FG%':>5} {'#CC':>5} {'Time':>5}")
    print("-" * 82)

    all_results = []
    per_volume_results = []

    for cfg in configs:
        vol_scores = []
        vol_fg = []
        vol_cc = []
        t0 = time.time()

        for vid in eval_ids:
            prob, lbl = data[vid]
            pred = apply_pp(cfg['method'], prob, cfg['params'])
            s = score_volume(pred, lbl, downsample=args.downsample)
            s['vid'] = vid
            s['config'] = cfg['name']
            s['method'] = cfg['method']
            s['fg_pct'] = fg_pct(pred)
            s['n_components'] = count_components(pred)
            vol_scores.append(s)
            vol_fg.append(s['fg_pct'])
            vol_cc.append(s['n_components'])
            per_volume_results.append(s)

        elapsed = time.time() - t0

        mean_scores = {
            k: np.mean([s[k] for s in vol_scores])
            for k in ['comp_score', 'topo', 'sdice', 'voi']
        }
        mean_fg = np.mean(vol_fg)
        mean_cc = np.mean(vol_cc)

        print(f"{cfg['name']:<40} {mean_scores['comp_score']:>6.4f} "
              f"{mean_scores['topo']:>6.4f} {mean_scores['sdice']:>6.4f} "
              f"{mean_scores['voi']:>6.4f} {mean_fg:>5.1f} {mean_cc:>5.0f} "
              f"{elapsed:>5.1f}s")

        all_results.append({
            'config': cfg['name'],
            'method': cfg['method'],
            **cfg['params'],
            **mean_scores,
            'fg_pct': mean_fg,
            'n_components': mean_cc,
            'n_vols': len(eval_ids),
            'time_s': elapsed,
        })

    # Sort by comp_score
    results_df = pd.DataFrame(all_results).sort_values('comp_score', ascending=False)

    print(f"\n{'='*82}")
    print("TOP CONFIGURATIONS (sorted by comp_score)")
    print(f"{'='*82}")
    for _, row in results_df.head(15).iterrows():
        print(f"  {row['config']:<40} comp={row['comp_score']:.4f} "
              f"topo={row['topo']:.4f} sdice={row['sdice']:.4f} "
              f"voi={row['voi']:.4f} fg={row['fg_pct']:.1f}%")

    # Method-level summary
    print(f"\n{'='*82}")
    print("BEST PER METHOD")
    print(f"{'='*82}")
    for method in ['baseline', 'gap_fill', 'dilate_merge_erode', 'two_pass', 'combined']:
        method_rows = results_df[results_df['method'] == method]
        if len(method_rows) > 0:
            best = method_rows.iloc[0]
            print(f"  [{method:<20}] {best['config']:<35} comp={best['comp_score']:.4f} "
                  f"topo={best['topo']:.4f} sdice={best['sdice']:.4f} "
                  f"voi={best['voi']:.4f}")

    # Save results
    model_name = probmap_dir.name
    summary_path = ROOT / "logs" / f"connectivity_pp_{model_name}.csv"
    summary_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

    # Per-volume results
    pv_df = pd.DataFrame(per_volume_results)
    pv_path = ROOT / "logs" / f"connectivity_pp_{model_name}_per_vol.csv"
    pv_df.to_csv(pv_path, index=False)
    print(f"Per-volume results saved to {pv_path}")


if __name__ == '__main__':
    main()
