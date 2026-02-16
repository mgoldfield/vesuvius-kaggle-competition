#!/usr/bin/env python3
"""
Cross-scroll evaluation of v9 probmaps.

Computes comp_score on ALL volumes (not just 5), grouped by scroll_id.
Uses pre-computed probmaps + hand-tuned post-processing.
No model inference needed — just thresholding + metric computation.

Usage:
    python scripts/eval_cross_scroll.py [--max-per-scroll N] [--downsample D]
"""

import argparse
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from scipy.ndimage import (
    binary_closing, generate_binary_structure, binary_propagation,
    label as scipy_label, zoom as scipy_zoom,
)
from topometrics.leaderboard import compute_leaderboard_score
import time

ROOT = Path("/workspace/vesuvius-kaggle-competition")
PROBMAP_DIR = ROOT / "data" / "refinement_data" / "probmaps"
TRAIN_LBL = ROOT / "data" / "train_labels"

# Best thresholds from sweep
T_LOW = 0.35
T_HIGH = 0.80
CLOSING_Z_RADIUS = 2
CLOSING_XY_RADIUS = 1
DUST_MIN_SIZE = 100


def hysteresis_threshold(prob, t_low, t_high):
    strong = prob >= t_high
    weak = prob >= t_low
    struct = generate_binary_structure(3, 3)
    return binary_propagation(strong, structure=struct, mask=weak).astype(np.uint8)


def build_anisotropic_struct(z_radius=2, xy_radius=1):
    sz = 2 * z_radius + 1
    sxy = 2 * xy_radius + 1
    struct = np.zeros((sz, sxy, sxy), dtype=bool)
    cy, cx = xy_radius, xy_radius
    for y in range(sxy):
        for x in range(sxy):
            if (y - cy) ** 2 + (x - cx) ** 2 <= xy_radius ** 2:
                struct[:, y, x] = True
    return struct


def postprocess(probs, t_low=T_LOW, t_high=T_HIGH):
    binary = hysteresis_threshold(probs, t_low, t_high)
    struct = build_anisotropic_struct(CLOSING_Z_RADIUS, CLOSING_XY_RADIUS)
    closed = binary_closing(binary, structure=struct)
    labeled, n_components = scipy_label(closed)
    if n_components > 0:
        sizes = np.bincount(labeled.ravel())
        small = sizes < DUST_MIN_SIZE
        small[0] = False
        closed[small[labeled]] = 0
    return closed.astype(np.uint8)


def score_volume(vol_id, downsample=4):
    prob = np.load(PROBMAP_DIR / f"{vol_id}.npy")
    lbl = tifffile.imread(TRAIN_LBL / f"{vol_id}.tif")

    pred = postprocess(prob)

    if downsample > 1:
        scale = 1.0 / downsample
        pred = scipy_zoom(pred, scale, order=0).astype(np.uint8)
        lbl = scipy_zoom(lbl, scale, order=0).astype(np.uint8)

    report = compute_leaderboard_score(
        pred, lbl, ignore_label=2, spacing=(1, 1, 1),
        surface_tolerance=2.0, voi_alpha=0.3,
        combine_weights=(0.3, 0.35, 0.35),
    )
    return report.score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-scroll", type=int, default=0,
                        help="Max volumes per scroll (0=all)")
    parser.add_argument("--downsample", type=int, default=4,
                        help="Downsample factor for metric (4=fast, 1=full res)")
    args = parser.parse_args()

    train_df = pd.read_csv(ROOT / "data" / "train.csv")

    # Find volumes with both probmap and label
    probmap_ids = set(int(p.stem) for p in PROBMAP_DIR.glob("*.npy"))
    label_ids = set(int(p.stem) for p in TRAIN_LBL.glob("*.tif"))
    available = probmap_ids & label_ids
    train_df = train_df[train_df.id.isin(available)].reset_index(drop=True)

    print(f"Total volumes with probmap + label: {len(train_df)}")
    print(f"Downsample: {args.downsample}x")
    print(f"Thresholds: T_low={T_LOW}, T_high={T_HIGH}")
    print()

    # Group by scroll
    scroll_groups = train_df.groupby("scroll_id")
    print(f"Scrolls: {len(scroll_groups)}")
    for sid, group in scroll_groups:
        print(f"  scroll {sid}: {len(group)} volumes")
    print()

    all_results = []

    for sid, group in scroll_groups:
        ids = group.id.tolist()
        if args.max_per_scroll > 0:
            ids = ids[:args.max_per_scroll]

        print(f"--- Scroll {sid} ({len(ids)} volumes) ---")
        scroll_scores = []

        for i, vid in enumerate(ids):
            t0 = time.time()
            score = score_volume(vid, downsample=args.downsample)
            elapsed = time.time() - t0
            scroll_scores.append(score)
            all_results.append({
                "scroll_id": sid,
                "vol_id": vid,
                "comp_score": score,
            })
            if (i + 1) % 10 == 0 or i == len(ids) - 1:
                print(f"  [{i+1}/{len(ids)}] latest={score:.4f}, "
                      f"scroll_mean={np.mean(scroll_scores):.4f} ({elapsed:.1f}s)")

        print(f"  Scroll {sid}: mean={np.mean(scroll_scores):.4f}, "
              f"std={np.std(scroll_scores):.4f}, "
              f"min={np.min(scroll_scores):.4f}, max={np.max(scroll_scores):.4f}")
        print()

    # Overall summary
    results_df = pd.DataFrame(all_results)
    print("=" * 70)
    print("CROSS-SCROLL EVALUATION SUMMARY")
    print("=" * 70)

    # Per-scroll table
    scroll_summary = results_df.groupby("scroll_id").agg(
        n=("comp_score", "count"),
        mean=("comp_score", "mean"),
        std=("comp_score", "std"),
        min=("comp_score", "min"),
        max=("comp_score", "max"),
    ).round(4)
    print("\nPer-scroll breakdown:")
    print(scroll_summary.to_string())

    # Val scroll vs training scrolls
    val_mask = results_df.scroll_id == 26002
    val_mean = results_df[val_mask].comp_score.mean()
    train_mean = results_df[~val_mask].comp_score.mean()
    overall_mean = results_df.comp_score.mean()

    print(f"\nVal scroll (26002):     {val_mean:.4f} (n={val_mask.sum()})")
    print(f"Training scrolls:      {train_mean:.4f} (n={(~val_mask).sum()})")
    print(f"Overall:               {overall_mean:.4f} (n={len(results_df)})")
    print(f"Val - Train gap:       {val_mean - train_mean:+.4f}")

    # Distribution
    print(f"\nScore distribution (all volumes):")
    for pct in [10, 25, 50, 75, 90]:
        print(f"  p{pct}: {np.percentile(results_df.comp_score, pct):.4f}")

    # Save detailed results
    out_path = ROOT / "logs" / "cross_scroll_eval.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
