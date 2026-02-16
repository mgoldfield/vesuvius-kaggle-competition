#!/usr/bin/env python3
"""
Compare adaptive T_HIGH strategies across all scrolls.
Uses pre-computed probmaps — no GPU needed.
"""

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
PROBMAP_DIR = ROOT / "data" / "refinement_data" / "probmaps"
TRAIN_LBL = ROOT / "data" / "train_labels"
T_LOW = 0.35


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


STRUCT = build_anisotropic_struct(2, 1)


def postprocess(probs, t_high):
    binary = hysteresis_threshold(probs, T_LOW, t_high)
    closed = binary_closing(binary, structure=STRUCT)
    labeled, n = scipy_label(closed)
    if n > 0:
        sizes = np.bincount(labeled.ravel())
        small = sizes < 100
        small[0] = False
        closed[small[labeled]] = 0
    return closed.astype(np.uint8)


def score(pred, lbl, ds=4):
    scale = 1.0 / ds
    pred = scipy_zoom(pred, scale, order=0).astype(np.uint8)
    lbl = scipy_zoom(lbl, scale, order=0).astype(np.uint8)
    report = compute_leaderboard_score(
        pred, lbl, ignore_label=2, spacing=(1, 1, 1),
        surface_tolerance=2.0, voi_alpha=0.3, combine_weights=(0.3, 0.35, 0.35),
    )
    return report.score


def adaptive_percentile(prob, min_prob=0.3, percentile=95):
    """T_HIGH = percentile of probabilities above min_prob."""
    high_probs = prob[prob > min_prob]
    if len(high_probs) > 0:
        return np.clip(float(np.percentile(high_probs, percentile)), 0.50, 0.90)
    return 0.50


def adaptive_max_fraction(prob, fraction=0.95):
    """T_HIGH = fraction * max probability."""
    return np.clip(float(prob.max()) * fraction, 0.50, 0.90)


def adaptive_max_minus(prob, margin=0.05):
    """T_HIGH = max probability - margin."""
    return np.clip(float(prob.max()) - margin, 0.50, 0.90)


strategies = {
    "fixed_080": lambda p: 0.80,
    "fixed_075": lambda p: 0.75,
    "fixed_070": lambda p: 0.70,
    "max*0.95": lambda p: adaptive_max_fraction(p, 0.95),
    "max-0.05": lambda p: adaptive_max_minus(p, 0.05),
    "p95_above_0.3": lambda p: adaptive_percentile(p, 0.3, 95),
    "p99_above_0.3": lambda p: adaptive_percentile(p, 0.3, 99),
    "p95_above_0.2": lambda p: adaptive_percentile(p, 0.2, 95),
}


def main():
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    probmap_ids = set(int(p.stem) for p in PROBMAP_DIR.glob("*.npy"))
    label_ids = set(int(p.stem) for p in TRAIN_LBL.glob("*.tif"))
    available = probmap_ids & label_ids
    train_df = train_df[train_df.id.isin(available)].reset_index(drop=True)

    MAX_PER_SCROLL = 20

    results = []
    for sid, group in train_df.groupby("scroll_id"):
        ids = group.id.tolist()[:MAX_PER_SCROLL]
        for i, vid in enumerate(ids):
            prob = np.load(PROBMAP_DIR / f"{vid}.npy")
            lbl = tifffile.imread(TRAIN_LBL / f"{vid}.tif")

            row = {"scroll_id": sid, "vol_id": vid, "max_prob": float(prob.max())}
            for name, strat in strategies.items():
                t_h = strat(prob)
                pred = postprocess(prob, t_h)
                s = score(pred, lbl)
                row[name] = s
                row[f"{name}_th"] = t_h
            results.append(row)

            if (i + 1) % 5 == 0:
                print(f"  Scroll {sid}: {i+1}/{len(ids)}")
        print(f"Scroll {sid}: done ({len(ids)} vols)")

    df = pd.DataFrame(results)

    print()
    print("=" * 80)
    print("PER-SCROLL MEAN COMP_SCORE BY STRATEGY")
    print("=" * 80)

    strat_names = list(strategies.keys())
    scroll_ids = sorted(df.scroll_id.unique())

    # Header
    header = f"{'scroll':>8} {'n':>4}"
    for name in strat_names:
        header += f" {name:>15}"
    print(header)
    print("-" * len(header))

    for sid in scroll_ids:
        mask = df.scroll_id == sid
        n = mask.sum()
        line = f"{sid:>8} {n:>4}"
        for name in strat_names:
            line += f" {df[mask][name].mean():>15.4f}"
        print(line)

    # Overall
    line = f"{'OVERALL':>8} {len(df):>4}"
    for name in strat_names:
        line += f" {df[name].mean():>15.4f}"
    print("-" * len(header))
    print(line)

    # Best strategy
    print()
    print("BEST STRATEGY BY OVERALL MEAN:")
    means = {name: df[name].mean() for name in strat_names}
    best = max(means, key=means.get)
    print(f"  {best}: {means[best]:.4f}")

    # Show adaptive thresholds chosen for each scroll
    print()
    print("ADAPTIVE THRESHOLD VALUES (mean per scroll):")
    for name in strat_names:
        if "fixed" not in name:
            print(f"\n  {name}:")
            for sid in scroll_ids:
                mask = df.scroll_id == sid
                th_col = f"{name}_th"
                print(f"    scroll {sid}: T_HIGH = {df[mask][th_col].mean():.4f} "
                      f"(range {df[mask][th_col].min():.4f} - {df[mask][th_col].max():.4f})")

    # Save
    out = ROOT / "logs" / "adaptive_threshold_eval.csv"
    df.to_csv(out, index=False)
    print(f"\nDetailed results: {out}")


if __name__ == "__main__":
    main()
