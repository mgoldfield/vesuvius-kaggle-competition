#!/usr/bin/env python3
"""
Analysis of scroll 35360 poor performance in Vesuvius competition.

Investigates:
1. Data characteristics (label distributions, probmap statistics) for 35360 vs 26002
2. Sub-metric breakdown (topo, surface_dice, voi) for good vs bad volumes
3. Probmap quality visual check (false positives vs false negatives)
"""

import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from scipy.ndimage import (
    binary_closing, generate_binary_structure, binary_propagation,
    label as scipy_label, zoom as scipy_zoom,
)
import sys
import time

# Add topometrics to path
sys.path.insert(0, "/workspace/vesuvius-kaggle-competition/libs/topometrics_download/topological-metrics-kaggle")
from topometrics.leaderboard import compute_leaderboard_score

ROOT = Path("/workspace/vesuvius-kaggle-competition")
PROBMAP_DIR = ROOT / "data" / "refinement_data" / "probmaps"
TRAIN_LBL = ROOT / "data" / "train_labels"
TRAIN_IMG = ROOT / "data" / "train_images"

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


def load_probmap(vol_id):
    return np.load(PROBMAP_DIR / f"{vol_id}.npy")


def load_label(vol_id):
    return tifffile.imread(TRAIN_LBL / f"{vol_id}.tif")


def print_separator(title):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()


# ============================================================
# PART 1: Data Characteristics
# ============================================================
def analyze_data_characteristics():
    print_separator("PART 1: DATA CHARACTERISTICS — scroll 35360 vs scroll 26002")

    train_df = pd.read_csv(ROOT / "data" / "train.csv")

    # Get available volumes (those with both probmap and label)
    probmap_ids = set(int(p.stem) for p in PROBMAP_DIR.glob("*.npy"))
    label_ids = set(int(p.stem) for p in TRAIN_LBL.glob("*.tif"))
    available = probmap_ids & label_ids
    train_df = train_df[train_df.id.isin(available)].reset_index(drop=True)

    # Get volumes for each scroll
    scroll_35360 = train_df[train_df.scroll_id == 35360].id.tolist()
    scroll_26002 = train_df[train_df.scroll_id == 26002].id.tolist()

    # Sample up to 10 from each
    np.random.seed(42)
    sample_35360 = sorted(np.random.choice(scroll_35360, min(10, len(scroll_35360)), replace=False))
    sample_26002 = sorted(np.random.choice(scroll_26002, min(10, len(scroll_26002)), replace=False))

    print(f"Scroll 35360: {len(scroll_35360)} available volumes, sampling {len(sample_35360)}")
    print(f"Scroll 26002: {len(scroll_26002)} available volumes, sampling {len(sample_26002)}")
    print()

    for scroll_name, sample_ids in [("35360", sample_35360), ("26002", sample_26002)]:
        print(f"--- Scroll {scroll_name} ---")
        print(f"{'vol_id':>12s} | {'fg%':>6s} {'unlbl%':>7s} {'bg%':>6s} | "
              f"{'mean_p':>6s} {'min_p':>6s} {'max_p':>6s} {'std_p':>6s} | "
              f"{'p>0.35':>7s} {'p>0.80':>7s}")
        print("-" * 95)

        all_fg_pct = []
        all_unlbl_pct = []
        all_bg_pct = []
        all_mean_p = []
        all_above_low = []
        all_above_high = []

        for vid in sample_ids:
            lbl = load_label(vid)
            prob = load_probmap(vid)

            total = lbl.size
            fg_pct = (lbl == 1).sum() / total * 100
            unlbl_pct = (lbl == 2).sum() / total * 100
            bg_pct = (lbl == 0).sum() / total * 100

            mean_p = float(prob.astype(np.float32).mean())
            min_p = float(prob.min())
            max_p = float(prob.max())
            std_p = float(prob.astype(np.float32).std())
            above_low = (prob > T_LOW).sum() / total * 100
            above_high = (prob > T_HIGH).sum() / total * 100

            all_fg_pct.append(fg_pct)
            all_unlbl_pct.append(unlbl_pct)
            all_bg_pct.append(bg_pct)
            all_mean_p.append(mean_p)
            all_above_low.append(above_low)
            all_above_high.append(above_high)

            print(f"{vid:>12d} | {fg_pct:5.2f}% {unlbl_pct:6.2f}% {bg_pct:5.1f}% | "
                  f"{mean_p:6.4f} {min_p:6.4f} {max_p:6.4f} {std_p:6.4f} | "
                  f"{above_low:6.2f}% {above_high:6.2f}%")

        print(f"{'MEAN':>12s} | {np.mean(all_fg_pct):5.2f}% {np.mean(all_unlbl_pct):6.2f}% "
              f"{np.mean(all_bg_pct):5.1f}% | "
              f"{np.mean(all_mean_p):6.4f} {'':>6s} {'':>6s} {'':>6s} | "
              f"{np.mean(all_above_low):6.2f}% {np.mean(all_above_high):6.2f}%")
        print()


# ============================================================
# PART 2: Sub-metric Breakdown
# ============================================================
def analyze_submetrics():
    print_separator("PART 2: SUB-METRIC BREAKDOWN — good vs bad volumes from scroll 35360")

    eval_df = pd.read_csv(ROOT / "logs" / "cross_scroll_eval.csv")
    scroll_35360 = eval_df[eval_df.scroll_id == 35360].sort_values("comp_score")

    print("All scroll 35360 volumes ranked by comp_score:")
    for _, row in scroll_35360.iterrows():
        print(f"  {row.vol_id:>12.0f}  comp_score={row.comp_score:.4f}")
    print()

    # Pick 5 worst and 5 best
    bad_vols = scroll_35360.head(5).vol_id.astype(int).tolist()
    good_vols = scroll_35360.tail(5).vol_id.astype(int).tolist()

    print(f"Bad volumes (score < 0.30):  {bad_vols}")
    print(f"Good volumes (score > 0.55): {good_vols}")
    print()

    print(f"{'vol_id':>12s} | {'comp':>6s} {'topo':>6s} {'sdice':>6s} {'voi':>6s} | "
          f"{'pred_fg%':>8s} {'lbl_fg%':>7s} {'overlap%':>8s}")
    print("-" * 85)

    for group_name, vol_list in [("BAD", bad_vols), ("GOOD", good_vols)]:
        print(f"  --- {group_name} ---")
        for vid in vol_list:
            t0 = time.time()
            prob = load_probmap(vid)
            lbl = load_label(vid)
            pred = postprocess(prob)

            # Downsample 4x for speed
            scale = 0.25
            pred_ds = scipy_zoom(pred, scale, order=0).astype(np.uint8)
            lbl_ds = scipy_zoom(lbl, scale, order=0).astype(np.uint8)

            report = compute_leaderboard_score(
                pred_ds, lbl_ds, ignore_label=2, spacing=(1, 1, 1),
                surface_tolerance=2.0, voi_alpha=0.3,
                combine_weights=(0.3, 0.35, 0.35),
            )

            # Compute overlap stats on full res for labeled region
            mask = lbl != 2  # labeled region
            pred_fg_in_labeled = pred[mask].sum() / mask.sum() * 100
            lbl_fg_in_labeled = (lbl[mask] == 1).sum() / mask.sum() * 100

            # True positive overlap
            tp = ((pred == 1) & (lbl == 1)).sum()
            fp = ((pred == 1) & (lbl == 0)).sum()
            fn = ((pred == 0) & (lbl == 1)).sum()
            precision = tp / (tp + fp + 1e-8) * 100
            recall = tp / (tp + fn + 1e-8) * 100

            elapsed = time.time() - t0

            print(f"{vid:>12d} | {report.score:.4f} {report.topo.toposcore:.4f} "
                  f"{report.surface_dice:.4f} {report.voi.voi_score:.4f} | "
                  f"{pred_fg_in_labeled:7.2f}% {lbl_fg_in_labeled:6.2f}% "
                  f"P={precision:.1f}% R={recall:.1f}%  ({elapsed:.1f}s)")
        print()


# ============================================================
# PART 3: Probmap Quality Visual Check
# ============================================================
def analyze_probmap_quality():
    print_separator("PART 3: PROBMAP QUALITY — middle-slice analysis (z=160)")

    eval_df = pd.read_csv(ROOT / "logs" / "cross_scroll_eval.csv")
    scroll_35360 = eval_df[eval_df.scroll_id == 35360].sort_values("comp_score")

    # Pick the best and worst
    worst_vol = int(scroll_35360.iloc[0].vol_id)
    best_vol = int(scroll_35360.iloc[-1].vol_id)

    print(f"Worst volume: {worst_vol} (score={scroll_35360.iloc[0].comp_score:.4f})")
    print(f"Best volume:  {best_vol} (score={scroll_35360.iloc[-1].comp_score:.4f})")
    print()

    for name, vid in [("WORST", worst_vol), ("BEST", best_vol)]:
        print(f"--- {name} volume: {vid} ---")
        prob = load_probmap(vid)
        lbl = load_label(vid)
        pred = postprocess(prob)

        z = 160
        prob_slice = prob[z]
        lbl_slice = lbl[z]
        pred_slice = pred[z]

        total_px = prob_slice.size
        labeled_mask = lbl_slice != 2

        print(f"  Full volume stats:")
        print(f"    Label: fg={((lbl==1).sum()/lbl.size*100):.2f}%, "
              f"unlabeled={((lbl==2).sum()/lbl.size*100):.2f}%, "
              f"bg={((lbl==0).sum()/lbl.size*100):.2f}%")
        print(f"    Probmap: mean={prob.mean():.4f}, std={prob.std():.4f}")
        print(f"    Pred: fg={((pred==1).sum()/pred.size*100):.2f}%")
        print()

        print(f"  Middle slice z={z} stats:")
        print(f"    Total pixels: {total_px}")
        print(f"    Labeled pixels: {labeled_mask.sum()} ({labeled_mask.sum()/total_px*100:.1f}%)")
        print(f"    Unlabeled pixels: {(~labeled_mask).sum()} ({(~labeled_mask).sum()/total_px*100:.1f}%)")
        print()

        # Overall slice
        print(f"    Prob > 0.5:  {(prob_slice > 0.5).sum():>7d} ({(prob_slice > 0.5).sum()/total_px*100:.2f}%)")
        print(f"    Prob > 0.35: {(prob_slice > 0.35).sum():>7d} ({(prob_slice > 0.35).sum()/total_px*100:.2f}%)")
        print(f"    Prob > 0.80: {(prob_slice > 0.80).sum():>7d} ({(prob_slice > 0.80).sum()/total_px*100:.2f}%)")
        print(f"    Label == 1:  {(lbl_slice == 1).sum():>7d} ({(lbl_slice == 1).sum()/total_px*100:.2f}%)")
        print(f"    Pred == 1:   {(pred_slice == 1).sum():>7d} ({(pred_slice == 1).sum()/total_px*100:.2f}%)")
        print()

        # In labeled region only
        if labeled_mask.sum() > 0:
            lbl_l = lbl_slice[labeled_mask]
            pred_l = pred_slice[labeled_mask]
            prob_l = prob_slice[labeled_mask]

            tp = ((pred_l == 1) & (lbl_l == 1)).sum()
            fp = ((pred_l == 1) & (lbl_l == 0)).sum()
            fn = ((pred_l == 0) & (lbl_l == 1)).sum()
            tn = ((pred_l == 0) & (lbl_l == 0)).sum()

            print(f"    In labeled region (z={z}):")
            print(f"      True Positives:  {tp:>7d}  (correctly detected fg)")
            print(f"      False Positives: {fp:>7d}  (predicted fg where label=bg)")
            print(f"      False Negatives: {fn:>7d}  (missed fg where label=fg)")
            print(f"      True Negatives:  {tn:>7d}  (correctly rejected bg)")
            precision = tp / (tp + fp + 1e-8) * 100
            recall = tp / (tp + fn + 1e-8) * 100
            print(f"      Precision: {precision:.1f}%  Recall: {recall:.1f}%")
            print()

            # Where label==1, what does the probmap look like?
            fg_mask = lbl_slice == 1
            if fg_mask.sum() > 0:
                fg_probs = prob_slice[fg_mask]
                print(f"    Probmap values where label==1 (z={z}):")
                print(f"      n={fg_mask.sum()}, mean={fg_probs.mean():.4f}, "
                      f"median={np.median(fg_probs):.4f}, "
                      f"std={fg_probs.std():.4f}")
                print(f"      min={fg_probs.min():.4f}, max={fg_probs.max():.4f}")
                pcts = [10, 25, 50, 75, 90]
                vals = np.percentile(fg_probs, pcts)
                print(f"      Percentiles: " + ", ".join(f"p{p}={v:.3f}" for p, v in zip(pcts, vals)))
                print(f"      Below T_LOW (0.35): {(fg_probs < T_LOW).sum()} "
                      f"({(fg_probs < T_LOW).sum()/fg_mask.sum()*100:.1f}%)")
                print(f"      Below T_HIGH (0.80): {(fg_probs < T_HIGH).sum()} "
                      f"({(fg_probs < T_HIGH).sum()/fg_mask.sum()*100:.1f}%)")
            else:
                print(f"    No foreground labels in z={z} slice")

            # Where label==0 (bg), what does the probmap look like?
            bg_mask = (lbl_slice == 0) & labeled_mask
            if bg_mask.sum() > 0:
                bg_probs = prob_slice[bg_mask]
                print(f"    Probmap values where label==0 (bg) (z={z}):")
                print(f"      n={bg_mask.sum()}, mean={bg_probs.mean():.4f}, "
                      f"median={np.median(bg_probs):.4f}")
                print(f"      Above T_LOW (0.35): {(bg_probs > T_LOW).sum()} "
                      f"({(bg_probs > T_LOW).sum()/bg_mask.sum()*100:.1f}%)")
                print(f"      Above T_HIGH (0.80): {(bg_probs > T_HIGH).sum()} "
                      f"({(bg_probs > T_HIGH).sum()/bg_mask.sum()*100:.1f}%)")

        print()

    # Also check: across all z-slices, how does the fg distribution look?
    print("--- Across all Z slices (worst volume) ---")
    vid = worst_vol
    prob = load_probmap(vid)
    lbl = load_label(vid)
    pred = postprocess(prob)

    for z in [0, 40, 80, 120, 160, 200, 240, 280, 319]:
        lbl_z = lbl[z]
        prob_z = prob[z]
        pred_z = pred[z]
        labeled_z = lbl_z != 2
        fg_z = (lbl_z == 1).sum()
        pred_fg_z = pred_z.sum()
        prob_above_half = (prob_z > 0.5).sum()
        if labeled_z.sum() > 0:
            tp = ((pred_z == 1) & (lbl_z == 1)).sum()
            fp = ((pred_z == 1) & (lbl_z == 0)).sum()
            fn = ((pred_z == 0) & (lbl_z == 1)).sum()
        else:
            tp = fp = fn = 0
        print(f"  z={z:>3d}: labeled={labeled_z.sum():>6d} fg={fg_z:>5d} "
              f"pred_fg={pred_fg_z:>5d} prob>0.5={prob_above_half:>5d} "
              f"TP={tp:>5d} FP={fp:>5d} FN={fn:>5d}")


# ============================================================
# PART 4: Summary and Diagnosis
# ============================================================
def summary_diagnosis():
    print_separator("PART 4: SUMMARY DIAGNOSIS")

    eval_df = pd.read_csv(ROOT / "logs" / "cross_scroll_eval.csv")

    # Per-scroll stats
    for sid in sorted(eval_df.scroll_id.unique()):
        sub = eval_df[eval_df.scroll_id == sid]
        print(f"Scroll {sid}: n={len(sub)}, mean={sub.comp_score.mean():.4f}, "
              f"std={sub.comp_score.std():.4f}, "
              f"min={sub.comp_score.min():.4f}, max={sub.comp_score.max():.4f}")

    # Bimodal check for 35360
    s35 = eval_df[eval_df.scroll_id == 35360].comp_score.values
    print(f"\nScroll 35360 score distribution:")
    print(f"  Below 0.35: {(s35 < 0.35).sum()} volumes")
    print(f"  0.35-0.50:  {((s35 >= 0.35) & (s35 < 0.50)).sum()} volumes")
    print(f"  Above 0.50: {(s35 >= 0.50).sum()} volumes")

    # Check if there's a pattern — do bad volumes have specific characteristics?
    print(f"\nScroll 35360 volumes sorted by score:")
    scroll_35 = eval_df[eval_df.scroll_id == 35360].sort_values("comp_score")
    for _, row in scroll_35.iterrows():
        vid = int(row.vol_id)
        # Quick check: label distribution
        lbl = load_label(vid)
        fg_pct = (lbl == 1).sum() / lbl.size * 100
        unlbl_pct = (lbl == 2).sum() / lbl.size * 100
        prob = load_probmap(vid)
        mean_p = prob.mean()
        above_high = (prob > T_HIGH).sum() / prob.size * 100
        print(f"  {vid:>12d} score={row.comp_score:.4f} | fg={fg_pct:.2f}% unlbl={unlbl_pct:.1f}% "
              f"| mean_p={mean_p:.4f} p>0.80={above_high:.2f}%")


def main():
    print("=" * 80)
    print("  SCROLL 35360 PERFORMANCE ANALYSIS")
    print("  Vesuvius Surface Detection Competition")
    print("=" * 80)

    t_total = time.time()

    analyze_data_characteristics()
    analyze_submetrics()
    analyze_probmap_quality()
    summary_diagnosis()

    elapsed = time.time() - t_total
    print(f"\nTotal analysis time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
