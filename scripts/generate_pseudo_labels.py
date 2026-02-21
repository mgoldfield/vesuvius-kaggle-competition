#!/usr/bin/env python3
"""
Generate pseudo-labeled .tif files from probmaps.

Stage 2 of pseudo-labeling pipeline:
- Reads original labels and model probmaps
- For label=2 (unlabeled) voxels with high-confidence predictions,
  converts them to label=1 (fg) or label=0 (bg)
- Original label=0 and label=1 are never changed

Usage:
    # Dry run (2 volumes, print stats)
    python scripts/generate_pseudo_labels.py --dry-run

    # Full run with default thresholds (0.85/0.15)
    python scripts/generate_pseudo_labels.py

    # Custom thresholds
    python scripts/generate_pseudo_labels.py --fg-threshold 0.90 --bg-threshold 0.10
"""

import argparse
import time
import numpy as np
import tifffile
from pathlib import Path

ROOT = Path("/workspace/vesuvius-kaggle-competition")
TRAIN_LBL = ROOT / "data" / "train_labels"
DEFAULT_PROBMAP_DIR = ROOT / "data" / "pseudo_probmaps"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "pseudo_labels"


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo-labeled .tif files')
    parser.add_argument('--probmap-dir', type=str, default=str(DEFAULT_PROBMAP_DIR),
                        help='Directory containing .npy probmaps')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help='Output directory for pseudo-labeled .tif files')
    parser.add_argument('--fg-threshold', type=float, default=0.85,
                        help='Probability above which label=2 becomes label=1')
    parser.add_argument('--bg-threshold', type=float, default=0.15,
                        help='Probability below which label=2 becomes label=0')
    parser.add_argument('--dry-run', action='store_true',
                        help='Process only 2 volumes, print detailed stats')
    args = parser.parse_args()

    probmap_dir = Path(args.probmap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all probmaps
    probmap_files = sorted(probmap_dir.glob("*.npy"))
    if not probmap_files:
        print(f"ERROR: No .npy files found in {probmap_dir}")
        return

    if args.dry_run:
        probmap_files = probmap_files[:2]

    print(f"=== Pseudo-Label Generation ===")
    print(f"  Probmap dir: {probmap_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  FG threshold: >= {args.fg_threshold}")
    print(f"  BG threshold: <= {args.bg_threshold}")
    print(f"  Volumes: {len(probmap_files)}")
    if args.dry_run:
        print(f"  DRY RUN: processing only {len(probmap_files)} volumes")
    print()

    t_global = time.time()
    total_stats = {
        'total_voxels': 0,
        'orig_fg': 0,
        'orig_bg': 0,
        'orig_unlabeled': 0,
        'pseudo_fg': 0,
        'pseudo_bg': 0,
        'remaining_unlabeled': 0,
    }

    for i, probmap_path in enumerate(probmap_files):
        t0 = time.time()
        vid = probmap_path.stem

        # Load original label and probmap
        lbl_path = TRAIN_LBL / f"{vid}.tif"
        if not lbl_path.exists():
            print(f"  WARNING: No label file for {vid}, skipping")
            continue

        orig_lbl = tifffile.imread(str(lbl_path))
        prob = np.load(str(probmap_path)).astype(np.float32)

        # Verify shapes match
        if orig_lbl.shape != prob.shape:
            print(f"  WARNING: Shape mismatch for {vid}: lbl={orig_lbl.shape} prob={prob.shape}, skipping")
            continue

        # Create pseudo-labels
        new_lbl = orig_lbl.copy()
        unlabeled = (orig_lbl == 2)

        # Convert high-confidence unlabeled voxels
        pseudo_fg_mask = unlabeled & (prob >= args.fg_threshold)
        pseudo_bg_mask = unlabeled & (prob <= args.bg_threshold)

        new_lbl[pseudo_fg_mask] = 1
        new_lbl[pseudo_bg_mask] = 0

        # Stats
        n_total = orig_lbl.size
        n_orig_fg = int((orig_lbl == 1).sum())
        n_orig_bg = int((orig_lbl == 0).sum())
        n_orig_unlabeled = int(unlabeled.sum())
        n_pseudo_fg = int(pseudo_fg_mask.sum())
        n_pseudo_bg = int(pseudo_bg_mask.sum())
        n_remaining = int((new_lbl == 2).sum())
        n_converted = n_pseudo_fg + n_pseudo_bg
        pct_converted = 100.0 * n_converted / max(n_orig_unlabeled, 1)

        total_stats['total_voxels'] += n_total
        total_stats['orig_fg'] += n_orig_fg
        total_stats['orig_bg'] += n_orig_bg
        total_stats['orig_unlabeled'] += n_orig_unlabeled
        total_stats['pseudo_fg'] += n_pseudo_fg
        total_stats['pseudo_bg'] += n_pseudo_bg
        total_stats['remaining_unlabeled'] += n_remaining

        # Save
        tifffile.imwrite(str(output_dir / f"{vid}.tif"), new_lbl)

        elapsed = time.time() - t0

        print(f"[{i+1}/{len(probmap_files)}] vol={vid} | "
              f"unlabeled: {n_orig_unlabeled} → pseudo_fg: +{n_pseudo_fg}, pseudo_bg: +{n_pseudo_bg} "
              f"({pct_converted:.1f}% converted) | remaining: {n_remaining} | {elapsed:.1f}s")

    # Summary
    total_elapsed = time.time() - t_global
    s = total_stats
    total_converted = s['pseudo_fg'] + s['pseudo_bg']
    pct = 100.0 * total_converted / max(s['orig_unlabeled'], 1)

    print(f"\n=== Summary ===")
    print(f"  Volumes processed: {len(probmap_files)}")
    print(f"  Total voxels: {s['total_voxels']:,}")
    print(f"  Original labels: fg={s['orig_fg']:,}, bg={s['orig_bg']:,}, unlabeled={s['orig_unlabeled']:,}")
    print(f"  Pseudo-labeled: +fg={s['pseudo_fg']:,}, +bg={s['pseudo_bg']:,} ({pct:.1f}% of unlabeled converted)")
    print(f"  Remaining unlabeled: {s['remaining_unlabeled']:,}")
    print(f"  New fg total: {s['orig_fg'] + s['pseudo_fg']:,} ({100.0 * (s['orig_fg'] + s['pseudo_fg']) / max(s['total_voxels'], 1):.1f}% of volume)")
    print(f"  Time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    main()
