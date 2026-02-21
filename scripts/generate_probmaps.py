#!/usr/bin/env python3
"""
Generate probmaps for all training volumes using a pretrained/SWA model.

This is the first stage of the pseudo-labeling pipeline:
1. generate_probmaps.py  — this script (inference → .npy probmaps)
2. generate_pseudo_labels.py — probmaps → pseudo-labeled .tif files
3. train with --label-dir pointing to pseudo-labels

Usage:
    # Dry run (2 volumes)
    python scripts/generate_probmaps.py --dry-run

    # Full run with best SWA blend
    python scripts/generate_probmaps.py --weights checkpoints/swa_topo/swa_70pre_30topo_ep5.weights.h5

    # Resume (skip already-generated probmaps)
    python scripts/generate_probmaps.py --skip-existing

    # Specific scrolls only
    python scripts/generate_probmaps.py --scroll-ids 34117 35360
"""

import os
os.environ.setdefault('KERAS_BACKEND', 'torch')

# Force TensorFlow to CPU-only
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

ROOT = Path("/workspace/vesuvius-kaggle-competition")
TRAIN_IMG = ROOT / "data" / "train_images"
TRAIN_LBL = ROOT / "data" / "train_labels"
DEFAULT_WEIGHTS = ROOT / "checkpoints" / "swa_topo" / "swa_70pre_30topo_ep5.weights.h5"
DEFAULT_OUTPUT = ROOT / "data" / "pseudo_probmaps"

ROI = (160, 160, 160)
VAL_SCROLL = 26002


# ── Reuse inference utilities from eval_transunet.py ──
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


def multiclass_logits_to_binary_prob(logits_5d):
    x = np.asarray(logits_5d, dtype=np.float32)[0]
    L0, L1, L2 = x[..., 0], x[..., 1], x[..., 2]
    binary_logit = logsumexp2(L1, L2) - L0
    return sigmoid_stable(binary_logit)


def iter_tta(volume):
    volume = np.asarray(volume)
    yield volume, (lambda y: y)
    for axis in [1, 2, 3]:
        v = np.flip(volume, axis=axis).copy()
        inv = (lambda y, ax=axis: np.flip(np.asarray(y), axis=ax).copy())
        yield v, inv
    for k in [1, 2, 3]:
        v = np.rot90(volume, k=k, axes=(2, 3)).copy()
        inv = (lambda y, kk=k: np.rot90(np.asarray(y), k=-kk, axes=(2, 3)).copy())
        yield v, inv


def gpu_cleanup():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_volume(model, swi, volume_5d, use_tta=False):
    import torch

    if not use_tta:
        with torch.no_grad():
            logits = np.asarray(swi(volume_5d))
        prob = multiclass_logits_to_binary_prob(logits)
        del logits
        gpu_cleanup()
        return prob

    s_sum = None
    n = 0
    for v, inv in iter_tta(volume_5d):
        with torch.no_grad():
            logits = np.asarray(swi(v))
        logits = inv(logits)
        x = logits[0].astype(np.float32)
        del logits
        L0, L1, L2 = x[..., 0], x[..., 1], x[..., 2]
        s = logsumexp2(L1, L2) - L0
        del x, L0, L1, L2
        s_sum = s if s_sum is None else s_sum + s
        del s
        n += 1
        gpu_cleanup()

    return sigmoid_stable(s_sum / float(n))


def normalize_volume(vol_5d):
    from medicai.transforms import Compose, NormalizeIntensity
    pipeline = Compose([
        NormalizeIntensity(keys=["image"], nonzero=True, channel_wise=False),
    ])
    result = pipeline({"image": vol_5d})["image"]
    return np.asarray(result, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description='Generate probmaps for pseudo-labeling')
    parser.add_argument('--weights', type=str, default=str(DEFAULT_WEIGHTS),
                        help='Path to model weights')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT),
                        help='Output directory for probmaps')
    parser.add_argument('--volume-ids', type=int, nargs='+', default=None,
                        help='Specific volume IDs to process')
    parser.add_argument('--scroll-ids', type=int, nargs='+', default=None,
                        help='Specific scroll IDs to process')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip volumes that already have a probmap')
    parser.add_argument('--tta', action='store_true',
                        help='Use 7-fold TTA (7x slower but better quality)')
    parser.add_argument('--overlap', type=float, default=0.50,
                        help='SlidingWindowInference overlap')
    parser.add_argument('--dry-run', action='store_true',
                        help='Process only 2 volumes then stop')
    parser.add_argument('--include-val', action='store_true',
                        help='Include validation scroll (26002). Default: training only.')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load volume list
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    available = set(int(p.stem) for p in TRAIN_IMG.glob("*.tif"))
    train_df = train_df[train_df.id.isin(available)]

    # Filter to training scrolls (exclude validation by default)
    if not args.include_val:
        train_df = train_df[train_df.scroll_id != VAL_SCROLL]

    # Filter by scroll IDs if specified
    if args.scroll_ids:
        train_df = train_df[train_df.scroll_id.isin(args.scroll_ids)]

    # Filter by specific volume IDs if specified
    if args.volume_ids:
        train_df = train_df[train_df.id.isin(args.volume_ids)]

    vol_ids = train_df.id.tolist()

    # Skip existing probmaps
    if args.skip_existing:
        before = len(vol_ids)
        vol_ids = [v for v in vol_ids if not (output_dir / f"{v}.npy").exists()]
        skipped = before - len(vol_ids)
        if skipped > 0:
            print(f"Skipping {skipped} volumes with existing probmaps")

    if args.dry_run:
        vol_ids = vol_ids[:2]

    print(f"=== Probmap Generation ===")
    print(f"  Weights: {args.weights}")
    print(f"  Output: {output_dir}")
    print(f"  Volumes: {len(vol_ids)}")
    print(f"  TTA: {args.tta}")
    print(f"  Overlap: {args.overlap}")
    if args.dry_run:
        print(f"  DRY RUN: processing only {len(vol_ids)} volumes")
    print()

    # Load model
    from medicai.models import TransUNet
    from medicai.utils.inference import SlidingWindowInference

    model = TransUNet(
        input_shape=(160, 160, 160, 1),
        encoder_name='seresnext50',
        classifier_activation=None,
        num_classes=3,
    )
    model.load_weights(str(args.weights))
    print(f"Model loaded: {model.count_params() / 1e6:.1f}M params")

    swi = SlidingWindowInference(
        model, num_classes=3, roi_size=ROI,
        sw_batch_size=1, mode='gaussian', overlap=float(args.overlap),
    )

    # Warmup
    print("Warming up (first SWI pass compiles the model)...")
    dummy = np.zeros((1, 160, 160, 160, 1), dtype=np.float32)
    import torch
    with torch.no_grad():
        _ = np.asarray(swi(dummy))
    del dummy
    gpu_cleanup()
    print("Warmup done.\n")

    # Process volumes
    t_global = time.time()
    total_bytes = 0

    for i, vid in enumerate(vol_ids):
        t0 = time.time()
        gpu_cleanup()

        # Load and normalize
        img = tifffile.imread(TRAIN_IMG / f"{vid}.tif").astype(np.float32)
        vol_5d = img[None, ..., None]
        del img
        vol_5d = normalize_volume(vol_5d)

        # Inference
        with torch.no_grad():
            prob = predict_volume(model, swi, vol_5d, use_tta=args.tta)
        del vol_5d
        gpu_cleanup()

        # Save
        out_path = output_dir / f"{vid}.npy"
        prob_f16 = prob.astype(np.float16)
        np.save(out_path, prob_f16)
        file_size = out_path.stat().st_size
        total_bytes += file_size

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_global
        eta_min = (total_elapsed / (i + 1)) * (len(vol_ids) - i - 1) / 60

        scroll_id = train_df[train_df.id == vid].scroll_id.values[0]
        print(f"[{i+1}/{len(vol_ids)}] vol={vid} scroll={scroll_id} | "
              f"shape={prob.shape} prob_max={prob.max():.3f} | "
              f"{file_size/1e6:.1f}MB | {elapsed:.0f}s (ETA {eta_min:.1f}min)")

        del prob, prob_f16

    total_elapsed = time.time() - t_global
    print(f"\n=== Done ===")
    print(f"  Volumes processed: {len(vol_ids)}")
    print(f"  Total size: {total_bytes / 1e9:.1f} GB")
    print(f"  Total time: {total_elapsed / 60:.1f} min ({total_elapsed / 3600:.1f} hrs)")
    print(f"  Avg per volume: {total_elapsed / max(len(vol_ids), 1):.0f}s")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    main()
