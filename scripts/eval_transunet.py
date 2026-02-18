#!/usr/bin/env python3
"""
Evaluate pretrained TransUNet on validation volumes.

Runs SlidingWindowInference, applies post-processing, scores at full resolution (ds=1).
Saves probmaps for reuse by post-processing sweep and exploration notebook.

Usage:
    # Quick test (3 volumes, no TTA)
    python scripts/eval_transunet.py --n-eval 3

    # Full validation (all 88 scroll-26002 volumes, with TTA)
    python scripts/eval_transunet.py --tta

    # Dry run (1 volume, fast check)
    python scripts/eval_transunet.py --dry-run

    # Use specific weights
    python scripts/eval_transunet.py --weights pretrained_weights/transunet/transunet.seresnext50.160px.weights.h5

    # Also eval on other scrolls
    python scripts/eval_transunet.py --cross-scroll --max-per-scroll 5
"""

import os
os.environ.setdefault('KERAS_BACKEND', 'torch')

# Force TensorFlow to CPU-only — Keras 3 imports TF even with torch backend,
# and TF grabs ~15 GiB GPU by default, causing OOM alongside PyTorch.
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

ROOT = Path("/workspace/vesuvius-kaggle-competition")
TRAIN_IMG = ROOT / "data" / "train_images"
TRAIN_LBL = ROOT / "data" / "train_labels"
DEFAULT_WEIGHTS = ROOT / "pretrained_weights" / "transunet" / "transunet.seresnext50.160px.comboloss.weights.h5"
PROBMAP_DIR = ROOT / "data" / "transunet_probmaps"

ROI = (160, 160, 160)
VAL_SCROLL = 26002


def load_model(weights_path):
    from medicai.models import TransUNet
    model = TransUNet(
        input_shape=(160, 160, 160, 1),
        encoder_name='seresnext50',
        classifier_activation=None,
        num_classes=3,
    )
    model.load_weights(str(weights_path))
    print(f"Model loaded: {model.count_params() / 1e6:.1f}M params")
    return model


def build_swi(model, overlap=0.50):
    from medicai.utils.inference import SlidingWindowInference
    return SlidingWindowInference(
        model,
        num_classes=3,
        roi_size=ROI,
        sw_batch_size=1,
        mode='gaussian',
        overlap=float(overlap),
    )


def normalize_volume(vol_5d):
    from medicai.transforms import Compose, NormalizeIntensity
    pipeline = Compose([
        NormalizeIntensity(keys=["image"], nonzero=True, channel_wise=False),
    ])
    result = pipeline({"image": vol_5d})["image"]
    return np.asarray(result, dtype=np.float32)  # ensure numpy, not TF tensor


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
    """Convert 3-class logits (N,D,H,W,3) to binary probability (D,H,W)."""
    x = np.asarray(logits_5d, dtype=np.float32)[0]  # (D,H,W,3)
    L0, L1, L2 = x[..., 0], x[..., 1], x[..., 2]
    binary_logit = logsumexp2(L1, L2) - L0
    return sigmoid_stable(binary_logit)


def iter_tta(volume):
    """Generate 7 TTA augmentations: identity + 3 flips + 3 rotations."""
    # Ensure numpy (medicai normalize can return TF EagerTensor)
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
    """Aggressively free GPU memory."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_volume(model, swi, volume_5d, use_tta=False):
    """Run inference on a volume, return binary probability map."""
    import torch

    if not use_tta:
        with torch.no_grad():
            logits = np.asarray(swi(volume_5d))
        prob = multiclass_logits_to_binary_prob(logits)
        del logits
        gpu_cleanup()
        return prob

    # TTA: average binary logits across 7 augmentations
    s_sum = None
    n = 0
    for v, inv in iter_tta(volume_5d):
        with torch.no_grad():
            logits = np.asarray(swi(v))
        logits = inv(logits)
        x = logits[0].astype(np.float32)  # (D,H,W,3)
        del logits
        L0, L1, L2 = x[..., 0], x[..., 1], x[..., 2]
        s = logsumexp2(L1, L2) - L0
        del x, L0, L1, L2
        s_sum = s if s_sum is None else s_sum + s
        del s
        n += 1
        gpu_cleanup()

    return sigmoid_stable(s_sum / float(n))


# ── Post-processing ──────────────────────────────────────
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


def postprocess(prob, t_low=0.50, t_high=0.90, z_radius=3, xy_radius=2,
                dust_min_size=100):
    """Hysteresis + closing + dust removal. Uses competitor defaults."""
    strong = prob >= t_high
    weak = prob >= t_low
    struct_hyst = generate_binary_structure(3, 3)

    if not strong.any():
        return np.zeros_like(prob, dtype=np.uint8)

    mask = binary_propagation(strong, structure=struct_hyst, mask=weak)

    struct_close = build_anisotropic_struct(z_radius, xy_radius)
    mask = binary_closing(mask, structure=struct_close)

    # Dust removal
    labeled, n = scipy_label(mask)
    if n > 0:
        sizes = np.bincount(labeled.ravel())
        small = sizes < dust_min_size
        small[0] = False
        mask[small[labeled]] = 0

    return mask.astype(np.uint8)


# ── Scoring ──────────────────────────────────────────────
def score_volume(pred, lbl, downsample=1):
    from topometrics.leaderboard import compute_leaderboard_score
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
    parser.add_argument('--weights', type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument('--n-eval', type=int, default=0,
                        help='Number of val volumes (0=all)')
    parser.add_argument('--tta', action='store_true', help='Use 7-fold TTA')
    parser.add_argument('--overlap', type=float, default=0.50)
    parser.add_argument('--downsample', type=int, default=1,
                        help='Metric downsample (1=full res, 2=2x)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Test on 1 volume only')
    parser.add_argument('--cross-scroll', action='store_true',
                        help='Evaluate on all scrolls, not just val')
    parser.add_argument('--max-per-scroll', type=int, default=5)
    parser.add_argument('--save-probmaps', action='store_true',
                        help='Save probmaps to disk for reuse')
    parser.add_argument('--t-low', type=float, default=0.50)
    parser.add_argument('--t-high', type=float, default=0.90)
    args = parser.parse_args()

    if args.dry_run:
        args.n_eval = 1

    # Load model
    print(f"Loading TransUNet from {args.weights}")
    model = load_model(args.weights)
    swi = build_swi(model, overlap=args.overlap)

    # Warmup (compile once)
    print("Warming up (first SWI pass compiles the model)...")
    dummy = np.zeros((1, 160, 160, 160, 1), dtype=np.float32)
    _ = np.asarray(swi(dummy))
    print("Warmup done.")

    # Get volume IDs
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    available = set(int(p.stem) for p in TRAIN_IMG.glob("*.tif")) & \
                set(int(p.stem) for p in TRAIN_LBL.glob("*.tif"))
    train_df = train_df[train_df.id.isin(available)]

    if args.cross_scroll:
        eval_ids = []
        for sid, group in train_df.groupby('scroll_id'):
            ids = group.id.tolist()[:args.max_per_scroll]
            eval_ids.extend(ids)
        print(f"Cross-scroll eval: {len(eval_ids)} volumes")
    else:
        val_df = train_df[train_df.scroll_id == VAL_SCROLL]
        eval_ids = val_df.id.tolist()
        if args.n_eval > 0:
            eval_ids = eval_ids[:args.n_eval]
        print(f"Val eval (scroll {VAL_SCROLL}): {len(eval_ids)} volumes")

    # Probmap output dir
    if args.save_probmaps:
        PROBMAP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nConfig: overlap={args.overlap}, TTA={args.tta}, "
          f"T_low={args.t_low}, T_high={args.t_high}, ds={args.downsample}")
    print(f"{'='*70}")

    all_results = []
    t_global = time.time()

    import torch

    for i, vid in enumerate(eval_ids):
        t0 = time.time()

        # Aggressively free GPU memory between volumes
        gpu_cleanup()

        # Load volume
        img = tifffile.imread(TRAIN_IMG / f"{vid}.tif").astype(np.float32)
        lbl = tifffile.imread(TRAIN_LBL / f"{vid}.tif")

        # Normalize
        vol_5d = img[None, ..., None]  # (1,D,H,W,1)
        del img
        vol_5d = normalize_volume(vol_5d)

        # Inference
        with torch.no_grad():
            prob = predict_volume(model, swi, vol_5d, use_tta=args.tta)
        del vol_5d
        gpu_cleanup()

        # Save probmap
        if args.save_probmaps:
            np.save(PROBMAP_DIR / f"{vid}.npy", prob.astype(np.float16))

        # Post-process and score
        pred = postprocess(prob, t_low=args.t_low, t_high=args.t_high)
        scores = score_volume(pred, lbl, downsample=args.downsample)

        # Capture stats before cleanup
        prob_max = float(prob.max())
        prob_p95 = float(np.percentile(prob[prob > 0.01], 95)) if (prob > 0.01).any() else 0
        fg_voxels = int(pred.sum())

        # Find scroll_id for this volume
        scroll_id = train_df[train_df.id == vid].scroll_id.values[0]

        result = {
            'vol_id': vid,
            'scroll_id': scroll_id,
            'prob_max': prob_max,
            'prob_p95': prob_p95,
            'fg_voxels': fg_voxels,
            **scores,
        }
        all_results.append(result)

        # Cleanup large arrays
        del prob, pred, lbl

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_global
        eta_min = (total_elapsed / (i + 1)) * (len(eval_ids) - i - 1) / 60

        print(f"[{i+1}/{len(eval_ids)}] vol={vid} scroll={scroll_id} | "
              f"comp={scores['comp_score']:.4f} topo={scores['topo']:.4f} "
              f"sdice={scores['sdice']:.4f} voi={scores['voi']:.4f} | "
              f"prob_max={prob_max:.3f} fg={fg_voxels} | "
              f"{elapsed:.0f}s (ETA {eta_min:.1f}min)")

    # Summary
    df = pd.DataFrame(all_results)
    print(f"\n{'='*70}")
    print("TRANSUNET EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Weights: {Path(args.weights).name}")
    print(f"TTA: {args.tta}, Overlap: {args.overlap}, DS: {args.downsample}")
    print(f"Thresholds: T_low={args.t_low}, T_high={args.t_high}")
    print(f"Total time: {(time.time() - t_global) / 60:.1f} min")
    print()

    # Per-scroll breakdown
    if args.cross_scroll:
        print("Per-scroll breakdown:")
        for sid, group in df.groupby('scroll_id'):
            print(f"  scroll {sid} (n={len(group)}): "
                  f"comp={group.comp_score.mean():.4f} +/- {group.comp_score.std():.4f}")
        print()

    print(f"Overall (n={len(df)}):")
    print(f"  comp_score: {df.comp_score.mean():.4f} +/- {df.comp_score.std():.4f}")
    print(f"  topo:       {df.topo.mean():.4f}")
    print(f"  sdice:      {df.sdice.mean():.4f}")
    print(f"  voi:        {df.voi.mean():.4f}")
    print(f"  prob_max:   {df.prob_max.mean():.3f} (range {df.prob_max.min():.3f}-{df.prob_max.max():.3f})")

    # Save results
    out_path = ROOT / "logs" / "transunet_eval.csv"
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nDetailed results saved to {out_path}")

    # Compare to SegResNet baseline
    print(f"\nFor reference: SegResNet v9 at ds=1 scored ~0.4113 (5 volumes)")


if __name__ == '__main__':
    main()
