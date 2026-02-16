#!/usr/bin/env python3
"""
Post-training checkpoint sweep: evaluate all periodic checkpoints with full
inference pipeline to find the best model.

In-training comp_score uses simplified inference; the full pipeline (Gaussian
SWI + logit TTA + hysteresis) gives ~2x better scores. So the "best" epoch
during training may not be the best with full inference.

Two-stage approach for efficiency:
  Stage 1 (fast): Gaussian SWI only (no TTA) on N val volumes — rank checkpoints
  Stage 2 (precise): Full 7-fold TTA on top K checkpoints with more volumes

Usage:
    # Quick sweep of all v12 checkpoints (no TTA, 10 volumes)
    python scripts/eval_checkpoint_sweep.py --version v12

    # Full eval of top 3 from sweep (with TTA, 20 volumes)
    python scripts/eval_checkpoint_sweep.py --version v12 --tta --n-volumes 20

    # Sweep v13 (3-class model)
    python scripts/eval_checkpoint_sweep.py --version v13 --three-class

    # Evaluate a specific checkpoint
    python scripts/eval_checkpoint_sweep.py --checkpoint checkpoints/models/segresnet_v12_ep30.pth
"""

import argparse
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
import time
from pathlib import Path
from scipy.ndimage import (
    generate_binary_structure, binary_propagation,
    binary_closing, label as scipy_label, zoom as scipy_zoom,
)
from monai.networks.nets import SegResNet
from topometrics.leaderboard import compute_leaderboard_score

ROOT = Path("/workspace/vesuvius-kaggle-competition")
TRAIN_IMG = ROOT / "data" / "train_images"
TRAIN_LBL = ROOT / "data" / "train_labels"
CKPT_DIR = ROOT / "checkpoints" / "models"

PATCH_SIZE = 160
STRIDE = 80
T_LOW = 0.35
T_HIGH = 0.75
CLOSING_Z_RADIUS = 2
CLOSING_XY_RADIUS = 1
DUST_MIN_SIZE = 100
METRIC_DOWNSAMPLE = 4
VAL_SCROLL = 26002


# ── Gaussian importance map ───────────────────────────────
def build_gaussian_map(patch_size, sigma_scale=0.125):
    sigma = patch_size * sigma_scale
    ax = np.arange(patch_size, dtype=np.float32)
    center = (patch_size - 1) / 2.0
    gauss_1d = np.exp(-0.5 * ((ax - center) / sigma) ** 2)
    gauss_3d = gauss_1d[:, None, None] * gauss_1d[None, :, None] * gauss_1d[None, None, :]
    gauss_3d /= gauss_3d.max()
    gauss_3d = np.clip(gauss_3d, 1e-6, None)
    return gauss_3d


GAUSSIAN_MAP = build_gaussian_map(PATCH_SIZE)


# ── Sliding window positions ─────────────────────────────
def _positions(length, ps, stride):
    pos = list(range(0, length - ps, stride))
    if not pos or pos[-1] + ps < length:
        pos.append(length - ps)
    return pos


# ── Model loading ─────────────────────────────────────────
def load_model(checkpoint_path, out_channels=1):
    """Load SegResNet from a fastai checkpoint."""
    model = SegResNet(
        spatial_dims=3, in_channels=1, out_channels=out_channels,
        init_filters=16, blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model, device


# ── Gaussian SWI (logit space) ────────────────────────────
def swi_gaussian(model, volume, device, out_channels=1):
    """Gaussian-weighted SWI in logit space."""
    D, H, W = volume.shape
    ps = PATCH_SIZE
    gauss = GAUSSIAN_MAP
    output = np.zeros((out_channels, D, H, W), dtype=np.float32)
    weights = np.zeros((D, H, W), dtype=np.float32)

    z_pos = _positions(D, ps, STRIDE)
    y_pos = _positions(H, ps, STRIDE)
    x_pos = _positions(W, ps, STRIDE)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for z in z_pos:
            for y in y_pos:
                for x in x_pos:
                    patch = volume[z:z+ps, y:y+ps, x:x+ps]
                    patch_t = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
                    logits = model(patch_t).squeeze(0).cpu().numpy()  # (C, D, H, W)
                    if out_channels == 1:
                        logits = logits.squeeze(0)  # (D, H, W)
                        output[0, z:z+ps, y:y+ps, x:x+ps] += logits * gauss
                    else:
                        for c in range(out_channels):
                            output[c, z:z+ps, y:y+ps, x:x+ps] += logits[c] * gauss
                    weights[z:z+ps, y:y+ps, x:x+ps] += gauss

    for c in range(out_channels):
        output[c] /= weights
    return output  # raw logits, shape (C, D, H, W)


def logits_to_prob(logits, three_class=False):
    """Convert logits to foreground probability."""
    if three_class:
        # logits shape: (3, D, H, W) — softmax, take class 1
        logits_t = torch.from_numpy(logits).unsqueeze(0)  # (1, 3, D, H, W)
        probs = F.softmax(logits_t, dim=1).squeeze(0).numpy()
        return probs[1]  # foreground class probability
    else:
        # logits shape: (1, D, H, W) — sigmoid
        return 1.0 / (1.0 + np.exp(-logits[0]))


# ── TTA (logit space) ────────────────────────────────────
def tta_logit(model, volume, device, out_channels=1):
    """7-fold TTA: 3 flips + 3 rotations + identity, average logits."""
    C = out_channels
    D, H, W = volume.shape
    logits_sum = swi_gaussian(model, volume, device, C)

    # 3 axis flips
    for axis in range(3):
        flipped = np.flip(volume, axis=axis).copy()
        l = swi_gaussian(model, flipped, device, C)
        for c in range(C):
            logits_sum[c] += np.flip(l[c], axis=axis).copy()

    # 3 rotations (90, 180, 270 around z-axis)
    for k in [1, 2, 3]:
        rotated = np.rot90(volume, k=k, axes=(1, 2)).copy()
        l = swi_gaussian(model, rotated, device, C)
        for c in range(C):
            logits_sum[c] += np.rot90(l[c], k=-k, axes=(1, 2)).copy()

    return logits_sum / 7.0


# ── Post-processing ──────────────────────────────────────
def hysteresis_threshold(prob, t_low, t_high):
    strong = prob >= t_high
    weak = prob >= t_low
    struct = generate_binary_structure(3, 3)
    return binary_propagation(strong, structure=struct, mask=weak).astype(np.uint8)


def build_anisotropic_struct(z_radius, xy_radius):
    sz = 2 * z_radius + 1
    sxy = 2 * xy_radius + 1
    struct = np.zeros((sz, sxy, sxy), dtype=bool)
    cy, cx = xy_radius, xy_radius
    for y in range(sxy):
        for x in range(sxy):
            if (y - cy)**2 + (x - cx)**2 <= xy_radius**2:
                struct[:, y, x] = True
    return struct


ANISO_STRUCT = build_anisotropic_struct(CLOSING_Z_RADIUS, CLOSING_XY_RADIUS)


def postprocess(prob):
    binary = hysteresis_threshold(prob, T_LOW, T_HIGH)
    closed = binary_closing(binary, structure=ANISO_STRUCT)
    labeled, n = scipy_label(closed)
    if n > 0:
        sizes = np.bincount(labeled.ravel())
        small = sizes < DUST_MIN_SIZE
        small[0] = False
        closed[small[labeled]] = 0
    return closed.astype(np.uint8)


# ── Scoring ──────────────────────────────────────────────
def score_volume(pred, lbl, downsample=METRIC_DOWNSAMPLE):
    if downsample > 1:
        s = 1.0 / downsample
        pred = scipy_zoom(pred, s, order=0).astype(np.uint8)
        lbl = scipy_zoom(lbl, s, order=0).astype(np.uint8)
    report = compute_leaderboard_score(
        pred, lbl, ignore_label=2, spacing=(1, 1, 1),
        surface_tolerance=2.0, voi_alpha=0.3, combine_weights=(0.3, 0.35, 0.35),
    )
    return report


# ── Main ─────────────────────────────────────────────────
def find_checkpoints(version):
    """Find all periodic + best checkpoints for a version."""
    ckpts = []
    for p in sorted(CKPT_DIR.glob(f"segresnet_{version}_ep*.pth")):
        epoch = int(p.stem.split("_ep")[-1])
        ckpts.append((epoch, p))
    # Also include best if exists
    best = CKPT_DIR / f"best_segresnet_{version}.pth"
    if best.exists():
        ckpts.append(("best", best))
    return ckpts


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--version", type=str, help="Version to sweep (e.g., v12, v13)")
    group.add_argument("--checkpoint", type=str, help="Single checkpoint to evaluate")
    parser.add_argument("--three-class", action="store_true",
                        help="3-class model (v13+)")
    parser.add_argument("--tta", action="store_true",
                        help="Use 7-fold TTA (7x slower, more accurate)")
    parser.add_argument("--n-volumes", type=int, default=10,
                        help="Number of val volumes to evaluate")
    parser.add_argument("--scroll", type=int, default=VAL_SCROLL,
                        help="Scroll ID for validation")
    args = parser.parse_args()

    out_channels = 3 if args.three_class else 1

    # Find checkpoints
    if args.checkpoint:
        ckpts = [("custom", Path(args.checkpoint))]
    else:
        ckpts = find_checkpoints(args.version)
        if not ckpts:
            print(f"No checkpoints found for {args.version} in {CKPT_DIR}")
            return

    print(f"Found {len(ckpts)} checkpoint(s)")
    for epoch, path in ckpts:
        print(f"  ep {epoch}: {path.name}")

    # Get val volumes
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    available = set(int(p.stem) for p in TRAIN_IMG.glob("*.tif")) & \
                set(int(p.stem) for p in TRAIN_LBL.glob("*.tif"))
    val_df = train_df[(train_df.scroll_id == args.scroll) & (train_df.id.isin(available))]
    eval_ids = val_df.id.tolist()[:args.n_volumes]
    print(f"\nEvaluating on {len(eval_ids)} val volumes (scroll {args.scroll})")
    mode = "TTA (7-fold)" if args.tta else "SWI only (no TTA)"
    print(f"Mode: {mode}, {'3-class' if args.three_class else 'binary'}\n")

    # Pre-load all labels and images
    print("Loading volumes...")
    volumes = {}
    labels = {}
    for vid in eval_ids:
        volumes[vid] = tifffile.imread(TRAIN_IMG / f"{vid}.tif")
        labels[vid] = tifffile.imread(TRAIN_LBL / f"{vid}.tif")
    print(f"  Loaded {len(volumes)} volumes\n")

    # Evaluate each checkpoint
    all_results = []
    for epoch, ckpt_path in ckpts:
        print(f"{'='*60}")
        print(f"Checkpoint: ep {epoch} ({ckpt_path.name})")
        print(f"{'='*60}")

        model, device = load_model(ckpt_path, out_channels)
        scores = []
        t_ckpt_start = time.time()

        for i, vid in enumerate(eval_ids):
            t0 = time.time()
            img = volumes[vid]
            lbl = labels[vid]

            if args.tta:
                logits = tta_logit(model, img, device, out_channels)
            else:
                logits = swi_gaussian(model, img, device, out_channels)

            prob = logits_to_prob(logits, args.three_class)
            pred = postprocess(prob)
            report = score_volume(pred, lbl)
            elapsed = time.time() - t0

            scores.append(report.score)
            print(f"  [{i+1}/{len(eval_ids)}] {vid}: {report.score:.4f} "
                  f"(topo={report.topo.toposcore:.3f} sdice={report.surface_dice:.3f} "
                  f"voi={report.voi.voi_score:.3f}) {elapsed:.1f}s")

        mean_score = np.mean(scores)
        ckpt_elapsed = time.time() - t_ckpt_start
        print(f"\n  Mean comp_score: {mean_score:.4f} ({ckpt_elapsed:.0f}s)")

        all_results.append({
            "epoch": epoch,
            "checkpoint": ckpt_path.name,
            "mean_score": mean_score,
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            "scores": scores,
        })

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"CHECKPOINT SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"{'Epoch':>8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}  Checkpoint")
    print(f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*30}")

    # Sort by mean score descending
    all_results.sort(key=lambda r: r["mean_score"], reverse=True)
    for r in all_results:
        marker = " <-- BEST" if r == all_results[0] else ""
        print(f"{str(r['epoch']):>8} {r['mean_score']:>8.4f} {r['std_score']:>8.4f} "
              f"{r['min_score']:>8.4f} {r['max_score']:>8.4f}  {r['checkpoint']}{marker}")

    best = all_results[0]
    print(f"\nBest checkpoint: ep {best['epoch']} ({best['checkpoint']}) "
          f"with mean comp_score = {best['mean_score']:.4f}")

    # Save results
    version = args.version or "custom"
    out_path = ROOT / "logs" / f"checkpoint_sweep_{version}.csv"
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "scores"} for r in all_results])
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
