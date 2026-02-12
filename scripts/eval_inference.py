"""
Evaluate inference pipeline improvements on validation set.

Compares old (uniform SWI + probability TTA) vs new (Gaussian SWI + logit TTA)
on the same val volumes, using the exact competition metric.

Optionally evaluates killer-ant surface splitting post-processing.

Usage:
    python scripts/eval_inference.py --checkpoint checkpoints/models/best_segresnet_v9.pth
    python scripts/eval_inference.py --checkpoint checkpoints/models/best_segresnet_v10.pth --split
    python scripts/eval_inference.py --checkpoint checkpoints/models/best_segresnet_v10.pth --sweep --split

Requires: topometrics, monai, torch, scipy, tifffile, numpy, pandas
Optional: cc3d, dijkstra3d (for --split)
"""
import argparse
import sys
import time
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
from pathlib import Path
from scipy.ndimage import (
    generate_binary_structure, binary_propagation,
    binary_closing, label as scipy_label, zoom as scipy_zoom,
)
from monai.networks.nets import SegResNet
from topometrics.leaderboard import compute_leaderboard_score

# Add killer-ant to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "libs" / "killer-ant"))
try:
    from surface_splitter import split_merged_surfaces_binary
    HAS_SURFACE_SPLITTER = True
except ImportError:
    HAS_SURFACE_SPLITTER = False

# ── Config ────────────────────────────────────────────────
ROOT = Path("/home/mongomatt/Projects/vesuvius")
TRAIN_IMG = ROOT / "train_images"
TRAIN_LBL = ROOT / "train_labels"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATCH_SIZE = 160
STRIDE = 80
T_LOW = 0.40
T_HIGH = 0.85
CLOSING_Z_RADIUS = 2
CLOSING_XY_RADIUS = 1
DUST_MIN_SIZE = 100
METRIC_DOWNSAMPLE = 4
VAL_SCROLL = 26002
N_EVAL = 5  # number of val volumes for comp_score


# ── Model loading ─────────────────────────────────────────
def load_model(checkpoint_path):
    """Load SegResNet from a fastai checkpoint."""
    model = SegResNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        init_filters=16, blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()
    return model


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


# ── OLD: Uniform SWI (probability space) ─────────────────
def swi_uniform(model, volume):
    """Original: uniform counting, returns probabilities."""
    D, H, W = volume.shape
    ps = PATCH_SIZE
    output = np.zeros((D, H, W), dtype=np.float32)
    counts = np.zeros((D, H, W), dtype=np.float32)

    z_pos = _positions(D, ps, STRIDE)
    y_pos = _positions(H, ps, STRIDE)
    x_pos = _positions(W, ps, STRIDE)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for z in z_pos:
            for y in y_pos:
                for x in x_pos:
                    patch = volume[z:z+ps, y:y+ps, x:x+ps]
                    patch_t = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0
                    logits = model(patch_t)
                    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                    output[z:z+ps, y:y+ps, x:x+ps] += prob
                    counts[z:z+ps, y:y+ps, x:x+ps] += 1.0
    return output / counts


# ── NEW: Gaussian SWI (logit space) ──────────────────────
def swi_gaussian(model, volume):
    """New: Gaussian-weighted, returns raw logits."""
    D, H, W = volume.shape
    ps = PATCH_SIZE
    gauss = GAUSSIAN_MAP
    output = np.zeros((D, H, W), dtype=np.float32)
    weights = np.zeros((D, H, W), dtype=np.float32)

    z_pos = _positions(D, ps, STRIDE)
    y_pos = _positions(H, ps, STRIDE)
    x_pos = _positions(W, ps, STRIDE)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for z in z_pos:
            for y in y_pos:
                for x in x_pos:
                    patch = volume[z:z+ps, y:y+ps, x:x+ps]
                    patch_t = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0
                    logits = model(patch_t).squeeze().cpu().numpy()
                    output[z:z+ps, y:y+ps, x:x+ps] += logits * gauss
                    weights[z:z+ps, y:y+ps, x:x+ps] += gauss
    return output / weights  # raw logits


# ── OLD TTA: probability-space averaging ──────────────────
def tta_prob(model, volume, swi_fn):
    """Old: average probabilities across 7 augmentations."""
    probs = []
    probs.append(swi_fn(model, volume))
    for axis in range(3):
        flipped = np.flip(volume, axis=axis).copy()
        p = swi_fn(model, flipped)
        probs.append(np.flip(p, axis=axis).copy())
    for k in [1, 2, 3]:
        rotated = np.rot90(volume, k=k, axes=(1, 2)).copy()
        p = swi_fn(model, rotated)
        probs.append(np.rot90(p, k=-k, axes=(1, 2)).copy())
    return np.mean(probs, axis=0)


# ── NEW TTA: logit-space averaging ────────────────────────
def tta_logit(model, volume, swi_fn):
    """New: average logits across 7 augmentations, then sigmoid."""
    logits_sum = np.zeros_like(volume, dtype=np.float32)
    logits_sum += swi_fn(model, volume)
    for axis in range(3):
        flipped = np.flip(volume, axis=axis).copy()
        l = swi_fn(model, flipped)
        logits_sum += np.flip(l, axis=axis).copy()
    for k in [1, 2, 3]:
        rotated = np.rot90(volume, k=k, axes=(1, 2)).copy()
        l = swi_fn(model, rotated)
        logits_sum += np.rot90(l, k=-k, axes=(1, 2)).copy()
    mean_logits = logits_sum / 7.0
    return 1.0 / (1.0 + np.exp(-mean_logits))


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


def postprocess(probs, t_low, t_high, z_radius, xy_radius, min_size):
    binary = hysteresis_threshold(probs, t_low, t_high)
    struct = build_anisotropic_struct(z_radius, xy_radius)
    closed = binary_closing(binary, structure=struct)
    labeled, n = scipy_label(closed)
    if n > 0:
        sizes = np.bincount(labeled.ravel())
        small = sizes < min_size
        small[0] = False
        closed[small[labeled]] = 0
    return closed.astype(np.uint8)


# ── Scoring ──────────────────────────────────────────────
def score_volume(pred, lbl, downsample=METRIC_DOWNSAMPLE):
    ds = downsample
    if ds > 1:
        pred = scipy_zoom(pred, 1.0/ds, order=0).astype(np.uint8)
        lbl = scipy_zoom(lbl, 1.0/ds, order=0).astype(np.uint8)
    report = compute_leaderboard_score(
        pred, lbl, ignore_label=2, spacing=(1,1,1),
        surface_tolerance=2.0, voi_alpha=0.3, combine_weights=(0.3, 0.35, 0.35),
    )
    return report.score


# ── Main ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-eval", type=int, default=N_EVAL)
    parser.add_argument("--no-tta", action="store_true", help="Skip TTA (faster, just compare SWI methods)")
    parser.add_argument("--sweep", action="store_true", help="Also sweep T_low/T_high thresholds")
    parser.add_argument("--split", action="store_true", help="Also evaluate killer-ant surface splitting")
    args = parser.parse_args()

    if args.split and not HAS_SURFACE_SPLITTER:
        print("ERROR: --split requires cc3d and dijkstra3d. Install with:")
        print("  pip install cc3d dijkstra3d")
        return

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint)

    # Get val volume IDs
    train_df = pd.read_csv(ROOT / "train.csv")
    available = set(int(p.stem) for p in TRAIN_IMG.glob("*.tif")) & \
                set(int(p.stem) for p in TRAIN_LBL.glob("*.tif"))
    train_df = train_df[train_df.id.isin(available)]
    val_ids = train_df[train_df.scroll_id == VAL_SCROLL].id.tolist()[:args.n_eval]
    print(f"Evaluating on {len(val_ids)} validation volumes from scroll {VAL_SCROLL}")

    # ── Define configurations to compare ──
    configs = {}
    if args.no_tta:
        configs["old_uniform"] = lambda vol: swi_uniform(model, vol)
        configs["new_gaussian"] = lambda vol: 1.0 / (1.0 + np.exp(-swi_gaussian(model, vol)))
    else:
        configs["old (uniform SWI + prob TTA)"] = lambda vol: tta_prob(model, vol, swi_uniform)
        configs["new (gaussian SWI + logit TTA)"] = lambda vol: tta_logit(model, vol, swi_gaussian)

    # Build list of all result keys
    result_keys = list(configs.keys())
    if args.split:
        # Add "+split" variants for each config
        for name in list(configs.keys()):
            result_keys.append(name + " +split")
    results = {name: [] for name in result_keys}

    for i, vid in enumerate(val_ids):
        print(f"\n[{i+1}/{len(val_ids)}] Volume {vid}")
        img = tifffile.imread(TRAIN_IMG / f"{vid}.tif")
        lbl = tifffile.imread(TRAIN_LBL / f"{vid}.tif")

        for name, infer_fn in configs.items():
            t0 = time.time()
            prob = infer_fn(img)
            pred = postprocess(prob, T_LOW, T_HIGH, CLOSING_Z_RADIUS, CLOSING_XY_RADIUS, DUST_MIN_SIZE)
            score = score_volume(pred, lbl)
            elapsed = time.time() - t0
            results[name].append(score)
            print(f"  {name}: comp_score={score:.4f} ({elapsed:.1f}s)")

            if args.split:
                t0 = time.time()
                pred_split = split_merged_surfaces_binary(pred, min_component_size=DUST_MIN_SIZE)
                score_split = score_volume(pred_split, lbl)
                split_elapsed = time.time() - t0
                results[name + " +split"].append(score_split)
                print(f"  {name} +split: comp_score={score_split:.4f} ({split_elapsed:.1f}s)")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, scores in results.items():
        mean = np.mean(scores)
        std = np.std(scores)
        print(f"  {name}: {mean:.4f} +/- {std:.4f}")

    # Compare old vs new (base configs only)
    base_names = list(configs.keys())
    if len(base_names) == 2:
        old_mean = np.mean(results[base_names[0]])
        new_mean = np.mean(results[base_names[1]])
        delta = new_mean - old_mean
        print(f"\n  Delta (new - old): {delta:+.4f}")

    # Show split improvement if applicable
    if args.split:
        print(f"\n  Surface splitting impact:")
        for name in base_names:
            base_mean = np.mean(results[name])
            split_mean = np.mean(results[name + " +split"])
            delta = split_mean - base_mean
            print(f"    {name}: {delta:+.4f} ({base_mean:.4f} -> {split_mean:.4f})")

    # ── Optional threshold sweep ──
    if args.sweep:
        print("\n" + "=" * 60)
        print("THRESHOLD SWEEP (using new inference pipeline)")
        print("=" * 60)
        # Pre-compute prob maps with new pipeline
        prob_maps = {}
        for vid in val_ids:
            img = tifffile.imread(TRAIN_IMG / f"{vid}.tif")
            best_fn = list(configs.values())[-1]  # use new pipeline
            prob_maps[vid] = best_fn(img)

        t_lows = [0.30, 0.35, 0.40, 0.45, 0.50]
        t_highs = [0.80, 0.85, 0.90]
        best_score, best_tl, best_th = 0, T_LOW, T_HIGH

        if args.split:
            print(f"{'T_low':>6} {'T_high':>6} {'comp_score':>12} {'+ split':>12}")
            print("-" * 42)
        else:
            print(f"{'T_low':>6} {'T_high':>6} {'comp_score':>12}")
            print("-" * 28)

        best_split_score, best_split_tl, best_split_th = 0, T_LOW, T_HIGH
        for th in t_highs:
            for tl in t_lows:
                scores = []
                split_scores = []
                for vid in val_ids:
                    lbl = tifffile.imread(TRAIN_LBL / f"{vid}.tif")
                    pred = postprocess(prob_maps[vid], tl, th,
                                       CLOSING_Z_RADIUS, CLOSING_XY_RADIUS, DUST_MIN_SIZE)
                    scores.append(score_volume(pred, lbl))
                    if args.split:
                        pred_split = split_merged_surfaces_binary(pred, min_component_size=DUST_MIN_SIZE)
                        split_scores.append(score_volume(pred_split, lbl))
                mean = np.mean(scores)
                if args.split:
                    split_mean = np.mean(split_scores)
                    print(f"{tl:>6.2f} {th:>6.2f} {mean:>12.4f} {split_mean:>12.4f}")
                    if split_mean > best_split_score:
                        best_split_score, best_split_tl, best_split_th = split_mean, tl, th
                else:
                    print(f"{tl:>6.2f} {th:>6.2f} {mean:>12.4f}")
                if mean > best_score:
                    best_score, best_tl, best_th = mean, tl, th

        print(f"\nBest (no split): T_low={best_tl}, T_high={best_th}, comp_score={best_score:.4f}")
        if args.split:
            print(f"Best (+ split):  T_low={best_split_tl}, T_high={best_split_th}, comp_score={best_split_score:.4f}")


if __name__ == "__main__":
    main()
