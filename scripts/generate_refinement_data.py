"""
Generate training data for the refinement (post-processing) model.

Runs the main segmentation model (Gaussian SWI) on all training volumes
and saves the raw probability maps. These become the refinement model's
input, with GT labels as targets.

Output: {ROOT}/data/refinement_data/probmaps/{volume_id}.npy (float16, 320^3)

Usage:
    # With traced model (no MONAI needed):
    python scripts/generate_refinement_data.py \
        --traced kaggle/kaggle_weights_download/best_segresnet_v9_traced.pt --tta

    # With checkpoint (needs MONAI):
    python scripts/generate_refinement_data.py \
        --checkpoint checkpoints/models/best_segresnet_v9.pth --tta

    # With 3-class model (v13):
    python scripts/generate_refinement_data.py \
        --checkpoint checkpoints/models/segresnet_v13_ep15.pth --tta --three-class \
        --output-dir data/refinement_data/probmaps_v13
"""
import argparse
import time
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from pathlib import Path

# ── Config ────────────────────────────────────────────────
ROOT = Path("/workspace/vesuvius-kaggle-competition")
TRAIN_IMG = ROOT / "data" / "train_images"
TRAIN_LBL = ROOT / "data" / "train_labels"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "refinement_data" / "probmaps"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATCH_SIZE = 160
STRIDE = 80


# ── Model loading ─────────────────────────────────────────
def load_model(checkpoint_path=None, traced_path=None, out_channels=1):
    """Load model from traced .pt (preferred) or fastai checkpoint."""
    if traced_path:
        model = torch.jit.load(traced_path, map_location=DEVICE)
        model.eval()
        return model

    from monai.networks.nets import SegResNet
    model = SegResNet(
        spatial_dims=3, in_channels=1, out_channels=out_channels,
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


# ── Gaussian SWI (logit space) ────────────────────────────
def swi_gaussian(model, volume, out_channels=1):
    """Gaussian-weighted sliding window, returns raw logits (C, D, H, W)."""
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
                    patch_t = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0
                    logits = model(patch_t).squeeze(0).cpu().numpy()  # (C, D, H, W) or (D, H, W)
                    if out_channels == 1:
                        if logits.ndim == 4:
                            logits = logits[0]
                        output[0, z:z+ps, y:y+ps, x:x+ps] += logits * gauss
                    else:
                        for c in range(out_channels):
                            output[c, z:z+ps, y:y+ps, x:x+ps] += logits[c] * gauss
                    weights[z:z+ps, y:y+ps, x:x+ps] += gauss
    for c in range(out_channels):
        output[c] /= weights
    return output  # raw logits (C, D, H, W)


def logits_to_prob(logits, three_class=False):
    """Convert logits (C, D, H, W) to foreground probability map (D, H, W)."""
    if three_class:
        logits_t = torch.from_numpy(logits).unsqueeze(0)  # (1, 3, D, H, W)
        probs = F.softmax(logits_t, dim=1).squeeze(0).numpy()  # (3, D, H, W)
        return probs[1]  # class 1 = foreground
    else:
        return 1.0 / (1.0 + np.exp(-logits[0]))


# ── TTA: logit-space averaging ────────────────────────────
def tta_logit(model, volume, out_channels=1, three_class=False):
    """Average logits across 7 augmentations, then convert to probability."""
    logits_sum = swi_gaussian(model, volume, out_channels).astype(np.float64)
    for axis in range(3):
        flipped = np.flip(volume, axis=axis).copy()
        l = swi_gaussian(model, flipped, out_channels)
        for c in range(out_channels):
            logits_sum[c] += np.flip(l[c], axis=axis)
    for k in [1, 2, 3]:
        rotated = np.rot90(volume, k=k, axes=(1, 2)).copy()
        l = swi_gaussian(model, rotated, out_channels)
        for c in range(out_channels):
            logits_sum[c] += np.rot90(l[c], k=-k, axes=(1, 2))
    mean_logits = (logits_sum / 7.0).astype(np.float32)
    return logits_to_prob(mean_logits, three_class)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="Path to fastai model checkpoint (.pth)")
    group.add_argument("--traced", help="Path to traced model (.pt)")
    parser.add_argument("--tta", action="store_true", help="Use 7-fold TTA (7x slower)")
    parser.add_argument("--three-class", action="store_true",
                        help="3-class model (v13): use softmax + extract class-1 prob")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: data/refinement_data/probmaps)")
    args = parser.parse_args()

    out_channels = 3 if args.three_class else 1
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    src = args.traced or args.checkpoint
    print(f"Loading model from {src}")
    print(f"  out_channels={out_channels}, three_class={args.three_class}")
    model = load_model(checkpoint_path=args.checkpoint, traced_path=args.traced,
                       out_channels=out_channels)

    # Get all training volume IDs that have files on disk
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    available_ids = set(int(p.stem) for p in TRAIN_IMG.glob("*.tif"))
    vol_ids = sorted(train_df[train_df.id.isin(available_ids)].id.tolist())

    # Skip already-generated volumes (for resumability)
    remaining = [vid for vid in vol_ids if not (output_dir / f"{vid}.npy").exists()]
    print(f"Total volumes: {len(vol_ids)}, already done: {len(vol_ids) - len(remaining)}, remaining: {len(remaining)}")
    print(f"Mode: {'Gaussian SWI + logit TTA (7-fold)' if args.tta else 'Gaussian SWI only'}")
    print(f"Output: {output_dir}")
    print()

    t_total = time.time()
    for i, vid in enumerate(remaining):
        t0 = time.time()
        img = tifffile.imread(TRAIN_IMG / f"{vid}.tif")

        if args.tta:
            prob = tta_logit(model, img, out_channels=out_channels,
                             three_class=args.three_class)
        else:
            logits = swi_gaussian(model, img, out_channels=out_channels)
            prob = logits_to_prob(logits, three_class=args.three_class)

        # Save as float16 to save space (~65 MB vs 130 MB per volume)
        np.save(output_dir / f"{vid}.npy", prob.astype(np.float16))

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_total
        avg = total_elapsed / (i + 1)
        eta = avg * (len(remaining) - i - 1)
        print(f"  [{i+1}/{len(remaining)}] {vid} — {elapsed:.1f}s "
              f"(avg {avg:.1f}s, ETA {eta/60:.0f}min)")

    total_time = time.time() - t_total
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.npy")) / 1e9
    print(f"\nDone! {len(vol_ids)} volumes in {total_time/60:.1f} min")
    print(f"Total size: {total_size:.1f} GB at {output_dir}")


if __name__ == "__main__":
    main()
