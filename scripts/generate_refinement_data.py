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
"""
import argparse
import time
import numpy as np
import pandas as pd
import tifffile
import torch
from pathlib import Path

# ── Config ────────────────────────────────────────────────
ROOT = Path("/workspace/vesuvius-kaggle-competition")
TRAIN_IMG = ROOT / "data" / "train_images"
TRAIN_LBL = ROOT / "data" / "train_labels"
OUTPUT_DIR = ROOT / "data" / "refinement_data" / "probmaps"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATCH_SIZE = 160
STRIDE = 80


# ── Model loading ─────────────────────────────────────────
def load_model(checkpoint_path=None, traced_path=None):
    """Load model from traced .pt (preferred) or fastai checkpoint."""
    if traced_path:
        model = torch.jit.load(traced_path, map_location=DEVICE)
        model.eval()
        return model

    from monai.networks.nets import SegResNet
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


# ── Gaussian SWI (logit space) ────────────────────────────
def swi_gaussian(model, volume):
    """Gaussian-weighted sliding window, returns raw logits."""
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


# ── TTA: logit-space averaging ────────────────────────────
def tta_logit(model, volume):
    """Average logits across 7 augmentations, then sigmoid."""
    logits_sum = swi_gaussian(model, volume).astype(np.float64)
    for axis in range(3):
        flipped = np.flip(volume, axis=axis).copy()
        l = swi_gaussian(model, flipped)
        logits_sum += np.flip(l, axis=axis)
    for k in [1, 2, 3]:
        rotated = np.rot90(volume, k=k, axes=(1, 2)).copy()
        l = swi_gaussian(model, rotated)
        logits_sum += np.rot90(l, k=-k, axes=(1, 2))
    mean_logits = (logits_sum / 7.0).astype(np.float32)
    return 1.0 / (1.0 + np.exp(-mean_logits))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="Path to fastai model checkpoint (.pth)")
    group.add_argument("--traced", help="Path to traced model (.pt)")
    parser.add_argument("--tta", action="store_true", help="Use 7-fold TTA (7x slower)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    src = args.traced or args.checkpoint
    print(f"Loading model from {src}")
    model = load_model(checkpoint_path=args.checkpoint, traced_path=args.traced)

    # Get all training volume IDs that have files on disk
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    available_ids = set(int(p.stem) for p in TRAIN_IMG.glob("*.tif"))
    vol_ids = sorted(train_df[train_df.id.isin(available_ids)].id.tolist())

    # Skip already-generated volumes (for resumability)
    remaining = [vid for vid in vol_ids if not (OUTPUT_DIR / f"{vid}.npy").exists()]
    print(f"Total volumes: {len(vol_ids)}, already done: {len(vol_ids) - len(remaining)}, remaining: {len(remaining)}")
    print(f"Mode: {'Gaussian SWI + logit TTA (7-fold)' if args.tta else 'Gaussian SWI only'}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    t_total = time.time()
    for i, vid in enumerate(remaining):
        t0 = time.time()
        img = tifffile.imread(TRAIN_IMG / f"{vid}.tif")

        if args.tta:
            prob = tta_logit(model, img)
        else:
            logits = swi_gaussian(model, img)
            prob = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

        # Save as float16 to save space (~65 MB vs 130 MB per volume)
        np.save(OUTPUT_DIR / f"{vid}.npy", prob.astype(np.float16))

        elapsed = time.time() - t0
        total_elapsed = time.time() - t_total
        avg = total_elapsed / (i + 1)
        eta = avg * (len(remaining) - i - 1)
        print(f"  [{i+1}/{len(remaining)}] {vid} — {elapsed:.1f}s "
              f"(avg {avg:.1f}s, ETA {eta/60:.0f}min)")

    total_time = time.time() - t_total
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.npy")) / 1e9
    print(f"\nDone! {len(vol_ids)} volumes in {total_time/60:.1f} min")
    print(f"Total size: {total_size:.1f} GB at {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
