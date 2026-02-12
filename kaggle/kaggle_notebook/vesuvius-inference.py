"""
Vesuvius Challenge - Surface Detection: Inference Notebook
SegResNet sliding window inference on test volumes.
Run 9: Lower LR (1e-5) + FG-biased sampling + TTA + hysteresis + 160^3.
Model exported via torch.jit.trace (no MONAI dependency needed).
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import zipfile
import time
from scipy.ndimage import (
    generate_binary_structure, binary_propagation,
    binary_closing, label as scipy_label,
)

# ── Paths ──────────────────────────────────────────────────
WEIGHTS_DIR = Path("/kaggle/input/vesuvius-unet3d-weights")
DATA_DIR = Path("/kaggle/input/vesuvius-challenge-surface-detection")
TEST_IMG = DATA_DIR / "test_images"
SUBMISSION_DIR = Path("/kaggle/working/submission")
SUBMISSION_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 160
STRIDE = 80
T_LOW = 0.40
T_HIGH = 0.85
CLOSING_Z_RADIUS = 2
CLOSING_XY_RADIUS = 1
DUST_MIN_SIZE = 100
USE_TTA = True

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Patch size: {PATCH_SIZE}^3, Stride: {STRIDE}")
print(f"Hysteresis: T_low={T_LOW}, T_high={T_HIGH}")
print(f"TTA: {'ON (7-fold)' if USE_TTA else 'OFF'}")

# ── Load Model (traced — no MONAI needed) ──────────────────
weights_path = WEIGHTS_DIR / "best_segresnet_v9_traced.pt"
model = torch.jit.load(weights_path, map_location=DEVICE)
model.eval()
print(f"Model loaded from {weights_path}")


# ── TIFF I/O using Pillow (handles LZW without imagecodecs) ─
def read_tiff_volume(path):
    """Read a multi-page TIFF as a 3D numpy array using Pillow."""
    img = Image.open(path)
    slices = []
    for i in range(img.n_frames):
        img.seek(i)
        slices.append(np.array(img))
    return np.stack(slices)


def write_tiff_volume(path, volume):
    """Write a 3D numpy array as a multi-page TIFF using Pillow."""
    frames = [Image.fromarray(volume[i]) for i in range(volume.shape[0])]
    frames[0].save(path, save_all=True, append_images=frames[1:])


# ── Sliding Window Inference ───────────────────────────────
def sliding_window_inference(model, volume, patch_size=160, stride=80, device="cuda"):
    """
    Run inference on a full 320^3 volume using overlapping patches.
    Overlapping regions are averaged for smoother predictions.
    """
    D, H, W = volume.shape
    ps = patch_size

    output = np.zeros((D, H, W), dtype=np.float32)
    counts = np.zeros((D, H, W), dtype=np.float32)

    def _positions(length, ps, stride):
        pos = list(range(0, length - ps, stride))
        if not pos or pos[-1] + ps < length:
            pos.append(length - ps)
        return pos

    z_pos = _positions(D, ps, stride)
    y_pos = _positions(H, ps, stride)
    x_pos = _positions(W, ps, stride)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for z in z_pos:
            for y in y_pos:
                for x in x_pos:
                    patch = volume[z:z+ps, y:y+ps, x:x+ps]
                    patch_t = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
                    logits = model(patch_t)
                    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                    output[z:z+ps, y:y+ps, x:x+ps] += prob
                    counts[z:z+ps, y:y+ps, x:x+ps] += 1.0

    output /= counts
    return output


# ── TTA: 7-fold test-time augmentation ─────────────────────
def sliding_window_inference_tta(model, volume, patch_size=160, stride=80, device="cuda"):
    """
    7-fold TTA: original + 3 axis flips + 3 HW-plane rotations (90/180/270).
    """
    probs = []

    # 1. Original
    probs.append(sliding_window_inference(model, volume, patch_size, stride, device))

    # 2-4. Axis flips (z, y, x)
    for axis in range(3):
        flipped = np.flip(volume, axis=axis).copy()
        prob = sliding_window_inference(model, flipped, patch_size, stride, device)
        probs.append(np.flip(prob, axis=axis).copy())

    # 5-7. HW-plane (axes 1,2) rotations: 90, 180, 270 degrees
    for k in [1, 2, 3]:
        rotated = np.rot90(volume, k=k, axes=(1, 2)).copy()
        prob = sliding_window_inference(model, rotated, patch_size, stride, device)
        probs.append(np.rot90(prob, k=-k, axes=(1, 2)).copy())

    return np.mean(probs, axis=0)


# ── Hysteresis thresholding ────────────────────────────────
def hysteresis_threshold(prob, t_low=0.35, t_high=0.85):
    """Dual-threshold seed-and-propagate with 26-connectivity."""
    strong = prob >= t_high
    weak = prob >= t_low
    struct = generate_binary_structure(3, 3)  # 26-connectivity
    return binary_propagation(strong, structure=struct, mask=weak).astype(np.uint8)


# ── Anisotropic closing ───────────────────────────────────
def build_anisotropic_struct(z_radius=2, xy_radius=1):
    """Z-heavy structuring element: disk in XY, extended in Z."""
    sz = 2 * z_radius + 1
    sxy = 2 * xy_radius + 1
    struct = np.zeros((sz, sxy, sxy), dtype=bool)
    cy, cx = xy_radius, xy_radius
    for y in range(sxy):
        for x in range(sxy):
            if (y - cy)**2 + (x - cx)**2 <= xy_radius**2:
                struct[:, y, x] = True
    return struct


# ── Post-processing pipeline ──────────────────────────────
def postprocess(probs, t_low=0.35, t_high=0.85, z_radius=2, xy_radius=1, min_size=100):
    """Hysteresis thresholding + anisotropic closing + dust removal."""
    # 1. Hysteresis thresholding
    binary = hysteresis_threshold(probs, t_low, t_high)

    # 2. Anisotropic closing
    struct = build_anisotropic_struct(z_radius, xy_radius)
    closed = binary_closing(binary, structure=struct)

    # 3. Dust removal
    labeled, n_components = scipy_label(closed)
    if n_components > 0:
        component_sizes = np.bincount(labeled.ravel())
        small_mask = component_sizes < min_size
        small_mask[0] = False
        closed[small_mask[labeled]] = 0

    return closed.astype(np.uint8)


# ── Run on Test Volumes ────────────────────────────────────
test_df = pd.read_csv(DATA_DIR / "test.csv")
print(f"\nTest volumes: {len(test_df)}")

infer_fn = sliding_window_inference_tta if USE_TTA else sliding_window_inference

for i, row in test_df.iterrows():
    vol_id = row.id
    img_path = TEST_IMG / f"{vol_id}.tif"
    if not img_path.exists():
        print(f"  [{i+1}/{len(test_df)}] Skipping {vol_id}: not found")
        continue

    print(f"  [{i+1}/{len(test_df)}] Predicting {vol_id}...", end=" ", flush=True)
    t0 = time.time()
    img = read_tiff_volume(img_path)
    prob = infer_fn(model, img, patch_size=PATCH_SIZE, stride=STRIDE, device=DEVICE)
    pred = postprocess(prob, t_low=T_LOW, t_high=T_HIGH,
                       z_radius=CLOSING_Z_RADIUS, xy_radius=CLOSING_XY_RADIUS,
                       min_size=DUST_MIN_SIZE)

    out_path = SUBMISSION_DIR / f"{vol_id}.tif"
    write_tiff_volume(out_path, pred)
    fg = pred.sum()
    elapsed = time.time() - t0
    print(f"fg={fg}/{pred.size} ({fg/pred.size*100:.1f}%) [{elapsed:.1f}s]")

print("\nInference complete.")


# ── Create submission.zip ──────────────────────────────────
zip_path = Path("/kaggle/working/submission.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for tif in sorted(SUBMISSION_DIR.glob("*.tif")):
        zf.write(tif, tif.name)

print(f"\nSubmission: {zip_path}")
print(f"Size: {zip_path.stat().st_size / 1e6:.1f} MB")
print(f"Files: {len(list(SUBMISSION_DIR.glob('*.tif')))}")
