"""
Vesuvius Challenge - Surface Detection: Inference Notebook
SegResNet sliding window inference on test volumes.
Inference v3: Gaussian SWI + logit TTA + hysteresis + surface splitting + 160^3.
Model exported via torch.jit.trace (no MONAI dependency needed).
"""
import subprocess
import sys
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

# ── Install surface splitting dependencies from bundled wheels ──
# Wheels may be at wheels/ (direct upload) or wheels/wheels/ (--dir-mode zip nesting)
_WEIGHTS_BASE = Path("/kaggle/input/vesuvius-unet3d-weights")
_WHEELS_CANDIDATES = [
    _WEIGHTS_BASE / "wheels",
    _WEIGHTS_BASE / "wheels" / "wheels",
]
_WHEELS_DIR = None
for _candidate in _WHEELS_CANDIDATES:
    if _candidate.exists() and any(_candidate.glob("*.whl")):
        _WHEELS_DIR = _candidate
        break

if _WHEELS_DIR is None:
    print("ERROR: No wheels directory with .whl files found!")
    print(f"  Searched: {[str(c) for c in _WHEELS_CANDIDATES]}")
    print(f"  Contents of {_WEIGHTS_BASE}:")
    for p in sorted(_WEIGHTS_BASE.rglob("*"))[:30]:
        print(f"    {p}")
    raise FileNotFoundError(f"Wheels not found in any of: {_WHEELS_CANDIDATES}")

print(f"Found wheels at: {_WHEELS_DIR}")
print(f"  Contents: {[p.name for p in _WHEELS_DIR.glob('*.whl')]}")
try:
    import cc3d
    import dijkstra3d
    print("cc3d and dijkstra3d already available")
except ImportError:
    print("Installing cc3d and dijkstra3d from bundled wheels...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--no-index", "--find-links", str(_WHEELS_DIR),
        "connected-components-3d", "dijkstra3d",
    ])
    import cc3d
    import dijkstra3d
    print(f"Installed cc3d and dijkstra3d")
HAS_SURFACE_SPLITTER = True
import scipy.ndimage as ndi

# ── Paths ──────────────────────────────────────────────────
WEIGHTS_DIR = Path("/kaggle/input/vesuvius-unet3d-weights")
DATA_DIR = Path("/kaggle/input/vesuvius-challenge-surface-detection")
TEST_IMG = DATA_DIR / "test_images"
SUBMISSION_DIR = Path("/kaggle/working/submission")
SUBMISSION_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 160
STRIDE = 80
T_LOW = 0.35
T_HIGH_FIXED = 0.75  # fallback if adaptive fails
USE_ADAPTIVE_THRESHOLD = True  # per-volume adaptive T_HIGH
CLOSING_Z_RADIUS = 2
CLOSING_XY_RADIUS = 1
DUST_MIN_SIZE = 100
USE_TTA = True
USE_SURFACE_SPLIT = False  # disabled — never evaluated, too slow (~80 min/vol on P100)

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Patch size: {PATCH_SIZE}^3, Stride: {STRIDE}")
print(f"Hysteresis: T_low={T_LOW}, T_high={'adaptive (p95>0.3)' if USE_ADAPTIVE_THRESHOLD else T_HIGH_FIXED}")
print(f"TTA: {'ON (7-fold)' if USE_TTA else 'OFF'}")
print(f"Surface splitting: {'ON' if USE_SURFACE_SPLIT and HAS_SURFACE_SPLITTER else 'OFF'}")

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


# ── Gaussian Importance Map ───────────────────────────────
def _build_gaussian_map(patch_size, sigma_scale=0.125):
    """
    Build a 3D Gaussian importance map for SWI blending.
    Center voxels get weight ~1.0, edges taper to near-zero.
    sigma_scale: fraction of patch_size for Gaussian sigma (0.125 = MONAI default).
    """
    sigma = patch_size * sigma_scale
    ax = np.arange(patch_size, dtype=np.float32)
    center = (patch_size - 1) / 2.0
    gauss_1d = np.exp(-0.5 * ((ax - center) / sigma) ** 2)
    gauss_3d = gauss_1d[:, None, None] * gauss_1d[None, :, None] * gauss_1d[None, None, :]
    # Normalize so max is 1.0 (avoids numerical issues)
    gauss_3d /= gauss_3d.max()
    # Clamp to small positive value to avoid division by zero
    gauss_3d = np.clip(gauss_3d, 1e-6, None)
    return gauss_3d


# Pre-build the Gaussian map (reused for all patches)
GAUSSIAN_MAP = _build_gaussian_map(PATCH_SIZE)
print(f"Gaussian importance map: min={GAUSSIAN_MAP.min():.4f}, max={GAUSSIAN_MAP.max():.4f}")


# ── Sliding Window Inference (Gaussian-weighted, returns logits) ──
def sliding_window_inference(model, volume, patch_size=160, stride=80, device="cuda"):
    """
    Run inference on a full 320^3 volume using overlapping patches.
    Uses Gaussian importance weighting for smooth blending (center > edges).
    Returns raw logits (pre-sigmoid) for logit-space TTA averaging.
    """
    D, H, W = volume.shape
    ps = patch_size
    gauss = GAUSSIAN_MAP

    output = np.zeros((D, H, W), dtype=np.float32)
    weights = np.zeros((D, H, W), dtype=np.float32)

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
                    logits = model(patch_t).squeeze().cpu().numpy()
                    output[z:z+ps, y:y+ps, x:x+ps] += logits * gauss
                    weights[z:z+ps, y:y+ps, x:x+ps] += gauss

    output /= weights
    return output  # raw logits, not probabilities


# ── TTA: 7-fold test-time augmentation (logit-space) ──────
def sliding_window_inference_tta(model, volume, patch_size=160, stride=80, device="cuda"):
    """
    7-fold TTA: original + 3 axis flips + 3 HW-plane rotations (90/180/270).
    Averages in logit space (before sigmoid) for more principled aggregation.
    Returns probability map (sigmoid applied after averaging logits).
    """
    logits_sum = np.zeros_like(volume, dtype=np.float32)

    # 1. Original
    logits_sum += sliding_window_inference(model, volume, patch_size, stride, device)

    # 2-4. Axis flips (z, y, x)
    for axis in range(3):
        flipped = np.flip(volume, axis=axis).copy()
        logits = sliding_window_inference(model, flipped, patch_size, stride, device)
        logits_sum += np.flip(logits, axis=axis).copy()

    # 5-7. HW-plane (axes 1,2) rotations: 90, 180, 270 degrees
    for k in [1, 2, 3]:
        rotated = np.rot90(volume, k=k, axes=(1, 2)).copy()
        logits = sliding_window_inference(model, rotated, patch_size, stride, device)
        logits_sum += np.rot90(logits, k=-k, axes=(1, 2)).copy()

    # Average logits, then apply sigmoid
    mean_logits = logits_sum / 7.0
    return 1.0 / (1.0 + np.exp(-mean_logits))  # sigmoid


# ── Adaptive T_HIGH threshold ─────────────────────────────
def adaptive_t_high(prob, min_prob=0.3, percentile=95):
    """
    Per-volume adaptive T_HIGH = 95th percentile of probabilities above min_prob.
    Prevents catastrophic failure when a volume's max prob is below a fixed threshold.
    Clamped to [0.50, 0.90] for safety.
    """
    high_probs = prob[prob > min_prob].astype(np.float32)
    if len(high_probs) > 0:
        t_high = float(np.percentile(high_probs, percentile))
        return np.clip(t_high, 0.50, 0.90)
    return 0.50


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


# ── Surface Splitting (Killer Ant algorithm) ─────────────────
# Adapted from hengck23's "Marching Ants" post-processing.
# Detects merged papyrus sheets via raycasting and splits them
# using Dijkstra3D shortest paths. Directly improves TopoScore + VOI.

SPLIT_RAYCAST_IOU = 0.1
SPLIT_RANGE = 20
SPLIT_MIN_POINTS = 8
SPLIT_MAX_TRIAL = 100
SPLIT_REMOVE_THR = 0.9
SPLIT_MAX_DEPTH = 50


def _split_range_intervals(d, min_size=64):
    k = max(1, d // min_size)
    edges = np.linspace(0, d, k + 1, dtype=int)
    edges[-1] = d
    return [(edges[i], edges[i + 1]) for i in range(k)]


def _raycast_iou(p1, p2, h, w):
    """Check if two 2D point sets (Nx2, yx format) overlap in YX projection."""
    by1 = np.bincount(p1[:, 0], minlength=h)
    by2 = np.bincount(p2[:, 0], minlength=h)
    bx1 = np.bincount(p1[:, 1], minlength=w)
    bx2 = np.bincount(p2[:, 1], minlength=w)
    combined1 = np.concatenate([by1, bx1]) != 0
    combined2 = np.concatenate([by2, bx2]) != 0
    inter = (combined1 & combined2).sum()
    union = (combined1 | combined2).sum()
    return inter / (union + 1e-6)


def _find_multi_surface_seeds(component):
    """Detect if a 3D component contains multiple overlapping surfaces."""
    d, h, w = component.shape
    for z1, z2 in _split_range_intervals(d, min_size=SPLIT_RANGE):
        z = (z1 + z2) // 2
        ccz = cc3d.connected_components(component[z])
        points = []
        for i in range(1, ccz.max() + 1):
            p = np.stack(np.where(ccz == i)).T
            if len(p) >= SPLIT_MIN_POINTS:
                points.append(p)
        if len(points) <= 1:
            continue
        for i1 in range(len(points)):
            for i2 in range(i1 + 1, len(points)):
                if _raycast_iou(points[i1], points[i2], h, w) >= SPLIT_RAYCAST_IOU:
                    z_col = np.full((len(points[i1]), 1), z, dtype=points[i1].dtype)
                    p1_zyx = np.concatenate([z_col, points[i1]], -1)
                    z_col2 = np.full((len(points[i2]), 1), z, dtype=points[i2].dtype)
                    p2_zyx = np.concatenate([z_col2, points[i2]], -1)
                    return True, p1_zyx, p2_zyx
    return False, None, None


def _split_component(component, p1_zyx, p2_zyx):
    """Split a component into two via Dijkstra path removal."""
    problem = component.copy()
    for trial in range(SPLIT_MAX_TRIAL):
        paths = []
        k1 = np.arange(len(p1_zyx))
        np.random.shuffle(k1)
        for (sz, sy, sx) in p1_zyx[k1[:8]]:
            parent = dijkstra3d.parental_field(
                np.where(problem, 1.0, 1e6).astype(np.float32),
                source=(sz, sy, sx), connectivity=26)
            k2 = np.arange(len(p2_zyx))
            np.random.shuffle(k2)
            for (ez, ey, ex) in p2_zyx[k2[:8]]:
                paths.append(dijkstra3d.path_from_parents(parent, (ez, ey, ex)))
        path_flat = np.concatenate(paths)
        if np.any(~problem[path_flat[:, 0], path_flat[:, 1], path_flat[:, 2]]):
            return True, cc3d.connected_components(problem)
        uniq, cnt = np.unique(path_flat, axis=0, return_counts=True)
        order = np.argsort(-cnt)
        uniq, cnt = uniq[order], cnt[order]
        threshold = SPLIT_REMOVE_THR * cnt.max()
        u = uniq[cnt >= threshold]
        larger = np.zeros_like(problem, dtype=bool)
        larger[u[:, 0], u[:, 1], u[:, 2]] = True
        larger = ndi.binary_dilation(larger, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
        problem[larger] = False
    return False, problem


def _recursive_split(component, result, depth=0):
    """Recursively split until each piece is a single surface."""
    if depth >= SPLIT_MAX_DEPTH:
        result.append(component)
        return
    is_multi, p1, p2 = _find_multi_surface_seeds(component)
    if not is_multi:
        result.append(component)
        return
    success, solved = _split_component(component, p1, p2)
    if not success:
        result.append(component)
        return
    _recursive_split(solved == 1, result, depth + 1)
    _recursive_split(solved == 2, result, depth + 1)


def split_merged_surfaces(binary_mask, min_size=100):
    """Split merged papyrus sheets in a binary mask. Returns binary uint8."""
    labeled = cc3d.connected_components(binary_mask.astype(np.uint8))
    n = labeled.max()
    if n == 0:
        return binary_mask
    surfaces = []
    for cid in range(1, n + 1):
        comp = (labeled == cid)
        if comp.sum() < min_size:
            surfaces.append(comp)
            continue
        result = []
        _recursive_split(comp, result)
        surfaces.extend(result)
    output = np.zeros_like(binary_mask, dtype=np.uint8)
    for s in surfaces:
        output[s > 0] = 1
    return output


# ── Run on Test Volumes ────────────────────────────────────
test_df = pd.read_csv(DATA_DIR / "test.csv")
print(f"\nTest volumes: {len(test_df)}")

def sliding_window_inference_prob(model, volume, patch_size=160, stride=80, device="cuda"):
    """Non-TTA path: sliding window returning probabilities (sigmoid applied)."""
    logits = sliding_window_inference(model, volume, patch_size, stride, device)
    return 1.0 / (1.0 + np.exp(-logits))

infer_fn = sliding_window_inference_tta if USE_TTA else sliding_window_inference_prob

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
    t_high = adaptive_t_high(prob) if USE_ADAPTIVE_THRESHOLD else T_HIGH_FIXED
    pred = postprocess(prob, t_low=T_LOW, t_high=t_high,
                       z_radius=CLOSING_Z_RADIUS, xy_radius=CLOSING_XY_RADIUS,
                       min_size=DUST_MIN_SIZE)

    if USE_SURFACE_SPLIT and HAS_SURFACE_SPLITTER:
        pred = split_merged_surfaces(pred, min_size=DUST_MIN_SIZE)

    out_path = SUBMISSION_DIR / f"{vol_id}.tif"
    write_tiff_volume(out_path, pred)
    fg = pred.sum()
    elapsed = time.time() - t0
    print(f"T_high={t_high:.3f} fg={fg}/{pred.size} ({fg/pred.size*100:.1f}%) [{elapsed:.1f}s]")

print("\nInference complete.")


# ── Create submission.zip ──────────────────────────────────
zip_path = Path("/kaggle/working/submission.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for tif in sorted(SUBMISSION_DIR.glob("*.tif")):
        zf.write(tif, tif.name)

print(f"\nSubmission: {zip_path}")
print(f"Size: {zip_path.stat().st_size / 1e6:.1f} MB")
print(f"Files: {len(list(SUBMISSION_DIR.glob('*.tif')))}")
