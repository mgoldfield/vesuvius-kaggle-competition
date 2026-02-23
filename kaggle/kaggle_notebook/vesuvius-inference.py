"""
Vesuvius Challenge - Surface Detection: TransUNet Inference
Dual-stream inference with seeded hysteresis post-processing.
Uses pretrained TransUNet SEResNeXt50 (comboloss weights, LB 0.545).
Based on Tony Li's 0.552 approach with public-anchored weak region expansion.
"""
import subprocess
import sys
import os
import time
import zipfile

# ── Install offline dependencies ──────────────────────────
# Protobuf stability
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ["KERAS_BACKEND"] = "jax"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

WHEELS_DIR = "/kaggle/input/vsdetection-packages-offline-installer-only/whls"
print(f"Installing offline wheels from {WHEELS_DIR}...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--no-index",
    "--find-links", WHEELS_DIR,
    "keras-nightly", "tifffile", "imagecodecs", "medicai",
])
print("Wheels installed.")

# ── Protobuf compatibility patch ──────────────────────────
try:
    from google.protobuf import message_factory as _message_factory
    if not hasattr(_message_factory.MessageFactory, "GetPrototype"):
        from google.protobuf.message_factory import GetMessageClass
        def _GetPrototype(self, descriptor):
            return GetMessageClass(descriptor)
        _message_factory.MessageFactory.GetPrototype = _GetPrototype
        print("Patched protobuf: added MessageFactory.GetPrototype")
except Exception as e:
    print("Could not patch protobuf MessageFactory:", e)

# ── Imports ───────────────────────────────────────────────
import numpy as np
import pandas as pd
import tifffile
import scipy.ndimage as ndi
from skimage.morphology import remove_small_objects
import keras
from medicai.transforms import Compose, NormalizeIntensity
from medicai.models import TransUNet
from medicai.utils.inference import SlidingWindowInference

print(f"Keras backend: {keras.config.backend()}, version: {keras.version()}")

# ── Configuration ─────────────────────────────────────────
CFG = dict(
    # Model weights (SWA 70/30 pretrained/margin_dist blend, local val 0.5551)
    weights_path="/kaggle/input/vesuvius-unet3d-weights/swa_70pre_30margin_dist_ep5.weights.h5",

    # SWI overlaps: dual-stream (Tony Li's 0.552 approach)
    overlap_public=0.42,   # public stream (argmax labels)
    overlap_base=0.43,     # private stream (TTA augmentations)
    overlap_hi=0.60,       # private stream (original orientation only)
    OV06_MAIN_ONLY=True,   # Use high-overlap only for identity augmentation

    # TTA: 7-fold (identity + 3 flips + 3 rotations)
    USE_TTA=True,

    # Adaptive TTA timer
    TIME_BUDGET_SEC=9 * 3600,     # 9 hour Kaggle limit
    BUFFER_SEC=15 * 60,           # 15 min safety buffer for ZIP + overhead

    # Binary logit mode: fg12 = logsumexp(L1, L2) - L0
    INK_MODE="fg12",

    # Hysteresis thresholds (T_low=0.40 + close_erode PP, local val 0.5595)
    T_low=0.40,
    T_high=0.90,

    # Close-erode post-processing: closing bridges gaps, erosion thins back
    closing_iters=1,       # face-connected binary closing iterations
    erosion_iters=1,       # face-connected binary erosion iterations
    dust_min_size=100,

    # Warmup (JAX JIT compilation)
    DO_WARMUP=True,
)

# ── Paths ─────────────────────────────────────────────────
ROOT_DIR = "/kaggle/input/vesuvius-challenge-surface-detection"
TEST_DIR = f"{ROOT_DIR}/test_images"
OUTPUT_DIR = "/kaggle/working/submission_masks"
ZIP_PATH = "/kaggle/working/submission.zip"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROI = (160, 160, 160)

test_df = pd.read_csv(f"{ROOT_DIR}/test.csv")
ids = test_df["id"].tolist()
print(f"Test volumes: {len(ids)}")

# ── Data transforms ───────────────────────────────────────
_val_pipeline = Compose([
    NormalizeIntensity(keys=["image"], nonzero=True, channel_wise=False),
])

def val_transformation(image):
    return _val_pipeline({"image": image})["image"]

def load_volume(path):
    vol = tifffile.imread(path).astype(np.float32)
    return vol[None, ..., None]  # (1, D, H, W, 1)

# ── Numerics ──────────────────────────────────────────────
def sigmoid_stable(x):
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

def logsumexp2(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m) + 1e-12)

def binary_logit_from_multiclass_logits(logits_5d, mode="fg12"):
    """Convert 3-class logits (1,D,H,W,3) to binary logit (D,H,W)."""
    x = np.asarray(logits_5d, dtype=np.float32)[0]  # (D,H,W,3)
    L0, L1, L2 = x[..., 0], x[..., 1], x[..., 2]
    if mode == "fg12":
        return (logsumexp2(L1, L2) - L0).astype(np.float32, copy=False)
    elif mode == "class1":
        return (L1 - logsumexp2(L0, L2)).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown INK_MODE={mode}")

# ── Post-processing helpers ───────────────────────────────
def seeded_hysteresis_close_erode(
    prob, pub_fg_bool,
    T_low=0.40, T_high=0.90,
    closing_iters=1, erosion_iters=1, dust_min_size=100,
):
    """Seeded hysteresis + close_erode: closing bridges gaps, erosion thins back."""
    prob = np.asarray(prob, dtype=np.float32)
    strong = prob >= float(T_high)

    # Weak includes private weak OR public foreground (the key insight)
    weak = (prob >= float(T_low)) | pub_fg_bool

    if not strong.any():
        return np.zeros_like(prob, dtype=np.uint8)

    struct_full = ndi.generate_binary_structure(3, 3)  # full connectivity for hysteresis
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_full)

    if not mask.any():
        return np.zeros_like(prob, dtype=np.uint8)

    # Close-erode: face-connected structuring element
    struct_face = ndi.generate_binary_structure(3, 1)  # face-connected (6-neighbor)

    if closing_iters > 0:
        mask = ndi.binary_closing(mask, structure=struct_face, iterations=closing_iters)

    if erosion_iters > 0:
        mask = ndi.binary_erosion(mask, structure=struct_face, iterations=erosion_iters)

    if int(dust_min_size) > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=int(dust_min_size))

    return mask.astype(np.uint8)

# ── TTA augmentations ─────────────────────────────────────
TTA_LEVEL_NAMES = {3: "7-fold", 2: "4-fold (rotations)", 1: "identity-only"}
TTA_LEVEL_VIEWS = {3: 7, 2: 4, 1: 1}

def iter_tta(volume, level=3):
    """Generate TTA augmentations based on level.
    Level 3: identity + 3 flips + 3 rotations = 7 views (full quality)
    Level 2: identity + 3 rotations = 4 views (drop flips for speed)
    Level 1: identity only = 1 view (fastest)
    """
    # Identity (always included)
    yield volume, (lambda y: y)
    if level <= 1:
        return
    # Rotations (levels 2 and 3)
    for k in [1, 2, 3]:
        v = np.rot90(volume, k=k, axes=(2, 3))
        inv = (lambda y, k=k: np.rot90(y, k=-k, axes=(2, 3)))
        yield v, inv
    if level <= 2:
        return
    # Flips (level 3 only)
    for axis in [1, 2, 3]:
        v = np.flip(volume, axis=axis)
        inv = (lambda y, axis=axis: np.flip(y, axis=axis))
        yield v, inv

# ── Model + SWI setup ─────────────────────────────────────
print(f"Loading TransUNet from {CFG['weights_path']}")
model = TransUNet(
    input_shape=(160, 160, 160, 1),
    encoder_name="seresnext50",
    classifier_activation=None,  # raw logits for binary conversion
    num_classes=3,
)
model.load_weights(CFG["weights_path"])
print(f"Model params: {model.count_params() / 1e6:.1f}M")

def build_swi(overlap):
    return SlidingWindowInference(
        model,
        num_classes=3,
        roi_size=ROI,
        sw_batch_size=1,
        mode="gaussian",
        overlap=float(overlap),
    )

swi_public = build_swi(CFG["overlap_public"])  # public stream (argmax)
swi_base = build_swi(CFG["overlap_base"])      # private stream (TTA augs)
swi_hi = build_swi(CFG["overlap_hi"])          # private stream (identity)

# ── Dual-stream prediction ────────────────────────────────
def predict_pub_labels_and_private_prob(volume, tta_level=3):
    """
    Dual-stream inference:
    - Public stream: average multiclass logits -> argmax labels
    - Private stream: average binary logits -> probability
    tta_level: 3=7-fold, 2=4-fold (rotations), 1=identity only
    """
    mode = CFG["INK_MODE"]

    if not CFG["USE_TTA"] or tta_level <= 0:
        l_pub = np.asarray(swi_public(volume))
        pub_labels = l_pub.argmax(-1).astype(np.uint8).squeeze()
        l_prv = np.asarray(swi_hi(volume))
        s = binary_logit_from_multiclass_logits(l_prv, mode=mode)
        prob = sigmoid_stable(s)
        return pub_labels, prob

    logits_sum = None
    s_sum = None
    n = 0

    for t, (v, inv) in enumerate(iter_tta(volume, level=tta_level)):
        # Public stream
        l_pub = np.asarray(swi_public(v))
        l_pub = inv(l_pub)
        logits_sum = l_pub.astype(np.float32) if logits_sum is None else (logits_sum + l_pub.astype(np.float32))

        # Private stream (high overlap for identity only)
        if CFG["OV06_MAIN_ONLY"]:
            swi_use = swi_hi if (t == 0) else swi_base
        else:
            swi_use = swi_hi

        l_prv = np.asarray(swi_use(v))
        l_prv = inv(l_prv)
        s = binary_logit_from_multiclass_logits(l_prv, mode=mode)
        s_sum = s.astype(np.float32) if s_sum is None else (s_sum + s.astype(np.float32))

        n += 1

    mean_logits = logits_sum / float(n)
    pub_labels = mean_logits.argmax(-1).astype(np.uint8).squeeze()

    s_mean = (s_sum / float(n)).astype(np.float32, copy=False)
    prob = sigmoid_stable(s_mean)
    return pub_labels, prob

# ── Warmup (JAX JIT compilation) ──────────────────────────
def warmup(volume):
    print("Warming up JAX (compile once)...")
    _ = np.asarray(swi_public(volume))
    _ = np.asarray(swi_base(volume))
    _ = np.asarray(swi_hi(volume))
    print("Warmup done.")

# ── Main inference loop ───────────────────────────────────
print(f"\nConfig: overlap_public={CFG['overlap_public']}, overlap_base={CFG['overlap_base']}, "
      f"overlap_hi={CFG['overlap_hi']}")
print(f"INK_MODE={CFG['INK_MODE']}, T_low={CFG['T_low']}, T_high={CFG['T_high']}")
print(f"TTA={'ON (7-fold)' if CFG['USE_TTA'] else 'OFF'}, "
      f"OV06_MAIN_ONLY={CFG['OV06_MAIN_ONLY']}")
print(f"PP: close_erode (close={CFG['closing_iters']}, erode={CFG['erosion_iters']}), dust={CFG['dust_min_size']}")
print(f"Time budget: {CFG['TIME_BUDGET_SEC']/3600:.1f}h, "
      f"buffer: {CFG['BUFFER_SEC']/60:.0f}min")

t_global0 = time.perf_counter()
tta_level = 3           # start at full 7-fold TTA
vol_times = []          # track per-volume times for ETA estimation
total_vols = len(ids)

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for i, image_id in enumerate(ids):
        t0 = time.perf_counter()

        tif_path = f"{TEST_DIR}/{image_id}.tif"
        volume = load_volume(tif_path)
        volume = val_transformation(volume)

        if i == 0 and CFG["DO_WARMUP"]:
            warmup(volume)

        pub_labels, prob = predict_pub_labels_and_private_prob(
            volume, tta_level=tta_level
        )

        # Public foreground anchor (weak region expansion)
        pub_fg = (pub_labels != 0)

        # Seeded hysteresis + close_erode PP
        output = seeded_hysteresis_close_erode(
            prob,
            pub_fg_bool=pub_fg,
            T_low=CFG["T_low"],
            T_high=CFG["T_high"],
            closing_iters=CFG["closing_iters"],
            erosion_iters=CFG["erosion_iters"],
            dust_min_size=CFG["dust_min_size"],
        )

        out_path = f"{OUTPUT_DIR}/{image_id}.tif"
        tifffile.imwrite(out_path, output.astype(np.uint8))
        zf.write(out_path, arcname=f"{image_id}.tif")
        os.remove(out_path)

        dt = time.perf_counter() - t0
        elapsed = time.perf_counter() - t_global0
        vol_times.append(dt)
        remaining_vols = total_vols - (i + 1)

        # ── Adaptive TTA timer ──
        status = "OK"
        if remaining_vols > 0:
            # Rolling average of last 5 volumes for ETA (adapts to level changes)
            recent = vol_times[-5:]
            avg_time = sum(recent) / len(recent)
            time_needed = remaining_vols * avg_time
            time_available = CFG["TIME_BUDGET_SEC"] - CFG["BUFFER_SEC"] - elapsed

            if time_needed > time_available and tta_level > 1:
                old_level = tta_level
                if tta_level == 3:
                    tta_level = 2
                    # Re-estimate: 4/7 of current average
                    est_new_avg = avg_time * (4 / 7)
                elif tta_level == 2:
                    tta_level = 1
                    # Re-estimate: 1/4 of current average
                    est_new_avg = avg_time * (1 / 4)
                # Check if reduction is enough, otherwise drop further
                if remaining_vols * est_new_avg > time_available and tta_level > 1:
                    tta_level = 1
                status = f"REDUCED TTA {old_level}->{tta_level}"
                print(f"  [TIMER] {status} ({TTA_LEVEL_NAMES[tta_level]}). "
                      f"Need {time_needed/60:.0f}min, have {time_available/60:.0f}min")

            eta_min = (elapsed + remaining_vols * avg_time) / 60
        else:
            eta_min = elapsed / 60

        print(f"[{i+1}/{total_vols}] id={image_id} | {dt/60:.1f}min | "
              f"TTA={TTA_LEVEL_VIEWS[tta_level]} | "
              f"elapsed {elapsed/3600:.2f}h | "
              f"ETA {eta_min/60:.2f}h | {status}")

print(f"\nSubmission ZIP: {ZIP_PATH}")
total_time = time.perf_counter() - t_global0
print(f"Total time: {total_time/3600:.2f} hours")
print(f"Final TTA level: {tta_level} ({TTA_LEVEL_NAMES[tta_level]})")
print(f"Volumes processed: {total_vols}")
