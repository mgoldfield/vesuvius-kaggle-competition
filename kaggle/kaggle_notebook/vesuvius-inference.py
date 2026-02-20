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
    # Model weights (comboloss, LB 0.545 as single-stream)
    weights_path="/kaggle/input/vesuvius-unet3d-weights/transunet.seresnext50.160px.comboloss.weights.h5",

    # SWI overlaps: dual-stream (Tony Li's 0.552 approach)
    overlap_public=0.42,   # public stream (argmax labels)
    overlap_base=0.43,     # private stream (TTA augmentations)
    overlap_hi=0.60,       # private stream (original orientation only)
    OV06_MAIN_ONLY=True,   # Use high-overlap only for identity augmentation

    # TTA: 7-fold (identity + 3 flips + 3 rotations)
    USE_TTA=True,

    # Binary logit mode: fg12 = logsumexp(L1, L2) - L0
    INK_MODE="fg12",

    # Hysteresis thresholds
    T_low=0.70,
    T_high=0.90,

    # Closing and dust removal
    z_radius=3,
    xy_radius=2,
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
def build_anisotropic_struct(z_radius, xy_radius):
    z, r = int(z_radius), int(xy_radius)
    if z == 0 and r == 0:
        return None
    if z == 0 and r > 0:
        size = 2 * r + 1
        struct = np.zeros((1, size, size), dtype=bool)
        cy = cx = r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[0, cy + dy, cx + dx] = True
        return struct
    if z > 0 and r == 0:
        struct = np.zeros((2 * z + 1, 1, 1), dtype=bool)
        struct[:, 0, 0] = True
        return struct
    depth = 2 * z + 1
    size = 2 * r + 1
    struct = np.zeros((depth, size, size), dtype=bool)
    cz = z
    cy = cx = r
    for dz in range(-z, z + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[cz + dz, cy + dy, cx + dx] = True
    return struct

def seeded_hysteresis_with_topology(
    prob, pub_fg_bool,
    T_low=0.70, T_high=0.90,
    z_radius=3, xy_radius=2, dust_min_size=100,
):
    """Seeded hysteresis with public-stream weak region expansion."""
    prob = np.asarray(prob, dtype=np.float32)
    strong = prob >= float(T_high)

    # Weak includes private weak OR public foreground (the key insight)
    weak = (prob >= float(T_low)) | pub_fg_bool

    if not strong.any():
        return np.zeros_like(prob, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros_like(prob, dtype=np.uint8)

    struct_close = build_anisotropic_struct(z_radius, xy_radius)
    if struct_close is not None:
        mask = ndi.binary_closing(mask, structure=struct_close)

    if int(dust_min_size) > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=int(dust_min_size))

    return mask.astype(np.uint8)

# ── TTA augmentations ─────────────────────────────────────
def iter_tta(volume):
    """Generate 7 TTA augmentations: identity + 3 flips + 3 rotations."""
    yield volume, (lambda y: y)
    for axis in [1, 2, 3]:
        v = np.flip(volume, axis=axis)
        inv = (lambda y, axis=axis: np.flip(y, axis=axis))
        yield v, inv
    for k in [1, 2, 3]:
        v = np.rot90(volume, k=k, axes=(2, 3))
        inv = (lambda y, k=k: np.rot90(y, k=-k, axes=(2, 3)))
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
def predict_pub_labels_and_private_prob(volume):
    """
    Dual-stream inference:
    - Public stream: average multiclass logits -> argmax labels
    - Private stream: average binary logits -> probability
    """
    mode = CFG["INK_MODE"]

    if not CFG["USE_TTA"]:
        l_pub = np.asarray(swi_public(volume))
        pub_labels = l_pub.argmax(-1).astype(np.uint8).squeeze()
        l_prv = np.asarray(swi_hi(volume))
        s = binary_logit_from_multiclass_logits(l_prv, mode=mode)
        prob = sigmoid_stable(s)
        return pub_labels, prob

    logits_sum = None
    s_sum = None
    n = 0

    for t, (v, inv) in enumerate(iter_tta(volume)):
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
print(f"Closing: z={CFG['z_radius']}, xy={CFG['xy_radius']}, dust={CFG['dust_min_size']}")

t_global0 = time.perf_counter()

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for i, image_id in enumerate(ids):
        t0 = time.perf_counter()

        tif_path = f"{TEST_DIR}/{image_id}.tif"
        volume = load_volume(tif_path)
        volume = val_transformation(volume)

        if i == 0 and CFG["DO_WARMUP"]:
            warmup(volume)

        pub_labels, prob = predict_pub_labels_and_private_prob(volume)

        # Public foreground anchor (weak region expansion)
        pub_fg = (pub_labels != 0)

        # Seeded hysteresis + closing + dust removal
        output = seeded_hysteresis_with_topology(
            prob,
            pub_fg_bool=pub_fg,
            T_low=CFG["T_low"],
            T_high=CFG["T_high"],
            z_radius=CFG["z_radius"],
            xy_radius=CFG["xy_radius"],
            dust_min_size=CFG["dust_min_size"],
        )

        out_path = f"{OUTPUT_DIR}/{image_id}.tif"
        tifffile.imwrite(out_path, output.astype(np.uint8))
        zf.write(out_path, arcname=f"{image_id}.tif")
        os.remove(out_path)

        dt = time.perf_counter() - t0
        elapsed = time.perf_counter() - t_global0
        print(f"[{i+1}/{len(ids)}] id={image_id} | {dt/60:.2f} min | "
              f"elapsed {elapsed/3600:.2f} h | positives={int(output.sum())}")

print(f"\nSubmission ZIP: {ZIP_PATH}")
total_time = time.perf_counter() - t_global0
print(f"Total time: {total_time/3600:.2f} hours")
