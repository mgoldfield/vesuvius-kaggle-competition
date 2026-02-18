#!/usr/bin/env python3
"""
Benchmark TransUNet performance: per-iteration timing and full-volume SWI.

Measures forward pass, backward pass, and full SWI inference time.
Reports estimates for epoch and full training duration.

Usage:
    python scripts/benchmark_transunet.py
    python scripts/benchmark_transunet.py --dry-run  # quick sanity check
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
import torch
import tifffile
from pathlib import Path

ROOT = Path("/workspace/vesuvius-kaggle-competition")
DEFAULT_WEIGHTS = ROOT / "pretrained_weights" / "transunet" / "transunet.seresnext50.160px.comboloss.weights.h5"


def benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    from medicai.models import TransUNet
    from medicai.transforms import Compose, NormalizeIntensity
    from medicai.utils.inference import SlidingWindowInference

    print("=" * 60)
    print("TransUNet Benchmark")
    print("=" * 60)

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("WARNING: No CUDA GPU detected")

    # Load model
    print("\nLoading model...")
    model = TransUNet(
        input_shape=(160, 160, 160, 1),
        encoder_name='seresnext50',
        classifier_activation=None,
        num_classes=3,
    )
    model.load_weights(str(DEFAULT_WEIGHTS))
    print(f"Model params: {model.count_params() / 1e6:.1f}M")

    # ── Benchmark 1: Single patch forward/backward ──
    print("\n--- Single Patch (160^3) ---")

    patch = np.random.randn(1, 160, 160, 160, 1).astype(np.float32)

    # Warmup
    print("Warmup (2 forward passes)...")
    for _ in range(2):
        with torch.no_grad():
            _ = model(patch)
    torch.cuda.synchronize()

    # Forward pass timing
    n_fwd = 5 if not args.dry_run else 2
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    fwd_times = []
    for _ in range(n_fwd):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(patch)
        torch.cuda.synchronize()
        fwd_times.append(time.perf_counter() - t0)

    fwd_mean = np.mean(fwd_times)
    fwd_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"Forward:  {fwd_mean:.3f}s (peak {fwd_peak:.2f} GB)")

    # Backward pass timing
    # Need to enable gradients via the underlying torch model
    torch_model = model  # Keras wraps torch
    n_bwd = 3 if not args.dry_run else 1
    bwd_times = []

    # Access underlying torch parameters
    import keras
    patch_t = torch.tensor(patch, dtype=torch.float32, device='cuda', requires_grad=False)

    for _ in range(n_bwd):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        # Enable training mode
        out = model(patch)
        # Convert output to torch tensor and compute dummy loss
        if isinstance(out, torch.Tensor):
            loss = out.sum()
        else:
            out_t = torch.tensor(np.asarray(out), dtype=torch.float32, device='cuda',
                                 requires_grad=True)
            loss = out_t.sum()
        loss.backward()
        torch.cuda.synchronize()
        bwd_times.append(time.perf_counter() - t0)

    bwd_mean = np.mean(bwd_times)
    bwd_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"Fwd+Bwd:  {bwd_mean:.3f}s (peak {bwd_peak:.2f} GB)")

    iter_time = bwd_mean
    print(f"Per-iteration estimate: {iter_time:.3f}s")

    # ── Benchmark 2: Full volume SWI ──
    print("\n--- Full Volume SWI (320^3) ---")

    # Load a real volume
    vol_path = ROOT / "data" / "train_images" / "26894125.tif"
    if vol_path.exists():
        print(f"Loading {vol_path.name}...")
        vol = tifffile.imread(vol_path).astype(np.float32)
        vol_5d = vol[None, ..., None]

        normalize = Compose([
            NormalizeIntensity(keys=["image"], nonzero=True, channel_wise=False),
        ])
        vol_5d = normalize({"image": vol_5d})["image"]

        for overlap in [0.25, 0.50]:
            swi = SlidingWindowInference(
                model,
                num_classes=3,
                roi_size=(160, 160, 160),
                sw_batch_size=1,
                mode='gaussian',
                overlap=overlap,
            )

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = np.asarray(swi(vol_5d))
            torch.cuda.synchronize()
            swi_time = time.perf_counter() - t0

            # Count patches
            D, H, W = vol.shape
            stride = int(160 * (1 - overlap))
            n_patches = (len(range(0, D - 160, stride)) + 1) * \
                        (len(range(0, H - 160, stride)) + 1) * \
                        (len(range(0, W - 160, stride)) + 1)

            print(f"  overlap={overlap}: {swi_time:.1f}s ({n_patches} patches, "
                  f"{swi_time/n_patches:.3f}s/patch)")

        # SWI with 7-fold TTA estimate
        print(f"\n  Estimated with 7-fold TTA at overlap=0.50: ~{swi_time * 7:.0f}s "
              f"({swi_time * 7 / 60:.1f} min)")
    else:
        print(f"  Volume not found at {vol_path}, skipping SWI benchmark")

    # ── Training time estimates ──
    print("\n--- Training Time Estimates ---")
    n_train = 780  # number of training patches (from competitors)

    epoch_time_min = (n_train * iter_time) / 60
    print(f"Per-epoch ({n_train} patches × {iter_time:.3f}s): {epoch_time_min:.1f} min")
    print(f"  5 epochs (fine-tune):  {epoch_time_min * 5 / 60:.1f} hours")
    print(f"  10 epochs (moderate):  {epoch_time_min * 10 / 60:.1f} hours")
    print(f"  25 epochs (full):      {epoch_time_min * 25 / 60:.1f} hours")

    # With gradient accumulation (doesn't change wall time, just optimizer steps)
    print(f"\nNote: gradient_accumulation=4 doesn't change wall time,")
    print(f"just reduces optimizer steps from {n_train} to {n_train // 4} per epoch.")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Forward pass:      {fwd_mean:.3f}s")
    print(f"Forward+backward:  {bwd_mean:.3f}s")
    print(f"Peak VRAM (train): {bwd_peak:.2f} GB")
    print(f"Est. epoch time:   {epoch_time_min:.1f} min")
    print(f"Est. 10-epoch:     {epoch_time_min * 10 / 60:.1f} hours")


if __name__ == '__main__':
    benchmark()
