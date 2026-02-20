#!/usr/bin/env python3
"""
Stochastic Weight Averaging (SWA) for TransUNet checkpoints.

Averages weights from multiple checkpoints to find a better solution in the
loss basin. Can also interpolate between pretrained and fine-tuned weights
(model soup / weight interpolation).

Usage:
    # Average all baseline_v3 checkpoints
    python scripts/swa_average.py \
        --checkpoints checkpoints/transunet_baseline_v3/transunet_ep5.weights.h5 \
                      checkpoints/transunet_baseline_v3/transunet_ep10.weights.h5 \
                      checkpoints/transunet_baseline_v3/transunet_ep15.weights.h5 \
                      checkpoints/transunet_baseline_v3/transunet_ep20.weights.h5 \
                      checkpoints/transunet_baseline_v3/transunet_ep25.weights.h5 \
        --output checkpoints/swa/swa_v3_all.weights.h5

    # Interpolate: 70% pretrained + 30% fine-tuned
    python scripts/swa_average.py \
        --checkpoints pretrained_weights/transunet/transunet.seresnext50.160px.comboloss.weights.h5 \
                      checkpoints/transunet_baseline_v3/transunet_ep5.weights.h5 \
        --weights 0.7 0.3 \
        --output checkpoints/swa/interp_70pre_30ep5.weights.h5

    # Dry run (just prints weight shapes, no saving)
    python scripts/swa_average.py --checkpoints ckpt1.h5 ckpt2.h5 --dry-run
"""

import os
os.environ.setdefault('KERAS_BACKEND', 'torch')

try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='SWA weight averaging for TransUNet')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='Paths to checkpoint .weights.h5 files')
    parser.add_argument('--weights', nargs='*', type=float, default=None,
                        help='Optional per-checkpoint weights (must sum to 1). '
                             'If not specified, uses uniform average.')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for averaged weights')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just load and print info, do not save')
    args = parser.parse_args()

    n_ckpts = len(args.checkpoints)
    if n_ckpts < 2:
        print("ERROR: Need at least 2 checkpoints to average")
        return

    # Determine weights
    if args.weights is not None:
        if len(args.weights) != n_ckpts:
            print(f"ERROR: --weights has {len(args.weights)} values but {n_ckpts} checkpoints")
            return
        w = np.array(args.weights, dtype=np.float64)
        if abs(w.sum() - 1.0) > 0.01:
            print(f"WARNING: weights sum to {w.sum():.4f}, normalizing to 1.0")
            w = w / w.sum()
    else:
        w = np.ones(n_ckpts, dtype=np.float64) / n_ckpts

    print(f"SWA Weight Averaging")
    print(f"  Checkpoints: {n_ckpts}")
    for i, (ckpt, wi) in enumerate(zip(args.checkpoints, w)):
        print(f"    [{i}] weight={wi:.4f}  {Path(ckpt).name}")
    print(f"  Output: {args.output}")
    print()

    # Load model architecture
    import keras
    from medicai.models import TransUNet

    model = TransUNet(
        input_shape=(160, 160, 160, 1),
        encoder_name='seresnext50',
        classifier_activation='softmax',
        num_classes=3,
    )
    print(f"Model: {model.count_params() / 1e6:.1f}M params, {len(model.weights)} weight tensors")

    # Load first checkpoint to get shapes
    model.load_weights(args.checkpoints[0])
    avg_weights = [np.array(v) * w[0] for v in model.weights]

    if args.dry_run:
        print(f"\nWeight tensor shapes:")
        for i, v in enumerate(model.weights[:10]):
            print(f"  [{i}] {v.name}: {np.array(v).shape}")
        if len(model.weights) > 10:
            print(f"  ... and {len(model.weights) - 10} more")
        print("\nDry run complete — not saving.")
        return

    # Accumulate weighted sum from remaining checkpoints
    for i in range(1, n_ckpts):
        print(f"  Loading checkpoint {i+1}/{n_ckpts}: {Path(args.checkpoints[i]).name}")
        model.load_weights(args.checkpoints[i])
        for j, v in enumerate(model.weights):
            avg_weights[j] += np.array(v) * w[i]

    # Set averaged weights back into model
    print(f"\nSetting averaged weights...")
    for j, v in enumerate(model.weights):
        v.assign(avg_weights[j])

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(output_path))
    print(f"Saved: {output_path} ({output_path.stat().st_size / 1e6:.0f} MB)")

    # Verify: reload and check a few weights match
    model.load_weights(str(output_path))
    for j in [0, len(avg_weights) // 2, -1]:
        diff = np.abs(np.array(model.weights[j]) - avg_weights[j]).max()
        assert diff < 1e-5, f"Verification failed for weight {j}: max diff {diff}"
    print("Verification passed — saved weights match averaged weights.")


if __name__ == '__main__':
    main()
