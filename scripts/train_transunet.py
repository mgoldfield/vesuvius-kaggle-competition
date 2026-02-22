#!/usr/bin/env python3
"""
Fine-tune pretrained TransUNet SEResNeXt50 on our training data.

Implements the competitor training pipeline:
- Loss: SparseDiceCE + w_srec*SkeletonRecall + w_fp*FP_Volume + w_dist*DistFromSkeleton
- Augmentation: flips, rotations, intensity shift, 3D CutOut
- Schedule: AdamW, cosine decay from 5e-5, weight_decay=1e-5
- Checkpoints every epoch

Usage:
    # Run 1: Competitor baseline (default weights)
    python scripts/train_transunet.py --run-name baseline --epochs 25

    # Run 2: High FP_Volume for thinner predictions
    python scripts/train_transunet.py --run-name thin_fp15 --fp-weight 1.5 --epochs 25

    # Run 3: Distance-from-skeleton penalty
    python scripts/train_transunet.py --run-name thin_dist --dist-weight 1.0 --epochs 25

    # Resume from a checkpoint
    python scripts/train_transunet.py --run-name thin_fp15 --weights checkpoints/transunet_baseline/transunet_best.weights.h5

    # Dry run (1 epoch, 5 samples)
    python scripts/train_transunet.py --dry-run
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
import random
import numpy as np
import pandas as pd
import tifffile
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.morphology import skeletonize

ROOT = Path("/workspace/vesuvius-kaggle-competition")
TRAIN_IMG = ROOT / "data" / "train_images"
TRAIN_LBL = ROOT / "data" / "train_labels"
DEFAULT_WEIGHTS = ROOT / "pretrained_weights" / "transunet" / "transunet.seresnext50.160px.comboloss.weights.h5"
CKPT_DIR = ROOT / "checkpoints" / "transunet"

ROI = (160, 160, 160)
VAL_SCROLL = 26002


# ── Dataset ──────────────────────────────────────────────
class VesuviusPatchDataset(Dataset):
    """Loads TIF volumes and extracts random 160^3 patches.

    Caches volumes in memory after first load (~33MB each, ~23GB for 700 vols).
    """

    def __init__(self, vol_ids, train=True, patch_size=160):
        self.vol_ids = vol_ids
        self.train = train
        self.patch_size = patch_size
        self._cache = {}

    def __len__(self):
        return len(self.vol_ids)

    def _load(self, vid):
        if vid not in self._cache:
            img = tifffile.imread(TRAIN_IMG / f"{vid}.tif")
            lbl = tifffile.imread(TRAIN_LBL / f"{vid}.tif")
            # Compute z-score normalization stats from full volume (nonzero voxels)
            # This matches medicai's NormalizeIntensity(nonzero=True, channel_wise=False)
            # which is used in our eval pipeline and Kaggle inference notebook.
            img_f = img.astype(np.float32)
            nonzero = img_f != 0
            if nonzero.any():
                mean = float(img_f[nonzero].mean())
                std = float(img_f[nonzero].std())
                if std == 0:
                    std = 1.0
            else:
                mean, std = 0.0, 1.0
            # Store raw uint8 + stats to save memory (~33MB per vol vs ~131MB for float32)
            self._cache[vid] = (img, lbl, mean, std)
        return self._cache[vid]

    def __getitem__(self, idx):
        vid = self.vol_ids[idx]
        img_raw, lbl_raw, mean, std = self._load(vid)
        img = img_raw.astype(np.float32)
        lbl = lbl_raw.astype(np.float32)

        ps = self.patch_size
        D, H, W = img.shape

        # Random crop (fg-biased: 50% chance to center on a foreground voxel)
        if self.train and random.random() < 0.5:
            fg_coords = np.argwhere(lbl == 1)
            if len(fg_coords) > 0:
                center = fg_coords[random.randint(0, len(fg_coords) - 1)]
                d0 = max(0, min(center[0] - ps // 2, D - ps))
                h0 = max(0, min(center[1] - ps // 2, H - ps))
                w0 = max(0, min(center[2] - ps // 2, W - ps))
            else:
                d0 = random.randint(0, max(0, D - ps))
                h0 = random.randint(0, max(0, H - ps))
                w0 = random.randint(0, max(0, W - ps))
        else:
            d0 = random.randint(0, max(0, D - ps))
            h0 = random.randint(0, max(0, H - ps))
            w0 = random.randint(0, max(0, W - ps))

        img_patch = img[d0:d0+ps, h0:h0+ps, w0:w0+ps]
        lbl_patch = lbl[d0:d0+ps, h0:h0+ps, w0:w0+ps]

        # Z-score normalize using full-volume stats (matches eval pipeline)
        # Apply BEFORE augmentation so intensity shift operates in z-score space
        nonzero = img_patch != 0
        img_patch = img_patch.copy()
        img_patch[nonzero] = (img_patch[nonzero] - mean) / std

        if self.train:
            img_patch, lbl_patch = self._augment(img_patch, lbl_patch)

        # Generate skeleton target and distance map for training
        skel = self._generate_skeleton(lbl_patch)
        dist = self._generate_dist_from_skeleton(lbl_patch)
        bdist = self._generate_boundary_dist(lbl_patch)

        # channels-last format: (D, H, W, 1) and (D, H, W, 4)
        img_out = img_patch[..., None]  # (D,H,W,1)
        lbl_out = np.stack([lbl_patch, skel, dist, bdist], axis=-1)  # (D,H,W,4)

        return img_out.astype(np.float32), lbl_out.astype(np.float32)

    def _augment(self, img, lbl):
        """Augmentation: flips, rotations, intensity shift, 3D CutOut.

        Input img is already z-score normalized. Augmentations operate in z-score space.
        """
        # Random flips (all 3 axes, p=0.5)
        for axis in range(3):
            if random.random() < 0.5:
                img = np.flip(img, axis=axis).copy()
                lbl = np.flip(lbl, axis=axis).copy()

        # Random 90-degree rotations (axes 0,1, p=0.4)
        if random.random() < 0.4:
            k = random.randint(1, 3)
            img = np.rot90(img, k=k, axes=(0, 1)).copy()
            lbl = np.rot90(lbl, k=k, axes=(0, 1)).copy()

        # Random intensity shift in z-score space (p=0.5)
        # ±0.10 std devs, matching competitor TPU notebook's
        # RandShiftIntensity(offsets=0.10) applied after z-score normalization
        if random.random() < 0.5:
            shift = random.uniform(-0.10, 0.10)
            img = img + shift

        # 3D CutOut (up to 6 random cuboid blocks, p=1.0)
        # Fill with 0 = mean intensity in z-score space
        for _ in range(6):
            bs = random.randint(2, 8)
            d0 = random.randint(0, max(0, img.shape[0] - bs))
            h0 = random.randint(0, max(0, img.shape[1] - bs))
            w0 = random.randint(0, max(0, img.shape[2] - bs))
            img[d0:d0+bs, h0:h0+bs, w0:w0+bs] = 0

        return img, lbl

    def _generate_skeleton(self, lbl):
        """Skeletonize foreground and dilate by 1."""
        mask = (lbl == 1)
        if not mask.any():
            return np.zeros_like(lbl, dtype=np.float32)
        skel = skeletonize(mask)
        tubed = binary_dilation(skel, iterations=1)
        return tubed.astype(np.float32)

    def _generate_dist_from_skeleton(self, lbl):
        """Distance transform from GT skeleton. Used by dist-from-skeleton loss."""
        mask = (lbl == 1)
        if not mask.any():
            return np.zeros_like(lbl, dtype=np.float32)
        skel = skeletonize(mask)
        if not skel.any():
            return np.zeros_like(lbl, dtype=np.float32)
        # Distance from each voxel to nearest skeleton voxel
        dist = distance_transform_edt(~skel).astype(np.float32)
        # Normalize: cap at 10 voxels, scale to [0, 1]
        dist = np.clip(dist / 10.0, 0, 1)
        return dist

    def _generate_boundary_dist(self, lbl):
        """Signed distance transform from GT boundary (Kervadec et al. 2019).
        Negative inside foreground, positive outside, zero at boundary.
        Normalized by max distance per patch."""
        fg = (lbl == 1)
        if not fg.any():
            return np.zeros_like(lbl, dtype=np.float32)
        bg = ~fg
        dist_in = distance_transform_edt(fg)
        dist_out = distance_transform_edt(bg)
        signed = dist_out - dist_in
        max_d = max(np.abs(signed).max(), 1.0)
        signed = signed / max_d
        # Zero out unlabeled regions
        signed[lbl == 2] = 0.0
        return signed.astype(np.float32)


def collate_fn(batch):
    """Stack into numpy arrays (Keras expects numpy, not torch tensors for channels-last)."""
    imgs, lbls = zip(*batch)
    return np.stack(imgs), np.stack(lbls)


# ── Loss function (Keras) ────────────────────────────────
def build_loss(num_classes=3, w_srec=0.75, w_fp=0.50, w_dist=0.0, dist_power=1.0,
               w_boundary=0.0, w_cldice=0.0):
    """Build the SkeletonRecallPlusDiceLoss using Keras ops.

    Args:
        w_srec: Weight for skeleton recall loss (default 0.75)
        w_fp: Weight for FP volume loss (default 0.50). Higher = thinner predictions.
        w_dist: Weight for distance-from-skeleton loss (default 0.0 = disabled).
                Penalizes predictions that are far from the GT skeleton centerline.
        w_boundary: Weight for boundary loss (default 0.0 = disabled).
                    Signed distance from GT boundary — penalizes predictions outside GT,
                    rewards predictions inside GT (Kervadec et al. 2019).
        w_cldice: Weight for centerline Dice loss (default 0.0 = disabled).
                  Measures skeleton overlap between pred and GT. Rewards correct
                  topology — complementary to dist_sq (penalty) and boundary (penalty).
    """
    import keras
    from medicai.losses import SparseDiceCELoss

    base_loss_fn = SparseDiceCELoss(
        from_logits=False,
        num_classes=num_classes,
        ignore_class_ids=2,
    )

    # clDice loss (optional) — uses soft-skeletonization, differentiable
    cldice_loss_fn = None
    if w_cldice > 0:
        from medicai.losses import SparseCenterlineDiceLoss
        cldice_loss_fn = SparseCenterlineDiceLoss(
            from_logits=False,
            num_classes=num_classes,
            target_class_ids=1,
            ignore_class_ids=2,
            iters=10,
            memory_efficient_skeleton=True,
        )

    class SkeletonRecallPlusDiceLoss(keras.losses.Loss):
        def call(self, y_true, y_pred):
            y_true_mask = y_true[..., 0]
            y_true_skel = y_true[..., 1]
            y_true_dist = y_true[..., 2]

            pred_ink_prob = y_pred[..., 1]
            valid_mask = keras.ops.cast(y_true_mask != 2, "float32")

            # 1. Base Dice+CE
            base_loss = base_loss_fn(y_true_mask[..., None], y_pred)

            # 2. Skeleton Recall
            intersection = keras.ops.sum(
                pred_ink_prob * y_true_skel * valid_mask, axis=(1, 2, 3)
            )
            skeleton_sum = keras.ops.sum(
                y_true_skel * valid_mask, axis=(1, 2, 3)
            )
            has_skeleton = keras.ops.cast(skeleton_sum > 0, "float32")
            recall = (intersection + 1e-6) / (skeleton_sum + 1e-6)
            skel_loss = keras.ops.mean((1.0 - recall) * has_skeleton)

            # 3. FP Volume
            gt_bg = keras.ops.cast(y_true_mask == 0, "float32")
            fp_volume = pred_ink_prob * gt_bg * valid_mask
            fp_loss = keras.ops.sum(fp_volume) / (
                keras.ops.sum(gt_bg * valid_mask) + 1e-6
            )

            total = base_loss + w_srec * skel_loss + w_fp * fp_loss

            # 4. Distance-from-skeleton penalty (optional)
            if w_dist > 0:
                # Penalize predictions proportionally to their distance from
                # the GT skeleton. Predictions ON the skeleton get zero penalty.
                # Predictions far from it get high penalty.
                dist_term = y_true_dist ** dist_power if dist_power != 1.0 else y_true_dist
                dist_penalty = pred_ink_prob * dist_term * valid_mask
                dist_loss = keras.ops.sum(dist_penalty) / (
                    keras.ops.sum(valid_mask) + 1e-6
                )
                total = total + w_dist * dist_loss

            # 5. Boundary loss (optional, Kervadec et al. 2019)
            if w_boundary > 0:
                # Signed distance: positive outside GT, negative inside GT.
                # Minimizing pred * signed_dist pushes predictions toward GT boundary.
                y_true_bdist = y_true[..., 3]
                bd_loss = keras.ops.sum(pred_ink_prob * y_true_bdist * valid_mask) / (
                    keras.ops.sum(valid_mask) + 1e-6
                )
                total = total + w_boundary * bd_loss

            # 6. Centerline Dice loss (optional)
            if cldice_loss_fn is not None:
                cldice = cldice_loss_fn(y_true_mask[..., None], y_pred)
                total = total + w_cldice * cldice

            return total

    return SkeletonRecallPlusDiceLoss(name="skel_recall_fp_loss")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--save-every', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--run-name', type=str, default='baseline',
                        help='Name for this run (determines checkpoint dir)')
    parser.add_argument('--fp-weight', type=float, default=0.50,
                        help='FP_Volume loss weight (default 0.50, try 1.5 for thinner)')
    parser.add_argument('--skel-weight', type=float, default=0.75,
                        help='SkeletonRecall loss weight (default 0.75)')
    parser.add_argument('--dist-weight', type=float, default=0.0,
                        help='Distance-from-skeleton loss weight (0=disabled)')
    parser.add_argument('--dist-power', type=float, default=1.0,
                        help='Power for distance term (1=linear, 2=quadratic)')
    parser.add_argument('--boundary-weight', type=float, default=0.0,
                        help='Boundary loss weight (0=disabled). Penalizes distance from GT surface boundary.')
    parser.add_argument('--cldice-weight', type=float, default=0.0,
                        help='Centerline Dice loss weight (0=disabled). Rewards skeleton overlap between pred and GT.')
    parser.add_argument('--label-dir', type=str, default=None,
                        help='Custom label directory (default: data/train_labels/). Use for pseudo-labels.')
    parser.add_argument('--freeze-encoder', action='store_true',
                        help='Freeze CNN encoder (SEResNeXt50), only train decoder + head')
    parser.add_argument('--discriminative-lr', action='store_true',
                        help='Use per-layer-group LRs: encoder=lr/100, decoder=lr/10, head=lr')
    args = parser.parse_args()

    if args.dry_run:
        args.epochs = 1

    # Override label directory if specified (for pseudo-labels)
    global TRAIN_LBL
    if args.label_dir:
        TRAIN_LBL = Path(args.label_dir)
        print(f"Using custom label dir: {TRAIN_LBL}")

    import keras
    from medicai.models import TransUNet
    from medicai.metrics import SparseDiceMetric

    # Use bfloat16 — same exponent range as float32 (no gradient underflow)
    # but half the memory like float16. mixed_float16 caused underflow on
    # SkeletonRecall and FP_Volume gradient signals. Competitor trained on TPU
    # which uses bfloat16 natively.
    keras.mixed_precision.set_global_policy('mixed_bfloat16')
    print("Mixed precision: bfloat16 compute, float32 weights")

    # Per-run checkpoint directory
    ckpt_dir = ROOT / "checkpoints" / f"transunet_{args.run_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data split ──
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    available = set(int(p.stem) for p in TRAIN_IMG.glob("*.tif")) & \
                set(int(p.stem) for p in TRAIN_LBL.glob("*.tif"))
    train_df = train_df[train_df.id.isin(available)]

    val_ids = train_df[train_df.scroll_id == VAL_SCROLL].id.tolist()
    train_ids = train_df[train_df.scroll_id != VAL_SCROLL].id.tolist()

    if args.dry_run:
        train_ids = train_ids[:5]
        val_ids = val_ids[:2]

    print(f"Train: {len(train_ids)} volumes, Val: {len(val_ids)} volumes")

    # ── DataLoaders ──
    train_ds = VesuviusPatchDataset(train_ids, train=True)
    val_ds = VesuviusPatchDataset(val_ids, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=2,
        collate_fn=collate_fn, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn, pin_memory=False,
    )

    # ── Model ──
    print(f"Loading TransUNet from {args.weights}")
    model = TransUNet(
        input_shape=(160, 160, 160, 1),
        encoder_name='seresnext50',
        classifier_activation='softmax',
        num_classes=3,
    )
    model.load_weights(str(args.weights))
    print(f"Model params: {model.count_params() / 1e6:.1f}M")

    # ── Freeze encoder if requested ──
    if args.freeze_encoder:
        n_frozen = 0
        for layer in model.layers:
            name = layer.name
            if (name.startswith(('conv1', 'pool1', 'stack'))
                    or name.startswith('transunet_vit')):
                layer.trainable = False
                n_frozen += 1
        trainable_params = sum(
            np.prod(v.shape) for v in model.trainable_weights
        )
        print(f"Frozen encoder: {n_frozen} layers frozen, "
              f"{trainable_params / 1e6:.1f}M trainable params remaining")

    # ── Training setup ──
    use_torch_optim = args.discriminative_lr

    total_steps = len(train_loader) * args.epochs
    loss_fn = build_loss(w_srec=args.skel_weight, w_fp=args.fp_weight,
                         w_dist=args.dist_weight, dist_power=args.dist_power,
                         w_boundary=args.boundary_weight,
                         w_cldice=args.cldice_weight)

    if use_torch_optim:
        # Discriminative LR: use PyTorch optimizer with parameter groups.
        # encoder=lr/100, vit=lr/10, decoder=lr/10, head=lr
        enc_params, vit_params, dec_params, head_params = [], [], [], []
        for var in model.trainable_weights:
            name = var.name
            tensor = var.value  # underlying torch tensor
            if name.startswith(('conv1', 'pool1', 'stack')):
                enc_params.append(tensor)
            elif name.startswith('transunet_vit'):
                vit_params.append(tensor)
            elif name.startswith('final_conv'):
                head_params.append(tensor)
            else:
                dec_params.append(tensor)

        lr_enc = args.lr / 100   # e.g., 5e-7 for lr=5e-5
        lr_vit = args.lr / 10    # e.g., 5e-6
        lr_dec = args.lr / 10    # e.g., 5e-6
        lr_head = args.lr        # e.g., 5e-5

        param_groups = []
        if enc_params:
            param_groups.append({'params': enc_params, 'lr': lr_enc})
        if vit_params:
            param_groups.append({'params': vit_params, 'lr': lr_vit})
        if dec_params:
            param_groups.append({'params': dec_params, 'lr': lr_dec})
        if head_params:
            param_groups.append({'params': head_params, 'lr': lr_head})

        torch_optimizer = torch.optim.AdamW(
            param_groups, weight_decay=args.weight_decay
        )

        # Cosine decay scheduler for PyTorch
        torch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            torch_optimizer,
            T_max=total_steps // args.grad_accum,  # total optimizer steps
            eta_min=args.lr * 0.1 / 100,  # min LR for slowest group
        )

        print(f"\nDiscriminative LR (PyTorch optimizer):")
        print(f"  Encoder: {lr_enc:.1e} ({len(enc_params)} tensors)")
        print(f"  ViT:     {lr_vit:.1e} ({len(vit_params)} tensors)")
        print(f"  Decoder: {lr_dec:.1e} ({len(dec_params)} tensors)")
        print(f"  Head:    {lr_head:.1e} ({len(head_params)} tensors)")
    else:
        # Standard Keras optimizer
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.lr,
            decay_steps=total_steps,
            alpha=0.1,  # min LR = lr * 0.1
        )

        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=args.weight_decay,
        )

        # Note: gradient_accumulation_steps in Keras 3 optimizer
        if args.grad_accum > 1:
            optimizer.gradient_accumulation_steps = args.grad_accum
            print(f"Gradient accumulation: {args.grad_accum} steps")

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
        )

    print(f"\nTraining config:")
    print(f"  LR: {args.lr}" + (" (discriminative)" if use_torch_optim else
          f" → {args.lr * 0.1} (cosine decay)"))
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Effective batch size: {args.grad_accum}")
    print(f"  Steps/epoch: {len(train_loader)}")
    print(f"  Total steps: {total_steps}")
    print(f"  Loss weights: skel={args.skel_weight}, fp={args.fp_weight}, dist={args.dist_weight}, dist_power={args.dist_power}, boundary={args.boundary_weight}, cldice={args.cldice_weight}")
    print(f"  Freeze encoder: {args.freeze_encoder}")
    print(f"  Checkpoint dir: {ckpt_dir}")

    # ── Training loop ──
    # Manual loop for better control over checkpointing and logging
    best_loss = float('inf')
    t_start = time.time()

    # Enable mixed precision context for PyTorch manual loop
    amp_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if use_torch_optim else None

    for epoch in range(args.epochs):
        t_epoch = time.time()
        train_losses = []

        # Training (don't reset trainable if encoder is frozen)
        if not args.freeze_encoder:
            model.trainable = True
        if use_torch_optim:
            torch_optimizer.zero_grad()

        for step, (x_batch, y_batch) in enumerate(train_loader):
            if use_torch_optim:
                # Manual PyTorch training step for discriminative LR
                x_t = torch.tensor(x_batch, device='cuda')
                y_t = torch.tensor(y_batch, device='cuda')
                with amp_ctx:
                    y_pred = model(x_t, training=True)
                    loss = loss_fn(y_t, y_pred)
                    loss_scaled = loss / args.grad_accum
                loss_scaled.backward()

                if (step + 1) % args.grad_accum == 0:
                    torch_optimizer.step()
                    torch_scheduler.step()
                    torch_optimizer.zero_grad()

                loss_val = float(loss.detach().cpu())
                del x_t, y_t, y_pred, loss, loss_scaled
            else:
                loss_val = model.train_on_batch(x_batch, y_batch)
                if isinstance(loss_val, (list, tuple)):
                    loss_val = loss_val[0]

            train_losses.append(float(loss_val))

            if (step + 1) % 50 == 0 or step == 0:
                elapsed = time.time() - t_epoch
                eta = elapsed / (step + 1) * (len(train_loader) - step - 1)
                print(f"  Epoch {epoch+1}/{args.epochs} step {step+1}/{len(train_loader)}: "
                      f"loss={np.mean(train_losses[-50:]):.4f} "
                      f"({elapsed/60:.1f}min, ETA {eta/60:.1f}min)")

        # Flush any remaining accumulated gradients
        if use_torch_optim and len(train_loader) % args.grad_accum != 0:
            torch_optimizer.step()
            torch_scheduler.step()
            torch_optimizer.zero_grad()

        # Validation
        val_losses = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                if use_torch_optim:
                    x_t = torch.tensor(x_val, device='cuda')
                    y_t = torch.tensor(y_val, device='cuda')
                    with amp_ctx:
                        y_pred = model(x_t, training=False)
                        vl = float(loss_fn(y_t, y_pred).detach().cpu())
                    del x_t, y_t, y_pred
                else:
                    vl = model.test_on_batch(x_val, y_val)
                    if isinstance(vl, (list, tuple)):
                        vl = vl[0]
                val_losses.append(float(vl))

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses) if val_losses else float('nan')
        epoch_time = time.time() - t_epoch

        print(f"\nEpoch {epoch+1}/{args.epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"time={epoch_time/60:.1f}min")

        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = ckpt_dir / f"transunet_ep{epoch+1}.weights.h5"
            model.save_weights(str(ckpt_path))
            print(f"  Checkpoint saved: {ckpt_path}")

        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            best_path = ckpt_dir / "transunet_best.weights.h5"
            model.save_weights(str(best_path))
            print(f"  New best model saved: {best_path} (val_loss={val_loss:.4f})")

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Training complete: {total_time/3600:.1f} hours")
    print(f"Best val_loss: {best_loss:.4f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
