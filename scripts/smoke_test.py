"""
Quick smoke test for v12 and refinement notebooks.
Tests all imports, model creation, loss functions, custom callbacks,
and runs 1 training step to verify the full pipeline works.

Usage:
    python scripts/smoke_test.py
"""
import sys
import time
import traceback

ROOT = "/workspace/vesuvius-kaggle-competition"
t0 = time.time()


def test(name, fn):
    try:
        result = fn()
        elapsed = time.time() - t0
        print(f"  [{elapsed:6.1f}s] PASS: {name}" + (f" — {result}" if result else ""))
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{elapsed:6.1f}s] FAIL: {name}")
        traceback.print_exc()
        return False


print("=" * 60)
print("Smoke Test: v12 + Refinement Pipeline")
print("=" * 60)
failures = []

# --- v12 Notebook Components ---
print("\n--- v12 Notebook Components ---")

def test_imports():
    import torch, numpy as np, pandas as pd, tifffile
    from pathlib import Path
    from fastai.learner import Learner
    from fastai.callback.core import Callback
    from fastai.callback.schedule import fit_flat_cos
    from fastai.callback.tracker import SaveModelCallback, TrackerCallback
    from fastai.callback.fp16 import MixedPrecision
    from fastai.data.core import DataLoaders
    from fastai.metrics import Metric
    from monai.networks.nets import SegResNet
    from scipy.ndimage import distance_transform_edt
    from topometrics.leaderboard import compute_leaderboard_score
    return "all imports OK"
if not test("v12 imports", test_imports): failures.append("v12 imports")

def test_fit_flat_cos_exists():
    from fastai.learner import Learner
    assert hasattr(Learner, 'fit_flat_cos'), "fit_flat_cos not found!"
    return "Learner.fit_flat_cos exists"
if not test("fit_flat_cos API", test_fit_flat_cos_exists): failures.append("fit_flat_cos API")

def test_paths():
    from pathlib import Path
    p = Path(ROOT)
    checks = {
        "train_images": (p / "data" / "train_images").exists(),
        "train_labels": (p / "data" / "train_labels").exists(),
        "test_images": (p / "data" / "test_images").exists(),
        "train.csv": (p / "data" / "train.csv").exists(),
        "test.csv": (p / "data" / "test.csv").exists(),
        "SuPreM weights": (p / "pretrained_weights" / "supervised_suprem_segresnet_2100.pth").exists(),
        "v9 traced": (p / "kaggle" / "kaggle_weights_download" / "best_segresnet_v9_traced.pt").exists(),
    }
    missing = [k for k, v in checks.items() if not v]
    if missing:
        raise FileNotFoundError(f"Missing: {missing}")
    return f"all {len(checks)} paths exist"
if not test("data/weight paths", test_paths): failures.append("paths")

def test_model_creation():
    import torch
    from monai.networks.nets import SegResNet
    model = SegResNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        init_filters=16, blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    )
    dummy = torch.randn(1, 1, 32, 32, 32)
    out = model(dummy)
    assert out.shape == (1, 1, 32, 32, 32)
    return f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params"
if not test("SegResNet model", test_model_creation): failures.append("model")

def test_suprem_weights():
    import torch
    from pathlib import Path
    from monai.networks.nets import SegResNet
    model = SegResNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        init_filters=16, blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    )
    ckpt = torch.load(
        Path(ROOT) / "pretrained_weights" / "supervised_suprem_segresnet_2100.pth",
        map_location="cpu", weights_only=False
    )
    loaded = 0
    store_dict = model.state_dict()
    for key in ckpt["net"]:
        new_key = key.replace("module.", "")
        if new_key in store_dict and "conv_final" not in new_key:
            if store_dict[new_key].shape == ckpt["net"][key].shape:
                loaded += 1
    return f"loaded {loaded} params"
if not test("SuPreM weight loading", test_suprem_weights): failures.append("SuPreM")

def test_delayed_save_callback():
    import numpy as np
    from fastai.callback.tracker import TrackerCallback

    class DelayedSaveCallback(TrackerCallback):
        order = 61
        def __init__(self, monitor='comp_score', comp=np.greater, fname='best', start_epoch=25):
            super().__init__(monitor=monitor, comp=comp)
            self.fname = fname
            self.start_epoch = start_epoch
        def after_epoch(self):
            if self.epoch < self.start_epoch:
                return
            super().after_epoch()
            if self.new_best:
                self.learn.save(self.fname)

    cb = DelayedSaveCallback(start_epoch=25)
    assert cb.start_epoch == 25
    return "created OK"
if not test("DelayedSaveCallback", test_delayed_save_callback): failures.append("DelayedSave")

def test_loss_function():
    import torch
    import torch.nn.functional as F

    # Minimal loss test
    pred = torch.randn(1, 1, 32, 32, 32)
    target = (torch.rand(1, 1, 32, 32, 32) > 0.5).float()
    mask = torch.ones(1, 1, 32, 32, 32)
    dist_map = torch.randn(1, 1, 32, 32, 32) * 0.1

    pred_sig = torch.sigmoid(pred)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    bce_masked = (bce * mask).sum() / mask.sum()

    intersection = (pred_sig * mask * target * mask).sum()
    dice = (2.0 * intersection + 1.0) / (pred_sig.sum() + target.sum() + 1.0)

    return f"BCE={bce_masked:.3f}, Dice={dice:.3f}"
if not test("loss function", test_loss_function): failures.append("loss")

def test_tiff_loading():
    import tifffile
    from pathlib import Path
    imgs = sorted((Path(ROOT) / "data" / "train_images").glob("*.tif"))[:1]
    img = tifffile.imread(imgs[0])
    assert img.shape == (320, 320, 320) and img.dtype.name == 'uint8'
    return f"loaded {imgs[0].stem}, shape={img.shape}"
if not test("TIFF loading (imagecodecs)", test_tiff_loading): failures.append("TIFF")

def test_v12_1batch_gpu():
    """Run 1 forward + backward pass on GPU to verify everything works end-to-end."""
    import torch
    import torch.nn.functional as F
    import numpy as np
    import tifffile
    from pathlib import Path
    from monai.networks.nets import SegResNet
    from scipy.ndimage import distance_transform_edt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SegResNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        init_filters=16, blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    ).to(device)

    # Create a small batch (32^3 patch for speed)
    ps = 32
    img = torch.randn(2, 1, ps, ps, ps, device=device)
    target = (torch.rand(2, 1, ps, ps, ps, device=device) > 0.5).float()
    mask = torch.ones_like(target)

    # Forward + backward
    with torch.amp.autocast("cuda"):
        out = model(img)
        pred_sig = torch.sigmoid(out)
        bce = F.binary_cross_entropy_with_logits(out, target, reduction='mean')
        intersection = (pred_sig * target).sum()
        dice = 1 - (2 * intersection + 1) / (pred_sig.sum() + target.sum() + 1)
        loss = 0.5 * bce + 0.5 * dice

    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

    return f"loss={loss.item():.3f}, grad_norm={grad_norm:.3f}, device={device}"
if not test("v12 1-batch GPU", test_v12_1batch_gpu): failures.append("v12 GPU")


# --- Refinement Components ---
print("\n--- Refinement Components ---")

def test_traced_v9():
    import torch
    from pathlib import Path
    traced = torch.jit.load(
        str(Path(ROOT) / "kaggle" / "kaggle_weights_download" / "best_segresnet_v9_traced.pt"),
        map_location="cpu"
    )
    dummy = torch.randn(1, 1, 32, 32, 32)
    out = traced(dummy)
    assert out.shape == (1, 1, 32, 32, 32)
    return f"output range [{out.min():.2f}, {out.max():.2f}]"
if not test("traced v9 model", test_traced_v9): failures.append("traced v9")

def test_refinement_model():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.conv(x)

    class RefinementUNet3D(nn.Module):
        def __init__(self, in_ch=1, out_ch=1, channels=[8, 16, 32, 64]):
            super().__init__()
            self.encoders = nn.ModuleList()
            self.pools = nn.ModuleList()
            prev = in_ch
            for ch in channels:
                self.encoders.append(ConvBlock(prev, ch))
                self.pools.append(nn.MaxPool3d(2))
                prev = ch
            self.bottleneck = ConvBlock(channels[-1], channels[-1])
            self.dropout = nn.Dropout3d(0.2)
            self.decoders = nn.ModuleList()
            for i in range(len(channels) - 1, -1, -1):
                in_c = channels[i] + (channels[i] if i < len(channels) - 1 else channels[-1])
                self.decoders.append(ConvBlock(in_c, channels[i]))
            self.final = nn.Conv3d(channels[0], out_ch, 1)

        def forward(self, x):
            skips = []
            for enc, pool in zip(self.encoders, self.pools):
                x = enc(x)
                skips.append(x)
                x = pool(x)
            x = self.dropout(self.bottleneck(x))
            for i, dec in enumerate(self.decoders):
                x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
                skip = skips[-(i + 1)]
                if x.shape != skip.shape:
                    x = F.pad(x, [0, skip.shape[4]-x.shape[4], 0, skip.shape[3]-x.shape[3], 0, skip.shape[2]-x.shape[2]])
                x = dec(torch.cat([x, skip], dim=1))
            return self.final(x)

    model = RefinementUNet3D()
    n_params = sum(p.numel() for p in model.parameters())
    dummy = torch.randn(1, 1, 32, 32, 32)
    out = model(dummy)
    assert out.shape == dummy.shape
    return f"{n_params/1e3:.0f}K params, shape OK"
if not test("RefinementUNet3D model", test_refinement_model): failures.append("refinement model")

def test_fit_flat_cos_with_slice():
    """Test that fit_flat_cos actually works with discriminative LR (slice)."""
    import torch
    from fastai.learner import Learner
    from fastai.data.core import DataLoaders
    from torch.utils.data import DataLoader, TensorDataset

    # Create minimal learner with 2 samples
    x = torch.randn(4, 1, 16, 16, 16)
    y = torch.randn(4, 1, 16, 16, 16)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=2)
    dls = DataLoaders(dl, dl)

    model = torch.nn.Conv3d(1, 1, 3, padding=1)

    def simple_splitter(m):
        return [list(m.parameters())[:1], list(m.parameters())[1:]]

    learn = Learner(dls, model, loss_func=torch.nn.MSELoss(), splitter=simple_splitter)
    learn.fit_flat_cos(1, lr=slice(1e-5, 1e-3))
    return "1 epoch with slice LR completed"
if not test("fit_flat_cos with slice LR", test_fit_flat_cos_with_slice): failures.append("flat_cos_slice")


# --- Summary ---
print("\n" + "=" * 60)
elapsed = time.time() - t0
if failures:
    print(f"SMOKE TEST FAILED ({len(failures)} failures in {elapsed:.1f}s):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print(f"ALL SMOKE TESTS PASSED ({elapsed:.1f}s)")
    print("Safe to start overnight pipeline.")
    sys.exit(0)
