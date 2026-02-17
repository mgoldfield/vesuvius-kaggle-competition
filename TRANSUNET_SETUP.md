# TransUNet Setup Guide

How to install and run the TransUNet SEResNeXt50 model used by top competitors.

## Installation (Local — PyTorch backend)

### 1. Install Keras 3 and medicai

```bash
pip install keras tensorflow-cpu
pip install git+https://github.com/innat/medic-ai.git --force-reinstall --no-deps
```

**CRITICAL:** Do NOT use `pip install medicai`. The PyPI version (0.0.3) creates a model
with 29.4M params — the wrong architecture. The GitHub source creates the correct 70.1M
param model that matches the pretrained weights. This caused `ValueError: A total of 317
objects could not be loaded` when loading weights.

The `tensorflow-cpu` package is needed by `medicai.transforms` (NormalizeIntensity etc.)
even when using the PyTorch backend.

### 2. Download pretrained weights from Kaggle

```bash
# Requires kaggle CLI configured with API key
kaggle models instances versions download ipythonx/vsd-model/keras/transunet/3
# Extracts to: pretrained_weights/transunet/
```

Three weight files (803 MB each):
- `transunet.seresnext50.160px.comboloss.weights.h5` — **Best, LB 0.545** (SparseDiceCE + SkeletonRecall + FP_Volume)
- `transunet.seresnext50.160px.weights.h5` — LB 0.505 (base loss only)
- `transunet.seresnext50.128px.weights.h5` — LB 0.500 (smaller patches)

### 3. Download offline wheels (for Kaggle submission)

```bash
kaggle datasets download tonylica/vsdetection-packages-offline-installer-only
# Contains: keras_nightly-3.12.0, medicai-0.0.3, tifffile, imagecodecs wheels
# NOTE: The medicai wheel in this package is the Kaggle-specific version that
# works correctly with the pretrained weights (unlike pip install medicai).
```

## Usage

### Loading the model (PyTorch backend)

```python
import os
os.environ['KERAS_BACKEND'] = 'torch'  # MUST be set before importing keras

import numpy as np
import torch
from medicai.models import TransUNet
from medicai.transforms import Compose, NormalizeIntensity

# Build model (channels-last: input and output are NDHWC)
model = TransUNet(
    input_shape=(160, 160, 160, 1),
    encoder_name='seresnext50',
    classifier_activation=None,  # Raw logits (not softmax)
    num_classes=3,               # bg=0, surface=1, unlabeled=2
)
model.load_weights('pretrained_weights/transunet/transunet.seresnext50.160px.comboloss.weights.h5')
print(f"Params: {model.count_params() / 1e6:.1f}M")  # Should print 70.1M
```

### Preprocessing

```python
normalize = Compose([
    NormalizeIntensity(keys=["image"], nonzero=True, channel_wise=False),
])

import tifffile
vol = tifffile.imread('data/train_images/26894125.tif').astype(np.float32)
vol_5d = vol[None, ..., None]  # (1, D, H, W, 1) — batch + channel dims
vol_norm = normalize({"image": vol_5d})["image"]
```

### Single patch inference

```python
# Extract a 160^3 patch
D, H, W = vol.shape
d0, h0, w0 = (D-160)//2, (H-160)//2, (W-160)//2
patch = vol_norm[:, d0:d0+160, h0:h0+160, w0:w0+160, :]

with torch.no_grad():
    logits = model(patch)  # (1, 160, 160, 160, 3) — channels-last

logits_np = logits.detach().cpu().numpy()
```

### Full-volume SlidingWindowInference

```python
from medicai.utils.inference import SlidingWindowInference

swi = SlidingWindowInference(
    model,
    num_classes=3,
    roi_size=(160, 160, 160),
    sw_batch_size=1,
    mode='gaussian',
    overlap=0.50,  # 0.42-0.60 range used by competitors
)

logits_full = np.asarray(swi(vol_norm))  # (1, D, H, W, 3)
```

### Converting 3-class logits to binary probability

```python
def logsumexp2(a, b):
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m) + 1e-12)

def sigmoid_stable(x):
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

logits = logits_full[0]  # (D, H, W, 3)
L0, L1, L2 = logits[..., 0], logits[..., 1], logits[..., 2]
binary_logit = logsumexp2(L1, L2) - L0  # Combines surface + unlabeled vs background
prob = sigmoid_stable(binary_logit)
```

## Architecture Details

TransUNet + SEResNeXt50 = 70.1M params:
- **Encoder:** SEResNeXt50 (ImageNet pretrained) — squeeze-and-excitation blocks + grouped convolutions
- **Bridge:** Vision Transformer (12 layers, 12 heads, hidden_dim=768)
- **Decoder:** U-Net-style upsampling with skip connections from encoder

For comparison, our SegResNet had 4.7M params (15x smaller).

## Memory Requirements (RTX 5090, 33.7 GB)

| Config | Peak VRAM | Notes |
|--------|-----------|-------|
| Model load only | 0.28 GB | Lightweight |
| Forward pass (bs=1, 160^3) | 14.33 GB | Activation memory |
| Forward + backward (bs=1, 160^3) | 19.89 GB | Training peak |
| Estimated bs=2 | ~33.8 GB | Likely OOM without mixed precision |

**Recommended training config:** bs=1 with gradient_accumulation=4 (simulates bs=4).
Mixed precision (fp16) could roughly halve activation memory, enabling bs=2.

## Kaggle Submission Notes

On Kaggle, use JAX backend (faster XLA compilation, competitors' default):
```python
os.environ['KERAS_BACKEND'] = 'jax'
```

Install from offline wheels (no internet on Kaggle):
```python
!pip install /kaggle/input/vsdetection-packages-offline-installer-only/whls/*.whl \
    --no-index --find-links /kaggle/input/vsdetection-packages-offline-installer-only/whls/
```

The offline `medicai` wheel from this Kaggle dataset works correctly (unlike pip install).

## Key Gotchas

1. **pip medicai is WRONG** — Always install from GitHub source locally
2. **Set KERAS_BACKEND before import** — Cannot change after keras is imported
3. **Channels-last format** — Input is (N,D,H,W,1), output is (N,D,H,W,3)
4. **classifier_activation=None** — Must use raw logits, not softmax, for proper averaging
5. **tensorflow-cpu required** — medicai transforms import tensorflow even with torch backend
