# Pre-trained 3D Model Research

Research into pre-trained 3D models for the Vesuvius Surface Detection competition.
Competition rules allow **"Freely & publicly available external data, including pre-trained models."**

## Summary of Options

| Model | Params | Pre-training Data | Architecture Match | VRAM (train 128^3 bs=2) | Verdict |
|-------|--------|-------------------|-------------------|------------------------|---------|
| SuPreM SegResNet | 4.7M | 2,100 CT (supervised, 25 classes) | New arch (MONAI) | ~5-6 GB est. | **Best option** |
| SuPreM UNet | 19.1M | 2,100 CT (supervised, 25 classes) | Close but different channels | ~11 GB est. | Good option |
| Models Genesis UNet | ~19M | 623 CT (self-supervised) | Close but different channels | ~11 GB est. | Decent option |
| MONAI SwinUNETR | 62.2M | 5,050 CT (self-supervised) | New arch (MONAI) | OOM at 128^3 | Too VRAM-hungry |
| Med3D ResNet | varies | 23 datasets (supervised) | Needs adapter | Unknown | More work |
| nnUNet | varies | Trains from scratch | Different framework | ~10 GB | Workflow change |

## Recommended: SuPreM SegResNet

**Why:**
- Pre-trained with supervision on 2,100 CT volumes (25 organ classes from AbdomenAtlas 1.1)
- Only 4.7M params — very VRAM efficient, trains fast
- Uses GroupNorm by default — perfect for our batch_size=2
- SegResNet won BraTS 2023, KiTS 2023, Seg.A 2023 competitions
- MONAI provides clean, well-tested implementation
- Weights are freely available on HuggingFace (56.5 MB)
- Small enough for Kaggle P100 (16GB) inference easily

**Trade-off:** Smaller model may have less capacity than our 22.6M UNet, but pre-training
compensates significantly. Can also try init_filters=32 (18.8M params) without pre-trained
weights as a compromise.

## Alternative: SuPreM UNet

Uses the Models Genesis 3D UNet architecture (1→64→128→256→512). Different channel
progression from our model (1→32→64→128→256→512), so we'd need to rebuild our UNet
to match. More parameters but also more pre-trained capacity.

---

## Detailed Architecture Analysis

### Our Current UNet3D (22.6M params)
```
Encoder: 1→32→64→128→256 (4 levels, ConvBlock = 2x [Conv3d→Norm→ReLU])
Bottleneck: 256→512
Decoder: 512→256→128→64→32 (4 levels with skip connections)
Output: 32→1 (1x1x1 conv)
Normalization: BatchNorm (Run 1) / GroupNorm (Run 2)
```

### Models Genesis / SuPreM UNet3D (19.1M params)
```
Encoder: 1→64→128→256→512 (4 levels, no separate bottleneck)
Decoder: 512→256→128→64 (3 levels with skip connections)
Output: 64→n_class (1x1x1 conv with sigmoid)
Normalization: ContBatchNorm3d (continuous batch norm)
```
- GitHub: https://github.com/MrGiovanni/ModelsGenesis
- Architecture in: `pytorch/unet3d.py`
- **NOT compatible** with our model — different channel dims at every level

### SuPreM SegResNet (4.7M params, MONAI)
```
init_filters=16, channels: 16→32→64→128
blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1]
Uses residual blocks with GroupNorm
in_channels=1, out_channels=configurable
```
- Part of MONAI framework (`monai.networks.nets.SegResNet`)
- Clean API, well-tested, actively maintained
- Weights on HuggingFace: `supervised_suprem_segresnet_2100.pth` (56.5 MB)

### MONAI SwinUNETR (62.2M params)
```
3D Swin Transformer encoder + CNN decoder
Pre-trained on 5,050 CT scans (self-supervised)
feature_size=48 default, depths=(2,2,2,2)
```
- **VRAM problem:** OOMs at 128^3 bs=1 on 12GB GPU
- Users report max comfortable patch is 96^3
- Would not fit on Kaggle P100 (16GB) for inference
- **Not recommended for our setup**

---

## How to Load SuPreM SegResNet Weights

### 1. Download weights
```bash
# From HuggingFace (56.5 MB)
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_segresnet_2100.pth \
  -O pretrained_weights/supervised_suprem_segresnet_2100.pth
```

### 2. Create model and load weights
```python
from monai.networks.nets import SegResNet
import torch

# Match SuPreM's exact config
model = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,        # binary segmentation
    init_filters=16,
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    dropout_prob=0.0,
)

# Load SuPreM weights (skip final conv since out_channels differs)
checkpoint = torch.load("pretrained_weights/supervised_suprem_segresnet_2100.pth")
store_dict = model.state_dict()
for key in checkpoint.keys():
    # SuPreM saves with 1-level prefix to strip
    new_key = '.'.join(key.split('.')[1:])
    if new_key in store_dict.keys() and 'conv_final.2.conv' not in new_key:
        store_dict[new_key] = checkpoint[key]
model.load_state_dict(store_dict)
```

### 3. Key notes
- SuPreM was trained with `out_channels=25` (25 organ classes), so the final conv layer
  (`conv_final.2.conv`) will NOT match our `out_channels=1`. That layer is excluded from
  loading and will be randomly initialized — the rest of the encoder/decoder is pre-trained.
- Input must be single-channel float32, values normalized (we use [0,1])
- Output is raw logits (no sigmoid), same as our current setup

---

## How to Load SuPreM UNet Weights

### 1. Download weights
```bash
# From HuggingFace (233 MB)
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_unet_2100.pth \
  -O pretrained_weights/supervised_suprem_unet_2100.pth
```

### 2. Requires matching architecture
Must use the Models Genesis UNet3D architecture (1→64→128→256→512) from
`https://github.com/MrGiovanni/ModelsGenesis/blob/master/pytorch/unet3d.py`

```python
# Weight loading (from SuPreM's code):
model = UNet3D(n_class=1)
checkpoint = torch.load("pretrained_weights/supervised_suprem_unet_2100.pth")['net']
store_dict = model.state_dict()
for key in checkpoint.keys():
    new_key = '.'.join(key.split('.')[2:])  # strip 2-level prefix
    if new_key in store_dict.keys():
        store_dict[new_key] = checkpoint[key]
model.load_state_dict(store_dict)
```

### 3. Architecture incompatibility with our model
Our UNet: features=(32, 64, 128, 256), bottleneck=512
Genesis UNet: features=(64, 128, 256, 512), no separate bottleneck

Would need to either:
- Rewrite our UNet to match Genesis architecture
- Or copy the `unet3d.py` from Models Genesis repo directly

---

## Models Genesis (Original) Weights

Older, less data (623 CT scans, self-supervised only).
- GitHub: https://github.com/MrGiovanni/ModelsGenesis
- Weights: `Genesis_Chest_CT.pt` (requires form submission to download)
- Architecture: same as SuPreM UNet (1→64→128→256→512)
- Less recommended than SuPreM (less data, self-supervised vs supervised)

---

## VRAM Measurements (local, RTX 5090)

| Model | Config | Inference VRAM | Training VRAM |
|-------|--------|---------------|---------------|
| Our UNet3D (22.6M) | 128^3 bs=2 | 8.1 GB | 11.0 GB |
| MONAI SegResNet (4.7M) | 128^3 bs=2 | 3.5 GB | TBD* |
| MONAI SegResNet (18.8M) | 128^3 bs=2 | TBD | TBD* |

*Training VRAM tests blocked by Run 2 using GPU. Will measure when free.

---

## Implementation Plan

### Run 3: SuPreM SegResNet (recommended next)
1. Download `supervised_suprem_segresnet_2100.pth` from HuggingFace
2. Create `vesuvius_train_v3.ipynb` with:
   - MONAI SegResNet replacing our UNet3D
   - SuPreM weight loading (skip final conv)
   - Same data pipeline, loss, training loop
   - May try larger patches (160^3) since model is much smaller
3. Fine-tune with our masked BCE+Dice loss
4. Re-run lr_find() (pre-trained model will want a lower LR)
5. Submit to Kaggle

### Run 4: SuPreM UNet (if SegResNet doesn't beat UNet)
1. Download `supervised_suprem_unet_2100.pth`
2. Copy Models Genesis `unet3d.py` architecture
3. Load SuPreM weights, fine-tune
4. Compare against SegResNet results

---

## Other Models Investigated

### Med3D / MedicalNet (Tencent)
- Pre-trained 3D ResNets (ResNet-10/18/34/50) on 23 datasets
- GitHub: https://github.com/Tencent/MedicalNet
- HuggingFace: https://huggingface.co/TencentMedicalNet/
- Would need ResNet→UNet adapter (encoder only, need custom decoder)
- More engineering work than SuPreM options

### SAM-Med3D
- 3D version of Segment Anything Model
- GitHub: https://github.com/uni-medical/SAM-Med3D
- Prompt-based segmentation (different paradigm)
- Overkill for our binary segmentation task

### SuPreM's Other Weights
- `self_supervised_models_genesis_unet_620.pt` (153 MB) — Models Genesis retrained
- `supervised_clip_driven_universal_unet_2100.pth` (77.8 MB) — CLIP-driven
- `supervised_dodnet_unet_920.pth` (69.3 MB) — DODNet
- `self_supervised_nv_swin_unetr_5050.pt` (256 MB) — NVIDIA self-supervised
- All use different architectures or training approaches

---

## References
- SuPreM paper: https://arxiv.org/abs/2310.16754 (ICLR 2024 Oral)
- Models Genesis paper: https://arxiv.org/abs/1908.06912 (MICCAI 2019)
- SegResNet: https://docs.monai.io/en/stable/networks.html#segresnet
- SuPreM weights: https://huggingface.co/MrGiovanni/SuPreM
