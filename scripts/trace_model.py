"""
Trace SegResNet model(s) for Kaggle submission.
Traced models load with just torch.jit.load — no MONAI dependency needed.

Supports:
  - Plain SegResNet (v9, v12): out_channels=1
  - 3-class SegResNet (v13): out_channels=3
  - SegResNetDSAttn (v10, v11): out_channels=1, attention gates + deep supervision

Usage:
  # Trace v12 (plain SegResNet):
  python scripts/trace_model.py --checkpoint checkpoints/models/best_segresnet_v12.pth \
                                --output kaggle/kaggle_weights/best_segresnet_v12_traced.pt

  # Trace v13 (3-class):
  python scripts/trace_model.py --checkpoint checkpoints/models/best_segresnet_v13.pth \
                                --output kaggle/kaggle_weights/best_segresnet_v13_traced.pt \
                                --out-channels 3

  # Trace v10 (attention gates + deep supervision):
  python scripts/trace_model.py --checkpoint checkpoints/models/best_segresnet_v10.pth \
                                --output kaggle/kaggle_weights/best_segresnet_v10_traced.pt \
                                --model-type dsattn
"""
import argparse
import torch
import torch.nn as nn
from monai.networks.nets import SegResNet
from pathlib import Path


class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv3d(F_g, F_int, kernel_size=1, bias=True)
        self.W_x = nn.Conv3d(F_l, F_int, kernel_size=1, bias=True)
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class SegResNetDSAttn(SegResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        filters = [self.init_filters * (2 ** i) for i in range(len(self.blocks_down))]
        up_channels = list(reversed(filters[:-1]))
        self.attention_gates = nn.ModuleList([
            AttentionGate3D(F_g=ch, F_l=ch, F_int=max(ch // 2, 4))
            for ch in up_channels
        ])
        self.ds_heads = nn.ModuleList([
            nn.Conv3d(ch, 1, kernel_size=1)
            for ch in up_channels[:-1]
        ])

    def decode(self, x, down_x, return_intermediates=False):
        intermediates = []
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x_up = up(x)
            skip = down_x[i + 1]
            skip = self.attention_gates[i](g=x_up, x=skip)
            x = x_up + skip
            x = upl(x)
            if return_intermediates and i < len(self.ds_heads):
                intermediates.append(self.ds_heads[i](x))
        if self.use_conv_final:
            x = self.conv_final(x)
        if return_intermediates:
            return x, intermediates
        return x

    def forward(self, x):
        x, down_x = self.encode(x)
        down_x.reverse()
        if self.training:
            main, aux = self.decode(x, down_x, return_intermediates=True)
            return [main] + aux
        else:
            return self.decode(x, down_x)


def create_model(model_type="plain", out_channels=1):
    if model_type == "dsattn":
        return SegResNetDSAttn(
            spatial_dims=3, in_channels=1, out_channels=out_channels, init_filters=16,
            blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], dropout_prob=0.2,
        )
    else:
        return SegResNet(
            spatial_dims=3, in_channels=1, out_channels=out_channels, init_filters=16,
            blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], dropout_prob=0.2,
        )


def trace_checkpoint(ckpt_path, output_path, model_type="plain", out_channels=1, device="cpu"):
    model = create_model(model_type, out_channels)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 1, 160, 160, 160)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"  Traced: {ckpt_path} -> {output_path} ({size_mb:.1f} MB, {out_channels}ch)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--output", nargs="+", required=True)
    parser.add_argument("--model-type", default="plain", choices=["plain", "dsattn"],
                        help="Model architecture (plain=SegResNet, dsattn=SegResNetDSAttn)")
    parser.add_argument("--out-channels", type=int, default=1,
                        help="Number of output channels (1=binary, 3=3-class)")
    args = parser.parse_args()

    assert len(args.checkpoint) == len(args.output), "Must have same number of checkpoints and outputs"

    for ckpt, out in zip(args.checkpoint, args.output):
        trace_checkpoint(ckpt, out, args.model_type, args.out_channels)

    print(f"\nDone: {len(args.checkpoint)} model(s) traced.")
