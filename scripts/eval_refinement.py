#!/usr/bin/env python3
"""
Evaluate refinement model vs hand-tuned post-processing.
Uses pre-computed probmaps — no GPU needed for baseline.
Refinement model runs on GPU for speed but could use CPU.

Usage:
    python scripts/eval_refinement.py [--n-volumes 20] [--phase 2]
"""

import argparse
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path
from scipy.ndimage import (
    binary_closing, generate_binary_structure, binary_propagation,
    label as scipy_label, zoom as scipy_zoom,
)
from topometrics.leaderboard import compute_leaderboard_score

ROOT = Path("/workspace/vesuvius-kaggle-competition")
PROBMAP_DIR = ROOT / "data" / "refinement_data" / "probmaps"
TRAIN_LBL = ROOT / "data" / "train_labels"

# Baseline post-processing parameters (proven best)
T_LOW = 0.35
T_HIGH = 0.75
CLOSING_Z_RADIUS = 2
CLOSING_XY_RADIUS = 1
DUST_MIN_SIZE = 100
METRIC_DOWNSAMPLE = 4


# ── RefinementUNet3D (must match notebook definition) ──────────

class ConvBlock3D(nn.Module):
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
    def __init__(self, in_ch=1, out_ch=1, channels=(8, 16, 32, 64), dropout=0.2):
        super().__init__()
        self.enc1 = ConvBlock3D(in_ch, channels[0])
        self.enc2 = ConvBlock3D(channels[0], channels[1])
        self.enc3 = ConvBlock3D(channels[1], channels[2])
        self.bottleneck = ConvBlock3D(channels[2], channels[3])
        self.dropout = nn.Dropout3d(p=dropout)
        self.dec3 = ConvBlock3D(channels[3] + channels[2], channels[2])
        self.dec2 = ConvBlock3D(channels[2] + channels[1], channels[1])
        self.dec1 = ConvBlock3D(channels[1] + channels[0], channels[0])
        self.conv_final = nn.Conv3d(channels[0], out_ch, 1)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        b = self.dropout(b)
        d3 = F.interpolate(b, size=e3.shape[2:], mode='trilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.conv_final(d1)


# ── Baseline post-processing ──────────────────────────────────

def hysteresis_threshold(prob, t_low, t_high):
    strong = prob >= t_high
    weak = prob >= t_low
    struct = generate_binary_structure(3, 3)
    return binary_propagation(strong, structure=struct, mask=weak).astype(np.uint8)


def build_anisotropic_struct(z_radius=2, xy_radius=1):
    sz = 2 * z_radius + 1
    sxy = 2 * xy_radius + 1
    struct = np.zeros((sz, sxy, sxy), dtype=bool)
    cy, cx = xy_radius, xy_radius
    for y in range(sxy):
        for x in range(sxy):
            if (y - cy) ** 2 + (x - cx) ** 2 <= xy_radius ** 2:
                struct[:, y, x] = True
    return struct


STRUCT = build_anisotropic_struct(CLOSING_Z_RADIUS, CLOSING_XY_RADIUS)


def postprocess_baseline(prob):
    """Hand-tuned: hysteresis + anisotropic closing + dust removal."""
    binary = hysteresis_threshold(prob, T_LOW, T_HIGH)
    closed = binary_closing(binary, structure=STRUCT)
    labeled, n = scipy_label(closed)
    if n > 0:
        sizes = np.bincount(labeled.ravel())
        small = sizes < DUST_MIN_SIZE
        small[0] = False
        closed[small[labeled]] = 0
    return closed.astype(np.uint8)


def postprocess_refinement(prob, model, device):
    """Refinement model: probmap → model → sigmoid → threshold 0.5."""
    with torch.no_grad():
        inp = torch.from_numpy(prob.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        logits = model(inp)
        pred = (torch.sigmoid(logits).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    return pred


def postprocess_refinement_then_baseline(prob, model, device):
    """Option A: probmap → model → sigmoid → baseline post-processing."""
    with torch.no_grad():
        inp = torch.from_numpy(prob.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        logits = model(inp)
        refined_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    return postprocess_baseline(refined_prob)


# ── Scoring ───────────────────────────────────────────────────

def score(pred, lbl, ds=METRIC_DOWNSAMPLE):
    scale = 1.0 / ds
    pred_ds = scipy_zoom(pred, scale, order=0).astype(np.uint8)
    lbl_ds = scipy_zoom(lbl, scale, order=0).astype(np.uint8)
    report = compute_leaderboard_score(
        pred_ds, lbl_ds, ignore_label=2, spacing=(1, 1, 1),
        surface_tolerance=2.0, voi_alpha=0.3, combine_weights=(0.3, 0.35, 0.35),
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-volumes", type=int, default=20,
                        help="Number of val volumes to evaluate")
    parser.add_argument("--phase", type=int, default=2, choices=[1, 2, 3],
                        help="Which refinement phase checkpoint to load")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to specific checkpoint (overrides --phase)")
    parser.add_argument("--device", default="cuda",
                        help="Device for refinement model")
    args = parser.parse_args()

    # Load refinement model
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = ROOT / "checkpoints" / "models" / f"best_refinement_phase{args.phase}.pth"
    print(f"Loading refinement model: {ckpt_path}")
    model = RefinementUNet3D()
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} params, device={device}")

    # Get val volumes (scroll 26002)
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    probmap_ids = set(int(p.stem) for p in PROBMAP_DIR.glob("*.npy"))
    label_ids = set(int(p.stem) for p in TRAIN_LBL.glob("*.tif"))
    available = probmap_ids & label_ids

    val_df = train_df[(train_df.scroll_id == 26002) & (train_df.id.isin(available))]
    eval_ids = val_df.id.tolist()[:args.n_volumes]
    print(f"Evaluating on {len(eval_ids)} val volumes (scroll 26002)\n")

    # Evaluate
    results = []
    t_start = time.time()
    for i, vid in enumerate(eval_ids):
        prob = np.load(PROBMAP_DIR / f"{vid}.npy")
        lbl = tifffile.imread(TRAIN_LBL / f"{vid}.tif")

        # Baseline
        t0 = time.time()
        pred_base = postprocess_baseline(prob)
        t_base = time.time() - t0

        # Refinement (raw threshold)
        t0 = time.time()
        pred_ref = postprocess_refinement(prob, model, device)
        t_ref = time.time() - t0

        # Refinement + baseline post-processing (Option A)
        t0 = time.time()
        pred_ref_pp = postprocess_refinement_then_baseline(prob, model, device)
        t_ref_pp = time.time() - t0

        # Score all three
        report_base = score(pred_base, lbl)
        report_ref = score(pred_ref, lbl)
        report_ref_pp = score(pred_ref_pp, lbl)

        delta = report_ref.score - report_base.score
        delta_pp = report_ref_pp.score - report_base.score
        results.append({
            "vol_id": vid,
            "baseline": report_base.score,
            "refined": report_ref.score,
            "refined_pp": report_ref_pp.score,
            "delta": delta,
            "delta_pp": delta_pp,
            "base_topo": report_base.topo.toposcore,
            "ref_topo": report_ref.topo.toposcore,
            "ref_pp_topo": report_ref_pp.topo.toposcore,
            "base_sdice": report_base.surface_dice,
            "ref_sdice": report_ref.surface_dice,
            "ref_pp_sdice": report_ref_pp.surface_dice,
            "base_voi": report_base.voi.voi_score,
            "ref_voi": report_ref.voi.voi_score,
            "ref_pp_voi": report_ref_pp.voi.voi_score,
        })
        m1 = "+" if delta > 0 else "-" if delta < 0 else "="
        m2 = "+" if delta_pp > 0 else "-" if delta_pp < 0 else "="
        print(f"  [{i+1}/{len(eval_ids)}] {vid}: "
              f"base={report_base.score:.4f} ref={report_ref.score:.4f}({m1}) "
              f"ref+pp={report_ref_pp.score:.4f}({m2})")

    elapsed = time.time() - t_start
    df = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print(f"HEAD-TO-HEAD RESULTS ({len(eval_ids)} val volumes, {elapsed:.0f}s total)")
    print(f"{'='*70}")
    print(f"  Baseline (hand-tuned):     {df.baseline.mean():.4f} mean comp_score")
    print(f"  Refinement (phase {args.phase}):     {df.refined.mean():.4f} mean comp_score  "
          f"delta={df.delta.mean():+.4f}  W/L={(df.delta > 0).sum()}/{(df.delta < 0).sum()}")
    print(f"  Refinement + postproc (A): {df.refined_pp.mean():.4f} mean comp_score  "
          f"delta={df.delta_pp.mean():+.4f}  W/L={(df.delta_pp > 0).sum()}/{(df.delta_pp < 0).sum()}")

    print(f"\n  Component breakdown (mean):")
    print(f"    {'':20s} {'Baseline':>10s} {'Refined':>10s} {'Ref+PP':>10s}")
    print(f"    {'TopoScore':20s} {df.base_topo.mean():10.4f} {df.ref_topo.mean():10.4f} {df.ref_pp_topo.mean():10.4f}")
    print(f"    {'SurfaceDice':20s} {df.base_sdice.mean():10.4f} {df.ref_sdice.mean():10.4f} {df.ref_pp_sdice.mean():10.4f}")
    print(f"    {'VOI':20s} {df.base_voi.mean():10.4f} {df.ref_voi.mean():10.4f} {df.ref_pp_voi.mean():10.4f}")

    best_method = "Refinement+PP" if df.delta_pp.mean() > df.delta.mean() else "Refinement"
    best_delta = max(df.delta.mean(), df.delta_pp.mean())
    if best_delta > 0:
        print(f"\n>>> PASS: {best_method} beats baseline by {best_delta:+.4f} <<<")
    else:
        print(f"\n>>> FAIL: Baseline wins (best delta: {best_delta:+.4f}) <<<")

    # Save detailed results
    out = ROOT / "logs" / "refinement_eval.csv"
    df.to_csv(out, index=False)
    print(f"\nDetailed results: {out}")


if __name__ == "__main__":
    main()
