#!/usr/bin/env python3
"""Export a Diffusion Policy single-step denoiser (ConditionalUnet1D) to ONNX.

Diffusion Policy (Chi et al., "Diffusion Policy: Visuomotor Policy Learning via
Action Diffusion", real-stanford/diffusion_policy) predicts an *action chunk*
via iterative denoising with a FiLM-conditioned 1D U-Net. This script exports
the **single denoising step**:

  - inputs : noisy_action  [1, action_dim, horizon]  float32
             timestep      [1]                        int64
             obs           [1, obs_dim]               float32   (low-dim state)
  - output : noise         [1, action_dim, horizon]  float32   (predicted eps)

The host side drives the DDPM loop in numpy (see infer_*). Default shape matches
the PushT low-dim example (action_dim=2, obs_dim=2, horizon=16).

A checkpoint from diffusion_policy training (``*.ckpt`` containing the
``ema_unet`` / ``model`` state_dict) can be loaded with ``--checkpoint``; key
names are matched loosely, and task-2 should confirm against the trained config.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embedding (diffusers convention)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / max(half, 1))
    args = timesteps.to(torch.float32)[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLMBlock(nn.Module):
    """1D conv residual block with FiLM (scale+bias) conditioning."""

    def __init__(self, in_ch, out_ch, cond_dim, groups=8):
        super().__init__()
        g1 = min(groups, in_ch)
        g2 = min(groups, out_ch)
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2)
        self.cond = nn.Linear(cond_dim, 2 * out_ch)
        self.act = nn.Mish()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        """x [B,C,T], cond [B,cond_dim] -> [B,out_ch,T]."""
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.cond(cond).chunk(2, dim=-1)
        h = h * (1.0 + scale[:, :, None]) + shift[:, :, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return self.skip(x) + h


class Downsample(nn.Module):
    """Stride-2 1D downsample."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Stride-2 1D upsample (transposed conv)."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ConditionalUnet1D(nn.Module):
    """FiLM-conditioned 1D U-Net for one diffusion denoising step."""

    def __init__(self, action_dim, obs_dim, emb_dim=256, down_dims=(256, 512, 1024), groups=8):
        super().__init__()
        self.emb_dim = int(emb_dim)
        cond_dim = self.emb_dim + int(obs_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim), nn.Mish(), nn.Linear(self.emb_dim, self.emb_dim))

        d = list(down_dims)
        in_dims = [action_dim] + d[:-1]
        self.enc_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i, out in enumerate(d):
            self.enc_blocks.append(nn.ModuleList([
                FiLMBlock(in_dims[i], out, cond_dim, groups),
                FiLMBlock(out, out, cond_dim, groups),
            ]))
            self.down_samples.append(Downsample(out) if i < len(d) - 1 else nn.Identity())

        self.mid_blocks = nn.ModuleList([
            FiLMBlock(d[-1], d[-1], cond_dim, groups),
            FiLMBlock(d[-1], d[-1], cond_dim, groups),
        ])

        rev = list(reversed(d))
        prev_dims = [d[-1]] + rev[:-1]
        self.up_samples = nn.ModuleList([nn.Identity()] + [Upsample(rev[i]) for i in range(len(rev) - 1)])
        self.dec_blocks = nn.ModuleList()
        for i, out in enumerate(rev):
            self.dec_blocks.append(nn.ModuleList([
                FiLMBlock(prev_dims[i] + rev[i], out, cond_dim, groups),
                FiLMBlock(out, out, cond_dim, groups),
            ]))
        self.final = nn.Conv1d(rev[-1], action_dim, 1)

    def forward(self, noisy_action, timestep, obs):
        """Return predicted noise; layout [B, action_dim, horizon]."""
        temb = self.time_mlp(timestep_embedding(timestep, self.emb_dim))
        cond = torch.cat([temb, obs], dim=-1)
        x = noisy_action
        skips = []
        for blocks, ds in zip(self.enc_blocks, self.down_samples):
            for blk in blocks:
                x = blk(x, cond)
            skips.append(x)
            x = ds(x)
        for blk in self.mid_blocks:
            x = blk(x, cond)
        for blocks, us in zip(self.dec_blocks, self.up_samples):
            x = us(x)
            x = torch.cat([x, skips.pop()], dim=1)
            for blk in blocks:
                x = blk(x, cond)
        return self.final(x)


def _parse_args():
    p = argparse.ArgumentParser(description="Export a Diffusion Policy ConditionalUnet1D step to ONNX.")
    p.add_argument("--output-dir", type=str, default="./diffusion_policy_onnx")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--checkpoint", type=str, default="",
                   help="diffusion_policy .ckpt (ema_unet/model); task-2 confirm key names.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-dim", type=int, default=2)
    p.add_argument("--obs-dim", type=int, default=2)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--emb-dim", type=int, default=256)
    p.add_argument("--down-dims", type=str, default="256,512,1024")
    return p.parse_args()


def main():
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    down_dims = tuple(int(x) for x in args.down_dims.split(",") if x.strip())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = ConditionalUnet1D(args.action_dim, args.obs_dim, args.emb_dim, down_dims)

    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu")
        sd = sd.get("model_state_dict", sd)
        if "ema_unet" in sd:
            sd = {k.replace("ema_unet.", ""): v for k, v in sd["ema_unet"].items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"random-init demo ConditionalUnet1D (seed={args.seed}). "
              f"Use --checkpoint for a trained diffusion_policy model.")

    model = model.to(args.device).eval()

    noisy_action = torch.randn(1, args.action_dim, args.horizon, device=args.device)
    timestep = torch.zeros(1, dtype=torch.int64, device=args.device)
    obs = torch.randn(1, args.obs_dim, device=args.device)
    onnx_path = out / "diffusion_policy.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model, (noisy_action, timestep, obs), str(onnx_path),
            input_names=["noisy_action", "timestep", "obs"],
            output_names=["noise"],
            opset_version=int(args.opset),
            do_constant_folding=False,
            dynamo=False,
        )

    n_params = sum(p.numel() for p in model.parameters())
    print("Export complete.")
    print(f"ONNX saved to: {onnx_path}")
    print(f"  params={n_params/1e6:.1f}M  action_dim={args.action_dim} obs_dim={args.obs_dim} "
          f"horizon={args.horizon} down_dims={down_dims}")


if __name__ == "__main__":
    main()
