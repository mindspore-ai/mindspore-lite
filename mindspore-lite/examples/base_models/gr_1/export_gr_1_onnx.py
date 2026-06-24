#!/usr/bin/env python3
"""Export a GR-1-style video-conditioned policy to ONNX.

GR-1 (ByteDance, "Unleashing Large-Scale Video Generative Pre-training for
Visual Robot Manipulation") pre-trains a video generative backbone (ViT over
video tokens) and fine-tunes it into a robot policy. The deployed policy maps a
short clip of observations to an action chunk.

This script provides a self-contained PyTorch regression skeleton:

  - input  : video     [1, num_frames, 3, 224, 224]   float32
  - output : action    [1, horizon, action_dim]        float32

``--random-init`` runs the pipeline. GR-1/GR-2 open-source status is uncertain;
real weights / modeling API must be confirmed in task-2 (network access).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ManualMHA(nn.Module):
    """Manual multi-head attention exporting to standard ONNX MatMul/Softmax."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = int(heads)
        self.hdim = int(dim) // int(heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.hdim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.hdim ** -0.5)
        out = attn.softmax(dim=-1) @ v
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ManualMHA(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GR1Policy(nn.Module):
    """Video -> action chunk regression skeleton (patch + temporal tokens)."""

    def __init__(self, num_frames=4, img_size=224, patch=16, dim=384, depth=4, heads=6,
                 horizon=16, action_dim=7):
        super().__init__()
        self.num_frames = int(num_frames)
        self.dim = int(dim)
        self.horizon = int(horizon)
        grid = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames * grid, dim))
        self.readout = nn.Parameter(torch.zeros(1, horizon, dim))
        self.trunk = nn.Sequential(*[TransformerBlock(dim, heads) for _ in range(depth)])
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, action_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.readout, std=0.02)

    def forward(self, video):
        """video [B, T, 3, H, W] -> action [B, horizon, action_dim]."""
        bsz, t = video.shape[0], video.shape[1]
        feats = self.patch_embed(video.reshape(bsz * t, *video.shape[2:]))  # [B*T, dim, h, w]
        ntok = feats.shape[2] * feats.shape[3]
        feats = feats.flatten(2).transpose(1, 2).reshape(bsz, t * ntok, self.dim)
        feats = feats + self.pos_embed
        readout = self.readout.expand(bsz, -1, -1)
        x = torch.cat([readout, feats], dim=1)
        x = self.trunk(x)
        return self.head(x[:, : self.horizon])


def _parse_args():
    p = argparse.ArgumentParser(description="Export a GR-1-style video policy to ONNX.")
    p.add_argument("--output-dir", type=str, default="./gr_1_onnx")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--checkpoint", type=str, default="", help="GR-1 state_dict (task-2: confirm open-source).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--action-dim", type=int, default=7)
    return p.parse_args()


def main():
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = GR1Policy(args.num_frames, args.img_size, args.patch, args.dim, args.depth, args.heads,
                      args.horizon, args.action_dim)

    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu")
        sd = sd.get("model_state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"random-init demo GR-1 video policy (seed={args.seed}). "
              f"GR-1/GR-2 open-source status to confirm in task-2.")

    model = model.to(args.device).eval()
    video = torch.randn(1, args.num_frames, 3, args.img_size, args.img_size, device=args.device)
    onnx_path = out / "gr_1_policy.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model, (video,), str(onnx_path),
            input_names=["video"], output_names=["action"],
            opset_version=int(args.opset), do_constant_folding=False, dynamo=False)

    n_params = sum(p.numel() for p in model.parameters())
    print("Export complete.")
    print(f"ONNX saved to: {onnx_path}")
    print(f"  params={n_params/1e6:.1f}M  frames={args.num_frames} action_dim={args.action_dim} horizon={args.horizon}")


if __name__ == "__main__":
    main()
