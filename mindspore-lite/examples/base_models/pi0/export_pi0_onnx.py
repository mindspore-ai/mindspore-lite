#!/usr/bin/env python3
"""Export a pi0-style Flow Matching action policy (velocity net) to ONNX.

pi0 / pi0-FAST (Physical Intelligence, "openpi") = PaliGemma VLM backbone +
flow-matching action expert predicting action chunks via an ODE. π0.5 is already
adapted (skipped here); this covers π0 / π0-FAST.

This script exports the **single velocity step** (same Flow Matching pattern as
the GR00T-N1 example):

  - inputs : image   [1, 3, 224, 224]         float32
             x_t     [1, horizon, action_dim]  float32
             t       [1]                       float32   (flow time in [0,1])
  - output : velocity [1, horizon, action_dim] float32

Euler ODE sampling runs host-side. Real π0 weights via the official ``openpi``
package in task-2.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


def sinusoidal_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / max(half, 1))
    args = timesteps.to(torch.float32)[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


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


class DiTBlock(nn.Module):
    """Transformer block with adaLN-zero conditioning."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = ManualMHA(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        # nn.init.zeros_(self.adaLN[1].weight)
        # nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + sc1.unsqueeze(1)) + s1.unsqueeze(1)
        a = self.attn(h)
        x = x + g1.unsqueeze(1) * a
        h = self.norm2(x) * (1 + sc2.unsqueeze(1)) + s2.unsqueeze(1)
        x = x + g2.unsqueeze(1) * self.mlp(h)
        return x


class ImageTokenizer(nn.Module):
    """Patchify image + a few transformer layers -> image tokens."""

    def __init__(self, img_size=224, patch=16, dim=384, depth=2, heads=6):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch) ** 2, dim))
        self.blocks = nn.Sequential(*[
            nn.Sequential(nn.LayerNorm(dim), ManualMHA(dim, heads)) for _ in range(depth)])

    def forward(self, image):
        x = self.patch_embed(image).flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.blocks:
            x = x + blk[1](blk[0](x))
        return x


class Pi0VelocityNet(nn.Module):
    """pi0 Flow Matching velocity net: (image, x_t, t) -> velocity."""

    def __init__(self, img_size=224, patch=16, dim=384, depth=4, heads=6, horizon=16, action_dim=7):
        super().__init__()
        self.dim = int(dim)
        self.horizon = int(horizon)
        self.img_tok = ImageTokenizer(img_size, patch, dim, depth=2, heads=heads)
        self.action_in = nn.Linear(action_dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, horizon, dim))
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.norm_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_f = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.head = nn.Linear(dim, action_dim)
        # nn.init.zeros_(self.adaLN_f[1].weight)
        # nn.init.zeros_(self.adaLN_f[1].bias)
        # nn.init.zeros_(self.head.weight)
        # nn.init.zeros_(self.head.bias)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, image, x_t, t):
        img = self.img_tok(image)
        c = self.t_mlp(sinusoidal_embedding(t * 1000.0, self.dim))
        act = self.action_in(x_t) + self.pos
        x = torch.cat([act, img], dim=1)
        for blk in self.blocks:
            x = blk(x, c)
        s, sc = self.adaLN_f(c).chunk(2, dim=-1)
        x = self.norm_f(x) * (1 + sc.unsqueeze(1)) + s.unsqueeze(1)
        return self.head(x[:, : self.horizon])


def _parse_args():
    p = argparse.ArgumentParser(description="Export a pi0 Flow Matching velocity step to ONNX.")
    p.add_argument("--output-dir", type=str, default="./pi0_onnx")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--checkpoint", type=str, default="", help="pi0 state_dict (task-2: openpi package).")
    p.add_argument("--seed", type=int, default=0)
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
    model = Pi0VelocityNet(args.img_size, args.patch, args.dim, args.depth, args.heads,
                           args.horizon, args.action_dim)

    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu")
        sd = sd.get("model_state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"random-init demo pi0 velocity net (seed={args.seed}). "
              f"Real pi0 = PaliGemma VLM + flow-matching action expert via openpi (task-2).")

    model = model.to(args.device).eval()
    image = torch.randn(1, 3, args.img_size, args.img_size, device=args.device)
    x_t = torch.randn(1, args.horizon, args.action_dim, device=args.device)
    t = torch.tensor([0.5], dtype=torch.float32, device=args.device)
    onnx_path = out / "pi0_velocity.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model, (image, x_t, t), str(onnx_path),
            input_names=["image", "x_t", "t"], output_names=["velocity"],
            opset_version=int(args.opset), do_constant_folding=False, dynamo=False)

    n_params = sum(p.numel() for p in model.parameters())
    print("Export complete.")
    print(f"ONNX saved to: {onnx_path}")
    print(f"  params={n_params/1e6:.1f}M  action_dim={args.action_dim} horizon={args.horizon}")


if __name__ == "__main__":
    main()
