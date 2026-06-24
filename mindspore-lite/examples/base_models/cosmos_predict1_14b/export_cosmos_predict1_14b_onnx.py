#!/usr/bin/env python3
"""Export a Cosmos-Predict1-style video diffusion DiT (latent denoiser) to ONNX.

Cosmos-Predict1 (NVIDIA) is a world model that diffuses over a *latent* of a
video clip with a DiT, plus a VAE encoder/decoder. This script exports the
**single latent-denoising step** (a DiT, like RDT but over video-latent tokens):

  - inputs : noisy_latent  [1, num_tokens, latent_dim]   float32
             timestep      [1]                            int64
             cond          [1, cond_dim]                  float32   (text/context)
  - output : noise         [1, num_tokens, latent_dim]   float32

The host side runs the DDPM loop (numpy, see infer_*). A VAE is required to map
pixels<->latent (out of scope for the single DiT step). 14B fp16 ~= 28GB+ on
300I Duo is at high OOM risk (task-2: split sub-graphs / lower resolution).
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


def _modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


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
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        # nn.init.zeros_(self.adaLN[1].weight)
        # nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)
        h = _modulate(self.norm1(x), s1.unsqueeze(1), sc1.unsqueeze(1))
        a = self.attn(h)
        x = x + g1.unsqueeze(1) * a
        h = _modulate(self.norm2(x), s2.unsqueeze(1), sc2.unsqueeze(1))
        x = x + g2.unsqueeze(1) * self.mlp(h)
        return x


class CosmosDenoise(nn.Module):
    """Cosmos latent DiT denoiser: (noisy_latent, timestep, cond) -> noise."""

    def __init__(self, num_tokens=256, latent_dim=16, cond_dim=256, dim=256, depth=6, heads=4):
        super().__init__()
        self.dim = int(dim)
        self.latent_in = nn.Linear(latent_dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.cond_proj = nn.Linear(cond_dim, dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.norm_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_f = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.head = nn.Linear(dim, latent_dim)
        # nn.init.zeros_(self.adaLN_f[1].weight)
        # nn.init.zeros_(self.adaLN_f[1].bias)
        # nn.init.zeros_(self.head.weight)
        # nn.init.zeros_(self.head.bias)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, noisy_latent, timestep, cond):
        c = self.t_mlp(sinusoidal_embedding(timestep, self.dim)) + self.cond_proj(cond)
        x = self.latent_in(noisy_latent) + self.pos
        for blk in self.blocks:
            x = blk(x, c)
        shift, scale = self.adaLN_f(c).chunk(2, dim=-1)
        x = _modulate(self.norm_f(x), shift.unsqueeze(1), scale.unsqueeze(1))
        return self.head(x)


def _parse_args():
    p = argparse.ArgumentParser(description="Export a Cosmos-Predict1 latent DiT denoiser step to ONNX.")
    p.add_argument("--output-dir", type=str, default="./cosmos_predict1_14b_onnx")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--checkpoint", type=str, default="", help="Cosmos state_dict (task-2: NVIDIA package + VAE).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-tokens", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--cond-dim", type=int, default=256)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    return p.parse_args()


def main():
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = CosmosDenoise(args.num_tokens, args.latent_dim, args.cond_dim, args.dim, args.depth, args.heads)

    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu")
        sd = sd.get("model_state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"random-init demo Cosmos latent DiT (seed={args.seed}). "
              f"Real Cosmos-Predict1 ~14B + VAE (task-2).")

    model = model.to(args.device).eval()
    noisy_latent = torch.randn(1, args.num_tokens, args.latent_dim, device=args.device)
    timestep = torch.zeros(1, dtype=torch.int64, device=args.device)
    cond = torch.randn(1, args.cond_dim, device=args.device)
    onnx_path = out / "cosmos_denoise.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model, (noisy_latent, timestep, cond), str(onnx_path),
            input_names=["noisy_latent", "timestep", "cond"], output_names=["noise"],
            opset_version=int(args.opset), do_constant_folding=False, dynamo=False)

    n_params = sum(p.numel() for p in model.parameters())
    print("Export complete.")
    print(f"ONNX saved to: {onnx_path}")
    print(f"  params={n_params/1e6:.1f}M  num_tokens={args.num_tokens} latent_dim={args.latent_dim}")


if __name__ == "__main__":
    main()
