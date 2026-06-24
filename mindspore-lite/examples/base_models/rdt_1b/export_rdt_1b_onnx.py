#!/usr/bin/env python3
"""Export an RDT-1B-style diffusion Transformer (DiT) single-step denoiser to ONNX.

RDT-1B (Tsinghua, "RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation")
predicts a bimanual *action chunk* via an adaLN-zero Diffusion Transformer
conditioned on (vision + language + proprio) embeddings.

This script provides a self-contained PyTorch DiT implementing the single
denoising step:

  - inputs : noisy_action  [1, horizon, action_dim]   float32   (action_dim=14: 7+7 arms)
             timestep      [1]                         int64
             cond          [1, cond_dim]               float32   (pooled vis+lang+proprio)
  - output : noise         [1, horizon, action_dim]   float32

The host side drives the DDPM loop (numpy, see infer_*). Load real RDT weights
with ``--checkpoint`` (task-2: confirm the official ``RDT-model`` key layout).
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
        """x [B,N,dim], c [B,dim] -> [B,N,dim]."""
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = self.adaLN(c).chunk(6, dim=-1)
        h = _modulate(self.norm1(x), s_msa.unsqueeze(1), sc_msa.unsqueeze(1))
        a = self.attn(h)
        x = x + g_msa.unsqueeze(1) * a
        h = _modulate(self.norm2(x), s_mlp.unsqueeze(1), sc_mlp.unsqueeze(1))
        x = x + g_mlp.unsqueeze(1) * self.mlp(h)
        return x


class RDTDenoise(nn.Module):
    """RDT DiT single-step denoiser: (noisy_action, timestep, cond) -> noise."""

    def __init__(self, action_dim=14, horizon=64, cond_dim=256, dim=256, depth=6, heads=4):
        super().__init__()
        self.dim = int(dim)
        self.action_in = nn.Linear(action_dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, horizon, dim))
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.cond_proj = nn.Linear(cond_dim, dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.norm_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_final = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.head = nn.Linear(dim, action_dim)
        # nn.init.zeros_(self.adaLN_final[1].weight)
        # nn.init.zeros_(self.adaLN_final[1].bias)
        # nn.init.zeros_(self.head.weight)
        # nn.init.zeros_(self.head.bias)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, noisy_action, timestep, cond):
        c = self.t_mlp(sinusoidal_embedding(timestep, self.dim)) + self.cond_proj(cond)
        x = self.action_in(noisy_action) + self.pos
        for blk in self.blocks:
            x = blk(x, c)
        shift, scale = self.adaLN_final(c).chunk(2, dim=-1)
        x = _modulate(self.norm_f(x), shift.unsqueeze(1), scale.unsqueeze(1))
        return self.head(x)


def _parse_args():
    p = argparse.ArgumentParser(description="Export an RDT-1B-style DiT denoiser step to ONNX.")
    p.add_argument("--output-dir", type=str, default="./rdt_1b_onnx")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--checkpoint", type=str, default="", help="RDT-1B state_dict (task-2 confirm layout).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-dim", type=int, default=14)
    p.add_argument("--horizon", type=int, default=64)
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
    model = RDTDenoise(args.action_dim, args.horizon, args.cond_dim, args.dim, args.depth, args.heads)

    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu")
        sd = sd.get("model_state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"random-init demo RDT DiT (seed={args.seed}). Use --checkpoint for real RDT-1B weights.")

    model = model.to(args.device).eval()
    noisy_action = torch.randn(1, args.horizon, args.action_dim, device=args.device)
    timestep = torch.zeros(1, dtype=torch.int64, device=args.device)
    cond = torch.randn(1, args.cond_dim, device=args.device)

    onnx_path = out / "rdt_1b_denoise.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model, (noisy_action, timestep, cond), str(onnx_path),
            input_names=["noisy_action", "timestep", "cond"],
            output_names=["noise"],
            opset_version=int(args.opset), do_constant_folding=False, dynamo=False)

    n_params = sum(p.numel() for p in model.parameters())
    print("Export complete.")
    print(f"ONNX saved to: {onnx_path}")
    print(f"  params={n_params/1e6:.1f}M  action_dim={args.action_dim} horizon={args.horizon} "
          f"dim={args.dim} depth={args.depth} (real RDT-1B ~1B; use --dim/depth to scale up in task-2)")


if __name__ == "__main__":
    main()
