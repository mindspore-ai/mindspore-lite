#!/usr/bin/env python3
"""Export an HPT-style (Heterogeneous Pre-trained Transformer) policy to ONNX.

HPT (Lirui Wang et al., "Scaling Proprioceptive-Visual-Learning with
Heterogeneous Pre-trained Transformers") factorizes a robot policy into:
  - a per-embodiment **stem**  : observation -> shared latent tokens
  - a shared **trunk**           : transformer over latent tokens
  - a per-embodiment **head**  : latent tokens -> action chunk

This script exports the single forward pass:

  - input  : observation   [1, obs_dim]                float32
  - output : action        [1, horizon, action_dim]    float32

The deployed graph bundles stem+trunk+head for a single embodiment. To deploy
across embodiments, swap stem/head weights (trunk is shared). ``--random-init``
runs the pipeline end-to-end; load real HPT weights via ``--checkpoint``
(task-2: confirm the heterogeneous stem/head key layout).
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
    """Pre-norm transformer block (self-attention + MLP)."""

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


class HPTPolicy(nn.Module):
    """Stem + shared trunk + head; observation -> action chunk (single forward)."""

    def __init__(self, obs_dim, action_dim, horizon, dim=256, depth=4, heads=4):
        super().__init__()
        self.horizon = int(horizon)
        # Per-embodiment stem: observation -> latent seed.
        self.stem = nn.Sequential(nn.Linear(obs_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        # Learnable latent query tokens (one per action step).
        self.latent_query = nn.Parameter(torch.zeros(1, horizon, dim))
        # Shared trunk.
        self.trunk = nn.Sequential(*[TransformerBlock(dim, heads) for _ in range(depth)])
        # Per-embodiment head.
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, action_dim))
        nn.init.trunc_normal_(self.latent_query, std=0.02)

    def forward(self, observation):
        """observation [B, obs_dim] -> action [B, horizon, action_dim]."""
        bsz = observation.shape[0]
        seed = self.stem(observation)[:, None, :]
        tokens = self.latent_query.expand(bsz, -1, -1) + seed
        tokens = self.trunk(tokens)
        return self.head(tokens)


def _parse_args():
    p = argparse.ArgumentParser(description="Export an HPT-style policy to ONNX.")
    p.add_argument("--output-dir", type=str, default="./hpt_onnx")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--checkpoint", type=str, default="", help="HPT state_dict (task-2 confirm stem/head layout).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs-dim", type=int, default=30)
    p.add_argument("--action-dim", type=int, default=14)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    return p.parse_args()


def main():
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = HPTPolicy(args.obs_dim, args.action_dim, args.horizon, args.dim, args.depth, args.heads)

    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu")
        sd = sd.get("model_state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"random-init demo HPT policy (seed={args.seed}). Use --checkpoint for real HPT weights.")

    model = model.to(args.device).eval()
    observation = torch.randn(1, args.obs_dim, device=args.device)
    onnx_path = out / "hpt_policy.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model, (observation,), str(onnx_path),
            input_names=["observation"], output_names=["action"],
            opset_version=int(args.opset), do_constant_folding=False, dynamo=False)

    n_params = sum(p.numel() for p in model.parameters())
    print("Export complete.")
    print(f"ONNX saved to: {onnx_path}")
    print(f"  params={n_params/1e6:.1f}M  obs_dim={args.obs_dim} action_dim={args.action_dim} "
          f"horizon={args.horizon} dim={args.dim} depth={args.depth}")


if __name__ == "__main__":
    main()
