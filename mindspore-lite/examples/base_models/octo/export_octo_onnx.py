#!/usr/bin/env python3
"""Export an Octo-style generalist robot policy (single-step denoiser) to ONNX.

Octo (Berkeley, "Octo: An Open-Source Generalist Robot Policy") is a Transformer
robot policy: image tokenizer + Transformer trunk with learnable readout tokens +
a diffusion action head that predicts an *action chunk* via iterative denoising.

The official Octo release is JAX/Flax (``rail-berkeley/octo-*``). This script
provides a **self-contained PyTorch reference implementation** of the single-step
denoiser network so the full export/convert/infer/align pipeline can run. The
exported ONNX is the *single denoising step*:

  - inputs : image         [1, 3, 224, 224]   float32
             proprio       [1, proprio_dim]    float32   (robot state)
             timestep      [1]                 int32     (diffusion step index)
             noisy_action  [1, horizon, act_dim] float32
  - output : noise         [1, horizon, act_dim] float32  (predicted epsilon)

The host side drives the DDPM sampling loop (pure numpy, see infer_*). Swap in
real weights via ``--checkpoint`` once the Octo PyTorch port / flax->torch
conversion is verified (task-2 / after network access is restored).
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


def sinusoidal_embedding(timesteps, dim):
    """Sinusoidal timestep embedding, matching the diffusion-policy convention."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / max(half, 1))
    args = timesteps.to(torch.float32)[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ManualMHA(nn.Module):
    """Manual multi-head attention exporting to standard ONNX MatMul/Softmax.

    ``nn.MultiheadAttention`` traces to ``aten::_native_multi_head_attention``,
    unsupported on ONNX opset 17; this avoids that op.
    """

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
    """Pre-norm Transformer block (self-attention + MLP)."""

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


class ImageTokenizer(nn.Module):
    """Patchify the image and refine with a few Transformer layers -> image tokens."""

    def __init__(self, img_size=224, patch=16, in_ch=3, dim=384, depth=2, heads=6):
        super().__init__()
        self.num_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])

    def forward(self, image):
        """image [B, C, H, W] -> tokens [B, num_patches, dim]."""
        x = self.patch_embed(image).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return x


class OctoDenoiseNet(nn.Module):
    """Octo single-step denoiser: (image, proprio, timestep, noisy_action) -> noise."""

    def __init__(self, img_size=224, patch=16, dim=384, trunk_depth=4, heads=6,
                 num_readout=16, proprio_dim=7, action_dim=7, horizon=4):
        super().__init__()
        self.num_readout = int(num_readout)
        self.dim = int(dim)
        self.img_tok = ImageTokenizer(img_size, patch, 3, dim, depth=2, heads=heads)
        self.proprio_proj = nn.Linear(proprio_dim, dim)
        self.readout = nn.Parameter(torch.zeros(1, num_readout, dim))
        self.trunk = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(trunk_depth)])
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.action_in = nn.Linear(action_dim, dim)
        self.action_blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(2)])
        self.head = nn.Linear(dim, action_dim)
        nn.init.trunc_normal_(self.readout, std=0.02)

    def forward(self, image, proprio, timestep, noisy_action):
        """Predict the noise epsilon for one diffusion step."""
        bsz = image.shape[0]
        img_tokens = self.img_tok(image)
        prop = self.proprio_proj(proprio)[:, None, :]
        readout = self.readout.expand(bsz, -1, -1)
        cond = torch.cat([readout, img_tokens, prop], dim=1)
        for block in self.trunk:
            cond = block(cond)
        pooled = cond[:, : self.num_readout].mean(dim=1)
        temb = self.time_mlp(sinusoidal_embedding(timestep, self.dim))
        act = self.action_in(noisy_action)
        act = act + (pooled + temb)[:, None, :]
        for block in self.action_blocks:
            act = block(act)
        return self.head(act)


def _parse_args():
    p = argparse.ArgumentParser(description="Export an Octo-style denoiser network to ONNX.")
    p.add_argument("--output-dir", type=str, default="./octo_onnx")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--checkpoint", type=str, default="",
                   help="Octo PyTorch state_dict (task-2: verify flax->torch format).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--trunk-depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--num-readout", type=int, default=16)
    p.add_argument("--proprio-dim", type=int, default=7)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--horizon", type=int, default=4)
    return p.parse_args()


def main():
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = OctoDenoiseNet(
        img_size=args.img_size, patch=args.patch, dim=args.dim, trunk_depth=args.trunk_depth,
        heads=args.heads, num_readout=args.num_readout, proprio_dim=args.proprio_dim,
        action_dim=args.action_dim, horizon=args.horizon,
    )

    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu")
        sd = sd.get("model_state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"random-init demo Octo denoiser (seed={args.seed}). "
              f"Use --checkpoint for real flax->torch weights.")

    model = model.to(args.device).eval()

    image = torch.randn(1, 3, args.img_size, args.img_size, device=args.device)
    proprio = torch.randn(1, args.proprio_dim, device=args.device)
    timestep = torch.zeros(1, dtype=torch.int64, device=args.device)
    noisy_action = torch.randn(1, args.horizon, args.action_dim, device=args.device)

    onnx_path = out / "octo_denoise.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            (image, proprio, timestep, noisy_action),
            str(onnx_path),
            input_names=["image", "proprio", "timestep", "noisy_action"],
            output_names=["noise"],
            opset_version=int(args.opset),
            do_constant_folding=False,
            dynamo=False,
        )

    n_params = sum(p.numel() for p in model.parameters())
    print("Export complete.")
    print(f"ONNX saved to: {onnx_path}")
    print(f"  params={n_params/1e6:.1f}M  dim={args.dim} trunk={args.trunk_depth} "
          f"readout={args.num_readout} horizon={args.horizon} action_dim={args.action_dim}")


if __name__ == "__main__":
    main()
