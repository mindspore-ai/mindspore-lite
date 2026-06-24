#!/usr/bin/env python3
"""Export an OpenVLA-style policy (vision+language -> action chunk) to ONNX.

OpenVLA (Stanford/Berkeley/TRI, Kim et al.) is a Prismatic VLM
(SigLIP+DINOv2 vision tower + Llama-2 LLM) that emits actions as autoregressive
action tokens. The full model is a VLM with prefill/decode (see internvl3_5_1b
for the reference prefill/decode split once real weights are wired in).

This script provides a self-contained PyTorch **regression skeleton** of the
vision-conditioned policy for the single forward pass:

  - inputs : image        [1, 3, 224, 224]   float32
             task_tokens  [1, task_len]       int64   (tokenized language instruction)
  - output : action       [1, horizon, action_dim]  float32

``--random-init`` runs the pipeline; real OpenVLA weights should be loaded via
the official ``prismatic``/transformers path with a prefill/decode split (task-2).
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


class ImageTokenizer(nn.Module):
    """Patchify image + a few transformer layers -> image tokens."""

    def __init__(self, img_size=224, patch=16, dim=384, depth=2, heads=6):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch) ** 2, dim))
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])

    def forward(self, image):
        x = self.patch_embed(image).flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return x


class OpenVLARegPolicy(nn.Module):
    """Simplified OpenVLA regression policy: image + task tokens -> action chunk."""

    def __init__(self, img_size=224, patch=16, dim=384, depth=4, heads=6,
                 num_readout=16, action_dim=7, horizon=16, vocab_size=32000, task_len=16):
        super().__init__()
        self.num_readout = int(num_readout)
        self.img_tok = ImageTokenizer(img_size, patch, dim, depth=2, heads=heads)
        self.task_embed = nn.Embedding(vocab_size, dim)
        self.readout = nn.Parameter(torch.zeros(1, horizon, dim))
        self.trunk = nn.Sequential(*[TransformerBlock(dim, heads) for _ in range(depth)])
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, action_dim))
        nn.init.trunc_normal_(self.readout, std=0.02)

    def forward(self, image, task_tokens):
        """image [B,3,H,W], task_tokens [B,L] -> action [B,horizon,action_dim]."""
        bsz = image.shape[0]
        img = self.img_tok(image)
        task = self.task_embed(task_tokens.long())
        readout = self.readout.expand(bsz, -1, -1)
        x = torch.cat([readout, img, task], dim=1)
        x = self.trunk(x)
        pooled = x[:, : self.readout.shape[1]]
        return self.head(pooled)


def _parse_args():
    p = argparse.ArgumentParser(description="Export an OpenVLA-style policy to ONNX.")
    p.add_argument("--output-dir", type=str, default="./openvla_7b_onnx")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--checkpoint", type=str, default="", help="OpenVLA state_dict (task-2: use prismatic/transformers).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--task-len", type=int, default=16)
    return p.parse_args()


def main():
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = OpenVLARegPolicy(args.img_size, args.patch, args.dim, args.depth, args.heads,
                             args.horizon, args.action_dim, args.horizon, args.vocab_size, args.task_len)

    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu")
        sd = sd.get("model_state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"random-init demo OpenVLA regression skeleton (seed={args.seed}). "
              f"Real OpenVLA = prismatic VLM + autoregressive action tokens (task-2).")

    model = model.to(args.device).eval()
    image = torch.randn(1, 3, args.img_size, args.img_size, device=args.device)
    task_tokens = torch.randint(0, args.vocab_size, (1, args.task_len), device=args.device)
    onnx_path = out / "openvla_7b_policy.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model, (image, task_tokens), str(onnx_path),
            input_names=["image", "task_tokens"], output_names=["action"],
            opset_version=int(args.opset), do_constant_folding=False, dynamo=False)

    n_params = sum(p.numel() for p in model.parameters())
    print("Export complete.")
    print(f"ONNX saved to: {onnx_path}")
    print(f"  params={n_params/1e6:.1f}M  action_dim={args.action_dim} horizon={args.horizon} "
          f"(real OpenVLA-7B uses Llama-2-7B; wire via prismatic in task-2)")


if __name__ == "__main__":
    main()
