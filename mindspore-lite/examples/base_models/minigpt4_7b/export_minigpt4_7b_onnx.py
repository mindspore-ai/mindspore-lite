#!/usr/bin/env python3
"""Export a MiniGPT-4-style VLM skeleton (Q-Former + LLM) to ONNX.

MiniGPT-4 (Vision-CAIR) aligns a frozen vision encoder with an LLM (Vicuna)
through a Q-Former + linear projector (BLIP-2 lineage). The deployed graph here
is a self-contained skeleton of one forward pass:

  - inputs : image      [1, 3, 224, 224]   float32
             input_ids  [1, seq_len]        int64
  - output : logits     [1, num_query+seq_len, vocab_size]  float32

``--random-init`` runs the pipeline. Real MiniGPT-4 should be loaded via its
official package with prefill/decode in task-2 (reuses BLIP-2 Q-Former know-how).
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


class ManualCrossAttention(nn.Module):
    """Manual cross-attention (query, key/value) for ONNX export."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = int(heads)
        self.hdim = int(dim) // int(heads)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, q, kv):
        b, nq, d = q.shape
        nk = kv.shape[1]
        q = self.q_proj(q).reshape(b, nq, self.heads, self.hdim).transpose(1, 2)
        kver = self.kv_proj(kv).reshape(b, nk, 2, self.heads, self.hdim).permute(2, 0, 3, 1, 4)
        k, v = kver[0], kver[1]
        attn = (q @ k.transpose(-2, -1)) * (self.hdim ** -0.5)
        out = attn.softmax(dim=-1) @ v
        out = out.transpose(1, 2).reshape(b, nq, d)
        return self.out_proj(out)


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


class VisionEncoder(nn.Module):
    """ViT-like patch encoder -> image features."""

    def __init__(self, img_size=224, patch=16, dim=384, depth=2, heads=6):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch) ** 2, dim))
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads) for _ in range(depth)])

    def forward(self, image):
        x = self.patch_embed(image).flatten(2).transpose(1, 2) + self.pos_embed
        return self.blocks(x)


class QFormer(nn.Module):
    """Learnable queries with self-attention + cross-attention to vision features."""

    def __init__(self, dim, num_query=32, heads=6):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, num_query, dim))
        self.self_block = TransformerBlock(dim, heads)
        self.cross_attn = ManualCrossAttention(dim, heads)
        self.norm = nn.LayerNorm(dim)
        self.ffn = TransformerBlock(dim, heads)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, vision_feat):
        q = self.self_block(self.query.expand(vision_feat.shape[0], -1, -1))
        a = self.cross_attn(self.norm(q), vision_feat)
        return self.ffn(q + a)


class MiniGPT4Skel(nn.Module):
    """MiniGPT-4 skeleton: vision encoder + Q-Former + projector + LLM head."""

    def __init__(self, img_size=224, patch=16, dim=384, depth=4, heads=6,
                 num_query=32, vocab_size=32000, seq_len=32):
        super().__init__()
        self.vision = VisionEncoder(img_size, patch, dim, depth=2, heads=heads)
        self.qformer = QFormer(dim, num_query, heads)
        self.projector = nn.Linear(dim, dim)
        self.tok_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_query + seq_len, dim))
        self.llm = nn.Sequential(*[TransformerBlock(dim, heads) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab_size, bias=False)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, image, input_ids):
        vis_feat = self.vision(image)
        query = self.projector(self.qformer(vis_feat))
        txt = self.tok_embed(input_ids.long())
        x = torch.cat([query, txt], dim=1) + self.pos_embed
        x = self.llm(x)
        return self.head(x)


def _parse_args():
    p = argparse.ArgumentParser(description="Export a MiniGPT-4-style VLM skeleton to ONNX.")
    p.add_argument("--output-dir", type=str, default="./minigpt4_7b_onnx")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--checkpoint", type=str, default="", help="MiniGPT-4 state_dict (task-2: official package).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--num-query", type=int, default=32)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--seq-len", type=int, default=32)
    return p.parse_args()


def main():
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = MiniGPT4Skel(args.img_size, args.patch, args.dim, args.depth, args.heads,
                         args.num_query, args.vocab_size, args.seq_len)

    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu")
        sd = sd.get("model_state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"random-init demo MiniGPT-4 skeleton (seed={args.seed}). "
              f"Real MiniGPT-4 = ViT + Q-Former + Vicuna-7B (task-2).")

    model = model.to(args.device).eval()
    image = torch.randn(1, 3, args.img_size, args.img_size, device=args.device)
    input_ids = torch.randint(0, args.vocab_size, (1, args.seq_len), device=args.device)
    onnx_path = out / "minigpt4_7b.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model, (image, input_ids), str(onnx_path),
            input_names=["image", "input_ids"], output_names=["logits"],
            opset_version=int(args.opset), do_constant_folding=False, dynamo=False)

    n_params = sum(p.numel() for p in model.parameters())
    print("Export complete.")
    print(f"ONNX saved to: {onnx_path}")
    print(f"  params={n_params/1e6:.1f}M  num_query={args.num_query} vocab={args.vocab_size} seq={args.seq_len}")


if __name__ == "__main__":
    main()
