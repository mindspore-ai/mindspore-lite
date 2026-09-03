"""Torch definitions for the ``MsTransRopeScatterNDUpdate`` custom operator.

The ``forward`` method provides an eager reference implementation; ``symbolic``
emits the ``custom::MsTransRopeScatterNDUpdate`` node consumed by the Kirin OMG
framework plugins.

Semantics (LLM 推理 KV-cache 动态更新):
    q/k/v       : fp16 [B, S, Hq/Hk, D]（BSHD，投影后布局）
    cos/sin     : fp16 [1, S, rotary_dim]（跨 batch/heads 共享）
    kcache/vcache: fp16 [B, Hk, L, D]
    indices     : int32 [1] —— 写入起点 idx（"下一个空位"）
    rope_q      : [B, Hq, S, D]        （transpose(1,2) + RoPE）
    scatternd_k/v: [B, Hk, L, D]       （cache 克隆后写入 [idx, idx+S) 行）
"""

from __future__ import annotations

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """NeoX rotate_half: cat([-x2, x1])，x1/x2 为最后一维的前后两半。"""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class MsTransRopeScatterNDUpdate(torch.autograd.Function):
    """FP16 fused RoPE + Transpose + ScatterND-Update as ``custom::MsTransRopeScatterNDUpdate``."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kcache: torch.Tensor,
        vcache: torch.Tensor,
        indices: torch.Tensor,
        num_attention_heads: int | None = None,
        rotary_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the fused operator reference without mutating input caches."""
        del ctx
        _, seq_len, num_heads_q, head_dim = q.shape
        kv_len = kcache.shape[2]
        # 注：GQA 整除性（Hq%Hk==0）由 host 侧 ValidateShapes 校验；golden 不重复
        # （重复校验会让"导出非法形状 ONNX → omg 拒绝"的负例测试在导出阶段就失败）。
        if num_attention_heads is not None and num_attention_heads != num_heads_q:
            raise ValueError(
                f"num_attention_heads({num_attention_heads}) != q heads({num_heads_q})"
            )
        rd = cos.shape[-1] if rotary_dim is None else int(rotary_dim)
        if rd != cos.shape[-1] or rd != sin.shape[-1] or rd % 2 != 0 or rd > head_dim:
            raise ValueError(
                f"rotary_dim({rd}) must match cos/sin last dim "
                f"({cos.shape[-1]}/{sin.shape[-1]}), be even and <= head_dim({head_dim})"
            )
        idx = int(indices.reshape(-1)[0])
        if idx + seq_len > kv_len:
            raise ValueError(
                f"indices[{idx}] + seq_len({seq_len}) exceeds kv_len({kv_len})"
            )

        # RoPE（fp32 数学）：cos/sin [1,S,rd] → [1,1,S,rd] 广播到 [B,H,S,rd]。
        def rope(x_bhsd: torch.Tensor) -> torch.Tensor:
            xf = x_bhsd.to(torch.float32)
            x_rot, x_pass = xf[..., :rd], xf[..., rd:]
            cosf = cos.to(torch.float32).reshape(seq_len, rd).unsqueeze(0).unsqueeze(0)
            sinf = sin.to(torch.float32).reshape(seq_len, rd).unsqueeze(0).unsqueeze(0)
            out_rot = x_rot * cosf + _rotate_half(x_rot) * sinf
            return torch.cat((out_rot, x_pass), dim=-1).to(torch.float16)

        rope_q = rope(q.transpose(1, 2))          # [B, Hq, S, D]
        rope_k = rope(k.transpose(1, 2))          # [B, Hk, S, D]
        v_t = v.transpose(1, 2).to(torch.float16) # [B, Hk, S, D]，无 RoPE

        scatternd_k = kcache.clone()
        scatternd_k[:, :, idx : idx + seq_len, :] = rope_k
        scatternd_v = vcache.clone()
        scatternd_v[:, :, idx : idx + seq_len, :] = v_t
        return rope_q, scatternd_k, scatternd_v

    @staticmethod
    def symbolic(
        g,
        q,
        k,
        v,
        cos,
        sin,
        kcache,
        vcache,
        indices,
        num_attention_heads: int | None = None,
        rotary_dim: int | None = None,
    ):
        """Emit the fused custom node with BNSD output shapes."""
        q_sizes = q.type().sizes()
        batch, seq_len, num_heads_q, head_dim = q_sizes
        num_heads_k = k.type().sizes()[2]
        kv_len = kcache.type().sizes()[2]
        rd = int(rotary_dim) if rotary_dim is not None else cos.type().sizes()[-1]
        outputs = g.op(
            "custom::MsTransRopeScatterNDUpdate",
            q,
            k,
            v,
            cos,
            sin,
            kcache,
            vcache,
            indices,
            num_attention_heads_i=int(
                num_attention_heads if num_attention_heads is not None else num_heads_q
            ),
            rotary_dim_i=rd,
            outputs=3,
        )
        outputs[0].setType(
            outputs[0]
            .type()
            .with_dtype(torch.float16)
            .with_sizes([batch, num_heads_q, seq_len, head_dim])
        )
        outputs[1].setType(
            outputs[1]
            .type()
            .with_dtype(torch.float16)
            .with_sizes([batch, num_heads_k, kv_len, head_dim])
        )
        outputs[2].setType(
            outputs[2]
            .type()
            .with_dtype(torch.float16)
            .with_sizes([batch, num_heads_k, kv_len, head_dim])
        )
        return outputs
