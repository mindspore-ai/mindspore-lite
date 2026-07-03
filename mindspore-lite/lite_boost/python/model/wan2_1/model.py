#!/usr/bin/env python3
# modified from
# https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
reference from wan2.1
Wan2.1 Ulysses Sequence Parallel (USP) adapters.

Provides usp_attn_forward and usp_dit_forward as drop-in replacements
for WanSelfAttention.forward and WanModel.forward respectively.
"""

import torch
import torch.distributed as dist
from torch.npu import amp

from lite_boost.parallel.context_parallel import all_to_all_4d
from lite_boost.layers.rope import rope_apply


def _sinusoidal_embedding_1d(dim, position):
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, but got {dim}.")
    half = dim // 2
    position = position.float()
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half, device=position.device, dtype=torch.float32).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


# ---------------------------------------------------------------------------
# usp_attn_forward — replaces WanSelfAttention.forward
# ---------------------------------------------------------------------------

def usp_attn_forward(self,
                     x,
                     seq_lens,
                     grid_sizes,
                     freqs,
                     dtype=torch.bfloat16):  # pylint: disable=unused-argument
    """Ulysses Sequence Parallel self-attention forward."""
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    sp_size = dist.get_world_size()

    # QKV projection
    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)

    # RoPE
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    # all_to_all forward (scatter heads, gather seq)
    q = all_to_all_4d(q)
    k = all_to_all_4d(k)
    v = all_to_all_4d(v)

    # Strip padding tokens added for torch.chunk alignment
    if self.seq_pad > 0:
        q = q.reshape(b, sp_size, -1, n // sp_size, d)
        q = q[:, :, :-self.seq_pad, :, :].reshape(b, -1, n // sp_size, d)
        k = k.reshape(b, sp_size, -1, n // sp_size, d)
        k = k[:, :, :-self.seq_pad, :, :].reshape(b, -1, n // sp_size, d)
        v = v.reshape(b, sp_size, -1, n // sp_size, d)
        v = v[:, :, :-self.seq_pad, :, :].reshape(b, -1, n // sp_size, d)

    from lite_boost.layers.attention import flash_attention

    x = flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=seq_lens,
        k_lens=seq_lens,
        window_size=self.window_size)

    # flash_attention now always returns 4D [B, L_q, N//sp_size, D]

    # Re-insert padding tokens before reverse all_to_all
    if self.seq_pad > 0:
        x = x.reshape(b, sp_size, -1, n // sp_size, d)
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, self.seq_pad, 0, 0, 0, 0), value=0)
        x = x.reshape(b, -1, n // sp_size, d)

    # all_to_all reverse (scatter seq, gather heads)
    x = all_to_all_4d(x, scatter_idx=1, gather_idx=2)

    # output projection
    x = x.flatten(2)
    x = self.o(x)
    return x


# ---------------------------------------------------------------------------
# usp_dit_forward — replaces WanModel.forward
# ---------------------------------------------------------------------------

def _prepare_usp_dit_inputs(self, x, t, context, clip_fea, y, seq_len):
    """Prepare embeddings, time, context, and kwargs for USP DiT."""
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if self.model_type != 'vace' and y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    if seq_lens.max() > seq_len:
        raise ValueError(
            f"max seq_len ({seq_lens.max()}) must not exceed seq_len ({seq_len}).")
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            _sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        if e.dtype != torch.float32 or e0.dtype != torch.float32:
            raise RuntimeError(
                f"time embedding must be float32, got e.dtype={e.dtype}, e0.dtype={e0.dtype}.")

    # context
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if self.model_type != 'vace' and clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    kwargs = {
        "e": e0,
        "seq_lens": seq_lens,
        "grid_sizes": grid_sizes,
        "freqs": self.freqs,
        "context": context,
        "context_lens": None,
    }

    return x, grid_sizes, e, kwargs


def usp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    vace_context=None,
    vace_context_scale=1.0,
    clip_fea=None,
    y=None,
):
    """Ulysses Sequence Parallel DiT forward."""
    if self.model_type == 'i2v':
        if clip_fea is None or y is None:
            raise ValueError(
                "clip_fea and y must not be None for 'i2v' model_type.")

    x, grid_sizes, e, kwargs = _prepare_usp_dit_inputs(
        self, x, t, context, clip_fea, y, seq_len)

    # Pad sequence dim so torch.chunk splits evenly across world_size
    x_seq_len = x.shape[1]
    seq_pad = 0
    world_size = dist.get_world_size()
    if x_seq_len % world_size != 0:
        seq_pad = world_size - x_seq_len % world_size
        x = torch.nn.functional.pad(x, (0, 0, 0, seq_pad), value=0)
    for block in self.blocks:
        block.self_attn.seq_pad = seq_pad

    # Context Parallel — split sequence across ranks
    x = torch.chunk(x, world_size, dim=1)[dist.get_rank()]

    if self.model_type == 'vace':
        hints = self.forward_vace(x, vace_context, seq_len, kwargs)
        kwargs['hints'] = hints
        kwargs['context_scale'] = vace_context_scale

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel — gather sequence across ranks
    x_gather_shape = list(x.shape)
    x_gather_shape[1] *= world_size
    x_gather = torch.empty(x_gather_shape, device=x.device, dtype=x.dtype)
    dist.all_gather_into_tensor(x_gather, x)
    x = x_gather

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


# ---------------------------------------------------------------------------
# usp_dit_forward_vace — replaces VaceWanModel.forward_vace
# ---------------------------------------------------------------------------

def usp_dit_forward_vace(self, x, vace_context, seq_len, kwargs):
    """Ulysses Sequence Parallel VACE forward."""
    # embeddings
    c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
    c = [u.flatten(2).transpose(1, 2) for u in c]
    c = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in c
    ])

    # arguments
    new_kwargs = {"x": x}
    new_kwargs.update(kwargs)

    # Context Parallel — split vace context sequence across ranks
    c = torch.chunk(c, dist.get_world_size(), dim=1)[dist.get_rank()]

    hints = []
    for block in self.vace_blocks:
        c, c_skip = block(c, **new_kwargs)
        hints.append(c_skip)
    return hints
