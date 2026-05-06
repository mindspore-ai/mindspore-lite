#!/usr/bin/env python3
# modified from
# https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py
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
Wan2.2 Ulysses Sequence Parallel (USP) adapters.

Provides usp_attn_forward and usp_dit_forward as drop-in replacements
for WanSelfAttention.forward and WanModel.forward respectively.
"""

import torch
import torch.distributed as dist
from torch.npu import amp

from lite_boost.parallel.context_parallel import all_to_all_4d
from lite_boost.model.wan2_2.rope import rope_apply


def _sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
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

    # Strip padding tokens added for torch.chunk alignment.
    # Padding was appended to the END of x before chunking, so after
    # all_to_all gathers the full sequence, strip from the end of dim=1.
    if self.seq_pad > 0:
        q = q[:, :-self.seq_pad, :, :]
        k = k[:, :-self.seq_pad, :, :]
        v = v[:, :-self.seq_pad, :, :]

    from lite_boost.layers.attention import flash_attention

    x = flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=seq_lens,
        k_lens=seq_lens,
        window_size=self.window_size)

    # flash_attention now always returns 4D [B, L_q, N//world_size, D]

    # Re-insert padding tokens at the END before reverse all_to_all
    if self.seq_pad > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, self.seq_pad, 0, 0), value=0)

    # all_to_all reverse (scatter seq, gather heads)
    x = all_to_all_4d(x, scatter_idx=1, gather_idx=2)

    # output projection
    x = x.flatten(2)
    x = self.o(x)
    return x


# ---------------------------------------------------------------------------
# usp_dit_forward — replaces WanModel.forward
# ---------------------------------------------------------------------------

def _prepare_usp_dit_inputs(self, x, t, context, y, seq_len):
    """Prepare embeddings, time, context, and kwargs for USP DiT."""
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with amp.autocast(dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            _sinusoidal_embedding_1d(self.freq_dim,
                                     t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

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
    y=None,
):
    """Ulysses Sequence Parallel DiT forward."""
    if self.model_type == 'i2v':
        assert y is not None

    x, grid_sizes, e, kwargs = _prepare_usp_dit_inputs(
        self, x, t, context, y, seq_len)

    # Pad sequence dim so torch.chunk splits evenly across world_size
    x_seq_len = x.shape[1]
    seq_pad = 0
    world_size = dist.get_world_size()
    if x_seq_len % world_size != 0:
        seq_pad = world_size - x_seq_len % world_size
        x = torch.nn.functional.pad(x, (0, 0, 0, seq_pad), value=0)
        e = torch.nn.functional.pad(e, (0, 0, 0, seq_pad), value=0)
        kwargs["e"] = torch.nn.functional.pad(kwargs["e"], (0, 0, 0, 0, 0, seq_pad), value=0)
    for block in self.blocks:
        block.self_attn.seq_pad = seq_pad

    # Context Parallel — split sequence across ranks
    x = torch.chunk(x, world_size, dim=1)[dist.get_rank()]
    e = torch.chunk(e, world_size, dim=1)[dist.get_rank()]
    kwargs["e"] = torch.chunk(kwargs["e"], world_size, dim=1)[dist.get_rank()]

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
