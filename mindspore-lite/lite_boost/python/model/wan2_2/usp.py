#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Fine-grained Ulysses Sequence Parallel (USP) primitives.

Shared by the upstream Wan2.2 adapter (model/wan2_2/model.py) and the
DiffSynth pipeline, which inlines the DiT forward loop and therefore
cannot use the whole-block ``usp_dit_forward`` replacement:

- prepare_usp_sequence: zero-pad the sequence dim to a world_size
  multiple and chunk it across ranks (shared by both forward loops).
- gather_usp_sequence:  all_gather the sequence dim and explicitly trim
  the tail padding.  Upstream's ``unpatchify(x, grid_sizes)`` discards
  padding by grid sizes; DiffSynth's ``unpatchify(x, (f, h, w))``
  reshapes the whole tensor, so it needs the explicit trim.
- usp_attn_forward_cos_sin: self-attention replacement for models that
  receive an expanded ``(S, 1, C)`` complex freqs tensor (DiffSynth
  style) instead of ``(grid_sizes, 1D freqs)`` (upstream style).
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.npu import amp

from lite_boost.parallel.context_parallel import all_to_all_4d


def prepare_usp_sequence(x, e=None, e0=None):
    """Zero-pad the sequence dim to a multiple of world_size and chunk it.

    Padding tokens are appended at the END of the sequence so
    ``gather_usp_sequence`` can trim them from the tail after gathering.

    Args:
        x:  ``[B, S, D]`` main sequence.
        e:  Optional ``[B, S, ...]`` sequence-length-aligned tensor.
        e0: Optional ``[B, S, N, D]`` sequence-length-aligned tensor.

    Returns:
        ``(x_local, e_local, e0_local, seq_pad)`` — per-rank slices and
        the number of padded tokens appended to the sequence.
    """
    world_size = dist.get_world_size()
    if world_size <= 1:
        return x, e, e0, 0

    x_seq_len = x.shape[1]
    seq_pad = 0
    if x_seq_len % world_size != 0:
        seq_pad = world_size - x_seq_len % world_size
        x = F.pad(x, (0, 0, 0, seq_pad), value=0)
        if e is not None:
            e = F.pad(e, (0, 0, 0, seq_pad), value=0)
        if e0 is not None:
            e0 = F.pad(e0, (0, 0, 0, 0, 0, seq_pad), value=0)

    rank = dist.get_rank()
    x = torch.chunk(x, world_size, dim=1)[rank]
    if e is not None:
        e = torch.chunk(e, world_size, dim=1)[rank]
    if e0 is not None:
        e0 = torch.chunk(e0, world_size, dim=1)[rank]
    return x, e, e0, seq_pad


def gather_usp_sequence(x, seq_pad=0):
    """all_gather the sequence dim across ranks and trim tail padding.

    ``seq_pad=0`` makes the trim a no-op, so upstream callers see no
    behavior change.
    """
    world_size = dist.get_world_size()
    if world_size <= 1:
        return x

    x_gather_shape = list(x.shape)
    x_gather_shape[1] *= world_size
    x_gather = torch.empty(x_gather_shape, device=x.device, dtype=x.dtype)
    dist.all_gather_into_tensor(x_gather, x)
    if seq_pad > 0:
        x_gather = x_gather[:, :-seq_pad]
    return x_gather


@amp.autocast(enabled=False)
def _rope_apply_expanded(x, freqs, s_local):
    """Component-wise RoPE with the per-rank slice of an expanded freqs.

    ``freqs`` is the DiffSynth-style expanded complex tensor ``(S, 1, C)``
    covering the FULL sequence; this function slices this rank's segment
    (same indexing as the old xfuser adapter: ``freqs[rank*s:(rank+1)*s]``)
    and applies RoPE with the component-wise formulation used across
    lite_boost (avoids complex64 view ops on NPU).

    Args:
        x:      ``[B, S_local, N, D]`` query/key tensor.
        freqs:  ``[S_full, 1, D]`` complex64 RoPE table (expanded).
        s_local: Local sequence length (including ``seq_pad``).

    Returns:
        RoPE-applied tensor, same shape as ``x``.
    """
    sp_size = dist.get_world_size()
    if sp_size > 1:
        sp_rank = dist.get_rank()
        full = s_local * sp_size
        if freqs.shape[0] < full:
            freqs = F.pad(freqs, (0, 0, 0, 0, 0, full - freqs.shape[0]))
        freqs = freqs[sp_rank * s_local:(sp_rank + 1) * s_local]

    # [1, S, 1, D] — broadcasts to any batch size (upstream assumes B=1)
    cos = freqs.real.float().unsqueeze(0)
    sin = freqs.imag.float().unsqueeze(0)

    x_r = x.reshape(*x.shape[:3], -1, 2)
    a = x_r[..., 0]
    b = x_r[..., 1]

    out_a = a * cos - b * sin
    out_b = b * cos + a * sin
    return torch.stack([out_a, out_b], dim=-1).flatten(3).float()


def usp_attn_forward_cos_sin(self, x, freqs, seq_lens=None,
                             dtype=torch.bfloat16):  # pylint: disable=unused-argument
    """Ulysses SP self-attention for expanded-cos/sin RoPE (DiffSynth style).

    Drop-in replacement for ``WanSelfAttention.forward(x, freqs)`` on
    DiffSynth models.  Attribute contract (``norm_q/norm_k/q/k/v/o/
    num_heads/head_dim``) matches both upstream Wan2.2 and DiffSynth;
    ``window_size`` is optional (DiffSynth SelfAttention has none).

    Sequence padding inserted by ``prepare_usp_sequence`` is stripped
    before flash attention and re-inserted before the reverse all_to_all,
    exactly like ``usp_attn_forward``.
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    # QKV projection
    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)

    # RoPE (per-rank slice of the expanded freqs table)
    q = _rope_apply_expanded(q, freqs, s)
    k = _rope_apply_expanded(k, freqs, s)

    # all_to_all forward (scatter heads, gather seq)
    q = all_to_all_4d(q)
    k = all_to_all_4d(k)
    v = all_to_all_4d(v)

    # Strip padding tokens added for torch.chunk alignment.
    seq_pad = getattr(self, "seq_pad", 0)
    if seq_pad > 0:
        q = q[:, :-seq_pad, :, :]
        k = k[:, :-seq_pad, :, :]
        v = v[:, :-seq_pad, :, :]

    from lite_boost.layers.attention import flash_attention

    x = flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=seq_lens,
        k_lens=seq_lens,
        window_size=getattr(self, "window_size", None))

    # Re-insert padding tokens at the END before reverse all_to_all
    if seq_pad > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, seq_pad, 0, 0), value=0)

    # all_to_all reverse (scatter seq, gather heads)
    x = all_to_all_4d(x, scatter_idx=1, gather_idx=2)

    # output projection
    x = x.flatten(2)
    x = self.o(x)
    return x
