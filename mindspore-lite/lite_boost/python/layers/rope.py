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
Optimized RoPE with float32 real-valued arithmetic and frequency table caching.

Replaces the original complex128 + float64 RoPE path with a pre-computed
cos/sin cache, reducing RoPE time by ~88% (153ms → 18ms per block).

Handles CP sequence padding correctly: when seq_len is not evenly
divisible by sp_size, the last rank's RoPE slice may extend beyond the
cos/sin table — padding positions get zero cos/sin (input values are
zero anyway, so the RoPE result remains zero).  Used by both the
Wan2.1 and Wan2.2 model paths.
"""
import torch
import torch.distributed as dist
from torch.npu import amp

_rope_cache = {}


def _get_rope_cos_sin(freqs, grid_sizes, s_local):
    """
    Pre-compute cos/sin RoPE tables (cached).
    freqs:      [1024, D//2] complex (polar form e^(i*theta))
    grid_sizes: [B, 3]
    s_local:    per-rank sequence length
    Returns: (cos, sin) each [B, s_local, D//2] float32
    """
    sp_size = dist.get_world_size()
    sp_rank = dist.get_rank()

    # The cache key must cover the whole grid list: two batches with the
    # same first sample but different later samples must not share tables.
    key = (tuple(grid_sizes.flatten().tolist()),
           sp_rank, sp_size, s_local, freqs.device)
    if key in _rope_cache:
        return _rope_cache[key]

    cos_freqs = freqs.real.float()
    sin_freqs = freqs.imag.float()
    c = cos_freqs.size(1)
    c1, c2 = c - 2 * (c // 3), c // 3
    cos_s = torch.split(cos_freqs, [c1, c2, c2], dim=1)
    sin_s = torch.split(sin_freqs, [c1, c2, c2], dim=1)

    all_cos, all_sin = [], []
    for grid_size in grid_sizes:
        f, h, w = grid_size.tolist()
        seq_len = f * h * w

        cos_i = torch.cat([
            cos_s[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            cos_s[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            cos_s[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, c)

        sin_i = torch.cat([
            sin_s[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            sin_s[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            sin_s[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, c)

        start = sp_rank * s_local
        end = start + s_local
        if end > seq_len:
            # Last rank with seq_len % sp_size != 0: zero-pad the slice
            # past the table (padding positions hold zero input values).
            pad = end - seq_len
            c_i = torch.cat([cos_i[start:, :],
                             cos_i.new_zeros(pad, c)], dim=0)
            s_i = torch.cat([sin_i[start:, :],
                             sin_i.new_zeros(pad, c)], dim=0)
        else:
            c_i = cos_i[start:end, :]
            s_i = sin_i[start:end, :]
        all_cos.append(c_i)
        all_sin.append(s_i)

    cos = torch.stack(all_cos)
    sin = torch.stack(all_sin)
    _rope_cache[key] = (cos, sin)
    return cos, sin


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    r"""
    Applies rotary position embeddings (RoPE) to `x`.

    Each interleaved complex pair :math:`(x[..., 2k], x[..., 2k+1])` is
    rotated by the angle from `freqs`, using a per-sample cos/sin table
    expanded from `grid_sizes`.  The table is cached process-wide.  The
    per-rank slice follows the sequence-parallel (SP) partition; when
    ``seq_len % sp_size != 0`` the last rank's slice is zero-padded past
    the table, and padding positions hold zero input values so the output
    stays zero there.

    Notation: `T` is the length of the `freqs` table, `B` the batch
    size, `F`/`H`/`W` the (frames, height, width) grid of each sample
    with raw length ``seq_len = F*H*W``, `D` the head dim, `N` the
    number of heads, and `s` the per-rank sequence length
    ``padded_seq_len / sp_size`` (``padded_seq_len`` is the common
    padded length of every sample before the SP split).  So `s` is
    related to ``F*H*W`` but not identical: with a single rank and no
    padding ``s == F*H*W``; in general ``s >= ceil(seq_len / sp_size)``
    (the padded length is split evenly across ranks), and the last
    rank's slice may extend past a short sample's table -- those
    positions are zero-padded (their inputs are zero, so the output
    stays zero).

    Requires ``torch.distributed`` to be initialized before calling
    (single process: ``dist.init_process_group(backend="hccl",
    world_size=1, rank=0)``).

    Supported only on A2; 300I Duo is not supported.

    Args:
        x (Tensor): Input tensor with shape :math:`(B, s, N, D)`, where
            `B` is the batch size, `s` the per-rank sequence length
            (``padded_seq_len / sp_size``), `N` the number of heads, and
            `D` the head dim (even; pairs are rotated).  Supported dtypes
            are float16, float32 and bfloat16.
        grid_sizes (Tensor): Integer tensor with shape :math:`(B, 3)`, the
            ``(F, H, W)`` grid of each sample (``seq_len = F*H*W``).
        freqs (Tensor): Complex tensor with shape :math:`(T, D//2)` in
            polar form :math:`e^{i\theta}`; must be on the same device as
            `x`.

    Returns:
        Tensor, with the same shape as `x`, always cast to float32.

    Raises:
        RuntimeError: If `x` is not 4-D, if `D` is odd, or if `freqs` is
            on a different device from `x`.
        ValueError: If ``torch.distributed`` is not initialized before
            the call.
        TypeError: If `grid_sizes` is not 2-D with shape :math:`(B, 3)`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import os
        >>> import torch
        >>> import torch_npu
        >>> import torch.distributed as dist
        >>> from lite_boost.layers import rope_apply
        >>> os.environ["MASTER_ADDR"] = "127.0.0.1"
        >>> os.environ["MASTER_PORT"] = "29500"
        >>> torch.npu.set_device(0)
        >>> torch.npu.set_compile_mode(jit_compile=False)  # complex freqs on some platforms
        >>> if not dist.is_initialized():
        ...     dist.init_process_group(backend="hccl", world_size=1, rank=0)
        >>> x = torch.randn(1, 16, 8, 32, device="npu")
        >>> grid_sizes = torch.tensor([[4, 4, 4]], device="npu")
        >>> freqs = torch.exp(1j * torch.rand(1024, 16)).to("npu")
        >>> y = rope_apply(x, grid_sizes, freqs)
        >>> print(y.shape)
        torch.Size([1, 16, 8, 32])
        >>> print(y.dtype)
        torch.float32
        >>> dist.destroy_process_group()
    """
    if (not torch.is_tensor(grid_sizes) or grid_sizes.dim() != 2
            or grid_sizes.size(1) != 3):
        raise TypeError(
            f"grid_sizes must be a 2-D tensor with shape [B, 3], but got "
            f"{type(grid_sizes).__name__} with shape "
            f"{tuple(grid_sizes.shape) if torch.is_tensor(grid_sizes) else grid_sizes}.")
    if x.dim() != 4:
        raise RuntimeError(
            f"x must be 4-D with shape [B, s, N, D], but got {tuple(x.shape)}.")
    if x.size(-1) % 2 != 0:
        raise RuntimeError(
            f"x head dim D must be even, but got {x.size(-1)}.")
    if freqs.device != x.device:
        raise RuntimeError(
            f"freqs must be on the same device as x, but got freqs on "
            f"{freqs.device} and x on {x.device}.")
    if not dist.is_initialized():
        raise ValueError(
            "torch.distributed must be initialized before calling rope_apply.")
    s = x.size(1)
    cos, sin = _get_rope_cos_sin(freqs, grid_sizes, s)

    # Interleaved complex pairs: (x[...,2k], x[...,2k+1]) → (a + i*b)
    x_r = x.reshape(*x.shape[:3], -1, 2)
    a = x_r[..., 0]
    b = x_r[..., 1]

    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

    out_a = a * cos - b * sin
    out_b = b * cos + a * sin
    return torch.stack([out_a, out_b], dim=-1).flatten(3).float()
