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
Optimized RoPE for Wan2.2 — handles CP sequence padding correctly.

Derived from lite_boost.layers.rope with a fix: when seq_len is not evenly
divisible by sp_size, the last rank's RoPE slice may extend beyond the
cos/sin table.  Padding positions get zero cos/sin (input values are zero
anyway, so the RoPE result remains zero).
"""
import torch
import torch.distributed as dist
from torch.npu import amp

_rope_cache = {}


def _get_rope_cos_sin(freqs, grid_sizes, s_local):
    """Compute per-rank cos/sin RoPE tables for CP-sharded sequences."""
    sp_size = dist.get_world_size()
    sp_rank = dist.get_rank()

    f0, h0, w0 = grid_sizes[0].tolist()
    key = (f0, h0, w0, sp_rank, sp_size, s_local, freqs.device)
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
    """Apply rotary position embedding with CP-aware per-rank slicing."""
    s = x.size(1)
    cos, sin = _get_rope_cos_sin(freqs, grid_sizes, s)

    x_r = x.reshape(*x.shape[:3], -1, 2)
    a = x_r[..., 0]
    b = x_r[..., 1]

    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

    out_a = a * cos - b * sin
    out_b = b * cos + a * sin
    return torch.stack([out_a, out_b], dim=-1).flatten(3).float()
