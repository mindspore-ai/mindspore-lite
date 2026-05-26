#!/usr/bin/env python3
# modified from
# https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/vae2_2.py
# Copyright 2024-2025 The Alibaba Wan Team Authors
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
Wan2.2 VAE DP — drop-in replacements for Wan2_2_VAE.encode / Wan2_2_VAE.decode.

Derived from the upstream Wan2.2 vae2_2 module.  Each DP rank processes a
contiguous chunk of frames at full spatial resolution; overlapping frames
ensure causal convolution caches are correctly initialized at boundaries.
"""
import torch

from lite_boost.parallel.data_parallel import dp_temporal_process


def dp_encode(self, videos):
    """DP temporal tiling encode — replaces Wan2_2_VAE.encode.

    Reads DP configuration from ``self.dp_cfg``, ``self.dp_world_size``,
    and ``self.dp_rank`` (set by ``apply_vae_dp`` before binding).
    """
    if not isinstance(videos, list):
        raise TypeError("videos should be a list")

    cfg = self.dp_cfg
    dtype = self.dtype
    device = self.device
    ws = self.dp_world_size
    rank = self.dp_rank

    results = []
    for u in videos:
        def _enc(chunk):
            with torch.amp.autocast("npu", dtype=dtype):
                return self.model.encode(
                    chunk.unsqueeze(0), self.scale).float().squeeze(0)

        results.append(dp_temporal_process(
            u, _enc, t_dim=1,
            chunk_frames=cfg.chunk_frames, overlap_frames=cfg.overlap_frames,
            temporal_stride=cfg.temporal_stride,
            world_size=ws, rank=rank, device=device))
    return results


def dp_decode(self, zs):
    """DP temporal tiling decode — replaces Wan2_2_VAE.decode.

    Reads DP configuration from ``self.dp_cfg``, ``self.dp_world_size``,
    and ``self.dp_rank`` (set by ``apply_vae_dp`` before binding).
    """
    if not isinstance(zs, list):
        raise TypeError("zs should be a list")

    cfg = self.dp_cfg
    dtype = self.dtype
    device = self.device
    ws = self.dp_world_size
    rank = self.dp_rank

    results = []
    for u in zs:
        def _dec(chunk):
            with torch.amp.autocast("npu", dtype=dtype):
                return self.model.decode(
                    chunk.unsqueeze(0), self.scale).float().clamp_(-1, 1).squeeze(0)

        results.append(dp_temporal_process(
            u, _dec, t_dim=1,
            chunk_frames=cfg.chunk_frames, overlap_frames=cfg.overlap_frames,
            temporal_stride=cfg.temporal_stride,
            world_size=ws, rank=rank, device=device))
    return results
