#!/usr/bin/env python3
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
Model-specific adaptation for Wan2.2 on NPU.

Entry point: ``boost_wan2_2(target)`` accepts a Wan pipeline or raw WanModel
and patches everything in-place for parallel inference.
"""
import types
from dataclasses import dataclass

import torch.distributed as dist

from lite_boost.layers.attention import flash_attention as lite_flash_attention
from lite_boost.model.wan2_2.model import usp_attn_forward, usp_dit_forward
from lite_boost.model.wan2_2.vae2_2 import dp_encode, dp_decode

_PIPELINE_CLASSES = ('WanT2V', 'WanTI2V', 'WanI2V', 'WanS2V')
_DIT_MODEL_ATTRS = ('model', 'low_noise_model', 'high_noise_model')


@dataclass
class DPConfig:
    """Configuration for VAE temporal DP tiling.

    Defaults are tuned for Wan2.2 VAEs (spatial_scale=16).
    For Wan2.1 set spatial_scale=8.
    """
    spatial_scale: int = 16
    temporal_stride: int = 4       # vae_stride[0] (4 for TI2V-5B, 4 for T2V/I2V)
    enable_encoder_dp: bool = True
    enable_decoder_dp: bool = True
    chunk_frames: int = 12         # frames per temporal chunk
    overlap_frames: int = 8        # overlap between adjacent chunks

    def __post_init__(self):
        if self.chunk_frames <= self.overlap_frames:
            raise ValueError(
                f"chunk_frames ({self.chunk_frames}) must exceed "
                f"overlap_frames ({self.overlap_frames})")


def _unwrap_fsdp(obj):
    """Return the underlying module if *obj* is FSDP-wrapped, otherwise *obj*."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(obj, FSDP):
            return getattr(obj, '_fsdp_wrapped_module', obj)
    except ImportError:
        pass
    return obj


def _find_dit_models(pipe) -> list:
    """Find all WanModel instances attached to a pipeline.

    Returns the **unwrapped** module so ``_boost_dit`` can safely replace
    ``model.forward`` without interfering with FSDP.
    """
    from wan.modules.model import WanModel
    models = []
    seen = set()
    for attr in _DIT_MODEL_ATTRS:
        obj = getattr(pipe, attr, None)
        if obj is None:
            continue
        inner = _unwrap_fsdp(obj)
        if isinstance(inner, WanModel) and id(inner) not in seen:
            models.append(inner)
            seen.add(id(inner))
    return models


def _find_vae(pipe):
    """Find the VAE attached to a pipeline."""
    return getattr(pipe, 'vae', None)


def boost_wan2_2(target):
    """Boost a Wan2.2 pipeline or raw WanModel for parallel inference.

    - WanModel instances → Context Parallel (USP) for DiT
    - VAE → Data Parallel (temporal tiling) for encoder/decoder
    """
    cls_name = target.__class__.__name__

    if cls_name in _PIPELINE_CLASSES:
        models = _find_dit_models(target)
        if not models:
            raise RuntimeError(
                f"No WanModel found in pipeline '{cls_name}'")
        for m in models:
            _boost_dit(m)
        vae = _find_vae(target)
        if vae is not None:
            _boost_vae(vae)
    else:
        _boost_dit(target)


def _boost_dit(model):
    """Patch a WanModel in-place for NPU Ulysses SP.

    Operations:
    1. Replace flash_attention in wan.modules with NPU-compatible version
    2. Replace each block.self_attn.forward → usp_attn_forward
    3. Replace model.forward → usp_dit_forward
    """
    world_size = dist.get_world_size()

    if model.num_heads % world_size != 0:
        raise ValueError(
            f"num_heads ({model.num_heads}) must be divisible by "
            f"world_size ({world_size})"
        )

    # Replace wan's flash_attention with NPU-compatible version.
    # Must patch both the source module AND model.py's cached reference
    # (model.py does `from .attention import flash_attention` at module level).
    import wan.modules.attention as _wan_attn
    import wan.modules.model as _wan_model
    _wan_attn.flash_attention = lite_flash_attention
    _wan_model.flash_attention = lite_flash_attention

    for block in model.blocks:
        block.self_attn.seq_pad = 0
        block.self_attn.forward = types.MethodType(
            usp_attn_forward, block.self_attn
        )

    model.forward = types.MethodType(usp_dit_forward, model)


def apply_vae_dp(
    vae,
    spatial_scale: int = 16,
    temporal_stride: int = 4,
    enable_encoder_dp: bool = True,
    enable_decoder_dp: bool = True,
    chunk_frames: int = 12,
    overlap_frames: int = 8,
):
    """Apply VAE encode/decode in-place for DP temporal tiling.

    Replaces ``vae.encode`` and ``vae.decode`` directly on the VAE
    instance so the original reference remains valid.  Safe to call
    unconditionally — returns the VAE unchanged if world_size <= 1.

    Args:
        vae:              Loaded Wan2_2_VAE or Wan2_1_VAE instance.
        spatial_scale:    VAE spatial stride (16 for Wan2.2, 8 for Wan2.1).
        temporal_stride:  VAE temporal stride (vae_stride[0]).
        enable_encoder_dp: Replace encode() with DP temporal tiling.
        enable_decoder_dp: Replace decode() with DP temporal tiling.
        chunk_frames:     Frames per temporal chunk (includes overlap).
        overlap_frames:   Overlap between adjacent chunks.

    Returns:
        The same VAE instance, with encode/decode applied in-place.
    """
    world_size = dist.get_world_size()
    if world_size <= 1:
        return vae
    if not enable_encoder_dp and not enable_decoder_dp:
        return vae

    rank = dist.get_rank()
    cfg = DPConfig(
        spatial_scale=spatial_scale,
        temporal_stride=temporal_stride,
        enable_encoder_dp=enable_encoder_dp,
        enable_decoder_dp=enable_decoder_dp,
        chunk_frames=chunk_frames,
        overlap_frames=overlap_frames,
    )

    vae.dp_cfg = cfg
    vae.dp_world_size = world_size
    vae.dp_rank = rank

    if enable_encoder_dp:
        vae.encode = types.MethodType(dp_encode, vae)
    if enable_decoder_dp:
        vae.decode = types.MethodType(dp_decode, vae)

    return vae


def _boost_vae(vae):
    """Patch VAE encode/decode in-place for DP temporal tiling."""
    return apply_vae_dp(vae)
