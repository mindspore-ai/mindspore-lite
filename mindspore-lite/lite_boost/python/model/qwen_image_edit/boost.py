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
Model-specific adaptation for Qwen-Image-Edit Ulysses Sequence Parallel on NPU.

Called by model.setup_model() to patch the pipeline in-place.
Accepts QwenImageEditPlusPipeline objects.
"""
import types

import torch
import torch.distributed as dist

from . import USPQwenDoubleStreamAttnProcessor, usp_dit_forward, patch_eager_sdpa

# Parallel algorithms currently supported per module (Parallel.dit/vae.alg).
_DIT_PARALLEL_ALGS = ('CP',)
_VAE_PARALLEL_ALGS = ('DP',)


def boost_qwen_image_edit(pipe, config=None):
    """
    Patch a QwenImageEditPlusPipeline in-place for NPU multi-card USP.

    Operations:
    1. Globally replace F.scaled_dot_product_attention with an eager
       implementation (NPU fusion SDPA op is unsupported).
    2. Replace each transformer block's attention processor with the USP
       joint-attention processor (text replication + head slicing, image
       latent sequence split via all_to_all).
    3. Replace transformer.forward -> usp_dit_forward (sequence split/gather).
    4. Replace the VAE with the parallel tiled encode/decode version
       (``Parallel.vae`` is validated when configured).

    ``pipe`` may be a QwenImageEditPlusPipeline or the raw
    QwenImageTransformer2DModel (in which case only the DiT is patched).

    ``config`` is the dict parsed from the boost YAML file (see
    ``qwen_image_edit.yaml``). Unconfigured sections (or ``config=None``)
    keep the best-performance defaults: DiT CP at the distributed world
    size with the parallel VAE enabled.
    """
    transformer = pipe.transformer if hasattr(pipe, "transformer") else pipe
    is_pipeline = transformer is not pipe

    world_size = _parse_parallel_config(config)

    num_heads = transformer.config.num_attention_heads
    if num_heads % world_size != 0:
        raise ValueError(
            f"num_attention_heads ({num_heads}) must be divisible by "
            f"world_size ({world_size})"
        )

    patch_eager_sdpa()

    if world_size > 1:
        for block in transformer.transformer_blocks:
            block.attn.processor = USPQwenDoubleStreamAttnProcessor()
        transformer.forward = types.MethodType(usp_dit_forward, transformer)

    if is_pipeline:
        _patch_vae_edit(pipe)
        restore_fp16_params(pipe.vae, dtype=pipe.vae.dtype)

    return pipe


def _parse_parallel_config(config):
    """Read the ``Parallel.dit`` / ``Parallel.vae`` sections of the boost config.

    Returns the DiT world size. Unconfigured sections (or ``config=None``)
    keep the best-performance defaults — DiT CP at the distributed world
    size with the parallel VAE enabled; a missing key falls back to its
    default (``alg`` → the sole supported algorithm, ``world_size`` → the
    DiT world size).

    Configured values are validated: ``dit.alg`` must be one of
    ``_DIT_PARALLEL_ALGS`` and ``vae.alg`` one of ``_VAE_PARALLEL_ALGS``;
    ``dit.world_size`` must match the distributed world size, and
    ``vae.world_size`` must equal ``dit.world_size``.

    The yaml schema (see ``qwen_image_edit.yaml``) is::

        Parallel:
          dit:
            alg: CP  # current support [CP]
            world_size: 2
          vae:
            alg: DP  # current support [DP]
            world_size: 2
    """
    dist_world_size = dist.get_world_size()
    parallel = (config or {}).get('Parallel') or {}

    dit = parallel.get('dit')
    if dit is None:
        dit_world_size = dist_world_size  # section absent → best-performance default
    else:
        _check_alg(dit, 'dit', _DIT_PARALLEL_ALGS)
        dit_world_size = dit.get('world_size') or dist_world_size
        _check_world_size(dit_world_size, 'dit', dist_world_size, 'the distributed world size')

    vae = parallel.get('vae')
    if vae is not None:
        _check_alg(vae, 'vae', _VAE_PARALLEL_ALGS)
        vae_world_size = vae.get('world_size') or dit_world_size
        _check_world_size(vae_world_size, 'vae', dit_world_size, 'Parallel.dit.world_size')

    return dit_world_size


def _check_alg(section, module, supported_algs):
    """Validate Parallel.<module>.alg against the supported algorithms."""
    alg = section.get('alg', supported_algs[0])
    if alg not in supported_algs:
        raise ValueError(
            f"Parallel.{module}.alg {alg!r} is unsupported; "
            f"expected one of {supported_algs}"
        )


def _check_world_size(world_size, module, expected_world_size, expected_desc):
    """Ensure the configured world size matches the expected value."""
    if world_size != expected_world_size:
        raise ValueError(
            f"Parallel.{module}.world_size ({world_size}) must match "
            f"{expected_desc} ({expected_world_size})"
        )


def _patch_vae_edit(pipe):
    """Swap the VAE for the parallel tiled encode/decode implementation."""
    from lite_boost.model.qwenimage import AutoencoderKLQwenImage as ParallelVAE

    vae = pipe.vae
    if isinstance(vae, ParallelVAE):
        new_vae = vae
    else:
        new_vae = ParallelVAE.from_config(vae.config)
        new_vae.load_state_dict(vae.state_dict())
        new_vae.to(device=vae.device, dtype=vae.dtype)
        pipe.vae = new_vae
    new_vae.enable_tiling()


def restore_fp16_params(module, dtype=torch.float16):
    """torch_npu casts some low-bit params (4D conv weights, norm gammas) back
    to fp32 during .to("npu"); restore every fp32 param to the target dtype."""
    for p in module.parameters():
        if p.dtype == torch.float32:
            p.data = p.data.to(dtype)
