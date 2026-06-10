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
Model-specific adaptation for Wan2.1 Ulysses Sequence Parallel on NPU.

Called by model.setup_model() to patch the pipeline in-place.
Accepts Wan2.1 pipeline objects: WanT2V, WanI2V, WanVace, WanVaceMP.
"""
import types

import torch.distributed as dist

from lite_boost.layers.attention import flash_attention as lite_flash_attention
from . import usp_attn_forward, usp_dit_forward


def boost_wan2_1(pipe):
    """
    Patch a Wan2.1 pipeline in-place for NPU Ulysses SP.

    The WanModel is extracted from pipe.model.

    Operations:
    1. Replace flash_attention in wan.modules with NPU-compatible version
    2. Replace each block.self_attn.forward → usp_attn_forward
    3. Replace model.forward → usp_dit_forward
    4. If VACE model: replace vace_blocks and forward_vace
    """
    model = pipe.model
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

    # Also patch vace_blocks for VACE models
    if hasattr(model, 'vace_blocks'):
        from . import usp_dit_forward_vace
        for block in model.vace_blocks:
            block.self_attn.seq_pad = 0
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn
            )
        model.forward_vace = types.MethodType(
            usp_dit_forward_vace, model
        )

    model.forward = types.MethodType(usp_dit_forward, model)
