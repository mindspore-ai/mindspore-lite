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
Wan2.2 Ulysses Sequence Parallel (USP) adapters.

Provides usp_attn_forward and usp_dit_forward as drop-in replacements
for WanSelfAttention.forward and WanModel.forward respectively.
"""
from lite_boost.model.wan2_2.model import (
    usp_attn_forward,
    usp_dit_forward,
)

# Re-export VAE DP
from lite_boost.model.wan2_2.boost import DPConfig, apply_vae_dp  # noqa: F401

__all__ = [
    "usp_attn_forward",
    "usp_dit_forward",
]
