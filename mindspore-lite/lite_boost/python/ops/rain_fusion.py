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
lite_boost custom ops
- rain_fusion_attention
"""

import os
from pathlib import Path
import torch

_LOADED = False


def _resolve_default_so_path() -> str:
    """Resolve default shared library path."""
    env_path = os.getenv("LITE_BOOST_OPS_LIB")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return str(p.resolve())

    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / "lib" / "liblite_boost_ops.so",
        pkg_dir / "lib" / "lite_boost_ops.so",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "lite_boost_ops shared library not found. "
        "Set LITE_BOOST_OPS_LIB or install a wheel containing lite_boost/lib/*.so."
    )


def _load_library(path=None):
    """Load shared library."""
    global _LOADED
    if _LOADED:
        return
    lib_path = path if path is not None else _resolve_default_so_path()
    torch.ops.load_library(lib_path)
    _LOADED = True


def ops():
    """Get ops."""
    _load_library()
    return torch.ops.lite_boost


_load_library()


def rain_fusion_attention(
    query,
    key,
    value,
    select_idx,
    select_num_idx,
    block_shape,
    attn_mask=None,
    actual_seq_lengths=None,
    actual_seq_lengths_kv=None,
    block_table=None,
    q_input_layout="TND",
    kv_input_layout="TND",
    num_key_value_heads=1,
    mask_type=0,
    scale_value=1.0,
    inner_precise=1,
    block_size=0,
):
    """Rain fusion attention."""
    return torch.ops.lite_boost.rain_fusion_attention(
        query,
        key,
        value,
        select_idx,
        select_num_idx,
        block_shape,
        attn_mask,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        block_table,
        q_input_layout,
        kv_input_layout,
        num_key_value_heads,
        mask_type,
        scale_value,
        inner_precise,
        block_size,
    )


__all__ = ["rain_fusion_attention"]
