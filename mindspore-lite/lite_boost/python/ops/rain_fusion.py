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
- sparse_attention
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import torch
from .sparse_attention import sparse_attention  # pylint: disable=unused-import

_LOADED = False


def _resolve_default_so_path() -> str:
    """Resolve default shared library path."""
    env_path = os.getenv("LITE_BOOST_OPS_LIB")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return str(p.resolve())

    pkg_dir = Path(__file__).resolve().parent.parent
    candidates = [
        pkg_dir / "lib" / "liblite_boost_ops.so",
        pkg_dir / "lib" / "lite_boost_ops.so",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    # Fallback: search installed package location via sys.path
    import sys as _sys  # pylint: disable=import-outside-toplevel
    for entry in _sys.path:
        for soname in ("liblite_boost_ops.so", "lite_boost_ops.so"):
            p = Path(entry) / "lite_boost" / "lib" / soname
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


def rain_fusion_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    select_idx: torch.Tensor,
    select_num_idx: torch.Tensor,
    block_shape: list[int],
    attn_mask: Optional[torch.Tensor] = None,
    actual_seq_lengths: Optional[list[int]] = None,
    actual_seq_lengths_kv: Optional[list[int]] = None,
    block_table: Optional[torch.Tensor] = None,
    q_input_layout: str = "TND",
    kv_input_layout: str = "TND",
    num_key_value_heads: int = 1,
    mask_type: int = 0,
    scale_value: float = 1.0,
    inner_precise: int = 1,
    block_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Block-sparse fusion attention forward computation.

    Calls the NPU native ``aclnnRainFusionAttention`` operator to perform block-level sparse fused attention.
    Supports flexible non-uniform sparse attention patterns by specifying per query-block KV block indices
    via ``select_idx`` and ``select_num_idx``. Input tensors support ``float16`` and ``bfloat16`` dtypes.

    Args:
        query (Tensor): Query tensor. Layout depends on ``q_input_layout``.
        key (Tensor): Key tensor. Layout depends on ``kv_input_layout``.
        value (Tensor): Value tensor. Layout depends on ``kv_input_layout``.
        select_idx (Tensor): Sparse selection index matrix. Shape ``[q_blocks, num_heads, kv_blocks]``,
            dtype ``torch.int64``. Valid KV block indices are sorted in ascending order per row,
            with remaining positions filled with ``-1``.
        select_num_idx (Tensor): Number of valid KV blocks per query block and head.
            Shape ``[q_blocks, num_heads]``, dtype ``torch.int64``.
        block_shape (list[int]): Block tile size in ``[block_rows, block_cols]``, typically ``[128, 128]``.
        attn_mask (Tensor, optional): Attention mask tensor. Default: ``None``.
        actual_seq_lengths (list[int], optional): Actual Q sequence length per batch.
            Required when layout is ``"TND"`` to correctly compute sequence boundaries.
            Default: ``None``.
        actual_seq_lengths_kv (list[int], optional): Actual KV sequence length per batch.
            Default: ``None``.
        block_table (Tensor, optional): Block table for PagedAttention scenarios. Default: ``None``.
        q_input_layout (str, optional): Query input layout, ``"TND"`` or ``"BNSD"``. Default: ``"TND"``.
        kv_input_layout (str, optional): Key/Value input layout, ``"TND"`` or ``"BNSD"``. Default: ``"TND"``.
        num_key_value_heads (int, optional): Number of KV heads for GQA/MQA. Default: ``1``.
        mask_type (int, optional): Mask type (``0`` for causal mask). Default: ``0``.
        scale_value (float, optional): Attention scale factor. Recommended: ``head_dim ** -0.5``.
            Default: ``1.0``.
        inner_precise (int, optional): Precision mode. ``0`` for high-precision, ``1`` for
            high-performance. Default: ``1``.
        block_size (int, optional): Block size. ``0`` for automatic inference. Default: ``0``.

    Returns:
        tuple[Tensor, Tensor]:

            - **attention_out** (Tensor) — Attention output, same shape as ``query``.
            - **softmax_lse** (Tensor) — Softmax log-sum-exp values with shape ``[T, N, H]``,
              for debugging and gradient backpropagation.

    Raises:
        TypeError: If ``query``, ``key``, ``value``, ``select_idx`` or ``select_num_idx`` is not a Tensor.

    Examples:
        >>> # Build dense attention with rain_fusion_attention
        >>> import math
        >>> import torch
        >>> import lite_boost.ops as lite_ops
        >>> device = torch.device("npu:0")
        >>> batch_size, num_heads, seq_len, head_dim = 1, 3, 4096, 128
        >>> block_size = 128
        >>> scale = head_dim ** -0.5
        >>> q = torch.randn(batch_size * seq_len, num_heads, head_dim,
        ...                 dtype=torch.float16, device=device)
        >>> k = torch.randn(batch_size * seq_len, num_heads, head_dim,
        ...                 dtype=torch.float16, device=device)
        >>> v = torch.randn(batch_size * seq_len, num_heads, head_dim,
        ...                 dtype=torch.float16, device=device)
        >>> q_blocks = math.ceil(seq_len / block_size)
        >>> kv_blocks = math.ceil(seq_len / block_size)
        >>> select_idx = torch.full((q_blocks, num_heads, kv_blocks), -1,
        ...                         dtype=torch.int64, device=device)
        >>> base_indices = torch.arange(kv_blocks, dtype=torch.int64, device=device)
        >>> select_idx[...] = base_indices.repeat(q_blocks, num_heads, 1)
        >>> select_num_idx = torch.full((q_blocks, num_heads), kv_blocks,
        ...                             dtype=torch.int64, device=device)
        >>> attention_out, softmax_lse = lite_ops.rain_fusion_attention(
        ...     query=q, key=k, value=v,
        ...     select_idx=select_idx, select_num_idx=select_num_idx,
        ...     block_shape=[block_size, block_size],
        ...     scale_value=scale,
        ...     actual_seq_lengths=[seq_len],
        ...     actual_seq_lengths_kv=[seq_len])
        >>> print(attention_out.shape)
        torch.Size([4096, 3, 128])
    """
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


_load_library()


__all__ = ["rain_fusion_attention", "sparse_attention"]
