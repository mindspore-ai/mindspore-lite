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
QuantMatmulW4a8 operator Python binding

INT4xINT8 matmul -> BF16 output via CANN aclnnQuantMatmulW4a8.
"""

import torch

# Ensure the C++ extension .so is loaded (torch.ops.lite_boost.*).
# _load_library is idempotent -- guarded by a module-level _LOADED flag.
from .rain_fusion import _load_library

_load_library()


def quant_matmul_w4a8(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    pertoken_scale: torch.Tensor,
    output_bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    r"""
    INT4xINT8 quantised matrix multiplication -> BF16 output.

    Computes ``out = ((x1_i8 @ x2_i4) * scale + bias) * pertoken_scale``
    with per-token activation quantisation (pertoken_scale) and
    optional output bias (pass None to skip; defaults to zeros).

    Args:
        x1 (Tensor): INT8 activation of shape ``[M, K]``.
        x2 (Tensor): Packed INT4 weight of shape ``[N, K//8]``, dtype=int32.
            Each int32 packs 8 consecutive int4 along the K dimension.
        scale (Tensor): Per-channel dequant scale of shape ``[N]``, dtype=float32.
        bias (Tensor): Per-channel bias (host pre-multiplied by scale).
            Shape ``[N]``, dtype=float32.  Required.
        pertoken_scale (Tensor): Per-token activation scale of shape
            ``[M]``, dtype=float32.  Required.
        output_bias (Tensor, optional): Output bias of shape ``[N]``,
            dtype=float32.  Defaults to None (filled with zeros internally).
        output_dtype (torch.dtype, optional): Output data type.  Currently only
            ``torch.bfloat16`` is supported.

    Returns:
        Tensor, output of shape ``[M, N]``.
    """
    if output_dtype != torch.bfloat16:
        raise ValueError(f"output_dtype={output_dtype} is not supported. "
                         "Only torch.bfloat16 is supported.")
    if output_bias is None:
        output_bias = torch.zeros(bias.shape[0], dtype=torch.float32, device=x1.device)
    # clone x1 — kernel AIV split overwrites act in-place (int8→int4b_t)
    return torch.ops.lite_boost.quant_matmul_w4a8(
        x1.clone(), x2, scale, bias, pertoken_scale, output_bias
    )
