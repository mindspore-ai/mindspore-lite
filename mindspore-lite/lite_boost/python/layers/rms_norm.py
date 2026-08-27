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
"""NPU RMS normalization primitives.

Fused replacement for the expanded ``F.normalize(x, dim) * sqrt(dim) *
gamma`` chain (7+ ops) used by Wan-series VAE ``RMS_norm`` layers, and
for the ``x / RMS(x) * weight`` form used by DiT qk-norm (``WanRMSNorm``).

``npu_rms_norm`` accepts N-D inputs natively on most SoCs (910B etc.);
the 310P3 binary set is 2D-only (probe-verified: higher-rank inputs tile
incorrectly; fp16 / fp32 accepted, C >= 8).  This primitive therefore
passes N-D inputs straight through on supporting SoCs and folds to 2D
``(N, C)`` internally on 310P3 — callers only need to place the
normalized dim last (e.g. transpose for channel-first layouts); they
never reshape.

Norm is per-last-dim: ``y = x / RMS(x) * gamma`` with
``RMS(x) = sqrt(mean(x^2))``, so the ``sqrt(dim)`` scale is NOT baked
into gamma (RMS normalization cancels it; equivalence verified with
rel error ~6e-4 in fp16).
"""
import torch
import torch_npu

__all__ = ["rms_norm"]

_FORCE_2D = None


def _needs_2d_collapse():
    """True only on 310P3 (npu_rms_norm is 2D-only there).  Cached."""
    global _FORCE_2D
    if _FORCE_2D is None:
        try:
            name = torch_npu.npu.get_device_name(torch_npu.npu.current_device())
            _FORCE_2D = "310P" in name
        except Exception:
            _FORCE_2D = False
    return _FORCE_2D


def rms_norm(x, gamma, eps=1e-6):
    """Apply per-row RMS normalization over the last dim.

    Args:
        x:      Input of any shape; the normalized dim must be last.
        gamma:  Per-column scale, ``(C,)`` broadcastable.
        eps:    Epsilon (kept for API parity; no effect on 310P3).

    Returns:
        Normalized tensor, same shape as the input (the ``rstd`` second
        return value of ``npu_rms_norm`` is discarded).
    """
    if not torch.is_tensor(x) or not torch.is_tensor(gamma):
        raise TypeError(
            "rms_norm: x and gamma must be torch.Tensor, got "
            f"{type(x).__name__} / {type(gamma).__name__}")
    if x.dim() != 2:
        # The pipeline consumer (diffsynth adapter, VAE RMS_norm layers)
        # always calls with a folded (N, C) input; higher-rank inputs
        # are not supported.
        raise ValueError(
            "rms_norm: supports 2-D (N, C) inputs only "
            f"(reshape higher-rank tensors before calling), got dim="
            f"{x.dim()} shape={tuple(x.shape)}")
    if gamma.dim() != 1 or gamma.numel() != x.shape[-1]:
        raise ValueError(
            "rms_norm: gamma must be 1-D and match the last dim of x "
            f"(x.shape={tuple(x.shape)}, gamma.shape={tuple(gamma.shape)})")
    if _needs_2d_collapse():
        # 310P3 only — restrictions (measured on 2026-08-26):
        #   * bf16 has no kernel under jit_compile=False (which the rope
        #     module sets at import time): RuntimeError 161002 — reject
        #     up front
        #   * results are unreliable below C = 16: C < 8 always wrong
        #     (fp16/fp32), fp16 C = 8 wrong / NaN for > 2 rows
        #     (fp32 C = 8 happens to be fine, but the bound is tightened
        #     to a single C >= 16 rule for simplicity)
        # Restrict those combinations explicitly instead of returning
        # wrong results.  The wan2.2 pipeline (VAE RMS_norm, fp16,
        # C >= 128) is never affected.
        if x.dtype == torch.bfloat16:
            raise ValueError(
                "rms_norm: on 310P3 bf16 is not supported (no bf16 "
                "RmsNorm kernel under jit_compile=False), got dtype="
                f"{x.dtype}")
        if x.shape[-1] < 16:
            raise ValueError(
                "rms_norm: on 310P3 requires last dim C >= 16 "
                f"(results unreliable below), got C={x.shape[-1]}")
    y, _ = torch_npu.npu_rms_norm(x, gamma, eps)
    return y
