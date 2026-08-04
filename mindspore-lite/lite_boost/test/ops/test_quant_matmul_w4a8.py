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
# pylint: disable=attribute-defined-outside-init
"""
lite_boost verification test for quant_matmul_w4a8 (ascend_a2 / ascend910b op).

The AscendC op computes a quantised INT4×INT8 matmul:
    out = ((x1_i8 @ x2_i4) * scale + bias) * pertoken_scale + output_bias
with BF16 output.  x1 is split in-place (int8→int4b_t) by the AIV phase,
so the Python wrapper clones it internally.

Cases:
  - test_output_shape_and_finite : output shape/dtype/finite, parametrised by (M, K, N).
  - test_accuracy_vs_bf16_ref    : cosine similarity vs torch.matmul BF16 reference ≥ 0.98.
  - test_optional_output_bias    : output_bias=None defaults to zeros, matches explicit zeros.
  - test_performance_vs_a8w8     : W4A8 vs A8W8 timing, parametrised by (M, K, N).
                                   Only production shape (8192,3072,3072) asserts W4A8 ≤ A8W8 × 2.

Runs on ascend910b only (``-m "ascend_a2"``).
"""

import time
import logging

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torch_npu
import lite_boost.ops as lite_ops

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COSINE_THRESHOLD = 0.98
TEST_SHAPES = [
    (8, 64, 16),
    (16, 128, 32),
    (64, 128, 128),
    (128, 256, 64),
    (128, 256, 128),
    (2128, 3072, 3072),
    (8192, 3072, 3072),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a, b):
    """Cosine similarity between two tensors (flattened to f32)."""
    a_f32 = a.float().reshape(-1)
    b_f32 = b.float().reshape(-1)
    if a_f32.norm() == 0 or b_f32.norm() == 0:
        return 1.0 if a_f32.norm() == b_f32.norm() else 0.0
    return F.cosine_similarity(a_f32.unsqueeze(0), b_f32.unsqueeze(0)).item()


def _pack_weight_int4(w_npu, K, N):
    """Per-channel absmax/7 quantise BF16 weight → int4, pack to int32 [N, K/8].

    Returns (w_packed, w_scale, bias) on NPU.
    """
    k8 = K // 8
    w_fp32 = w_npu.cpu().float().numpy()

    w_scale = np.abs(w_fp32).max(axis=0).clip(min=1e-8) / 7.0
    w_int4 = np.round(w_fp32 / w_scale).clip(-8, 7).astype(np.int8)

    bias_np = (8.0 * w_int4.astype(np.float32).sum(axis=0)).astype(np.float32)

    w_grp = w_int4.reshape(k8, 8, N).astype(np.int32) & 0xF
    shifts = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.int32).reshape(1, 8, 1)
    w_packed = (w_grp.astype(np.int64) << shifts).sum(axis=1)
    w_packed = w_packed.T.astype(np.int32).ravel()

    w_packed_t = torch.from_numpy(w_packed).to(torch.int32).reshape(N, k8).npu()
    w_scale_t = torch.from_numpy(w_scale.copy()).float().npu()
    bias_t = torch.from_numpy(bias_np.copy()).float().npu()
    return w_packed_t, w_scale_t, bias_t


def _generate_test_data(M, K, N, device, seed=42):
    """Build test tensors for shape [M,K] × [K,N] with K padded to 32-align.

    Returns dict with all inputs for both W4A8 and A8W8 paths.
    """
    k_pad = ((K + 31) // 32) * 32
    rng = np.random.RandomState(seed)

    x_bf16 = torch.from_numpy(rng.randn(M, K).astype(np.float32)).to(torch.bfloat16)
    w_bf16 = torch.from_numpy(rng.randn(K, N).astype(np.float32)).to(torch.bfloat16)
    if K != k_pad:
        x_bf16 = F.pad(x_bf16, (0, k_pad - K))
        w_bf16 = F.pad(w_bf16, (0, 0, 0, k_pad - K))

    x_npu = x_bf16.to(device)
    w_npu = w_bf16.to(device)

    # Shared activation quantisation
    x_int8, x_scale = torch_npu.npu_dynamic_quant(x_npu)

    # W4A8 weight packing
    w_packed, w4_scale, bias = _pack_weight_int4(w_npu, k_pad, N)

    # A8W8 weight quantisation
    w_int8_t, w8_scale = torch_npu.npu_dynamic_quant(w_npu.T)
    w_int8 = w_int8_t.T.contiguous()

    # BF16 ground truth
    out_bf16 = torch.matmul(x_npu, w_npu)

    return {
        "x_npu": x_npu,
        "w_npu": w_npu,
        "x_int8": x_int8,
        "x_scale": x_scale,
        "w_packed": w_packed,
        "w4_scale": w4_scale,
        "bias": bias,
        "w_int8": w_int8,
        "w8_scale": w8_scale,
        "out_bf16": out_bf16,
        "M": M, "K": K, "N": N, "k_pad": k_pad,
    }


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestQuantMatmulW4a8:
    """Verify quant_matmul_w4a8 against BF16 matmul reference (ascend910b)."""

    def setup_method(self):
        """Setup test fixtures: device, op availability check, warm-up."""
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)

        if "310P" in torch.npu.get_device_name(0):
            pytest.skip("310P detected — quant_matmul_w4a8 requires ascend910b.")

        if not hasattr(torch.ops.lite_boost, "quant_matmul_w4a8"):
            pytest.skip("torch.ops.lite_boost.quant_matmul_w4a8 not found. "
                        "Build lite_boost first: cd lite_boost && bash build.sh")

        # Warm-up: prime the NPU stream and JIT-compile the AscendC kernel
        # so that subsequent timed runs are not skewed by cold-start overhead.
        M, K, N = 128, 256, 128
        wx = torch.randn(M, K, dtype=torch.bfloat16, device=self.device)
        ww = torch.randn(K, N, dtype=torch.bfloat16, device=self.device)
        torch.matmul(wx, ww)
        torch.npu.synchronize()

        wx_int8, wx_scale = torch_npu.npu_dynamic_quant(wx)
        wp, ws, wb = _pack_weight_int4(ww, K, N)
        lite_ops.quant_matmul_w4a8(wx_int8, wp, ws, wb, pertoken_scale=wx_scale)
        torch.npu.synchronize()

    # ------------------------------------------------------------------
    # Shape + finiteness
    # ------------------------------------------------------------------

    @pytest.mark.ascend_a2
    @pytest.mark.L0
    @pytest.mark.parametrize("M,K,N", TEST_SHAPES)
    def test_output_shape_and_finite(self, M, K, N):
        """Output has correct shape [M,N], dtype bf16, and all values finite."""
        d = _generate_test_data(M, K, N, self.device)
        out = lite_ops.quant_matmul_w4a8(
            d["x_int8"], d["w_packed"], d["w4_scale"], d["bias"],
            pertoken_scale=d["x_scale"])
        torch.npu.synchronize()

        assert out.shape == (M, N), \
            f"expected ({M},{N}), got {tuple(out.shape)}"
        assert out.dtype == torch.bfloat16, \
            f"expected bfloat16, got {out.dtype}"
        assert torch.isfinite(out).all(), \
            f"output has NaN/Inf for shape ({M},{K},{N})"

    # ------------------------------------------------------------------
    # Accuracy vs BF16 reference
    # ------------------------------------------------------------------

    @pytest.mark.ascend_a2
    @pytest.mark.L0
    @pytest.mark.parametrize("M,K,N", TEST_SHAPES)
    def test_accuracy_vs_bf16_ref(self, M, K, N):
        """W4A8 cosine similarity vs torch.matmul BF16 reference ≥ 0.98."""
        d = _generate_test_data(M, K, N, self.device)
        out = lite_ops.quant_matmul_w4a8(
            d["x_int8"], d["w_packed"], d["w4_scale"], d["bias"],
            pertoken_scale=d["x_scale"])
        torch.npu.synchronize()

        cos = _cosine_similarity(out.cpu(), d["out_bf16"].cpu())
        logging.info("[accuracy %dx%dx%d] cos vs BF16 ref = %.6f", M, K, N, cos)
        assert cos >= COSINE_THRESHOLD, \
            f"cosine {cos:.6f} < {COSINE_THRESHOLD} for shape ({M},{K},{N})"

    # ------------------------------------------------------------------
    # Optional output_bias
    # ------------------------------------------------------------------

    @pytest.mark.ascend_a2
    @pytest.mark.L0
    def test_optional_output_bias(self):
        """output_bias=None (default zeros) matches explicit zero output_bias."""
        M, K, N = 64, 128, 128
        d = _generate_test_data(M, K, N, self.device)

        out_default = lite_ops.quant_matmul_w4a8(
            d["x_int8"], d["w_packed"], d["w4_scale"], d["bias"],
            pertoken_scale=d["x_scale"])  # output_bias=None
        torch.npu.synchronize()

        zeros = torch.zeros(N, dtype=torch.float32, device=self.device)
        out_explicit = lite_ops.quant_matmul_w4a8(
            d["x_int8"], d["w_packed"], d["w4_scale"], d["bias"],
            pertoken_scale=d["x_scale"], output_bias=zeros)
        torch.npu.synchronize()

        max_diff = (out_default.float() - out_explicit.float()).abs().max().item()
        logging.info("[optional_output_bias] max diff (default vs explicit zeros) = %.6e",
                     max_diff)
        assert max_diff == 0.0, \
            f"output_bias default vs explicit zeros mismatch: max diff {max_diff}"

    # ------------------------------------------------------------------
    # Performance vs A8W8
    # ------------------------------------------------------------------

    @pytest.mark.ascend_a2
    @pytest.mark.parametrize("M,K,N", TEST_SHAPES)
    def test_performance_vs_a8w8(self, M, K, N):
        """W4A8 vs A8W8 timing.  Only production shape (8192,3072,3072) has a
        W4A8 ≤ A8W8 × 2 threshold; all other shapes are informational."""
        PROD_SHAPE = (8192, 3072, 3072)
        d = _generate_test_data(M, K, N, self.device)
        n_warmup = 3
        n_iters = 10

        # W4A8 warm-up + benchmark
        for _ in range(n_warmup):
            lite_ops.quant_matmul_w4a8(
                d["x_int8"], d["w_packed"], d["w4_scale"], d["bias"],
                pertoken_scale=d["x_scale"])
        torch.npu.synchronize()

        t0 = time.time()
        for _ in range(n_iters):
            lite_ops.quant_matmul_w4a8(
                d["x_int8"], d["w_packed"], d["w4_scale"], d["bias"],
                pertoken_scale=d["x_scale"])
        torch.npu.synchronize()
        w4a8_ms = (time.time() - t0) / n_iters * 1000

        # A8W8 warm-up + benchmark
        for _ in range(n_warmup):
            torch_npu.npu_quant_matmul(
                d["x_int8"], d["w_int8"], d["w8_scale"],
                pertoken_scale=d["x_scale"],
                output_dtype=torch.bfloat16)
        torch.npu.synchronize()

        t0 = time.time()
        for _ in range(n_iters):
            torch_npu.npu_quant_matmul(
                d["x_int8"], d["w_int8"], d["w8_scale"],
                pertoken_scale=d["x_scale"],
                output_dtype=torch.bfloat16)
        torch.npu.synchronize()
        a8w8_ms = (time.time() - t0) / n_iters * 1000

        ratio = w4a8_ms / a8w8_ms if a8w8_ms > 0 else float("inf")
        logging.info("[perf %dx%dx%d] W4A8 %.3f ms  |  A8W8 %.3f ms  |  ratio %.2fx",
                     M, K, N, w4a8_ms, a8w8_ms, ratio)

        if (M, K, N) == PROD_SHAPE:
            assert w4a8_ms <= a8w8_ms * 2, \
                (f"W4A8 ({w4a8_ms:.3f} ms) > 2x A8W8 ({a8w8_ms:.3f} ms) "
                 f"for production shape ({M},{K},{N})")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t = TestQuantMatmulW4a8()
    t.setup_method()
    for shape in TEST_SHAPES:
        t.test_output_shape_and_finite(*shape)
        logging.info("shape_and_finite    %-20s: PASS", str(shape))
        t.test_accuracy_vs_bf16_ref(*shape)
        logging.info("accuracy            %-20s: PASS", str(shape))
        t.test_performance_vs_a8w8(*shape)
        logging.info("performance         %-20s: PASS", str(shape))
    t.test_optional_output_bias()
    logging.info("optional_output_bias:    PASS")
