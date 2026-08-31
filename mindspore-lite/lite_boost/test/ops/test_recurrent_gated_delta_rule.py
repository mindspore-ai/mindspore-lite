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
lite_boost test recurrent_gated_delta_rule

CANN operator constraints (from official documentation):
  - 0 < Li <= 8 (per-batch sequence length)
  - 0 < Nk <= 256, Nk <= Nv <= 256, Nv % Nk == 0
  - 0 < Dk <= 512, 0 < Dv <= 512
  - 0 <= query[i][j][k] <= 1 (L2 normalized)
  - 0 <= key[i][j][k] <= 1  (L2 normalized)
  - g[i][j] < 0 (decay gate)
  - gk[i][j][k] < 0 (key gate)
  - 0 < beta[i][j] < 1
"""

import time
import logging

import pytest
import torch
import lite_boost.ops as lite_ops
logging.basicConfig(level=logging.INFO, format='%(message)s')


# ---------------------------------------------------------------------------
# PyTorch reference implementation (ported from Qwen3.5 export script)
# ---------------------------------------------------------------------------
def _l2norm(x, dim=-1, eps=1e-6):
    """Apply L2 normalization along the given dimension."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _pytorch_recurrent_gated_delta_rule(
    query, key, value, g, gk, beta, initial_state, scale=1.0
):
    # pylint: disable=too-many-locals
    """Reference PyTorch implementation of recurrent GatedDeltaRule for decode.

    Args:
        query:         [B, T, H, D_k] float32 (L2 normalized)
        key:           [B, T, H, D_k] float32 (L2 normalized)
        value:         [B, T, H, D_v] float32
        g:             [B, T, H] float32 (negative decay gate)
        gk:            [B, T, H, D_k] float32 (negative key gate)
        beta:          [B, T, H] float32 in (0, 1)
        initial_state: [B, H, D_k, D_v] float32
        scale:         float

    Returns:
        (output, final_state)
        output:       [B, T, H, D_v] float32
        final_state:  [B, H, D_k, D_v] float32
    """
    batch_size, seq_len, num_key_heads, _ = query.shape
    num_value_heads = value.shape[2]
    if num_value_heads % num_key_heads != 0:
        raise ValueError("num_value_heads must be divisible by num_key_heads")
    v_head_dim = value.shape[-1]

    # Transpose to [B, H, T, D]
    q = query.transpose(1, 2)  # [B, H, T, D_k]
    k = key.transpose(1, 2)  # [B, H, T, D_k]
    v = value.transpose(1, 2)  # [B, H, T, D_v]
    head_ratio = num_value_heads // num_key_heads
    q = q.repeat_interleave(head_ratio, dim=1)
    k = k.repeat_interleave(head_ratio, dim=1)
    g_t = g.transpose(1, 2)  # [B, H, T]
    gk_t = gk.transpose(1, 2)  # [B, H, T, D_k]
    beta_t = beta.transpose(1, 2)  # [B, H, T]

    q = q * scale

    core_attn_out = torch.zeros(
        batch_size,
        num_value_heads,
        seq_len,
        v_head_dim,
        dtype=torch.float32,
        device=query.device,
    )
    last_recurrent_state = initial_state.clone()

    for i in range(seq_len):
        q_i = q[:, :, i]  # [B, H, D_k]
        k_i = k[:, :, i]  # [B, H, D_k]
        v_i = v[:, :, i]  # [B, H, D_v]
        g_i = g_t[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        gk_i = gk_t[:, :, i, :].exp().unsqueeze(-1)  # [B, H, D_k, 1]
        beta_i = beta_t[:, :, i].unsqueeze(-1)  # [B, H, 1]

        last_recurrent_state = last_recurrent_state * g_i
        last_recurrent_state = last_recurrent_state * gk_i
        kv_mem = (last_recurrent_state * k_i.unsqueeze(-1)).sum(dim=-2)  # [B, H, D_v]
        delta = (v_i - kv_mem) * beta_i  # [B, H, D_v]
        last_recurrent_state = last_recurrent_state + k_i.unsqueeze(
            -1
        ) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_i.unsqueeze(-1)).sum(dim=-2)

    # Transpose back to [B, T, H, D_v]
    output = core_attn_out.transpose(1, 2).contiguous()
    return output, last_recurrent_state


def _print_accuracy_comparison(tag, out_cann, out_ref, state_cann, state_ref):
    """Print structured accuracy comparison between CANN op and PyTorch reference.

    Args:
        tag:       Test case label (e.g. "L0 single token").
        out_cann:  CANN operator output (bfloat16, BNSD layout).
        out_ref:   Reference output (bfloat16, BNSD layout, same shape as out_cann).
        state_cann:  CANN operator state (bfloat16).
        state_ref:   Reference state (float32 or bfloat16, same shape as state_cann).
    """
    out_diff = (out_cann.float() - out_ref.float()).abs()
    state_diff = (state_cann.float() - state_ref.float()).abs()

    logging.info("=" * 60)
    logging.info("  [%s] Accuracy comparison: Optimized (CANN op) vs Baseline (PyTorch ref)", tag)
    logging.info("=" * 60)
    logging.info("  Output  shape: %s", out_cann.shape)
    logging.info("    Max Absolute Diff:  %.6f", out_diff.max().item())
    logging.info("    Mean Absolute Diff: %.6f", out_diff.mean().item())
    logging.info("  State   shape: %s", state_cann.shape)
    logging.info("    Max Absolute Diff:  %.6f", state_diff.max().item())
    logging.info("    Mean Absolute Diff: %.6f", state_diff.mean().item())
    logging.info("=" * 60)


def _print_performance_comparison(tag, cann_time, ref_time):
    """Print structured performance comparison between CANN op and PyTorch reference.

    Args:
        tag:       Test case label.
        cann_time: CANN operator average time in ms.
        ref_time:  PyTorch reference average time in ms.
    """
    speedup = ref_time / cann_time if cann_time > 0 else float("inf")
    logging.info("=" * 60)
    logging.info("  [%s] Performance comparison: Optimized (CANN op) vs Baseline (PyTorch ref)", tag)
    logging.info("=" * 60)
    logging.info("  Optimized (CANN op)    Avg time: %.3f ms", cann_time)
    logging.info("  Baseline (PyTorch ref) Avg time: %.3f ms", ref_time)
    logging.info("  Speedup: %.2fx", speedup)
    logging.info("=" * 60)


# pylint: disable=too-many-locals
def _generate_test_data(batch_size, num_heads, seq_len, k_head_dim, v_head_dim, device):
    """Generate test data satisfying CANN operator constraints.

    Returns dict with all tensors in BNSD layout (as expected by CANN op).
    """
    scale = 1.0 / (k_head_dim**0.5)

    # query/key: L2 normalize to ensure values in [0, 1]
    query_f32 = _l2norm(
        torch.randn(batch_size, seq_len, num_heads, k_head_dim, device=device), dim=-1
    )
    key_f32 = _l2norm(
        torch.randn(batch_size, seq_len, num_heads, k_head_dim, device=device), dim=-1
    )
    value_f32 = torch.randn(batch_size, seq_len, num_heads, v_head_dim, device=device)

    # beta: sigmoid ensures (0, 1)
    beta_f32 = torch.randn(batch_size, seq_len, num_heads, device=device).sigmoid()

    # g: must be negative (decay gate)
    g_f32 = -(
        torch.rand(batch_size, num_heads, seq_len, dtype=torch.float32, device=device)
        + 0.01
    )

    # gk: must be negative (key gate), shape [B, H, T, D_k]
    gk_f32 = -(
        torch.rand(
            batch_size,
            num_heads,
            seq_len,
            k_head_dim,
            dtype=torch.float32,
            device=device,
        )
        + 0.01
    )

    # state: [B, H, D_k, D_v]
    state_f32 = torch.randn(
        batch_size, num_heads, k_head_dim, v_head_dim, device=device
    )

    # Convert to CANN op layout: BNSD for q/k/v, BN for g/beta
    query_bnsd = query_f32.transpose(1, 2).contiguous()  # [B, H, T, D_k]
    key_bnsd = key_f32.transpose(1, 2).contiguous()  # [B, H, T, D_k]
    value_bnsd = value_f32.transpose(1, 2).contiguous()  # [B, H, T, D_v]
    beta_bn = beta_f32.transpose(1, 2).contiguous()  # [B, H, T]

    return {
        "query_bf16": query_bnsd.bfloat16(),
        "key_bf16": key_bnsd.bfloat16(),
        "value_bf16": value_bnsd.bfloat16(),
        "beta_bf16": beta_bn.bfloat16(),
        "state_bf16": state_f32.bfloat16(),
        "g_f32": g_f32,
        "gk_f32": gk_f32,
        "query_f32_bthd": query_f32,
        "key_f32_bthd": key_f32,
        "value_f32_bthd": value_f32,
        "beta_f32_bth": beta_f32,
        "g_f32_bth": g_f32.permute(0, 2, 1).contiguous(),  # [B, T, H] for reference
        "gk_f32_bthd": gk_f32.permute(
            0, 2, 1, 3
        ).contiguous(),  # [B, T, H, D_k] for reference
        "state_f32": state_f32,
        "actual_seq_lengths": torch.tensor(
            [seq_len] * batch_size, dtype=torch.int32, device=device
        ),
        "ssm_state_indices": torch.tensor(
            list(range(batch_size)), dtype=torch.int32, device=device
        ),
        "num_accepted_tokens": torch.tensor(
            [seq_len] * batch_size, dtype=torch.int32, device=device
        ),
        "scale": scale,
    }


def _generate_300iduo_test_data(batch_size, seq_len, device):
    """Generate FP16 inputs using the Qwen3.5-4B recurrent-attention shape."""
    num_key_heads, num_value_heads = 16, 32
    k_head_dim = v_head_dim = 128
    scale = 1.0 / (k_head_dim**0.5)
    query = _l2norm(
        torch.randn(batch_size, seq_len, num_key_heads, k_head_dim),
        dim=-1,
    )
    key = _l2norm(
        torch.randn(batch_size, seq_len, num_key_heads, k_head_dim),
        dim=-1,
    )
    value = torch.randn(batch_size, seq_len, num_value_heads, v_head_dim) * 0.05
    beta = torch.rand(batch_size, seq_len, num_value_heads) * 0.8 + 0.1
    g = -(torch.rand(batch_size, seq_len, num_value_heads) + 0.01)
    gk = -(torch.rand(batch_size, seq_len, num_value_heads, k_head_dim) + 0.01)
    state = (
        torch.randn(
            batch_size,
            num_value_heads,
            k_head_dim,
            v_head_dim,
        )
        * 0.05
    )
    return {
        "query": query.transpose(1, 2).contiguous().half().to(device),
        "key": key.transpose(1, 2).contiguous().half().to(device),
        "value": value.transpose(1, 2).contiguous().half().to(device),
        "beta": beta.transpose(1, 2).contiguous().half().to(device),
        "state": state.half().to(device),
        "g": g.transpose(1, 2).contiguous().to(device),
        "gk": gk.permute(0, 2, 1, 3).contiguous().to(device),
        "actual_seq_lengths": torch.full(
            (batch_size,), seq_len, dtype=torch.int32, device=device
        ),
        "ssm_state_indices": torch.arange(batch_size, dtype=torch.int32)
        .repeat_interleave(seq_len)
        .to(device),
        "num_accepted_tokens": torch.full(
            (batch_size,), seq_len, dtype=torch.int32, device=device
        ),
        "query_ref": query.half().float(),
        "key_ref": key.half().float(),
        "value_ref": value.half().float(),
        "beta_ref": beta.half().float(),
        "state_ref": state.half().float(),
        "g_ref": g,
        "gk_ref": gk,
        "scale": scale,
    }


class TestRecurrentGatedDeltaRule:
    """
    Test recurrent_gated_delta_rule operator.

    CANN operator constraints:
      - Li <= 8 (per-batch sequence length upper bound is 8)
      - query/key must be L2-normalized (value range [0, 1])
      - g < 0, gk < 0 (negative decay gates)
      - 0 < beta < 1
    """

    def setup_method(self):
        """
        Setup test fixtures before each test method.
        Uses Qwen3.5-2B decode config: num_heads=64, Dk=64, Dv=512, seq_len=1.
        """
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)

        self.batch_size = 1
        self.num_heads = 64
        self.seq_len = 1
        self.k_head_dim = 64
        self.v_head_dim = 512

        self.data = _generate_test_data(
            self.batch_size,
            self.num_heads,
            self.seq_len,
            self.k_head_dim,
            self.v_head_dim,
            self.device,
        )

    @pytest.mark.ascend_a2
    @pytest.mark.L0
    def test_accuracy_vs_reference(self):
        """
        L0 accuracy test: CANN operator vs PyTorch reference implementation (single token decode).
        """
        d = self.data

        # Invoke CANN operator
        out_cann, state_cann = lite_ops.recurrent_gated_delta_rule(
            d["query_bf16"],
            d["key_bf16"],
            d["value_bf16"],
            d["beta_bf16"],
            d["state_bf16"],
            d["actual_seq_lengths"],
            d["ssm_state_indices"],
            d["g_f32"],
            d["gk_f32"],
            d["num_accepted_tokens"],
            scale_value=d["scale"],
        )

        # Run PyTorch reference implementation (float32, BTHD layout)
        out_ref, state_ref = _pytorch_recurrent_gated_delta_rule(
            d["query_f32_bthd"],
            d["key_f32_bthd"],
            d["value_f32_bthd"],
            d["g_f32_bth"],
            d["gk_f32_bthd"],
            d["beta_f32_bth"],
            d["state_f32"],
            scale=d["scale"],
        )

        # Convert reference output to BNSD layout and bfloat16
        out_ref_bnsd = out_ref.permute(0, 2, 1, 3).contiguous().bfloat16()
        state_ref_bf16 = state_ref.bfloat16()

        # Print accuracy comparison
        _print_accuracy_comparison(
            "L0 single token decode",
            out_cann,
            out_ref_bnsd,
            state_cann,
            state_ref_bf16,
        )

        max_diff_out = (out_cann.float() - out_ref_bnsd.float()).abs().max().item()
        max_diff_state = (
            (state_cann.float() - state_ref_bf16.float()).abs().max().item()
        )
        assert (
            max_diff_out < 0.05
        ), f"Output max diff {max_diff_out:.6f} exceeds threshold 0.05"
        assert (
            max_diff_state < 0.05
        ), f"State max diff {max_diff_state:.6f} exceeds threshold 0.05"


    @pytest.mark.ascend_a2
    @pytest.mark.L0
    def test_performance(self):
        """
        L0 performance test: CANN operator vs PyTorch reference implementation.
        Uses Qwen3.5-2B decode config, 100 iterations averaged.
        """
        d = self.data
        num_warmup = 10
        num_iters = 100

        # Warmup CANN
        for _ in range(num_warmup):
            lite_ops.recurrent_gated_delta_rule(
                d["query_bf16"],
                d["key_bf16"],
                d["value_bf16"],
                d["beta_bf16"],
                d["state_bf16"],
                d["actual_seq_lengths"],
                d["ssm_state_indices"],
                d["g_f32"],
                d["gk_f32"],
                d["num_accepted_tokens"],
                scale_value=d["scale"],
            )
        torch.npu.synchronize()

        # Benchmark CANN
        start = time.time()
        for _ in range(num_iters):
            lite_ops.recurrent_gated_delta_rule(
                d["query_bf16"],
                d["key_bf16"],
                d["value_bf16"],
                d["beta_bf16"],
                d["state_bf16"],
                d["actual_seq_lengths"],
                d["ssm_state_indices"],
                d["g_f32"],
                d["gk_f32"],
                d["num_accepted_tokens"],
                scale_value=d["scale"],
            )
        torch.npu.synchronize()
        cann_time = (time.time() - start) / num_iters * 1000  # ms

        # Benchmark PyTorch reference
        for _ in range(num_warmup):
            _ = _pytorch_recurrent_gated_delta_rule(
                d["query_f32_bthd"],
                d["key_f32_bthd"],
                d["value_f32_bthd"],
                d["g_f32_bth"],
                d["gk_f32_bthd"],
                d["beta_f32_bth"],
                d["state_f32"],
                scale=d["scale"],
            )
        torch.npu.synchronize()

        start = time.time()
        for _ in range(num_iters):
            _ = _pytorch_recurrent_gated_delta_rule(
                d["query_f32_bthd"],
                d["key_f32_bthd"],
                d["value_f32_bthd"],
                d["g_f32_bth"],
                d["gk_f32_bthd"],
                d["beta_f32_bth"],
                d["state_f32"],
                scale=d["scale"],
            )
        torch.npu.synchronize()
        ref_time = (time.time() - start) / num_iters * 1000  # ms

        # Print performance comparison
        _print_performance_comparison("L0 Qwen3.5-2B decode", cann_time, ref_time)


class TestRecurrentGatedDeltaRule300IDuo:
    """Validate the 300I Duo custom kernel with Qwen3.5-4B GQA dimensions."""

    @pytest.mark.ascend_300iduo
    @pytest.mark.L0
    @pytest.mark.parametrize("batch_size,seq_len", [(1, 1), (2, 1)])
    def test_qwen35_4b_accuracy(self, batch_size, seq_len):
        """Compare the custom kernel with the FP32 reference at decode shapes."""
        device = torch.device("npu:0")
        torch.npu.set_device(device)
        torch.manual_seed(2026 + batch_size * 10 + seq_len)
        data = _generate_300iduo_test_data(batch_size, seq_len, device)

        out_npu, state_npu = lite_ops.recurrent_gated_delta_rule(
            data["query"],
            data["key"],
            data["value"],
            data["beta"],
            data["state"],
            data["actual_seq_lengths"],
            data["ssm_state_indices"],
            data["g"],
            data["gk"],
            data["num_accepted_tokens"],
            scale_value=data["scale"],
        )
        out_ref, state_ref = _pytorch_recurrent_gated_delta_rule(
            data["query_ref"],
            data["key_ref"],
            data["value_ref"],
            data["g_ref"],
            data["gk_ref"],
            data["beta_ref"],
            data["state_ref"],
            scale=data["scale"],
        )
        out_npu = out_npu.cpu()
        state_npu = state_npu.cpu()
        out_ref = out_ref.permute(0, 2, 1, 3).contiguous().half()
        state_ref = state_ref.half()

        torch.testing.assert_close(out_npu, out_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(state_npu, state_ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test = TestRecurrentGatedDeltaRule()
    test.setup_method()
    test.test_accuracy_vs_reference()
    test.test_performance()
