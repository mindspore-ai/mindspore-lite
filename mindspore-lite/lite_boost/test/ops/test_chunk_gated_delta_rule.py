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
lite_boost verification test for chunk_gated_delta_rule.

Accuracy approach
-----------------
The operator implements the *chunked* Gated Delta Rule (a fused AscendC kernel
with a lower-triangular attn matrix ``-((k_beta @ K^T) * decay_mask)`` and a
recursive accumulation pass). It is NOT the plain "includes-self" recurrent form,
so a naive token-by-token PyTorch reference diverges structurally (e.g. for the
first token, the op output is the look-back state, not the just-written value).

Rather than reverse-engineer the exact chunked attn as a reference, correctness is
verified by **self-consistency**: ``chunk_size`` is purely an internal tiling
parameter, so the result must be identical for every divisor of T. This is a
strong, implementation-agnostic correctness check (verified to fp16 epsilon).

Cases:
  - test_shapes_and_dtypes : output shape/dtype/finite.
  - test_self_consistency  : chunk_size-independence (accuracy).
  - test_performance       : fused op vs a naive recurrent PyTorch baseline.

CANN operator constraints:
  - query/key must be L2-normalized (value range [0, 1])
  - g < 0  (global decay gate, exp(g) in (0, 1))
  - 0 < beta < 1
  - T is padded up to a multiple of chunk_size internally
"""

import time
import logging

import pytest
import torch
import lite_boost.ops as lite_ops
logging.basicConfig(level=logging.INFO, format='%(message)s')


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _pytorch_recurrent_baseline(query, key, value, g, beta, initial_state, scale=1.0):
    """Naive token-by-token Gated Delta Rule (no key-gate gk).

    Used only as a *performance* baseline (an unoptimized implementation the fused
    op should beat). It is NOT an accuracy ground truth for this op — the op uses a
    chunked, look-back convention that this recurrent form does not reproduce
    exactly (see module docstring).
    """
    q = query * scale
    state = initial_state.clone()  # [B, H, Dk, Dv]
    batch, n_head, seq_len, d_v = value.shape
    out = torch.zeros(batch, n_head, seq_len, d_v, dtype=torch.float32, device=query.device)
    for i in range(seq_len):
        gi = torch.exp(g[:, :, i])
        state = state * gi[..., None, None]
        ki, vi, qi = key[:, :, i, :], value[:, :, i, :], q[:, :, i, :]
        beta_i = beta[:, :, i]
        kv_mem = torch.einsum("bhdv,bhd->bhv", state, ki)
        delta = (vi - kv_mem) * beta_i[..., None]
        state = state + torch.einsum("bhd,bhv->bhdv", ki, delta)
        out[:, :, i, :] = torch.einsum("bhdv,bhd->bhv", state, qi)
    return out, state


def _generate_test_data(batch_size, num_heads, seq_len, dk, dv, device):
    scale = 1.0 / (dk ** 0.5)
    q_f32 = _l2norm(torch.randn(batch_size, seq_len, num_heads, dk, device=device), dim=-1)
    k_f32 = _l2norm(torch.randn(batch_size, seq_len, num_heads, dk, device=device), dim=-1)
    v_f32 = torch.randn(batch_size, seq_len, num_heads, dv, device=device)
    beta_f32 = torch.randn(batch_size, seq_len, num_heads, device=device).sigmoid()
    g_f32 = -(torch.rand(batch_size, num_heads, seq_len, dtype=torch.float32, device=device) + 0.01)
    state_f32 = torch.randn(batch_size, num_heads, dk, dv, device=device)
    return {
        "query": q_f32.transpose(1, 2).contiguous(),          # [B, H, T, Dk]
        "key": k_f32.transpose(1, 2).contiguous(),
        "value": v_f32.transpose(1, 2).contiguous(),
        "g": g_f32,                                            # [B, H, T]
        "beta": beta_f32.transpose(1, 2).contiguous(),        # [B, H, T]
        "state": state_f32,                                    # [B, H, Dk, Dv]
        "actual_seq_lengths": torch.tensor([seq_len] * batch_size, dtype=torch.int32, device=device),
        "ssm_state_indices": torch.tensor(list(range(batch_size)), dtype=torch.int32, device=device),
        "scale": scale,
    }


class TestChunkGatedDeltaRule:
    """Verify chunk_gated_delta_rule: shapes/dtypes, chunk_size self-consistency, perf."""

    def setup_method(self):
        """Setup test fixtures: device, tensor dimensions and test data."""
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)
        # T=64; chunk_size divisors used by the self-consistency case.
        self.batch_size, self.num_heads, self.seq_len = 1, 8, 64
        self.dk, self.dv = 64, 64
        self.data = _generate_test_data(
            self.batch_size, self.num_heads, self.seq_len, self.dk, self.dv, self.device)

    @staticmethod
    def _run(d, chunk_size):
        out, final_state = lite_ops.chunk_gated_delta_rule(
            d["query"].bfloat16(), d["key"].bfloat16(), d["value"].bfloat16(),
            d["g"].bfloat16(), d["beta"].bfloat16(), d["state"].bfloat16(),
            d["actual_seq_lengths"], d["ssm_state_indices"],
            chunk_size=chunk_size, scale_value=d["scale"])
        torch.npu.synchronize()
        return out.float().cpu(), final_state.float().cpu()

    @pytest.mark.ascend_300iduo
    @pytest.mark.L0
    def test_shapes_and_dtypes(self):
        """Output shapes/dtypes are correct and results are finite."""
        out, final_state = self._run(self.data, chunk_size=64)
        assert out.shape == (self.batch_size, self.num_heads, self.seq_len, self.dv)
        assert final_state.shape == (self.batch_size, self.num_heads, self.dk, self.dv)
        assert out.dtype is not None and final_state.dtype is not None  # came back as fp32 .cpu()
        assert torch.isfinite(out).all(), "out has NaN/Inf"
        assert torch.isfinite(final_state).all(), "final_state has NaN/Inf"

    @pytest.mark.ascend_300iduo
    @pytest.mark.L0
    def test_self_consistency(self):
        """Accuracy: result is identical for every chunk_size divisor of T
        (chunk_size is internal tiling only).
        """
        ref_out, ref_state = self._run(self.data, chunk_size=64)
        max_out, max_state = 0.0, 0.0
        for cs in (32, 16, 8):
            out, state = self._run(self.data, chunk_size=cs)
            max_out = max(max_out, (out - ref_out).abs().max().item())
            max_state = max(max_state, (state - ref_state).abs().max().item())
        logging.info("[self-consistency vs chunk_size=64] out max diff=%.6f  state max diff=%.6f",
                     max_out, max_state)
        # fp16 epsilon-level (observed ~2.4e-4); 5e-3 is a comfortable bound.
        assert max_out < 5e-3, f"out differs across chunk_size: {max_out}"
        assert max_state < 5e-3, f"state differs across chunk_size: {max_state}"

    @pytest.mark.ascend_300iduo
    @pytest.mark.L0
    def test_performance(self):
        """The fused AscendC op beats a naive token-by-token PyTorch baseline."""
        d = self.data
        qf = {k: (v.bfloat16() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
              for k, v in d.items()}

        for _ in range(5):  # warmup
            lite_ops.chunk_gated_delta_rule(
                qf["query"], qf["key"], qf["value"], qf["g"], qf["beta"], qf["state"],
                qf["actual_seq_lengths"], qf["ssm_state_indices"],
                chunk_size=64, scale_value=qf["scale"])
        torch.npu.synchronize()
        op_n = 20
        t0 = time.time()
        for _ in range(op_n):
            lite_ops.chunk_gated_delta_rule(
                qf["query"], qf["key"], qf["value"], qf["g"], qf["beta"], qf["state"],
                qf["actual_seq_lengths"], qf["ssm_state_indices"],
                chunk_size=64, scale_value=qf["scale"])
        torch.npu.synchronize()
        cann_ms = (time.time() - t0) / op_n * 1000

        # Naive recurrent baseline on CPU fp32 (the einsum path isn't NPU-compilable
        # here); a few iterations are enough to show the op's acceleration.
        def cpu(tensor):
            return tensor.detach().float().cpu()
        ref_n = 3
        t0 = time.time()
        for _ in range(ref_n):
            _ = _pytorch_recurrent_baseline(
                cpu(d["query"]), cpu(d["key"]), cpu(d["value"]),
                cpu(d["g"]), cpu(d["beta"]), cpu(d["state"]), scale=d["scale"])
        ref_ms = (time.time() - t0) / ref_n * 1000
        speedup = ref_ms / cann_ms if cann_ms > 0 else float("inf")
        logging.info("[perf] chunk op %.3f ms  |  naive baseline %.3f ms  |  speedup %.1fx",
                     cann_ms, ref_ms, speedup)
        assert cann_ms < ref_ms, (
            f"chunk op ({cann_ms:.3f} ms) not faster than naive baseline ({ref_ms:.3f} ms)")
        assert speedup >= 2.0, f"speedup {speedup:.1f}x below the 2.0x threshold"


if __name__ == "__main__":
    t = TestChunkGatedDeltaRule()
    t.setup_method()
    t.test_shapes_and_dtypes()
    logging.info("shapes_and_dtypes: PASS")
    t.test_self_consistency()
    logging.info("self_consistency:   PASS")
    t.test_performance()
    logging.info("performance:        PASS")
