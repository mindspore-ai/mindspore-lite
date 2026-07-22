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
lite_boost verification test for chunk_gated_delta_rule (ascend_a2 / ascend910b op).

The wrapped AscendC op implements the *chunked* Gated Delta Rule (prefill). Its interface
matches the ascend_a2 op prototype verbatim: TND layout, q/k/v/beta/state/out/final_state each
accept BOTH bf16 and fp16 (DataTypeList), float32 optional gate ``g``, int32
``actual_seq_lengths``, ``scale_value`` attr (no cu_seqlens / ssm_state_indices / chunk_size —
those belong to the ascend_300iduo op).

The functional cases are parametrised over dtype in {bf16, fp16} so both low-dtype paths are
exercised. The wrapper auto-detects the low dtype from the query tensor.

Accuracy approach
-----------------
A closed-form reference exists for the degenerate ``beta = 0, g = None`` case: with no
state update (beta=0) and no decay gate, the recurrent state stays at ``initial_state`` for
every step and the intra-chunk attention contribution vanishes, so the output collapses to
the pure inter-chunk term ``out_t = scale * (q_t @ S_0^T)`` for all t. This is checked against
a plain ``torch.einsum`` reference at the low dtype's tolerance.

Cases:
  - test_shapes_and_dtypes : output shape/dtype/finite (with g), bf16 + fp16.
  - test_optional_g        : the g=None (hasGamma=0) path runs and is well-formed, bf16 + fp16.
  - test_accuracy_beta_zero: closed-form reference (beta=0, g=None), bf16 + fp16.
  - test_performance       : fused op vs a naive token-by-token PyTorch baseline (bf16).

Runs on ascend910b only (``-m "ascend_a2"``).
"""

import time
import logging

import pytest
import torch
import lite_boost.ops as lite_ops
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Both low dtypes the 910B op accepts.
LOW_DTYPES = [torch.bfloat16, torch.float16]


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _pytorch_recurrent_baseline(query, key, value, g, beta, initial_state, scale=1.0):
    # pylint: disable=too-many-locals
    """Naive token-by-token Gated Delta Rule (no key-gate gk).

    Used only as a *performance* baseline (an unoptimized implementation the fused op should
    beat). It is NOT an accuracy ground truth for the chunked op — see module docstring.
    """
    q = query * scale
    state = initial_state.clone()  # [B, H, Dk, Dv]
    batch, n_head, seq_len, d_v = value.shape
    out = torch.zeros(batch, n_head, seq_len, d_v, dtype=torch.float32, device=query.device)
    for i in range(seq_len):
        gi = torch.exp(g[:, :, i]) if g is not None else 1.0
        state = state * (gi[..., None, None] if g is not None else 1.0)
        ki, vi, qi = key[:, :, i, :], value[:, :, i, :], q[:, :, i, :]
        beta_i = beta[:, :, i]
        kv_mem = torch.einsum("bhdv,bhd->bhv", state, ki)
        delta = (vi - kv_mem) * beta_i[..., None]
        state = state + torch.einsum("bhd,bhv->bhdv", ki, delta)
        out[:, :, i, :] = torch.einsum("bhdv,bhd->bhv", state, qi)
    return out, state


def _generate_test_data(batch_size, num_heads, seq_len, dk, dv, device):
    """Build BNSD test inputs (fp32 on device). ``num_heads`` is used for both Nk and Nv."""
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
        "scale": 1.0,
    }


class TestChunkGatedDeltaRule:
    """Verify chunk_gated_delta_rule against the ascend_a2 op spec (bf16 + fp16)."""

    def setup_method(self):
        """Setup test fixtures: device, tensor dimensions and test data."""
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)
        # T=64; Nk=Nv=8 (no GQA); Dk=Dv=64.
        self.batch_size, self.num_heads, self.seq_len = 1, 8, 64
        self.dk, self.dv = 64, 64
        self.data = _generate_test_data(
            self.batch_size, self.num_heads, self.seq_len, self.dk, self.dv, self.device)

    @staticmethod
    def _run(d, dtype, g=None, scale=None):
        """Run one chunk_gated_delta_rule forward and return (out, final_state)."""
        # Cast q/k/v/beta/state to the target low dtype; the wrapper auto-detects it.
        # g stays float32 (op dtype for g is FLOAT).
        g_in = (g if g is not None else d["g"])
        if g_in is not None:
            g_in = g_in.float()
        out, final_state = lite_ops.chunk_gated_delta_rule(
            d["query"].to(dtype), d["key"].to(dtype), d["value"].to(dtype),
            d["beta"].to(dtype), d["state"].to(dtype),
            d["actual_seq_lengths"], g=g_in,
            scale_value=d["scale"] if scale is None else scale)
        torch.npu.synchronize()
        return out.float().cpu(), final_state.float().cpu()

    @pytest.mark.ascend_a2
    @pytest.mark.L0
    @pytest.mark.parametrize("dtype", LOW_DTYPES)
    def test_shapes_and_dtypes(self, dtype):
        """Output shapes/dtypes are correct and results are finite (with g)."""
        out, final_state = self._run(self.data, dtype)
        assert out.shape == (self.batch_size, self.num_heads, self.seq_len, self.dv)
        assert final_state.shape == (self.batch_size, self.num_heads, self.dk, self.dv)
        assert torch.isfinite(out).all(), f"out has NaN/Inf ({dtype})"
        assert torch.isfinite(final_state).all(), f"final_state has NaN/Inf ({dtype})"

    @pytest.mark.ascend_a2
    @pytest.mark.L0
    @pytest.mark.parametrize("dtype", LOW_DTYPES)
    def test_optional_g(self, dtype):
        """The g=None (hasGamma=0) path runs and produces well-formed output."""
        out, final_state = self._run(self.data, dtype, g=None)
        assert out.shape == (self.batch_size, self.num_heads, self.seq_len, self.dv)
        assert final_state.shape == (self.batch_size, self.num_heads, self.dk, self.dv)
        assert torch.isfinite(out).all(), f"out (g=None) has NaN/Inf ({dtype})"
        assert torch.isfinite(final_state).all(), f"final_state (g=None) has NaN/Inf ({dtype})"

    @pytest.mark.ascend_a2
    @pytest.mark.L0
    @pytest.mark.parametrize("dtype", LOW_DTYPES)
    def test_accuracy_beta_zero(self, dtype):
        """Closed-form check: with beta=0 and g=None the state is frozen and
        out_t = scale * (q_t @ S_0^T) for every t (intra-chunk attn vanishes).
        """
        scale = 1.0
        d = self.data
        beta_zero = torch.zeros_like(d["beta"])
        out, _ = lite_ops.chunk_gated_delta_rule(
            d["query"].to(dtype), d["key"].to(dtype), d["value"].to(dtype),
            beta_zero.to(dtype), d["state"].to(dtype),
            d["actual_seq_lengths"], g=None, scale_value=scale)
        torch.npu.synchronize()
        out = out.float().cpu()

        # Reference from the SAME low-dtype values the op actually sees.
        q_low = d["query"].to(dtype).float().cpu()             # [B, H, T, Dk]
        state_low = d["state"].to(dtype).float().cpu()          # [B, H, Dk, Dv]
        out_ref = scale * torch.einsum("bhkv,bhtk->bhtv", state_low, q_low)  # [B, H, T, Dv]

        max_diff = (out - out_ref).abs().max().item()
        logging.info("[beta=0 / g=None / %s] out max diff vs einsum ref = %.6f", dtype, max_diff)
        # bf16/fp16, Dk=64 accumulation -> ~1e-2; 5e-2 is the project's standard tolerance.
        assert max_diff < 5e-2, f"beta=0 output diverges from reference ({dtype}): {max_diff}"

    @pytest.mark.ascend_a2
    @pytest.mark.L0
    def test_performance(self):
        """The fused AscendC op beats a naive token-by-token PyTorch baseline (bf16)."""
        dtype = torch.bfloat16
        d = self.data
        for _ in range(5):  # warmup
            lite_ops.chunk_gated_delta_rule(
                d["query"].to(dtype), d["key"].to(dtype), d["value"].to(dtype),
                d["beta"].to(dtype), d["state"].to(dtype),
                d["actual_seq_lengths"], g=d["g"].float(), scale_value=d["scale"])
        torch.npu.synchronize()
        op_n = 20
        t0 = time.time()
        for _ in range(op_n):
            lite_ops.chunk_gated_delta_rule(
                d["query"].to(dtype), d["key"].to(dtype), d["value"].to(dtype),
                d["beta"].to(dtype), d["state"].to(dtype),
                d["actual_seq_lengths"], g=d["g"].float(), scale_value=d["scale"])
        torch.npu.synchronize()
        cann_ms = (time.time() - t0) / op_n * 1000

        # Naive recurrent baseline on CPU fp32; a few iterations are enough to show the
        # op's acceleration.
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
        logging.info("[perf / %s] chunk op %.3f ms  |  naive baseline %.3f ms  |  speedup %.1fx",
                     dtype, cann_ms, ref_ms, speedup)
        assert cann_ms < ref_ms, (
            f"chunk op ({cann_ms:.3f} ms) not faster than naive baseline ({ref_ms:.3f} ms)")
        assert speedup >= 2.0, f"speedup {speedup:.1f}x below the 2.0x threshold"


if __name__ == "__main__":
    t = TestChunkGatedDeltaRule()
    t.setup_method()
    for dt in LOW_DTYPES:
        t.test_shapes_and_dtypes(dt)
        logging.info("shapes_and_dtypes [%s]: PASS", dt)
        t.test_optional_g(dt)
        logging.info("optional_g [%s]:        PASS", dt)
        t.test_accuracy_beta_zero(dt)
        logging.info("accuracy_beta_zero [%s]:PASS", dt)
    t.test_performance()
    logging.info("performance:       PASS")
