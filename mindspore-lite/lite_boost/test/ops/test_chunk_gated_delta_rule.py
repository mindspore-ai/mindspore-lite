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
LiteBoost verification tests for the common ChunkGatedDeltaRule interface.

Both ascend_a2 and ascend_300iduo use the same public signature:
query, key, value, beta, initial_state, actual_seq_lengths, optional g, and
scale_value. Ascend 300I Duo exercises float16; A2 additionally exercises
bfloat16. Accuracy is checked against an independent token-by-token CPU
reference, while performance is guarded by latency baselines measured on 300I Duo.
"""

import logging
import time

import pytest
import torch

import lite_boost.ops as lite_ops

logging.basicConfig(level=logging.INFO, format="%(message)s")

LOW_DTYPES = (torch.float16, torch.bfloat16)
PERF_REGRESSION_TOLERANCE = 0.50
# Baselines are median LiteBoost end-to-end latencies measured on Ascend 300I Duo
# with CANN 8.5. Each measurement uses 5 warmup calls followed by 20 synchronized
# iterations, and the median of 3 measurements is recorded. The 50% tolerance
# absorbs shared-machine and runtime variation while still detecting regressions.
PERFORMANCE_CASES_300IDUO = (
    pytest.param(1, 8, 64, 64, 64, 1.242, id="small"),
    pytest.param(1, 16, 128, 128, 128, 7.241, id="representative"),
    pytest.param(1, 1, 128, 128, 256, 3.616, id="low_head_dv256"),
    pytest.param(2, 8, 512, 128, 128, 25.145, id="long_t512_batch2"),
    pytest.param(1, 16, 1024, 128, 128, 49.211, id="long_t1024"),
    pytest.param(1, 16, 2048, 128, 128, 97.188, id="long_t2048"),
)
_DEFAULT_G = object()


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _is_300iduo():
    return "310" + "P" in torch.npu.get_device_name(0).upper()


def _pytorch_recurrent_baseline(query, key, value, g, beta, initial_state, scale=1.0):
    """Naive token-by-token FP32 CPU reference for Gated Delta Rule."""
    query = query * scale
    state = initial_state.clone()
    batch_size, num_heads, seq_len, dv = value.shape
    out = torch.zeros(
        batch_size,
        num_heads,
        seq_len,
        dv,
        dtype=torch.float32,
        device=query.device,
    )
    for token_idx in range(seq_len):
        if g is not None:
            decay = torch.exp(g[:, :, token_idx])
            state = state * decay[..., None, None]
        key_token = key[:, :, token_idx, :]
        value_token = value[:, :, token_idx, :]
        query_token = query[:, :, token_idx, :]
        beta_token = beta[:, :, token_idx]
        value_memory = torch.einsum("bhkv,bhk->bhv", state, key_token)
        delta = (value_token - value_memory) * beta_token[..., None]
        state = state + torch.einsum("bhk,bhv->bhkv", key_token, delta)
        out[:, :, token_idx, :] = torch.einsum(
            "bhkv,bhk->bhv", state, query_token
        )
    return out, state


def _accuracy_metrics(actual, expected):
    """Return max absolute error, cosine similarity, and normalized RMSE."""
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    max_diff = (actual - expected).abs().max().item()
    cosine = (
        torch.sum(actual * expected)
        / (
            torch.sqrt(torch.sum(actual * actual))
            * torch.sqrt(torch.sum(expected * expected))
        ).clamp_min(1e-12)
    ).item()
    nrmse = (
        torch.sqrt(torch.mean((actual - expected) ** 2))
        / torch.sqrt(torch.mean(expected ** 2)).clamp_min(1e-12)
    ).item()
    return max_diff, cosine, nrmse


def _generate_test_data(batch_size, num_heads, seq_len, dk, dv, device):
    """Build BNSD inputs; the LiteBoost wrapper converts them to TND."""
    query = _l2norm(torch.randn(batch_size, seq_len, num_heads, dk, device=device), dim=-1)
    key = _l2norm(torch.randn(batch_size, seq_len, num_heads, dk, device=device), dim=-1)
    value = torch.randn(batch_size, seq_len, num_heads, dv, device=device)
    beta = torch.randn(batch_size, seq_len, num_heads, device=device).sigmoid()
    g = -(torch.rand(batch_size, num_heads, seq_len, dtype=torch.float32, device=device) + 0.01)
    state = torch.randn(batch_size, num_heads, dk, dv, device=device)
    return {
        "query": query.transpose(1, 2).contiguous(),
        "key": key.transpose(1, 2).contiguous(),
        "value": value.transpose(1, 2).contiguous(),
        "beta": beta.transpose(1, 2).contiguous(),
        "state": state,
        "actual_seq_lengths": torch.tensor(
            [seq_len] * batch_size, dtype=torch.int32, device=device
        ),
        "g": g,
        "scale": 1.0 / (dk ** 0.5),
    }


def _generate_gqa_test_data(
    batch_size, num_qk_heads, num_value_heads, seq_len, dk, dv, device
):
    """Build deterministic GQA inputs with a zero initial state."""
    generator = torch.Generator().manual_seed(42)
    query = _l2norm(
        torch.randn(batch_size, num_qk_heads, seq_len, dk, generator=generator), dim=-1
    )
    key = _l2norm(
        torch.randn(batch_size, num_qk_heads, seq_len, dk, generator=generator), dim=-1
    )
    value = torch.randn(
        batch_size, num_value_heads, seq_len, dv, generator=generator
    )
    beta = torch.randn(
        batch_size, num_value_heads, seq_len, generator=generator
    ).sigmoid()
    g = -(
        torch.rand(
            batch_size,
            num_value_heads,
            seq_len,
            dtype=torch.float32,
            generator=generator,
        )
        + 0.01
    )
    return {
        "query": query.to(device),
        "key": key.to(device),
        "value": value.to(device),
        "beta": beta.to(device),
        "state": torch.zeros(batch_size, num_value_heads, dk, dv, device=device),
        "actual_seq_lengths": torch.tensor(
            [seq_len] * batch_size, dtype=torch.int32, device=device
        ),
        "g": g.to(device),
        "scale": 1.0 / (dk ** 0.5),
    }


def _run_op(data, dtype, g=_DEFAULT_G, beta=None):
    g_input = data["g"] if g is _DEFAULT_G else g
    beta_input = data["beta"] if beta is None else beta
    return lite_ops.chunk_gated_delta_rule(
        data["query"].to(dtype),
        data["key"].to(dtype),
        data["value"].to(dtype),
        beta_input.to(dtype),
        data["state"].to(dtype),
        data["actual_seq_lengths"],
        g=g_input,
        scale_value=data["scale"],
    )


def _measure_latency_ms(data, dtype, warmup=5, repeats=20):
    for _ in range(warmup):
        _run_op(data, dtype)
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        _run_op(data, dtype)
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


@pytest.mark.ascend_a2
@pytest.mark.ascend_300iduo
class TestChunkGatedDeltaRule:
    """Verify the shared interface, optional input, accuracy, and latency."""

    def setup_method(self):
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)
        self.batch_size, self.num_heads, self.seq_len = 1, 8, 64
        self.dk, self.dv = 64, 64
        self.data = _generate_test_data(
            self.batch_size, self.num_heads, self.seq_len, self.dk, self.dv, self.device
        )

    @staticmethod
    def _skip_unsupported_dtype(dtype):
        if dtype == torch.bfloat16 and _is_300iduo():
            pytest.skip("Ascend 300I Duo ChunkGatedDeltaRule supports float16 only")

    @pytest.mark.L0
    @pytest.mark.parametrize("dtype", LOW_DTYPES)
    def test_shapes_and_dtypes(self, dtype):
        """Outputs preserve the public BNSD shapes and low dtype."""
        self._skip_unsupported_dtype(dtype)
        out, final_state = _run_op(self.data, dtype)
        torch.npu.synchronize()
        assert out.shape == (self.batch_size, self.num_heads, self.seq_len, self.dv)
        assert final_state.shape == (self.batch_size, self.num_heads, self.dk, self.dv)
        assert out.dtype == dtype
        assert final_state.dtype == dtype
        assert torch.isfinite(out).all(), f"out has NaN/Inf ({dtype})"
        assert torch.isfinite(final_state).all(), f"final_state has NaN/Inf ({dtype})"

    @pytest.mark.L0
    @pytest.mark.parametrize("dtype", LOW_DTYPES)
    def test_optional_g(self, dtype):
        """The optional g input can be omitted on both backends."""
        self._skip_unsupported_dtype(dtype)
        out, final_state = _run_op(self.data, dtype, g=None)
        torch.npu.synchronize()
        assert out.shape == (self.batch_size, self.num_heads, self.seq_len, self.dv)
        assert final_state.shape == (self.batch_size, self.num_heads, self.dk, self.dv)
        assert torch.isfinite(out).all(), f"out has NaN/Inf with g=None ({dtype})"
        assert torch.isfinite(final_state).all(), f"final_state has NaN/Inf with g=None ({dtype})"

    @pytest.mark.L0
    @pytest.mark.parametrize("dtype", LOW_DTYPES)
    def test_accuracy_beta_zero(self, dtype):
        """With beta=0 and g=None, state is frozen and out=q@state."""
        self._skip_unsupported_dtype(dtype)
        beta_zero = torch.zeros_like(self.data["beta"])
        out, final_state = _run_op(self.data, dtype, g=None, beta=beta_zero)
        torch.npu.synchronize()

        query = self.data["query"].to(dtype).float().cpu()
        initial_state = self.data["state"].to(dtype).float().cpu()
        out_ref = self.data["scale"] * torch.einsum("bhkv,bhtk->bhtv", initial_state, query)
        state_ref = initial_state
        out_diff = (out.float().cpu() - out_ref).abs().max().item()
        state_diff = (final_state.float().cpu() - state_ref).abs().max().item()
        logging.info(
            "[beta=0 / g=None / %s] out max diff %.6f, state max diff %.6f",
            dtype,
            out_diff,
            state_diff,
        )
        assert out_diff < 5e-2, f"output diverges from reference ({dtype}): {out_diff}"
        assert state_diff < 5e-3, f"final_state diverges from input ({dtype}): {state_diff}"

    @pytest.mark.L0
    @pytest.mark.parametrize("dtype", LOW_DTYPES)
    def test_accuracy_recurrent_reference(self, dtype):
        """Normal beta/g path matches an independent FP32 CPU recurrence."""
        self._skip_unsupported_dtype(dtype)
        torch.manual_seed(42)
        data = _generate_test_data(2, 2, 64, 32, 32, self.device)
        out, final_state = _run_op(data, dtype)
        torch.npu.synchronize()

        def cpu_low(tensor):
            return tensor.to(dtype).float().cpu()

        out_ref, state_ref = _pytorch_recurrent_baseline(
            cpu_low(data["query"]),
            cpu_low(data["key"]),
            cpu_low(data["value"]),
            data["g"].float().cpu(),
            cpu_low(data["beta"]),
            cpu_low(data["state"]),
            scale=data["scale"],
        )
        out_metrics = _accuracy_metrics(out.float().cpu(), out_ref)
        state_metrics = _accuracy_metrics(final_state.float().cpu(), state_ref)
        logging.info(
            "[recurrent ref / %s] out(max=%.6f cos=%.9f nrmse=%.6f), "
            "state(max=%.6f cos=%.9f nrmse=%.6f)",
            dtype,
            *out_metrics,
            *state_metrics,
        )
        assert out_metrics[1] >= 0.999 and out_metrics[2] <= 0.05, (
            f"output diverges from CPU recurrence ({dtype}): "
            f"cosine={out_metrics[1]}, nrmse={out_metrics[2]}"
        )
        assert state_metrics[1] >= 0.999 and state_metrics[2] <= 0.02, (
            f"final_state diverges from CPU recurrence ({dtype}): "
            f"cosine={state_metrics[1]}, nrmse={state_metrics[2]}"
        )

    @pytest.mark.L0
    def test_gqa_beta_cast_regression(self):
        """A representative GQA shape stays finite and matches the CPU recurrence."""
        num_qk_heads, num_value_heads = 16, 32
        data = _generate_gqa_test_data(
            1, num_qk_heads, num_value_heads, 64, 128, 128, self.device
        )
        out, final_state = _run_op(data, torch.float16)
        torch.npu.synchronize()
        assert torch.isfinite(out).all(), "GQA output has NaN/Inf"
        assert torch.isfinite(final_state).all(), "GQA final_state has NaN/Inf"

        repeat_factor = num_value_heads // num_qk_heads

        def cpu_low(tensor):
            return tensor.to(torch.float16).float().cpu()

        out_ref, state_ref = _pytorch_recurrent_baseline(
            cpu_low(data["query"]).repeat_interleave(repeat_factor, dim=1),
            cpu_low(data["key"]).repeat_interleave(repeat_factor, dim=1),
            cpu_low(data["value"]),
            data["g"].float().cpu(),
            cpu_low(data["beta"]),
            cpu_low(data["state"]),
            scale=data["scale"],
        )
        out_metrics = _accuracy_metrics(out.float().cpu(), out_ref)
        state_metrics = _accuracy_metrics(final_state.float().cpu(), state_ref)
        logging.info(
            "[GQA] out(max=%.6f cos=%.9f nrmse=%.6f), "
            "state(max=%.6f cos=%.9f nrmse=%.6f)",
            *out_metrics,
            *state_metrics,
        )
        assert out_metrics[1] >= 0.999 and out_metrics[2] <= 0.02, (
            "GQA output diverges from CPU recurrence: "
            f"cosine={out_metrics[1]}, nrmse={out_metrics[2]}"
        )
        assert state_metrics[1] >= 0.999 and state_metrics[2] <= 0.02, (
            "GQA final_state diverges from CPU recurrence: "
            f"cosine={state_metrics[1]}, nrmse={state_metrics[2]}"
        )

    @pytest.mark.L0
    def test_gqa_multichunk_l0c_sync(self):
        """Repeated multi-chunk GQA calls must not race Cube writes and Vector reads."""
        if not _is_300iduo():
            pytest.skip("300I Duo-specific L0C synchronization regression")
        data = _generate_gqa_test_data(1, 16, 32, 512, 128, 128, self.device)
        results = []
        for _ in range(32):
            out, final_state = _run_op(data, torch.float16)
            results.append((out, final_state))
        torch.npu.synchronize()

        reference_out = results[0][0].float().cpu()
        reference_state = results[0][1].float().cpu()
        for repeat, (out, final_state) in enumerate(results):
            assert torch.isfinite(out).all(), f"GQA output has NaN/Inf at repeat {repeat}"
            assert torch.isfinite(final_state).all(), f"GQA state has NaN/Inf at repeat {repeat}"
            out_metrics = _accuracy_metrics(out.float().cpu(), reference_out)
            state_metrics = _accuracy_metrics(final_state.float().cpu(), reference_state)
            assert out_metrics[1] >= 0.999 and out_metrics[2] <= 0.02, (
                f"GQA output is unstable at repeat {repeat}: "
                f"cosine={out_metrics[1]}, nrmse={out_metrics[2]}"
            )
            assert state_metrics[1] >= 0.999 and state_metrics[2] <= 0.02, (
                f"GQA state is unstable at repeat {repeat}: "
                f"cosine={state_metrics[1]}, nrmse={state_metrics[2]}"
            )

    @pytest.mark.L0
    @pytest.mark.parametrize("dtype", LOW_DTYPES)
    @pytest.mark.parametrize("dk", (63, 65, 79, 80, 81, 95, 96, 97, 127))
    def test_non_multiple_reduce_width(self, dtype, dk):
        """Bucketed Cube tails match both zero padding and the CPU recurrence."""
        self._skip_unsupported_dtype(dtype)
        if dk <= 64:
            padded_dk = 64
        elif dk <= 80:
            padded_dk = 80
        elif dk <= 96:
            padded_dk = 96
        else:
            padded_dk = 128
        data = _generate_test_data(1, 2, 64, dk, 32, self.device)
        padded_data = dict(data)
        padded_data["query"] = torch.nn.functional.pad(data["query"], (0, padded_dk - dk))
        padded_data["key"] = torch.nn.functional.pad(data["key"], (0, padded_dk - dk))
        padded_data["state"] = torch.nn.functional.pad(data["state"], (0, 0, 0, padded_dk - dk))

        out, final_state = _run_op(data, dtype)
        padded_out, padded_final_state = _run_op(padded_data, dtype)
        torch.npu.synchronize()

        def cpu_low(tensor):
            return tensor.to(dtype).float().cpu()

        out_ref, state_ref = _pytorch_recurrent_baseline(
            cpu_low(data["query"]),
            cpu_low(data["key"]),
            cpu_low(data["value"]),
            data["g"].float().cpu(),
            cpu_low(data["beta"]),
            cpu_low(data["state"]),
            scale=data["scale"],
        )

        out_diff = (out.float() - padded_out.float()).abs().max().item()
        state_diff = (
            final_state.float() - padded_final_state[:, :, :dk, :].float()
        ).abs().max().item()
        out_metrics = _accuracy_metrics(out.float().cpu(), out_ref)
        state_metrics = _accuracy_metrics(final_state.float().cpu(), state_ref)
        assert out_diff < 5e-2, f"tail reduction output mismatch ({dtype}): {out_diff}"
        assert state_diff < 5e-2, f"tail reduction state mismatch ({dtype}): {state_diff}"
        assert out_metrics[1] >= 0.999 and out_metrics[2] <= 0.02, (
            f"tail output diverges from CPU recurrence ({dtype}, dk={dk}): "
            f"cosine={out_metrics[1]}, nrmse={out_metrics[2]}"
        )
        assert state_metrics[1] >= 0.999 and state_metrics[2] <= 0.02, (
            f"tail state diverges from CPU recurrence ({dtype}, dk={dk}): "
            f"cosine={state_metrics[1]}, nrmse={state_metrics[2]}"
        )

    @pytest.mark.L0
    @pytest.mark.parametrize(
        "batch_size,num_heads,seq_len,dk,dv,baseline_ms",
        PERFORMANCE_CASES_300IDUO,
    )
    def test_performance_regression_300iduo(
        self, batch_size, num_heads, seq_len, dk, dv, baseline_ms
    ):
        """300I Duo latency may vary by at most 50% above its verified baseline."""
        if not _is_300iduo():
            pytest.skip("300I Duo-specific performance baseline")
        data = _generate_test_data(
            batch_size, num_heads, seq_len, dk, dv, self.device
        )
        latency_ms = _measure_latency_ms(data, torch.float16)
        limit_ms = baseline_ms * (1.0 + PERF_REGRESSION_TOLERANCE)
        logging.info(
            "[perf %dx%dx%dx%dx%d] %.3f ms, limit %.3f ms",
            batch_size,
            num_heads,
            seq_len,
            dk,
            dv,
            latency_ms,
            limit_ms,
        )
        assert latency_ms <= limit_ms, (
            f"performance regression: {latency_ms:.3f} ms exceeds "
            f"{limit_ms:.3f} ms (baseline {baseline_ms:.3f} ms + 50%)"
        )
