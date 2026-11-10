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
LiteBoost verification tests for FusedQkL2NormChunkGatedDeltaRule (ascend_a2).

The fused op performs the host-side L2 normalization of query/key inside the
ChunkGatedDeltaRule kernel, and can additionally zero the padding rows of its
output.  It must agree with `F.normalize` + ChunkGatedDeltaRule on the same
inputs, and with `out_zero_pad_rows=True` the rows past each sequence's actual
length must be exactly zero.
"""

import pytest
import torch
import torch.nn.functional as F

import lite_boost.ops  # noqa: F401  # pylint: disable=unused-import  (registers torch.ops)

# bf16 carries 8 mantissa bits; two mathematically equal paths differ only by
# the rounding of the normalization (host `F.normalize` vs in-kernel rsqrt).
BF16_RTOL = 2e-3


def _tnd_inputs(t, num_k_heads, num_v_heads, dk, dv, batch, device):
    """Random TND inputs; every batch row is full length."""
    query = torch.randn(t, num_k_heads, dk, device=device, dtype=torch.bfloat16)
    key = torch.randn(t, num_k_heads, dk, device=device, dtype=torch.bfloat16)
    value = torch.randn(t, num_v_heads, dv, device=device, dtype=torch.bfloat16)
    beta = torch.rand(t, num_v_heads, device=device, dtype=torch.bfloat16)
    initial_state = torch.randn(batch, num_v_heads, dv, dk, device=device, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.rand(t, num_v_heads, device=device, dtype=torch.float32))
    seqlens = torch.tensor([t // batch] * batch, dtype=torch.int32, device=device)
    return query, key, value, beta, initial_state, seqlens, g


@pytest.mark.L0
@pytest.mark.ascend_a2
class TestFusedQkL2NormChunkGatedDeltaRule:
    """Fused normalization must not change the chunk-level GDR result."""

    @pytest.mark.parametrize("t,num_k,num_v,dk,dv", [
        pytest.param(128, 16, 32, 128, 128, id="gqa_16x32"),
        pytest.param(256, 16, 16, 128, 128, id="mha"),
        pytest.param(64, 4, 8, 128, 64, id="small_gqa"),
    ])
    def test_matches_host_normalization(self, t, num_k, num_v, dk, dv):
        """Fusing the normalization must reproduce normalize-then-op results."""
        device = "npu:0"
        torch.manual_seed(0)
        query, key, value, beta, state, seqlens, g = _tnd_inputs(
            t, num_k, num_v, dk, dv, 1, device)
        scale = dk ** -0.5

        ref_out, ref_state = torch.ops.lite_boost.chunk_gated_delta_rule(
            F.normalize(query, p=2, dim=-1), F.normalize(key, p=2, dim=-1),
            value, beta, state, seqlens, g, scale)
        out, final_state = torch.ops.lite_boost.fused_qk_l2norm_chunk_gated_delta_rule(
            query, key, value, beta, state, seqlens, g, scale, False)

        assert out.shape == ref_out.shape
        assert final_state.shape == ref_state.shape
        assert torch.allclose(out.float(), ref_out.float(), rtol=BF16_RTOL, atol=0)
        assert torch.allclose(final_state.float(), ref_state.float(), rtol=BF16_RTOL, atol=0)

    def test_zero_pad_rows(self):
        """Rows at or past each sequence's length are zeroed in-kernel."""
        device = "npu:0"
        torch.manual_seed(0)
        total, lengths = 256, [160, 64]
        query, key, value, beta, state, _, g = _tnd_inputs(
            total, 16, 32, 128, 128, len(lengths), device)
        seqlens = torch.tensor(lengths, dtype=torch.int32, device=device)

        out, _ = torch.ops.lite_boost.fused_qk_l2norm_chunk_gated_delta_rule(
            query, key, value, beta, state, seqlens, g, 128 ** -0.5, True)

        valid = int(sum(lengths))
        tail = out[valid:]
        assert tail.numel() > 0
        assert bool((tail == 0).all()), "padding rows were not zeroed"
        assert bool((out[:valid] != 0).any()), "valid rows were zeroed as well"

    def test_optional_g_absent(self):
        """A missing decay gate takes the same hasGamma=0 path as the plain op."""
        device = "npu:0"
        torch.manual_seed(0)
        query, key, value, beta, state, seqlens, _ = _tnd_inputs(
            128, 16, 32, 128, 128, 1, device)
        scale = 128 ** -0.5

        ref_out, ref_state = torch.ops.lite_boost.chunk_gated_delta_rule(
            F.normalize(query, p=2, dim=-1), F.normalize(key, p=2, dim=-1),
            value, beta, state, seqlens, None, scale)
        out, final_state = torch.ops.lite_boost.fused_qk_l2norm_chunk_gated_delta_rule(
            query, key, value, beta, state, seqlens, None, scale, False)

        assert torch.allclose(out.float(), ref_out.float(), rtol=BF16_RTOL, atol=0)
        assert torch.allclose(final_state.float(), ref_state.float(), rtol=BF16_RTOL, atol=0)
