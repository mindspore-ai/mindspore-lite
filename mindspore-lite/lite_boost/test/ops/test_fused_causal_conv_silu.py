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
LiteBoost verification tests for FusedCausalConvSilu (ascend_a2).

The fused op computes a depthwise causal conv1d with a 4-tap kernel and SiLU in
one kernel for the GDN prefill path.  ``y`` must match a left-padded conv1d +
SiLU reference and ``conv_state`` must hold the trailing 4 taps of the real
sequence so the decode step can be seeded from it.  The kernel accepts
``seq_len`` as an int32 vector whose first entry is the number of valid rows;
rows past it are still convolved (they are padding and get sliced away by the
model), but ``conv_state`` is read from the valid prefix only.
"""

import pytest
import torch
import torch.nn.functional as F

import lite_boost.ops  # noqa: F401  # pylint: disable=unused-import  (registers torch.ops)

KERNEL_SIZE = 4
SEQ_LEN_ELEMS = 8

# The kernel accumulates in fp32 but rounds the convolution to bf16 before the
# SiLU, so the reference follows the same order; only the sigmoid evaluation
# can differ by one bf16 ulp (2**-8).
BF16_RTOL = 4e-3
BF16_ATOL = 1e-3


def _reference(x, weight, bias, real_len):
    """Left-padded conv1d + SiLU plus the trailing-tap conv state."""
    channels = weight.shape[1]
    x_used = x[:, :, :channels].float()
    padded = F.pad(x_used, (0, 0, KERNEL_SIZE - 1, 0))
    conv = torch.zeros_like(x_used)
    for tap in range(KERNEL_SIZE):
        conv = conv + padded[:, tap:tap + x.shape[1], :] * weight[tap].float()
    conv = (conv + bias.float()).to(torch.bfloat16).float()
    y = (conv * torch.sigmoid(conv)).to(torch.bfloat16)

    rows = [max(real_len - KERNEL_SIZE + tap, 0) for tap in range(KERNEL_SIZE)]
    state = x[:, rows, :channels].contiguous()
    return y, state


def _inputs(seq, channels, real_len, device, extra_channels=0):
    """Random bf16 inputs; ``x`` may carry unused trailing channel columns."""
    torch.manual_seed(0)
    x = torch.randn(1, seq, channels + extra_channels, device=device, dtype=torch.bfloat16)
    weight = torch.randn(KERNEL_SIZE, channels, device=device, dtype=torch.bfloat16)
    bias = torch.randn(channels, device=device, dtype=torch.bfloat16)
    seq_len = torch.full((SEQ_LEN_ELEMS,), real_len, dtype=torch.int32, device=device)
    return x, weight, bias, seq_len


def _run(x, weight, bias, seq_len):
    return torch.ops.lite_boost.fused_causal_conv_silu(x, weight, bias, seq_len)


@pytest.mark.L0
@pytest.mark.ascend_a2
class TestFusedCausalConvSilu:
    """Fused conv1d+SiLU must match the framework reference and keep the taps."""

    @pytest.mark.parametrize("seq,channels", [
        pytest.param(1, 512, id="single_token"),
        pytest.param(16, 512, id="one_chunk"),
        pytest.param(17, 512, id="chunk_halo"),
        pytest.param(40, 1024, id="two_channel_tiles"),
    ])
    def test_matches_reference(self, seq, channels):
        """Full-length sequences reproduce conv1d+SiLU and the trailing taps."""
        device = "npu:0"
        x, weight, bias, seq_len = _inputs(seq, channels, seq, device)
        y, state = _run(x, weight, bias, seq_len)
        ref_y, ref_state = _reference(x, weight, bias, seq)

        assert y.shape == (1, seq, channels)
        assert state.shape == (1, KERNEL_SIZE, channels)
        torch.testing.assert_close(
            y.float(), ref_y.float(), rtol=BF16_RTOL, atol=BF16_ATOL)
        assert torch.equal(state, ref_state), "conv_state is not the trailing 4 taps"

    @pytest.mark.parametrize("seq,real_len", [
        pytest.param(16, 5, id="shorter_than_seq"),
        pytest.param(40, 2, id="shorter_than_kernel"),
        pytest.param(40, 1, id="single_valid_row"),
    ])
    def test_conv_state_uses_real_len(self, seq, real_len):
        """``conv_state`` is read from ``real_len``, clamped at the first row."""
        device = "npu:0"
        channels = 512
        x, weight, bias, seq_len = _inputs(seq, channels, real_len, device)
        y, state = _run(x, weight, bias, seq_len)
        ref_y, ref_state = _reference(x, weight, bias, real_len)

        assert torch.equal(state, ref_state), (
            f"conv_state for real_len={real_len} does not match the reference taps")
        # Valid rows are unaffected by the length; padding rows are sliced away.
        torch.testing.assert_close(
            y[:, :real_len].float(), ref_y[:, :real_len].float(),
            rtol=BF16_RTOL, atol=BF16_ATOL)

    def test_extra_input_channels_ignored(self):
        """Trailing columns of ``x`` beyond ``channels`` are not convolved."""
        device = "npu:0"
        seq, channels, extra = 18, 512, 16
        x, weight, bias, seq_len = _inputs(seq, channels, seq, device, extra)
        y, state = _run(x, weight, bias, seq_len)
        ref_y, ref_state = _reference(x, weight, bias, seq)

        assert y.shape == (1, seq, channels)
        torch.testing.assert_close(
            y.float(), ref_y.float(), rtol=BF16_RTOL, atol=BF16_ATOL)
        assert torch.equal(state, ref_state)
