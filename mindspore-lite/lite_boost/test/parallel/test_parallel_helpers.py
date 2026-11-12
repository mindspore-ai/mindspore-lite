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
LiteBoost tests for the shared tensor-parallel / sequence-parallel helpers.

The Qwen3.5-9B adapter shards its projections with these helpers, so their
partition arithmetic is pinned here: every case states the exact slice the
caller expects, which is what keeps a refactor of the adapter numerically
inert.  The all-gather / GQA-expansion cases run without a process group and
therefore exercise the single-rank fallbacks.
"""

import pytest
import torch
from torch import nn

from lite_boost.parallel.context_parallel import all_gather_seq
from lite_boost.parallel.tensor_parallel import (
    get_overlap_kv_heads_per_rank,
    gqa_expand_kv_overlap,
    shard_colwise,
    shard_kv_overlap,
    shard_rowwise,
)


def _linear(out_features, in_features):
    linear = nn.Linear(in_features, out_features)
    with torch.no_grad():
        linear.weight.copy_(torch.randn_like(linear.weight))
        linear.bias.copy_(torch.randn_like(linear.bias))
    return linear


@pytest.mark.L0
@pytest.mark.ascend_a2
class TestLinearSharding:
    """In-place partition of nn.Linear weights."""

    @pytest.mark.parametrize("world", [1, 2, 4])
    def test_colwise_is_a_contiguous_row_slice(self, world):
        """Column-parallel sharding takes rows [rank*chunk, (rank+1)*chunk)."""
        for rank in range(world):
            linear = _linear(64, 32)
            expected_w = linear.weight.detach().clone()
            expected_b = linear.bias.detach().clone()
            chunk = 64 // world

            shard_colwise(linear, rank, world)

            assert linear.out_features == chunk
            assert torch.equal(linear.weight, expected_w[rank * chunk:(rank + 1) * chunk])
            assert torch.equal(linear.bias, expected_b[rank * chunk:(rank + 1) * chunk])

    @pytest.mark.parametrize("world", [1, 2, 4])
    def test_rowwise_is_a_contiguous_column_slice(self, world):
        """Row-parallel sharding takes columns [rank*chunk, (rank+1)*chunk) and keeps bias."""
        for rank in range(world):
            linear = _linear(64, 32)
            expected_w = linear.weight.detach().clone()
            expected_b = linear.bias.detach().clone()
            chunk = 32 // world

            shard_rowwise(linear, rank, world)

            assert linear.in_features == chunk
            assert torch.equal(linear.weight, expected_w[:, rank * chunk:(rank + 1) * chunk])
            assert torch.equal(linear.bias, expected_b)

    def test_none_is_ignored(self):
        assert shard_colwise(None, 0, 2) is None
        assert shard_rowwise(None, 0, 2) is None

    @pytest.mark.parametrize("world", [1, 2, 4])
    @pytest.mark.parametrize("num_kv_heads", [4, 5, 8])
    def test_kv_overlap_keeps_one_shared_head(self, world, num_kv_heads):
        """Neighbouring ranks overlap by one KV head so every query head finds its group."""
        head_dim = 8
        for rank in range(world):
            start, count = get_overlap_kv_heads_per_rank(rank, world, num_kv_heads)
            if rank > 0:
                heads_per_rank = (num_kv_heads + world - 1) // world
                assert start == rank * (heads_per_rank - 1)
            assert count >= 1
            assert start + count <= num_kv_heads

            linear = _linear(num_kv_heads * head_dim, 16)
            expected_w = linear.weight.detach().clone()
            shard_kv_overlap(linear, rank, world, num_kv_heads, head_dim)
            rows = slice(start * head_dim, (start + count) * head_dim)
            assert linear.out_features == count * head_dim
            assert torch.equal(linear.weight, expected_w[rows])


@pytest.mark.L0
@pytest.mark.ascend_a2
class TestSingleRankFallbacks:
    """Without an initialized process group the helpers must be no-ops."""

    def test_gather_returns_input(self):
        x = torch.randn(1, 4, 8, 16)
        assert all_gather_seq(x, dim=1) is x
        assert all_gather_seq(x, dim=2) is x

    def test_gqa_expansion_is_repeat_interleave(self):
        key = torch.randn(1, 6, 4, 8)
        value = torch.randn(1, 6, 4, 8)
        out_k, out_v = gqa_expand_kv_overlap(key, value, 16, 4, 0, 4)
        assert torch.equal(out_k, key.repeat_interleave(4, dim=-3))
        assert torch.equal(out_v, value.repeat_interleave(4, dim=-3))
