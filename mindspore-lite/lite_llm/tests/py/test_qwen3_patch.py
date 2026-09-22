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
"""Focused tests for Qwen3 attention projection shape handling."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

torch = pytest.importorskip("torch")

_EXPORT_DIR = Path(__file__).resolve().parents[2] / "export"
sys.path.insert(0, str(_EXPORT_DIR))

from models._base.nnrt_decoder_wrapper import NnrtAttention  # pylint: disable=wrong-import-position
from models.qwen3.qwen3_wrapper import Qwen3OpSet, MsTransRopeScatterNDUpdate  # pylint: disable=wrong-import-position


class _Projection:
    def __init__(self, output_width):
        self.output_width = output_width

    def __call__(self, tensor):
        return torch.zeros((*tensor.shape[:-1], self.output_width), dtype=tensor.dtype)


class _OutputProjection:
    def __init__(self, output_width):
        self.output_width = output_width
        self.input_width = None

    def __call__(self, tensor):
        self.input_width = tensor.shape[-1]
        return torch.zeros((*tensor.shape[:-1], self.output_width), dtype=tensor.dtype)


class _GroupMatmul:
    @staticmethod
    def apply(lhs, rhs, transpose_rhs):
        if transpose_rhs:
            return torch.zeros((lhs.shape[0], lhs.shape[1], lhs.shape[2], rhs.shape[2]), dtype=lhs.dtype)
        return torch.zeros((lhs.shape[0], lhs.shape[1], lhs.shape[2], rhs.shape[3]), dtype=lhs.dtype)


class _Rotary:
    @staticmethod
    def apply(query, key, cos, sin):
        del cos, sin
        return query, key


class _AddSoftmax:
    @staticmethod
    def apply(scores, mask):
        del mask
        return torch.softmax(scores, dim=-1)


def _run_attention(monkeypatch, hidden_size, num_heads, num_kv_heads, head_dim):
    """Exercise projected width independently of attention kernels."""
    ops = Qwen3OpSet(SimpleNamespace(num_attention_heads=num_heads, head_dim=head_dim))
    monkeypatch.setattr(ops, "qk_matmul", lambda q, k: _GroupMatmul.apply(q, k, True))
    monkeypatch.setattr(ops, "pv_matmul", lambda q, v: _GroupMatmul.apply(q, v, False))
    monkeypatch.setattr(ops, "mask_softmax", _AddSoftmax.apply)

    output_projection = _OutputProjection(hidden_size)
    attention = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
        ),
        head_dim=head_dim,
        q_proj=_Projection(num_heads * head_dim),
        k_proj=_Projection(num_kv_heads * head_dim),
        v_proj=_Projection(num_kv_heads * head_dim),
        q_norm=lambda tensor: tensor,
        k_norm=lambda tensor: tensor,
        o_proj=output_projection,
        attention_dropout=0.0,
        training=False,
    )
    sequence = 2
    wrapper = NnrtAttention(attention, attention.config, ops).eval()
    output, _ = wrapper(
        torch.zeros((1, sequence, hidden_size), dtype=torch.float16),
        attention_mask=torch.zeros((1, 1, sequence, sequence), dtype=torch.float16),
        rope_cos=torch.zeros((1, sequence, head_dim), dtype=torch.float16),
        rope_sin=torch.zeros((1, sequence, head_dim), dtype=torch.float16),
        past_key_value=(torch.zeros((1, num_kv_heads, sequence, head_dim), dtype=torch.float16),
                        torch.zeros((1, num_kv_heads, sequence, head_dim), dtype=torch.float16)),
        valid_seq_len=torch.tensor([0], dtype=torch.int32),
    )
    assert output.shape == (1, sequence, hidden_size)
    return output_projection.input_width


def test_attention_uses_projected_head_width(monkeypatch):
    """Qwen3-4B and MiniMind both feed the correct projected width to o_proj."""
    assert _run_attention(monkeypatch, hidden_size=2560, num_heads=32, num_kv_heads=8, head_dim=128) == 4096
    assert _run_attention(monkeypatch, hidden_size=768, num_heads=8, num_kv_heads=4, head_dim=96) == 768


@pytest.mark.parametrize("seq,offset", [(1, 7), (3, 2)])
def test_fused_adapter_preserves_cache_and_transposes(seq, offset):
    """Preserve untouched cache entries and expose BNSD outputs."""
    q = torch.arange(seq * 4 * 16, dtype=torch.float16).reshape(1, seq, 4, 16)
    k = q[:, :, :2, :].contiguous()
    v = -k
    cache = torch.ones((1, 2, 8, 16), dtype=torch.float16)
    qout, kout, vout = MsTransRopeScatterNDUpdate.apply(
        q, k, v, torch.ones((1, seq, 16), dtype=torch.float16),
        torch.zeros((1, seq, 16), dtype=torch.float16), cache, cache,
        torch.tensor([offset], dtype=torch.int32), 4, 16)
    torch.testing.assert_close(qout, q.transpose(1, 2))
    expected_k = cache.clone()
    expected_v = cache.clone()
    expected_k[:, :, offset:offset+seq] = k.transpose(1, 2)
    expected_v[:, :, offset:offset+seq] = v.transpose(1, 2)
    torch.testing.assert_close(kout, expected_k)
    torch.testing.assert_close(vout, expected_v)
    torch.testing.assert_close(cache, torch.ones_like(cache))


@pytest.mark.parametrize("sequence,offset", [(4, 0), (1, 4)])
def test_fused_wrapper_matches_unfused_reference(sequence, offset):
    """Compare full wrapper logits and all caches before checking ONNX fusion."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from models._base.nnrt_decoder_wrapper import NnrtOpSet
    from models.qwen3.qwen3_wrapper import Qwen3NnrtWrapper

    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(config).half().eval()
    fused = Qwen3NnrtWrapper(model, config).eval()
    reference = Qwen3NnrtWrapper(model, config, NnrtOpSet()).eval()
    caches = tuple(
        (torch.randn(1, 2, 8, 32).half(), torch.randn(1, 2, 8, 32).half())
        for _ in range(2)
    )
    angles = torch.randn(1, sequence, 16).repeat(1, 1, 2)
    mask = torch.zeros(1, 1, sequence, 8).half()
    for index in range(sequence):
        mask[:, :, index, offset + index + 1:] = -65504
    inputs = dict(  # pylint: disable=use-dict-literal
        valid_seq_len=torch.tensor([offset], dtype=torch.int32),
        lmhead_idx=torch.tensor([sequence - 1], dtype=torch.int64),
        rope_cos=angles.cos().half(), rope_sin=angles.sin().half(),
        inputs_embeds=torch.randn(1, sequence, 128).half(),
        attention_mask=mask, past_key_values=caches,
    )
    with torch.no_grad():
        actual = fused(**inputs)
        expected = reference(**inputs)
    for output, golden in zip(actual, expected):
        torch.testing.assert_close(output, golden, atol=0.002, rtol=0.002)
