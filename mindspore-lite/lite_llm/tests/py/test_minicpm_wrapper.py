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
"""Tests for the MiniCPM NNRT wrapper, exporter and GGUF weight mapping.

MiniCPM 1/2 is a verified LLaMA derivative (same projection/layernorm/MLP
attribute names, same attention math), so the numerical reference is a tiny
real ``LlamaForCausalLM`` — no MiniCPM weights required (transformers has no
built-in ``minicpm`` anymore).  ``scale_emb``, the one MiniCPM numerical
delta, is exercised as a parametrized case on both sides.

Covers, in order:
* eager numerics — wrapper logits/KV vs the HF reference (fp16 tolerances)
* traced graph — node-name scope + Ms* op multiset (2 layers)
* NNRT contract — 7 inputs + interleaved per-layer KV (shared-weight path)
* quant chain — W4A16 apply_quant + every GGUF loader map key resolvable

Requires the qwen2.5 export extras (``pip install -r requirements.txt``).
"""

import sys
from pathlib import Path

import pytest

_EXPORT_DIR = Path(__file__).resolve().parents[2] / "export"
sys.path.insert(0, str(_EXPORT_DIR))

# pylint: disable=wrong-import-position  # export/ added to sys.path above
from utils.onnx_postprocess import _save_onnx, validate_contract  # noqa: E402
from utils.export_quant import ModelConfig, QuantizationConfig, apply_quant, apply_shared_weight  # noqa: E402

onnx = pytest.importorskip("onnx")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from models.minicpm.minicpm_gguf_loader import QUANT_MATMUL_MAP  # noqa: E402
from models.minicpm.minicpm_wrapper import MiniCpmNnrtWrapper  # noqa: E402

SEQ_LEN, MAX_SEQ_LEN, NUM_LAYERS = 32, 64, 2
HEAD_DIM = 16  # hidden 64 / 4 heads


def _cache_kv(past_key_values, layer_idx):
    """Layer (key, value) from a transformers KV cache (4.x and 5.x layouts)."""
    if hasattr(past_key_values, "key_cache"):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]
    layer = past_key_values.layers[layer_idx]
    return layer.keys, layer.values


@pytest.fixture(scope="module", name="llama_ref")
def _llama_ref():
    """A tiny fp16 LlamaForCausalLM (deterministic weights) + input ids."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=176,
        num_hidden_layers=NUM_LAYERS, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=MAX_SEQ_LEN, rms_norm_eps=1e-5,
        tie_word_embeddings=True, rope_theta=10000.0,
    )
    config._attn_implementation = "eager"  # pylint: disable=protected-access
    model = LlamaForCausalLM(config).to(torch.float16).eval()
    input_ids = torch.randint(0, 128, (1, SEQ_LEN))
    return model, config, input_ids


@pytest.mark.parametrize("scale_emb", [1, 12])
def test_minicpm_wrapper_matches_hf_reference(llama_ref, scale_emb):
    """Wrapper logits/top-1/KV match the HF reference for both scale_emb cases."""
    model, config, input_ids = llama_ref
    if scale_emb == 1:
        config.__dict__.pop("scale_emb", None)
    else:
        config.scale_emb = scale_emb

    with torch.no_grad():
        if scale_emb == 1:
            reference = model(input_ids=input_ids, use_cache=True)
        else:  # MiniCPM applies scale_emb to the (CPU-side) embedding output.
            embeds = model.model.embed_tokens(input_ids) * scale_emb
            reference = model(inputs_embeds=embeds, use_cache=True)
        ref_logits = reference.logits[0, -1].float()

        dummy = torch.zeros(1, SEQ_LEN, 64, dtype=torch.float16)
        position_ids = torch.arange(SEQ_LEN).unsqueeze(0)
        rope_cos, rope_sin = model.model.rotary_emb(dummy, position_ids)  # [1, S, D]
        full = torch.full((SEQ_LEN, MAX_SEQ_LEN), torch.finfo(torch.float16).min, dtype=torch.float16)
        # NNRT attention faces the FULL cache: [1, 1, chunk, max_seq_len].
        mask = torch.triu(full, diagonal=1)[None, None]
        past = [
            [torch.zeros(1, 2, MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float16) for _ in range(2)]
            for _ in range(NUM_LAYERS)
        ]
        wrapper = MiniCpmNnrtWrapper(model, config).eval()
        out = wrapper(
            None,
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([SEQ_LEN - 1], dtype=torch.int32),
            rope_cos, rope_sin,
            model.model.embed_tokens(input_ids), mask, past,
        )
        wrapper_logits = out[0][0].float()

    rel_err = (wrapper_logits - ref_logits).abs().max().item() / (ref_logits.abs().max().item() + 1e-6)
    assert rel_err < 2e-2, f"logits rel_err {rel_err:.3e}"
    assert wrapper_logits.argmax().item() == ref_logits.argmax().item()

    for i in range(NUM_LAYERS):
        ref_key, ref_val = _cache_kv(reference.past_key_values, i)
        # Wrapper KV is the full cache; compare the chunk actually written.
        out_key = out[1 + 2 * i][:, :, :SEQ_LEN, :]
        out_val = out[2 + 2 * i][:, :, :SEQ_LEN, :]
        assert torch.allclose(out_key.float(), ref_key.float(), atol=5e-3)
        assert torch.allclose(out_val.float(), ref_val.float(), atol=5e-3)


def test_minicpm_export_traces_ms_ops_with_scope(llama_ref, tmp_path):
    """The traced graph keeps /model/... scopes and the expected Ms* multiset."""
    from models.minicpm.minicpm_exporter import MiniCpmOnnx

    model, config, _ = llama_ref
    config.__dict__.pop("scale_emb", None)
    exporter = MiniCpmOnnx()
    exporter.model, exporter.config = model, config
    exporter.num_layers, exporter.hidden_size, exporter.num_kv_heads = NUM_LAYERS, 64, 2
    onnx_path = str(tmp_path / "minicpm.onnx")
    exporter.export(onnx_path, max_seq_len=MAX_SEQ_LEN, chunk_size=SEQ_LEN)

    graph = onnx.load(onnx_path).graph
    node_names = [node.name for node in graph.node]
    op_counts = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    assert any(name.startswith("/model/layers.0/self_attn/q_proj/") for name in node_names), \
        "module scope lost — GGUF loaders map by node name"
    # Per layer: 1 rotary (q/k in one call) + 2 scatter (k, v) + 2 group
    # matmul (qk, pv) + 1 add-softmax; plus the final model norm.  Mul x2 per
    # layer = SwiGLU (silu*up), no scale_emb in this case.
    expected = {
        "MsRotaryPosEmb": NUM_LAYERS,
        "MsScatterND": 2 * NUM_LAYERS,
        "MsGroupMatmul": 2 * NUM_LAYERS,
        "MsAddSoftmax": NUM_LAYERS,
        "MsRmsNorm": 1,  # post-fuse: layer norms became MsAddRmsNorm
        "Mul": 2 * NUM_LAYERS,
    }
    for op_type, count in expected.items():
        assert op_counts.get(op_type, 0) == count, f"{op_type}: {op_counts.get(op_type, 0)} != {count}"
    assert "MsAddRmsNorm" in op_counts  # fuse_add_rmsnorm ran

    # Shared-weight (fp16) contract: embedding_weight inserted at index 6.
    model_onnx = onnx.load(onnx_path)
    apply_shared_weight(model_onnx)
    fp16_path = str(tmp_path / "minicpm_fp16.onnx")
    _save_onnx(model_onnx, fp16_path)
    validate_contract(fp16_path, NUM_LAYERS, embedding_quant=False)


def test_minicpm_quant_chain_resolves_gguf_map(llama_ref, tmp_path):
    """W4A16 apply_quant produces MsQuant4N0Group32 and every GGUF map key exists."""
    from models.minicpm.minicpm_exporter import MiniCpmOnnx

    model, config, _ = llama_ref
    config.__dict__.pop("scale_emb", None)
    exporter = MiniCpmOnnx()
    exporter.model, exporter.config = model, config
    exporter.num_layers, exporter.hidden_size, exporter.num_kv_heads = NUM_LAYERS, 64, 2
    onnx_path = str(tmp_path / "minicpm.onnx")
    exporter.export(onnx_path, max_seq_len=MAX_SEQ_LEN, chunk_size=SEQ_LEN)

    quant_config = ModelConfig(
        max_length=MAX_SEQ_LEN, chunk_size=SEQ_LEN, vocab_size=128, hidden_size=64,
        num_attention_heads=4, num_key_value_heads=2, eos_id=2,
        embedding_quant=QuantizationConfig("W4A16"),
        decoder_quant=QuantizationConfig("W4A16"),
    )
    quant_path = str(tmp_path / "minicpm_quant.onnx")
    apply_quant(onnx_path, quant_path, quant_config)

    quant_model = onnx.load(quant_path)
    op_counts = {}
    node_names = set()
    for node in quant_model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        node_names.add(node.name)

    assert op_counts.get("MsQuant4N0Group32", 0) > 0, "quant kernel missing"
    gguf_keys = {key.format(i) for i in range(NUM_LAYERS) for key in QUANT_MATMUL_MAP}
    missing = gguf_keys - node_names
    assert not missing, f"GGUF map keys unresolved: {sorted(missing)[:3]}"
    validate_contract(quant_path, NUM_LAYERS, embedding_quant=True)
