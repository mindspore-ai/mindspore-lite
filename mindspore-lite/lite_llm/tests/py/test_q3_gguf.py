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
"""Private type41 file, adapter packing, graph and rejection contracts."""

from enum import IntEnum
from types import SimpleNamespace
from typing import Union

import gguf
import numpy as np
import onnx
import pytest

from utils import gguf_mapping
from utils.export_quant import QuantizationConfig, quant_node_3bit_gp32
from utils.quantization import PackingLayout, QuantType, get_quant_preset
from mslite_llm_quant.q3_0 import BLOCK_BYTES, QH_BYTES, QS_BYTES, quantize_q3_0_blocks, register_private_q3_0_type


def _decode_shape(input1_shape):
    """Validate [K, N] for the test-only plane decoder (mirrors mslite_llm_quant.q3_0._shape)."""
    if not isinstance(input1_shape, (tuple, list)) or len(input1_shape) != 2:
        raise ValueError("input1_shape must be a list/tuple [K, N]")
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_))
           for v in input1_shape):
        raise ValueError("input1_shape must contain integers")
    k_dim, n_dim = map(int, input1_shape)
    if k_dim <= 0 or k_dim > 16384 or k_dim % 32 or n_dim <= 0 or n_dim % 16:
        raise ValueError("input1_shape requires 0<K<=16384, K%32==0 and positive N%16==0")
    if n_dim * k_dim // 32 * BLOCK_BYTES > np.iinfo(np.uint32).max:
        raise ValueError("weight exceeds kernel address limits")
    return k_dim, n_dim


def _tiles(k_dim, n_dim):
    """Enumerate the compact N64/K256 wire tiles, including both tails."""
    groups = k_dim // 32
    for n_start in range(0, n_dim, 64):
        for group_start in range(0, groups, 8):
            yield n_start, min(64, n_dim - n_start), group_start, min(8, groups - group_start)


def dequantize_q3_0_planes(blob: Union[bytes, np.ndarray], input1_shape) -> np.ndarray:
    """Reference decode of MsQuant3N0Group32 plane blobs to FP16 [K, N].

    Test-only helper; not a production runtime path.
    """
    k_dim, n_dim = _decode_shape(input1_shape)
    if isinstance(blob, bytes):
        raw = np.frombuffer(blob, dtype=np.uint8)
    elif isinstance(blob, np.ndarray) and blob.dtype == np.uint8:
        raw = np.ascontiguousarray(blob).reshape(-1)
    else:
        raise ValueError("weight must be bytes or a UINT8 ndarray")
    expected = n_dim * k_dim // 32 * BLOCK_BYTES
    if raw.size != expected:
        raise ValueError("packed weight byte size does not match input1_shape")
    result = np.empty((k_dim, n_dim), dtype=np.float16)
    j = np.arange(8, dtype=np.uint8)
    offset = 0
    for n_start, n_count, group_start, group_count in _tiles(k_dim, n_dim):
        count = n_count * group_count
        tile = raw[offset:offset + count * BLOCK_BYTES]
        qh = tile[:count * QH_BYTES]
        qs = tile[count * QH_BYTES:count * (QH_BYTES + QS_BYTES)].reshape(-1, 2)
        low = (qs[:, j // 4] >> (2 * (j % 4))) & 3
        high = (qh[:, None] >> j) & 1
        ordered = (low | (high << 2)).reshape(n_count // 16, group_count, 2, 4, 8, 4, 2)
        codes = ordered.transpose(0, 5, 3, 1, 2, 4, 6).reshape(count, 32)
        values = codes.astype(np.float32) - np.float32(4)
        scales = tile[count * (QH_BYTES + QS_BYTES):].view("<f2").astype(np.float32)
        decoded = (values * scales[:, None]).reshape(n_count, group_count * 32)
        result[group_start * 32:(group_start + group_count) * 32, n_start:n_start + n_count] = decoded.T
        offset += count * BLOCK_BYTES
    return result



def _raw_blocks():
    """Two distinguishable groups per row, including signed fp16 scales."""
    blocks = np.empty((16, 2, 14), dtype=np.uint8)
    scales = np.arange(1, 33, dtype=np.float16).reshape(16, 2) / 16
    scales[:, 1] *= -1
    blocks[:, :, :2] = np.ascontiguousarray(scales).view(np.uint8).reshape(16, 2, 2)
    blocks[:, :, 2:6] = np.arange(128, dtype=np.uint8).reshape(16, 2, 4)
    blocks[:, :, 6:] = np.arange(256, dtype=np.uint8).reshape(16, 2, 8)
    return blocks


def _write_mixed_file(path, blocks):
    """Write actual type41 and standard Q4 tensor records."""
    q3_type = register_private_q3_0_type()
    writer = gguf.GGUFWriter(str(path), "qwen2")
    try:
        writer.add_name("q3-roundtrip")
        writer.add_tensor("blk.0.attn_q.weight", blocks.reshape(16, 28), raw_dtype=q3_type)
        writer.add_tensor("token_embd.weight", np.zeros((16, 36), dtype=np.uint8),
                          raw_dtype=gguf.GGMLQuantizationType.Q4_0)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()


def test_type41_file_roundtrip_and_weight_values(tmp_path):
    """Real file injection preserves logical Q3 values across physical layouts."""
    blocks = _raw_blocks()
    path = tmp_path / "mixed.gguf"
    _write_mixed_file(path, blocks)
    reader = gguf_mapping.open_gguf_reader(path)
    assert gguf_mapping.detect_gguf_quantization(reader) == (QuantType.Q3_0, QuantType.Q4_0)
    tensor = next(item for item in reader.tensors if item.name == "blk.0.attn_q.weight")
    assert int(tensor.tensor_type) == 41
    assert tuple(tensor.shape) == (64, 16)
    assert tensor.n_bytes == 16 * 2 * 14
    np.testing.assert_array_equal(tensor.data, blocks.reshape(16, 28))
    embedding_path = tmp_path / "embedding.bin"
    weights = gguf_mapping.load_file_from_tensors(reader.tensors, embedding_path,
                                                  QuantType.Q3_0, QuantType.Q4_0)
    assert embedding_path.stat().st_size == 16 * 2 * 18
    restored = dequantize_q3_0_planes(weights[tensor.name], (64, 16))
    # Scalar decoding is intentionally independent of the adapter implementation.
    reference = np.empty((64, 16), dtype=np.float32)
    for row in range(16):
        for group in range(2):
            block = blocks[row, group]
            scale = block[:2].copy().view("<f2")[0]
            for lane in range(32):
                low = (int(block[6 + lane // 4]) >> (2 * (lane % 4))) & 3
                high = (int(block[2 + lane // 8]) >> (lane % 8)) & 1
                reference[group * 32 + lane, row] = ((low | (high << 2)) - 4) * scale
    np.testing.assert_array_equal(restored, reference)


def test_registration_is_idempotent():
    """Repeated registration keeps the same enum member and size contract."""
    first = register_private_q3_0_type()
    enum = gguf.GGMLQuantizationType
    assert register_private_q3_0_type() is first
    assert gguf.GGMLQuantizationType is enum
    assert first.name == "Q3_0"
    assert first.value == 41
    assert gguf.GGML_QUANT_SIZES[first] == (32, 14)


@pytest.mark.parametrize("members", [{"UPSTREAM_TYPE": 41}, {"Q3_0": 42}])
def test_registration_rejects_wire_type_collision(monkeypatch, members):
    """Neither a reused wire ID nor a conflicting symbolic name is overwritten."""
    monkeypatch.setattr(gguf, "GGMLQuantizationType", IntEnum("ConflictingType", members))
    with pytest.raises(ValueError, match="41"):
        register_private_q3_0_type()


def test_old_type1024_rejected(tmp_path):
    """Legacy decoder wire IDs are rejected before gguf resolves the enum."""
    path = tmp_path / "old.gguf"
    _write_mixed_file(path, _raw_blocks())
    data = path.read_bytes()
    name = b"blk.0.attn_q.weight"
    position = data.index(name) + len(name)
    position += 4 + 2 * 8
    assert data[position:position + 4] == (41).to_bytes(4, "little")
    path.write_bytes(data[:position] + (1024).to_bytes(4, "little") + data[position + 4:])
    with pytest.raises(ValueError, match="legacy private type-1024 Q3 GGUF is no longer supported"):
        gguf_mapping.open_gguf_reader(path)


def test_reader_preserves_parse_errors(monkeypatch, tmp_path):
    """Malformed input errors must remain the original exception."""
    error = ValueError("unrelated malformed file, no enum wording")

    def malformed(path):
        del path
        raise error

    monkeypatch.setattr(gguf, "GGUFReader", malformed)
    with pytest.raises(ValueError) as caught:
        gguf_mapping.open_gguf_reader(tmp_path / "invalid.gguf")
    assert caught.value is error


@pytest.mark.parametrize("shape", [(16, 13), (16, 0), (0, 14), (15, 14), (16, 7182)])
def test_repack_rejects_invalid_dimensions(shape):
    """Reject partial blocks, empty dimensions and unsupported kernel shapes."""
    with pytest.raises(ValueError):
        gguf_mapping.rearrange_q3_0_g32(np.zeros(shape, dtype=np.uint8))


@pytest.mark.parametrize("data", [np.zeros(14, dtype=np.uint8), np.zeros((16, 14), dtype=np.float16)])
def test_repack_rejects_rank_or_dtype(data):
    """Only two-dimensional packed UINT8 rows are accepted."""
    with pytest.raises(ValueError):
        gguf_mapping.rearrange_q3_0_g32(data)


@pytest.mark.parametrize("shape", [(32, 16), (16384, 16)])
def test_q3_minimum_and_maximum_k(shape):
    """Both supported K boundaries pack and decode zero weights exactly."""
    weight = np.zeros(shape, dtype=np.float32)
    blocks = quantize_q3_0_blocks(weight).reshape(shape[1], shape[0] // 32 * 14)
    packed = gguf_mapping.rearrange_q3_0_g32(blocks)
    assert packed.size == shape[0] * shape[1] // 32 * 14
    np.testing.assert_array_equal(dequantize_q3_0_planes(packed, shape), weight)


@pytest.mark.parametrize("value", ["msl_q3_0", "msl_q3_0".upper()])
def test_q3_rejects_removed_config_name(value):
    """The removed export name must not remain an API/config alias."""
    with pytest.raises(ValueError):
        QuantType(value)
    with pytest.raises(ValueError):
        QuantizationConfig(value)


def test_q3_config_serializes_canonical_name():
    """Exported quantization metadata uses q3_0, not the GGUF wire name."""
    config = QuantizationConfig("q3_0")
    assert config.asdict() == {"quant_method": "q3_0", "bits": 3, "group_size": 32}


def test_q3_graph_attribute_and_shared_preset():
    """The graph uses ListInt shapes and the canonical decoder-only preset."""
    weight = np.tile(np.arange(-4, 4, dtype=np.float32), (16, 4)).T.copy()
    initializer = onnx.numpy_helper.from_array(weight, "projection.weight")
    node = onnx.helper.make_node("MatMul", ["hidden", initializer.name], ["out"], name="projection/MatMul")
    nodes, weights = quant_node_3bit_gp32({"hidden": [1, 1, 32]}, node, {initializer.name: initializer})
    assert nodes[0].op_type == "MsQuant3N0Group32"
    assert onnx.helper.get_attribute_value(nodes[0].attribute[0]) == [32, 16]
    assert list(weights[0].dims) == [16 * 14]
    config = QuantizationConfig("W3A16")
    assert (config.bits, config.group_size, config.quant_method) == (3, 32, QuantType.Q3_0)
    preset = get_quant_preset(QuantType.Q3_0)
    assert preset.packing_layout is PackingLayout.Q3_PLANAR
    assert preset.embedding_format is None
    preset.validate_target("kirin9020")
    with pytest.raises(ValueError, match="kirin9020"):
        preset.validate_target("kirin9030")


@pytest.mark.parametrize("decoder_types,embedding_type", [([41, 2], 2), ([], 2), ([41], 1)])
def test_detection_rejects_mixed_or_missing_contract(decoder_types, embedding_type):
    """A mixed decoder or non-Q4 embedding cannot masquerade as supported Q3."""
    tensors = [SimpleNamespace(name=f"blk.{i}.attn_q.weight", tensor_type=kind)
               for i, kind in enumerate(decoder_types)]
    tensors.append(SimpleNamespace(name="token_embd.weight", tensor_type=embedding_type))
    with pytest.raises(ValueError):
        gguf_mapping.detect_gguf_quantization(SimpleNamespace(tensors=tensors))


def test_hf_safetensors_to_real_mixed_gguf(tmp_path):
    """Preserve HF->GGUF without materializing a transformers model."""
    from safetensors.numpy import save_file
    from mslite_llm_quant.gguf_quant import _add_weights

    source = {
        "model.layers.0.self_attn.q_proj.weight": np.tile(np.arange(-4, 4, dtype=np.float32), (16, 8)),
        "model.embed_tokens.weight": np.zeros((16, 64), dtype=np.float32),
        "model.norm.weight": np.ones(64, dtype=np.float32),
    }
    save_file(source, str(tmp_path / "model.safetensors"))
    path = tmp_path / "from-hf.gguf"
    writer = gguf.GGUFWriter(str(path), "qwen2", use_temp_file=True)
    try:
        _add_weights(writer, tmp_path, "q3_0")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    reader = gguf_mapping.open_gguf_reader(path)
    assert gguf_mapping.detect_gguf_quantization(reader) == (QuantType.Q3_0, QuantType.Q4_0)
    tensor = next(item for item in reader.tensors if item.name == "blk.0.attn_q.weight")
    assert tuple(tensor.shape) == (64, 16)
    assert tensor.data.shape == (16, 28)
    packed = gguf_mapping.rearrange_q3_0_g32(tensor.data)
    actual = dequantize_q3_0_planes(packed, (64, 16))
    np.testing.assert_array_equal(actual, source["model.layers.0.self_attn.q_proj.weight"].T)


def test_reader_registers_before_parsing_real_type41(tmp_path, monkeypatch):
    """A reader with no Q3 enum entry succeeds without retrying a ValueError."""
    from importlib import import_module

    path = tmp_path / "fresh-reader.gguf"
    _write_mixed_file(path, _raw_blocks())
    monkeypatch.delitem(gguf.GGML_QUANT_SIZES, 41, raising=False)
    original = IntEnum("GGMLQuantizationType", {
        name: member.value for name, member in gguf.GGMLQuantizationType.__members__.items()
        if name != "Q3_0"
    })
    for name in ("gguf", "gguf.constants", "gguf.gguf_reader", "gguf.gguf_writer", "gguf.quants"):
        monkeypatch.setattr(import_module(name), "GGMLQuantizationType", original)
    reader = gguf_mapping.open_gguf_reader(path)
    assert int(reader.tensors[0].tensor_type) == 41
    assert gguf_mapping.detect_gguf_quantization(reader)[0] is QuantType.Q3_0


def test_hf_checkpoint_index_rejects_outside_shard(tmp_path):
    """Checkpoint index paths cannot escape the supplied model directory."""
    import json
    from mslite_llm_quant.gguf_quant import _checkpoint_files

    outside = tmp_path / "outside.safetensors"
    outside.touch()
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.norm.weight": "../outside.safetensors"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="out-of-directory"):
        _checkpoint_files(model)


@pytest.mark.parametrize("override", [{"hidden_size": 1024}, {"tie_word_embeddings": False}])
def test_hf_gguf_export_rejects_unsupported_architecture(override):
    """HF GGUF conversion is restricted to the supported tied architecture."""
    from mslite_llm_quant.gguf_quant import SUPPORTED_MODEL, validate_config

    config = SimpleNamespace(**{**SUPPORTED_MODEL, "tie_word_embeddings": True, **override})
    with pytest.raises(ValueError, match="Qwen2.5-0.5B"):
        validate_config(config, "q3_0")


def test_hf_gguf_tokenizer_preserves_nonspecial_added_tokens_and_absent_bos(tmp_path):
    """Added-token boundaries and unset BOS survive a real GGUF metadata roundtrip."""
    import json
    from mslite_llm_quant.gguf_quant import _add_tokenizer
    from utils.export_tokenizer import _GGUFTokenizerAdapter, _qwen_bpe_backend

    tokenizer = SimpleNamespace(
        get_vocab=lambda: {"a": 0, "b": 1, "<eos>": 2, "<tool>": 3},
        get_added_vocab=lambda: {"<eos>": 2, "<tool>": 3},
        all_special_tokens=["<eos>"], bos_token_id=None, eos_token_id=2,
        pad_token_id=2, add_bos_token=False, chat_template="",
    )
    config = SimpleNamespace(vocab_size=4, bos_token_id=2)
    (tmp_path / "tokenizer.json").write_text(
        json.dumps({"model": {"merges": []}}), encoding="utf-8"
    )
    path = tmp_path / "tokenizer.gguf"
    writer = gguf.GGUFWriter(str(path), "qwen2")
    try:
        _add_tokenizer(writer, config, tokenizer, tmp_path)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    actual = _GGUFTokenizerAdapter(str(path))
    assert actual.get_vocab() == tokenizer.get_vocab()
    assert actual.bos_token_id is None
    assert (actual.eos_token_id, actual.pad_token_id) == (2, 2)
    assert [item["id"] for item in _qwen_bpe_backend(actual)["added_tokens"]] == [2, 3]


def _qwen_tensor_metadata(decoder_type=41):
    """Describe all 24 layers without allocating any checkpoint payload."""
    layer_shapes = {
        "attn_q.weight": (896, 896),
        "attn_k.weight": (896, 128),
        "attn_v.weight": (896, 128),
        "attn_output.weight": (896, 896),
        "ffn_gate.weight": (896, 4864),
        "ffn_up.weight": (896, 4864),
        "ffn_down.weight": (4864, 896),
        "attn_q.bias": (896,),
        "attn_k.bias": (128,),
        "attn_v.bias": (128,),
        "attn_norm.weight": (896,),
        "ffn_norm.weight": (896,),
    }
    tensors = [
        SimpleNamespace(
            name=f"blk.{layer}.{name}", shape=shape,
            tensor_type=decoder_type if len(shape) == 2 else 0,
        )
        for layer in range(24) for name, shape in layer_shapes.items()
    ]
    tensors.extend([
        SimpleNamespace(name="token_embd.weight", shape=(896, 151936), tensor_type=2),
        SimpleNamespace(name="output_norm.weight", shape=(896,), tensor_type=0),
    ])
    return tensors


@pytest.mark.parametrize("decoder_type", [41, 2])
@pytest.mark.parametrize("include_output", [False, True])
def test_qwen_gguf_valid_logical_shapes_without_payloads(decoder_type, include_output):
    """Mixed Q3/Q4 and all-Q4 metadata validate without accessing tensor data."""
    from models.qwen2_5.qwen2_5_gguf_loader import _validate_tensor_shapes

    tensors = _qwen_tensor_metadata(decoder_type)
    if include_output:
        tensors.append(SimpleNamespace(name="output.weight", shape=(896, 151936), tensor_type=8))
    _validate_tensor_shapes(tensors, 24)


@pytest.mark.parametrize("name,shape", [
    ("blk.0.attn_q.weight", (448, 1792)),
    ("blk.0.attn_q.weight", (896, 896, 1)),
    ("blk.0.attn_k.weight", (128, 896)),
    ("blk.0.attn_v.weight", (448, 256)),
    ("blk.0.attn_output.weight", (1792, 448)),
    ("blk.0.ffn_gate.weight", (4864, 896)),
    ("blk.0.ffn_up.weight", (448, 9728)),
    ("blk.23.ffn_down.weight", (896, 4864)),
    ("blk.0.attn_q.bias", (1, 896)),
    ("blk.0.attn_k.bias", (127,)),
    ("blk.0.attn_v.bias", (129,)),
    ("blk.0.attn_norm.weight", (895,)),
    ("blk.23.ffn_norm.weight", (897,)),
    ("output_norm.weight", (0,)),
    ("output_norm.weight", ()),
    ("token_embd.weight", (151936, 896)),
    ("token_embd.weight", (896, 151935)),
    ("output.weight", (448, 303872)),
])
def test_qwen_gguf_rejects_logical_shape_mismatches(name, shape):
    """Check axes and rank, every required tensor role, and both layer edges."""
    from models.qwen2_5.qwen2_5_gguf_loader import _validate_tensor_shapes

    tensors = _qwen_tensor_metadata()
    if name == "output.weight":
        tensors.append(SimpleNamespace(name=name, shape=shape, tensor_type=8))
    else:
        next(tensor for tensor in tensors if tensor.name == name).shape = shape
    with pytest.raises(ValueError, match=rf"{name} has logical shape .*expected"):
        _validate_tensor_shapes(tensors, 24)


@pytest.mark.parametrize("name", [
    "blk.0.attn_q.weight", "blk.23.ffn_down.weight", "blk.0.attn_k.bias",
    "blk.23.ffn_norm.weight", "output_norm.weight", "token_embd.weight",
])
def test_qwen_gguf_rejects_missing_required_tensor(name):
    """Missing tensors fail metadata validation before any payload is packed."""
    from models.qwen2_5.qwen2_5_gguf_loader import _validate_tensor_shapes

    tensors = [tensor for tensor in _qwen_tensor_metadata() if tensor.name != name]
    with pytest.raises(ValueError, match=f"Missing required GGUF tensor {name}"):
        _validate_tensor_shapes(tensors, 24)


@pytest.mark.parametrize("decoder_type,block_bytes", [(41, 14), (2, 18)])
def test_qwen_loader_rejects_aligned_same_byte_count_tensor(tmp_path, decoder_type, block_bytes):
    """A real GGUF's wrong [448,1792] Q projection cannot enter packing/injection."""
    from models.qwen2_5.qwen2_5_gguf_loader import gguf_loader

    register_private_q3_0_type()
    path = tmp_path / "wrong-shape.gguf"
    blocks = np.zeros((1792, 448 // 32 * block_bytes), dtype=np.uint8)
    writer = gguf.GGUFWriter(str(path), "qwen2")
    try:
        writer.add_tensor("blk.0.attn_q.weight", blocks,
                          raw_dtype=gguf.GGMLQuantizationType(decoder_type))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    tensor = gguf_mapping.open_gguf_reader(path).tensors[0]
    assert tuple(tensor.shape) == (448, 1792)
    assert tensor.n_bytes == 896 * 896 // 32 * block_bytes
    embedding_path = tmp_path / "embedding.bin"
    output_path = tmp_path / "output.onnx"
    with pytest.raises(ValueError, match=r"attn_q.weight has logical shape .*expected \(896, 896\)"):
        gguf_loader(
            path, tmp_path / "not-loaded.onnx", output_path, embedding_path,
            decoder_quantize_config=QuantType.Q3_0 if decoder_type == 41 else QuantType.Q4_0,
        )
    assert not embedding_path.exists()
    assert not output_path.exists()


def test_registration_rejects_foreign_type41(monkeypatch):
    """An upstream type already using 41 is never overwritten."""
    members = {name: member.value for name, member in gguf.GGMLQuantizationType.__members__.items()
               if name != "Q3_0"}
    foreign = IntEnum("GGMLQuantizationType", {**members, "FOREIGN_41": 41})
    monkeypatch.setattr(gguf, "GGMLQuantizationType", foreign)
    monkeypatch.setitem(gguf.GGML_QUANT_SIZES, foreign.FOREIGN_41, (32, 18))
    with pytest.raises(ValueError, match="type 41 conflicts"):
        register_private_q3_0_type()
