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
"""Compiler input sizes and opt-in external OMC package weights."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest


EXPORT_DIR = Path(__file__).resolve().parents[2] / "export"
sys.path.insert(0, str(EXPORT_DIR))

from utils import msl_pack, omc_compiler, ensure_custom_ops, export_quant  # pylint: disable=wrong-import-position


COMPACT_LAYOUT = "q4_0_nzf_compact_phase4"


def _architecture():
    return {
        "num_layers": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 16,
        "vocab_size": 64,
        "max_position_embeddings": 128,
    }


@pytest.mark.parametrize("n,k", [(16, 32), (48, 896), (80, 1056)])
def test_omg_embedding_input_matches_compact_payload(n, k):
    """OMG must receive live byte counts even when both tile dimensions end."""
    ensure_custom_ops()
    from torch_custom.ms_quant4_n0_group32 import MsQuant4N0Group32  # pylint: disable=import-outside-toplevel

    arch = _architecture()
    arch.update(vocab_size=n, hidden_size=k)
    command = omc_compiler.build_omg_command(
        "input.onnx", "output", arch, 1024, (1, 64), "W4A16", omg="omg", save_external_weights=True
    )
    shapes = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--input_shape="))
    dimensions = dict(part.split(":", 1) for part in shapes.split(";"))
    payload = MsQuant4N0Group32.quantize_weight_g32_4bit(np.zeros((k, n), dtype=np.float16))
    assert dimensions["embedding_weight"] == str(payload.size) == str(n * k // 32 * 18)
    assert dimensions["past_key_0"] == "1,1,1024,16"
    assert "--dynamic_dims=1,1,1,1;64,64,64,64" in command
    assert "--save_weights_as_external_data=true" in command


@pytest.mark.parametrize("n,k", [(1, 32), (17, 32), (16, 31), (16, 0), (0, 32)])
def test_omg_rejects_shapes_outside_compact_kernel_contract(n, k):
    arch = _architecture()
    arch.update(vocab_size=n, hidden_size=k)
    with pytest.raises(ValueError):
        omc_compiler.build_omg_command("input.onnx", "output", arch, 1024, (64,), "W4A16", omg="omg")


@pytest.mark.parametrize("quant,expected", [(None, 17 * 128), ("FP16", 17 * 128), ("W4A8", 32 * 68)])
def test_non_w4a16_embedding_sizes_keep_their_original_contract(quant, expected):
    assert omc_compiler.embedding_weight_elems(17, 128, quant) == expected


def _package_inputs(tmp_path, layout):
    """Create real package assets and caller-owned metadata for both modes."""
    files = {}
    for name in ("model.omc", "vocab.bin", "embedding.bin", "cos.bin", "sin.bin", "mask.bin"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = str(path)
    npu = {
        "max_length": 128, "chunk_size": 32, "embedding_quant": layout is not None,
        "om_weight_dir": "stale_directory",
    }
    if layout is not None:
        npu["q4_0_weight_layout"] = layout
    return {
        "omc_path": files["model.omc"], "vocab_path": files["vocab.bin"],
        "embedding_path": files["embedding.bin"], "rope_cos": files["cos.bin"],
        "rope_sin": files["sin.bin"], "attention_mask": files["mask.bin"],
        "architecture": _architecture(), "npu_config": npu,
        "generation_policy": {}, "package_name": "test",
    }


@pytest.mark.parametrize("layout", [None, COMPACT_LAYOUT])
@pytest.mark.parametrize("weight_dir", ["weights", "decoder_payload"])
def test_external_weight_package_is_opt_in(tmp_path, layout, weight_dir):
    """Only an explicit external payload can create a packaged weight resource."""
    common = _package_inputs(tmp_path, layout)
    external = tmp_path / "SubGraph_0.weight"
    external.write_bytes(b"external-weights")
    embedded = tmp_path / "embedded.msl"
    msl_pack.build_single_file_msl(output_path=str(embedded), **common)
    embedded_dir = tmp_path / "embedded"
    embedded_kv = msl_pack.unpack(str(embedded), str(embedded_dir))
    assert "npu.om_weight_dir" not in embedded_kv
    assert not (embedded_dir / "SubGraph_0.weight").exists()
    assert embedded_kv.get("npu.q4_0_weight_layout") == layout

    split = tmp_path / "split.msl"
    msl_pack.build_single_file_msl(
        output_path=str(split), external_weight_path=str(external),
        external_weight_dir=weight_dir, **common
    )
    split_dir = tmp_path / "split"
    split_kv = msl_pack.unpack(str(split), str(split_dir))
    assert split_kv["npu.om_weight_dir"] == weight_dir
    assert (split_dir / "SubGraph_0.weight").read_bytes() == b"external-weights"
    assert split_kv["npu.embedding_quant"] is (layout is not None)
    assert split_kv.get("npu.q4_0_weight_layout") == layout
    assert (split_dir / split_kv["litert.prefill.path"]).read_bytes() == b"model.omc"
    assert common["npu_config"]["om_weight_dir"] == "stale_directory"


@pytest.mark.parametrize("layout,embedding_quant", [
    (None, False), (None, True), (COMPACT_LAYOUT, True), ("q4_0_nzf_phase4", True),
    ("q4_0_nzf", True), ("planar", True), ("future_layout_v2", True),
])
def test_manifest_preserves_layout_for_runtime_validation(tmp_path, layout, embedding_quant):
    """Packaging never upgrades legacy/unknown blobs by rewriting their marker.

    The runtime, not the generic KV serializer, rejects missing/legacy markers
    for W4A16. FP16 serialization must not invent a quantized layout either.
    """
    npu_config = {"max_length": 128, "chunk_size": 32, "embedding_quant": embedding_quant}
    if layout is not None:
        npu_config["q4_0_weight_layout"] = layout
    manifest = msl_pack.build_manifest("qwen", _architecture(), npu_config, {}, "model.omc")
    kv = msl_pack.manifest_to_kv(manifest)
    package = tmp_path / "metadata.msl"
    msl_pack.pack(str(package), kv, [])
    decoded = msl_pack.unpack(str(package), str(tmp_path / "unpacked"))
    assert decoded == kv
    if layout is None:
        assert "q4_0_weight_layout" not in manifest["npu"]
        assert "npu.q4_0_weight_layout" not in decoded
    else:
        assert manifest["npu"]["q4_0_weight_layout"] == layout
        assert decoded["npu.q4_0_weight_layout"] == layout
    assert "om_weight_dir" not in npu_config


@pytest.mark.parametrize("family", [
    ("models.qwen2_5.qwen2_5_exporter", "Qwen2Onnx", "Qwen2"),
    ("models.qwen3.qwen3_exporter", "Qwen3Onnx", "Qwen3"),
    ("models.minicpm.minicpm_exporter", "MiniCpmOnnx", "Llama"),
])
@pytest.mark.parametrize("quant_method,layout", [(None, None), ("W4A8", None), ("W4A16", COMPACT_LAYOUT)])
def test_quantized_exporter_fragment_preserves_layout_in_package(tmp_path, family, quant_method, layout):
    """Each producer's layout declaration must survive both package-entry paths."""
    module_name, exporter_name, hf_family = family
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config_type = getattr(transformers, hf_family + "Config")
    model_type = getattr(transformers, hf_family + "ForCausalLM")
    config = config_type(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=128, eos_token_id=2,
    )
    exporter = getattr(importlib.import_module(module_name), exporter_name)()
    with torch.random.fork_rng(devices=[]):
        exporter.model = model_type(config)
    quant = export_quant.QuantizationConfig(quant_method)
    extra = {"model_name": "tiny"} if hf_family == "Qwen3" else {}
    fragment = exporter.build_config(128, 32, quant, quant, **extra)
    manifest = msl_pack.build_manifest(
        "tiny", fragment["architecture"], fragment["npu"], fragment["generation"], "model.omc"
    )
    for name, metadata in (("fragment", fragment), ("manifest", manifest)):
        package = tmp_path / f"{name}.msl"
        msl_pack.pack(str(package), msl_pack.manifest_to_kv(metadata), [])
        decoded = msl_pack.unpack(str(package), str(tmp_path / name))
        assert decoded["npu.embedding_quant"] is (quant_method is not None)
        assert decoded.get("npu.q4_0_weight_layout") == layout
