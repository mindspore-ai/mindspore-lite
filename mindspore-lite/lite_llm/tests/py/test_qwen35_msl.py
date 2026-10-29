"""Single-file paged packaging preserves independently addressable resources."""

import json
import struct
from types import SimpleNamespace

import pytest

from models.qwen3_5 import package
from utils import msl_pack


@pytest.mark.parametrize("output_name", ["model", "model.onnx"])
def test_export_rejects_directory_output(tmp_path, monkeypatch, output_name):
    """Reject directory outputs before accessing checkpoint weights."""
    from models.qwen3_5 import pipeline

    def unexpected_checkpoint_validation(model):
        del model
        pytest.fail("Output validation must run before checkpoint processing")

    monkeypatch.setattr(pipeline, "validate_checkpoint", unexpected_checkpoint_validation)
    args = SimpleNamespace(output=str(tmp_path / output_name), model="unused")
    with pytest.raises(ValueError, match="ending in .msl"):
        pipeline.export_qwen3_5(args, tmp_path)
    assert not (tmp_path / output_name).exists()


def test_paged_package_resources(tmp_path, monkeypatch):
    """Preserve individually addressable resources in a paged package."""
    root = tmp_path / "package"
    root.mkdir()
    manifest = {
        "model_name": "Qwen3.5-35B-A3B",
        "dtype": "float16",
        "architecture": {"name": "qwen3.5", "num_layers": 40, "hidden_size": 2048, "vocab_size": 248320},
        "assets": {"tokenizer": "tokenizer/vocab.bin"},
        "npu": {"max_length": 8192, "chunk_size": 64},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    long_name = "omc/resident/qwen35_resident_linear_s1_s64/qwen35_resident_linear_s1_s64.omc"
    payloads = {long_name: b"graph", "experts/layer_00/expert_000.q4": b"expert-q4",
                "experts_s16s4/layer_00/expert_000.s4": b"expert-s4", "tokenizer/vocab.bin": b"vocab"}
    for name, payload in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    # This fixture tests packaging, independently of full 40x256 export validation.
    monkeypatch.setattr(package, "validate_artifacts", lambda directory: None)
    output = tmp_path / "model.msl"
    package.pack_package(root, output)
    data = output.read_bytes()
    assert struct.unpack_from("<I", data, 4)[0] == 1
    assert struct.unpack_from("<I", data, 16)[0] == 65536
    unpacked = tmp_path / "unpacked"
    kv = msl_pack.unpack(output, unpacked)
    assert kv["arch.name"] == "qwen3.5"
    assert "npu.paged_moe_dir" not in kv
    assert kv["model.dtype"] == "float16"
    assert kv["npu.chunk_size"] == 64
    for name, payload in payloads.items():
        assert (unpacked / package.resource_name(name)).read_bytes() == payload
    assert package.resource_name(long_name) == "r/2fc75cb570689390"


def test_pack_streams_payloads(tmp_path, monkeypatch):
    """Stream resource payloads without loading entire files into memory."""
    source = tmp_path / "weight.bin"
    source.write_bytes(b"x" * (3 * 1024 * 1024 + 17))
    copied = []
    original = msl_pack.shutil.copyfileobj

    def copy(source_file, destination, length):
        copied.append(length)
        return original(source_file, destination, length)

    monkeypatch.setattr(msl_pack.shutil, "copyfileobj", copy)
    output = tmp_path / "model.msl"
    msl_pack.pack(output, {}, [("weight.bin", source, msl_pack.ACCESS_MMAP)])
    assert copied == [1024 * 1024]
    assert output.stat().st_size == 4096 + source.stat().st_size


def test_pack_rejects_duplicate_resources(tmp_path):
    source = tmp_path / "weight.bin"
    source.write_bytes(b"weight")
    with pytest.raises(msl_pack.MslPackError, match="duplicate resource"):
        msl_pack.pack(tmp_path / "bad.msl", {}, [("w", source, 0), ("w", source, 0)])
