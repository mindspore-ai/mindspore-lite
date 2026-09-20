"""Finalize and pack paged Qwen3.5 artifacts for the public runtime."""

import json
from pathlib import Path

from utils.export_tokenizer import export_tokenizer
from utils.msl_pack import ACCESS_MMAP, ACCESS_READ, manifest_to_kv, pack
from .gptq import quantization, text_config
from .export_qwen_tokenizer import export_binary


CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}"
    "{%- endif %}"
)


def resource_name(relative):
    """Match ArtifactResourceName: compact long paths without changing v1."""
    encoded = relative.encode("utf-8")
    if len(encoded) < 64:
        return relative
    value = 14695981039346656037
    for byte in encoded:
        value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return f"r/{value:016x}"


def pack_package(package_dir, output_path):
    """Stream a validated paged directory into a directly pageable v1 .msl."""
    package_dir = Path(package_dir)
    validate_artifacts(package_dir)
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    resources = []
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or path.name.endswith(".compile.log"):
            continue
        relative = path.relative_to(package_dir).as_posix()
        access = ACCESS_READ if path.suffix == ".json" or relative.startswith("tokenizer/") else ACCESS_MMAP
        resources.append((resource_name(relative), str(path), access))
    # 64 KiB supports 4/16/64 KiB mmap offsets on 64-bit device kernels.
    return pack(str(output_path), manifest_to_kv(manifest), resources, alignment=65536)


def validate_checkpoint(model_dir):
    """Reject incompatible models before exporting fixed-shape custom graphs."""
    model_dir = Path(model_dir)
    config = text_config(model_dir)
    expected = {
        "hidden_size": 2048, "vocab_size": 248320, "num_hidden_layers": 40,
        "num_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 512,
        "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"Qwen3.5-35B requires {field}={value}, got {config.get(field)!r}")
    layers = ["full_attention" if index % 4 == 3 else "linear_attention" for index in range(40)]
    if config.get("layer_types") != layers:
        raise ValueError("Qwen3.5-35B requires the 30 linear / 10 full attention layer layout")
    rope = config.get("rope_parameters", {})
    if rope.get("rope_theta") != 10000000 or rope.get("partial_rotary_factor") != 0.25:
        raise ValueError("Qwen3.5-35B requires theta=10000000 and rotary fraction=0.25")
    quantization(model_dir)
    return config


def finalize_package(model_dir, output_dir):
    """Add the native tokenizer and manifest to an exported artifact directory.

    Large expert shards stay as individual files for bounded-memory paging.
    Call this only after all resident weights, expert shards and OMCs exist.
    """
    config = validate_checkpoint(model_dir)
    output_dir = Path(output_dir)
    validate_artifacts(output_dir)
    required = ["model.json", "global/embedding_fp16.bin", "global/lm_head_nz_fp16.bin",
                "experts/manifest.json", "experts_s16s4/manifest.json", "experts_s16s4/READY"]
    for relative in required:
        if not (output_dir / relative).is_file():
            raise FileNotFoundError(output_dir / relative)
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    export_binary(Path(model_dir), tokenizer_dir / "qwen35.bin")
    export_tokenizer(model_dir, tokenizer_dir, chat_template=CHAT_TEMPLATE, native_encoding=True)
    generation = json.loads((tokenizer_dir / "generation_policy.json").read_text())
    manifest = {
        "model_name": "Qwen3.5-35B-A3B", "version": "1", "format_version": "1.0",
        "dtype": "float16",
        "architecture": {
            "name": "qwen3.5",
            "num_layers": 40, "hidden_size": 2048, "intermediate_size": 512,
            "num_heads": 16, "num_kv_heads": 2, "head_dim": 256,
            "vocab_size": 248320, "max_position_embeddings": 8192,
            "rope_theta": 10000000, "norm_eps": config["rms_norm_eps"],
            "tie_word_embeddings": False,
        },
        "assets": {"tokenizer": "tokenizer/vocab.bin"},
        "npu": {"max_length": 8192, "chunk_size": 64},
        "generation": generation,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def validate_artifacts(root):
    """Reject missing layers, partial expert exports and truncated weight files."""
    root = Path(root).resolve()

    def require(relative, size=None):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or unsafe artifact: {relative}")
        actual = path.stat().st_size
        if actual == 0 or (size is not None and actual != size):
            raise ValueError(f"Invalid artifact size: {relative}: {actual}, expected {size}")
        return path

    require("model.json")
    require("global/embedding_fp16.bin", 248320 * 2048 * 2)
    require("global/lm_head_nz_fp16.bin", 248320 * 2048 * 2)
    require("global/final_norm_fp16.bin", 2048 * 2)
    for layer in range(40):
        directory = Path("weights") / f"layer_{layer:02d}"
        manifest = json.loads(require(directory / "manifest.json").read_text())
        if manifest.get("layer") != layer or not manifest.get("weight_layout"):
            raise ValueError(f"Invalid resident layer manifest: {directory}")
        for name, layout in manifest["weight_layout"].items():
            require(directory / manifest["inputs"][name], layout["length"])
    for directory, suffix in (("experts", "q4"), ("experts_s16s4", "s4")):
        manifest = json.loads(require(f"{directory}/manifest.json").read_text())
        layers = manifest.get("layers", {})
        if set(layers) != {str(index) for index in range(40)}:
            raise ValueError(f"Incomplete expert layers: {directory}")
        for layer, info in layers.items():
            experts = info.get("experts", {})
            if set(experts) != {str(index) for index in range(256)}:
                raise ValueError(f"Incomplete expert shards: {directory}/layer_{layer}")
            for expert, record in experts.items():
                relative = f"layer_{int(layer):02d}/expert_{int(expert):03d}.{suffix}"
                if record["file"] != relative:
                    raise ValueError(f"Unexpected expert path: {record['file']}")
                require(f"{directory}/{relative}", record["bytes"])
    require("experts_s16s4/READY")
    for kind in ("linear", "full"):
        stem = f"qwen35_resident_{kind}_s1_s64"
        require(f"omc/resident/{stem}/{stem}.omc")
        require(f"omc/resident/{stem}/SubGraph_0.weight")
    for directory, prefix, sequences in (
        ("expert", "qwen35_moe", (1, 8, 64)),
        ("expert_s16s4", "qwen35_moe_s16s4", (1, 64)),
    ):
        for sequence in sequences:
            for stage in ("gate_up", "down"):
                require(f"omc/{directory}/{prefix}_{stage}_s{sequence}.omc")
    require("omc/expert_s16s4/qwen35_moe_s16s4_grouped_top8_packed32_s1.omc")
    require("omc/lm_head/final_norm.omc")
    require("omc/lm_head/lm_head_62080.omc")
