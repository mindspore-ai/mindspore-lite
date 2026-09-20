# Adapted from HMOS-Device-AI converter/qwen3.5-35b-a3b; see source_provenance.json.
#!/usr/bin/env python3
"""Build the paged-runtime manifest without loading checkpoint tensors."""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path

EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(qweight|qzeros|scales|g_idx)$"
)
LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def align(value: int, alignment: int = 4096) -> int:
    return (value + alignment - 1) // alignment * alignment


def fused_expert_shard_bytes() -> int:
    # Eight Q35GPTQ tensors: fused Gate/Up and Down qweight/qzeros/scales/g_idx.
    """Calculate the packed byte count for one fused expert shard."""
    tensor_bytes = (
        256 * 1024 * 4,
        16 * 128 * 4,
        16 * 1024 * 2,
        2048 * 4,
        512 * 256 * 4,
        4 * 256 * 4,
        4 * 2048 * 2,
        512 * 4,
    )
    offset = align(64 + len(tensor_bytes) * 128)
    for size in tensor_bytes:
        offset = align(offset)
        offset += size
    return align(offset)


def tensor_metadata(model_dir: Path, weight_map: dict[str, str]) -> dict[str, dict]:
    """Read safetensors headers only; tensor payloads remain unmapped by NumPy."""
    by_file: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        by_file[shard].append(name)

    result: dict[str, dict] = {}
    for shard, names in sorted(by_file.items()):
        path = model_dir / shard
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise ValueError(f"Truncated safetensors header: {path}")
            header_length = struct.unpack("<Q", length_bytes)[0]
            if header_length > min(file_size - 8, 100_000_000):
                raise ValueError(f"Invalid safetensors header length: {path}")
            header = json.loads(stream.read(header_length))
            for name in names:
                info = header[name]
                shape = list(info["shape"])
                dtype = info["dtype"]
                elements = 1
                for dim in shape:
                    elements *= dim
                if dtype not in DTYPE_BYTES:
                    raise ValueError(f"unsupported safetensors dtype {dtype}: {name}")
                begin, end = info["data_offsets"]
                size = elements * DTYPE_BYTES[dtype]
                if begin < 0 or end - begin != size or 8 + header_length + end > file_size:
                    raise ValueError(f"Invalid safetensors tensor range: {name}")
                result[name] = {
                    "shape": shape,
                    "dtype": dtype,
                    "bytes": size,
                    "source_shard": shard,
                }
    return result


def component(name: str) -> str:
    """Describe a compiled component and its external weight resource."""
    if name == "model.language_model.embed_tokens.weight":
        return "embedding"
    if name == "lm_head.weight":
        return "lm_head"
    if name.startswith("model.visual."):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    match = LAYER_RE.match(name)
    if match:
        return f"layer_{int(match.group(1)):02d}_resident"
    return "text_global"


def parse_arguments():
    """Parse and validate command-line options for this exporter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-root", default="experts")
    parser.add_argument("--layer-omc-root", default="omc/layers")
    parser.add_argument(
        "--expert-gate-up-omc",
        default="omc/expert/qwen35_moe_gate_up_s{sequence}.omc",
    )
    parser.add_argument(
        "--expert-down-omc",
        default="omc/expert/qwen35_moe_down_s{sequence}.omc",
    )
    parser.add_argument("--s16s4-expert-root", default="experts_s16s4")
    parser.add_argument(
        "--s16s4-gate-up-omc",
        default="omc/expert_s16s4/qwen35_moe_s16s4_gate_up_s{sequence}.omc",
    )
    parser.add_argument(
        "--s16s4-down-omc",
        default="omc/expert_s16s4/qwen35_moe_s16s4_down_s{sequence}.omc",
    )
    parser.add_argument("--cache-shards", type=int, default=16)
    return parser.parse_args()


def classify_checkpoint(metadata, num_layers, num_experts):
    """Classify resident tensors and validate every expert projection."""
    expected_suffixes = {
        (projection, suffix)
        for projection in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("qweight", "qzeros", "scales", "g_idx")
    }
    expert_members: dict[tuple[int, int], set[tuple[str, str]]] = defaultdict(set)
    expert_projection_bytes: Counter[str] = Counter()
    expert_tensor_bytes = 0
    resident: dict[str, list[dict]] = defaultdict(list)
    resident_bytes: Counter[str] = Counter()

    for name, info in metadata.items():
        expert = EXPERT_RE.match(name)
        if expert:
            layer, expert_id = int(expert.group(1)), int(expert.group(2))
            projection, suffix = expert.group(3), expert.group(4)
            expert_members[(layer, expert_id)].add((projection, suffix))
            expert_projection_bytes[projection] += info["bytes"]
            expert_tensor_bytes += info["bytes"]
            continue
        group = component(name)
        entry = {"name": name, **info}
        resident[group].append(entry)
        resident_bytes[group] += info["bytes"]

    incomplete = []
    for layer in range(num_layers):
        for expert in range(num_experts):
            missing = sorted(expected_suffixes - expert_members[(layer, expert)])
            if missing:
                incomplete.append(
                    {"layer": layer, "expert": expert, "missing": missing}
                )
    if incomplete:
        preview = incomplete[:4]
        raise ValueError(f"checkpoint has incomplete routed experts: {preview}")

    return expert_members, expert_projection_bytes, expert_tensor_bytes, resident, resident_bytes


def layer_manifest(text, resident_bytes, args):
    """Describe resident and expert resources for each layer."""
    layer_types = list(text["layer_types"])
    layers = []
    for layer, layer_type in enumerate(layer_types):
        group = f"layer_{layer:02d}_resident"
        layers.append(
            {
                "index": layer,
                "type": layer_type,
                "resident_omc": (
                    "omc/resident/qwen35_resident_linear_s1_s64/"
                    "qwen35_resident_linear_s1_s64.omc"
                    if layer_type == "linear_attention"
                    else "omc/resident/qwen35_resident_full_s1_s64/"
                    "qwen35_resident_full_s1_s64.omc"
                ),
                "resident_weight_dir": f"weights/layer_{layer:02d}",
                "resident_checkpoint_bytes": resident_bytes[group],
                "expert_pattern": (
                    f"{args.expert_root}/layer_{layer:02d}/expert_{{expert:03d}}.q4"
                ),
                "s16s4_expert_pattern": (
                    f"{args.s16s4_expert_root}/layer_{layer:02d}/"
                    "expert_{expert:03d}.s4"
                ),
            }
        )

    return layers


def main() -> None:
    args = parse_arguments()

    model_dir = args.model_dir.resolve()
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", config)
    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map: dict[str, str] = index["weight_map"]
    metadata = tensor_metadata(model_dir, weight_map)

    num_layers = int(text["num_hidden_layers"])
    num_experts = int(text["num_experts"])
    (expert_members, expert_projection_bytes, expert_tensor_bytes,
     resident, resident_bytes) = classify_checkpoint(metadata, num_layers, num_experts)
    layers = layer_manifest(text, resident_bytes, args)
    components = {}
    for name in sorted(resident):
        entries = sorted(resident[name], key=lambda item: item["name"])
        components[name] = {
            "bytes": resident_bytes[name],
            "tensor_count": len(entries),
            "tensors": entries,
        }

    manifest = {
        "format": "hmos-device-ai-paged-moe",
        "version": 1,
        "model_family": "qwen3.5-moe",
        "model_id": "Qwen3.5-35B-A3B-GPTQ-Int4",
        "runtime": {
            "hidden_size": int(text["hidden_size"]),
            "vocab_size": int(text["vocab_size"]),
            "num_hidden_layers": num_layers,
            "num_experts": num_experts,
            "experts_per_token": int(text["num_experts_per_tok"]),
            "moe_intermediate_size": int(text["moe_intermediate_size"]),
            "cache_length": 8192,
            "gears": [1, 64],
            "expert_gears": [1, 8, 64],
            "expert_cache_shards": args.cache_shards,
            "expert_gate_up_omc": args.expert_gate_up_omc,
            "expert_down_omc": args.expert_down_omc,
            "s16s4_expert_root": args.s16s4_expert_root,
            "s16s4_expert_gate_up_omc": args.s16s4_gate_up_omc,
            "s16s4_expert_down_omc": args.s16s4_down_omc,
            "s16s4_activation_scale": 1024.0,
            "s16s4_dispatch": {"rows": [1, 9, 64], "fallback_rows": [2, 8]},
            "expert_abi": "gate_up_swiglu_then_down_v1",
            "expert_staging_buffers": 2,
        },
        "quantization": {
            "method": "gptq",
            "bits": int(config["quantization_config"]["bits"]),
            "group_size": int(config["quantization_config"]["group_size"]),
            "symmetric": bool(config["quantization_config"]["sym"]),
            "routed_expert_tensor_bytes": expert_tensor_bytes,
            "routed_expert_projection_bytes": dict(expert_projection_bytes),
            "expert_shard_bytes_aligned": fused_expert_shard_bytes(),
            "expert_shard_count": num_layers * num_experts,
        },
        "memory_policy": {
            "resident_layer_window": 1,
            "embedding": "cpu_mmap_row_lookup",
            "lm_head": "npu_vocab_tiles",
            "routed_experts": "mmap_lru_to_reused_nnrt_inputs",
            "prefetch": "next_layer_top_experts",
        },
        "layers": layers,
        "components": components,
        "checkpoint": {
            "metadata_total_size": int(index.get("metadata", {}).get("total_size", 0)),
            "tensor_count": len(weight_map),
            "source_shard_count": len(set(weight_map.values())),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "routed_expert_bytes": expert_tensor_bytes,
                "resident_bytes": sum(resident_bytes.values()),
                "expert_count": len(expert_members),
                "components": {name: value["bytes"] for name, value in components.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
