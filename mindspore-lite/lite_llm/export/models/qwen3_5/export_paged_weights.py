# Adapted from HMOS-Device-AI converter/qwen3.5-35b-a3b; see source_provenance.json.
#!/usr/bin/env python3
"""Extract GPTQ MoE experts into independently mmap-able runtime shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from .gptq import (
    PROJECTIONS,
    IndexedSafetensors,
    cube_qweight,
    load_projection,
    quantization,
    text_config,
)


MAGIC = b"Q35GPTQ\0"
VERSION = 1
ALIGNMENT = 4096
DTYPE_CODES = {
    np.dtype("int32"): 1,
    np.dtype("float16"): 2,
    np.dtype("int8"): 3,
}


def align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def parse_range(text: str, limit: int, label: str) -> list[int]:
    """Parse a comma-separated list of indices and inclusive ranges."""
    result: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first_text, last_text = part.split("-", 1)
            first, last = int(first_text), int(last_text)
            if first > last:
                raise ValueError(f"invalid {label} range: {part}")
            result.update(range(first, last + 1))
        else:
            result.add(int(part))
    if not result or min(result) < 0 or max(result) >= limit:
        raise ValueError(f"{label} must be within [0,{limit - 1}]")
    return sorted(result)


def write_expert(
    path: Path,
    tensors: list[tuple[str, np.ndarray]],
    magic: bytes = MAGIC,
) -> dict:
    """Write the gate, up, and down weights for one expert."""
    if len(magic) != 8:
        raise ValueError("expert shard magic must contain exactly 8 bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    header_bytes = align(64 + len(tensors) * 128)
    offset = header_bytes
    for name, value in tensors:
        value = np.ascontiguousarray(value)
        offset = align(offset)
        entries.append(
            {
                "name": name,
                "dtype": value.dtype.name,
                "dtype_code": DTYPE_CODES[value.dtype],
                "shape": list(value.shape),
                "offset": offset,
                "bytes": value.nbytes,
            }
        )
        offset += value.nbytes

    with path.open("wb") as handle:
        handle.write(magic)
        handle.write(struct.pack("<IIIIQ", VERSION, len(entries), ALIGNMENT, 0, header_bytes))
        handle.write(b"\0" * (64 - handle.tell()))
        for entry in entries:
            encoded = entry["name"].encode("ascii")
            if len(encoded) > 63 or len(entry["shape"]) > 4:
                raise ValueError(f"tensor metadata is too large: {entry['name']}")
            shape = entry["shape"] + [0] * (4 - len(entry["shape"]))
            record = struct.pack(
                "<64sII4QQQ",
                encoded,
                entry["dtype_code"],
                len(entry["shape"]),
                *shape,
                entry["offset"],
                entry["bytes"],
            )
            handle.write(record)
            handle.write(b"\0" * (128 - len(record)))
        handle.write(b"\0" * (header_bytes - handle.tell()))
        for entry, (_, value) in zip(entries, tensors):
            handle.write(b"\0" * (entry["offset"] - handle.tell()))
            handle.write(np.ascontiguousarray(value).tobytes(order="C"))
        handle.truncate(align(handle.tell()))

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest, "tensors": entries}


def parse_arguments():
    """Parse and validate command-line options for this exporter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="0", help="comma/range list, for example 0,4-7")
    parser.add_argument("--experts", default="0", help="comma/range list, for example 0,8-15")
    parser.add_argument(
        "--layout",
        choices=("cube", "source"),
        default="cube",
        help="cube stores four-row split-half signed INT4 as [N/4,K/2]",
    )
    parser.add_argument(
        "--fuse-gate-up",
        action="store_true",
        help="Store Gate/Up as one 2048x1024 projection for the accepted stage ABI",
    )
    return parser.parse_args()


def pack_expert(checkpoint, layer, expert, args):
    """Pack one expert, optionally fusing Gate and Up with the Cube layout."""
    packed = []
    projection_values = {}
    for projection in PROJECTIONS:
        projection_tensors = load_projection(
            checkpoint, layer, expert, projection
        )
        projection_values[projection] = projection_tensors
    if args.fuse_gate_up:
        if args.layout != "cube":
            raise ValueError("fused Gate/Up requires the Cube qweight layout")
        gate = projection_values["gate_proj"]
        up = projection_values["up_proj"]
        if not np.array_equal(gate["g_idx"], up["g_idx"]):
            raise ValueError("Gate and Up g_idx tensors differ")
        fused = {
            "qweight": np.concatenate(
                (cube_qweight(gate), cube_qweight(up)), axis=0
            ),
            "qzeros": np.concatenate(
                (gate["qzeros"], up["qzeros"]), axis=1
            ),
            "scales": np.concatenate(
                (gate["scales"], up["scales"]), axis=1
            ),
            "g_idx": gate["g_idx"],
        }
        for suffix, value in fused.items():
            packed.append((f"gate_up.{suffix}", value))
        projections_to_store = ("down_proj",)
    else:
        projections_to_store = PROJECTIONS
    for projection in projections_to_store:
        projection_tensors = projection_values[projection]
        if args.layout == "cube":
            projection_tensors = dict(projection_tensors)
            projection_tensors["qweight"] = cube_qweight(
                projection_tensors
            )
        for suffix, value in projection_tensors.items():
            packed.append((f"{projection}.{suffix}", value))
    return packed


def main() -> None:
    args = parse_arguments()

    config = text_config(args.model_dir)
    qconfig = quantization(args.model_dir)
    layers = parse_range(args.layers, int(config["num_hidden_layers"]), "layer")
    experts = parse_range(args.experts, int(config["num_experts"]), "expert")
    checkpoint = IndexedSafetensors(args.model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "qwen3.5-moe-gptq-paged-v1",
        "source_model": str(args.model_dir.resolve()),
        "model": {
            key: int(config[key])
            for key in (
                "hidden_size",
                "moe_intermediate_size",
                "num_hidden_layers",
                "num_experts",
                "num_experts_per_tok",
                "vocab_size",
            )
        },
        "quantization": {
            "bits": qconfig.bits,
            "group_size": qconfig.group_size,
            "symmetric": qconfig.symmetric,
            "zero_point_encoding": "direct",
            "qweight_layout": (
                "N4_K2_INT32_SIGNED_NIBBLE_EVEN_ODD_INTERLEAVED4"
                if args.layout == "cube"
                else "K8_N_INT32_LITTLE_NIBBLE"
            ),
            "expert_abi": (
                "gate_up_swiglu_then_down_v1"
                if args.fuse_gate_up else "three_projection_v1"
            ),
        },
        "alignment": ALIGNMENT,
        "layers": {},
    }
    total = 0
    for layer in layers:
        layer_items = {}
        for expert in experts:
            packed = pack_expert(checkpoint, layer, expert, args)
            relative = Path(f"layer_{layer:02d}") / f"expert_{expert:03d}.q4"
            item = write_expert(args.output_dir / relative, packed)
            item["file"] = relative.as_posix()
            layer_items[str(expert)] = item
            total += item["bytes"]
            print(f"wrote layer={layer} expert={expert}: {item['bytes'] / 1048576:.3f} MiB")
        manifest["layers"][str(layer)] = {"experts": layer_items}
    manifest["selected_bytes"] = total
    manifest["selection"] = {"layers": layers, "experts": experts}
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")
    print(f"selected expert storage: {total / 1048576:.3f} MiB")


if __name__ == "__main__":
    main()
