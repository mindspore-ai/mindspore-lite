# Adapted from HMOS-Device-AI converter/qwen3.5-35b-a3b; see source_provenance.json.
#!/usr/bin/env python3
"""Export Qwen3.5 GPTQ routed experts as mmap-able S16S4 shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .export_paged_weights import ALIGNMENT, parse_range, write_expert
from .gptq import IndexedSafetensors, load_projection, quantization, text_config
from .s16s4 import materialize_projection


MAGIC = b"Q35S16\0\0"
ACTIVATION_SCALE = 1024.0


def write_checksums(manifest_path: Path) -> Path:
    """Write resource sizes and SHA-256 digests to the manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lines = []
    for layer in manifest["layers"].values():
        for expert in layer["experts"].values():
            lines.append(f"{expert['sha256']}  {expert['file']}")
    checksums = manifest_path.with_name("SHA256SUMS")
    checksums.write_text("\n".join(sorted(lines)) + "\n", encoding="ascii")
    manifest_path.with_name("READY").write_text(
        f"{manifest['format']}\nshards={len(lines)}\n", encoding="ascii"
    )
    return checksums


def combine_gate_up(
    gate: dict[str, np.ndarray], up: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        "w": np.ascontiguousarray(np.concatenate((gate["w"], up["w"]), axis=0)),
        "bias": np.ascontiguousarray(
            np.concatenate((gate["bias"], up["bias"]), axis=0)
        ),
        "w_scale": np.ascontiguousarray(
            np.concatenate((gate["w_scale"], up["w_scale"]), axis=0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="0", help="comma/range list")
    parser.add_argument("--experts", default="0", help="comma/range list")
    parser.add_argument("--activation-scale", type=float, default=ACTIVATION_SCALE)
    args = parser.parse_args()

    config = text_config(args.model_dir)
    qconfig = quantization(args.model_dir)
    if qconfig.bits != 4 or qconfig.group_size != 128 or not qconfig.symmetric:
        raise ValueError(
            "MatmulS16S4 requires symmetric GPTQ INT4 with group_size=128"
        )
    layers = parse_range(args.layers, int(config["num_hidden_layers"]), "layer")
    experts = parse_range(args.experts, int(config["num_experts"]), "expert")
    checkpoint = IndexedSafetensors(args.model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "format": "qwen3.5-moe-s16s4-paged-v1",
        "source_model": str(args.model_dir.resolve()),
        "model": {
            key: int(config[key])
            for key in (
                "hidden_size",
                "moe_intermediate_size",
                "num_hidden_layers",
                "num_experts",
                "num_experts_per_tok",
            )
        },
        "quantization": {
            "activation": "s16_symmetric_per_tensor",
            "activation_scale": args.activation_scale,
            "weight": "s4_symmetric_groupwise",
            "group_size": 128,
            "weight_layout": "L311_FRACTAL_NZ_S4",
            "scale_layout": "L311_FIXPIPE_FP32_RECORD",
        },
        "alignment": ALIGNMENT,
        "layers": {},
    }
    total = 0
    layer_manifest: dict[str, object] = {}
    for layer in layers:
        expert_manifest: dict[str, object] = {}
        for expert in experts:
            gate = materialize_projection(
                load_projection(checkpoint, layer, expert, "gate_proj"),
                args.activation_scale,
            )
            up = materialize_projection(
                load_projection(checkpoint, layer, expert, "up_proj"),
                args.activation_scale,
            )
            gate_up = combine_gate_up(gate, up)
            down = materialize_projection(
                load_projection(checkpoint, layer, expert, "down_proj"),
                args.activation_scale,
            )
            tensors = [
                (f"gate_up.{name}", gate_up[name])
                for name in ("w", "bias", "w_scale")
            ] + [
                (f"down.{name}", down[name])
                for name in ("w", "bias", "w_scale")
            ]
            relative = Path(f"layer_{layer:02d}") / f"expert_{expert:03d}.s4"
            item = write_expert(args.output_dir / relative, tensors, magic=MAGIC)
            item["file"] = relative.as_posix()
            expert_manifest[str(expert)] = item
            total += int(item["bytes"])
            print(
                f"wrote layer={layer} expert={expert}: "
                f"{int(item['bytes']) / 1048576:.3f} MiB",
                flush=True,
            )
        layer_manifest[str(layer)] = {"experts": expert_manifest}
    manifest["layers"] = layer_manifest
    manifest["selected_bytes"] = total
    manifest["selection"] = {"layers": layers, "experts": experts}
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksums = write_checksums(manifest_path)
    print(f"manifest: {manifest_path}")
    print(f"checksums: {checksums}")
    print(f"selected expert storage: {total / 1048576:.3f} MiB")


if __name__ == "__main__":
    main()
