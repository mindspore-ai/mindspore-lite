#!/usr/bin/env python3
"""Align an OMG enum-shape gear table with the public graph input order."""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")




def input_shapes(path: Path) -> dict[str, list[int]]:
    report = json.loads(path.read_text())
    if report.get("status") != "PASS":
        raise ValueError(f"ABI report is not accepted: {path}")
    return {item["name"]: item["shape"] for item in report["inputs"]}


def flatten_gear(names: list[str], shapes: dict[str, list[int]]) -> list[int]:
    missing = [name for name in names if name not in shapes]
    if missing:
        raise ValueError(f"ABI report is missing graph inputs: {missing}")
    return [dim for name in names for dim in shapes[name]]


def onnx_input_shapes(path: Path, sequence: int, lmhead_count: int) -> dict[str, list[int]]:
    """Resolve public ONNX input shapes for one sequence gear."""
    import onnx
    model = onnx.load(path, load_external_data=False)
    shapes: dict[str, list[int]] = {}
    for value in model.graph.input:
        dims: list[int] = []
        for dim in value.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(dim.dim_value)
            elif dim.dim_param == "seq_len":
                dims.append(sequence)
            elif dim.dim_param == "lmhead_len":
                dims.append(lmhead_count)
            else:
                raise ValueError(
                    f"unsupported symbolic dimension {dim.dim_param!r} "
                    f"in ONNX input {value.name}"
                )
        shapes[value.name] = dims
    return shapes


def parse_arguments():
    """Parse and validate command-line options for this exporter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_omc", type=Path)
    parser.add_argument("output_omc", type=Path)
    parser.add_argument(
        "--onnx", type=Path,
        help="dynamic ONNX used to derive public gear shapes",
    )
    parser.add_argument("--gear-sequences", default="1,64")
    parser.add_argument("--lmhead-counts", default="1,1")
    parser.add_argument("--decode-abi", type=Path)
    parser.add_argument("--prefill64-abi", type=Path)
    parser.add_argument(
        "--ge-ir-module-dir", type=Path, required=True,
    )
    return parser, parser.parse_args()


def resolve_gears(args, parser, names):
    """Resolve public gear shapes from ONNX or accepted ABI reports."""
    if args.onnx is not None:
        if args.decode_abi is not None or args.prefill64_abi is not None:
            parser.error("--onnx cannot be combined with ABI report options")
        shape_source = str(args.onnx.resolve())
        sequences = [int(item) for item in args.gear_sequences.split(",") if item]
        lmhead_counts = [int(item) for item in args.lmhead_counts.split(",") if item]
        if len(sequences) != len(lmhead_counts) or not sequences:
            parser.error("--gear-sequences and --lmhead-counts must have equal nonzero length")
        gears = [
            flatten_gear(names, onnx_input_shapes(args.onnx, sequence, lmhead_count))
            for sequence, lmhead_count in zip(sequences, lmhead_counts)
        ]
    else:
        if args.decode_abi is None or args.prefill64_abi is None:
            parser.error("use --onnx or provide both --decode-abi and --prefill64-abi")
        shape_source = (
            f"{args.decode_abi.resolve()};{args.prefill64_abi.resolve()}"
        )
        gears = [
            flatten_gear(names, input_shapes(args.decode_abi)),
            flatten_gear(names, input_shapes(args.prefill64_abi)),
        ]
    if len(gears[0]) != len(gears[1]):
        raise ValueError("gear shape-vector lengths differ")

    return gears, shape_source


def main() -> None:
    from .omc_container import HEADER_SIZE, PARTITION_MODEL_DEF, load_ge_ir, read_model

    parser, args = parse_arguments()

    ge_ir_pb2 = load_ge_ir(args.ge_ir_module_dir)
    model, partitions, _ = read_model(args.input_omc, ge_ir_pb2)
    if len(model.graph) != 1:
        raise ValueError(f"expected one graph, got {len(model.graph)}")
    graph = model.graph[0]
    names = [name.removesuffix(":0") for name in graph.input]
    names.remove("enum_shape_graph/data_shape")

    gears, shape_source = resolve_gears(args, parser, names)
    constants = [op for op in graph.op if op.name == "enum_shape_graph/const_shape"]
    if len(constants) != 1:
        raise ValueError(f"expected one enum shape constant, got {len(constants)}")
    value_attrs = [item for item in constants[0].attr if item.key == "value"]
    if len(value_attrs) != 1:
        raise ValueError("enum shape constant has no unique value attribute")
    old_data = bytes(value_attrs[0].value.t.data)
    old_values = struct.unpack(f"<{len(old_data) // 4}i", old_data)
    gear_width = len(gears[0])
    if len(old_values) != gear_width * len(gears):
        raise ValueError(
            f"gear table has {len(old_values)} values, expected {gear_width * len(gears)}"
        )
    new_values = [dim for gear in gears for dim in gear]
    new_data = struct.pack(f"<{len(new_values)}i", *new_values)

    raw = bytearray(args.input_omc.read_bytes())
    model_parts = [item for item in partitions if item[0] == PARTITION_MODEL_DEF]
    _, model_offset, model_size = model_parts[0]
    partition_base = HEADER_SIZE + 4 + 12 * len(partitions)
    begin = partition_base + model_offset
    end = begin + model_size
    position = raw.find(old_data, begin, end)
    if position < 0 or raw.find(old_data, position + 1, end) >= 0:
        raise ValueError("enum shape table is not unique in MODEL_DEF")
    raw[position:position + len(old_data)] = new_data

    args.output_omc.parent.mkdir(parents=True, exist_ok=True)
    args.output_omc.write_bytes(raw)
    report = {
        "input_omc": str(args.input_omc.resolve()),
        "output_omc": str(args.output_omc.resolve()),
        "graph_input_count": len(names),
        "gear_count": len(gears),
        "gear_width": gear_width,
        "shape_source": shape_source,
        "graph_input_order": names,
        "old_first_gear_prefix": list(old_values[:20]),
        "new_first_gear_prefix": new_values[:20],
    }
    args.output_omc.with_suffix(".gear_fix.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
