# Adapted from HMOS-Device-AI converter/qwen3.5-35b-a3b; see source_provenance.json.
#!/usr/bin/env python3
"""Export final RMSNorm and tiled LM Head custom-op ONNX graphs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HIDDEN = 2048
VOCAB = 248320


def value(name: str, shape: list[int]):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--kind", choices=("norm", "lm-head", "top1", "resident"), required=True
    )
    parser.add_argument(
        "--tile-rows", type=int,
        choices=(2560, 8192, 16384, 31040, 62080), default=8192
    )
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()
    initializers = []
    if args.kind == "norm":
        graph = helper.make_graph(
            [helper.make_node("RMSNorm", ["hidden", "weight"], ["normalized"])],
            "qwen35_final_norm",
            [value("hidden", [1, 2048]), value("weight", [2048])],
            [value("normalized", [1, 2048])],
        )
    elif args.kind == "lm-head":
        graph = helper.make_graph(
            [helper.make_node("MatMulFp16Nz", ["hidden", "weight"], ["logits"])],
            f"qwen35_lm_head_{args.tile_rows}",
            [value("hidden", [1, 2048]), value("weight", [args.tile_rows, 2048])],
            [value("logits", [1, args.tile_rows])],
        )
    elif args.kind == "top1":
        graph = helper.make_graph(
            [helper.make_node(
                "MatMulFp16NzTop1", ["hidden", "weight"], ["token_ids"]
            )],
            "qwen35_lm_head_top1",
            [value("hidden", [1, HIDDEN]), value("weight", [VOCAB, HIDDEN])],
            [helper.make_tensor_value_info(
                "token_ids", TensorProto.INT32, [1]
            )],
        )
    else:
        if args.artifact_root is None:
            parser.error("--artifact-root is required for --kind resident")
        norm_path = args.artifact_root / "global/final_norm_fp16.bin"
        weight_path = args.artifact_root / "global/lm_head_nz_fp16.bin"
        expected_bytes = VOCAB * HIDDEN * 2
        if norm_path.stat().st_size != HIDDEN * 2:
            raise ValueError(f"invalid final norm file: {norm_path}")
        if weight_path.stat().st_size != expected_bytes:
            raise ValueError(f"invalid LM Head weight file: {weight_path}")
        norm = numpy_helper.from_array(
            np.fromfile(norm_path, dtype=np.float16), name="norm_weight"
        )
        weight = TensorProto()
        weight.name = "weight"
        weight.data_type = TensorProto.FLOAT16
        weight.dims.extend([VOCAB, HIDDEN])
        weight.data_location = TensorProto.EXTERNAL
        relative_weight = os.path.relpath(weight_path, args.output.parent)
        for key, item in (
            ("location", relative_weight),
            ("offset", "0"),
            ("length", str(expected_bytes)),
        ):
            entry = weight.external_data.add()
            entry.key = key
            entry.value = item
        initializers = [norm, weight]
        graph = helper.make_graph(
            [
                helper.make_node(
                    "RMSNorm", ["hidden", "norm_weight"], ["normalized"]
                ),
                helper.make_node(
                    "MatMulFp16Nz", ["normalized", "weight"], ["logits"]
                ),
            ],
            "qwen35_resident_lm_head",
            [value("hidden", [1, HIDDEN])],
            [value("logits", [1, VOCAB])],
            initializer=initializers,
        )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="HMOS-Device-AI",
    )
    model.ir_version = 9
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
