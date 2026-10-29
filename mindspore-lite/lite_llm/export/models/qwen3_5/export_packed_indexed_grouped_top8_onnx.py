# Adapted from HMOS-Device-AI converter/qwen3.5-35b-a3b; see source_provenance.json.
#!/usr/bin/env python3
"""Export Top-8 S16S4 with one packed resident expert-cache input."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


HIDDEN = 2048
TOP_K = 8
SLOT_BYTES = 1_777_664


def value(name: str, dtype: int, shape: list[int]):
    return helper.make_tensor_value_info(name, dtype, shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-slots", type=int, choices=(16, 32, 64, 128, 384), required=True
    )
    parser.add_argument("--activation-scale", type=float, default=1024.0)
    args = parser.parse_args()

    graph_inputs = [
        value("hidden", TensorProto.FLOAT16, [1, HIDDEN]),
        value("expert_ids", TensorProto.INT32, [TOP_K]),
        value(
            "packed_weights",
            TensorProto.UINT8,
            [args.cache_slots, SLOT_BYTES],
        ),
    ]
    x_scale = onnx.numpy_helper.from_array(
        np.asarray([args.activation_scale], dtype=np.float16), name="x_scale"
    )
    node = helper.make_node(
        "GroupedTop8S16S4PackedIndexed",
        ["hidden", "x_scale", "expert_ids", "packed_weights"],
        ["experts"],
        name="grouped_top8_s16_s4_packed_indexed",
        domain="com.huawei.custom",
    )
    graph = helper.make_graph(
        [node],
        f"qwen3_5_35b_grouped_top8_s16s4_packed{args.cache_slots}",
        graph_inputs,
        [value("experts", TensorProto.FLOAT16, [TOP_K, HIDDEN])],
        initializer=[x_scale],
    )
    model = helper.make_model(
        graph,
        producer_name="HMOS-Device-AI/qwen35-packed-indexed-top8",
        opset_imports=[
            helper.make_opsetid("", 13),
            helper.make_opsetid("com.huawei.custom", 1),
        ],
        ir_version=9,
    )
    model.doc_string = (
        f"Qwen3.5-35B packed Top8 S16S4; {args.cache_slots} DMA slots "
        "selected by eight int32 expert ids"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(
        f"wrote {args.output}: cache_slots={args.cache_slots} "
        f"packed_bytes={args.cache_slots * SLOT_BYTES}"
    )


if __name__ == "__main__":
    main()
