# Adapted from HMOS-Device-AI converter/qwen3.5-35b-a3b; see source_provenance.json.
#!/usr/bin/env python3
"""Export swappable-weight S16S4 Gate/Up or Down expert stages."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def value(name: str, dtype: int, shape: list[int]):
    return helper.make_tensor_value_info(name, dtype, shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("gate-up", "down"), required=True)
    parser.add_argument("--tokens", type=int, choices=(1, 64), required=True)
    parser.add_argument("--activation-scale", type=float)
    parser.add_argument("--expose-linear", action="store_true")
    args = parser.parse_args()

    m = args.tokens
    k, n = (2048, 1024) if args.stage == "gate-up" else (512, 2048)
    activation = "hidden" if args.stage == "gate-up" else "gated"
    output = "gate_up_linear" if args.stage == "gate-up" else "output"
    default_scale = 1024.0
    activation_scale = args.activation_scale or default_scale
    scale = helper.make_tensor(
        "x_scale", TensorProto.FLOAT16, [1],
        np.asarray([activation_scale], dtype=np.float16).tobytes(), raw=True
    )
    inputs = [
        value(activation, TensorProto.FLOAT16, [m, k]),
        value("w", TensorProto.INT8, [n, k // 2]),
        value("bias", TensorProto.FLOAT16, [n]),
        value("w_scale", TensorProto.INT8, [n // 128, k // 128, 128, 8]),
    ]
    nodes = [
        helper.make_node(
            "MatmulS16S4",
            [activation, "x_scale", "w", "bias", "w_scale"],
            [output],
            name="MatmulS16S4",
            domain="ai.onnx",
            w_shape=[k, n],
            group_size=128,
        )
    ]
    outputs = [value(output, TensorProto.FLOAT16, [m, n])]
    value_info = []
    if args.stage == "gate-up":
        nodes.append(
            helper.make_node(
                "GateUpSwiGLU", [output], ["gated"],
                name="gate_up_swiglu", domain="com.huawei.custom"
            )
        )
        outputs = [value("gated", TensorProto.FLOAT16, [m, 512])]
        value_info = [value(output, TensorProto.FLOAT16, [m, n])]
        if args.expose_linear:
            outputs.insert(0, value(output, TensorProto.FLOAT16, [m, n]))

    graph = helper.make_graph(
        nodes,
        f"qwen3_5_35b_s16s4_{args.stage.replace('-', '_')}_s{m}",
        inputs,
        outputs,
        initializer=[scale],
        value_info=value_info,
    )
    model = helper.make_model(
        graph,
        producer_name="HMOS-Device-AI",
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("ai.onnx", 1),
            helper.make_opsetid("com.huawei.custom", 1),
        ],
    )
    model.ir_version = 9
    model.doc_string = (
        f"Qwen3.5-35B paged S16S4 {args.stage}; activation_scale={activation_scale}."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"wrote {args.output}: M={m} K={k} N={n} scale={activation_scale}")


if __name__ == "__main__":
    main()
