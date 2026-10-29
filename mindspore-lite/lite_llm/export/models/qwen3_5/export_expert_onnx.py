# Adapted from HMOS-Device-AI converter/qwen3.5-35b-a3b; see source_provenance.json.
#!/usr/bin/env python3
"""Create a swappable-weight single Qwen3.5 expert custom-op ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, checker, helper


def value(name: str, dtype: int, shape: list[int]):
    return helper.make_tensor_value_info(name, dtype, shape)


def projection_inputs(prefix: str, k: int, n: int, layout: str) -> list:
    qweight_shape = [n // 4, k // 2] if layout == "cube" else [k // 8, n]
    return [
        value(f"{prefix}_qweight", TensorProto.INT32, qweight_shape),
        value(f"{prefix}_qzeros", TensorProto.INT32, [k // 128, n // 8]),
        value(f"{prefix}_scales", TensorProto.FLOAT16, [k // 128, n]),
        value(f"{prefix}_g_idx", TensorProto.INT32, [k]),
    ]


def gptq_node(prefix: str, x: str, y: str, layout: str = "source"):
    return helper.make_node(
        "GptqMatMul",
        [
            x,
            f"{prefix}_qweight",
            f"{prefix}_qzeros",
            f"{prefix}_scales",
            f"{prefix}_g_idx",
        ],
        [y],
        name=f"{prefix}_gptq_matmul",
        domain="com.huawei.custom",
        weight_layout=1 if layout == "cube" else 0,
    )


def parse_arguments():
    """Parse and validate command-line options for this exporter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, choices=(1, 8, 64), default=1)
    parser.add_argument("--layout", choices=("cube", "source"), default="source")
    parser.add_argument(
        "--expose-intermediates",
        action="store_true",
        help="Expose projection and SwiGLU intermediate tensors as graph outputs",
    )
    parser.add_argument(
        "--fuse-gate-up",
        action="store_true",
        help="Use one 2048x1024 GPTQ projection and split Gate/Up outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    m, hidden, intermediate = args.tokens, 2048, 512
    if args.fuse_gate_up and args.layout != "cube":
        raise ValueError("fused Gate/Up currently requires the Cube qweight layout")
    inputs = [value("hidden", TensorProto.FLOAT16, [m, hidden])]
    inputs += projection_inputs(
        "gate_up" if args.fuse_gate_up else "gate",
        hidden,
        intermediate * 2 if args.fuse_gate_up else intermediate,
        args.layout,
    )
    if not args.fuse_gate_up:
        inputs += projection_inputs("up", hidden, intermediate, args.layout)
    inputs += projection_inputs("down", intermediate, hidden, args.layout)
    initializers = []
    if args.fuse_gate_up:
        projection_nodes = [
            gptq_node("gate_up", "hidden", "gate_up_linear", args.layout),
            helper.make_node(
                "GateUpSwiGLU",
                ["gate_up_linear"],
                ["gated"],
                name="gate_up_swiglu",
                domain="com.huawei.custom",
            ),
        ]
    else:
        projection_nodes = [
            gptq_node("gate", "hidden", "gate_linear", args.layout),
            gptq_node("up", "hidden", "up_linear", args.layout),
        ]
    activation_nodes = [] if args.fuse_gate_up else [
        helper.make_node("Sigmoid", ["gate_linear"], ["gate_sigmoid"], name="silu_sigmoid"),
        helper.make_node("Mul", ["gate_linear", "gate_sigmoid"], ["gate_silu"], name="silu_mul"),
        helper.make_node("Mul", ["gate_silu", "up_linear"], ["gated"], name="gated_mul"),
    ]
    nodes = projection_nodes + activation_nodes + [
        gptq_node("down", "gated", "output", args.layout),
    ]
    graph_outputs = [value("output", TensorProto.FLOAT16, [m, hidden])]
    if args.expose_intermediates:
        if args.fuse_gate_up:
            graph_outputs = [
                value("gate_up_linear", TensorProto.FLOAT16, [m, intermediate * 2]),
                value("gated", TensorProto.FLOAT16, [m, intermediate]),
            ] + graph_outputs
        else:
            graph_outputs = [
                value(name, TensorProto.FLOAT16, [m, intermediate])
                for name in (
                    "gate_linear", "up_linear", "gate_sigmoid",
                    "gate_silu", "gated",
                )
            ] + graph_outputs
    graph = helper.make_graph(
        nodes,
        "qwen3_5_35b_a3b_single_expert",
        inputs,
        graph_outputs,
        initializer=initializers,
        value_info=[
            *(
                [value("gate_up_linear", TensorProto.FLOAT16, [m, intermediate * 2])]
                if args.fuse_gate_up else []
            ),
            *(
                [] if args.fuse_gate_up else [
                    value("gate_linear", TensorProto.FLOAT16, [m, intermediate]),
                    value("up_linear", TensorProto.FLOAT16, [m, intermediate]),
                    value("gate_sigmoid", TensorProto.FLOAT16, [m, intermediate]),
                    value("gate_silu", TensorProto.FLOAT16, [m, intermediate]),
                ]
            ),
            value("gated", TensorProto.FLOAT16, [m, intermediate]),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.huawei.custom", 1),
        ],
        producer_name="HMOS-Device-AI",
    )
    model.ir_version = 9
    model.doc_string = (
        "Qwen3.5 MoE single expert with swappable GPTQ weights; "
        f"layout={args.layout}; fused_gate_up={args.fuse_gate_up}."
    )
    checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"wrote {args.output}: inputs={len(inputs)} nodes={len(nodes)}")


if __name__ == "__main__":
    main()
