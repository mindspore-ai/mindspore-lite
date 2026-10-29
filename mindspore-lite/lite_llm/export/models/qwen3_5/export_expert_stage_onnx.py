# Adapted from HMOS-Device-AI converter/qwen3.5-35b-a3b; see source_provenance.json.
#!/usr/bin/env python3
"""Export the accepted two-stage paged Qwen3.5 MoE Expert graphs."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, checker, helper

from .export_expert_onnx import gptq_node, projection_inputs, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("gate-up", "down"), required=True)
    parser.add_argument("--tokens", type=int, choices=(1, 8, 64), default=1)
    parser.add_argument("--expose-linear", action="store_true")
    args = parser.parse_args()

    m, hidden, intermediate = args.tokens, 2048, 512
    if args.stage == "gate-up":
        inputs = [value("hidden", TensorProto.FLOAT16, [m, hidden])]
        inputs += projection_inputs("gate_up", hidden, intermediate * 2, "cube")
        nodes = [
            gptq_node("gate_up", "hidden", "gate_up_linear", "cube"),
            helper.make_node(
                "GateUpSwiGLU",
                ["gate_up_linear"],
                ["gated"],
                name="gate_up_swiglu",
                domain="com.huawei.custom",
            ),
        ]
        outputs = [value("gated", TensorProto.FLOAT16, [m, intermediate])]
        if args.expose_linear:
            outputs.insert(
                0,
                value("gate_up_linear", TensorProto.FLOAT16, [m, intermediate * 2]),
            )
        value_info = [
            value("gate_up_linear", TensorProto.FLOAT16, [m, intermediate * 2])
        ]
    else:
        inputs = [value("gated", TensorProto.FLOAT16, [m, intermediate])]
        inputs += projection_inputs("down", intermediate, hidden, "cube")
        nodes = [gptq_node("down", "gated", "output", "cube")]
        outputs = [value("output", TensorProto.FLOAT16, [m, hidden])]
        value_info = []

    graph = helper.make_graph(
        nodes,
        f"qwen3_5_35b_expert_{args.stage.replace('-', '_')}",
        inputs,
        outputs,
        value_info=value_info,
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
    model.doc_string = f"Paged Qwen3.5 MoE Expert stage: {args.stage}."
    checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"wrote {args.output}: stage={args.stage} inputs={len(inputs)}")


if __name__ == "__main__":
    main()
