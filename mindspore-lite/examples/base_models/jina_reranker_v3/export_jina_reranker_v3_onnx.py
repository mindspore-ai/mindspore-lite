#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Export Jina Reranker V3 to ONNX format.
This script exports a single unified ONNX model for inference.
"""

import argparse
import os
import torch
import onnx
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class JinaRerankerV3(torch.nn.Module):
    """
    Jina Reranker V3 wrapper for ONNX export.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for reranking.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        logits = outputs[0]
        return logits


def _parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Export Jina Reranker V3 to ONNX")
    parser.add_argument(
        "--model-id",
        type=str,
        default="jinaai/jina-reranker-v3",
        help="Model ID or path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./onnx",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--max-length", type=int, default=8192, help="Maximum sequence length"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (cpu/cuda)"
    )
    return parser.parse_args()


def _load_model_and_tokenizer(args):
    """
    Load model and tokenizer from Hugging Face.
    """
    print(f"Loading model from {args.model_id}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    return model, tokenizer


def _prepare_model_for_export(model, device):
    """
    Prepare model for unified ONNX export.
    """
    reranker = JinaRerankerV3(model).to(device).eval()
    return reranker


def _create_dummy_inputs(args, tokenizer):
    """
    Create dummy inputs for ONNX export.
    """
    dummy_text = "Query: What is the capital of China? Document: The capital of China is Beijing."
    dummy_inputs = tokenizer(
        dummy_text,
        padding=False,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    dummy_input_ids = dummy_inputs["input_ids"]
    dummy_attention_mask = dummy_inputs["attention_mask"]

    return dummy_input_ids, dummy_attention_mask


def _export_to_onnx(model, output_path, dummy_input_ids, dummy_attention_mask):
    """
    Export model to ONNX format.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size"},
    }
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dummy_inputs = (dummy_input_ids, dummy_attention_mask)

    print(f"Exporting model to {output_path}")
    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"Model exported successfully to {output_path}")

    print("Optimizing ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx_model = _optimize_onnx_model(onnx_model)
    onnx.save(onnx_model, output_path)
    print(f"Optimized model saved to {output_path}")


def _optimize_onnx_model(onnx_model):
    """
    Optimize ONNX model by removing unnecessary nodes.
    """
    isnan_nodes = [node for node in onnx_model.graph.node if node.op_type == "IsNaN"]

    if isnan_nodes:
        print(
            f"Found {len(isnan_nodes)} IsNaN nodes, replacing Where(IsNaN(x), default, x) patterns with Identity(x)..."
        )

        nodes_to_remove_names = set()
        nodes_to_add = []

        for isnan_node in isnan_nodes:
            isnan_output = isnan_node.output[0]

            where_nodes = [
                node
                for node in onnx_model.graph.node
                if node.op_type == "Where" and isnan_output in node.input
            ]

            print(
                f"  IsNaN node {isnan_node.name}: found {len(where_nodes)} Where consumers"
            )

            for where_node in where_nodes:
                where_inputs = where_node.input
                if len(where_inputs) == 3 and where_inputs[0] == isnan_output:
                    where_output = where_node.output[0]
                    value_if_false = where_inputs[2]

                    identity_node = onnx.helper.make_node(
                        "Identity",
                        inputs=[value_if_false],
                        outputs=[where_output],
                        name=where_node.name + "_identity",
                    )

                    nodes_to_add.append(identity_node)
                    nodes_to_remove_names.add(where_node.name)
                    print(
                        f"    Will replace Where node {where_node.name} with Identity"
                    )

            nodes_to_remove_names.add(isnan_node.name)
            print(f"  Will remove IsNaN node {isnan_node.name}")

        print(
            f"Removing {len(nodes_to_remove_names)} nodes and adding {len(nodes_to_add)} nodes..."
        )

        new_nodes = []
        for node in onnx_model.graph.node:
            if node.name not in nodes_to_remove_names:
                new_nodes.append(node)

        new_nodes.extend(nodes_to_add)

        onnx_model.graph.ClearField("node")
        onnx_model.graph.node.extend(new_nodes)

        print(f"Successfully replaced {len(isnan_nodes)} IsNaN nodes")

        remaining_isnan = sum(
            1 for node in onnx_model.graph.node if node.op_type == "IsNaN"
        )
        print(f"Remaining IsNaN nodes after removal: {remaining_isnan}")
    else:
        print("No IsNaN nodes found in the model")

    # onnx.checker.check_model(onnx_model)
    return onnx_model


def main():
    """
    Main function to export Jina Reranker V3 model to ONNX format.
    """
    args = _parse_args()
    model, tokenizer = _load_model_and_tokenizer(args)
    dummy_input_ids, dummy_attention_mask = _create_dummy_inputs(args, tokenizer)

    print("Exporting unified model...")
    reranker = _prepare_model_for_export(model, args.device)

    output_path = os.path.join(args.output_dir, "jina_reranker_v3.onnx")
    _export_to_onnx(reranker, output_path, dummy_input_ids, dummy_attention_mask)

    print("\nExport completed successfully!")
    print(f"ONNX model saved to: {output_path}")

    print(f"Max sequence length: {args.max_length}")


if __name__ == "__main__":
    main()
