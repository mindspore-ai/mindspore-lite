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
Export Qwen3-Reranker model to ONNX format.
"""

import argparse
import os
from pathlib import Path
import torch
import onnx
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen3Reranker4B(torch.nn.Module):
    """
    Qwen3-Reranker model for ONNX export.
    """

    def __init__(self, model):
        """
        Initialize the Qwen3-Reranker model.
        """
        super().__init__()
        self.model = model.model
        self.lm_head = model.lm_head

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the Qwen3-Reranker model.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return logits


def _parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Export Qwen3-Reranker to ONNX")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-Reranker-4B",
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
    Load model and tokenizer from the specified model ID.
    """
    print(f"Loading model from {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
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
    Prepare model for ONNX export.
    """
    reranker = Qwen3Reranker4B(model).to(device).eval()
    return reranker


def _create_dummy_inputs(args, tokenizer):
    """
    Create dummy inputs for ONNX model.
    """
    prefix = (
        "<|im_start|>system\nJudge whether the Document meets "
        "the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    dummy_text = "<Instruct>: test\n<Query>: test\n<Document>: test"
    dummy_ids = tokenizer.encode(dummy_text, add_special_tokens=False)
    dummy_ids = prefix_tokens + dummy_ids + suffix_tokens

    if len(dummy_ids) > args.max_length:
        dummy_ids = dummy_ids[: args.max_length]

    dummy_input_ids = torch.tensor([dummy_ids], dtype=torch.long)
    dummy_attention_mask = torch.ones_like(dummy_input_ids)

    return dummy_input_ids, dummy_attention_mask


def _export_to_onnx(model, output_path, dummy_input_ids, dummy_attention_mask):
    """
    Export model to ONNX format with external data for large models.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size", 1: "sequence"},
    }

    print(f"Exporting model to {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=False,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    print(f"Model exported successfully to {output_path}")

    # Check if external data needs consolidation
    onnx_size = os.path.getsize(output_path)
    if onnx_size > 100 * 1024 * 1024:
        print(
            f"ONNX file is {onnx_size / 1024 / 1024:.1f}MB, "
            "consolidating external data..."
        )
        _consolidate_onnx_external_data(output_path)
    else:
        print(f"ONNX file size: {onnx_size / 1024 / 1024:.1f}MB")

    print("Removing IsNaN operators from ONNX model...")
    onnx_model = onnx.load(output_path, load_external_data=False)
    onnx_model = _remove_isnan_nodes(onnx_model)
    onnx.save(onnx_model, output_path)
    print(f"Optimized model saved to {output_path}")


def _remove_isnan_nodes(onnx_model):
    """
    Remove IsNaN nodes from ONNX model.
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
    return onnx_model


def _consolidate_onnx_external_data(onnx_path):
    """
    Consolidate ONNX external data into a single .data file.
    """
    from onnx.external_data_helper import convert_model_to_external_data as _convert

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        return
    print(f"Consolidating external data for {onnx_path.name}...")
    model = onnx.load_model(str(onnx_path), load_external_data=True)
    data_name = onnx_path.stem + ".data"
    _convert(
        model,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=0,
    )
    for f in onnx_path.parent.iterdir():
        if f.name.startswith("onnx__") or (
            f.name.startswith("model.")
            and f.suffix not in (".onnx", ".data", ".mindir")
        ):
            f.unlink(missing_ok=True)
    onnx.save_model(model, str(onnx_path))
    print(f"External data consolidated into {data_name}")


def main():
    """
    Main function.
    """
    args = _parse_args()
    model, tokenizer = _load_model_and_tokenizer(args)
    reranker = _prepare_model_for_export(model, args.device)
    dummy_input_ids, dummy_attention_mask = _create_dummy_inputs(args, tokenizer)

    output_path = os.path.join(args.output_dir, "qwen3_reranker_4b.onnx")
    _export_to_onnx(reranker, output_path, dummy_input_ids, dummy_attention_mask)

    print("\nExport completed successfully!")
    print(f"ONNX model saved to: {output_path}")
    print(f"Max sequence length: {args.max_length}")


if __name__ == "__main__":
    main()
