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
Export bert-base-chinese (BertForMaskedLM) to ONNX format.
"""

import argparse
import gc
import sys
from pathlib import Path

import onnx

try:
    import torch
    import torch._dynamo
    torch._dynamo.disable()
except Exception:
    pass

try:
    from transformers import AutoModelForMaskedLM
except ImportError:
    print("Error: transformers package not found or version too low.")
    print("Please install: pip install transformers")
    sys.exit(1)


class BertOnnxWrapper(torch.nn.Module):
    """Wrapper for BertForMaskedLM to export to ONNX."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.logits


def _ensure_single_file_onnx(onnx_path: Path) -> None:
    """_ensure_single_file_onnx"""
    model = onnx.load(str(onnx_path), load_external_data=True)
    try:
        from onnx.external_data_helper import convert_model_from_external_data

        convert_model_from_external_data(model)
    except Exception:
        pass
    onnx.save_model(model, str(onnx_path), save_as_external_data=False)
    data_path = onnx_path.with_name(onnx_path.name + ".data")
    if data_path.exists():
        data_path.unlink()


def _load_bert_mlm(model_path: str):
    load_kwargs = {
        "torch_dtype": torch.float32,
    }
    try:
        return AutoModelForMaskedLM.from_pretrained(model_path, attn_implementation="eager", **load_kwargs)
    except TypeError:
        return AutoModelForMaskedLM.from_pretrained(model_path, **load_kwargs)


def _export_onnx(model, output_path, device, opset=14):
    """Export BERT model to ONNX."""
    print(f"Exporting BERT model to {output_path}...")

    dummy_input_ids = torch.randint(0, 100, (1, 32), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones(1, 32, dtype=torch.long, device=device)
    dummy_token_type_ids = torch.zeros(1, 32, dtype=torch.long, device=device)

    wrapper = BertOnnxWrapper(model).to(device).eval()

    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "token_type_ids": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            dynamo=False,
            external_data=False,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=False,
            do_constant_folding=False,
            verbose=False,
        )
    _ensure_single_file_onnx(Path(output_path))
    print(f"Export successful: {output_path}")


def _optimize_onnx_model(onnx_model):
    """Optimize ONNX model by removing IsNaN nodes."""
    isnan_nodes = [node for node in onnx_model.graph.node if node.op_type == "IsNaN"]

    if isnan_nodes:
        print(f"Found {len(isnan_nodes)} IsNaN nodes, removing...")

        nodes_to_remove_names = set()
        nodes_to_add = []

        for isnan_node in isnan_nodes:
            isnan_output = isnan_node.output[0]

            where_nodes = [
                node
                for node in onnx_model.graph.node
                if node.op_type == "Where" and isnan_output in node.input
            ]

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

            nodes_to_remove_names.add(isnan_node.name)

        new_nodes = []
        for node in onnx_model.graph.node:
            if node.name not in nodes_to_remove_names:
                new_nodes.append(node)

        new_nodes.extend(nodes_to_add)
        onnx_model.graph.ClearField("node")
        onnx_model.graph.node.extend(new_nodes)

        remaining_isnan = sum(1 for node in onnx_model.graph.node if node.op_type == "IsNaN")
        print(f"Remaining IsNaN nodes after removal: {remaining_isnan}")
    else:
        print("No IsNaN nodes found in the model")

    return onnx_model


def main():
    parser = argparse.ArgumentParser(description="Export bert-base-chinese to ONNX")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./bert-base-chinese",
        help="Path to bert-base-chinese model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./bert_base_chinese_onnx",
        help="Output directory for ONNX file",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"]
    )
    parser.add_argument("--opset", type=int, default=14)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model from {args.model_path}...")
    model = _load_bert_mlm(args.model_path)

    if args.device == "cuda" and torch.cuda.is_available():
        model = model.to(args.device)
    else:
        args.device = "cpu"

    model.eval()

    output_path = output_dir / "bert_base_chinese.onnx"
    opset = int(args.opset)
    if opset < 14:
        print(f"Requested opset {args.opset} is too low for this model export, fallback to opset 14.")
        opset = 14
    _export_onnx(model, output_path, args.device, opset)

    print("Optimizing ONNX model...")
    onnx_model = onnx.load(str(output_path), load_external_data=True)
    onnx_model = _optimize_onnx_model(onnx_model)
    onnx.save_model(onnx_model, str(output_path), save_as_external_data=False)
    _ensure_single_file_onnx(output_path)
    print(f"Optimized model saved to: {output_path}")

    print("\nClearing memory after export...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nExport finished. ONNX model saved to: {output_path}")


if __name__ == "__main__":
    main()
