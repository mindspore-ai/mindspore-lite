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
Export Qwen3-Embedding-0.6B model to ONNX format.

The model produces text embeddings by:
1. Running the causal LM forward pass
2. Extracting the last token's hidden state
3. Applying L2 normalization
"""

import argparse
import os
from pathlib import Path
import torch
import onnx
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen3EmbeddingWrapper(torch.nn.Module):
    """
    Wrapper for Qwen3-Embedding-0.6B that outputs normalized embeddings.
    Uses last-token pooling with L2 normalization.
    """

    def __init__(self, model):
        """
        Initialize the embedding wrapper.
        """
        super().__init__()
        self.model = model.model
        self.lm_head = model.lm_head

    def forward(self, input_ids, attention_mask):
        """
        Forward pass producing pooled embeddings (last token).
        L2 normalization is done in the inference script, not in ONNX,
        to avoid compatibility issues with Clip/ConstantOfShape operators.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state
        # Last token pooling: get the last non-padding token's representation
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        embeddings = hidden_states[torch.arange(batch_size), sequence_lengths]
        return embeddings


def _parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Export Qwen3-Embedding-0.6B to ONNX")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
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
    Load model and tokenizer.
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


def _create_dummy_inputs(tokenizer):
    """
    Create dummy inputs for ONNX export.
    """
    dummy_text = "This is a test sentence for embedding."
    tokens = tokenizer(dummy_text, return_tensors="pt", padding=False, truncation=True)
    dummy_input_ids = tokens["input_ids"]
    dummy_attention_mask = tokens["attention_mask"]
    return dummy_input_ids, dummy_attention_mask


def _export_to_onnx(model, output_path, dummy_input_ids, dummy_attention_mask):
    """
    Export model to ONNX format.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "embeddings": {0: "batch_size"},
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
            output_names=["embeddings"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    print(f"Model exported successfully to {output_path}")

    # Check if external data needs consolidation
    onnx_size = os.path.getsize(output_path)
    if onnx_size > 100 * 1024 * 1024:  # > 100MB means scattered external data
        print(f"ONNX file is {onnx_size/1024/1024:.1f}MB, consolidating external data...")
        _consolidate_onnx_external_data(output_path)
    else:
        print(f"ONNX file size: {onnx_size/1024/1024:.1f}MB")


def _consolidate_onnx_external_data(onnx_path):
    """Consolidate ONNX external data into a single .data file."""
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
                and f.suffix not in (".onnx", ".data", ".mindir")):
            f.unlink(missing_ok=True)
    onnx.save_model(model, str(onnx_path))
    print(f"External data consolidated into {data_name}")


def main():
    """
    Main function.
    """
    args = _parse_args()
    model, tokenizer = _load_model_and_tokenizer(args)
    wrapper = Qwen3EmbeddingWrapper(model).to(args.device).eval()
    dummy_input_ids, dummy_attention_mask = _create_dummy_inputs(tokenizer)
    dummy_input_ids = dummy_input_ids.to(args.device)
    dummy_attention_mask = dummy_attention_mask.to(args.device)

    output_path = os.path.join(args.output_dir, "qwen3_embedding_0_6b.onnx")
    _export_to_onnx(wrapper, output_path, dummy_input_ids, dummy_attention_mask)

    print("\nExport completed successfully!")
    print(f"ONNX model saved to: {output_path}")
    print(f"Max sequence length: {args.max_length}")


if __name__ == "__main__":
    main()
