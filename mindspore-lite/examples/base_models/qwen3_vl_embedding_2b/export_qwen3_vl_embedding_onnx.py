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
Export Qwen3 VL Embedding 2B model to ONNX format.
"""

import sys
import argparse
import gc
from pathlib import Path
import torch

try:
    import torch._dynamo

    torch._dynamo.disable()
except:
    pass

try:
    from transformers import AutoModel
except ImportError:
    print("Error: transformers package not found.")
    print("Please install: pip install transformers")
    sys.exit(1)


class Qwen3VLEmbeddingModel(torch.nn.Module):
    """
    Qwen3 VL Embedding 2B model wrapper.
    """

    def __init__(self, model):
        """
        Initialize the Qwen3 VL Embedding 2B model wrapper.
        """
        super().__init__()
        self.model = model

    def forward(
        self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None
    ):
        """
        Forward pass of the Qwen3 VL Embedding 2B model wrapper.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
        return outputs.last_hidden_state


def export_embedding(model, output_dir, device="cpu"):
    """
    Export the Qwen3 VL Embedding 2B model to ONNX format.
    """
    model.eval()
    model.to(device)

    output_path = Path(output_dir) / "qwen3_vl_embedding_2b.onnx"

    dummy_seq = 128
    dummy_input_ids = torch.randint(
        0, 1000, (1, dummy_seq), dtype=torch.int64, device=device
    )
    dummy_attention_mask = torch.ones(1, dummy_seq, dtype=torch.int64, device=device)

    input_names = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
    output_names = ["last_hidden_state"]

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "pixel_values": {0: "batch"},
        "image_grid_thw": {0: "batch"},
        "last_hidden_state": {0: "batch", 1: "seq_len"},
    }

    embedding_wrapper = Qwen3VLEmbeddingModel(model).to(device).eval()

    print(f"Exporting embedding model to {output_path}...")

    with torch.no_grad():
        torch.onnx.export(
            embedding_wrapper,
            (dummy_input_ids, dummy_attention_mask, None, None),
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=18,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )

    print("Embedding model exported successfully.")


def main():
    """
    Main function for exporting the Qwen3 VL Embedding 2B model to ONNX format.
    """
    parser = argparse.ArgumentParser(description="Export Qwen3-VL-Embedding-2B to ONNX")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-VL-Embedding-2B",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./qwen3_vl_embedding_onnx",
        help="Output directory",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for export (cpu or cuda)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model {args.model_id} in FP16 for export...")
    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map=args.device,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    export_embedding(model, output_dir, args.device)

    print("Clearing memory after export...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nExport finished. Files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
