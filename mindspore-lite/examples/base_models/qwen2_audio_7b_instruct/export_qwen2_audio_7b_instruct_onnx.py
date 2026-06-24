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
Export Qwen2-Audio-7B-Instruct Language Model to ONNX format.

This script extracts the text LLM backbone from Qwen2-Audio-7B-Instruct
(Qwen2AudioForConditionalGeneration) and exports it as a standard causal LM
for text generation. Audio encoder (Whisper) and multi-modal projector are
not included — this focuses on the Qwen2-7B text generation component.
"""

import argparse
import gc
from pathlib import Path

import torch
from transformers import Qwen2AudioForConditionalGeneration


class Qwen2AudioTextPrefillWrapper(torch.nn.Module):
    """Wrapper for prefill: input_ids -> logits (no KV cache in ONNX)."""

    def __init__(self, text_model, lm_head):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask):
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        return logits


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen2-Audio-7B-Instruct Text LLM to ONNX"
    )
    parser.add_argument(
        "--model-id", type=str, default="Qwen/Qwen2-Audio-7B-Instruct",
        help="Model ID or local path",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./qwen2_audio_7b_instruct_onnx",
        help="Output directory for ONNX model",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for export",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load full Qwen2-Audio model
    print(f"Loading Qwen2-Audio-7B-Instruct model from {args.model_id}...")
    full_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Extract text LLM backbone + lm_head
    # Qwen2AudioForConditionalGeneration uses 'language_model' (Qwen2ForCausalLM)
    text_model = full_model.language_model.model
    lm_head = full_model.language_model.lm_head

    # Free non-text components (audio_tower, multi_modal_projector)
    del full_model
    gc.collect()

    # Create wrapper and export
    wrapper = Qwen2AudioTextPrefillWrapper(text_model, lm_head).to(device).eval()

    # Dummy inputs
    dummy_input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long).to(device)
    dummy_attention_mask = torch.ones_like(dummy_input_ids)

    output_path = str(output_dir / "qwen2_audio_7b_instruct_text.onnx")

    print(f"Exporting text LLM to {output_path}...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size", 1: "sequence"},
            },
        )

    print(f"\nExport completed successfully!")
    print(f"ONNX model saved to: {output_path}")


if __name__ == "__main__":
    main()
