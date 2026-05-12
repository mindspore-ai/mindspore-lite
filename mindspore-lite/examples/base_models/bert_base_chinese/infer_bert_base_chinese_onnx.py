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
ONNX Runtime inference for bert-base-chinese (BertForMaskedLM).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime package not found.")
    print("Please install: pip install onnxruntime")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers package not found.")
    print("Please install: pip install transformers")
    sys.exit(1)


def _pick_providers(device: str):
    """Select ONNX Runtime providers based on device."""
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _masked_language_modeling_predict(session, input_ids, attention_mask, token_type_ids):
    """Run masked language modeling prediction."""
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    logits = session.run(None, inputs)[0]
    return logits


def _find_mask_token_positions(input_ids, tokenizer):
    """Find positions of [MASK] tokens in the input."""
    mask_token_id = tokenizer.mask_token_id
    positions = []
    for i, token_id in enumerate(input_ids[0]):
        if token_id == mask_token_id:
            positions.append(i)
    return positions


def _replace_mask_with_predictions(logits, mask_positions, tokenizer, top_k=5):
    """Replace masked positions with top-k predicted tokens."""
    results = []
    for pos in mask_positions:
        token_probs = logits[0, pos]
        top_indices = np.argsort(token_probs)[::-1][:top_k]
        top_tokens = [(tokenizer.decode([idx]), token_probs[idx]) for idx in top_indices]
        results.append((pos, top_tokens))
    return results


class BertOnnxInferencer:
    """BERT ONNX inference class for masked language modeling."""

    def __init__(self, model_path: str, tokenizer_path: str, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        providers = _pick_providers(device)
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.device = device

    def infer(self, text: str, top_k: int = 5):
        """Run inference on text with [MASK] tokens."""
        inputs = self.tokenizer(text, return_tensors="np")
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs.get("attention_mask", np.ones_like(input_ids)).astype(np.int64)
        token_type_ids = inputs.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int64)

        logits = _masked_language_modeling_predict(
            self.session, input_ids, attention_mask, token_type_ids
        )

        mask_positions = _find_mask_token_positions(input_ids, self.tokenizer)

        if not mask_positions:
            print("No [MASK] token found in input.")
            return None

        predictions = _replace_mask_with_predictions(logits, mask_positions, self.tokenizer, top_k)
        return predictions

    def print_predictions(self, predictions):
        """Print prediction results nicely."""
        for pos, top_tokens in predictions:
            print(f"\n[MASK] at position {pos}:")
            for token, prob in top_tokens:
                print(f"  {token}: {prob:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="ONNX Runtime inference for bert-base-chinese"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to ONNX model"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="./bert-base-chinese",
        help="Path to tokenizer directory",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="今天天气很好，我[MASK]外面去玩。",
        help="Input text with [MASK] token",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions per mask",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for ONNX Runtime",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of inference runs for timing",
    )

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    print(f"Loading ONNX model from: {args.model}")
    print(f"Loading tokenizer from: {args.tokenizer}")

    inferencer = BertOnnxInferencer(args.model, args.tokenizer, args.device)

    print(f"\nInput text: {args.text}")
    predictions = inferencer.infer(args.text, args.top_k)

    if predictions:
        print("\n" + "=" * 50)
        print("Predictions:")
        inferencer.print_predictions(predictions)
        print("=" * 50)

    latencies = []
    for _ in range(args.runs):
        inputs = inferencer.tokenizer(args.text, return_tensors="np")
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs.get("attention_mask", np.ones_like(input_ids)).astype(np.int64)
        token_type_ids = inputs.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int64)

        start = time.perf_counter()
        _ = _masked_language_modeling_predict(
            inferencer.session, input_ids, attention_mask, token_type_ids
        )
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    print(f"\nAverage latency: {np.mean(latencies):.2f} ms")


if __name__ == "__main__":
    main()
