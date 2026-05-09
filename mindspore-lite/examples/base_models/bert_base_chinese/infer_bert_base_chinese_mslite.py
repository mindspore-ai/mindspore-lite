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
"""BERT MindSpore Lite inference script using MindIR model."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mindspore_lite as mslite
except ImportError:
    print("Error: mindspore_lite package not found.")
    print("Please install: pip install mindspore-lite")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers package not found.")
    print("Please install: pip install transformers")
    sys.exit(1)


class BertMsLiteInferencer:
    """BERT MindSpore Lite inference class for masked language modeling."""

    def __init__(self, model_path: str, tokenizer_path: str, device: str = "ascend", device_id: int = 0):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        self.context = mslite.Context()
        self.context.target = [device]
        if device == "ascend":
            self.context.ascend.device_id = device_id

        self.model = mslite.Model()
        self.model.build_from_file(model_path, mslite.ModelType.MINDIR, self.context)

        self.device = device

    def infer(self, text: str, top_k: int = 5):
        """Run inference on text with [MASK] tokens."""
        inputs = self.tokenizer(text, return_tensors="np")
        input_ids = inputs["input_ids"].astype(np.int32)
        attention_mask = inputs.get("attention_mask", np.ones_like(input_ids)).astype(np.int32)
        token_type_ids = inputs.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int32)

        inputs_mslite = [
            mslite.Tensor(input_ids),
            mslite.Tensor(attention_mask),
            mslite.Tensor(token_type_ids),
        ]

        outputs = self.model.predict(inputs_mslite)
        logits = outputs[0].get_data_to_numpy()

        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = []
        for i, token_id in enumerate(input_ids[0]):
            if token_id == mask_token_id:
                mask_positions.append(i)

        if not mask_positions:
            print("No [MASK] token found in input.")
            return None

        results = []
        for pos in mask_positions:
            token_probs = logits[0, pos]
            top_indices = np.argsort(token_probs)[::-1][:top_k]
            top_tokens = [(self.tokenizer.decode([idx]), token_probs[idx]) for idx in top_indices]
            results.append((pos, top_tokens))

        return results

    def print_predictions(self, predictions):
        """Print prediction results nicely."""
        for pos, top_tokens in predictions:
            print(f"\n[MASK] at position {pos}:")
            for token, prob in top_tokens:
                print(f"  {token}: {prob:.4f}")


def _pick_providers(device: str):
    """Select MindSpore Lite providers based on device."""
    if device == "ascend":
        return ["AscendExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def main():
    parser = argparse.ArgumentParser(description="BERT MindSpore Lite inference")
    parser.add_argument("--model", type=str, required=True, help="Path to MindIR model")
    parser.add_argument("--tokenizer", type=str, default="./bert-base-chinese", help="Path to tokenizer")
    parser.add_argument("--text", type=str, default="今天天气很好，我[MASK]外面去玩。", help="Input text with [MASK] token")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions per mask")
    parser.add_argument("--device", type=str, default="ascend", choices=["cpu", "ascend"], help="Device for inference")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID for ascend")
    parser.add_argument("--runs", type=int, default=10, help="Number of inference runs for timing")

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    print(f"Loading MindIR model from: {args.model}")
    print(f"Loading tokenizer from: {args.tokenizer}")
    print(f"Using device: {args.device}")

    inferencer = BertMsLiteInferencer(args.model, args.tokenizer, args.device, args.device_id)

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
        input_ids = inputs["input_ids"].astype(np.int32)
        attention_mask = inputs.get("attention_mask", np.ones_like(input_ids)).astype(np.int32)
        token_type_ids = inputs.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int32)

        inputs_mslite = [
            mslite.Tensor(input_ids),
            mslite.Tensor(attention_mask),
            mslite.Tensor(token_type_ids),
        ]

        start = time.perf_counter()
        _ = inferencer.model.predict(inputs_mslite)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    print(f"\nAverage latency: {np.mean(latencies):.2f} ms")


if __name__ == "__main__":
    main()
