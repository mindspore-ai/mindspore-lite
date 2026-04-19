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
Infer Qwen3 VL Embedding 2B model from ONNX format.
"""

import argparse
import sys
import numpy as np
import onnxruntime as ort

try:
    from transformers import AutoProcessor
except ImportError:
    print("Error: transformers package not found.")
    print("Please install: pip install transformers")
    sys.exit(1)


def run_inference(model_path, tokenizer_path, texts, device="cpu"):
    """
    Run inference on the Qwen3 VL Embedding 2B model from ONNX format.
    """
    print(f"Loading processor from {tokenizer_path}...")
    processor = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)

    print(f"Loading model from {model_path}...")
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
        if device == "cpu"
        else ["CUDAExecutionProvider"],
    )

    print(f"Processing {len(texts)} texts...")

    embeddings = []

    for i, text in enumerate(texts):
        print(f"Processing text {i + 1}/{len(texts)}...")

        inputs = processor(
            text=[text], return_tensors="np", padding=True, truncation=True
        )

        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)

        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)

        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            },
        )

        last_hidden_state = outputs[0]
        embedding = last_hidden_state.mean(axis=1)
        embeddings.append(embedding)

        print(f"  Text {i + 1} embedding shape: {embedding.shape}")

    embeddings = np.array(embeddings)

    print("=" * 50)
    print("Embeddings computed successfully!")
    print("=" * 50)
    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings


def compute_similarity(embeddings):
    """
    Compute cosine similarity between embeddings.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    similarity_matrix = cosine_similarity(embeddings)

    print("\nSimilarity Matrix:")
    print("=" * 50)
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            if i < j:
                print(f"Text {i + 1} vs Text {j + 1}: {similarity_matrix[i][j]:.4f}")

    return similarity_matrix


def main():
    """
    Main function for running inference on the Qwen3 VL Embedding 2B model from ONNX format.
    """
    parser = argparse.ArgumentParser(
        description="Inference with Qwen3-VL-Embedding-2B ONNX model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="./Qwen3-VL-Embedding-2B",
        help="HuggingFace tokenizer path",
    )
    parser.add_argument(
        "--texts",
        type=str,
        nargs="+",
        default=["Hello world", "Hi there", "Good morning"],
        help="List of texts to embed",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for inference (cpu or cuda)"
    )
    parser.add_argument(
        "--compute-similarity",
        action="store_true",
        help="Compute similarity matrix between embeddings",
    )

    args = parser.parse_args()

    embeddings = run_inference(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        texts=args.texts,
        device=args.device,
    )

    if args.compute_similarity:
        compute_similarity(embeddings)


if __name__ == "__main__":
    main()
