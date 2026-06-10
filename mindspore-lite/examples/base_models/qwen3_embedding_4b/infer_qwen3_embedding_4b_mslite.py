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
Infer Qwen3-Embedding-4B with MindSpore Lite.

Produces normalized text embeddings for similarity search and retrieval.
"""

import sys
import argparse
import time
import numpy as np

try:
    import mindspore_lite as mslite
    from transformers import AutoTokenizer
except ImportError:
    print("Error: mindspore_lite or transformers package not found.")
    print("Please install them first.")
    sys.exit(1)


TASK_LIST = [
    "Given a web search query, retrieve relevant passages that answer the query",
    "Retrieve semantically similar text",
    "Classify the sentiment of the text",
    "Retrieve the most relevant document for the query",
]


class Qwen3EmbeddingInferencer:
    """
    Qwen3-Embedding-4B inferencer with MindSpore Lite.
    """

    def __init__(self, model_path, tokenizer_id, device_id=0, device_type="cpu"):
        """
        Initialize the embedding inferencer.
        """
        print(f"Initializing MindSpore Lite context for {device_type}...")

        self.context = mslite.Context()
        self.context.target = [device_type]
        if device_type == "ascend":
            self.context.ascend.device_id = device_id

        print(f"Loading model from {model_path}...")
        self.model = mslite.Model()
        self.model.build_from_file(model_path, mslite.ModelType.MINDIR, self.context)

        print(f"Loading tokenizer from {tokenizer_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_detailed_instruct(self, task, query):
        """
        Format query with instruction for retrieval tasks.
        """
        return f"Instruct: {task}\nQuery: {query}"

    def encode(self, texts, instruction=None, max_length=8192, batch_size=1):
        """
        Encode texts into normalized embeddings.

        Args:
            texts: List of text strings to encode.
            instruction: Optional task instruction (applied to queries only).
            max_length: Maximum sequence length.
            batch_size: Batch size for encoding.

        Returns:
            numpy array of shape (len(texts), embedding_dim) with L2-normalized embeddings.
        """
        if instruction:
            formatted = [self._get_detailed_instruct(instruction, t) for t in texts]
        else:
            formatted = list(texts)

        all_embeddings = []
        for i in range(0, len(formatted), batch_size):
            batch = formatted[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )

            t0 = time.perf_counter()
            mslite_inputs = [
                mslite.Tensor(inputs["input_ids"].astype(np.int32)),
                mslite.Tensor(inputs["attention_mask"].astype(np.int32)),
            ]
            outputs = self.model.predict(mslite_inputs)
            embeddings = outputs[0].get_data_to_numpy()
            # L2 normalization
            norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            embeddings = embeddings / norms
            elapsed = (time.perf_counter() - t0) * 1000.0

            all_embeddings.append(embeddings)
            print(f"Batch {i // batch_size + 1}: {len(batch)} texts, {elapsed:.3f}ms")

        return np.concatenate(all_embeddings, axis=0)

    def compute_similarity(self, query_embeddings, doc_embeddings):
        """
        Compute cosine similarity between query and document embeddings.
        """
        return np.dot(query_embeddings, doc_embeddings.T)


def main():
    """
    Main function for Qwen3-Embedding-4B inference with MindSpore Lite.
    """
    parser = argparse.ArgumentParser(
        description="Qwen3-Embedding-4B Inference with MindSpore Lite"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to mindir model"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-Embedding-4B",
        help="Tokenizer model ID or path",
    )
    parser.add_argument(
        "--max-length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--device", type=str, default="ascend", help="Device for inference (cpu/ascend)"
    )
    parser.add_argument("--device-id", type=int, default=0, help="Device ID for ascend")

    args = parser.parse_args()

    inferencer = Qwen3EmbeddingInferencer(
        args.model, args.tokenizer, args.device_id, args.device
    )

    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a fundamental force of nature that attracts two bodies towards each other.",
        "The Eiffel Tower is located in Paris, France.",
        "Python is a popular programming language.",
    ]

    instruction = "Given a web search query, retrieve relevant passages that answer the query"

    print("\nEncoding queries...")
    query_embeddings = inferencer.encode(
        queries, instruction=instruction, max_length=args.max_length
    )
    print(f"Query embeddings shape: {query_embeddings.shape}")

    print("\nEncoding documents...")
    doc_embeddings = inferencer.encode(documents, max_length=args.max_length)
    print(f"Document embeddings shape: {doc_embeddings.shape}")

    similarities = inferencer.compute_similarity(query_embeddings, doc_embeddings)

    print("\nSimilarity scores:")
    for i, query in enumerate(queries):
        print(f"\nQuery: {query}")
        scores = similarities[i]
        ranked = np.argsort(scores)[::-1]
        for j in ranked:
            print(f"  [{scores[j]:.4f}] {documents[j]}")


if __name__ == "__main__":
    main()
