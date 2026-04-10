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
Inference with Jina Reranker V3 ONNX model.
Supports both unified and split model modes.
"""

import argparse
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def _parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Inference with Jina Reranker V3 ONNX model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./onnx/jina_reranker_v3.onnx",
        help="Path to ONNX model (unified) or encoder model (split mode)",
    )
    parser.add_argument(
        "--head-path",
        type=str,
        default=None,
        help="Path to head ONNX model (required for split mode)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="jinaai/jina-reranker-v3-base",
        help="Tokenizer model ID or path",
    )
    parser.add_argument(
        "--max-length", type=int, default=8192, help="Maximum sequence length"
    )
    parser.add_argument(
        "--device", type=str, default="CPU", help="Device for ONNX Runtime (CPU/CUDA)"
    )
    return parser.parse_args()


def _load_tokenizer(args):
    """
    Load tokenizer from specified model ID or path.
    """
    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_onnx_sessions(args):
    """
    Load ONNX sessions from specified paths.
    """
    print(f"Loading ONNX model from {args.model_path}")
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.device == "CUDA"
        else ["CPUExecutionProvider"]
    )
    encoder_session = ort.InferenceSession(args.model_path, providers=providers)

    if args.head_path:
        print(f"Loading head model from {args.head_path}")
        head_session = ort.InferenceSession(args.head_path, providers=providers)
        return encoder_session, head_session
    return encoder_session, None


def _format_query_doc_pair(query, doc):
    """
    Format query and document into input text.
    Jina Reranker V3 expects query and document to be concatenated.
    """
    return f"Query: {query} Document: {doc}"


def _prepare_inputs(pairs, tokenizer, max_length):
    """
    Prepare input tensors for inference.
    """
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    return inputs


def _compute_scores_unified(session, inputs):
    """
    Compute reranking scores using unified model.
    """
    batch_size = inputs["input_ids"].shape[0]
    scores = []

    for i in range(batch_size):
        ort_inputs = {
            "input_ids": inputs["input_ids"][i : i + 1].astype(np.int64),
            "attention_mask": inputs["attention_mask"][i : i + 1].astype(np.int64),
        }
        logits = session.run(["logits"], ort_inputs)[0]

        score = logits[0, 1]
        scores.append(score)

    return scores


def _compute_scores_split(encoder_session, head_session, inputs):
    """
    Compute reranking scores using split model (encoder + head).
    """
    batch_size = inputs["input_ids"].shape[0]
    scores = []

    for i in range(batch_size):
        encoder_inputs = {
            "input_ids": inputs["input_ids"][i : i + 1].astype(np.int64),
            "attention_mask": inputs["attention_mask"][i : i + 1].astype(np.int64),
        }
        hidden_states = encoder_session.run(["logits"], encoder_inputs)[0]

        head_inputs = {
            "hidden_states": hidden_states.astype(np.float16),
        }
        logits = head_session.run(["logits"], head_inputs)[0]

        score = logits[0, 1]
        scores.append(score)

    return scores


def main():
    """
    Main function to run inference with Jina Reranker V3 ONNX model.
    """
    args = _parse_args()
    tokenizer = _load_tokenizer(args)
    encoder_session, head_session = _load_onnx_sessions(args)

    queries = [
        "What is the capital of China?",
        "Explain gravity",
        "How does photosynthesis work?",
    ]

    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. "
        "It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide "
        "into glucose and oxygen.",
    ]

    pairs = [
        _format_query_doc_pair(query, doc) for query, doc in zip(queries, documents)
    ]
    inputs = _prepare_inputs(pairs, tokenizer, args.max_length)

    if head_session:
        print("\nRunning inference with split model (encoder + head)...")
        scores = _compute_scores_split(encoder_session, head_session, inputs)
    else:
        print("\nRunning inference with unified model...")
        scores = _compute_scores_unified(encoder_session, inputs)

    print("\nReranking scores:")
    for i, (query, doc, score) in enumerate(zip(queries, documents, scores)):
        print(f"\n[{i + 1}] Score: {score:.4f}")
        print(f"Query: {query}")
        print(f"Document: {doc}")

    print("\n" + "=" * 60)
    print("Higher scores indicate better relevance to the query.")
    print("=" * 60)


if __name__ == "__main__":
    main()
