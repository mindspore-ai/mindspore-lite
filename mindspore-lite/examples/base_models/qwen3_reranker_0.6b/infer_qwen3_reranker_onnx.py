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
Inference with Qwen3-Reranker ONNX model.
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
        description="Inference with Qwen3-Reranker ONNX model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./onnx/qwen3_reranker_0.6b.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-Reranker-0.6B",
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
    Load tokenizer from HuggingFace.
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


def _load_onnx_session(args):
    """
    Load ONNX model from file.
    """
    print(f"Loading ONNX model from {args.model_path}")
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.device == "CUDA"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(args.model_path, providers=providers)
    return session


def _format_instruction(instruction, query, doc):
    """
    Format instruction for ONNX model.
    """
    if instruction is None:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )
    return output


def _prepare_inputs(pairs, tokenizer, prefix_tokens, suffix_tokens, max_length):
    """
    Prepare inputs for ONNX model.
    """
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
    )
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(
        inputs, padding=True, return_tensors="np", max_length=max_length
    )
    return inputs


def _compute_scores(session, inputs, token_true_id, token_false_id):
    """
    Compute scores for ONNX model.
    """
    batch_size = inputs["input_ids"].shape[0]
    scores = []

    for i in range(batch_size):
        ort_inputs = {
            "input_ids": inputs["input_ids"][i : i + 1].astype(np.int64),
            "attention_mask": inputs["attention_mask"][i : i + 1].astype(np.int64),
        }
        logits = session.run(["logits"], ort_inputs)[0]

        last_token_logits = logits[0, -1, :]
        true_score = last_token_logits[token_true_id]
        false_score = last_token_logits[token_false_id]

        scores_array = np.array([false_score, true_score])
        scores_array = np.exp(scores_array - np.max(scores_array))
        scores_array = scores_array / np.sum(scores_array)
        scores.append(scores_array[1])

    return scores


def main():
    """
    Main function for inference.
    """
    args = _parse_args()
    tokenizer = _load_tokenizer(args)
    session = _load_onnx_session(args)

    prefix = (
        "<|im_start|>system\nJudge whether the Document meets "
        "the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" '
        'or "no".<|im_end|>\n<|im_start|>user\n'
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")

    task = "Given a web search query, retrieve relevant passages that answer the query"

    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]

    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. "
        "It gives weight to physical objects and "
        "is responsible for the movement of planets around the sun.",
    ]

    pairs = [
        _format_instruction(task, query, doc) for query, doc in zip(queries, documents)
    ]
    inputs = _prepare_inputs(
        pairs, tokenizer, prefix_tokens, suffix_tokens, args.max_length
    )
    scores = _compute_scores(session, inputs, token_true_id, token_false_id)

    print("\nReranking scores:")
    for i, (query, doc, score) in enumerate(zip(queries, documents, scores)):
        print(f"\n[{i + 1}] Score: {score:.4f}")
        print(f"Query: {query}")
        print(f"Document: {doc}")


if __name__ == "__main__":
    main()
