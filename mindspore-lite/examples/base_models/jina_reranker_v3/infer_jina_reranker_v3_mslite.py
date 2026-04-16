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
Inference with Jina Reranker V3 using MindSpore Lite.
Uses a single unified model.
"""

import argparse
import numpy as np

try:
    import mindspore_lite as mslite  # type: ignore
    from transformers import AutoTokenizer
except ImportError:
    print("Error: mindspore_lite or transformers package not found.")
    print("Please install them first.")


def _parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Inference with Jina Reranker V3 using MindSpore Lite"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to unified MindIR model",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="jinaai/jina-reranker-v3",
        help="Tokenizer model ID or path",
    )
    parser.add_argument(
        "--max-length", type=int, default=8192, help="Maximum sequence length"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for inference (cpu/ascend)"
    )
    parser.add_argument("--device-id", type=int, default=0, help="Device ID for ascend")
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


def _load_mslite_models(args):
    """
    Load MindSpore Lite model from specified path.
    """
    print(f"Initializing MindSpore Lite context for {args.device}...")

    context = mslite.Context()
    context.target = [args.device]
    if args.device == "ascend":
        context.ascend.device_id = args.device_id

    print(f"Loading model from {args.model_path}...")
    model = mslite.Model()
    model.build_from_file(args.model_path, mslite.ModelType.MINDIR, context)
    return model


def _format_query_doc_pair(query, doc):
    """
    Format query and document into input text.
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


def _compute_scores_unified(model, inputs):
    """
    Compute reranking scores using unified model.
    """
    batch_size = inputs["input_ids"].shape[0]
    scores = []

    for i in range(batch_size):
        # mindspore lite 输入要求 int32 类型
        mslite_inputs = [
            mslite.Tensor(inputs["input_ids"][i : i + 1].astype(np.int32)),
            mslite.Tensor(inputs["attention_mask"][i : i + 1].astype(np.int32)),
        ]
        outputs = model.predict(mslite_inputs)
        logits = outputs[0].get_data_to_numpy()

        score = logits[0, 1]
        scores.append(score)

    return scores


def main():
    """
    Main function to run inference with MindSpore Lite.
    """
    args = _parse_args()
    tokenizer = _load_tokenizer(args)
    model = _load_mslite_models(args)

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

    print("\nRunning inference with unified model...")
    scores = _compute_scores_unified(model, inputs)

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
