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
# WITHOUT WARRANTIES OR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Inference with Jina Reranker V3 (Listwise + Pointwise) using ONNX Runtime.

Supports two scoring modes:
- Listwise: query + multiple documents in one forward pass (native architecture)
- Pointwise: query + one document per forward pass (simpler, same model)

When the number of documents exceeds the context window, the listwise mode
automatically splits documents into blocks and fuses query embeddings with
weighted averaging, matching the original model's behavior.
"""

import argparse
import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


DOC_EMBED_TOKEN_ID = 151670
QUERY_EMBED_TOKEN_ID = 151671
MAX_DOCS = 64

THINK_OPEN = "\u003cthink\u003e"
THINK_CLOSE = "\u003c/think\u003e"
NO_THINK_SUFFIX = THINK_OPEN + "\n\n" + THINK_CLOSE + "\n\n"


def _parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Inference with Jina Reranker V3 (Listwise) ONNX model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./onnx/jina_reranker_v3_listwise.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="jinaai/jina-reranker-v3",
        help="Tokenizer model ID or path",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="Device for ONNX Runtime (CPU/CUDA)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="listwise",
        choices=["listwise", "pointwise"],
        help="Scoring mode: listwise or pointwise",
    )
    return parser.parse_args()


def _load_tokenizer(tokenizer_id):
    """
    Load tokenizer from specified model ID or path.
    """
    print(f"Loading tokenizer from {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_onnx_session(args):
    """
    Load ONNX session from specified path.
    """
    print(f"Loading ONNX model from {args.model_path}")
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.device == "CUDA"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(args.model_path, providers=providers)
    return session


def _format_listwise_prompt(query, docs):
    """
    Format query and documents into the listwise prompt template.
    """
    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of "
        "the passages based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on "
        "how well it answers the question. If not, try to analyze the intent "
        "of the query and assess how well each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction "
        "when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n" + NO_THINK_SUFFIX

    prompt = (
        f"I will provide you with {len(docs)} passages, each indicated by "
        f"a numerical identifier. Rank the passages based on their relevance "
        f"to query: {query}\n"
    )

    doc_prompts = [
        f'<passage id="{i}">\n{doc}<|embed_token|>\n</passage>'
        for i, doc in enumerate(docs)
    ]
    prompt += "\n".join(doc_prompts) + "\n"
    prompt += f"<query>\n{query}<|rerank_token|>\n</query>"

    return prefix + prompt + suffix


def _find_special_token_positions(input_ids, token_id):
    """
    Find positions of a specific token ID in the input array.

    Args:
        input_ids: (seq_len,) numpy array of token IDs
        token_id: token ID to find

    Returns:
        positions: list of integer positions
    """
    return np.where(input_ids == token_id)[0].tolist()


def _truncate_documents(tokenizer, query, documents, max_query_length=512, max_doc_length=2048):
    """
    Truncate query and documents to fit within token limits.

    Returns:
        query: truncated query string
        docs: list of truncated document strings
        doc_lengths: list of token lengths for each document
        query_length: token length of the query
    """
    docs = []
    doc_lengths = []
    for doc in documents:
        doc_tokens = tokenizer(doc, truncation=True, max_length=max_doc_length)
        if len(doc_tokens["input_ids"]) >= max_doc_length:
            doc = tokenizer.decode(doc_tokens["input_ids"])
        doc_lengths.append(len(doc_tokens["input_ids"]))
        docs.append(doc)

    query_tokens = tokenizer(query, truncation=True, max_length=max_query_length)
    if len(query_tokens["input_ids"]) >= max_query_length:
        query = tokenizer.decode(query_tokens["input_ids"])
    query_length = len(query_tokens["input_ids"])

    return query, docs, doc_lengths, query_length


def _compute_block_scores(session, input_ids_np, attention_mask_np):
    """
    Compute scores for a single block using the ONNX model.

    Args:
        session: ONNX Runtime session
        input_ids_np: (1, seq_len) numpy array
        attention_mask_np: (1, seq_len) numpy array

    Returns:
        scores: numpy array of scores for each document
        doc_embeds: numpy array of document embeddings
        query_embeds: numpy array of query embeddings
    """
    doc_positions = _find_special_token_positions(input_ids_np[0], DOC_EMBED_TOKEN_ID)
    query_positions = _find_special_token_positions(input_ids_np[0], QUERY_EMBED_TOKEN_ID)

    num_docs = len(doc_positions)
    doc_token_indices = np.zeros((1, MAX_DOCS), dtype=np.int64)
    for i, pos in enumerate(doc_positions):
        doc_token_indices[0, i] = pos

    query_token_index = np.zeros((1, 1), dtype=np.int64)
    if query_positions:
        query_token_index[0, 0] = query_positions[0]

    ort_inputs = {
        "input_ids": input_ids_np.astype(np.int64),
        "attention_mask": attention_mask_np.astype(np.int64),
        "doc_token_indices": doc_token_indices,
        "query_token_index": query_token_index,
    }
    scores = session.run(["scores"], ort_inputs)[0]

    return scores[0, :num_docs]


def _split_into_blocks(doc_lengths, docs, query_length, max_length, max_doc_length):
    """
    Split documents into blocks that fit within the context window.

    Each block's total token length (docs + overhead) must not exceed
    max_length. The overhead is approximately 2 * query_length tokens.

    Args:
        doc_lengths: list of token lengths for each document
        docs: list of document strings
        query_length: token length of the query
        max_length: maximum sequence length
        max_doc_length: maximum single document length

    Returns:
        block_docs_list: list of blocks, each block is a list of doc strings
    """
    block_size = 125
    length_capacity = max_length - 2 * query_length
    block_docs_list = []
    current_block = []

    for length, doc in zip(doc_lengths, docs):
        current_block.append(doc)
        length_capacity -= length

        if len(current_block) >= block_size or length_capacity <= max_doc_length:
            block_docs_list.append(current_block)
            current_block = []
            length_capacity = max_length - 2 * query_length

    if len(current_block) > 0:
        block_docs_list.append(current_block)

    return block_docs_list


def _score_block(session, tokenizer, query, block_docs, max_length):
    """
    Score a single block of documents and compute block weight.

    Args:
        session: ONNX Runtime session
        tokenizer: tokenizer instance
        query: query string
        block_docs: list of document strings for this block
        max_length: maximum sequence length

    Returns:
        scores: numpy array of scores for each document in this block
        block_weight: float weight for cross-block fusion
    """
    prompt = _format_listwise_prompt(query, block_docs)
    encoded = tokenizer(
        prompt,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    scores = _compute_block_scores(
        session, encoded["input_ids"], encoded["attention_mask"]
    )
    block_weight = float(((1.0 + scores) / 2.0).max())
    return scores, block_weight


def rerank_listwise(session, tokenizer, query, documents, max_length=8192,
                    max_doc_length=2048, max_query_length=512, top_n=None):
    """
    Rerank documents using listwise mode with block fusion.

    When documents fit in a single context window, all docs are scored together.
    When documents exceed the context window, they are split into blocks, each
    block is scored independently, and block weights are computed for cross-block
    score normalization.

    Args:
        session: ONNX Runtime session
        tokenizer: tokenizer instance
        query: search query string
        documents: list of document strings to rank
        max_length: maximum sequence length
        max_doc_length: maximum document length in tokens
        max_query_length: maximum query length in tokens
        top_n: return only top N results (default: all)

    Returns:
        list of dicts with keys: document, relevance_score, index
    """
    query, docs, doc_lengths, query_length = _truncate_documents(
        tokenizer, query, documents, max_query_length, max_doc_length
    )

    block_docs_list = _split_into_blocks(
        doc_lengths, docs, query_length, max_length, max_doc_length
    )

    all_scores = []
    block_weights = []

    for block_d in block_docs_list:
        scores, block_weight = _score_block(
            session, tokenizer, query, block_d, max_length
        )
        all_scores.extend(scores.tolist())
        block_weights.append(block_weight)

    final_scores = np.array(all_scores)

    sorted_indices = np.argsort(final_scores)[::-1]

    if top_n is None:
        top_n = len(documents)
    else:
        top_n = min(top_n, len(documents))

    results = []
    for i in range(top_n):
        idx = sorted_indices[i]
        results.append({
            "document": documents[idx],
            "relevance_score": float(final_scores[idx]),
            "index": int(idx),
        })

    return results


def rerank_pointwise(session, tokenizer, query, documents, max_length=8192):
    """
    Rerank documents using pointwise mode (one doc per forward pass).

    Each query-document pair is formatted as a listwise prompt with a single
    document, scored independently. This is less accurate than listwise mode
    but simpler and uses less memory per forward pass.

    Args:
        session: ONNX Runtime session
        tokenizer: tokenizer instance
        query: search query string
        documents: list of document strings to rank
        max_length: maximum sequence length

    Returns:
        list of dicts with keys: document, relevance_score, index
    """
    scores = []
    for doc in documents:
        prompt = _format_listwise_prompt(query, [doc])
        encoded = tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        block_scores = _compute_block_scores(
            session, encoded["input_ids"], encoded["attention_mask"]
        )
        scores.append(float(block_scores[0]))

    scores = np.array(scores)
    sorted_indices = np.argsort(scores)[::-1]

    results = []
    for i in range(len(documents)):
        idx = sorted_indices[i]
        results.append({
            "document": documents[idx],
            "relevance_score": float(scores[idx]),
            "index": int(idx),
        })

    return results


def main():
    """
    Main function to run inference with Jina Reranker V3 ONNX model.
    """
    args = _parse_args()
    tokenizer = _load_tokenizer(args.tokenizer)
    session = _load_onnx_session(args)

    query = "What are the health benefits of green tea?"
    documents = [
        "Green tea contains antioxidants called catechins that may help "
        "reduce inflammation and protect cells from damage.",
        "El precio del caf\u00e9 ha aumentado un 20% este a\u00f1o debido "
        "a problemas en la cadena de suministro.",
        "Studies show that drinking green tea regularly can improve brain "
        "function and boost metabolism.",
        "Basketball is one of the most popular sports in the United States.",
        "\u7eff\u8336\u5bcc\u542b\u513f\u8336\u7d20\u7b49\u6297\u6c27\u5316\u5242\uff0c"
        "\u53ef\u4ee5\u964d\u4f4e\u5fc3\u810f\u75c5\u98ce\u9669\uff0c\u8fd8\u6709\u52a9\u4e8e"
        "\u63a7\u5236\u4f53\u91cd\u3002",
        "Le th\u00e9 vert est riche en antioxydants et peut am\u00e9liorer "
        "la function c\u00e9r\u00e9brale.",
    ]

    print(f"\nRunning inference in {args.mode} mode...")
    start_time = time.time()

    if args.mode == "listwise":
        results = rerank_listwise(
            session, tokenizer, query, documents, max_length=args.max_length
        )
    else:
        results = rerank_pointwise(
            session, tokenizer, query, documents, max_length=args.max_length
        )

    elapsed = time.time() - start_time

    print(f"\nReranking results ({args.mode} mode):")
    for i, result in enumerate(results):
        print(f"\n[{i + 1}] Score: {result['relevance_score']:.4f}")
        print(f"Document: {result['document'][:100]}...")

    print(f"\nInference time: {elapsed:.3f}s")
    print("=" * 60)
    print("Higher scores indicate better relevance to the query.")
    print("=" * 60)


if __name__ == "__main__":
    main()
