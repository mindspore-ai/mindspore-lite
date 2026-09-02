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
Inference with Jina Reranker V3.5 (Listwise only) using MindSpore Lite.

Supports listwise scoring with block fusion: when documents exceed the context
window, they are split into blocks; each block is scored independently and the
block weights are used to normalize scores across blocks, matching the
original model's behavior.

Note: This script does NOT depend on torch. All preprocessing uses numpy.
"""

import argparse
import sys
import time

import numpy as np

try:
    import mindspore_lite as mslite
    from transformers import AutoTokenizer
except ImportError:
    print("Error: mindspore_lite or transformers package not found.")
    print("Please install them first.")
    sys.exit(1)


DOC_EMBED_TOKEN_ID = 151670
QUERY_EMBED_TOKEN_ID = 151671
MAX_DOCS = 64

THINK_OPEN = "think"
THINK_CLOSE = "/think"
NO_THINK_SUFFIX = THINK_OPEN + "\n\n" + THINK_CLOSE + "\n\n"

_PERF = {
    "tokenizer_load_s": 0.0,
    "model_build_s": 0.0,
    "tokenize_s": 0.0,
    "resize_s": 0.0,
    "predict_s": 0.0,
    "postprocess_s": 0.0,
}
_RUN_INFO = {
    "mode": None,
    "model_max_length": None,
    "io_shapes": None,
}
_WARMED_UP_SEQ_LENS = set()


def _sec_to_ms(v):
    return float(v) * 1000.0


def _print_perf_tables(mode, model_max_length, io_shapes):
    """Print input/output shape and latency tables for the current run."""
    print("\n### MSLite 推理输入输出（本次运行）")
    print("\n| 项目 | 值 |")
    print("|---|---|")
    print(f"| mode | `{mode}` |")
    print(f"| max_length(bucket) | `{int(model_max_length)}` |")
    for k, v in io_shapes.items():
        print(f"| {k} | `{v}` |")

    total_s = (
        _PERF["tokenize_s"]
        + _PERF["resize_s"]
        + _PERF["predict_s"]
        + _PERF["postprocess_s"]
    )
    print("\n### 端到端推理性能（本次运行）")
    print("\n| 指标 | 耗时 (ms) |")
    print("|---|---:|")
    print(f"| Tokenize + pad | {_sec_to_ms(_PERF['tokenize_s']):.2f} |")
    print(f"| Model resize | {_sec_to_ms(_PERF['resize_s']):.2f} |")
    print(f"| Model predict | {_sec_to_ms(_PERF['predict_s']):.2f} |")
    print(f"| Postprocess | {_sec_to_ms(_PERF['postprocess_s']):.2f} |")
    print(f"| **总耗时** | **{_sec_to_ms(total_s):.2f}** |")


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inference with Jina Reranker V3.5 (Listwise) using MindSpore Lite"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to MindIR model",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="jinaai/jina-reranker-v3.5",
        help="Tokenizer model ID or path",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1280,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (cpu/ascend)",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID for ascend",
    )
    return parser.parse_args()


def _load_tokenizer(tokenizer_id):
    """Load tokenizer from specified model ID or path."""
    print(f"Loading tokenizer from {tokenizer_id}")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _PERF["tokenizer_load_s"] += time.perf_counter() - t0
    return tokenizer


def _load_mslite_model(args):
    """Load MindSpore Lite model from specified path."""
    print(f"Initializing MindSpore Lite context for {args.device}...")

    context = mslite.Context()
    context.target = [args.device]
    if args.device == "ascend":
        context.ascend.device_id = args.device_id

    print(f"Loading model from {args.model_path}...")
    t0 = time.perf_counter()
    model = mslite.Model()
    model.build_from_file(args.model_path, mslite.ModelType.MINDIR, context)
    _PERF["model_build_s"] += time.perf_counter() - t0
    return model


def _resolve_model_max_length(model, fallback_max_length):
    """Resolve the fixed max_length required by the compiled MindIR."""
    inputs = model.get_inputs()
    for inp in inputs:
        if inp.name == "input_ids":
            shape = getattr(inp, "shape", None)
            if shape is None or len(shape) != 2:
                break
            seq_len = int(shape[1])
            if seq_len > 0:
                return seq_len
            break
    return int(fallback_max_length)


def _format_listwise_prompt(query, docs):
    """Format query and documents into the listwise prompt template."""
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
    """Find positions of a specific token ID in the input array."""
    return np.where(input_ids == token_id)[0].tolist()


def _build_mslite_inputs(model, input_ids_np, attention_mask_np,
                         doc_token_indices_np, query_token_index_np):
    """Build MindSpore Lite input tensors by matching input names."""
    inputs = model.get_inputs()

    feed = {
        "input_ids": input_ids_np.astype(np.int32),
        "attention_mask": attention_mask_np.astype(np.int32),
        "doc_token_indices": doc_token_indices_np.astype(np.int32),
        "query_token_index": query_token_index_np.astype(np.int32),
    }

    dims = []
    for inp in inputs:
        if inp.name not in feed:
            raise ValueError(f"Unknown model input: {inp.name}")
        dims.append(list(feed[inp.name].shape))
    t0 = time.perf_counter()
    model.resize(inputs, dims)
    _PERF["resize_s"] += time.perf_counter() - t0

    mslite_inputs = []
    for inp in inputs:
        mslite_inputs.append(mslite.Tensor(feed[inp.name]))

    return mslite_inputs


def _truncate_documents(tokenizer, query, documents, max_query_length=512, max_doc_length=2048):
    """Truncate query and documents to fit within token limits."""
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


def _compute_block_scores(model, input_ids_np, attention_mask_np):
    """Compute scores for a single block using the MindSpore Lite model."""
    doc_positions = _find_special_token_positions(input_ids_np[0], DOC_EMBED_TOKEN_ID)
    query_positions = _find_special_token_positions(input_ids_np[0], QUERY_EMBED_TOKEN_ID)

    num_docs = len(doc_positions)
    doc_token_indices = np.zeros((1, MAX_DOCS), dtype=np.int32)
    for i, pos in enumerate(doc_positions):
        doc_token_indices[0, i] = pos

    query_token_index = np.zeros((1, 1), dtype=np.int32)
    if query_positions:
        query_token_index[0, 0] = query_positions[0]

    mslite_inputs = _build_mslite_inputs(
        model, input_ids_np, attention_mask_np,
        doc_token_indices, query_token_index,
    )
    seq_len = int(input_ids_np.shape[1])
    if seq_len not in _WARMED_UP_SEQ_LENS:
        model.predict(mslite_inputs)
        _WARMED_UP_SEQ_LENS.add(seq_len)
    t0 = time.perf_counter()
    outputs = model.predict(mslite_inputs)
    _PERF["predict_s"] += time.perf_counter() - t0
    scores = outputs[0].get_data_to_numpy()

    return scores[0, :num_docs]


def _split_into_blocks(doc_lengths, docs, query_length, max_length, max_doc_length):
    """Split documents into blocks that fit within the context window."""
    del max_doc_length
    block_size = 125
    total_capacity = max_length - 2 * query_length
    block_docs_list = []
    current_block = []
    current_used = 0

    for length, doc in zip(doc_lengths, docs):
        length = int(length)
        if not current_block:
            current_block = [doc]
            current_used = length
            continue

        if len(current_block) >= block_size or (current_used + length) > total_capacity:
            block_docs_list.append(current_block)
            current_block = [doc]
            current_used = length
        else:
            current_block.append(doc)
            current_used += length

    if current_block:
        block_docs_list.append(current_block)

    return block_docs_list


def _score_block(model, tokenizer, query, block_docs, max_length):
    """Score a single block of documents and compute block weight."""
    prompt = _format_listwise_prompt(query, block_docs)
    t0 = time.perf_counter()
    encoded = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    _PERF["tokenize_s"] += time.perf_counter() - t0

    if _RUN_INFO["io_shapes"] is None:
        _RUN_INFO["mode"] = "listwise"
        _RUN_INFO["model_max_length"] = int(max_length)
        _RUN_INFO["io_shapes"] = {
            "input_ids Shape": encoded["input_ids"].shape,
            "attention_mask Shape": encoded["attention_mask"].shape,
            "doc_token_indices Shape": (1, MAX_DOCS),
            "query_token_index Shape": (1, 1),
            "scores Shape": (1, len(block_docs)),
        }
    scores = _compute_block_scores(
        model, encoded["input_ids"], encoded["attention_mask"]
    )
    block_weight = float(((1.0 + scores) / 2.0).max())
    return scores, block_weight


def rerank_listwise(model, tokenizer, query, documents, max_length=1280,
                    max_doc_length=2048, max_query_length=512, top_n=None):
    """Rerank documents using listwise mode with block fusion."""
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
            model, tokenizer, query, block_d, max_length
        )
        all_scores.extend(scores.tolist())
        block_weights.append(block_weight)

    final_scores = np.array(all_scores)
    t0 = time.perf_counter()

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
    _PERF["postprocess_s"] += time.perf_counter() - t0
    return results


def main():
    """Main function to run inference with MindSpore Lite."""
    args = _parse_args()
    tokenizer = _load_tokenizer(args.tokenizer)
    model = _load_mslite_model(args)
    model_max_length = _resolve_model_max_length(model, args.max_length)

    query = "What are the health benefits of green tea?"
    documents = [
        "Green tea contains antioxidants called catechins that may help "
        "reduce inflammation and protect cells from damage.",
        "El precio del café ha aumentado un 20% este año debido "
        "a problemas en la cadena de suministro.",
        "Studies show that drinking green tea regularly can improve brain "
        "function and boost metabolism.",
        "Basketball is one of the most popular sports in the United States.",
        "绿茶富含儿茶素等抗氧化剂，"
        "可以降低心脏病风险，还有助于"
        "控制体重。",
        "Le thé vert est riche en antioxydants et peut améliorer "
        "la function cérébrale.",
    ]

    print(f"\nRunning inference in listwise mode with max_length={model_max_length}...")

    results = rerank_listwise(
        model, tokenizer, query, documents, max_length=model_max_length
    )

    print("\nReranking results (listwise mode):")
    for i, result in enumerate(results):
        print(f"\n[{i + 1}] Score: {result['relevance_score']:.4f}")
        print(f"Document: {result['document'][:100]}...")

    if _RUN_INFO["io_shapes"] is not None:
        _print_perf_tables(
            _RUN_INFO["mode"] or "listwise",
            _RUN_INFO["model_max_length"] or model_max_length,
            _RUN_INFO["io_shapes"],
        )
    print("=" * 60)
    print("Higher scores indicate better relevance to the query.")
    print("=" * 60)


if __name__ == "__main__":
    main()
