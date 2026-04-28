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
Export Jina Reranker V3 to ONNX format (Listwise + Pointwise).

This script exports the JinaForRanking model (built on Qwen3-0.6B) with its
native "last but not late interaction" architecture. The ONNX model supports
both listwise (multiple docs per forward) and pointwise (one doc per forward)
scoring modes via the same graph.

Key design: instead of boolean indexing to find special token positions (which
is not ONNX-exportable with dynamic shapes), we pre-compute the positions of
<|embed_token|> and <|rerank_token|> and pass them as explicit inputs. The
model uses torch.gather to extract hidden states at those positions, then
projects them through the MLP projector and computes cosine similarity scores.
"""

import argparse
import os

import torch
import torch.nn.functional as F
import onnx
from transformers import AutoModel, AutoTokenizer


DOC_EMBED_TOKEN_ID = 151670
QUERY_EMBED_TOKEN_ID = 151671
MAX_DOCS = 64

THINK_OPEN = "\u003cthink\u003e"
THINK_CLOSE = "\u003c/think\u003e"
NO_THINK_SUFFIX = THINK_OPEN + "\n\n" + THINK_CLOSE + "\n\n"


class JinaRerankerV3ListwiseWrapper(torch.nn.Module):
    """
    Wrapper for Jina Reranker V3 listwise ONNX export.

    Uses torch.gather instead of boolean indexing to extract hidden states at
    special token positions, making the graph ONNX-exportable with dynamic
    number of documents.
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.model
        self.projector = model.projector

    def forward(self, input_ids, attention_mask, doc_token_indices, query_token_index):
        """
        Forward pass for listwise reranking.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) attention mask
            doc_token_indices: (batch, num_docs) positions of <|embed_token|>
            query_token_index: (batch, 1) position of <|rerank_token|>

        Returns:
            scores: (batch, num_docs) cosine similarity scores
        """
        seq_len = input_ids.shape[1]
        cache_position = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state
        _, _, dim = hidden_states.shape

        doc_idx = doc_token_indices.unsqueeze(-1).expand(-1, -1, dim)
        doc_embeds = torch.gather(hidden_states, 1, doc_idx)

        query_idx = query_token_index.unsqueeze(-1).expand(-1, -1, dim)
        query_embeds = torch.gather(hidden_states, 1, query_idx)

        doc_embeds = self.projector(doc_embeds)
        query_embeds = self.projector(query_embeds)

        query_embeds_expanded = query_embeds.expand_as(doc_embeds)
        scores = F.cosine_similarity(doc_embeds, query_embeds_expanded, dim=-1)

        return scores


def _parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Export Jina Reranker V3 (Listwise) to ONNX"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="jinaai/jina-reranker-v3",
        help="Model ID or local path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./onnx",
        help="Output directory for ONNX model",
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
        default="cpu",
        help="Device to use (cpu/cuda)",
    )
    return parser.parse_args()


def _load_model_and_tokenizer(args):
    """
    Load JinaForRanking model and tokenizer.

    The model is loaded in float32 to ensure the exported ONNX model uses
    float32 throughout. Loading in float16 would cause the ONNX graph to
    have float16 type declarations, which MindSpore Lite's Clip parser and
    other operators do not support.
    """
    print(f"Loading model from {args.model_id}")
    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    return model, tokenizer


def _format_listwise_prompt(query, docs):
    """
    Format query and documents into the listwise prompt template.

    Mirrors the format_docs_prompts_func from the original modeling.py.
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


def _create_dummy_inputs(args, tokenizer):
    """
    Create dummy inputs for ONNX export with listwise prompt format.
    """
    query = "What is the capital of China?"
    docs = [
        "The capital of China is Beijing.",
        "China has many large cities.",
        "Beijing is the political center of China.",
    ]

    prompt = _format_listwise_prompt(query, docs)
    encoded = tokenizer(
        prompt,
        padding=False,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    doc_token_indices = _find_token_positions(input_ids, DOC_EMBED_TOKEN_ID)
    query_token_index = _find_token_positions(input_ids, QUERY_EMBED_TOKEN_ID)

    return input_ids, attention_mask, doc_token_indices, query_token_index


def _find_token_positions(input_ids, token_id):
    """
    Find positions of a specific token ID in the input sequence.

    Args:
        input_ids: (batch, seq_len) token IDs
        token_id: token ID to find

    Returns:
        positions: (batch, num_occurrences) positions of the token
    """
    positions = (input_ids == token_id).nonzero(as_tuple=True)
    batch_indices = positions[0]
    seq_indices = positions[1]

    result = torch.zeros(
        1, MAX_DOCS if token_id == DOC_EMBED_TOKEN_ID else 1,
        dtype=torch.long,
    )
    count = 0
    for i, batch_idx in enumerate(batch_indices):
        if batch_idx == 0:
            col_idx = count if token_id == DOC_EMBED_TOKEN_ID else 0
            if col_idx < result.shape[1]:
                result[0, col_idx] = seq_indices[i]
                count += 1

    return result


def _export_to_onnx(model, output_path, dummy_inputs):
    """
    Export model to ONNX format using TorchScript-based tracing.

    The newer transformers (v4.50+) changed Qwen3's mask creation to use
    symbolic shape values during JIT tracing, which breaks the backward-
    compatibility check in masking_utils.sdpa_mask. We work around this by
    monkey-patching _preprocess_mask_arguments to return concrete integers.
    """
    import transformers.masking_utils as _mu

    _orig_preprocess = _mu._preprocess_mask_arguments

    def _patched_preprocess(*args, **kwargs):
        early_exit, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset = (
            _orig_preprocess(*args, **kwargs)
        )
        if isinstance(q_length, torch.Tensor) and q_length.dim() == 0:
            q_length = int(q_length.item())
        if isinstance(kv_length, torch.Tensor) and kv_length.dim() == 0:
            kv_length = int(kv_length.item())
        if isinstance(q_offset, torch.Tensor) and q_offset.dim() == 0:
            q_offset = int(q_offset.item())
        if isinstance(kv_offset, torch.Tensor) and kv_offset.dim() == 0:
            kv_offset = int(kv_offset.item())
        return early_exit, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset

    _mu._preprocess_mask_arguments = _patched_preprocess

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dummy_input_ids, dummy_attention_mask, dummy_doc_indices, dummy_query_idx = dummy_inputs

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "doc_token_indices": {0: "batch_size", 1: "num_docs"},
        "query_token_index": {0: "batch_size"},
        "scores": {0: "batch_size", 1: "num_docs"},
    }
    input_names = ["input_ids", "attention_mask", "doc_token_indices", "query_token_index"]
    output_names = ["scores"]

    print(f"Exporting model to {output_path}")
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_input_ids, dummy_attention_mask, dummy_doc_indices, dummy_query_idx),
                output_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
    finally:
        _mu._preprocess_mask_arguments = _orig_preprocess

    print(f"Model exported successfully to {output_path}")


def _optimize_onnx_model(onnx_model):
    """
    Optimize ONNX model by removing IsNaN + Where patterns.
    """
    isnan_nodes = [node for node in onnx_model.graph.node if node.op_type == "IsNaN"]

    if not isnan_nodes:
        print("No IsNaN nodes found in the model")
        return onnx_model

    print(
        f"Found {len(isnan_nodes)} IsNaN nodes, replacing "
        "Where(IsNaN(x), default, x) patterns with Identity(x)..."
    )

    nodes_to_remove_names = set()
    nodes_to_add = []

    for isnan_node in isnan_nodes:
        isnan_output = isnan_node.output[0]
        where_nodes = [
            node
            for node in onnx_model.graph.node
            if node.op_type == "Where" and isnan_output in node.input
        ]
        print(f"  IsNaN node {isnan_node.name}: found {len(where_nodes)} Where consumers")

        for where_node in where_nodes:
            where_inputs = where_node.input
            if len(where_inputs) == 3 and where_inputs[0] == isnan_output:
                where_output = where_node.output[0]
                value_if_false = where_inputs[2]
                identity_node = onnx.helper.make_node(
                    "Identity",
                    inputs=[value_if_false],
                    outputs=[where_output],
                    name=where_node.name + "_identity",
                )
                nodes_to_add.append(identity_node)
                nodes_to_remove_names.add(where_node.name)
                print(f"    Will replace Where node {where_node.name} with Identity")

        nodes_to_remove_names.add(isnan_node.name)
        print(f"  Will remove IsNaN node {isnan_node.name}")

    print(
        f"Removing {len(nodes_to_remove_names)} nodes and "
        f"adding {len(nodes_to_add)} nodes..."
    )

    new_nodes = [
        node for node in onnx_model.graph.node if node.name not in nodes_to_remove_names
    ]
    new_nodes.extend(nodes_to_add)

    onnx_model.graph.ClearField("node")
    onnx_model.graph.node.extend(new_nodes)

    print(f"Successfully replaced {len(isnan_nodes)} IsNaN nodes")
    remaining_isnan = sum(1 for node in onnx_model.graph.node if node.op_type == "IsNaN")
    print(f"Remaining IsNaN nodes after removal: {remaining_isnan}")

    return onnx_model


def main():
    """
    Main function to export Jina Reranker V3 model to ONNX format.
    """
    args = _parse_args()
    model, tokenizer = _load_model_and_tokenizer(args)

    print("Preparing model for export...")
    wrapper = JinaRerankerV3ListwiseWrapper(model).to(args.device).eval()

    print("Creating dummy inputs with listwise prompt format...")
    dummy_inputs = _create_dummy_inputs(args, tokenizer)
    dummy_input_ids = dummy_inputs[0].to(args.device)
    dummy_attention_mask = dummy_inputs[1].to(args.device)
    dummy_doc_indices = dummy_inputs[2].to(args.device)
    dummy_query_idx = dummy_inputs[3].to(args.device)
    dummy_inputs_device = (dummy_input_ids, dummy_attention_mask, dummy_doc_indices, dummy_query_idx)

    output_path = os.path.join(args.output_dir, "jina_reranker_v3_listwise.onnx")
    _export_to_onnx(wrapper, output_path, dummy_inputs_device)

    print("Optimizing ONNX model...")
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx_model = _optimize_onnx_model(onnx_model)

    onnx_path = os.path.join(args.output_dir, "jina_reranker_v3_listwise.onnx")
    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)
    onnx.save_model(
        onnx_model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="jina_reranker_v3_listwise.onnx.data",
        size_threshold=1024,
        convert_attribute=True,
    )
    print(f"Optimized model saved to {output_path}")

    print("\nExport completed successfully!")
    print(f"ONNX model saved to: {output_path}")
    print(f"Max sequence length: {args.max_length}")
    print(f"Max documents per query: {MAX_DOCS}")


if __name__ == "__main__":
    main()
