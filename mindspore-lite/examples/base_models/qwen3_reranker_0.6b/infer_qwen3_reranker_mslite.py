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
Infer Qwen3-Reranker-0.6B with MindSpore Lite.
"""

import sys
import argparse
import numpy as np

try:
    import mindspore_lite as mslite
    from transformers import AutoTokenizer
except ImportError:
    print("Error: mindspore_lite or transformers package not found.")
    print("Please install them first.")
    sys.exit(1)


class Qwen3RerankerInferencer:
    """
    Qwen3-Reranker-0.6B inferencer.
    """

    def __init__(self, model_path, tokenizer_id, device_id=0, device_type="cpu"):
        """
        Initialize Qwen3-Reranker-0.6B inferencer.
        """
        print(f"Initializing MindSpore Lite context for {device_type}...")

        # Configure context
        self.context = mslite.Context()
        self.context.target = [device_type]
        if device_type == "ascend":
            self.context.ascend.device_id = device_id

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = mslite.Model()
        self.model.build_from_file(model_path, mslite.ModelType.MINDIR, self.context)

        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get token IDs for "yes" and "no"
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        print(f"Token IDs - yes: {self.token_true_id}, no: {self.token_false_id}")

    def _format_instruction(self, instruction, query, doc):
        """
        Format instruction, query and document into input text.
        """
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        output = (
            "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                instruction=instruction, query=query, doc=doc
            )
        )
        return output

    def _prepare_inputs(self, pairs, prefix_tokens, suffix_tokens, max_length):
        """
        Prepare input tensors for inference.
        """
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors="np", max_length=max_length
        )
        return inputs

    def _compute_scores(self, inputs):
        """
        Compute reranking scores from model outputs.
        """
        batch_size = inputs["input_ids"].shape[0]
        scores = []

        for i in range(batch_size):
            mslite_inputs = [
                mslite.Tensor(inputs["input_ids"][i : i + 1].astype(np.int32)),
                mslite.Tensor(inputs["attention_mask"][i : i + 1].astype(np.int32)),
            ]
            outputs = self.model.predict(mslite_inputs)
            logits = outputs[0].get_data_to_numpy()

            last_token_logits = logits[0, -1, :]
            true_score = last_token_logits[self.token_true_id]
            false_score = last_token_logits[self.token_false_id]

            scores_array = np.array([false_score, true_score])
            scores_array = np.exp(scores_array - np.max(scores_array))
            scores_array = scores_array / np.sum(scores_array)
            scores.append(scores_array[1])

        return scores

    def rerank(self, queries, documents, instruction=None, max_length=8192):
        """
        Rerank documents based on queries.
        """
        prefix = (
            "<|im_start|>system\nJudge whether the Document meets "
            "the requirements based on the Query and the Instruct provided. "
            'Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n"
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"

        pairs = [
            self._format_instruction(instruction, query, doc)
            for query, doc in zip(queries, documents)
        ]
        inputs = self._prepare_inputs(pairs, prefix_tokens, suffix_tokens, max_length)
        scores = self._compute_scores(inputs)

        return scores


def main():
    """
    Main function for Qwen3-Reranker-0.6B inference with MindSpore Lite.
    """
    parser = argparse.ArgumentParser(
        description="Qwen3-Reranker-0.6B Inference with MindSpore Lite"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to qwen3_reranker_0.6b.mindir"
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
        "--device", type=str, default="ascend", help="Device for inference (cpu/ascend)"
    )
    parser.add_argument("--device-id", type=int, default=0, help="Device ID for ascend")

    args = parser.parse_args()

    inferencer = Qwen3RerankerInferencer(
        args.model, args.tokenizer, args.device_id, args.device
    )

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

    print("\nRunning reranking inference...")
    scores = inferencer.rerank(queries, documents, max_length=args.max_length)

    print("\nReranking scores:")
    for i, (query, doc, score) in enumerate(zip(queries, documents, scores)):
        print(f"\n[{i + 1}] Score: {score:.4f}")
        print(f"Query: {query}")
        print(f"Document: {doc}")


if __name__ == "__main__":
    main()
