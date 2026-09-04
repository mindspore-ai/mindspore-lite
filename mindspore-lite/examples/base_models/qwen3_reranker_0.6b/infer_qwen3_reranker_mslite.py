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
import time
import argparse
import numpy as np

try:
    import mindspore_lite as mslite
    from transformers import AutoTokenizer
except ImportError:
    print("Error: mindspore_lite or transformers package not found.")
    print("Please install them first.")
    sys.exit(1)

# 动态分档档位（seq_len），需与 converter config.ini 中 ge.dynamicDims 保持一致。
# batch 固定为 1（pointwise reranker），仅 seq_len 分档。
SEQ_BUCKETS = (128, 256, 512, 768, 1024, 1280, 1536, 2048, 3072, 4096, 8192)


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

    @staticmethod
    def _select_bucket(seq_len, max_length):
        """
        Select the smallest bucket >= seq_len, capped by max_length.

        动态分档不会自动 pad：业务侧必须把输入 pad 到某个档位值再 resize。
        """
        for bucket in SEQ_BUCKETS:
            if seq_len <= bucket <= max_length:
                return bucket
        # 超过 max_length 或所有档位都不够时，回退到 max_length（截断保证过）
        return max_length

    def _prepare_inputs(self, pairs, prefix_tokens, suffix_tokens, max_length):
        """
        Tokenize each pair and prepend/append prefix/suffix tokens.

        返回每条样本的真实 token id 列表（未 pad），后续逐条选档 + pad。
        """
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        )
        token_ids_list = []
        for ele in inputs["input_ids"]:
            full_ids = prefix_tokens + ele + suffix_tokens
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]
            token_ids_list.append(full_ids)
        return token_ids_list

    def _run_single(self, token_ids, bucket):
        """
        Left-pad one sample to `bucket`, resize model to the gear, and predict.

        返回该样本的 logits（numpy）。
        """
        pad_len = bucket - len(token_ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        input_ids = np.array([pad_id] * pad_len + list(token_ids), dtype=np.int32)
        attention_mask = np.array([0] * pad_len + [1] * len(token_ids), dtype=np.int32)
        mslite_inputs = [
            mslite.Tensor(input_ids.reshape(1, bucket)),
            mslite.Tensor(attention_mask.reshape(1, bucket)),
        ]
        # resize 需要按输入顺序给出每个输入的完整 shape 列表。
        self.model.resize(mslite_inputs, [[1, bucket], [1, bucket]])
        outputs = self.model.predict(mslite_inputs)
        return outputs[0].get_data_to_numpy()

    def _compute_scores(self, token_ids_list, max_length, timing):
        """
        Compute reranking scores from model outputs (one sample per forward).
        """
        scores = []
        for token_ids in token_ids_list:
            t0 = time.perf_counter()
            bucket = self._select_bucket(len(token_ids), max_length)
            t1 = time.perf_counter()
            logits = self._run_single(token_ids, bucket)
            t2 = time.perf_counter()

            last_token_logits = logits[0, -1, :]
            # Auto-detect lm_head-sliced output ([1,1,2], convention
            # row 0=yes, row 1=no) vs full-vocab output ([1,1,vocab]).
            # Slicing the lm_head weight to the [yes, no] two rows is
            # bit-identical to full-vocab lm_head then indexing those ids,
            # but cuts the lm_head MatMul + weight read + D2H by ~76000x.
            if int(last_token_logits.shape[-1]) == 2:
                true_score = float(last_token_logits[0])   # row 0 = yes
                false_score = float(last_token_logits[1])  # row 1 = no
            else:
                true_score = last_token_logits[self.token_true_id]
                false_score = last_token_logits[self.token_false_id]
            scores_array = np.array([false_score, true_score])
            scores_array = np.exp(scores_array - np.max(scores_array))
            scores_array = scores_array / np.sum(scores_array)
            scores.append(float(scores_array[1]))
            t3 = time.perf_counter()

            timing["select"].append(t1 - t0)
            timing["predict"].append(t2 - t1)
            timing["postprocess"].append(t3 - t2)
            timing["seq_len"].append(len(token_ids))
            timing["bucket"].append(bucket)
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

        t_tok0 = time.perf_counter()
        pairs = [
            self._format_instruction(instruction, query, doc)
            for query, doc in zip(queries, documents)
        ]
        token_ids_list = self._prepare_inputs(
            pairs, prefix_tokens, suffix_tokens, max_length
        )
        t_tok1 = time.perf_counter()

        timing = {
            "tokenize": t_tok1 - t_tok0,
            "select": [],
            "predict": [],
            "postprocess": [],
            "seq_len": [],
            "bucket": [],
        }
        scores = self._compute_scores(token_ids_list, max_length, timing)
        self.last_timing = timing
        return scores


def main():
    """
    Main function for Qwen3-Reranker-0.6B inference with MindSpore Lite.
    """
    parser = argparse.ArgumentParser(
        description="Qwen3-Reranker-0.6B Inference with MindSpore Lite"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to qwen3_reranker_0.6b_graph.mindir"
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
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup rounds to skip (Ascend first-run graph compile)",
    )

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

    # Warmup: Ascend 首次推理会触发图编译，计时前先跑 warmup 轮排除编译开销。
    for _ in range(args.warmup):
        inferencer.rerank(queries, documents, max_length=args.max_length)

    print("\nRunning reranking inference...")
    t_start = time.perf_counter()
    scores = inferencer.rerank(queries, documents, max_length=args.max_length)
    t_total = time.perf_counter() - t_start
    timing = inferencer.last_timing

    print("\nReranking scores:")
    for i, (query, doc, score) in enumerate(zip(queries, documents, scores)):
        print(f"\n[{i + 1}] Score: {score:.4f}")
        print(f"Query: {query}")
        print(f"Document: {doc}")

    print("\n=== 端到端推理性能（本次运行） ===")
    print("| 指标 | 耗时 (ms) |")
    print("|---|---:|")
    print(f"| Tokenize + pad | {timing['tokenize'] * 1000:.2f} |")
    n = len(timing["predict"])
    if n:
        print(
            f"| Bucket 选择 | {sum(timing['select']) * 1000:.2f} |"
        )
        print(
            f"| Model predict (单条平均 {sum(timing['predict']) * 1000 / n:.2f} ms × {n}) | "
            f"{sum(timing['predict']) * 1000:.2f} |"
        )
        print(
            f"| Postprocess | {sum(timing['postprocess']) * 1000:.2f} |"
        )
    print(f"| **总耗时** | **{t_total * 1000:.2f}** |")
    for i in range(n):
        print(
            f"  [sample {i + 1}] seq_len={timing['seq_len'][i]}, "
            f"bucket={timing['bucket'][i]}, predict={timing['predict'][i] * 1000:.2f} ms"
        )


if __name__ == "__main__":
    main()
