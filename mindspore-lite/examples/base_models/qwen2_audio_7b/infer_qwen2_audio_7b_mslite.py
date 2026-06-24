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
Infer Qwen2-Audio-7B with MindSpore Lite using MindIR model.

This script provides text generation inference using the Qwen2-Audio-7B
text LLM component exported as MindIR. It uses a single MindIR model for
greedy token-by-token generation (re-encode the full sequence each step).

Audio capabilities are not included in this inference script.
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


def _mslite_tensor(np_array):
    """Convert numpy array to MindSpore Lite tensor."""
    return mslite.Tensor(np_array)


class Qwen2Audio7BInferencer:
    """Qwen2-Audio-7B text inferencer with MindSpore Lite."""

    def __init__(
        self,
        model_path: str,
        tokenizer_id: str,
        device: str = "ascend",
        device_id: int = 0,
    ):
        if device not in ["cpu", "ascend"]:
            raise ValueError("device must be cpu or ascend")

        print(f"Initializing MindSpore Lite context for {device}...")
        self.context = mslite.Context()
        self.context.target = [device]
        if device == "ascend":
            self.context.ascend.device_id = device_id

        print(f"Loading model from {model_path}...")
        self.model = mslite.Model()
        self.model.build_from_file(
            model_path, mslite.ModelType.MINDIR, self.context
        )

        print(f"Loading tokenizer from {tokenizer_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

    def _tokenize(self, text):
        """Tokenize text using chat template."""
        enc = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="np",
        )
        if isinstance(enc, dict) or (hasattr(enc, "__getitem__") and hasattr(enc, "keys")):
            input_ids = np.array(enc["input_ids"])
            am = enc.get("attention_mask")
            attention_mask = np.array(am) if am is not None else np.ones_like(input_ids)
        elif hasattr(enc, "numpy"):
            input_ids = enc.numpy()
            attention_mask = np.ones_like(input_ids)
        else:
            input_ids = np.array(enc)
            attention_mask = np.ones_like(input_ids)

        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        if attention_mask.ndim == 1:
            attention_mask = attention_mask[None, :]

        return input_ids, attention_mask

    def _run_model(self, input_ids, attention_mask):
        """Run model inference and return logits."""
        inputs = self.model.get_inputs()
        feed_dict = {
            inputs[0].name: _mslite_tensor(input_ids.astype(np.int32)),
            inputs[1].name: _mslite_tensor(attention_mask.astype(np.int32)),
        }
        model_inputs = [feed_dict[inputs[i].name] for i in range(len(inputs))]
        outputs = self.model.predict(model_inputs)
        return outputs[0].get_data_to_numpy()

    def _stream_print_delta(self, generated_ids, prev_text):
        """Print incremental decoded text delta in stream mode."""
        new_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if prev_text and new_text.startswith(prev_text):
            delta = new_text[len(prev_text):]
        else:
            n = min(len(prev_text), len(new_text))
            i = 0
            while i < n and prev_text[i] == new_text[i]:
                i += 1
            delta = new_text[i:]
        if delta:
            delta = delta.replace("�", "")
        if delta:
            print(delta, end="", flush=True)
        return new_text

    def generate(
        self,
        text,
        max_new_tokens=128,
        max_input_length=256,
        stream=True,
    ):
        """Generate text using Qwen2-Audio-7B via greedy token-by-token."""
        input_ids, attention_mask = self._tokenize(text)
        input_ids = input_ids.astype(np.int32)
        attention_mask = attention_mask.astype(np.int32)

        # Truncate if too long
        if input_ids.shape[1] > max_input_length:
            input_ids = input_ids[:, -max_input_length:]
            attention_mask = attention_mask[:, -max_input_length:]

        actual_input_len = int(input_ids.shape[1])
        print(f"\n[Input Info]")
        print(f"  input_ids shape: {input_ids.shape}, dtype={input_ids.dtype}")
        print(f"  actual_input_len: {actual_input_len}")

        # Prefill step
        print(f"\n[Prefill] Running prefill ...", end="", flush=True)
        t0 = time.time()
        logits = self._run_model(input_ids, attention_mask)
        prefill_ms = (time.time() - t0) * 1000
        print(f" done in {prefill_ms:.2f} ms")
        print(f"  logits shape: {logits.shape}, dtype={logits.dtype}")

        # Get first generated token
        generated_ids = [int(np.argmax(logits[0, -1]))]
        streamed_text = ""
        if stream:
            streamed_text = self._stream_print_delta(generated_ids, streamed_text)

        # Decode steps
        print(f"\n[Decode] Running {max_new_tokens - 1} decode steps ...")
        decode_times = []
        for step_i in range(max_new_tokens - 1):
            if self.eos_token_id is not None and generated_ids[-1] == int(
                self.eos_token_id
            ):
                print(f"  [step {step_i}] EOS token reached, stopping early.")
                break

            # Append new token to sequence
            new_token = np.array([[generated_ids[-1]]], dtype=np.int32)
            input_ids = np.concatenate([input_ids, new_token], axis=1)
            attention_mask = np.concatenate(
                [attention_mask, np.ones((1, 1), dtype=np.int32)], axis=1
            )

            t1 = time.time()
            logits = self._run_model(input_ids, attention_mask)
            decode_ms = (time.time() - t1) * 1000
            decode_times.append(decode_ms)

            generated_ids.append(int(np.argmax(logits[0, -1])))
            if stream:
                streamed_text = self._stream_print_delta(generated_ids, streamed_text)

            if step_i == 0 or (step_i + 1) % 32 == 0:
                print(f"  [step {step_i:>3d}] decode={decode_ms:.2f}ms, "
                      f"seq_len={input_ids.shape[1]}")

        if stream:
            print()

        total_decode_ms = sum(decode_times)
        avg_decode_ms = total_decode_ms / len(decode_times) if decode_times else 0
        min_decode_ms = min(decode_times) if decode_times else 0
        max_decode_ms = max(decode_times) if decode_times else 0
        total_ms = prefill_ms + total_decode_ms
        throughput = len(generated_ids) / (total_ms / 1000) if total_ms > 0 else 0

        perf = {
            "prefill_ms": prefill_ms,
            "total_decode_ms": total_decode_ms,
            "avg_decode_ms": avg_decode_ms,
            "min_decode_ms": min_decode_ms,
            "max_decode_ms": max_decode_ms,
            "total_ms": total_ms,
            "generated_tokens": len(generated_ids),
            "throughput_tok_s": throughput,
            "actual_input_len": actual_input_len,
        }

        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return result, perf


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2-Audio-7B Inference with MindSpore Lite"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to .mindir graph file"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="./Qwen2-Audio-7B", help="Tokenizer path"
    )
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument(
        "--device", type=str, default="ascend", choices=["cpu", "ascend"]
    )
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    inferencer = Qwen2Audio7BInferencer(
        model_path=args.model,
        tokenizer_id=args.tokenizer,
        device=args.device,
        device_id=args.device_id,
    )

    print("\n" + "=" * 60)
    print(f"Input Prompt: {args.prompt}")
    print("=" * 60)
    print("Generated Response: ", end="", flush=True)
    result, perf = inferencer.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_input_length=args.max_input_length,
    )

    print("=" * 60)
    print(f"\n[Performance]")
    print(f"  Input Tokens:     {perf['actual_input_len']}")
    print(f"  Output Tokens:    {perf['generated_tokens']}")
    print(f"  Prefill:          {perf['prefill_ms']:.2f} ms")
    print(f"  Total Decode:     {perf['total_decode_ms']:.2f} ms")
    print(f"  Avg Decode:       {perf['avg_decode_ms']:.2f} ms/step")
    print(f"  Min Decode:       {perf['min_decode_ms']:.2f} ms/step")
    print(f"  Max Decode:       {perf['max_decode_ms']:.2f} ms/step")
    print(f"  Total Time:       {perf['total_ms']:.2f} ms")
    print(f"  Throughput:       {perf['throughput_tok_s']:.2f} tok/s")
    print("=" * 60)
    print(f"\n[Full Output Text]")
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()
