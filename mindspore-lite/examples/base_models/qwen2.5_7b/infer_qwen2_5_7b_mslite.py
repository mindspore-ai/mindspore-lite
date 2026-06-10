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
Infer Qwen2.5-7B with PyTorch prefill + MindSpore Lite decode.

For 7B+ models the prefill ONNX exceeds the GE protobuf 2GB limit,
so prefill runs via PyTorch (CPU) and decode runs on Ascend via MindSpore Lite.
"""

import argparse
import sys
import time

import numpy as np
import torch

try:
    import mindspore_lite as mslite
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: mindspore_lite, transformers or torch package not found.")
    print("Please install them first.")
    sys.exit(1)

KV_CACHE_LEN = 512


def _compute_position_ids(attention_mask):
    """Compute position ids from attention mask."""
    position_ids = np.cumsum(attention_mask.astype(np.int32), axis=-1) - 1
    position_ids = np.where(attention_mask > 0, position_ids, 0)
    return position_ids.astype(np.int32)


# MindSpore Lite DataType -> numpy dtype mapping
_MS_DTYPE_TO_NP = {
    mslite.DataType.FLOAT16: np.float16,
    mslite.DataType.FLOAT32: np.float32,
    mslite.DataType.FLOAT64: np.float64,
    mslite.DataType.INT32: np.int32,
    mslite.DataType.INT64: np.int64,
    mslite.DataType.BOOL: np.bool_,
}


def _mslite_tensor(np_array, target_dtype=None):
    """Convert numpy array to MindSpore Lite tensor, casting dtype if needed."""
    if target_dtype is not None:
        np_dtype = _MS_DTYPE_TO_NP.get(target_dtype)
        if np_dtype is not None and np_array.dtype != np_dtype:
            np_array = np_array.astype(np_dtype)
    return mslite.Tensor(np_array)


def _build_mslite_inputs(model: mslite.Model, feed_dict, preferred_order=None):
    """Build MindSpore Lite model inputs, auto-casting dtypes to match model expectations."""
    inputs = model.get_inputs()
    if not inputs:
        if preferred_order:
            return [_mslite_tensor(feed_dict[k]) for k in preferred_order]
        return [_mslite_tensor(v) for v in feed_dict.values()]

    ok_by_name = True
    for t in inputs:
        name = getattr(t, "name", None)
        if name is None or name not in feed_dict:
            ok_by_name = False
            break
    if ok_by_name:
        return [_mslite_tensor(feed_dict[t.name], t.dtype) for t in inputs]

    if preferred_order:
        # Build a dtype map from preferred_order index when names don't match
        dtype_map = {}
        for idx, key in enumerate(preferred_order):
            if idx < len(inputs):
                dtype_map[key] = inputs[idx].dtype
        return [_mslite_tensor(feed_dict[k], dtype_map.get(k)) for k in preferred_order]

    raise RuntimeError(
        f"input mismatch. model inputs={[getattr(x, 'name', '') for x in inputs]} "
        f"feed keys={list(feed_dict.keys())}"
    )


class Qwen257BInferencer:
    """
    Qwen2.5-7B inferencer: PyTorch prefill (CPU) + MindSpore Lite decode (Ascend).
    """

    def __init__(
        self,
        model_id: str,
        decode_model_path: str,
        device: str = "ascend",
        device_id: int = 0,
        torch_dtype: str = "float16",
    ):
        """
        Initialize Qwen2.5-7B inferencer.
        """
        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        torch_dtype_val = dtype_map.get(torch_dtype, torch.float16)

        print(f"Loading PyTorch model from {model_id} (dtype={torch_dtype}) for prefill...")
        self.torch_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype_val, low_cpu_mem_usage=True, attn_implementation="eager"
        )
        self.torch_model.eval()

        print(f"Loading tokenizer from {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

        print(f"Initializing MindSpore Lite context for decode ({device})...")
        self.decode_context = mslite.Context()
        self.decode_context.target = [device]
        if device == "ascend":
            self.decode_context.ascend.device_id = device_id

        print(f"Loading decode model from {decode_model_path}...")
        self.decode_model = mslite.Model()
        self.decode_model.build_from_file(
            decode_model_path, mslite.ModelType.MINDIR, self.decode_context
        )
        print("All models loaded successfully.")

    def _stream_print_delta(self, generated_ids, prev_text: str):
        """Print incremental decoded text delta in stream mode."""
        new_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
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

    @torch.no_grad()
    def _prefill_pytorch(self, input_ids_np, attention_mask_np):
        """
        Run prefill with PyTorch model and extract KV cache.
        Returns (logits_np, past_k_np, past_v_np).
        """
        device = next(self.torch_model.parameters()).device
        input_ids_t = torch.from_numpy(input_ids_np).to(device)
        attention_mask_t = torch.from_numpy(attention_mask_np).to(device)

        outputs = self.torch_model.model(
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            use_cache=True,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.torch_model.lm_head(hidden_states)

        # Extract KV cache and pad to KV_CACHE_LEN
        past_k_list = []
        past_v_list = []
        for layer_past in outputs.past_key_values:
            k = layer_past[0]  # (batch, num_kv_heads, seq_len, head_dim)
            v = layer_past[1]
            seq_len = k.shape[2]
            head_dim = k.shape[3]
            pad_len = KV_CACHE_LEN - seq_len
            if pad_len > 0:
                k = torch.cat([k, k.new_zeros(k.shape[0], k.shape[1], pad_len, head_dim)], dim=2)
                v = torch.cat([v, v.new_zeros(v.shape[0], v.shape[1], pad_len, head_dim)], dim=2)
            past_k_list.append(k[:, :, :KV_CACHE_LEN, :])
            past_v_list.append(v[:, :, :KV_CACHE_LEN, :])

        past_k = torch.stack(past_k_list, dim=0).cpu().numpy()  # (num_layers, batch, num_kv_heads, KV_CACHE_LEN, head_dim)
        past_v = torch.stack(past_v_list, dim=0).cpu().numpy()
        logits_np = logits[:, -1, :].cpu().numpy()  # (batch, vocab_size) - last token logits

        return logits_np, past_k, past_v

    def generate(
        self,
        text: str,
        max_new_tokens: int = 128,
        stream: bool = True,
    ):
        """
        Generate text. Returns (decoded_text, perf_dict).
        """
        enc = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True, add_generation_prompt=True, return_tensors="np",
        )
        if hasattr(enc, "keys"):
            input_ids = np.asarray(enc["input_ids"], dtype=np.int32)
            attention_mask = np.asarray(enc.get("attention_mask", np.ones_like(input_ids)), dtype=np.int32)
        elif isinstance(enc, np.ndarray):
            input_ids = enc.astype(np.int32)
            attention_mask = np.ones_like(input_ids, dtype=np.int32)
        else:
            input_ids = np.asarray(enc, dtype=np.int32)
            attention_mask = np.ones_like(input_ids, dtype=np.int32)

        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        if attention_mask.ndim == 1:
            attention_mask = attention_mask[None, :]

        input_ids = input_ids.astype(np.int32)
        attention_mask = attention_mask.astype(np.int32)

        # --- Prefill via PyTorch (CPU) ---
        t0 = time.perf_counter()
        logits_np, past_k, past_v = self._prefill_pytorch(input_ids, attention_mask)
        t_prefill = time.perf_counter() - t0

        actual_len = int(attention_mask[0].sum())
        print(f"[Prefill done: {actual_len} input tokens, {t_prefill:.3f}s]")

        # First generated token from prefill logits
        generated_ids = [int(np.argmax(logits_np[0]))]
        streamed_text = ""
        if stream:
            streamed_text = self._stream_print_delta(generated_ids, streamed_text)

        cur_attention_mask = np.zeros((1, KV_CACHE_LEN), dtype=np.int32)
        if actual_len > 0:
            cur_attention_mask[0, :actual_len] = 1
        valid_len = int(actual_len)

        # --- Decode loop via MindSpore Lite (Ascend) ---
        decode_times = []
        for _ in range(max_new_tokens - 1):
            if self.eos_token_id is not None and generated_ids[-1] == int(self.eos_token_id):
                break
            if valid_len >= KV_CACHE_LEN:
                break

            next_input_ids = np.array([[generated_ids[-1]]], dtype=np.int32)
            cur_attention_mask[0, valid_len] = 1
            next_position_ids = np.array([[valid_len]], dtype=np.int32)

            decode_feed = {
                "input_ids": next_input_ids,
                "attention_mask": cur_attention_mask,
                "position_ids": next_position_ids,
                "past_key_cache": past_k,
                "past_value_cache": past_v,
            }
            inputs = _build_mslite_inputs(
                self.decode_model, decode_feed,
                preferred_order=["input_ids", "attention_mask", "position_ids",
                                 "past_key_cache", "past_value_cache"],
            )
            td0 = time.perf_counter()
            decode_outputs = self.decode_model.predict(inputs)
            decode_times.append(time.perf_counter() - td0)

            logits = decode_outputs[0].get_data_to_numpy()
            past_k = decode_outputs[1].get_data_to_numpy()
            past_v = decode_outputs[2].get_data_to_numpy()
            valid_len += 1
            generated_ids.append(int(np.argmax(logits[0, -1])))
            if stream:
                streamed_text = self._stream_print_delta(generated_ids, streamed_text)

        if stream:
            print()

        total_decode = sum(decode_times)
        avg_decode = total_decode / len(decode_times) if decode_times else 0
        perf = {
            "prefill_ms": t_prefill * 1000,
            "total_decode_ms": total_decode * 1000,
            "avg_decode_ms": avg_decode * 1000,
            "total_ms": (t_prefill + total_decode) * 1000,
            "num_decode_steps": len(decode_times),
            "input_len": actual_len,
            "output_len": len(generated_ids),
        }
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True), perf


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Qwen2.5-7B Inference: PyTorch prefill + MindSpore Lite decode"
    )
    parser.add_argument("--model-id", type=str, default="./Qwen2.5-7B-Instruct",
                        help="HuggingFace model path for PyTorch prefill")
    parser.add_argument("--decode-model", type=str, required=True,
                        help="Path to decode .mindir")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer path (defaults to model-id)")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--device", type=str, default="ascend", choices=["cpu", "ascend"])
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--torch-dtype", type=str, default="float16",
                        choices=["float16", "float32", "bfloat16"])
    args = parser.parse_args()
    inferencer = Qwen257BInferencer(
        model_id=args.model_id,
        decode_model_path=args.decode_model,
        device=args.device,
        device_id=args.device_id,
        torch_dtype=args.torch_dtype,
    )

    print("\n" + "=" * 60)
    print(f"Input Prompt: {args.prompt}")
    print("=" * 60)
    print("Generated Response: ", end="", flush=True)
    _, perf = inferencer.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    print("=" * 60)
    print("\n--- Performance ---")
    print(f"  Input tokens:     {perf['input_len']}")
    print(f"  Output tokens:    {perf['output_len']}")
    print(f"  Prefill (PyTorch CPU): {perf['prefill_ms']:.2f} ms")
    print(f"  Total Decode (Ascend): {perf['total_decode_ms']:.2f} ms")
    print(f"  Avg decode step:  {perf['avg_decode_ms']:.2f} ms")
    print(f"  Total time:       {perf['total_ms']:.2f} ms")
    if perf['output_len'] > 1 and perf['total_decode_ms'] > 0:
        throughput = (perf['output_len'] - 1) / (perf['total_decode_ms'] / 1000)
        print(f"  Decode throughput: {throughput:.1f} tok/s")


if __name__ == "__main__":
    main()
