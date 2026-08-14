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
"""Qwen3-0.6B MindSpore Lite 推理脚本（zero-copy 优化版本）。
"""

import sys
import argparse
import time
import numpy as np

try:
    import mindspore_lite as mslite
    from transformers import AutoTokenizer
except ImportError:
    print("Error: mindspore_lite or transformers package not found.")
    sys.exit(1)


def _compute_position_ids(attention_mask: np.ndarray) -> np.ndarray:
    position_ids = np.cumsum(attention_mask.astype(np.int32), axis=-1) - 1
    position_ids = np.where(attention_mask > 0, position_ids, 0)
    return position_ids.astype(np.int32)


_DTYPE_MAP = {
    np.dtype("int32"): mslite.DataType.INT32,
    np.dtype("float16"): mslite.DataType.FLOAT16,
    np.dtype("float32"): mslite.DataType.FLOAT32,
}


def _np_to_mslite_dtype(np_dtype):
    return _DTYPE_MAP[np.dtype(np_dtype)]


class Qwen3InferencerZeroCopy:
    """Zero-copy variant of Qwen3Inferencer."""

    def __init__(
        self,
        prefill_model_path: str,
        decode_model_path: str,
        tokenizer_id: str,
        device: str = "ascend",
        device_id: int = 0,
        decode_buckets=None,
        prefill_buckets=None,
    ):
        if device not in ["cpu", "ascend"]:
            raise ValueError("device must be cpu or ascend")
        if device != "ascend":
            raise ValueError("Zero-copy path requires device='ascend'")

        # Tri-state: None = unknown (try pre-alloc first), True/False = cached choice.
        # Some mix-precision builds reject predict(outputs=...) with HW errors;
        # fall back to plain predict (which still reuses input device Tensors).
        self._use_pre_alloc = None

        print(f"Initializing MindSpore Lite context for {device}...")
        self.context = mslite.Context()
        self.context.target = [device]
        self.context.ascend.device_id = device_id
        self.device_id = device_id
        self.device_str = f"ascend:{int(device_id)}"

        print(f"Loading prefill model from {prefill_model_path}...")
        self.prefill_model = mslite.Model()
        self.prefill_model.build_from_file(
            prefill_model_path, mslite.ModelType.MINDIR, self.context
        )

        print(f"Loading decode model from {decode_model_path}...")
        self.decode_model = mslite.Model()
        self.decode_model.build_from_file(
            decode_model_path, mslite.ModelType.MINDIR, self.context
        )

        print(f"Loading tokenizer from {tokenizer_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.decode_buckets: list[int] = (
            sorted(int(b) for b in decode_buckets) if decode_buckets else []
        )
        self.prefill_buckets: list[int] = (
            sorted(int(b) for b in prefill_buckets) if prefill_buckets else []
        )

        # Inspect decode model I/O to get output shapes per bucket
        # (model has dynamic gears; output shape depends on selected gear)
        # We allocate per-bucket I/O buffers lazily as they're needed.
        self._bucket_io_cache = {}

        # Probe whether predict(outputs=...) works on this model. Some
        # mix-precision builds reject pre-allocated outputs with HW errors
        # and leave the model in a dirty state, so the immediate fallback
        # predict produces wrong output. We probe here in __init__ with
        # synthetic data so any dirty state is contained and the real
        # decode loop never has to recover from a probe failure.
        self._probe_pre_alloc_support()

    def _probe_pre_alloc_support(self):
        """Probe predict(outputs=...) support once at init time.

        Uses synthetic data on the smallest bucket. If predict(outputs=)
        fails, sets _use_pre_alloc=False so the real decode loop never
        attempts it. This avoids corrupting real decode state when the
        mix-precision build rejects pre-alloc mid-inference.
        """
        if self._use_pre_alloc is not None:
            # Forced via flag (e.g., --force-no-pre-alloc sets False before
            # this runs).
            return
        if not self.decode_buckets:
            # No buckets → can't probe (no pre-allocated buffers). Default
            # to plain predict; the no-bucket path doesn't go through
            # _get_bucket_io anyway.
            self._use_pre_alloc = False
            return

        smallest_bucket = self.decode_buckets[0]
        io = self._get_bucket_io(smallest_bucket)

        # Synthetic inputs matching the smallest gear's expected shapes.
        syn_input_ids = np.zeros((1, 1), dtype=np.int32)
        syn_attn = np.zeros((1, smallest_bucket + 1), dtype=np.int32)
        syn_attn[0, smallest_bucket] = 1
        syn_pos = np.array([[0]], dtype=np.int32)
        syn_past = np.zeros(
            (56, 1, 8, smallest_bucket, 128), dtype=np.float16
        )

        io["t_input_ids"].set_data_from_numpy(syn_input_ids)
        io["t_attention_mask"].set_data_from_numpy(syn_attn)
        io["t_position_ids"].set_data_from_numpy(syn_pos)
        io["t_past_in"].set_data_from_numpy(syn_past)

        inputs = [
            io["t_input_ids"],
            io["t_attention_mask"],
            io["t_position_ids"],
            io["t_past_in"],
        ]
        try:
            self.decode_model.predict(inputs, outputs=io["out_bufs"])
            self._use_pre_alloc = True
            print("[zero-copy] pre-allocated outputs: enabled (probe OK)")
        except (RuntimeError, ValueError) as e:
            self._use_pre_alloc = False
            print(
                f"[zero-copy] pre-allocated outputs disabled (probe failed: "
                f"{type(e).__name__}: {e!s:.120s}); using plain predict"
            )

    def _next_bucket(self, cur_len: int) -> int:
        if not self.decode_buckets:
            return cur_len
        for b in self.decode_buckets:
            if b >= cur_len:
                return b
        raise ValueError(
            f"cur_len={cur_len} exceeds max bucket {self.decode_buckets[-1]}"
        )

    def _get_bucket_io(self, bucket_kv_len: int):
        """Return pre-allocated device Tensor set for a given bucket.

        Returns dict with:
          t_input_ids, t_attention_mask, t_position_ids,
          t_past_in, t_past_out,
          out_bufs (=[t_logits, t_past_out])
        """
        if bucket_kv_len in self._bucket_io_cache:
            return self._bucket_io_cache[bucket_kv_len]

        amask_len = bucket_kv_len + 1
        device_str = self.device_str

        # Small inputs
        t_input_ids = mslite.Tensor(
            shape=[1, 1], dtype=mslite.DataType.INT32, device=device_str
        )
        t_attention_mask = mslite.Tensor(
            shape=[1, amask_len], dtype=mslite.DataType.INT32, device=device_str
        )
        t_position_ids = mslite.Tensor(
            shape=[1, 1], dtype=mslite.DataType.INT32, device=device_str
        )

        # KV cache input: (56, 1, 8, bucket_kv_len, 128) float16
        t_past_in = mslite.Tensor(
            shape=[56, 1, 8, bucket_kv_len, 128],
            dtype=mslite.DataType.FLOAT16,
            device=device_str,
        )

        # KV cache output: (56, 1, 8, bucket_kv_len + 1, 128) float16
        # (model appends one new K/V slot)
        t_past_out = mslite.Tensor(
            shape=[56, 1, 8, bucket_kv_len + 1, 128],
            dtype=mslite.DataType.FLOAT16,
            device=device_str,
        )

        # logits output shape: (1, 1, vocab_size) float16
        # Get from model's get_outputs() (which returns dynamic shapes; we use
        # a generic shape and let predict resize).
        # Use a placeholder; predict will write into it.
        # Actually we need correct shape. Use the model's output info.
        outs_info = self.decode_model.get_outputs()
        # outs_info[0] = logits; shape may have -1 for dynamic dims
        logits_shape = [int(x) if int(x) > 0 else 1 for x in outs_info[0].shape]
        # Override seq dim to 1 (decode) and vocab from info
        # Typical: [1, 1, vocab_size]
        if len(logits_shape) == 3:
            logits_shape = [1, 1, logits_shape[2]]
        elif len(logits_shape) == 2:
            logits_shape = [1, logits_shape[1]]
        else:
            logits_shape = [1, 1, logits_shape[-1]]
        t_logits = mslite.Tensor(
            shape=logits_shape,
            dtype=mslite.DataType.FLOAT16,
            device=device_str,
        )

        io = {
            "t_input_ids": t_input_ids,
            "t_attention_mask": t_attention_mask,
            "t_position_ids": t_position_ids,
            "t_past_in": t_past_in,
            "t_past_out": t_past_out,
            "t_logits": t_logits,
            "out_bufs": [t_logits, t_past_out],
        }
        self._bucket_io_cache[bucket_kv_len] = io
        return io

    def _predict(self, inputs, out_bufs):
        """Dispatch to predict based on cached pre-alloc support.

        _use_pre_alloc is set once during __init__ via _probe_pre_alloc_support
        (or forced via --force-no-pre-alloc). The dispatch is plain: by the
        time we get here, the model is in a known clean state.
        """
        if self._use_pre_alloc:
            return self.decode_model.predict(inputs, outputs=out_bufs)
        return self.decode_model.predict(inputs)

    def _prepare_inputs(self, text: str, max_length: int, use_chat_template: bool = True):
        """Tokenize ``text`` into input_ids/attention_mask/position_ids tensors for prefill."""
        if (
            use_chat_template
            and hasattr(self.tokenizer, "apply_chat_template")
            and getattr(self.tokenizer, "chat_template", None)
        ):
            enc = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="np",
            )
            if hasattr(enc, "keys") and "input_ids" in enc:
                input_ids = enc["input_ids"]
                attention_mask = enc.get("attention_mask", np.ones_like(input_ids))
            else:
                input_ids = enc
                attention_mask = np.ones_like(input_ids)
        else:
            enc = self.tokenizer(
                text,
                return_tensors="np",
                padding=False,
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", np.ones_like(input_ids))
        input_ids = input_ids.astype(np.int32)
        attention_mask = attention_mask.astype(np.int32)
        position_ids = _compute_position_ids(attention_mask)
        return input_ids, attention_mask, position_ids

    def generate(
        self,
        text: str,
        max_new_tokens: int = 128,
        max_length: int = 2048,
        use_chat_template: bool = True,
    ):
        """Run prefill + autoregressive decode and return the generated text."""
        input_ids, attention_mask, position_ids = self._prepare_inputs(
            text, max_length, use_chat_template
        )

        # Pad prefill inputs to the nearest prefill bucket boundary. The bucketed
        # MindIR is AOT-compiled for specific seq_lens; runtime only accepts
        # shapes that match one of the ge.dynamicDims entries exactly.
        actual_seq_len = int(input_ids.shape[1])
        if self.prefill_buckets:
            target_seq_len = actual_seq_len
            for b in self.prefill_buckets:
                if b >= actual_seq_len:
                    target_seq_len = b
                    break
            if target_seq_len < actual_seq_len:
                raise ValueError(
                    f"prompt seq_len={actual_seq_len} exceeds max prefill "
                    f"bucket {self.prefill_buckets[-1]}"
                )
            pad_len = target_seq_len - actual_seq_len
            if pad_len > 0:
                pad_token = int(self.tokenizer.pad_token_id)
                input_ids = np.concatenate(
                    [
                        input_ids,
                        np.full((1, pad_len), pad_token, dtype=np.int32),
                    ],
                    axis=1,
                )
                attention_mask = np.concatenate(
                    [attention_mask, np.zeros((1, pad_len), dtype=np.int32)],
                    axis=1,
                )
                position_ids = np.concatenate(
                    [position_ids, np.zeros((1, pad_len), dtype=np.int32)],
                    axis=1,
                )
            print(
                f"[prefill] seq_len={actual_seq_len} → bucket={target_seq_len} "
                f"(pad {pad_len})"
            )

        # Prefill (host-side, single shot)
        prefill_inputs = [
            mslite.Tensor(input_ids),
            mslite.Tensor(attention_mask),
            mslite.Tensor(position_ids),
        ]
        print("Running LLM prefill...")
        t0 = time.time()
        prefill_outputs = self.prefill_model.predict(prefill_inputs)
        logits = prefill_outputs[0].get_data_to_numpy()
        past_kv = prefill_outputs[1].get_data_to_numpy()  # numpy on host
        prefill_ms = (time.time() - t0) * 1000
        print(f"Prefill time: {prefill_ms:.2f} ms")

        # When inputs were padded, slice outputs back to the actual seq_len so
        # decode starts from the real prompt boundary, not the padding tail.
        if actual_seq_len != int(input_ids.shape[1]):
            past_kv = past_kv[:, :, :, :actual_seq_len, :]

        generated_ids = []
        next_token = int(np.argmax(logits[0, actual_seq_len - 1]))
        generated_ids.append(next_token)

        cur_pos = int(position_ids[0, actual_seq_len - 1])

        print("Running LLM decode...")
        decode_times = []
        for _ in range(max_new_tokens - 1):
            if self.eos_token_id is not None and generated_ids[-1] == int(
                self.eos_token_id
            ):
                break

            next_input_ids = np.array([[generated_ids[-1]]], dtype=np.int32)
            next_position_ids = np.array([[cur_pos + 1]], dtype=np.int32)

            if self.decode_buckets:
                cur_kv_len = past_kv.shape[3]
                target_len = self._next_bucket(cur_kv_len)
                if target_len > cur_kv_len:
                    pad = np.zeros(
                        (
                            past_kv.shape[0],
                            past_kv.shape[1],
                            past_kv.shape[2],
                            target_len - cur_kv_len,
                            past_kv.shape[4],
                        ),
                        dtype=past_kv.dtype,
                    )
                    past_kv_in = np.concatenate([past_kv, pad], axis=3)
                else:
                    past_kv_in = past_kv
                attn_in = np.zeros((1, target_len + 1), dtype=np.int32)
                attn_in[0, :cur_kv_len] = 1
                attn_in[0, target_len] = 1
            else:
                target_len = past_kv.shape[3]
                past_kv_in = past_kv
                attn_in = np.ones((1, target_len + 1), dtype=np.int32)

            io = self._get_bucket_io(target_len)

            # Update small inputs
            io["t_input_ids"].set_data_from_numpy(next_input_ids)
            io["t_attention_mask"].set_data_from_numpy(attn_in)
            io["t_position_ids"].set_data_from_numpy(next_position_ids)
            # Push past_kv_in (host→device; this is the unavoidable copy in v1)
            io["t_past_in"].set_data_from_numpy(past_kv_in)

            inputs = [
                io["t_input_ids"],
                io["t_attention_mask"],
                io["t_position_ids"],
                io["t_past_in"],
            ]
            # Reuse pre-allocated output buffers (fall back to no-pre-alloc if
            # the model doesn't support it, e.g. some mix-precision builds)
            t_step = time.time()
            decode_outputs = self._predict(inputs, io["out_bufs"])
            decode_times.append((time.time() - t_step) * 1000)

            # Copy only logits (small) back to host
            logits = decode_outputs[0].get_data_to_numpy()
            new_past_kv = decode_outputs[1].get_data_to_numpy()

            if self.decode_buckets and new_past_kv.shape[3] != past_kv.shape[3] + 1:
                tail_idx = past_kv_in.shape[3]
                new_kv = new_past_kv[:, :, :, tail_idx : tail_idx + 1, :]
                past_kv = np.concatenate([past_kv, new_kv], axis=3)
            else:
                past_kv = new_past_kv
            cur_pos += 1

            generated_ids.append(int(np.argmax(logits[0, -1])))

        total_decode_ms = sum(decode_times)
        avg_decode_ms = (
            total_decode_ms / len(decode_times) if decode_times else 0.0
        )
        print(
            f"Total decode time: {total_decode_ms:.2f} ms, "
            f"avg decode step: {avg_decode_ms:.2f} ms, steps: {len(decode_times)}"
        )

        total_ms = prefill_ms + total_decode_ms
        throughput = (
            len(generated_ids) / (total_ms / 1000.0) if total_ms > 0 else 0.0
        )
        print(
            f"Total time: {total_ms:.2f} ms, throughput: {throughput:.2f} tok/s"
        )

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-0.6B Inference (zero-copy decode) with split MindIR"
    )
    parser.add_argument("--prefill-model", type=str, required=True)
    parser.add_argument("--decode-model", type=str, required=True)
    parser.add_argument(
        "--tokenizer", type=str, default="Qwen/Qwen3-0.6B-Instruct"
    )
    parser.add_argument(
        "--prompt", type=str, default="你好，请介绍一下你自己。"
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--device", type=str, default="ascend")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--decode-buckets",
        type=str,
        default="16,32,64,96,128,192,256,384,512,768,1024,1536,2048",
    )
    parser.add_argument(
        "--prefill-buckets",
        type=str,
        default="",
        help="Prefill seq_len buckets (comma-separated). Required when the "
             "prefill MindIR is built with ge.dynamicDims (bucketed); empty "
             "for pure-dynamic prefill MindIR.",
    )
    parser.add_argument(
        "--force-no-pre-alloc",
        action="store_true",
        help="Skip the pre-alloc probe; use plain predict (some mix-precision builds need this).",
    )
    args = parser.parse_args()

    decode_buckets = [int(b) for b in args.decode_buckets.split(",") if b]
    prefill_buckets = [int(b) for b in args.prefill_buckets.split(",") if b]

    inferencer = Qwen3InferencerZeroCopy(
        prefill_model_path=args.prefill_model,
        decode_model_path=args.decode_model,
        tokenizer_id=args.tokenizer,
        device=args.device,
        device_id=args.device_id,
        decode_buckets=decode_buckets,
        prefill_buckets=prefill_buckets,
    )
    if args.force_no_pre_alloc:
        inferencer._use_pre_alloc = False
        print("[zero-copy] pre-alloc disabled by --force-no-pre-alloc flag")

    print("\n" + "=" * 60)
    print(f"Input Prompt: {args.prompt}")
    print("=" * 60)

    result = inferencer.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        use_chat_template=not args.no_chat_template,
    )

    print("\n" + "=" * 60)
    print(f"Generated Response: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
