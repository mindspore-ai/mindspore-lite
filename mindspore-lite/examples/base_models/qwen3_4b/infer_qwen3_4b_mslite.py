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
Infer Qwen3-4B with MindSpore Lite using split MindIR (prefill + decode).
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

KV_CACHE_LEN = 512
PREFILL_GEARS = [64, 128, 256]


def _select_prefill_gear(seq_len):
    """Select the nearest prefill gear size >= seq_len within configured gears."""
    seq_len = int(seq_len)
    for gear in PREFILL_GEARS:
        if seq_len <= gear:
            return gear
    return PREFILL_GEARS[-1]


def _compute_position_ids(attention_mask):
    """Compute position ids from an attention mask via cumulative sum.

    Parameters
    ----------
    attention_mask : numpy.ndarray
        A 0/1 integer array of shape ``(batch, seq_len)`` where 1 indicates
        a valid (non-padded) token position.

    Returns
    -------
    numpy.ndarray
        An ``int32`` array of the same shape as *attention_mask* containing
        the position index for each valid token and 0 for padded positions.
    """
    position_ids = np.cumsum(attention_mask.astype(np.int32), axis=-1) - 1
    position_ids = np.where(attention_mask > 0, position_ids, 0)
    return position_ids.astype(np.int32)


def _mslite_tensor(np_array):
    """Wrap a NumPy array as a MindSpore Lite ``Tensor``.

    Parameters
    ----------
    np_array : numpy.ndarray
        The data to convert.

    Returns
    -------
    mslite.Tensor
        A MindSpore Lite tensor backed by *np_array*.
    """
    return mslite.Tensor(np_array)


def _build_mslite_inputs(model, feed_dict, preferred_order=None):
    """Build the ordered list of ``mslite.Tensor`` inputs required by *model*.

    The function first attempts to match model inputs by name.  When that is
    not possible it falls back to *preferred_order* (if supplied) or raises.

    Parameters
    ----------
    model : mslite.Model
        The target MindSpore Lite model whose input slots are used as the
        reference ordering.
    feed_dict : dict[str, numpy.ndarray]
        Mapping from input names to NumPy arrays.
    preferred_order : list[str] or None, optional
        Fallback key order used when model inputs cannot be matched by name.

    Returns
    -------
    list[mslite.Tensor]
        An ordered list of tensors ready for ``model.predict()``.

    Raises
    ------
    RuntimeError
        If the model input names and *feed_dict* keys cannot be reconciled.
    """
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
        return [_mslite_tensor(feed_dict[t.name]) for t in inputs]

    if preferred_order:
        return [_mslite_tensor(feed_dict[k]) for k in preferred_order]

    raise RuntimeError(
        f"input mismatch. model inputs={[getattr(x, 'name', '') for x in inputs]} "
        f"feed keys={list(feed_dict.keys())}"
    )


class Qwen34BInferencer:
    """Qwen3-4B inferencer backed by MindSpore Lite (prefill + decode).

    This class loads a split MindIR model pair (one for the prefill phase and
    one for the decode phase) and provides a high-level ``generate`` method
    that handles tokenisation, KV-cache management, and optional streaming
    output.
    """

    def __init__(
        self,
        prefill_model_path: str,
        decode_model_path: str,
        tokenizer_id: str,
        device: str = "ascend",
        device_id: int = 0,
    ):
        """Initialise the inferencer by loading models and the tokenizer.

        Parameters
        ----------
        prefill_model_path : str
            Filesystem path to the prefill-phase MindIR model.
        decode_model_path : str
            Filesystem path to the decode-phase MindIR model.
        tokenizer_id : str
            HuggingFace model identifier or local path used to load the
            ``AutoTokenizer``.
        device : str, optional
            Target device — ``"cpu"`` or ``"ascend"``.  Defaults to
            ``"ascend"``.
        device_id : int, optional
            Ascend device index (ignored when *device* is ``"cpu"``).
            Defaults to ``0``.

        Raises
        ------
        ValueError
            If *device* is not ``"cpu"`` or ``"ascend"``.
        """
        if device not in ["cpu", "ascend"]:
            raise ValueError("device must be cpu or ascend")

        print(f"Initializing MindSpore Lite context for {device}...")
        self.context = mslite.Context()
        self.context.target = [device]
        if device == "ascend":
            self.context.ascend.device_id = device_id

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

    def _prepare_inputs(self, text, max_length):
        """Tokenise a user prompt and build padded model inputs.

        The chat template is applied, tokens are truncated to
        ``max_length``, and right-side padding is added to match the
        nearest prefill gear dimension.

        Parameters
        ----------
        text : str
            The raw user prompt.
        max_length : int or None
            Requested maximum sequence length.  Clamped to
            ``min(max_length, PREFILL_GEARS[-1])``; falls back to
            ``KV_CACHE_LEN`` when non-positive or ``None``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            ``(input_ids, attention_mask, position_ids)`` — each of shape
            ``(1, gear_len)`` and dtype ``int32``.
        """
        enc = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="np",
        )
        if hasattr(enc, "__getitem__") and "input_ids" in enc:
            input_ids = np.array(enc["input_ids"])
            attention_mask = np.array(enc.get("attention_mask", np.ones_like(input_ids)))
        else:
            input_ids = np.array(enc)
            attention_mask = np.ones_like(input_ids)

        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        if attention_mask.ndim == 1:
            attention_mask = attention_mask[None, :]

        max_length = int(max_length) if max_length is not None and int(max_length) > 0 else KV_CACHE_LEN
        max_length = min(max_length, PREFILL_GEARS[-1])
        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]

        gear_len = _select_prefill_gear(int(input_ids.shape[1]))
        input_ids, attention_mask = self._pad_to_gear(
            input_ids, attention_mask, gear_len
        )

        input_ids = input_ids.astype(np.int32, copy=False)
        attention_mask = attention_mask.astype(np.int32, copy=False)
        position_ids = _compute_position_ids(attention_mask)
        return input_ids, attention_mask, position_ids

    def _pad_to_gear(self, input_ids, attention_mask, gear_len):
        """Right-pad *input_ids* and *attention_mask* to *gear_len*.

        Parameters
        ----------
        input_ids : numpy.ndarray
            Token IDs of shape ``(batch, seq_len)``.
        attention_mask : numpy.ndarray
            Corresponding mask of the same shape.
        gear_len : int
            Target length from the prefill gear configuration.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The padded ``(input_ids, attention_mask)`` pair, each of shape
            ``(batch, gear_len)``.
        """
        seq_len = int(input_ids.shape[1])
        if gear_len <= seq_len:
            return input_ids, attention_mask
        pad_len = gear_len - seq_len
        pad_id = int(self.tokenizer.pad_token_id)
        pad_ids = np.full(
            (input_ids.shape[0], pad_len), pad_id, dtype=input_ids.dtype
        )
        pad_mask = np.zeros(
            (attention_mask.shape[0], pad_len), dtype=attention_mask.dtype
        )
        input_ids = np.concatenate([input_ids, pad_ids], axis=1)
        attention_mask = np.concatenate([attention_mask, pad_mask], axis=1)
        return input_ids, attention_mask

    def _stream_print_delta(self, generated_ids, prev_text):
        """Decode *generated_ids* and print only the newly produced text.

        Parameters
        ----------
        generated_ids : list[int]
            All token ids generated so far.
        prev_text : str
            Previously streamed decoded text.

        Returns
        -------
        str
            The full decoded text including the latest tokens (used as
            *prev_text* on the next call).
        """
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

    def _run_prefill(self, input_ids, attention_mask, position_ids):
        """Execute the prefill phase and return initial outputs.

        Parameters
        ----------
        input_ids : numpy.ndarray
            Token IDs of shape ``(1, gear_len)``.
        attention_mask : numpy.ndarray
            Mask of shape ``(1, gear_len)``.
        position_ids : numpy.ndarray
            Position IDs of shape ``(1, gear_len)``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]
            ``(logits, past_k, past_v, prefill_ms)`` from the prefill model.

        Raises
        ------
        RuntimeError
            If the KV-cache dimension returned by the model does not equal
            ``KV_CACHE_LEN``.
        """
        prefill_feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        inputs = _build_mslite_inputs(
            self.prefill_model,
            prefill_feed,
            preferred_order=["input_ids", "attention_mask", "position_ids"],
        )
        t0 = time.time()
        prefill_outputs = self.prefill_model.predict(inputs)
        prefill_ms = (time.time() - t0) * 1000.0
        logits = prefill_outputs[0].get_data_to_numpy()
        past_k = prefill_outputs[1].get_data_to_numpy()
        past_v = prefill_outputs[2].get_data_to_numpy()

        if int(past_k.shape[3]) != KV_CACHE_LEN or int(past_v.shape[3]) != KV_CACHE_LEN:
            raise RuntimeError(
                f"prefill cache len mismatch, expected {KV_CACHE_LEN}, "
                f"got k={past_k.shape}, v={past_v.shape}"
            )
        return logits, past_k, past_v, prefill_ms

    def _run_decode_loop(
        self, generated_ids, cur_attention_mask, past_k, past_v,
        valid_len, max_new_tokens, stream, streamed_text,
    ):
        """Iterate the decode phase, appending tokens until a stop condition.

        Parameters
        ----------
        generated_ids : list[int]
            Seed token ids (typically containing the first predicted token).
        cur_attention_mask : numpy.ndarray
            Current attention mask of shape ``(1, KV_CACHE_LEN)``.
        past_k : numpy.ndarray
            Key cache from the prefill phase.
        past_v : numpy.ndarray
            Value cache from the prefill phase.
        valid_len : int
            Number of valid positions already in the cache.
        max_new_tokens : int
            Maximum number of new tokens to generate (including the seed
            token already in *generated_ids*).
        stream : bool
            If ``True``, incremental text deltas are printed in real time.
        streamed_text : str
            Text already streamed to the console.

        Returns
        -------
        tuple[list[int], str, list[float]]
            The final ``generated_ids`` list, the full streamed text, and
            a list of per-step decode times in milliseconds.
        """
        decode_times = []
        for _ in range(max_new_tokens - 1):
            if self.eos_token_id is not None and generated_ids[-1] == int(
                self.eos_token_id
            ):
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
                self.decode_model,
                decode_feed,
                preferred_order=[
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "past_key_cache",
                    "past_value_cache",
                ],
            )
            t1 = time.time()
            decode_outputs = self.decode_model.predict(inputs)
            decode_ms = (time.time() - t1) * 1000.0
            decode_times.append(decode_ms)
            logits = decode_outputs[0].get_data_to_numpy()
            past_k = decode_outputs[1].get_data_to_numpy()
            past_v = decode_outputs[2].get_data_to_numpy()
            valid_len += 1
            generated_ids.append(int(np.argmax(logits[0, -1])))
            if stream:
                streamed_text = self._stream_print_delta(
                    generated_ids, streamed_text
                )

        if stream:
            print()

        return generated_ids, streamed_text, decode_times

    def generate(
        self,
        text: str,
        max_new_tokens: int = 128,
        max_length: int = 4096,
        stream: bool = True,
    ):
        """Generate a response for the given *text* prompt.

        The method runs the prefill phase to obtain the first predicted token
        and the initial KV-cache, then iterates the decode phase to produce
        additional tokens.

        Parameters
        ----------
        text : str
            The user prompt to feed into the model.
        max_new_tokens : int, optional
            Upper bound on the number of tokens to generate.  Defaults to 128.
        max_length : int, optional
            Maximum input sequence length (truncated before padding).  Defaults
            to 4096.
        stream : bool, optional
            When ``True`` (default), incremental text is printed to stdout as
            tokens are generated.

        Returns
        -------
        tuple[str, dict]
            The full decoded response string (excluding special tokens) and
            a dict with performance metrics.
        """
        input_ids, attention_mask, position_ids = self._prepare_inputs(
            text, max_length
        )

        logits, past_k, past_v, prefill_ms = self._run_prefill(
            input_ids, attention_mask, position_ids
        )

        actual_len = int(attention_mask[0].sum())
        cur_attention_mask = np.zeros((1, KV_CACHE_LEN), dtype=np.int32)
        if actual_len > 0:
            cur_attention_mask[0, :actual_len] = 1

        last_idx = max(actual_len - 1, 0)
        generated_ids = [int(np.argmax(logits[0, last_idx]))]
        streamed_text = ""
        if stream:
            streamed_text = self._stream_print_delta(generated_ids, streamed_text)

        generated_ids, _, decode_times = self._run_decode_loop(
            generated_ids, cur_attention_mask, past_k, past_v,
            int(actual_len), max_new_tokens, stream, streamed_text,
        )

        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        total_decode_ms = sum(decode_times)
        avg_decode_ms = total_decode_ms / len(decode_times) if decode_times else 0.0
        total_ms = prefill_ms + total_decode_ms
        perf = {
            "prefill_ms": prefill_ms, "total_decode_ms": total_decode_ms,
            "avg_decode_ms": avg_decode_ms, "total_ms": total_ms,
            "num_generated": len(generated_ids),
            "throughput_tok_s": len(generated_ids) / (total_ms / 1000.0) if total_ms > 0 else 0.0,
        }
        return result, perf


def main():
    """Parse CLI arguments and run a single Qwen3-4B inference pass.

    Accepted command-line flags
    ---------------------------
    ``--prefill-model``   Path to the prefill-phase MindIR file (required).
    ``--decode-model``    Path to the decode-phase MindIR file (required).
    ``--tokenizer``       Tokenizer path or HuggingFace id (default ``./Qwen3-4B``).
    ``--prompt``          Input prompt string.
    ``--max-new-tokens``  Maximum number of new tokens to generate.
    ``--max-length``      Maximum input sequence length.
    ``--device``          Target device (``cpu`` or ``ascend``).
    ``--device-id``       Ascend device index.
    """
    parser = argparse.ArgumentParser(
        description="Qwen3-4B Inference with MindSpore Lite (prefill + decode)"
    )
    parser.add_argument(
        "--prefill-model", type=str, required=True, help="Path to prefill .mindir"
    )
    parser.add_argument(
        "--decode-model", type=str, required=True, help="Path to decode .mindir"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="./Qwen3-4B", help="Tokenizer path"
    )
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--device", type=str, default="ascend", choices=["cpu", "ascend"]
    )
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    inferencer = Qwen34BInferencer(
        prefill_model_path=args.prefill_model,
        decode_model_path=args.decode_model,
        tokenizer_id=args.tokenizer,
        device=args.device,
        device_id=args.device_id,
    )

    print("\n" + "=" * 60)
    print(f"Input Prompt: {args.prompt}")
    print("=" * 60)
    print("Generated Response: ", end="", flush=True)
    _, perf = inferencer.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
    )
    print("\n--- Performance ---")
    print(f"  Prefill:           {perf['prefill_ms']:.2f} ms")
    print(f"  Total Decode:      {perf['total_decode_ms']:.2f} ms")
    print(f"  Avg Decode Step:   {perf['avg_decode_ms']:.2f} ms")
    print(f"  Total:             {perf['total_ms']:.2f} ms")
    print(f"  Tokens Generated:  {perf['num_generated']}")
    print(f"  Throughput:        {perf['throughput_tok_s']:.2f} tok/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
