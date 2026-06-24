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
"""InternVL3.5-2B-Flash vision-language inference with MindSpore Lite (Ascend).

Loads the three converted MindIR sub-models (InternViT vision encoder + Qwen3 LLM
prefill + decode), encodes the image to visual embeds, fuses them into the token
embedding stream (replacing ``<IMG_CONTEXT>`` positions), and runs greedy
autoregressive decoding with a fixed-shape KV cache. The decode sub-model scatters
the new KV into the cache internally and returns the full updated cache, so the
infer loop only feeds the cache back. All model compute is pure mslite + numpy;
``torch`` is never imported.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

import mindspore_lite as mslite

KV_CACHE_LEN = 1024
PREFILL_SEQ = 320
NUM_IMG_TOKENS = 256
IMAGE_SIZE = 448
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PREFILL_INPUT_ORDER = ["inputs_embeds", "attention_mask", "position_ids"]
DECODE_INPUT_ORDER = ["inputs_embeds", "attention_mask", "position_ids",
                      "past_key_cache", "past_value_cache"]

_MS_DTYPE = {"FLOAT16": np.float16, "FLOAT32": np.float32,
             "INT32": np.int32, "INT64": np.int64}


def _np_dtype(dtype_info):
    """Map a mslite tensor dtype to a numpy dtype, or None if unknown."""
    return _MS_DTYPE.get(getattr(dtype_info, "name", str(dtype_info)))


def _to_tensor(np_array, tensor_info):
    """Cast a numpy array to the model input dtype and wrap as mslite.Tensor."""
    target = _np_dtype(tensor_info.dtype)
    arr = np_array.astype(target, copy=False) if target is not None else np_array
    return mslite.Tensor(arr)


def _build_inputs(model, feed_dict, preferred_order):
    """Build mslite input tensors by name match, falling back to preferred order."""
    inputs = model.get_inputs()
    name_to_info = {t.name: t for t in inputs if getattr(t, "name", "")}
    tensors = []
    if name_to_info and all(k in name_to_info for k in feed_dict):
        for t in inputs:
            tensors.append(_to_tensor(feed_dict[t.name], t))
    else:
        for key in preferred_order:
            tensors.append(_to_tensor(feed_dict[key], inputs[preferred_order.index(key)]))
    return tensors


def _describe_io(model, tag):
    """Print model input/output dtype+shape once for dtype alignment debugging."""
    print(f"[{tag}] inputs:")
    for t in model.get_inputs():
        print(f"    {t.name}: {_np_dtype(t.dtype)} {t.shape}")
    print(f"[{tag}] outputs:")
    for t in model.get_outputs():
        print(f"    {t.name}: {_np_dtype(t.dtype)} {t.shape}")


class InternVLInferencer:
    """End-to-end InternVL3.5 multimodal inferencer over MindSpore Lite."""

    def __init__(self, mindir_dir, model_dir, vision_device=1, llm_device=0):
        """Load sub-models, tokenizer and the LLM token-embedding matrix."""
        mindir_dir = Path(mindir_dir)
        self.vision_device = vision_device
        self.llm_device = llm_device

        self.vision = self._build_model(mindir_dir / "internvl_vision.mindir", vision_device)
        self.prefill = self._build_model(mindir_dir / "internvl_llm_prefill_graph.mindir", llm_device)
        self.decode = self._build_model(mindir_dir / "internvl_llm_decode_graph.mindir", llm_device)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.img_context_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.embed_weights = np.load(mindir_dir / "embed_weights.npy").astype(np.float32)

    @staticmethod
    def _build_model(mindir_path, device_id):
        """Build an mslite Model from a ``*_graph.mindir`` on a given Ascend device."""
        ctx = mslite.Context()
        ctx.target = ["ascend"]
        ctx.ascend.device_id = int(device_id)
        model = mslite.Model()
        model.build_from_file(str(mindir_path), mslite.ModelType.MINDIR, ctx)
        return model

    def _preprocess_image(self, image):
        """Resize to IMAGE_SIZE square and normalize (CLIP mean/std), [1,3,H,W]."""
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - IMG_MEAN) / IMG_STD
        return arr.transpose(2, 0, 1)[None, :].astype(np.float32)

    def _run_vision(self, pixel_values):
        """Run the vision encoder, return image_embeds [1, num_img_tokens, hidden]."""
        feed = {"pixel_values": pixel_values}
        inputs = _build_inputs(self.vision, feed, ["pixel_values"])
        out = self.vision.predict(inputs)
        return out[0].get_data_to_numpy()

    def _build_inputs_embeds(self, input_ids, image_embeds):
        """Embed input_ids, replace ``<IMG_CONTEXT>`` positions with visual embeds."""
        ids = input_ids[0] if input_ids.ndim == 2 else input_ids
        embeds = self.embed_weights[ids]
        img_pos = np.where(ids == self.img_context_id)[0]
        n = min(len(img_pos), image_embeds.shape[1])
        for i, pos in enumerate(img_pos[:n]):
            embeds[pos] = image_embeds[0, i]
        return embeds[None, :, :]

    def _build_prompt_ids(self, question):
        """Apply chat template, inject image-context tokens, tokenize, pad to PREFILL_SEQ."""
        query = f"<image>\n{question}"
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": query}], tokenize=False, add_generation_prompt=True)
        text = text.replace("<image>", "<IMG_CONTEXT>" * NUM_IMG_TOKENS)
        enc = self.tokenizer(text, return_tensors="np", add_special_tokens=False)
        input_ids = np.asarray(enc["input_ids"])[0]
        actual_len = int(input_ids.shape[0])
        if actual_len > PREFILL_SEQ:
            input_ids = input_ids[:PREFILL_SEQ]
            actual_len = PREFILL_SEQ
        pad_id = int(self.tokenizer.pad_token_id)
        pad_len = PREFILL_SEQ - actual_len
        input_ids = np.concatenate([input_ids, np.full(pad_len, pad_id, dtype=np.int64)])
        attention_mask = np.concatenate([np.ones(actual_len, dtype=np.int64),
                                         np.zeros(pad_len, dtype=np.int64)])
        return input_ids[None, :], attention_mask[None, :], actual_len

    def generate(self, image, question, max_new_tokens=128):
        """Run vision encode + prefill + greedy decode with stage timing."""
        timing = {}
        t0 = time.perf_counter()
        pixel_values = self._preprocess_image(image)
        image_embeds = self._run_vision(pixel_values)
        timing["vision_ms"] = (time.perf_counter() - t0) * 1000.0

        input_ids, attention_mask, actual_len = self._build_prompt_ids(question)
        inputs_embeds = self._build_inputs_embeds(input_ids, image_embeds)
        position_ids = np.arange(PREFILL_SEQ, dtype=np.int64)[None, :]
        attention_mask = attention_mask.astype(np.int32, copy=False)
        position_ids = position_ids.astype(np.int32, copy=False)

        t0 = time.perf_counter()
        prefill_feed = {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask,
                        "position_ids": position_ids}
        out = self.prefill.predict(_build_inputs(self.prefill, prefill_feed, PREFILL_INPUT_ORDER))
        logits, past_k, past_v = (o.get_data_to_numpy() for o in out)
        timing["prefill_ms"] = (time.perf_counter() - t0) * 1000.0

        cur_mask = np.zeros((1, KV_CACHE_LEN), dtype=np.int32)
        cur_mask[0, :actual_len] = 1
        valid_len = actual_len
        generated = [int(np.argmax(logits[0, actual_len - 1]))]

        decode_times = []
        t0 = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            if self.eos_token_id is not None and generated[-1] == int(self.eos_token_id):
                break
            if valid_len >= KV_CACHE_LEN:
                break
            cur_mask[0, valid_len] = 1
            tok_embed = self.embed_weights[generated[-1]][None, None, :]
            pos = np.array([[valid_len]], dtype=np.int32)
            decode_feed = {"inputs_embeds": tok_embed, "attention_mask": cur_mask,
                           "position_ids": pos, "past_key_cache": past_k,
                           "past_value_cache": past_v}
            out = self.decode.predict(_build_inputs(self.decode, decode_feed, DECODE_INPUT_ORDER))
            logits, past_k, past_v = (o.get_data_to_numpy() for o in out)
            decode_times.append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            valid_len += 1
            generated.append(int(np.argmax(logits[0, -1])))
        timing["decode_total_ms"] = sum(decode_times)
        timing["decode_steps"] = len(generated) - 1
        timing["e2e_ms"] = timing["vision_ms"] + timing["prefill_ms"] + timing["decode_total_ms"]
        text_out = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text_out, timing


def _parse_args():
    """Parse command-line arguments for the inference script."""
    parser = argparse.ArgumentParser(description="InternVL3.5-2B-Flash MindSpore Lite inference")
    parser.add_argument("--mindir-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Describe this image in detail.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--vision-device", type=int, default=1)
    parser.add_argument("--llm-device", type=int, default=0)
    parser.add_argument("--describe-io", action="store_true")
    return parser.parse_args()


def main():
    """Parse arguments and run InternVL3.5-2B-Flash multimodal inference."""
    args = _parse_args()
    infer = InternVLInferencer(args.mindir_dir, args.model_dir, args.vision_device, args.llm_device)
    if args.describe_io:
        _describe_io(infer.vision, "vision")
        _describe_io(infer.prefill, "prefill")
        _describe_io(infer.decode, "decode")
    image = Image.open(args.image)
    text, timing = infer.generate(image, args.prompt, args.max_new_tokens)

    print("\n[output]", text)
    print("\n--- Performance ---")
    print(f"  Vision encode:   {timing['vision_ms']:.2f} ms")
    print(f"  LLM prefill:     {timing['prefill_ms']:.2f} ms (seq={PREFILL_SEQ})")
    print(f"  LLM decode:      {timing['decode_total_ms']:.2f} ms ({timing['decode_steps']} steps)")
    print(f"  End-to-end:      {timing['e2e_ms']:.2f} ms")
    if timing["decode_steps"] > 0:
        avg = timing["decode_total_ms"] / timing["decode_steps"]
        print(f"  Avg decode step: {avg:.2f} ms")
        print(f"  Throughput:      {1000.0 / avg:.2f} tok/s (decode)")


if __name__ == "__main__":
    main()
