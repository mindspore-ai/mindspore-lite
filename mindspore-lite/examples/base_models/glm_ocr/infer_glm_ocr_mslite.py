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
Infer GLM-OCR on Ascend with MindSpore Lite (vision + prefill + decode).

The whole pipeline is implemented with numpy / PIL (no torch). The only point
that touches torch is ``AutoProcessor.apply_chat_template(..., return_tensors="pt")``,
whose output is converted to numpy immediately.
"""

import argparse
import sys
import time
import urllib.request
from io import BytesIO

import numpy as np
from PIL import Image

try:
    import mindspore_lite as mslite
    from transformers import AutoConfig, AutoProcessor
except ImportError:
    print("Error: mindspore_lite or transformers package not found.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Image loading.
# ---------------------------------------------------------------------------


def _load_image(image_path_or_url):
    """Load an image from a local path or http(s) URL and convert to RGB."""
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        with urllib.request.urlopen(image_path_or_url) as resp:
            data = resp.read()
        return Image.open(BytesIO(data)).convert("RGB")
    return Image.open(image_path_or_url).convert("RGB")


def _pad_to_square(image):
    """Pad an image to a square with black background."""
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    out = Image.new("RGB", (side, side), (0, 0, 0))
    out.paste(image, ((side - w) // 2, (side - h) // 2))
    return out


# ---------------------------------------------------------------------------
# Multimodal RoPE (mRoPE) position ids, reimplemented in numpy.
# Mirrors transformers GlmOcrModel.get_vision_position_ids / get_rope_index.
# ---------------------------------------------------------------------------


def _get_vision_position_ids(start_position, grid_thw, spatial_merge_size):
    """Compute 3D (temporal, height, width) position ids for one image block."""
    grid_t = int(grid_thw[0])
    grid_h = int(grid_thw[1]) // spatial_merge_size
    grid_w = int(grid_thw[2]) // spatial_merge_size
    position_temporal = np.arange(grid_t)  # time_interval == 1
    position_width = np.arange(grid_w) + start_position
    position_height = np.arange(grid_h) + start_position
    position_width = np.tile(position_width, grid_h * grid_t)
    position_height = np.tile(np.repeat(position_height, grid_w), grid_t)
    position_temporal = np.repeat(position_temporal, grid_h * grid_w) + start_position
    return np.stack([position_temporal, position_height, position_width], axis=0)


def _get_rope_index(input_ids, mm_token_type_ids, image_grid_thw,
                    attention_mask, spatial_merge_size):
    """Compute mRoPE position ids [3, bsz, seq] and rope deltas [bsz, 1]."""
    bsz, seq_len = input_ids.shape
    position_ids = np.zeros((3, bsz, seq_len), dtype=np.int64)
    deltas = []
    image_iter = iter(image_grid_thw) if image_grid_thw is not None else iter([])
    for b in range(bsz):
        cur_types = mm_token_type_ids[b]
        mask = attention_mask[b].astype(bool) if attention_mask is not None else None
        if mask is not None:
            cur_types = cur_types[mask]
        types_list = cur_types.tolist()
        groups = []
        start = 0
        for i in range(1, len(types_list) + 1):
            if i == len(types_list) or types_list[i] != types_list[start]:
                groups.append((types_list[start], start, i))
                start = i
        current_pos = 0
        pos_blocks = []
        for modality_type, start_idx, end_idx in groups:
            if modality_type == 0:
                text_len = end_idx - start_idx
                block = np.arange(text_len).reshape(1, -1) + current_pos
                block = np.broadcast_to(block, (3, text_len))
                pos_blocks.append(block)
                current_pos += text_len
            elif modality_type == 1:
                grid = next(image_iter)
                vpos = _get_vision_position_ids(current_pos, grid, spatial_merge_size)
                pos_blocks.append(vpos)
                current_pos += max(int(grid[1]), int(grid[2])) // spatial_merge_size
            else:
                raise ValueError(f"Unsupported modality_type: {modality_type}")
        positions = np.concatenate(pos_blocks, axis=1).reshape(3, -1)
        if mask is not None:
            position_ids[:, b, mask] = positions
        else:
            position_ids[:, b] = positions
        n_tokens = int(attention_mask[b].sum()) if attention_mask is not None else seq_len
        deltas.append(int(positions.max()) + 1 - n_tokens)
    deltas = np.array(deltas, dtype=np.int64).reshape(bsz, 1)
    return position_ids, deltas


# ---------------------------------------------------------------------------
# MindSpore Lite tensor helpers.
# ---------------------------------------------------------------------------


def _np_dtype_to_mslite(dtype):
    """Convert a numpy dtype to the matching MindSpore Lite DataType."""
    dt = np.dtype(dtype)
    if dt == np.float16:
        return mslite.DataType.FLOAT16
    if dt == np.float32:
        return mslite.DataType.FLOAT32
    if dt == np.int32:
        return mslite.DataType.INT32
    if dt == np.int64:
        return mslite.DataType.INT64
    raise TypeError(f"unsupported numpy dtype for mslite.Tensor: {dt}")


def _build_inputs(model, feed_dict, preferred_order):
    """Build mslite input tensors by name (fallback to preferred order)."""
    inputs = model.get_inputs()
    if not inputs:
        return [mslite.Tensor(v) for v in feed_dict.values()]
    ok_by_name = all(getattr(t, "name", None) in feed_dict for t in inputs)
    if ok_by_name:
        return [mslite.Tensor(feed_dict[t.name]) for t in inputs]
    return [mslite.Tensor(feed_dict[k]) for k in preferred_order]


class GlmOcrInferencer:
    """Run GLM-OCR end-to-end on MindSpore Lite (vision + prefill + decode)."""

    def __init__(self, vision_model_path, prefill_model_path, decode_model_path,
                 processor_id, device="ascend", device_id=0, image_size=896,
                 kv_cache_len=2048, prefill_seq=1152, pad_to_square=True):
        """Build the three MindIR models and load the processor/config."""
        if device not in ("cpu", "ascend"):
            raise ValueError("device must be 'cpu' or 'ascend'")
        self.device = str(device)
        self.device_id = int(device_id)
        self.image_size = int(image_size)
        self.kv_cache_len = int(kv_cache_len)
        self.prefill_seq = int(prefill_seq)
        self.pad_to_square = bool(pad_to_square)

        self.context = mslite.Context()
        self.context.target = [self.device]
        if self.device == "ascend":
            self.context.ascend.device_id = self.device_id

        print(f"Loading vision model from {vision_model_path}...")
        self.vision_model = mslite.Model()
        self.vision_model.build_from_file(vision_model_path, mslite.ModelType.MINDIR, self.context)
        print(f"Loading prefill model from {prefill_model_path}...")
        self.prefill_model = mslite.Model()
        self.prefill_model.build_from_file(prefill_model_path, mslite.ModelType.MINDIR, self.context)
        print(f"Loading decode model from {decode_model_path}...")
        self.decode_model = mslite.Model()
        self.decode_model.build_from_file(decode_model_path, mslite.ModelType.MINDIR, self.context)
        # Match the decode past_key_values input dtype (fp16 or fp32) for the KV buffer.
        self.kv_dtype = np.float16
        for _inp in self.decode_model.get_inputs():
            if getattr(_inp, "name", "") == "past_key_values":
                self.kv_dtype = np.float16 if str(_inp.dtype).endswith("FLOAT16") else np.float32
                break

        print(f"Loading processor from {processor_id}...")
        self.cfg = AutoConfig.from_pretrained(processor_id)
        self.processor = AutoProcessor.from_pretrained(processor_id)
        self.image_token_id = int(self.cfg.image_token_id)
        self.eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        self._decode_io = None
        # Match the prefill image_embeds input dtype (fp16 or fp32 depending on convert).
        self.img_dtype = np.float32
        for _inp in self.prefill_model.get_inputs():
            if getattr(_inp, "name", "") == "image_embeds":
                self.img_dtype = np.float16 if str(_inp.dtype).endswith("FLOAT16") else np.float32
                break

    def _prepare_inputs(self, image_path_or_url, prompt):
        """Build (input_ids, attention_mask, mm_token_type_ids, pixel_values, grid_thw)."""
        image = _load_image(image_path_or_url)
        if self.pad_to_square:
            image = _pad_to_square(image)
        image = image.resize((self.image_size, self.image_size))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        enc = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        input_ids = np.array(enc["input_ids"]).astype(np.int32)
        attention_mask = np.array(enc["attention_mask"]).astype(np.int32)
        pixel_values = np.array(enc["pixel_values"]).astype(np.float32)
        image_grid_thw = np.array(enc["image_grid_thw"]).astype(np.int32)
        if "mm_token_type_ids" in enc:
            mm_token_type_ids = np.array(enc["mm_token_type_ids"]).astype(np.int32)
        else:
            mm_token_type_ids = (input_ids == self.image_token_id).astype(np.int32)
        return input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw

    def _run_vision(self, pixel_values):
        """Run vision model, return (image_embeds fp16, elapsed_ms)."""
        feed = {"pixel_values": pixel_values}
        t0 = time.perf_counter()
        out = self.vision_model.predict(_build_inputs(self.vision_model, feed, ["pixel_values"]))
        t_ms = (time.perf_counter() - t0) * 1000.0
        image_embeds = out[0].get_data_to_numpy()
        if image_embeds.dtype != self.img_dtype:
            image_embeds = image_embeds.astype(self.img_dtype)
        return image_embeds, t_ms

    def _ensure_decode_io(self, past_kv_fixed):
        """Pre-allocate Ascend decode IO tensors for performance (double-buffer)."""
        if self.device != "ascend":
            return None
        device_str = f"ascend:{self.device_id}"
        ms_kv = _np_dtype_to_mslite(past_kv_fixed.dtype)
        past_shape = list(past_kv_fixed.shape)
        io = {
            "device": device_str,
            "t_input_ids": mslite.Tensor(shape=[1, 1], dtype=mslite.DataType.INT32, device=device_str),
            "t_attention_mask": mslite.Tensor(shape=[1, self.kv_cache_len],
                                              dtype=mslite.DataType.INT32, device=device_str),
            "t_position_ids": mslite.Tensor(shape=[3, 1, 1], dtype=mslite.DataType.INT32, device=device_str),
            "t_cache_pos": mslite.Tensor(shape=[1], dtype=mslite.DataType.INT32, device=device_str),
            "t_past_in": mslite.Tensor(shape=past_shape, dtype=ms_kv, device=device_str),
        }
        io["t_past_in"].set_data_from_numpy(past_kv_fixed)
        outs = self.decode_model.get_outputs()
        io["out_bufs"] = None
        try:
            if outs and getattr(outs[0], "shape", None) and getattr(outs[1], "shape", None):
                logits_shape = [int(x) for x in outs[0].shape]
                past_shape_out = [int(x) for x in outs[1].shape]
                if all(x > 0 for x in logits_shape) and all(x > 0 for x in past_shape_out):
                    out_dt = _np_dtype_to_mslite(self.kv_dtype)
                    io["t_logits_out"] = mslite.Tensor(shape=logits_shape, dtype=out_dt, device=device_str)
                    io["t_past_out"] = mslite.Tensor(shape=past_shape_out, dtype=out_dt, device=device_str)
                    io["out_bufs"] = [io["t_logits_out"], io["t_past_out"]]
        except Exception:
            io["out_bufs"] = None
        return io

    def _run_decode_loop(self, generated, past_kv_fixed, attn_mask_fixed, cache_pos,
                         rope_delta, max_new_tokens):
        """Run autoregressive decode; return (total_ms, num_steps)."""
        total_ms = 0.0
        steps = 0
        io = self._ensure_decode_io(past_kv_fixed)
        for _ in range(max_new_tokens - 1):
            if self.eos_token_id is not None and generated[-1] == int(self.eos_token_id):
                break
            if cache_pos >= self.kv_cache_len:
                break
            step_id = np.array([[generated[-1]]], dtype=np.int32)
            attn_mask_fixed[0, :cache_pos + 1] = 1
            pos_val = int(cache_pos + int(rope_delta))
            position_ids_step = np.full((3, 1, 1), pos_val, dtype=np.int32)
            cache_pos_arr = np.array([cache_pos], dtype=np.int32)
            t0 = time.perf_counter()
            if io is not None:
                io["t_input_ids"].set_data_from_numpy(step_id)
                io["t_attention_mask"].set_data_from_numpy(attn_mask_fixed)
                io["t_position_ids"].set_data_from_numpy(position_ids_step)
                io["t_cache_pos"].set_data_from_numpy(cache_pos_arr)
                inputs = [io["t_input_ids"], io["t_attention_mask"], io["t_position_ids"],
                          io["t_past_in"], io["t_cache_pos"]]
                decode_out = self.decode_model.predict(inputs, outputs=io["out_bufs"])
                if io["out_bufs"] is not None:
                    prev = io["t_past_in"]
                    io["t_past_in"] = decode_out[1]
                    io["t_past_out"] = prev
                    io["out_bufs"][1] = io["t_past_out"]
                else:
                    io["t_past_in"] = decode_out[1]
            else:
                feed = {"input_ids": step_id, "attention_mask": attn_mask_fixed,
                        "position_ids": position_ids_step,
                        "past_key_values": past_kv_fixed.astype(np.float16),
                        "cache_pos": cache_pos_arr}
                decode_out = self.decode_model.predict(_build_inputs(
                    self.decode_model, feed,
                    ["input_ids", "attention_mask", "position_ids", "past_key_values", "cache_pos"]))
                past_kv_fixed = decode_out[1].get_data_to_numpy()
            total_ms += (time.perf_counter() - t0) * 1000.0
            logits = decode_out[0].get_data_to_numpy()
            generated.append(int(np.argmax(logits[0, -1])))
            cache_pos += 1
            steps += 1
        return total_ms, steps

    def infer(self, image_path_or_url, prompt, max_new_tokens=256):
        """Run the full GLM-OCR pipeline and return (decoded_text, timing_dict)."""
        t_start = time.perf_counter()
        input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw = (
            self._prepare_inputs(image_path_or_url, prompt))
        spatial_merge_size = int(self.cfg.vision_config.spatial_merge_size)
        position_ids, deltas = _get_rope_index(
            input_ids, mm_token_type_ids, image_grid_thw, attention_mask, spatial_merge_size)
        rope_delta = int(deltas[0, 0])

        image_embeds, t_vision = self._run_vision(pixel_values)
        image_token_cnt = int((input_ids[0] == self.image_token_id).sum())
        if int(image_embeds.shape[0]) != image_token_cnt:
            raise RuntimeError(
                f"image_embeds length {image_embeds.shape[0]} != image tokens {image_token_cnt}; "
                f"grid_thw={image_grid_thw.tolist()} (check --image-size matches export)")

        prompt_len = int(attention_mask.sum())
        if prompt_len <= 0:
            raise RuntimeError(f"invalid prompt_len={prompt_len}")
        if prompt_len > self.prefill_seq:
            raise RuntimeError(
                f"prompt_len={prompt_len} > fixed prefill_seq={self.prefill_seq}; "
                f"re-export/convert prefill with a larger seq or shorten the prompt.")
        if prompt_len >= self.kv_cache_len:
            raise RuntimeError(f"prompt_len={prompt_len} >= kv_cache_len={self.kv_cache_len}")

        # Fixed-shape prefill: pad input_ids/attention_mask/position_ids to prefill_seq.
        pad_len = self.prefill_seq - input_ids.shape[1]
        pad_id = int(getattr(self.processor.tokenizer, "pad_token_id", 0) or 0)
        input_ids = np.concatenate(
            [input_ids, np.full((1, pad_len), pad_id, dtype=np.int32)], axis=1)
        attention_mask = np.concatenate(
            [attention_mask, np.zeros((1, pad_len), dtype=np.int32)], axis=1)
        position_ids = np.concatenate(
            [position_ids, np.zeros((3, 1, pad_len), dtype=np.int32)], axis=2)

        prefill_feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids.astype(np.int32),
            "image_embeds": image_embeds,
        }
        t0 = time.perf_counter()
        prefill_out = self.prefill_model.predict(_build_inputs(
            self.prefill_model, prefill_feed,
            ["input_ids", "attention_mask", "position_ids", "image_embeds"]))
        t_prefill = (time.perf_counter() - t0) * 1000.0
        logits = prefill_out[0].get_data_to_numpy()
        past_kv = prefill_out[1].get_data_to_numpy()

        # Fixed-shape prefill emits logits for all prefill_seq positions; take the
        # last *real* (non-padded) token. KV cache holds all prefill_seq positions;
        # padded slots are masked out by the decode attention mask.
        generated = [int(np.argmax(logits[0, prompt_len - 1]))]

        past_kv_fixed = np.zeros(
            (past_kv.shape[0], past_kv.shape[1], past_kv.shape[2], self.kv_cache_len, past_kv.shape[4]),
            dtype=self.kv_dtype)
        past_kv_fixed[:, :, :, : self.prefill_seq, :] = past_kv.astype(self.kv_dtype)
        attn_mask_fixed = np.zeros((1, self.kv_cache_len), dtype=np.int32)
        attn_mask_fixed[0, :prompt_len] = 1
        cache_pos = prompt_len

        t_decode, steps = self._run_decode_loop(
            generated, past_kv_fixed, attn_mask_fixed, cache_pos, rope_delta, max_new_tokens)
        t_e2e = (time.perf_counter() - t_start) * 1000.0
        avg_decode = (t_decode / steps) if steps > 0 else 0.0
        timing = {"vision": t_vision, "prefill": t_prefill, "decode_total": t_decode,
                  "decode_steps": steps, "decode_avg": avg_decode, "e2e": t_e2e,
                  "prompt_len": prompt_len, "tokens": len(generated)}
        text = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
        return text, timing


def main():
    """Parse args and run GLM-OCR inference on MindSpore Lite."""
    parser = argparse.ArgumentParser(description="GLM-OCR MindSpore Lite inference")
    parser.add_argument("--vision-model", type=str, required=True, help="Path to glm_ocr_vision.mindir")
    parser.add_argument("--prefill-model", type=str, required=True, help="Path to glm_ocr_llm_prefill.mindir")
    parser.add_argument("--decode-model", type=str, required=True, help="Path to glm_ocr_llm_decode.mindir")
    parser.add_argument("--processor", type=str, default="./GLM-OCR", help="Processor / model dir")
    parser.add_argument("--image", type=str, required=True, help="Image URL or local path")
    parser.add_argument("--prompt", type=str, default="Text Recognition:", help="Text prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=896,
                        help="Fixed image size (must match exported vision model).")
    parser.add_argument("--kv-cache-len", type=int, default=2048)
    parser.add_argument("--prefill-seq", type=int, default=1152,
                        help="Fixed prefill seq length (must match the converted prefill MindIR).")
    parser.add_argument("--no-pad-to-square", action="store_true")
    parser.add_argument("--device", type=str, default="ascend", choices=["ascend", "cpu"])
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    inferencer = GlmOcrInferencer(
        vision_model_path=args.vision_model, prefill_model_path=args.prefill_model,
        decode_model_path=args.decode_model, processor_id=args.processor,
        device=args.device, device_id=args.device_id, image_size=args.image_size,
        kv_cache_len=args.kv_cache_len, prefill_seq=args.prefill_seq,
        pad_to_square=not bool(args.no_pad_to_square))

    result, timing = inferencer.infer(args.image, args.prompt, max_new_tokens=args.max_new_tokens)
    avg_decode = timing["decode_avg"]
    decode_total = timing["decode_total"]
    tokens = timing["tokens"]
    throughput = (tokens / (timing["e2e"] / 1000.0)) if timing["e2e"] > 0 else 0.0
    print(
        "\n--- Performance ---\n"
        f"  Vision:           {timing['vision']:.2f} ms\n"
        f"  Prefill:          {timing['prefill']:.2f} ms (prompt_len={timing['prompt_len']})\n"
        f"  Total Decode:     {decode_total:.2f} ms\n"
        f"  Avg Decode Step:  {avg_decode:.2f} ms\n"
        f"  Total:            {timing['e2e']:.2f} ms\n"
        f"  Tokens Generated: {tokens}\n"
        f"  Throughput:       {throughput:.2f} tok/s")

    print("\n" + "=" * 60)
    print(f"Prompt: {args.prompt}")
    print("-" * 60)
    print(f"Response:\n{result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
