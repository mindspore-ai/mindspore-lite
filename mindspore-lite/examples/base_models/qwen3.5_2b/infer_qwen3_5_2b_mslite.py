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
Infer Qwen3.5-2B on Ascend with MindSpore Lite.

Qwen3.5-2B is a multimodal VL model with hybrid linear attention
(GatedDeltaNet) and full attention architecture. This script runs
inference using three MindIR models:
  - Vision Tower
  - LLM Prefill (with image_embeds input)
  - LLM Decode (with conv_state + recurrent_state + KV cache)

This script does NOT depend on torch -- all computation uses numpy.
"""

import sys
import argparse
import urllib.request
import itertools
import time
from io import BytesIO

import numpy as np
from PIL import Image

try:
    import mindspore_lite as mslite
    from transformers import AutoConfig, AutoProcessor
except ImportError:
    print("Error: mindspore_lite or transformers package not found.")
    print("Please install them first.")
    sys.exit(1)


def _load_image(image_path_or_url: str) -> Image.Image:
    """Load an image from a local path or URL and convert to RGB."""
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        with urllib.request.urlopen(image_path_or_url) as resp:
            data = resp.read()
        return Image.open(BytesIO(data)).convert("RGB")
    return Image.open(image_path_or_url).convert("RGB")


def _pad_to_square(image: Image.Image) -> Image.Image:
    """Pad image to square with black pixels, keeping the original centered."""
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    out = Image.new("RGB", (side, side), (0, 0, 0))
    out.paste(image, ((side - w) // 2, (side - h) // 2))
    return out


def _get_vision_position_ids(start_position, grid_thw, spatial_merge_size):
    """Compute 3D (temporal, height, width) position ids for vision tokens."""
    llm_grid_t = int(grid_thw[0])
    llm_grid_h = int(grid_thw[1]) // spatial_merge_size
    llm_grid_w = int(grid_thw[2]) // spatial_merge_size

    position_temporal = np.arange(llm_grid_t, dtype=np.int32)
    position_width = np.arange(llm_grid_w, dtype=np.int32) + start_position
    position_height = np.arange(llm_grid_h, dtype=np.int32) + start_position

    position_width = np.tile(position_width, llm_grid_h * llm_grid_t)
    position_height = np.repeat(position_height, llm_grid_w)
    position_height = np.tile(position_height, llm_grid_t)
    position_temporal = np.repeat(position_temporal, llm_grid_h * llm_grid_w) + start_position

    return np.stack([position_temporal, position_height, position_width], axis=0)


def _get_rope_index(input_ids, mm_token_type_ids, image_grid_thw, attention_mask,
                    spatial_merge_size):
    """Compute multimodal rotary position ids and mrope deltas for text + vision tokens."""
    bsz, seq_len = input_ids.shape
    position_ids = np.zeros((3, bsz, seq_len), dtype=np.int32)
    mrope_position_deltas = []
    image_iter = iter(image_grid_thw) if image_grid_thw is not None else iter([])

    for b in range(bsz):
        cur_types = mm_token_type_ids[b]
        if attention_mask is not None:
            cur_mask = attention_mask[b].astype(bool)
            cur_types = cur_types[cur_mask]

        input_type_group = []
        for key, group in itertools.groupby(enumerate(cur_types.tolist()), lambda x: x[1]):
            grp = list(group)
            start_index = grp[0][0]
            end_index = grp[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in input_type_group:
            if modality_type == 0:
                text_len = end_idx - start_idx
                text_pos = np.arange(text_len, dtype=np.int32)
                text_pos = np.broadcast_to(text_pos.reshape(1, -1), (3, text_len)) + current_pos
                llm_pos_ids_list.append(text_pos)
                current_pos += text_len
            elif modality_type in (1, 2):
                grid = next(image_iter)
                vision_pos = _get_vision_position_ids(
                    current_pos, grid, spatial_merge_size
                )
                llm_pos_ids_list.append(vision_pos)
                current_pos += max(int(grid[1]), int(grid[2])) // spatial_merge_size
            else:
                raise ValueError(f"Unsupported modality_type: {modality_type}")

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        if attention_mask is not None:
            mask_bool = attention_mask[b].astype(bool)
            position_ids[:, b, mask_bool] = llm_positions
        else:
            position_ids[:, b] = llm_positions

        n_tokens = int(attention_mask[b].sum()) if attention_mask is not None else seq_len
        mrope_position_deltas.append(int(llm_positions.max()) + 1 - n_tokens)

    mrope_position_deltas = np.array(
        mrope_position_deltas, dtype=np.int32
    ).reshape(-1, 1)
    return position_ids, mrope_position_deltas


def _build_position_ids(cfg, input_ids, attention_mask, mm_token_type_ids, image_grid_thw):
    """Build 4D position ids (text + 3 multimodal) and rope deltas for the model."""
    position_ids_3, rope_deltas = _get_rope_index(
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
        spatial_merge_size=cfg.vision_config.spatial_merge_size,
    )
    text_pos = attention_mask.astype(np.int32).cumsum(axis=-1) - 1
    text_pos = np.where(attention_mask == 0, 0, text_pos)
    position_ids_4 = np.concatenate(
        [text_pos.reshape(1, *text_pos.shape), position_ids_3], axis=0
    ).astype(np.int32)
    return position_ids_4, rope_deltas


def _force_processor_image_size(processor, image_size):
    """Override processor image size to match the exported vision model."""
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
        size_pixels = int(image_size) * int(image_size)
        processor.image_processor.size = {
            "shortest_edge": size_pixels,
            "longest_edge": size_pixels,
        }


def _ms_dtype_to_numpy(ms_dtype):
    """Map MindSpore Lite DataType to numpy dtype."""
    dtype_map = {
        mslite.DataType.BOOL: np.bool_,
        mslite.DataType.INT8: np.int8,
        mslite.DataType.INT16: np.int16,
        mslite.DataType.INT32: np.int32,
        mslite.DataType.INT64: np.int64,
        mslite.DataType.FLOAT16: np.float16,
        mslite.DataType.FLOAT32: np.float32,
        mslite.DataType.FLOAT64: np.float64,
    }
    return dtype_map.get(ms_dtype)


def _build_mslite_inputs(model, feed_dict, preferred_order=None):
    """Build MindSpore Lite input tensor list from a feed dict, auto-casting dtypes to match model."""
    inputs = model.get_inputs()
    if not inputs:
        if preferred_order:
            return [mslite.Tensor(feed_dict[k]) for k in preferred_order]
        return [mslite.Tensor(v) for v in feed_dict.values()]

    match_by_name = all(
        getattr(t, "name", None) is not None and t.name in feed_dict
        for t in inputs
    )
    if match_by_name:
        tensors = []
        for t in inputs:
            arr = feed_dict[t.name]
            target_np_dtype = _ms_dtype_to_numpy(t.dtype)
            if target_np_dtype is not None and arr.dtype != target_np_dtype:
                arr = arr.astype(target_np_dtype)
            tensors.append(mslite.Tensor(arr))
        return tensors
    if preferred_order:
        return [mslite.Tensor(feed_dict[k]) for k in preferred_order]
    raise RuntimeError(
        f"input mismatch. model inputs={[getattr(x, 'name', '') for x in inputs]} "
        f"feed keys={list(feed_dict.keys())}"
    )


class Qwen352BInferencer:
    """Qwen3.5-2B MindSpore Lite inferencer with Vision + Prefill + Decode pipeline."""

    def __init__(self, vision_model_path, prefill_model_path, decode_model_path,
                 processor_id, device="ascend", device_id=0, image_size=128):
        """Load MindIR models and processor for Qwen3.5-2B inference."""
        if device not in ["cpu", "ascend"]:
            raise ValueError("device must be cpu or ascend")

        if device == "ascend":
            print("Initializing MindSpore Lite context for Ascend...")
        else:
            print("Initializing MindSpore Lite context for CPU...")

        self.context = mslite.Context()
        self.context.target = [device]
        if device == "ascend":
            self.context.ascend.device_id = device_id

        self.vision_model = mslite.Model()
        self.vision_model.build_from_file(
            vision_model_path, mslite.ModelType.MINDIR, self.context
        )

        self.prefill_model = mslite.Model()
        self.prefill_model.build_from_file(
            prefill_model_path, mslite.ModelType.MINDIR, self.context
        )

        self.decode_model = mslite.Model()
        self.decode_model.build_from_file(
            decode_model_path, mslite.ModelType.MINDIR, self.context
        )
        self.cfg = AutoConfig.from_pretrained(processor_id)
        self.processor = AutoProcessor.from_pretrained(processor_id)
        _force_processor_image_size(self.processor, image_size)

    def _prepare_inputs(self, image_path_or_url, prompt):
        """Tokenize image + text prompt and return numpy arrays for model input."""
        image = _pad_to_square(_load_image(image_path_or_url))
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        input_ids = inputs.input_ids.numpy().astype(np.int32)
        attention_mask = inputs.attention_mask.numpy().astype(np.int32)
        mm_token_type_ids = inputs.mm_token_type_ids.numpy().astype(np.int32)
        pixel_values = inputs.pixel_values.numpy().astype(np.float32)
        image_grid_thw = inputs.image_grid_thw.numpy().astype(np.int32)
        return input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw

    def _decode_generate(self, logits, past_conv, past_recurrent, past_kv,
                         attention_mask, rope_deltas, max_new_tokens, eos_token_id,
                         real_seq_len=None):
        """Autoregressively generate tokens using the decode model until eos or max_new_tokens."""
        generated = []
        last_pos = (real_seq_len - 1) if real_seq_len else -1
        first_token = int(np.argmax(logits[0, last_pos]))
        generated.append(first_token)

        attn_mask_np = attention_mask.astype(np.int32)
        rope_deltas_np = rope_deltas.astype(np.int32)

        decode_times = []
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and generated[-1] == int(eos_token_id):
                break

            step_id = np.array([[generated[-1]]], dtype=np.int32)
            attn_mask_np = np.concatenate(
                [attn_mask_np, np.ones((1, 1), dtype=np.int32)], axis=1
            )
            total_len = int(attn_mask_np.shape[1])

            text_pos_step = np.array([[[total_len - 1]]], dtype=np.int32)
            mm_pos_step = np.broadcast_to(
                text_pos_step + rope_deltas_np.reshape(1, 1, 1), (3, 1, 1)
            ).copy()
            position_ids_step = np.concatenate(
                [text_pos_step, mm_pos_step], axis=0
            ).astype(np.int32)

            decode_feed = {
                "input_ids": step_id,
                "attention_mask": attn_mask_np,
                "position_ids": position_ids_step,
                "past_conv_states": past_conv.astype(np.float32),
                "past_recurrent_states": past_recurrent.astype(np.float32),
                "past_kv_cache": past_kv.astype(np.float32),
            }
            t_step = time.time()
            decode_out = self.decode_model.predict(
                _build_mslite_inputs(self.decode_model, decode_feed,
                                     preferred_order=["input_ids", "attention_mask",
                                                       "position_ids", "past_conv_states",
                                                       "past_recurrent_states", "past_kv_cache"])
            )
            decode_times.append((time.time() - t_step) * 1000)
            logits = decode_out[0].get_data_to_numpy()
            past_conv = decode_out[1].get_data_to_numpy()
            past_recurrent = decode_out[2].get_data_to_numpy()
            past_kv = decode_out[3].get_data_to_numpy()
            generated.append(int(np.argmax(logits[0, -1])))

        total_decode_ms = sum(decode_times)
        avg_decode_ms = total_decode_ms / len(decode_times) if decode_times else 0
        return generated, total_decode_ms, avg_decode_ms, len(decode_times)

    def infer(self, image_path_or_url, text_prompt, max_new_tokens=128):
        """End-to-end inference: vision encoding, prefill, and autoregressive decode."""
        input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw = \
            self._prepare_inputs(image_path_or_url, text_prompt)
        input_seq_len = int(input_ids.shape[1])

        align = 16
        pad_len = (align - input_seq_len % align) % align
        if pad_len > 0:
            pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", 0) or 0
            input_ids = np.concatenate(
                [input_ids, np.full((1, pad_len), pad_token_id, dtype=np.int32)], axis=1
            )
            attention_mask = np.concatenate(
                [attention_mask, np.zeros((1, pad_len), dtype=np.int32)], axis=1
            )
            mm_token_type_ids = np.concatenate(
                [mm_token_type_ids, np.zeros((1, pad_len), dtype=np.int32)], axis=1
            )

        position_ids_4, rope_deltas = _build_position_ids(
            self.cfg, input_ids, attention_mask, mm_token_type_ids, image_grid_thw
        )

        # Vision
        t0 = time.time()
        vision_feed = {"pixel_values": pixel_values}
        vision_out = self.vision_model.predict(
            _build_mslite_inputs(self.vision_model, vision_feed,
                                 preferred_order=["pixel_values"])
        )
        image_embeds = vision_out[0].get_data_to_numpy()
        vision_ms = (time.time() - t0) * 1000

        image_token_cnt = int((input_ids == int(self.cfg.image_token_id)).sum())
        if int(image_embeds.shape[0]) != image_token_cnt:
            raise RuntimeError(
                f"image_embeds length mismatch: embeds={image_embeds.shape[0]} "
                f"vs image_token_cnt={image_token_cnt}"
            )

        # Prefill
        t0 = time.time()
        prefill_feed = {
            "input_ids": input_ids.astype(np.int32),
            "attention_mask": attention_mask.astype(np.int32),
            "position_ids": position_ids_4.astype(np.int32),
            "image_embeds": image_embeds.astype(np.float32),
        }
        prefill_out = self.prefill_model.predict(
            _build_mslite_inputs(self.prefill_model, prefill_feed,
                                 preferred_order=["input_ids", "attention_mask",
                                                   "position_ids", "image_embeds"])
        )
        logits = prefill_out[0].get_data_to_numpy()
        past_conv = prefill_out[1].get_data_to_numpy()
        past_recurrent = prefill_out[2].get_data_to_numpy()
        past_kv = prefill_out[3].get_data_to_numpy()
        prefill_ms = (time.time() - t0) * 1000

        # Decode
        real_seq_len = input_seq_len
        eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        generated, decode_ms, avg_decode_ms, decode_steps = self._decode_generate(
            logits, past_conv, past_recurrent, past_kv,
            attention_mask, rope_deltas, max_new_tokens, eos_token_id,
            real_seq_len=real_seq_len if pad_len > 0 else None,
        )

        total_ms = vision_ms + prefill_ms + decode_ms
        throughput = len(generated) / (total_ms / 1000) if total_ms > 0 else 0

        return {
            "text": self.processor.tokenizer.decode(generated, skip_special_tokens=True),
            "tokens": generated,
            "vision_ms": vision_ms,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "avg_decode_ms": avg_decode_ms,
            "decode_steps": decode_steps,
            "total_ms": total_ms,
            "throughput": throughput,
            "input_seq_len": input_seq_len,
            "padded_seq_len": int(input_ids.shape[1]),
            "pixel_values_shape": list(pixel_values.shape),
            "image_embeds_shape": list(image_embeds.shape),
            "prefill_logits_shape": list(logits.shape),
        }


def main():
    """
    Qwen3.5-2B MindSpore Lite 推理脚本
    """
    parser = argparse.ArgumentParser(
        description="Qwen3.5-2B MindSpore Lite inference (vision + prefill + decode)"
    )
    parser.add_argument("--vision-model", type=str,
                        default="qwen3_5_2b_onnx_final/qwen3_5_vision.mindir",
                        help="Path to qwen3_5_vision.mindir")
    parser.add_argument("--prefill-model", type=str,
                        default="qwen3_5_2b_onnx_final/qwen3_5_llm_prefill_graph.mindir",
                        help="Path to qwen3_5_llm_prefill_graph.mindir")
    parser.add_argument("--decode-model", type=str,
                        default="qwen3_5_2b_onnx_final/qwen3_5_llm_decode_graph.mindir",
                        help="Path to qwen3_5_llm_decode_graph.mindir")
    parser.add_argument("--processor", type=str,
                        default="./Qwen3.5-2B",
                        help="Processor ID or local path")
    parser.add_argument("--image", type=str,
                        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                        help="Image URL or path")
    parser.add_argument("--prompt", type=str, default="Describe this image.", help="Text prompt")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=128,
                        help="Force processor image size (must match exported vision model)")
    parser.add_argument("--device", type=str, default="ascend", choices=["ascend", "cpu"],
                        help="MindSpore Lite target device")
    parser.add_argument("--device-id", type=int, default=0, help="Ascend device ID")

    args = parser.parse_args()

    inferencer = Qwen352BInferencer(
        vision_model_path=args.vision_model,
        prefill_model_path=args.prefill_model,
        decode_model_path=args.decode_model,
        processor_id=args.processor,
        device=args.device,
        device_id=args.device_id,
        image_size=args.image_size,
    )
    result = inferencer.infer(args.image, args.prompt, max_new_tokens=args.max_new_tokens)

    print("\n" + ("=" * 60))
    print(f"Prompt: {args.prompt}")
    print(f"Response: {result['text']}")
    print("=" * 60)
    print("Input shape:")
    print(f"  pixel_values: {result['pixel_values_shape']}")
    print(f"  input_seq_len: {result['input_seq_len']} -> padded: {result['padded_seq_len']}")
    print(f"  image_embeds: {result['image_embeds_shape']}")
    print(f"  prefill logits: {result['prefill_logits_shape']}")
    print(f"Output tokens: {len(result['tokens'])}")
    print("=" * 60)
    print("Performance:")
    print(f"  Vision:  {result['vision_ms']:.2f} ms")
    prefill_info = f"  Prefill: {result['prefill_ms']:.2f} ms (seq_len={result['input_seq_len']})"
    if result['padded_seq_len'] != result['input_seq_len']:
        prefill_info += f" -> padded {result['padded_seq_len']}"
    print(prefill_info)
    print(f"  Decode:  {result['decode_ms']:.2f} ms ({result['decode_steps']} steps, "
          f"avg {result['avg_decode_ms']:.2f} ms/step)")
    print(f"  Total:   {result['total_ms']:.2f} ms")
    print(f"  Throughput: {result['throughput']:.2f} tok/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
