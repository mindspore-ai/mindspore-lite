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
Infer Qwen3.5-4B on Ascend with MindSpore Lite.

Qwen3.5-4B is a multimodal VL model with hybrid linear attention
(GatedDeltaNet) and full attention architecture. This script runs
inference using three MindIR models:
  - Vision Tower
  - LLM Prefill (with image_embeds input)
  - LLM Decode (with conv_state + recurrent_state + KV cache)

This script does NOT depend on torch -- all computation uses numpy.
"""

import sys
import argparse
import gc
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
    """
    Load image from local path or URL.
    """
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        with urllib.request.urlopen(image_path_or_url) as resp:
            data = resp.read()
        return Image.open(BytesIO(data)).convert("RGB")
    return Image.open(image_path_or_url).convert("RGB")


def _pad_to_square(image: Image.Image) -> Image.Image:
    """
    Pad image to square.
    """
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    out = Image.new("RGB", (side, side), (0, 0, 0))
    out.paste(image, ((side - w) // 2, (side - h) // 2))
    return out


def _get_vision_position_ids(start_position, grid_thw, spatial_merge_size):
    """
    Get vision position ids.
    """
    llm_grid_t = int(grid_thw[0])
    llm_grid_h = int(grid_thw[1]) // spatial_merge_size
    llm_grid_w = int(grid_thw[2]) // spatial_merge_size

    position_temporal = np.arange(llm_grid_t, dtype=np.int64)
    position_width = np.arange(llm_grid_w, dtype=np.int64) + start_position
    position_height = np.arange(llm_grid_h, dtype=np.int64) + start_position

    position_width = np.tile(position_width, llm_grid_h * llm_grid_t)
    position_height = np.repeat(position_height, llm_grid_w)
    position_height = np.tile(position_height, llm_grid_t)
    position_temporal = np.repeat(position_temporal, llm_grid_h * llm_grid_w) + start_position

    return np.stack([position_temporal, position_height, position_width], axis=0)


def _get_rope_index(input_ids, mm_token_type_ids, image_grid_thw, attention_mask,
                    spatial_merge_size):
    """
    Get rope index.
    """
    bsz, seq_len = input_ids.shape
    position_ids = np.zeros((3, bsz, seq_len), dtype=np.int64)
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
                text_pos = np.arange(text_len, dtype=np.int64)
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
        mrope_position_deltas, dtype=np.int64
    ).reshape(-1, 1)
    return position_ids, mrope_position_deltas


def _build_position_ids(cfg, input_ids, attention_mask, mm_token_type_ids, image_grid_thw):
    """
    Build position ids.
    """
    position_ids_3, rope_deltas = _get_rope_index(
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
        spatial_merge_size=cfg.vision_config.spatial_merge_size,
    )
    text_pos = attention_mask.astype(np.int64).cumsum(axis=-1) - 1
    text_pos = np.where(attention_mask == 0, 0, text_pos)
    position_ids_4 = np.concatenate(
        [text_pos.reshape(1, *text_pos.shape), position_ids_3], axis=0
    ).astype(np.int64)
    return position_ids_4, rope_deltas


def _force_processor_image_size(processor, image_size):
    """
    Force processor image size.
    """
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
        size_pixels = int(image_size) * int(image_size)
        processor.image_processor.size = {
            "shortest_edge": size_pixels,
            "longest_edge": size_pixels,
        }


def _mslite_tensor(np_array):
    """
    Convert numpy array to MindSpore Lite tensor.
    """
    if isinstance(np_array, mslite.Tensor):
        return np_array
    return mslite.Tensor(np_array)


def _device_output_tensor(template, shape, device):
    """Create an Ascend output tensor matching a model output descriptor."""
    tensor = mslite.Tensor(shape=list(shape), dtype=template.dtype, device=device)
    tensor.name = template.name
    tensor.format = template.format
    return tensor


def _build_mslite_inputs(model, feed_dict, preferred_order=None):
    """
    Build MindSpore Lite model inputs.
    """
    inputs = model.get_inputs()
    if not inputs:
        if preferred_order:
            return [_mslite_tensor(feed_dict[k]) for k in preferred_order]
        return [_mslite_tensor(v) for v in feed_dict.values()]
    tensors = []
    ok_by_name = True
    for t in inputs:
        name = getattr(t, "name", None)
        if name is None or name not in feed_dict:
            ok_by_name = False
            break
    if ok_by_name:
        for t in inputs:
            tensors.append(_mslite_tensor(feed_dict[t.name]))
        return tensors
    if preferred_order:
        return [_mslite_tensor(feed_dict[k]) for k in preferred_order]
    raise RuntimeError(
        f"input mismatch. model inputs={[getattr(x, 'name', '') for x in inputs]} "
        f"feed keys={list(feed_dict.keys())}"
    )


def _resize_dynamic_model_inputs(model, feed_dict, preferred_order=None):
    """Resize a dynamic MindSpore Lite model to the current input shapes."""
    inputs = model.get_inputs()
    if not inputs:
        return
    input_names = [getattr(tensor, "name", "") for tensor in inputs]
    if all(name in feed_dict for name in input_names):
        ordered_values = [feed_dict[name] for name in input_names]
    elif preferred_order and len(inputs) == len(preferred_order):
        ordered_values = [feed_dict[name] for name in preferred_order]
    else:
        raise RuntimeError(
            f"input mismatch. model inputs={input_names} "
            f"feed keys={list(feed_dict.keys())}"
        )
    target_shapes = [list(value.shape) for value in ordered_values]
    current_shapes = [list(tensor.shape) for tensor in inputs]
    if current_shapes != target_shapes:
        print(f"Resizing model inputs: {current_shapes} -> {target_shapes}")
        model.resize(inputs, target_shapes)


class Qwen354BInferencer:
    """Qwen3.5-4B MindSpore Lite inferencer with Vision + Prefill + Decode pipeline."""

    def __init__(self, vision_model_path, prefill_model_path, decode_model_path,
                 processor_id, device="ascend", device_id=0, image_size=128,
                 device_resident_states=True):
        """
        Initialize the inferencer.
        """
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
        self.device = device
        self.device_id = device_id
        self.device_resident_states = bool(device_resident_states and device == "ascend")

        # These three large models can exceed host memory when they are built
        # concurrently on a shared server.  Keep only one model alive at a
        # time; the NumPy outputs are sufficient to bridge the stages.
        self.vision_model_path = vision_model_path
        self.prefill_model_path = prefill_model_path
        self.decode_model_path = decode_model_path
        self.vision_model = None
        self.prefill_model = None
        self.decode_model = None
        self.fixed_decode_max_seq_len = None

        print(f"Loading processor from {processor_id}...")
        self.cfg = AutoConfig.from_pretrained(processor_id)
        self.processor = AutoProcessor.from_pretrained(processor_id)
        _force_processor_image_size(self.processor, image_size)

    def _prepare_inputs(self, image_path_or_url, prompt):
        """
        Prepare inputs for the inferencer.
        """
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
        input_ids = inputs.input_ids.numpy().astype(np.int64)
        attention_mask = inputs.attention_mask.numpy().astype(np.int64)
        mm_token_type_ids = inputs.mm_token_type_ids.numpy().astype(np.int64)
        pixel_values = inputs.pixel_values.numpy().astype(np.float16)
        image_grid_thw = inputs.image_grid_thw.numpy().astype(np.int64)
        return input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw

    def _stream_print_token(self, token_id: int):
        """
        Stream print token.
        """
        token_text = self.processor.tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if token_text:
            print(token_text, end="", flush=True)

    def _initialize_decode_device_states(self, logits, past_conv, past_recurrent, past_kv):
        """Upload decode states and allocate reusable device output tensors."""
        if not self.device_resident_states:
            return past_conv, past_recurrent, past_kv, 0.0, None

        device_name = f"ascend:{self.device_id}"
        state_init_start = time.perf_counter()
        past_conv = mslite.Tensor(
            past_conv.astype(np.float16), device=device_name
        )
        past_recurrent = mslite.Tensor(
            past_recurrent.astype(np.float32), device=device_name
        )
        past_kv = mslite.Tensor(
            past_kv.astype(np.float16), device=device_name
        )
        output_templates = self.decode_model.get_outputs()
        if len(output_templates) != 4:
            raise RuntimeError(
                f"expected 4 Decode outputs, got {len(output_templates)}"
            )
        device_outputs = {
            "device_name": device_name,
            "templates": output_templates,
            "conv": [
                _device_output_tensor(output_templates[1], past_conv.shape, device_name),
                _device_output_tensor(output_templates[1], past_conv.shape, device_name),
            ],
            "recurrent": [
                _device_output_tensor(
                    output_templates[2], past_recurrent.shape, device_name
                ),
                _device_output_tensor(
                    output_templates[2], past_recurrent.shape, device_name
                ),
            ],
            "logits": _device_output_tensor(
                output_templates[0], [1, 1, logits.shape[-1]], device_name
            ),
        }
        if self.fixed_decode_max_seq_len is not None:
            device_outputs["kv"] = [
                _device_output_tensor(output_templates[3], past_kv.shape, device_name),
                _device_output_tensor(output_templates[3], past_kv.shape, device_name),
            ]
        device_state_init_ms = (time.perf_counter() - state_init_start) * 1000
        print(f"Decode state upload time: {device_state_init_ms:.2f} ms")
        return past_conv, past_recurrent, past_kv, device_state_init_ms, device_outputs

    def _build_decode_step_io(self, step_id, attention_mask, position_ids,
                              past_conv, past_recurrent, past_kv, step_index,
                              device_outputs):
        """Build one decode step's inputs and optional reusable output tensors."""
        decode_feed = {
            "input_ids": step_id,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_conv_states": past_conv,
            "past_recurrent_states": past_recurrent,
            "past_kv_cache": past_kv,
        }
        if not self.device_resident_states:
            decode_feed["past_conv_states"] = past_conv.astype(np.float16)
            decode_feed["past_recurrent_states"] = past_recurrent.astype(np.float32)
            decode_feed["past_kv_cache"] = past_kv.astype(np.float16)
            return decode_feed, None

        output_index = step_index & 1
        output_templates = device_outputs["templates"]
        if self.fixed_decode_max_seq_len is None:
            kv_output_shape = list(past_kv.shape)
            kv_output_shape[-2] += 1
            kv_output = _device_output_tensor(
                output_templates[3], kv_output_shape, device_outputs["device_name"]
            )
        else:
            kv_output = device_outputs["kv"][output_index]
        output_buffers = [
            device_outputs["logits"],
            device_outputs["conv"][output_index],
            device_outputs["recurrent"][output_index],
            kv_output,
        ]
        return decode_feed, output_buffers

    def _detect_fixed_decode_capacity(self):
        """Return the fixed Decode cache capacity, or None for a growing cache."""
        input_list = self.decode_model.get_inputs()
        output_list = self.decode_model.get_outputs()
        inputs = {getattr(tensor, "name", ""): tensor for tensor in input_list}
        outputs = {getattr(tensor, "name", ""): tensor for tensor in output_list}
        attention = inputs.get("attention_mask")
        past_kv = inputs.get("past_kv_cache")
        present_kv = outputs.get("present_kv_cache")
        if attention is None and len(input_list) > 1:
            attention = input_list[1]
        if past_kv is None and len(input_list) > 5:
            past_kv = input_list[5]
        if present_kv is None and len(output_list) > 3:
            present_kv = output_list[3]
        if attention is None or past_kv is None or present_kv is None:
            return None
        attention_shape = [int(dim) for dim in attention.shape]
        past_shape = [int(dim) for dim in past_kv.shape]
        present_shape = [int(dim) for dim in present_kv.shape]
        if len(attention_shape) < 2 or len(past_shape) < 4 or len(present_shape) < 4:
            return None
        capacity = attention_shape[-1]
        if capacity > 0 and past_shape[-2] == capacity and present_shape[-2] == capacity:
            return capacity
        return None

    def _decode_generate(self, logits, past_conv, past_recurrent, past_kv,
                         attention_mask, rope_deltas, max_new_tokens, eos_token_id,
                         stream=True):
        """
        Decode and generate text.
        """
        generated = []
        generated.append(int(np.argmax(logits[0, -1])))
        if stream:
            self._stream_print_token(generated[-1])

        if self.fixed_decode_max_seq_len is None:
            attn_mask_np = attention_mask.astype(np.int32)
        else:
            if attention_mask.shape[0] != 1:
                raise ValueError("fixed Decode currently supports batch=1 only")
            if attention_mask.shape[1] > self.fixed_decode_max_seq_len:
                raise ValueError(
                    f"prompt length {attention_mask.shape[1]} exceeds fixed Decode "
                    f"capacity {self.fixed_decode_max_seq_len}"
                )
            if int(past_kv.shape[-2]) != self.fixed_decode_max_seq_len:
                raise ValueError(
                    f"Prefill KV capacity {past_kv.shape[-2]} does not match Decode "
                    f"capacity {self.fixed_decode_max_seq_len}"
                )
            if not np.all((attention_mask == 0) | (attention_mask == 1)):
                raise ValueError("fixed Decode attention mask must contain only 0 or 1")
            valid_tokens = int(attention_mask.sum())
            expected_mask = np.arange(attention_mask.shape[1]) < valid_tokens
            if not np.array_equal(attention_mask[0].astype(bool), expected_mask):
                raise ValueError(
                    "fixed Decode requires a contiguous prefix attention mask"
                )
            attn_mask_np = np.zeros(
                (1, self.fixed_decode_max_seq_len), dtype=np.int32
            )
            attn_mask_np[:, :attention_mask.shape[1]] = attention_mask.astype(np.int32)
        rope_deltas_np = rope_deltas.astype(np.int32)

        past_conv, past_recurrent, past_kv, device_state_init_ms, device_outputs = (
            self._initialize_decode_device_states(
                logits, past_conv, past_recurrent, past_kv
            )
        )

        print("Running LLM decode...")
        decode_times = []
        decode_wall_times = []
        for step_index in range(max_new_tokens - 1):
            if eos_token_id is not None and generated[-1] == int(eos_token_id):
                break

            wall_start = time.perf_counter()
            step_id = np.array([[generated[-1]]], dtype=np.int32)
            if self.fixed_decode_max_seq_len is None:
                attn_mask_np = np.concatenate(
                    [attn_mask_np, np.ones((1, 1), dtype=np.int32)], axis=1
                )
                next_pos = int(attn_mask_np.shape[1]) - 1
            else:
                next_pos = int(attn_mask_np.sum())
                if next_pos >= self.fixed_decode_max_seq_len:
                    raise ValueError(
                        f"Decode reached fixed capacity {self.fixed_decode_max_seq_len}"
                    )
                attn_mask_np[0, next_pos] = 1

            text_pos_step = np.array([[[next_pos]]], dtype=np.int32)
            mm_pos_step = np.broadcast_to(
                text_pos_step + rope_deltas_np.reshape(1, 1, 1), (3, 1, 1)
            ).copy()
            position_ids_step = np.concatenate(
                [text_pos_step, mm_pos_step], axis=0
            ).astype(np.int32)

            decode_feed, output_buffers = self._build_decode_step_io(
                step_id, attn_mask_np, position_ids_step,
                past_conv, past_recurrent, past_kv, step_index, device_outputs,
            )
            t_step = time.time()
            decode_inputs = _build_mslite_inputs(
                self.decode_model, decode_feed,
                preferred_order=["input_ids", "attention_mask", "position_ids",
                                 "past_conv_states", "past_recurrent_states",
                                 "past_kv_cache"],
            )
            if output_buffers is None:
                decode_out = self.decode_model.predict(decode_inputs)
            else:
                decode_out = self.decode_model.predict(decode_inputs, output_buffers)
            decode_times.append((time.time() - t_step) * 1000)
            logits = decode_out[0].get_data_to_numpy()
            if self.device_resident_states:
                past_conv, past_recurrent, past_kv = decode_out[1:]
            else:
                past_conv = decode_out[1].get_data_to_numpy()
                past_recurrent = decode_out[2].get_data_to_numpy()
                past_kv = decode_out[3].get_data_to_numpy()
            generated.append(int(np.argmax(logits[0, -1])))
            decode_wall_times.append((time.perf_counter() - wall_start) * 1000)
            if stream:
                self._stream_print_token(generated[-1])

        if stream:
            print()

        total_decode_ms = sum(decode_times)
        avg_decode_ms = total_decode_ms / len(decode_times) if decode_times else 0
        print(f"Total decode time: {total_decode_ms:.2f} ms, "
              f"avg decode step: {avg_decode_ms:.2f} ms, "
              f"steps: {len(decode_times)}")
        total_decode_wall_ms = device_state_init_ms + sum(decode_wall_times)
        avg_decode_wall_ms = (
            sum(decode_wall_times) / len(decode_wall_times) if decode_wall_times else 0
        )
        print(f"Total decode wall time: {total_decode_wall_ms:.2f} ms, "
              f"avg decode wall step: {avg_decode_wall_ms:.2f} ms")
        return generated, total_decode_wall_ms

    def infer(self, image_path_or_url, text_prompt, max_new_tokens=128, stream=True):
        """
        Infer text from image.
        """
        input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw = \
            self._prepare_inputs(image_path_or_url, text_prompt)
        position_ids_4, rope_deltas = _build_position_ids(
            self.cfg, input_ids, attention_mask, mm_token_type_ids, image_grid_thw
        )

        print(f"Loading vision model from {self.vision_model_path}...")
        self.vision_model = mslite.Model()
        try:
            self.vision_model.build_from_file(
                self.vision_model_path, mslite.ModelType.MINDIR, self.context
            )
            print("Running vision tower...")
            t0 = time.time()
            vision_feed = {"pixel_values": pixel_values}
            vision_out = self.vision_model.predict(
                _build_mslite_inputs(self.vision_model, vision_feed,
                                     preferred_order=["pixel_values"])
            )
            image_embeds = vision_out[0].get_data_to_numpy()
            vision_ms = (time.time() - t0) * 1000
            print(f"Vision time: {vision_ms:.2f} ms")
            del vision_out
        finally:
            self.vision_model = None
            gc.collect()

        image_token_cnt = int((input_ids == int(self.cfg.image_token_id)).sum())
        if int(image_embeds.shape[0]) != image_token_cnt:
            raise RuntimeError(
                f"image_embeds length mismatch: embeds={image_embeds.shape[0]} "
                f"vs image_token_cnt={image_token_cnt}"
            )

        print(f"Loading prefill model from {self.prefill_model_path}...")
        self.prefill_model = mslite.Model()
        try:
            self.prefill_model.build_from_file(
                self.prefill_model_path, mslite.ModelType.MINDIR, self.context
            )
            print("Running LLM prefill...")
            t0 = time.time()
            prefill_feed = {
                "input_ids": input_ids.astype(np.int32),
                "attention_mask": attention_mask.astype(np.int32),
                "position_ids": position_ids_4.astype(np.int32),
                "image_embeds": image_embeds.astype(np.float16),
            }
            prefill_order = [
                "input_ids", "attention_mask", "position_ids", "image_embeds"
            ]
            _resize_dynamic_model_inputs(
                self.prefill_model, prefill_feed, preferred_order=prefill_order
            )
            prefill_out = self.prefill_model.predict(
                _build_mslite_inputs(self.prefill_model, prefill_feed,
                                     preferred_order=prefill_order)
            )
            logits = prefill_out[0].get_data_to_numpy()
            past_conv = prefill_out[1].get_data_to_numpy()
            past_recurrent = prefill_out[2].get_data_to_numpy()
            past_kv = prefill_out[3].get_data_to_numpy()
            prefill_ms = (time.time() - t0) * 1000
            print(f"Prefill time: {prefill_ms:.2f} ms")
            del prefill_out
        finally:
            self.prefill_model = None
            gc.collect()

        print(f"Loading decode model from {self.decode_model_path}...")
        self.decode_model = mslite.Model()
        try:
            self.decode_model.build_from_file(
                self.decode_model_path, mslite.ModelType.MINDIR, self.context
            )
            self.fixed_decode_max_seq_len = self._detect_fixed_decode_capacity()
            if self.fixed_decode_max_seq_len is not None:
                print(
                    "Detected fixed Decode cache capacity: "
                    f"{self.fixed_decode_max_seq_len}"
                )
            eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
            generated, total_decode_ms = self._decode_generate(
                logits, past_conv, past_recurrent, past_kv,
                attention_mask, rope_deltas, max_new_tokens, eos_token_id,
                stream=stream,
            )
        finally:
            self.decode_model = None
            self.fixed_decode_max_seq_len = None
            gc.collect()

        total_ms = vision_ms + prefill_ms + total_decode_ms
        throughput = len(generated) / (total_ms / 1000) if total_ms > 0 else 0
        print(f"Total time: {total_ms:.2f} ms, throughput: {throughput:.2f} tok/s")

        return self.processor.tokenizer.decode(generated, skip_special_tokens=True)


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(
        description="Qwen3.5-4B MindSpore Lite inference (vision + prefill + decode)"
    )
    parser.add_argument("--vision-model", type=str, required=True,
                        help="Path to qwen3_5_vision.mindir")
    parser.add_argument("--prefill-model", type=str, required=True,
                        help="Path to qwen3_5_llm_prefill.mindir")
    parser.add_argument("--decode-model", type=str, required=True,
                        help="Path to qwen3_5_llm_decode.mindir")
    parser.add_argument("--processor", type=str,
                        default="./Qwen3.5-4B",
                        help="Processor ID or local path")
    parser.add_argument("--image", type=str,
                        default="./demo.jpeg",
                        help="Image path or URL")
    parser.add_argument("--prompt", type=str, default="Describe this image.", help="Text prompt")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=128,
                        help="Force processor image size (must match exported vision model)")
    parser.add_argument("--device", type=str, default="ascend", choices=["ascend", "cpu"],
                        help="MindSpore Lite target device")
    parser.add_argument("--device-id", type=int, default=0, help="Ascend device ID")
    parser.add_argument(
        "--host-state-roundtrip", action="store_true",
        help="Copy Conv/Recurrent/KV states through NumPy every decode step",
    )

    args = parser.parse_args()

    inferencer = Qwen354BInferencer(
        vision_model_path=args.vision_model,
        prefill_model_path=args.prefill_model,
        decode_model_path=args.decode_model,
        processor_id=args.processor,
        device=args.device,
        device_id=args.device_id,
        image_size=args.image_size,
        device_resident_states=not args.host_state_roundtrip,
    )
    result = inferencer.infer(args.image, args.prompt, max_new_tokens=args.max_new_tokens)

    print("\n" + "=" * 50)
    print(f"Input Prompt: {args.prompt}")
    print(f"Generated Response: {result}")
    print("=" * 50)


if __name__ == "__main__":
    main()
