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
Infer Qwen3-VL-4B-Thinking on Ascend with MindSpore Lite.

This script supports the Thinking variant of Qwen3-VL-4B which generates
a chain-of-thought reasoning block before producing the final answer.

The model outputs are structured as:
  <think_start> reasoning_content <think_end> answer_content

Usage:
    python infer_qwen3_vl_4b_thinking_mslite.py \
        --vision-model ./qwen3_vl_4b_thinking_onnx/qwen3_vl_vision.mindir \
        --prefill-model ./qwen3_vl_4b_thinking_onnx/qwen3_vl_llm_prefill_graph.mindir \
        --decode-model ./qwen3_vl_4b_thinking_onnx/qwen3_vl_llm_decode_graph.mindir \
        --processor ./Qwen/Qwen3-VL-4B-Thinking \
        --image https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg \
        --prompt "Describe this image." \
        --max-new-tokens 512 \
        --device ascend \
        --device-id 0
"""

import sys
import argparse
import urllib.request
from io import BytesIO
import time
import numpy as np
from PIL import Image

try:
    import mindspore_lite as mslite
    from transformers import AutoConfig, AutoProcessor
except ImportError:
    print("Error: mindspore_lite or transformers package not found.")
    print("Please install them first.")
    sys.exit(1)


# ============================================================================
# Image utilities
# ============================================================================


def _load_image(image_path_or_url: str) -> Image.Image:
    """Load an image from a local path or HTTP URL.

    Args:
        image_path_or_url: Local file path or http/https URL.

    Returns:
        PIL Image in RGB mode.
    """
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith(
        "https://"
    ):
        with urllib.request.urlopen(image_path_or_url) as resp:
            data = resp.read()
        return Image.open(BytesIO(data)).convert("RGB")
    return Image.open(image_path_or_url).convert("RGB")


def _pad_to_square(image: Image.Image) -> Image.Image:
    """Pad an image to square with black pixels, keeping original content centered."""
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    out = Image.new("RGB", (side, side), (0, 0, 0))
    out.paste(image, ((side - w) // 2, (side - h) // 2))
    return out


# ============================================================================
# Position encoding utilities (MRoPE)
# ============================================================================


def _get_vision_position_ids(start_position, grid_thw, spatial_merge_size, device):
    """Compute 3D vision position IDs for MRoPE encoding.

    Args:
        start_position: Starting position index.
        grid_thw: Grid tensor (t, h, w).
        spatial_merge_size: Spatial merge ratio.
        device: Torch device.

    Returns:
        Tensor of shape (3, image_seq_length) with temporal, height, width positions.
    """
    import torch

    llm_grid_t = int(grid_thw[0].item())
    llm_grid_h = int(grid_thw[1].item()) // spatial_merge_size
    llm_grid_w = int(grid_thw[2].item()) // spatial_merge_size
    image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
    position_width = torch.arange(
        start_position, start_position + llm_grid_w, device=device
    ).repeat(llm_grid_h * llm_grid_t)
    position_height = torch.arange(
        start_position, start_position + llm_grid_h, device=device
    ).repeat_interleave(llm_grid_w * llm_grid_t)
    position_temporal = torch.full(
        (image_seq_length,), start_position, device=device, dtype=torch.long
    )
    return torch.stack([position_temporal, position_height, position_width], dim=0)


def _compute_batch_rope_positions(
    input_ids, mm_token_type_ids, image_grid_thw, attention_mask, spatial_merge_size, batch_idx
):
    """Compute MRoPE position IDs for a single batch element.

    Groups tokens by modality type (text vs image) and computes the appropriate
    position IDs for each group, accumulating the overall position offset.

    Returns:
        tuple: (llm_positions, mrope_delta) for this batch element.
    """
    import torch

    b = batch_idx
    cur_types = mm_token_type_ids[b]
    cur_mask = attention_mask[b].bool() if attention_mask is not None else None
    if cur_mask is not None:
        cur_types = cur_types[cur_mask]
    cur_types_list = cur_types.tolist()

    # Group consecutive tokens by modality type
    groups = []
    start = 0
    for i in range(1, len(cur_types_list) + 1):
        if i == len(cur_types_list) or cur_types_list[i] != cur_types_list[start]:
            groups.append((cur_types_list[start], start, i))
            start = i

    # Compute position IDs for each group
    image_iter = iter(image_grid_thw) if image_grid_thw is not None else iter([])
    current_pos = 0
    llm_pos_ids_list = []
    for modality_type, start_idx, end_idx in groups:
        if modality_type == 0:
            text_len = end_idx - start_idx
            llm_pos_ids_list.append(
                torch.arange(text_len, device=input_ids.device)
                .view(1, -1)
                .expand(3, -1)
                + current_pos
            )
            current_pos += text_len
        elif modality_type == 1:
            grid = next(image_iter)
            vision_pos = _get_vision_position_ids(
                current_pos, grid, spatial_merge_size, device=input_ids.device
            )
            llm_pos_ids_list.append(vision_pos)
            current_pos += (
                max(int(grid[1].item()), int(grid[2].item())) // spatial_merge_size
            )
        else:
            raise ValueError(
                f"Unsupported modality_type in this tutorial: {modality_type}"
            )

    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    n_tokens = (
        int(attention_mask[b].sum().item())
        if attention_mask is not None
        else input_ids.shape[1]
    )
    mrope_delta = llm_positions.max() + 1 - n_tokens
    return llm_positions, mrope_delta


def _get_rope_index(
    input_ids, mm_token_type_ids, image_grid_thw, attention_mask, spatial_merge_size
):
    """Compute MRoPE position IDs and deltas for all batch elements.

    Iterates over batch dimension and delegates per-batch computation
    to _compute_batch_rope_positions.

    Returns:
        tuple: (position_ids, mrope_position_deltas).
    """
    import torch

    bsz, seq_len = input_ids.shape
    position_ids = torch.zeros(
        (3, bsz, seq_len), dtype=torch.long, device=input_ids.device
    )
    mrope_position_deltas = []
    for b in range(bsz):
        llm_positions, mrope_delta = _compute_batch_rope_positions(
            input_ids, mm_token_type_ids, image_grid_thw, attention_mask,
            spatial_merge_size, b
        )

        if attention_mask is not None:
            position_ids[:, b, attention_mask[b].bool()] = llm_positions.to(
                position_ids.device
            )
        else:
            position_ids[:, b] = llm_positions.to(position_ids.device)

        mrope_position_deltas.append(mrope_delta)

    mrope_position_deltas = torch.tensor(
        mrope_position_deltas, device=input_ids.device, dtype=torch.long
    ).unsqueeze(1)
    return position_ids, mrope_position_deltas


def _build_position_ids(
    cfg, input_ids, attention_mask, mm_token_type_ids, image_grid_thw
):
    """Build 4D position IDs combining text and multimodal MRoPE positions.

    Returns:
        tuple: (position_ids_4d, rope_deltas) where position_ids_4d has
               shape (4, batch, seq_len) and rope_deltas has shape (batch, 1).
    """
    import torch

    position_ids_3, rope_deltas = _get_rope_index(
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
        spatial_merge_size=cfg.vision_config.spatial_merge_size,
    )
    text_pos = attention_mask.long().cumsum(-1) - 1
    text_pos = text_pos.masked_fill(attention_mask == 0, 0)
    position_ids_4 = torch.cat([text_pos.unsqueeze(0), position_ids_3], dim=0).to(
        torch.long
    )
    return position_ids_4, rope_deltas


# ============================================================================
# MindSpore Lite utilities
# ============================================================================


def _force_processor_image_size(processor, image_size: int):
    """Force the processor to use a specific image size for preprocessing."""
    if hasattr(processor, "image_processor") and hasattr(
        processor.image_processor, "size"
    ):
        if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            size_pixels = int(image_size[0]) * int(image_size[1])
        else:
            size_pixels = int(image_size) * int(image_size)
        processor.image_processor.size = {
            "shortest_edge": size_pixels,
            "longest_edge": size_pixels,
        }


def _parse_image_size(image_size: str):
    """Parse image size string ('128' or '512x320') into integer or tuple."""
    s = str(image_size).strip().lower()
    if "x" in s:
        h, w = s.split("x", 1)
        return int(h), int(w)
    if "," in s:
        h, w = s.split(",", 1)
        return int(h), int(w)
    return int(s)


def _mslite_tensor(np_array: np.ndarray) -> mslite.Tensor:
    """Create a MindSpore Lite Tensor from a numpy array."""
    return mslite.Tensor(np_array)


def _np_dtype_to_mslite(dtype: np.dtype):
    """Convert numpy dtype to MindSpore Lite DataType enum."""
    dt = np.dtype(dtype)
    if dt == np.dtype(np.float16):
        return mslite.DataType.FLOAT16
    if dt == np.dtype(np.float32):
        return mslite.DataType.FLOAT32
    if dt == np.dtype(np.int32):
        return mslite.DataType.INT32
    if dt == np.dtype(np.int64):
        return mslite.DataType.INT64
    raise TypeError(f"unsupported numpy dtype for mslite.Tensor: {dt}")


def _build_mslite_inputs(model: mslite.Model, feed_dict, preferred_order=None):
    """Build MindSpore Lite input tensors from a feed dictionary.

    Matches inputs by name first; falls back to preferred_order for
    positional matching if names don't align.
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
        "input mismatch. model inputs={[getattr(x, 'name', '') for x in inputs]} "
        "feed keys={list(feed_dict.keys())}"
    )


def _ascend_device_str(device_id: int) -> str:
    """Return the Ascend device string for MindSpore Lite tensor allocation."""
    return f"ascend:{int(device_id)}"


# ============================================================================
# Inferencer class
# ============================================================================


class Qwen3VLThinkingInferencer:
    """
    Qwen3-VL-4B-Thinking inferencer with MindSpore Lite.

    Supports the Thinking variant which generates chain-of-thought reasoning
    before producing the final answer. The output structure is:
        <think_start> reasoning_content <think_end> answer_content
    """

    def __init__(
        self,
        vision_model_path: str,
        prefill_model_path: str,
        decode_model_path: str,
        processor_id: str,
        device: str = "ascend",
        device_id: int = 0,
        image_size: int = 128,
        pad_to_square: bool = True,
    ):
        """Initialize Qwen3-VL-4B-Thinking inferencer with three MSLite models."""
        if device not in ["cpu", "ascend"]:
            raise ValueError("device must be cpu or ascend")

        if device == "ascend":
            print("Initializing MindSpore Lite context for Ascend...")
        else:
            print("Initializing MindSpore Lite context for CPU...")

        self.context = mslite.Context()
        self.context.target = [device]
        self.device = str(device)
        self.device_id = int(device_id)
        if device == "ascend":
            self.context.ascend.device_id = device_id

        self.pad_to_square = bool(pad_to_square)
        self._decode_io_cache = None

        self._load_mslite_models(
            vision_model_path, prefill_model_path, decode_model_path
        )
        self._load_processor(processor_id, image_size)
        self._init_vision_patches(image_size)

    def _load_mslite_models(self, vision_path, prefill_path, decode_path):
        """Load the three MindSpore Lite models (vision, prefill, decode)."""
        print(f"Loading vision model from {vision_path}...")
        self.vision_model = mslite.Model()
        self.vision_model.build_from_file(
            vision_path, mslite.ModelType.MINDIR, self.context
        )

        print(f"Loading prefill model from {prefill_path}...")
        self.prefill_model = mslite.Model()
        self.prefill_model.build_from_file(
            prefill_path, mslite.ModelType.MINDIR, self.context
        )

        print(f"Loading decode model from {decode_path}...")
        self.decode_model = mslite.Model()
        self.decode_model.build_from_file(
            decode_path, mslite.ModelType.MINDIR, self.context
        )

    def _load_processor(self, processor_id, image_size):
        """Load the HuggingFace processor for tokenization and image preprocessing."""
        print(f"Loading processor from {processor_id}...")
        self.cfg = AutoConfig.from_pretrained(processor_id)
        self.processor = AutoProcessor.from_pretrained(processor_id)
        _force_processor_image_size(self.processor, image_size)

    def _init_vision_patches(self, image_size):
        """Determine the expected number of vision patches from model inputs."""
        import math

        inputs = self.vision_model.get_inputs()
        patches = None
        if inputs:
            shape = getattr(inputs[0], "shape", None)
            if shape and len(shape) >= 1 and shape[0] is not None:
                try:
                    patches = int(shape[0])
                except (TypeError, ValueError):
                    patches = None
        if patches is None:
            patch_size = int(self.cfg.vision_config.patch_size)
            if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
                image_h = int(image_size[0])
                image_w = int(image_size[1])
            else:
                image_h = int(image_size)
                image_w = int(image_size)
            grid_h = int(math.ceil(float(image_h) / float(patch_size)))
            grid_w = int(math.ceil(float(image_w) / float(patch_size)))
            patches = int(grid_h) * int(grid_w)
        self.vision_expected_patches = int(patches)

    def _ensure_decode_io(self, max_seq_len: int, past_kv_fixed: np.ndarray):
        """Pre-allocate decode input/output buffers on Ascend for performance.

        Reuses cached buffers if the shape and dtype match.
        """
        if self.device != "ascend":
            return None
        key = (
            int(max_seq_len),
            str(past_kv_fixed.dtype),
            tuple(int(x) for x in past_kv_fixed.shape),
        )
        if self._decode_io_cache is not None and self._decode_io_cache.get("key") == key:
            return self._decode_io_cache
        device_str = _ascend_device_str(self.device_id)

        t_input_ids = mslite.Tensor(
            shape=[1, 1], dtype=mslite.DataType.INT32, device=device_str
        )
        t_attention_mask = mslite.Tensor(
            shape=[1, int(max_seq_len)], dtype=mslite.DataType.INT32, device=device_str
        )
        t_position_ids = mslite.Tensor(
            shape=[4, 1, 1], dtype=mslite.DataType.INT32, device=device_str
        )
        t_cache_pos = mslite.Tensor(shape=[1], dtype=mslite.DataType.INT32, device=device_str)
        t_past_in = mslite.Tensor(
            shape=list(past_kv_fixed.shape),
            dtype=_np_dtype_to_mslite(past_kv_fixed.dtype),
            device=device_str,
        )
        t_past_in.set_data_from_numpy(past_kv_fixed)

        out_bufs = self._alloc_decode_outputs(device_str)
        self._decode_io_cache = {
            "key": key,
            "device": device_str,
            "t_input_ids": t_input_ids,
            "t_attention_mask": t_attention_mask,
            "t_position_ids": t_position_ids,
            "t_cache_pos": t_cache_pos,
            "t_past_in": t_past_in,
            "t_past_out": out_bufs[1] if out_bufs else None,
            "t_logits_out": out_bufs[0] if out_bufs else None,
            "out_bufs": out_bufs,
        }
        return self._decode_io_cache

    def _alloc_decode_outputs(self, device_str):
        """Allocate decode output tensors based on model output shapes.

        Returns:
            list: [logits_tensor, past_kv_tensor] or None if shapes unavailable.
        """
        try:
            outs = self.decode_model.get_outputs()
            if outs and getattr(outs[0], "shape", None) and getattr(outs[1], "shape", None):
                logits_shape = tuple(int(x) for x in outs[0].shape)
                past_shape = tuple(int(x) for x in outs[1].shape)
                if all(x > 0 for x in logits_shape) and all(x > 0 for x in past_shape):
                    t_logits_out = mslite.Tensor(
                        shape=list(logits_shape),
                        dtype=mslite.DataType.FLOAT16,
                        device=device_str,
                    )
                    t_past_out = mslite.Tensor(
                        shape=list(past_shape),
                        dtype=mslite.DataType.FLOAT16,
                        device=device_str,
                    )
                    return [t_logits_out, t_past_out]
        except Exception:
            pass
        return None

    def _prepare_inputs(self, image_path_or_url: str, prompt: str):
        """Prepare tokenized inputs from image and text prompt.

        Returns:
            tuple: (input_ids, attention_mask, mm_token_type_ids,
                    pixel_values, image_grid_thw).
        """
        import torch

        image = _load_image(image_path_or_url)
        if getattr(self, "pad_to_square", True):
            image = _pad_to_square(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(torch.long)
        attention_mask = inputs.attention_mask.to(torch.long)
        mm_token_type_ids = inputs.mm_token_type_ids.to(torch.int64)
        pixel_values = inputs.pixel_values.to(torch.float16)
        image_grid_thw = inputs.image_grid_thw.to(torch.long)
        return (
            input_ids,
            attention_mask,
            mm_token_type_ids,
            pixel_values,
            image_grid_thw,
        )

    def _run_vision_encoder(self, pixel_values_np, image_grid_thw_np):
        """Run vision model and return image embeddings with timing.

        Returns:
            tuple: (image_embeds, deepstack_embeds, elapsed_ms).
        """
        vision_inputs = self.vision_model.get_inputs()
        if len(vision_inputs) == 1:
            feed = {"pixel_values": pixel_values_np}
            preferred = ["pixel_values"]
        else:
            feed = {"pixel_values": pixel_values_np, "grid_thw": image_grid_thw_np}
            preferred = ["pixel_values", "grid_thw"]
        if int(self.vision_expected_patches) != int(pixel_values_np.shape[0]):
            raise RuntimeError(
                f"pixel_values.shape[0]={int(pixel_values_np.shape[0])} does not "
                f"match vision expected {int(self.vision_expected_patches)}. "
                f"image_grid_thw={image_grid_thw_np.tolist()}"
            )
        t0 = time.perf_counter()
        vision_out = self.vision_model.predict(
            _build_mslite_inputs(self.vision_model, feed, preferred_order=preferred)
        )
        t_vision_ms = (time.perf_counter() - t0) * 1000.0
        image_embeds = vision_out[0].get_data_to_numpy()
        deepstack_embeds = vision_out[1].get_data_to_numpy()
        if image_embeds.dtype != np.float16:
            image_embeds = image_embeds.astype(np.float16)
        if deepstack_embeds.dtype != np.float16:
            deepstack_embeds = deepstack_embeds.astype(np.float16)
        return image_embeds, deepstack_embeds, t_vision_ms

    def _run_prefill(self, input_ids, attention_mask, position_ids_4,
                     image_embeds, deepstack_embeds):
        """Run prefill model and return logits with KV cache.

        Returns:
            tuple: (logits, past_kv, elapsed_ms).
        """
        prefill_feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids_4,
            "image_embeds": image_embeds,
            "deepstack_embeds": deepstack_embeds,
        }
        t0 = time.perf_counter()
        prefill_out = self.prefill_model.predict(
            _build_mslite_inputs(
                self.prefill_model,
                prefill_feed,
                preferred_order=[
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "image_embeds",
                    "deepstack_embeds",
                ],
            )
        )
        t_prefill_ms = (time.perf_counter() - t0) * 1000.0
        logits = prefill_out[0].get_data_to_numpy()
        past_kv = prefill_out[1].get_data_to_numpy()
        return logits, past_kv, t_prefill_ms

    def _prepare_decode_step_inputs(self, generated, cache_pos, rope_deltas_np,
                                     attn_mask_fixed, past_kv_fixed):
        """Build the feed dictionary for a single decode step.

        Returns:
            dict: Feed dictionary with input_ids, attention_mask, position_ids,
                  past_key_values, and cache_pos.
        """
        step_id = np.array([[generated[-1]]], dtype=np.int32)
        attn_mask_fixed[0, : cache_pos + 1] = 1
        text_pos_step = np.array([[[cache_pos]]], dtype=np.int32)
        mm_pos_step = (text_pos_step + rope_deltas_np.reshape(1, 1, 1)).repeat(
            3, axis=0
        )
        position_ids_step = np.concatenate(
            [text_pos_step, mm_pos_step], axis=0
        ).astype(np.int32)
        return {
            "input_ids": step_id,
            "attention_mask": attn_mask_fixed,
            "position_ids": position_ids_step,
            "past_key_values": past_kv_fixed.astype(np.float16),
            "cache_pos": np.array([cache_pos], dtype=np.int32),
        }

    def _run_decode_step(self, decode_feed, io):
        """Execute a single decode step on MindSpore Lite.

        Returns:
            tuple: (logits, updated_past_kv, step_time_ms).
        """
        preferred = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "cache_pos",
        ]
        t0 = time.perf_counter()
        if io is not None:
            io["t_input_ids"].set_data_from_numpy(decode_feed["input_ids"])
            io["t_attention_mask"].set_data_from_numpy(decode_feed["attention_mask"])
            io["t_position_ids"].set_data_from_numpy(decode_feed["position_ids"])
            io["t_cache_pos"].set_data_from_numpy(decode_feed["cache_pos"])
            inputs = [
                io["t_input_ids"],
                io["t_attention_mask"],
                io["t_position_ids"],
                io["t_past_in"],
                io["t_cache_pos"],
            ]
            decode_out = self.decode_model.predict(inputs, outputs=io["out_bufs"])
        else:
            decode_out = self.decode_model.predict(
                _build_mslite_inputs(self.decode_model, decode_feed, preferred_order=preferred)
            )
        step_ms = (time.perf_counter() - t0) * 1000.0
        logits = decode_out[0].get_data_to_numpy()
        past_kv = decode_out[1].get_data_to_numpy()
        return logits, past_kv, step_ms

    def _run_decode_loop(self, generated, past_kv_fixed, attn_mask_fixed, cache_pos,
                         rope_deltas_np, eos_token_id, max_new_tokens):
        """Run the autoregressive decode loop.

        Generates tokens one at a time until max_new_tokens, EOS, or
        KV cache exhaustion.

        Returns:
            tuple: (total_decode_ms, decode_steps).
        """
        t_decode_ms = 0.0
        decode_steps = 0
        max_seq_len = attn_mask_fixed.shape[1]
        io = None
        if self.device == "ascend":
            io = self._ensure_decode_io(int(max_seq_len), past_kv_fixed)
            if io is not None:
                io["t_attention_mask"].set_data_from_numpy(attn_mask_fixed)
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and generated[-1] == int(eos_token_id):
                break
            if cache_pos >= max_seq_len:
                break

            decode_feed = self._prepare_decode_step_inputs(
                generated, cache_pos, rope_deltas_np, attn_mask_fixed, past_kv_fixed
            )
            logits, past_kv_fixed, step_ms = self._run_decode_step(decode_feed, io)
            t_decode_ms += step_ms

            if io is not None:
                if io["out_bufs"] is not None:
                    t_prev_in = io["t_past_in"]
                    io["t_past_in"] = past_kv_fixed if isinstance(past_kv_fixed, mslite.Tensor) else io["t_past_in"]
                    io["t_past_in"].set_data_from_numpy(past_kv_fixed)
                    io["t_past_out"] = t_prev_in
                    io["out_bufs"][1] = io["t_past_out"]
                else:
                    io["t_past_in"].set_data_from_numpy(past_kv_fixed)
            else:
                pass

            generated.append(int(np.argmax(logits[0, -1])))
            cache_pos += 1
            decode_steps += 1
        return t_decode_ms, decode_steps

    def _init_decode_state(self, past_kv, attention_mask, rope_deltas, max_seq_len=512):
        """Initialize fixed-size KV cache and attention mask for decode loop.

        Returns:
            tuple: (past_kv_fixed, attn_mask_fixed, cache_pos, rope_deltas_np).
        """
        prompt_len = int(attention_mask.sum().item())
        if prompt_len <= 0:
            raise RuntimeError(f"invalid prompt_len={prompt_len}")
        if prompt_len >= max_seq_len:
            raise RuntimeError(f"prompt_len={prompt_len} exceeds max_seq_len={max_seq_len}")
        if past_kv.ndim != 5:
            raise RuntimeError(f"unexpected past_kv shape: {past_kv.shape}")

        past_kv_fixed = np.zeros(
            (past_kv.shape[0], past_kv.shape[1], past_kv.shape[2], max_seq_len, past_kv.shape[4]),
            dtype=past_kv.dtype,
        )
        past_kv_fixed[:, :, :, : prompt_len, :] = past_kv

        attn_mask_fixed = np.zeros((1, max_seq_len), dtype=np.int32)
        attn_mask_fixed[0, :prompt_len] = 1

        rope_deltas_np = rope_deltas.cpu().numpy().astype(np.int32)
        return past_kv_fixed, attn_mask_fixed, int(prompt_len), rope_deltas_np

    def _parse_thinking_output(self, generated_ids):
        """Parse generated tokens into thinking and answer content.

        The Thinking model wraps reasoning in <|think_start|>...<|think_end|> tokens.
        """
        full_response = self.processor.tokenizer.decode(
            generated_ids, skip_special_tokens=False
        )
        thinking_content = ""
        content = full_response

        think_start_token = "<|think_start|>"
        think_end_token = "<|think_end|>"

        if think_start_token in full_response and think_end_token in full_response:
            think_start_idx = full_response.index(think_start_token) + len(think_start_token)
            think_end_idx = full_response.index(think_end_token)
            thinking_content = full_response[think_start_idx:think_end_idx].strip()
            content = full_response[think_end_idx + len(think_end_token):].strip()

        return thinking_content, content, full_response

    def infer(
        self, image_path_or_url: str, text_prompt: str, max_new_tokens: int = 512
    ):
        """Run end-to-end inference: image → vision → prefill → decode.

        The Thinking model generates:
          <think_start> reasoning <think_end> answer

        Returns:
            dict: Contains 'thinking_content', 'content', 'full_response',
                  and timing information.
        """
        t_e2e_start = time.perf_counter()

        # Phase 1: Prepare inputs and build position IDs
        input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw = (
            self._prepare_inputs(image_path_or_url, text_prompt)
        )
        position_ids_4, rope_deltas = _build_position_ids(
            self.cfg, input_ids, attention_mask, mm_token_type_ids, image_grid_thw
        )
        pixel_values_np = pixel_values.cpu().numpy()
        image_grid_thw_np = image_grid_thw.cpu().numpy().astype(np.int32)

        # Phase 2: Vision encoding
        image_embeds, deepstack_embeds, t_vision_ms = self._run_vision_encoder(
            pixel_values_np, image_grid_thw_np
        )
        image_token_cnt = int((input_ids == int(self.cfg.image_token_id)).sum().item())
        if int(image_embeds.shape[0]) != image_token_cnt:
            raise RuntimeError(
                f"image_embeds length mismatch: embeds={image_embeds.shape[0]} vs "
                f"image_token_cnt={image_token_cnt}. "
                f"grid_thw={image_grid_thw_np.tolist()}"
            )

        # Phase 3: Prefill
        logits, past_kv, t_prefill_ms = self._run_prefill(
            input_ids.cpu().numpy().astype(np.int32),
            attention_mask.cpu().numpy().astype(np.int32),
            position_ids_4.cpu().numpy().astype(np.int32),
            image_embeds,
            deepstack_embeds,
        )

        eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        generated = [int(np.argmax(logits[0, -1]))]

        # Phase 4: Initialize decode state and run loop
        past_kv_fixed, attn_mask_fixed, cache_pos, rope_deltas_np = self._init_decode_state(
            past_kv, attention_mask, rope_deltas
        )
        t_decode_ms, decode_steps = self._run_decode_loop(
            generated, past_kv_fixed, attn_mask_fixed, cache_pos,
            rope_deltas_np, eos_token_id, max_new_tokens
        )

        t_e2e_ms = (time.perf_counter() - t_e2e_start) * 1000.0
        avg_decode_ms = (t_decode_ms / float(decode_steps)) if decode_steps > 0 else 0.0

        # Phase 5: Parse output
        thinking_content, content, full_response = self._parse_thinking_output(generated)

        timing = {
            "vision_ms": t_vision_ms,
            "prefill_ms": t_prefill_ms,
            "decode_total_ms": t_decode_ms,
            "decode_steps": decode_steps,
            "decode_avg_ms": avg_decode_ms,
            "e2e_ms": t_e2e_ms,
        }

        print(
            f"Timing(ms): vision={t_vision_ms:.3f} prefill={t_prefill_ms:.3f} "
            f"decode_total={t_decode_ms:.3f} decode_steps={decode_steps} "
            f"decode_avg={avg_decode_ms:.3f} e2e={t_e2e_ms:.3f}"
        )

        return {
            "thinking_content": thinking_content,
            "content": content,
            "full_response": full_response,
            "timing": timing,
        }


# ============================================================================
# Main
# ============================================================================


def _parse_args():
    """Parse command-line arguments for the inference script."""
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-4B-Thinking MindSpore Lite inference (vision + prefill + decode)"
    )
    parser.add_argument(
        "--vision-model", type=str, required=True, help="Path to qwen3_vl_vision.mindir"
    )
    parser.add_argument(
        "--prefill-model",
        type=str,
        required=True,
        help="Path to qwen3_vl_llm_prefill_graph.mindir",
    )
    parser.add_argument(
        "--decode-model",
        type=str,
        required=True,
        help="Path to qwen3_vl_llm_decode_graph.mindir",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default="./Qwen/Qwen3-VL-4B-Thinking",
        help="Processor ID or path",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        help="Image URL or path (http/https or local path)",
    )
    parser.add_argument(
        "--prompt", type=str, default="Describe this image.", help="Text prompt"
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--image-size",
        type=str,
        default="128",
        help="Force processor image size (must match exported vision model). "
             "Example: 128 or 512x320 or 512,320",
    )
    parser.add_argument(
        "--no-pad-to-square",
        action="store_true",
        help="Disable padding image to square before processor",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="ascend",
        choices=["ascend", "cpu"],
        help="MindSpore Lite target device",
    )
    parser.add_argument("--device-id", type=int, default=0, help="Ascend device ID")
    return parser.parse_args()


def main():
    """Main entry point: parse args, create inferencer, and run inference."""
    args = _parse_args()
    image_size = _parse_image_size(args.image_size)

    inferencer = Qwen3VLThinkingInferencer(
        vision_model_path=args.vision_model,
        prefill_model_path=args.prefill_model,
        decode_model_path=args.decode_model,
        processor_id=args.processor,
        device=args.device,
        device_id=args.device_id,
        image_size=image_size,
        pad_to_square=not bool(args.no_pad_to_square),
    )
    result = inferencer.infer(
        args.image, args.prompt, max_new_tokens=args.max_new_tokens
    )

    print("\n" + "=" * 60)
    print(f"Input Prompt: {args.prompt}")
    print("-" * 60)
    if result["thinking_content"]:
        print(f"Thinking: {result['thinking_content']}")
        print("-" * 60)
    print(f"Response: {result['content']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
