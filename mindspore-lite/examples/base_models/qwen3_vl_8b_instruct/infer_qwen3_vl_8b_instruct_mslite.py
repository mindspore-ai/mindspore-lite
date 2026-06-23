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
Infer Qwen3-VL-8B-Instruct on Ascend with MindSpore Lite.

This script supports the Instruct variant of Qwen3-VL-8B which directly
generates answers without the thinking/reasoning step.

Usage:
    python infer_qwen3_vl_8b_instruct_mslite.py \
        --vision-model ./qwen3_vl_8b_instruct_onnx/qwen3_vl_vision.mindir \
        --prefill-model ./qwen3_vl_8b_instruct_onnx/qwen3_vl_llm_prefill_graph.mindir \
        --decode-model ./qwen3_vl_8b_instruct_onnx/qwen3_vl_llm_decode_graph.mindir \
        --processor ./Qwen3-VL-8B-Instruct \
        --image https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg \
        --prompt "Describe this image." \
        --max-new-tokens 512 \
        --device ascend \
        --device-id 0

    # Multi-card deployment example:
    python infer_qwen3_vl_8b_instruct_mslite.py \
        --vision-model ./qwen3_vl_8b_instruct_onnx/qwen3_vl_vision_graph.mindir \
        --prefill-model ./qwen3_vl_8b_instruct_onnx/qwen3_vl_llm_prefill_graph.mindir \
        --decode-model ./qwen3_vl_8b_instruct_onnx/qwen3_vl_llm_decode_graph.mindir \
        --processor ./Qwen3-VL-8B-Instruct \
        --image https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg \
        --prompt "Describe this image." \
        --max-new-tokens 512 \
        --device ascend \
        --device-id 0 \
        --vision-device-id 0 \
        --prefill-device-id 1 \
        --decode-device-id 1
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


def _load_image(image_path_or_url: str) -> Image.Image:
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith(
        "https://"
    ):
        with urllib.request.urlopen(image_path_or_url) as resp:
            data = resp.read()
        return Image.open(BytesIO(data)).convert("RGB")
    return Image.open(image_path_or_url).convert("RGB")


def _pad_to_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    out = Image.new("RGB", (side, side), (0, 0, 0))
    out.paste(image, ((side - w) // 2, (side - h) // 2))
    return out


def _get_vision_position_ids(start_position, grid_thw, spatial_merge_size, device):
    """
    Get vision position ids for Qwen3-VL-8B-Instruct.
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


def _get_rope_index(
    input_ids, mm_token_type_ids, image_grid_thw, attention_mask, spatial_merge_size
):
    """
    Get rope index for Qwen3-VL-8B-Instruct.
    """
    import torch

    bsz, seq_len = input_ids.shape
    position_ids = torch.zeros(
        (3, bsz, seq_len), dtype=torch.long, device=input_ids.device
    )
    mrope_position_deltas = []
    image_iter = iter(image_grid_thw) if image_grid_thw is not None else iter([])

    for b in range(bsz):
        cur_types = mm_token_type_ids[b]
        cur_mask = attention_mask[b].bool() if attention_mask is not None else None
        if cur_mask is not None:
            cur_types = cur_types[cur_mask]
        cur_types_list = cur_types.tolist()
        groups = []
        start = 0
        for i in range(1, len(cur_types_list) + 1):
            if i == len(cur_types_list) or cur_types_list[i] != cur_types_list[start]:
                groups.append((cur_types_list[start], start, i))
                start = i

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
        if attention_mask is not None:
            position_ids[:, b, attention_mask[b].bool()] = llm_positions.to(
                position_ids.device
            )
        else:
            position_ids[:, b] = llm_positions.to(position_ids.device)

        n_tokens = (
            int(attention_mask[b].sum().item())
            if attention_mask is not None
            else seq_len
        )
        mrope_position_deltas.append(llm_positions.max() + 1 - n_tokens)

    mrope_position_deltas = torch.tensor(
        mrope_position_deltas, device=input_ids.device, dtype=torch.long
    ).unsqueeze(1)
    return position_ids, mrope_position_deltas


def _build_position_ids(
    cfg, input_ids, attention_mask, mm_token_type_ids, image_grid_thw
):
    """
    Build position ids for Qwen3-VL-8B-Instruct.
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


def _force_processor_image_size(processor, image_size: int):
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
    s = str(image_size).strip().lower()
    if "x" in s:
        h, w = s.split("x", 1)
        return int(h), int(w)
    if "," in s:
        h, w = s.split(",", 1)
        return int(h), int(w)
    return int(s)


def _mslite_tensor(np_array: np.ndarray) -> mslite.Tensor:
    return mslite.Tensor(np_array)


def _np_dtype_to_mslite(dtype: np.dtype):
    """Convert numpy dtype to MindSpore Lite DataType."""
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
    """
    Build MindSpore Lite inputs for Qwen3-VL-8B-Instruct.
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


class Qwen3VLInstructInferencer:
    """
    Qwen3-VL-8B-Instruct inferencer with MindSpore Lite.

    The Instruct variant directly generates answers without a thinking step.
    """

    def __init__(
        self,
        vision_model_path: str,
        prefill_model_path: str,
        decode_model_path: str,
        processor_id: str,
        device: str = "ascend",
        device_id: int = 0,
        vision_device_id: int = -1,
        prefill_device_id: int = -1,
        decode_device_id: int = -1,
        prefill_device: str = "",
        image_size: int = 128,
        pad_to_square: bool = True,
    ):
        """
        Initialize Qwen3-VL-8B-Instruct inferencer.

        Supports multi-card deployment: each sub-model can be placed on a
        different Ascend device.  Set ``vision_device_id``, ``prefill_device_id``,
        ``decode_device_id`` to override the global ``device_id`` for each
        sub-model.  Set ``prefill_device="cpu"`` to run prefill on CPU (useful
        when the prefill model exceeds single-card memory).
        """
        if device not in ["cpu", "ascend"]:
            raise ValueError("device must be cpu or ascend")

        # Resolve per-model device ids
        _vid = vision_device_id if vision_device_id >= 0 else device_id
        _pid = prefill_device_id if prefill_device_id >= 0 else device_id
        _did = decode_device_id if decode_device_id >= 0 else device_id
        _pdev = prefill_device if prefill_device else device

        self.device = str(device)
        self.device_id = int(device_id)
        # Resolved per-model device ids (decode I/O buffers must live on the decode
        # model's own device, not the global device_id, or cross-device aclrtMemcpy
        # fails with "input data size is wrong").
        self.vision_device_id = int(_vid)
        self.prefill_device_id = int(_pid)
        self.decode_device_id = int(_did)

        # --- Vision model ---
        print(f"Initializing MindSpore Lite context for vision ({device}:{_vid})...")
        ctx_v = mslite.Context()
        ctx_v.target = [device]
        if device == "ascend":
            ctx_v.ascend.device_id = _vid
        print(f"Loading vision model from {vision_model_path}...")
        self.vision_model = mslite.Model()
        self.vision_model.build_from_file(
            vision_model_path, mslite.ModelType.MINDIR, ctx_v
        )

        # --- Prefill model ---
        print(f"Initializing MindSpore Lite context for prefill ({_pdev}:{_pid})...")
        ctx_p = mslite.Context()
        ctx_p.target = [_pdev]
        if _pdev == "ascend":
            ctx_p.ascend.device_id = _pid
        print(f"Loading prefill model from {prefill_model_path}...")
        self.prefill_model = mslite.Model()
        self.prefill_model.build_from_file(
            prefill_model_path, mslite.ModelType.MINDIR, ctx_p
        )

        # --- Decode model ---
        print(f"Initializing MindSpore Lite context for decode ({device}:{_did})...")
        ctx_d = mslite.Context()
        ctx_d.target = [device]
        if device == "ascend":
            ctx_d.ascend.device_id = _did
        print(f"Loading decode model from {decode_model_path}...")
        self.decode_model = mslite.Model()
        self.decode_model.build_from_file(
            decode_model_path, mslite.ModelType.MINDIR, ctx_d
        )

        self.pad_to_square = bool(pad_to_square)

        # Load processor
        print(f"Loading processor from {processor_id}...")
        self.cfg = AutoConfig.from_pretrained(processor_id)
        self.processor = AutoProcessor.from_pretrained(processor_id)
        _force_processor_image_size(self.processor, image_size)

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
            import math

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
        self._decode_io_cache = None

    def _ascend_device_str(self):
        # Used to allocate decode I/O buffers, which must reside on the decode
        # model's own device (may differ from the global device_id in multi-card
        # deployment).
        return f"ascend:{int(self.decode_device_id)}"

    def _ensure_decode_io(self, max_seq_len: int, past_kv_fixed: np.ndarray):
        """Pre-allocate decode output buffers on Ascend for performance."""
        if self.device != "ascend":
            return None
        key = (
            int(max_seq_len),
            str(past_kv_fixed.dtype),
            tuple(int(x) for x in past_kv_fixed.shape),
        )
        if self._decode_io_cache is not None and self._decode_io_cache.get("key") == key:
            return self._decode_io_cache
        device_str = self._ascend_device_str()
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
        t_logits_out = None
        t_past_out = None
        out_bufs = None
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
                    out_bufs = [t_logits_out, t_past_out]
        except Exception:
            out_bufs = None

        self._decode_io_cache = {
            "key": key,
            "device": device_str,
            "t_input_ids": t_input_ids,
            "t_attention_mask": t_attention_mask,
            "t_position_ids": t_position_ids,
            "t_cache_pos": t_cache_pos,
            "t_past_in": t_past_in,
            "t_past_out": t_past_out,
            "t_logits_out": t_logits_out,
            "out_bufs": out_bufs,
        }
        return self._decode_io_cache

    def _prepare_inputs(self, image_path_or_url: str, prompt: str):
        """
        Prepare inputs for Qwen3-VL-8B-Instruct.
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
        """Run vision model and return (image_embeds, deepstack_embeds, elapsed_ms)."""
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

    def _run_decode_loop(self, generated, past_kv_fixed, attn_mask_fixed, cache_pos,
                         rope_deltas_np, eos_token_id, max_new_tokens):
        """Run the autoregressive decode loop and return updated state."""
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

            step_id = np.array([[generated[-1]]], dtype=np.int32)
            attn_mask_fixed[0, : cache_pos + 1] = 1
            text_pos_step = np.array([[[cache_pos]]], dtype=np.int32)
            mm_pos_step = (text_pos_step + rope_deltas_np.reshape(1, 1, 1)).repeat(
                3, axis=0
            )
            position_ids_step = np.concatenate(
                [text_pos_step, mm_pos_step], axis=0
            ).astype(np.int32)

            t0 = time.perf_counter()
            if io is not None:
                io["t_input_ids"].set_data_from_numpy(step_id)
                io["t_attention_mask"].set_data_from_numpy(attn_mask_fixed)
                io["t_position_ids"].set_data_from_numpy(position_ids_step)
                io["t_cache_pos"].set_data_from_numpy(np.array([cache_pos], dtype=np.int32))
                inputs = [
                    io["t_input_ids"],
                    io["t_attention_mask"],
                    io["t_position_ids"],
                    io["t_past_in"],
                    io["t_cache_pos"],
                ]
                decode_out = self.decode_model.predict(inputs, outputs=io["out_bufs"])
            else:
                decode_feed = {
                    "input_ids": step_id,
                    "attention_mask": attn_mask_fixed,
                    "position_ids": position_ids_step,
                    "past_key_values": past_kv_fixed.astype(np.float16),
                    "cache_pos": np.array([cache_pos], dtype=np.int32),
                }
                decode_out = self.decode_model.predict(
                    _build_mslite_inputs(
                        self.decode_model,
                        decode_feed,
                        preferred_order=[
                            "input_ids",
                            "attention_mask",
                            "position_ids",
                            "past_key_values",
                            "cache_pos",
                        ],
                    )
                )
            t_decode_ms += (time.perf_counter() - t0) * 1000.0
            logits = decode_out[0].get_data_to_numpy()
            if io is not None:
                if io["out_bufs"] is not None:
                    t_prev_in = io["t_past_in"]
                    io["t_past_in"] = decode_out[1]
                    io["t_past_out"] = t_prev_in
                    io["out_bufs"][1] = io["t_past_out"]
                else:
                    io["t_past_in"] = decode_out[1]
            else:
                past_kv_fixed = decode_out[1].get_data_to_numpy()
            generated.append(int(np.argmax(logits[0, -1])))
            cache_pos += 1
            decode_steps += 1
        return t_decode_ms, decode_steps

    def infer(
        self, image_path_or_url: str, text_prompt: str, max_new_tokens: int = 512
    ):
        """
        Infer Qwen3-VL-8B-Instruct on MindSpore Lite.

        The Instruct model generates answers directly without thinking mode.

        Returns:
            dict: Contains 'content', 'full_response', and timing information.
        """
        t_e2e_start = time.perf_counter()
        input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw = (
            self._prepare_inputs(image_path_or_url, text_prompt)
        )
        position_ids_4, rope_deltas = _build_position_ids(
            self.cfg, input_ids, attention_mask, mm_token_type_ids, image_grid_thw
        )

        pixel_values_np = pixel_values.cpu().numpy()
        image_grid_thw_np = image_grid_thw.cpu().numpy().astype(np.int32)

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

        prefill_feed = {
            "input_ids": input_ids.cpu().numpy().astype(np.int32),
            "attention_mask": attention_mask.cpu().numpy().astype(np.int32),
            "position_ids": position_ids_4.cpu().numpy().astype(np.int32),
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

        eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        generated = [int(np.argmax(logits[0, -1]))]

        rope_deltas_np = rope_deltas.cpu().numpy().astype(np.int32)
        prompt_len = int(attention_mask.sum().item())
        max_seq_len = 512
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
        cache_pos = int(prompt_len)

        t_decode_ms, decode_steps = self._run_decode_loop(
            generated, past_kv_fixed, attn_mask_fixed, cache_pos,
            rope_deltas_np, eos_token_id, max_new_tokens
        )

        t_e2e_ms = (time.perf_counter() - t_e2e_start) * 1000.0
        avg_decode_ms = (t_decode_ms / float(decode_steps)) if decode_steps > 0 else 0.0

        # Decode full response
        full_response = self.processor.tokenizer.decode(generated, skip_special_tokens=True)

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
            "content": full_response,
            "full_response": full_response,
            "timing": timing,
        }


def _build_arg_parser():
    """Build the CLI argument parser for Qwen3-VL-8B-Instruct inference."""
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-8B-Instruct MindSpore Lite inference (vision + prefill + decode)"
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
        default="./Qwen3-VL-8B-Instruct",
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
    parser.add_argument(
        "--vision-device-id",
        type=int,
        default=-1,
        help="Ascend device ID for vision model (default: same as --device-id)",
    )
    parser.add_argument(
        "--prefill-device-id",
        type=int,
        default=-1,
        help="Ascend device ID for prefill model (default: same as --device-id)",
    )
    parser.add_argument(
        "--decode-device-id",
        type=int,
        default=-1,
        help="Ascend device ID for decode model (default: same as --device-id)",
    )
    parser.add_argument(
        "--prefill-device",
        type=str,
        default="",
        help="Device for prefill model, e.g. 'cpu' to run prefill on CPU",
    )
    return parser


def main():
    """
    Main function for Qwen3-VL-8B-Instruct inference on Ascend with MindSpore Lite.
    """
    args = _build_arg_parser().parse_args()
    image_size = _parse_image_size(args.image_size)

    inferencer = Qwen3VLInstructInferencer(
        vision_model_path=args.vision_model,
        prefill_model_path=args.prefill_model,
        decode_model_path=args.decode_model,
        processor_id=args.processor,
        device=args.device,
        device_id=args.device_id,
        vision_device_id=args.vision_device_id,
        prefill_device_id=args.prefill_device_id,
        decode_device_id=args.decode_device_id,
        prefill_device=args.prefill_device,
        image_size=image_size,
        pad_to_square=not bool(args.no_pad_to_square),
    )
    result = inferencer.infer(
        args.image, args.prompt, max_new_tokens=args.max_new_tokens
    )

    print("\n" + "=" * 60)
    print(f"Input Prompt: {args.prompt}")
    print("-" * 60)
    print(f"Response: {result['content']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
