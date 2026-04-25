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
Infer Qwen3.5-0.8B on ONNX Runtime.

Qwen3.5-0.8B is a multimodal VL model with hybrid linear attention
(GatedDeltaNet) and full attention architecture. This script runs
inference using three ONNX models:
  - Vision Tower
  - LLM Prefill (with image_embeds input)
  - LLM Decode (with conv_state + recurrent_state + KV cache)
"""

import argparse
import sys
import urllib.request
from io import BytesIO
import itertools

import numpy as np
import torch
from PIL import Image

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    from transformers import AutoConfig, AutoProcessor
except Exception:
    AutoConfig = None
    AutoProcessor = None


def _load_image(image_path_or_url: str) -> Image.Image:
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
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


def _pick_providers(device: str):
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _get_vision_position_ids(start_position, grid_thw, spatial_merge_size, device):
    """Compute 3D position ids for vision tokens in MRoPE format."""
    llm_grid_t = int(grid_thw[0].item())
    llm_grid_h = int(grid_thw[1].item()) // spatial_merge_size
    llm_grid_w = int(grid_thw[2].item()) // spatial_merge_size

    position_temporal = torch.arange(llm_grid_t, device=device)
    position_width = torch.arange(llm_grid_w, device=device) + start_position
    position_height = torch.arange(llm_grid_h, device=device) + start_position

    position_width = position_width.repeat(llm_grid_h * llm_grid_t)
    position_height = position_height.repeat_interleave(llm_grid_w).repeat(llm_grid_t)
    position_temporal = position_temporal.repeat_interleave(llm_grid_h * llm_grid_w) + start_position

    return torch.stack([position_temporal, position_height, position_width], dim=0)


def _get_rope_index(input_ids, mm_token_type_ids, image_grid_thw, attention_mask,
                    spatial_merge_size):
    """Compute MRoPE position ids and deltas for multimodal input."""
    bsz, seq_len = input_ids.shape
    position_ids = torch.zeros((3, bsz, seq_len), dtype=torch.long, device=input_ids.device)
    mrope_position_deltas = []
    image_iter = iter(image_grid_thw) if image_grid_thw is not None else iter([])

    for b in range(bsz):
        cur_types = mm_token_type_ids[b]
        cur_mask = attention_mask[b].bool() if attention_mask is not None else None
        if cur_mask is not None:
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
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device)
                    .view(1, -1).expand(3, -1) + current_pos
                )
                current_pos += text_len
            elif modality_type in (1, 2):
                grid = next(image_iter)
                vision_pos = _get_vision_position_ids(
                    current_pos, grid, spatial_merge_size, device=input_ids.device
                )
                llm_pos_ids_list.append(vision_pos)
                current_pos += max(int(grid[1].item()), int(grid[2].item())) // spatial_merge_size
            else:
                raise ValueError(f"Unsupported modality_type: {modality_type}")

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if attention_mask is not None:
            position_ids[:, b, attention_mask[b].bool()] = llm_positions.to(position_ids.device)
        else:
            position_ids[:, b] = llm_positions.to(position_ids.device)

        n_tokens = int(attention_mask[b].sum().item()) if attention_mask is not None else seq_len
        mrope_position_deltas.append(llm_positions.max() + 1 - n_tokens)

    mrope_position_deltas = torch.tensor(
        mrope_position_deltas, device=input_ids.device, dtype=torch.long
    ).unsqueeze(1)
    return position_ids, mrope_position_deltas


def _build_position_ids(cfg, input_ids, attention_mask, mm_token_type_ids, image_grid_thw):
    """Build 4D position ids (text_pos + 3D MRoPE) for Qwen3.5 multimodal input."""
    position_ids_3, rope_deltas = _get_rope_index(
        input_ids=input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
        spatial_merge_size=cfg.vision_config.spatial_merge_size,
    )
    text_pos = attention_mask.long().cumsum(-1) - 1
    text_pos = text_pos.masked_fill(attention_mask == 0, 0)
    position_ids_4 = torch.cat([text_pos.unsqueeze(0), position_ids_3], dim=0).to(torch.long)
    return position_ids_4, rope_deltas


def _check_deps():
    if ort is None:
        print("Error: onnxruntime not installed. Install with: pip install onnxruntime (or onnxruntime-gpu).")
        sys.exit(1)
    if AutoProcessor is None or AutoConfig is None:
        print("Error: transformers not installed.")
        sys.exit(1)


def _parse_args():
    """Parse command-line arguments for ONNX inference."""
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B ONNX inference")
    parser.add_argument("--vision", type=str, required=True, help="Path to qwen3_5_vision.onnx")
    parser.add_argument("--prefill", type=str, required=True, help="Path to qwen3_5_llm_prefill.onnx")
    parser.add_argument("--decode", type=str, required=True, help="Path to qwen3_5_llm_decode.onnx")
    parser.add_argument("--processor", type=str,
                        default="/Users/apple/git/models/models_weights/Qwen3.5-0.8B",
                        help="HuggingFace processor id or local path")
    parser.add_argument("--image", type=str, required=True, help="Image path or URL")
    parser.add_argument("--prompt", type=str, default="Describe this image.", help="Text prompt")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=128,
                        help="Force processor image size (must match exported vision ONNX)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def _prepare_inputs(processor, image_path, prompt):
    """Load image and tokenize prompt using processor."""
    image = _pad_to_square(_load_image(image_path))
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(torch.long)
    attention_mask = inputs.attention_mask.to(torch.long)
    mm_token_type_ids = inputs.mm_token_type_ids.to(torch.int64)
    pixel_values = inputs.pixel_values.to(torch.float16)
    image_grid_thw = inputs.image_grid_thw.to(torch.long)
    return input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw


def _create_sessions(vision_path, prefill_path, decode_path, device):
    providers = _pick_providers(device)
    so = ort.SessionOptions()
    vision_sess = ort.InferenceSession(vision_path, sess_options=so, providers=providers)
    prefill_sess = ort.InferenceSession(prefill_path, sess_options=so, providers=providers)
    decode_sess = ort.InferenceSession(decode_path, sess_options=so, providers=providers)
    return vision_sess, prefill_sess, decode_sess


def _run_vision(vision_sess, pixel_values):
    vision_out = vision_sess.run(None, {"pixel_values": pixel_values.cpu().numpy()})
    return vision_out[0]


def _run_prefill(prefill_sess, input_ids, attention_mask, position_ids, image_embeds):
    prefill_out = prefill_sess.run(None, {
        "input_ids": input_ids.cpu().numpy(),
        "attention_mask": attention_mask.cpu().numpy(),
        "position_ids": position_ids.cpu().numpy(),
        "image_embeds": image_embeds,
    })
    return prefill_out[0], prefill_out[1], prefill_out[2], prefill_out[3]


def _decode_generate(decode_sess, logits, past_conv, past_recurrent, past_kv,
                     attention_mask, rope_deltas, max_new_tokens, eos_token_id):
    """Autoregressive decode loop generating tokens one by one."""
    generated = []
    generated.append(int(np.argmax(logits[0, -1])))

    attn_mask_np = attention_mask.cpu().numpy().astype(np.int64)
    rope_deltas_np = rope_deltas.cpu().numpy().astype(np.int64)

    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and generated[-1] == eos_token_id:
            break

        step_id = np.array([[generated[-1]]], dtype=np.int64)
        attn_mask_np = np.concatenate([attn_mask_np, np.ones((1, 1), dtype=np.int64)], axis=1)
        total_len = int(attn_mask_np.shape[1])

        text_pos_step = np.array([[[total_len - 1]]], dtype=np.int64)
        mm_pos_step = (text_pos_step + rope_deltas_np.reshape(1, 1, 1)).repeat(3, axis=0)
        position_ids_step = np.concatenate([text_pos_step, mm_pos_step], axis=0).astype(np.int64)

        decode_out = decode_sess.run(None, {
            "input_ids": step_id,
            "attention_mask": attn_mask_np,
            "position_ids": position_ids_step,
            "past_conv_states": past_conv,
            "past_recurrent_states": past_recurrent,
            "past_kv_cache": past_kv,
        })
        logits = decode_out[0]
        past_conv = decode_out[1]
        past_recurrent = decode_out[2]
        past_kv = decode_out[3]
        generated.append(int(np.argmax(logits[0, -1])))

    return generated


def main():
    args = _parse_args()
    _check_deps()

    cfg = AutoConfig.from_pretrained(args.processor)
    processor = AutoProcessor.from_pretrained(args.processor)

    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
        size_pixels = int(args.image_size) * int(args.image_size)
        processor.image_processor.size = {
            "shortest_edge": size_pixels,
            "longest_edge": size_pixels,
        }

    input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw = \
        _prepare_inputs(processor, args.image, args.prompt)

    position_ids_4, rope_deltas = _build_position_ids(
        cfg, input_ids, attention_mask, mm_token_type_ids, image_grid_thw
    )

    vision_sess, prefill_sess, decode_sess = _create_sessions(
        args.vision, args.prefill, args.decode, args.device
    )

    print("Running vision tower...")
    image_embeds = _run_vision(vision_sess, pixel_values)

    image_token_cnt = int((input_ids == int(cfg.image_token_id)).sum().item())
    if int(image_embeds.shape[0]) != image_token_cnt:
        raise RuntimeError(
            f"image_embeds length mismatch: embeds={image_embeds.shape[0]} "
            f"vs image_token_cnt={image_token_cnt}. grid_thw={image_grid_thw.tolist()}"
        )

    print("Running LLM prefill...")
    logits, past_conv, past_recurrent, past_kv = _run_prefill(
        prefill_sess, input_ids, attention_mask, position_ids_4, image_embeds
    )

    eos_token_id = processor.tokenizer.eos_token_id
    print("Running LLM decode...")
    generated = _decode_generate(
        decode_sess, logits, past_conv, past_recurrent, past_kv,
        attention_mask, rope_deltas, args.max_new_tokens, eos_token_id
    )

    result = processor.tokenizer.decode(generated, skip_special_tokens=True)
    print("\n" + "=" * 50)
    print(f"Input Prompt: {args.prompt}")
    print(f"Generated Response: {result}")
    print("=" * 50)


if __name__ == "__main__":
    main()
