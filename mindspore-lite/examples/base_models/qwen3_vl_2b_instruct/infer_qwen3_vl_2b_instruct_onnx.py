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
Infer Qwen3-VL-2B-Instruct on ONNX.
"""

import argparse
import sys
import urllib.request
from io import BytesIO

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
    """
    Load image from path or URL.
    """
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith(
        "https://"
    ):
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


def _pick_providers(device: str):
    """
    Pick providers for ONNX inference.
    """
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _get_vision_position_ids(
    start_position: int, grid_thw: torch.Tensor, spatial_merge_size: int, device
):
    """
    Get position ids for Qwen3-VL-2B-Instruct.
    """
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
    Get rope index for Qwen3-VL-2B-Instruct.
    """
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


def _parse_args():
    """
    Parse command-line arguments for Qwen3-VL-2B-Instruct ONNX inference.
    """
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-2B-Instruct ONNX inference (vision + prefill + decode)"
    )
    parser.add_argument(
        "--vision", type=str, required=True, help="Path to qwen3_vl_vision.onnx"
    )
    parser.add_argument(
        "--prefill", type=str, required=True, help="Path to qwen3_vl_llm_prefill.onnx"
    )
    parser.add_argument(
        "--decode", type=str, required=True, help="Path to qwen3_vl_llm_decode.onnx"
    )
    parser.add_argument(
        "--processor",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="HuggingFace processor id",
    )
    parser.add_argument("--image", type=str, required=True, help="Image path or URL")
    parser.add_argument(
        "--prompt", type=str, default="Describe this image.", help="Text prompt"
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Force processor image size (must match exported vision ONNX)",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def _check_deps():
    """
    Check dependencies for Qwen3-VL-2B-Instruct ONNX inference.
    """
    if ort is None:
        print(
            "Error: onnxruntime not installed. Install with: pip install onnxruntime (or onnxruntime-gpu)."
        )
        sys.exit(1)
    if AutoProcessor is None or AutoConfig is None:
        print("Error: transformers not installed.")
        sys.exit(1)


def _load_cfg_and_processor(processor_id: str, image_size: int):
    """
    Load Qwen3-VL-2B-Instruct configuration and processor.

    Args:
        processor_id (str): HuggingFace processor id.
        image_size (int): Image size to force in processor.

    Returns:
        cfg (AutoConfig): Qwen3-VL-2B-Instruct configuration.
        processor (AutoProcessor): Qwen3-VL-2B-Instruct processor.
    """
    cfg = AutoConfig.from_pretrained(processor_id)
    processor = AutoProcessor.from_pretrained(processor_id)
    if hasattr(processor, "image_processor") and hasattr(
        processor.image_processor, "size"
    ):
        size_pixels = int(image_size) * int(image_size)
        processor.image_processor.size = {
            "shortest_edge": size_pixels,
            "longest_edge": size_pixels,
        }
    return cfg, processor


def _prepare_inputs(processor, image_path: str, prompt: str):
    """
    Prepare inputs for Qwen3-VL-2B-Instruct ONNX inference.
    """
    image = _pad_to_square(_load_image(image_path))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
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
    return input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw


def _build_position_ids(
    cfg, input_ids, attention_mask, mm_token_type_ids, image_grid_thw
):
    """
    Build position ids for Qwen3-VL-2B-Instruct ONNX inference.

    Args:
        cfg (AutoConfig): Qwen3-VL-2B-Instruct configuration.
        input_ids (torch.Tensor): Input ids.
        attention_mask (torch.Tensor): Attention mask.
        mm_token_type_ids (torch.Tensor): MM token type ids.
        image_grid_thw (torch.Tensor): Image grid thw.

    Returns:
        position_ids_4 (torch.Tensor): Position ids.
        rope_deltas (torch.Tensor): Rope deltas.
    """
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


def _create_sessions(
    vision_path: str, prefill_path: str, decode_path: str, device: str
):
    """
    Create ONNX inference sessions for Qwen3-VL-2B-Instruct ONNX inference.
    """
    providers = _pick_providers(device)
    so = ort.SessionOptions()
    vision_sess = ort.InferenceSession(
        vision_path, sess_options=so, providers=providers
    )
    prefill_sess = ort.InferenceSession(
        prefill_path, sess_options=so, providers=providers
    )
    decode_sess = ort.InferenceSession(
        decode_path, sess_options=so, providers=providers
    )
    return vision_sess, prefill_sess, decode_sess


def _validate_vision_inputs(vision_sess, pixel_values, image_grid_thw):
    """
    Validate vision inputs for Qwen3-VL-2B-Instruct ONNX inference.
    """
    expected_patches = int(vision_sess.get_inputs()[0].shape[0])
    if int(pixel_values.shape[0]) != expected_patches:
        raise RuntimeError(
            f"pixel_values.shape={tuple(pixel_values.shape)} does not match vision expected {expected_patches}."
        )
    if int(image_grid_thw.shape[0]) != 1 or int(image_grid_thw[0, 0].item()) != 1:
        raise RuntimeError(
            f"image_grid_thw={image_grid_thw.tolist()} is not a single image grid."
        )
    if (
        int(image_grid_thw[0, 1].item() * image_grid_thw[0, 2].item())
        != expected_patches
    ):
        raise RuntimeError(
            f"image_grid_thw={image_grid_thw.tolist()} does not match vision expected {expected_patches}."
        )


def _run_vision(vision_sess, pixel_values):
    """
    Run vision inference for Qwen3-VL-2B-Instruct ONNX inference.
    """
    vision_out = vision_sess.run(
        None,
        {
            "pixel_values": pixel_values.cpu().numpy(),
        },
    )
    return vision_out[0], vision_out[1]


def _run_prefill(
    prefill_sess,
    input_ids,
    attention_mask,
    position_ids,
    image_embeds,
    deepstack_embeds,
):
    """
    Run prefill inference for Qwen3-VL-2B-Instruct ONNX inference.
    """
    prefill_out = prefill_sess.run(
        None,
        {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
            "position_ids": position_ids.cpu().numpy(),
            "image_embeds": image_embeds,
            "deepstack_embeds": deepstack_embeds,
        },
    )
    return prefill_out[0], prefill_out[1]


def _decode_generate(
    decode_sess,
    logits,
    past_kv,
    attention_mask,
    rope_deltas,
    max_new_tokens,
    eos_token_id,
):
    """
    Decode and generate for Qwen3-VL-2B-Instruct ONNX inference.

    Args:
        decode_sess (ort.InferenceSession): ONNX inference session for Qwen3-VL-2B-Instruct ONNX inference.
        logits (torch.Tensor): Logits from prefill inference.
        past_kv (list): Past key values from prefill inference.
        attention_mask (torch.Tensor): Attention mask from prefill inference.
        rope_deltas (torch.Tensor): Rope deltas from prefill inference.
        max_new_tokens (int): Maximum number of tokens to generate.
        eos_token_id (int): EOS token id.

    Returns:
        generated (list): Generated token ids.
    """
    generated = []
    generated.append(int(np.argmax(logits[0, -1])))
    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and generated[-1] == eos_token_id:
            break
        step_id = np.array([[generated[-1]]], dtype=np.int64)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=torch.long)], dim=1
        )
        total_len = attention_mask.shape[1]
        text_pos_step = torch.tensor([[total_len - 1]], dtype=torch.long)
        mm_pos_step = (
            (text_pos_step + rope_deltas.to(torch.long)).view(1, 1, 1).expand(3, 1, 1)
        )
        position_ids_step = torch.cat(
            [text_pos_step.view(1, 1, 1), mm_pos_step], dim=0
        ).to(torch.long)
        decode_out = decode_sess.run(
            None,
            {
                "input_ids": step_id,
                "attention_mask": attention_mask.cpu().numpy(),
                "position_ids": position_ids_step.cpu().numpy(),
                "past_key_values": past_kv,
            },
        )
        logits = decode_out[0]
        past_kv = decode_out[1]
        generated.append(int(np.argmax(logits[0, -1])))
    return generated


def main():
    """
    Main function for Qwen3-VL-2B-Instruct ONNX inference.
    """
    args = _parse_args()
    _check_deps()
    cfg, processor = _load_cfg_and_processor(args.processor, args.image_size)
    input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw = (
        _prepare_inputs(processor, args.image, args.prompt)
    )
    position_ids_4, rope_deltas = _build_position_ids(
        cfg, input_ids, attention_mask, mm_token_type_ids, image_grid_thw
    )
    vision_sess, prefill_sess, decode_sess = _create_sessions(
        args.vision, args.prefill, args.decode, args.device
    )
    _validate_vision_inputs(vision_sess, pixel_values, image_grid_thw)
    image_embeds, deepstack_embeds = _run_vision(vision_sess, pixel_values)
    image_token_cnt = int((input_ids == cfg.image_token_id).sum().item())
    if int(image_embeds.shape[0]) != image_token_cnt:
        raise RuntimeError(
            f"image_embeds length mismatch: embeds={image_embeds.shape[0]} vs image_token_cnt={image_token_cnt}. "
            f"grid_thw={image_grid_thw.tolist()}"
        )
    logits, past_kv = _run_prefill(
        prefill_sess,
        input_ids,
        attention_mask,
        position_ids_4,
        image_embeds,
        deepstack_embeds,
    )
    eos_token_id = processor.tokenizer.eos_token_id
    generated = _decode_generate(
        decode_sess,
        logits,
        past_kv,
        attention_mask,
        rope_deltas,
        args.max_new_tokens,
        eos_token_id,
    )
    print(processor.tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
