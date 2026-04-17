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
ONNX Runtime inference for Qwen3-VL-Reranker-2B (vision + score).

The model is exported into two ONNX files:
- Vision: pixel_values -> image_embeds, deepstack_embeds
- Score:  input_ids/attention_mask/position_ids + (image_embeds/deepstack_embeds) -> score (sigmoid, 0..1)
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from io import BytesIO
from typing import List, Optional, Tuple

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


def _pick_providers(device: str) -> List[str]:
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        with urllib.request.urlopen(path_or_url) as resp:
            data = resp.read()
        return Image.open(BytesIO(data)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def _pad_to_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    out = Image.new("RGB", (side, side), (0, 0, 0))
    out.paste(image, ((side - w) // 2, (side - h) // 2))
    return out


def _pad_and_resize(image: Image.Image, image_size: int) -> Image.Image:
    image = _pad_to_square(image)
    size = int(image_size)
    if size <= 0:
        return image
    if image.size == (size, size):
        return image
    resample = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)
    return image.resize((size, size), resample=resample)


def _get_vision_position_ids(
    start_position: int, grid_thw: torch.Tensor, spatial_merge_size: int, device
) -> torch.Tensor:
    """Compute vision position ids from grid_thw for RoPE."""
    llm_grid_t = int(grid_thw[0].item())
    llm_grid_h = int(grid_thw[1].item()) // spatial_merge_size
    llm_grid_w = int(grid_thw[2].item()) // spatial_merge_size
    image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
    position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(
        llm_grid_h * llm_grid_t
    )
    position_height = torch.arange(start_position, start_position + llm_grid_h, device=device).repeat_interleave(
        llm_grid_w * llm_grid_t
    )
    position_temporal = torch.full((image_seq_length,), start_position, device=device, dtype=torch.long)
    return torch.stack([position_temporal, position_height, position_width], dim=0)


def _get_rope_index(
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    spatial_merge_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 3D RoPE position ids for multimodal inputs."""
    bsz, seq_len = input_ids.shape
    position_ids = torch.zeros((3, bsz, seq_len), dtype=torch.long, device=input_ids.device)
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
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                )
                current_pos += text_len
            elif modality_type == 1:
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

    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device, dtype=torch.long).unsqueeze(1)
    return position_ids, mrope_position_deltas


def _build_position_ids(cfg, input_ids, attention_mask, mm_token_type_ids, image_grid_thw):
    """Build 4D position_ids (text + 3D RoPE) and rope_deltas."""
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


def _build_messages(
    instruction: str,
    query_text: str,
    doc_text: str,
    query_image: Optional[Image.Image],
    doc_image: Optional[Image.Image],
):
    """Build chat messages for Qwen3-VL-Reranker processor."""
    system = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
                    'Note that the answer can only be "yes" or "no".'
                ),
            }
        ],
    }

    content = [{"type": "text", "text": f"<Instruct>: {instruction}"}]
    content.append({"type": "text", "text": "<Query>:"})
    if query_image is not None:
        content.append({"type": "image", "image": query_image})
    content.append({"type": "text", "text": query_text if query_text else "NULL"})
    content.append({"type": "text", "text": "\n<Document>:"})
    if doc_image is not None:
        content.append({"type": "image", "image": doc_image})
    content.append({"type": "text", "text": doc_text if doc_text else "NULL"})

    user = {"role": "user", "content": content}
    return [system, user]


def _split_pixel_values(
    pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
) -> List[torch.Tensor]:
    """Split batched pixel_values into per-image chunks by grid_thw."""
    if pixel_values is None or image_grid_thw is None:
        return []
    if pixel_values.ndim == 3:
        return [pixel_values[i] for i in range(int(pixel_values.shape[0]))]
    chunks = []
    offset = 0
    for row in image_grid_thw:
        t, h, w = [int(x) for x in row.tolist()]
        n = int(t * h * w)
        chunks.append(pixel_values[offset : offset + n])
        offset += n
    if int(offset) != int(pixel_values.shape[0]):
        raise RuntimeError(
            f"pixel_values length mismatch: used={offset} "
            f"total={int(pixel_values.shape[0])} grid_thw={image_grid_thw.tolist()}"
        )
    return chunks


def _ort_numpy_dtype(session: "ort.InferenceSession", input_name: str, fallback):
    for inp in session.get_inputs():
        if inp.name != input_name:
            continue
        if inp.type == "tensor(float16)":
            return np.float16
        if inp.type == "tensor(float)":
            return np.float32
        return fallback
    return fallback


def main():
    p = argparse.ArgumentParser(description="Qwen3-VL-Reranker-2B ONNX inference (vision + score)")
    p.add_argument("--vision", type=str, required=True, help="Path to qwen3_vl_reranker_vision.onnx")
    p.add_argument("--score", type=str, required=True, help="Path to qwen3_vl_reranker_score.onnx")
    p.add_argument("--processor", type=str, default="Qwen/Qwen3-VL-Reranker-2B", help="Processor id or local path")
    p.add_argument("--instruction", type=str, default="Retrieve images or text relevant to the user's query.")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--doc", type=str, required=True)
    p.add_argument("--query-image", type=str, default=None)
    p.add_argument("--doc-image", type=str, default=None)
    p.add_argument("--image-size", type=int, default=128, help="Must match --vision-image-size used in export")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    if ort is None:
        print("Error: onnxruntime not installed. Install with: pip install onnxruntime (or onnxruntime-gpu).")
        sys.exit(1)
    if AutoProcessor is None or AutoConfig is None:
        print("Error: transformers not installed or incompatible.")
        sys.exit(1)

    cfg = AutoConfig.from_pretrained(args.processor, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.processor, trust_remote_code=True)

    q_img = _pad_and_resize(_load_image(args.query_image), int(args.image_size)) if args.query_image else None
    d_img = _pad_and_resize(_load_image(args.doc_image), int(args.image_size)) if args.doc_image else None

    messages = _build_messages(args.instruction, args.query, args.doc, q_img, d_img)
    enc = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = enc.input_ids.to(torch.long)
    attention_mask = enc.attention_mask.to(torch.long)

    mm_token_type_ids = getattr(enc, "mm_token_type_ids", None)
    if mm_token_type_ids is None:
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
    else:
        mm_token_type_ids = mm_token_type_ids.to(torch.long)

    pixel_values = getattr(enc, "pixel_values", None)
    image_grid_thw = getattr(enc, "image_grid_thw", None)
    if image_grid_thw is None:
        image_grid_thw = getattr(enc, "grid_thw", None)

    if pixel_values is not None:
        pixel_values = pixel_values.to(torch.float16)
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(torch.long)

    position_ids_4, _ = _build_position_ids(cfg, input_ids, attention_mask, mm_token_type_ids, image_grid_thw)

    providers = _pick_providers(args.device)
    so = ort.SessionOptions()
    vision_sess = ort.InferenceSession(args.vision, sess_options=so, providers=providers)
    score_sess = ort.InferenceSession(args.score, sess_options=so, providers=providers)

    hidden_size = int(getattr(cfg, "hidden_size", 2048))
    num_deepstack = len(getattr(cfg.vision_config, "deepstack_visual_indexes", []))

    image_token_cnt = int((input_ids == int(cfg.image_token_id)).sum().item())

    score_emb_dtype = _ort_numpy_dtype(score_sess, "image_embeds", np.float16)
    score_deep_dtype = _ort_numpy_dtype(score_sess, "deepstack_embeds", score_emb_dtype)

    if image_token_cnt == 0:
        image_embeds = np.zeros((1, hidden_size), dtype=score_emb_dtype)
        deepstack_embeds = np.zeros((num_deepstack, 1, hidden_size), dtype=score_deep_dtype)
    else:
        if pixel_values is None or image_grid_thw is None:
            raise RuntimeError(
                "Found image_token_id in input_ids "
                "but processor did not return pixel_values/image_grid_thw."
            )

        chunks = _split_pixel_values(pixel_values, image_grid_thw)
        expected = vision_sess.get_inputs()[0].shape[0]
        image_embeds_list = []
        deepstack_list = []
        for i, pv in enumerate(chunks):
            if isinstance(expected, int) and int(pv.shape[0]) != int(expected):
                raise RuntimeError(
                    f"Vision ONNX expects pixel_values length {expected}, but image[{i}] has {int(pv.shape[0])}. "
                    f"Ensure --image-size matches export --vision-image-size."
                )
            out = vision_sess.run(None, {"pixel_values": pv.cpu().numpy()})
            image_embeds_list.append(out[0])
            deepstack_list.append(out[1])

        image_embeds = np.concatenate(image_embeds_list, axis=0).astype(score_emb_dtype, copy=False)
        deepstack_embeds = np.concatenate(deepstack_list, axis=1).astype(score_deep_dtype, copy=False)

        if int(image_embeds.shape[0]) != int(image_token_cnt):
            grid_info = image_grid_thw.tolist() if image_grid_thw is not None else None
            raise RuntimeError(
                f"image_embeds length mismatch: "
                f"embeds={int(image_embeds.shape[0])} vs image_token_cnt={image_token_cnt}. "
                f"image_grid_thw={grid_info}"
            )

    score = score_sess.run(
        None,
        {
            "input_ids": input_ids.cpu().numpy().astype(np.int64),
            "attention_mask": attention_mask.cpu().numpy().astype(np.int64),
            "position_ids": position_ids_4.cpu().numpy().astype(np.int64),
            "image_embeds": image_embeds,
            "deepstack_embeds": deepstack_embeds,
        },
    )[0]

    print("=" * 60)
    print(f"score: {float(score.reshape(-1)[0]):.6f}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
