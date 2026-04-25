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
MindSpore Lite (Ascend) inference for Qwen3-VL-Reranker-2B (vision + score).

This script:
- Uses MindSpore Lite MindIR models on Ascend
- Uses numpy/PIL + transformers tokenizer/image_processor for preprocessing
- Does NOT use torch

Inputs:
  - Query text (+ optional query image)
  - Document text (+ optional document image)
Output:
  - score in [0, 1]
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import mindspore_lite as mslite  # type: ignore
except Exception:
    mslite = None

try:
    from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer
except Exception:
    AutoConfig = None
    AutoImageProcessor = None
    AutoTokenizer = None


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


def _ms_tensor(arr: np.ndarray) -> "mslite.Tensor":
    return mslite.Tensor(np.ascontiguousarray(arr))


def _run_mslite(model: "mslite.Model", inputs) -> List[np.ndarray]:
    if inputs and isinstance(inputs[0], np.ndarray):
        inputs = [_ms_tensor(x) for x in inputs]
    outs = model.predict(inputs)
    return [t.get_data_to_numpy() for t in outs]


def _build_mslite_inputs(
    model: "mslite.Model",
    feed_dict: dict,
    preferred_order: Optional[List[str]] = None,
) -> list:
    """Build MSLite input tensor list from feed_dict, matching by name or order."""
    inputs = model.get_inputs()
    if not inputs:
        if preferred_order:
            return [_ms_tensor(feed_dict[k]) for k in preferred_order]
        return [_ms_tensor(v) for v in feed_dict.values()]
    ok_by_name = True
    for t in inputs:
        name = getattr(t, "name", None)
        if name is None or name not in feed_dict:
            ok_by_name = False
            break
    if ok_by_name:
        result = []
        for t in inputs:
            arr = feed_dict[t.name]
            result.append(_ms_tensor(arr))
        return result
    if preferred_order:
        return [_ms_tensor(feed_dict[k]) for k in preferred_order]
    model_names = [getattr(t, "name", "") for t in inputs]
    feed_names = list(feed_dict.keys())
    raise RuntimeError(
        f"Input name mismatch. model_inputs={model_names} feed_keys={feed_names}"
    )


def _maybe_get_fixed_seq_len(score_model: "mslite.Model") -> Optional[int]:
    """Try to get fixed sequence length from score model input shape."""
    try:
        inputs = score_model.get_inputs()
    except Exception:
        return None
    for t in inputs or []:
        name = str(getattr(t, "name", "") or "")
        if "input_ids" not in name:
            continue
        shape = getattr(t, "shape", None)
        if shape and len(shape) >= 2:
            try:
                s = int(shape[1])
                if s > 0:
                    return s
            except Exception:
                pass
    return None


def _left_pad_to_len(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    pad_token_id: int,
    target_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Left-pad input_ids and attention_mask to target_len."""
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        return input_ids, attention_mask
    if int(input_ids.shape[0]) != 1:
        return input_ids, attention_mask
    if int(input_ids.shape[1]) == int(target_len):
        return input_ids, attention_mask
    valid = int(attention_mask[0].sum())
    if valid > int(target_len):
        raise RuntimeError(
            f"Tokenized seq_len({valid}) > model fixed seq_len({int(target_len)}). "
            "Please shorten --query/--doc (or export/convert with a larger seq_len if your pipeline fixes shapes)."
        )
    ids = input_ids[0, :valid]
    pad_len = int(target_len) - valid
    out_ids = np.concatenate(
        [np.full((pad_len,), int(pad_token_id), dtype=input_ids.dtype), ids], axis=0
    )[None, :]
    out_mask = np.concatenate(
        [np.zeros((pad_len,), dtype=attention_mask.dtype), np.ones((valid,), dtype=attention_mask.dtype)],
        axis=0,
    )[None, :]
    return out_ids, out_mask


def _build_image_token_block(num_llm_tokens: int) -> str:
    return "<|vision_start|>" + ("<|image_pad|>" * int(num_llm_tokens)) + "<|vision_end|>"


def _build_prompt(
    instruction: str,
    query_text: str,
    doc_text: str,
    query_image_tokens: Optional[str],
    doc_image_tokens: Optional[str],
) -> List[dict]:
    """Build chat prompt messages for Qwen3-VL-Reranker."""
    system_text = (
        'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
        'Note that the answer can only be "yes" or "no".'
    )

    user = [f"<Instruct>: {instruction}", "<Query>:"]
    if query_image_tokens:
        user.append(query_image_tokens)
    user.append(query_text if query_text else "NULL")
    user.append("\n<Document>:")
    if doc_image_tokens:
        user.append(doc_image_tokens)
    user.append(doc_text if doc_text else "NULL")

    user_text = "\n".join(user)
    return [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]


def _get_vision_position_ids_np(
    start_position: int, grid_thw: np.ndarray, spatial_merge_size: int
) -> np.ndarray:
    """Compute vision position ids from grid_thw for RoPE."""
    llm_grid_t = int(grid_thw[0])
    llm_grid_h = int(grid_thw[1]) // spatial_merge_size
    llm_grid_w = int(grid_thw[2]) // spatial_merge_size
    image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
    position_width = np.arange(start_position, start_position + llm_grid_w, dtype=np.int64).repeat(
        llm_grid_h * llm_grid_t
    )
    position_height = np.repeat(
        np.arange(start_position, start_position + llm_grid_h, dtype=np.int64),
        llm_grid_w * llm_grid_t,
    )
    position_temporal = np.full((image_seq_length,), start_position, dtype=np.int64)
    return np.stack([position_temporal, position_height, position_width], axis=0)


def _get_rope_index_np(
    *,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    mm_token_type_ids: Optional[np.ndarray],
    image_grid_thw: Optional[np.ndarray],
    cfg,
) -> np.ndarray:
    """Compute 3D RoPE position ids for multimodal inputs (numpy version)."""
    spatial_merge_size = int(cfg.vision_config.spatial_merge_size)

    bsz, seq_len = input_ids.shape
    position_ids = np.zeros((3, bsz, seq_len), dtype=np.int64)

    if mm_token_type_ids is not None:
        image_iter = iter(image_grid_thw) if image_grid_thw is not None else iter([])
        for b in range(int(bsz)):
            cur_types = mm_token_type_ids[b]
            cur_mask = attention_mask[b].astype(bool)
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
                        np.arange(text_len, dtype=np.int64)[None, :].repeat(3, axis=0) + current_pos
                    )
                    current_pos += text_len
                elif modality_type == 1:
                    grid = next(image_iter)
                    vision_pos = _get_vision_position_ids_np(current_pos, grid, spatial_merge_size)
                    llm_pos_ids_list.append(vision_pos)
                    current_pos += max(int(grid[1]), int(grid[2])) // spatial_merge_size
                else:
                    raise ValueError(f"Unsupported modality_type: {modality_type}")

            llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
            mask_pos = np.where(attention_mask[b] == 1)[0]
            if int(llm_positions.shape[1]) != int(mask_pos.shape[0]):
                raise RuntimeError(
                    f"rope_index length mismatch: got {int(llm_positions.shape[1])} positions, "
                    f"but attention_mask has {int(mask_pos.shape[0])} valid tokens"
                )
            position_ids[:, b, mask_pos] = llm_positions
    else:
        for b in range(int(bsz)):
            valid_len = int(attention_mask[b].sum())
            if valid_len <= 0:
                continue
            arange_pos = np.arange(valid_len, dtype=np.int64)
            mask_pos = np.where(attention_mask[b] == 1)[0]
            position_ids[:, b, mask_pos] = arange_pos[None, :]

    return position_ids


def _build_position_ids_np(
    cfg,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    mm_token_type_ids: Optional[np.ndarray],
    image_grid_thw: Optional[np.ndarray],
) -> np.ndarray:
    """Build 4D position_ids (text + 3D RoPE) for score model."""
    mm_pos = _get_rope_index_np(
        input_ids=input_ids, attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids, image_grid_thw=image_grid_thw, cfg=cfg
    )
    text_pos = np.cumsum(attention_mask.astype(np.int64), axis=-1) - 1
    text_pos = np.where(attention_mask == 0, 0, text_pos)
    pos4 = np.concatenate([text_pos[None, ...], mm_pos], axis=0).astype(np.int64, copy=False)
    return pos4


def _run_vision_inference(args, image_processor, vision_model):
    """Run vision model on query/doc images, return (embeds_list, deepstack_list, grid_thw, vision_ms)."""
    image_embeds_list = []
    deepstack_list = []
    grid_list = []
    vision_total_ms = 0.0

    def handle_one_image(path: str):
        img = _pad_and_resize(_load_image(path), int(args.image_size))
        feats = image_processor.preprocess(img, do_resize=False, return_tensors="np")
        pv = feats["pixel_values"].astype(np.float16, copy=False)
        grid = feats["image_grid_thw"][0].astype(np.int64, copy=False)
        vision_feed = {"pixel_values": pv}
        vision_inputs = _build_mslite_inputs(
            vision_model, vision_feed, preferred_order=["pixel_values"]
        )
        t0 = time.perf_counter()
        out = _run_mslite(vision_model, vision_inputs)
        t1 = time.perf_counter()
        image_embeds_list.append(out[0])
        deepstack_list.append(out[1])
        grid_list.append(grid)
        return (t1 - t0) * 1000.0

    if args.query_image:
        vision_total_ms += handle_one_image(args.query_image)
    if args.doc_image:
        vision_total_ms += handle_one_image(args.doc_image)

    image_grid_thw = np.stack(grid_list, axis=0) if grid_list else None
    return image_embeds_list, deepstack_list, image_grid_thw, vision_total_ms


def _build_image_tokens(args, cfg, image_grid_thw):
    """Compute image token blocks for query and doc from grid_thw."""
    query_img_tokens = None
    doc_img_tokens = None
    spatial = int(cfg.vision_config.spatial_merge_size)
    if args.query_image:
        t, h, w = [int(x) for x in image_grid_thw[0].tolist()]
        num_llm = t * (h // spatial) * (w // spatial)
        query_img_tokens = _build_image_token_block(num_llm)
    if args.doc_image:
        idx = 1 if args.query_image else 0
        t, h, w = [int(x) for x in image_grid_thw[idx].tolist()]
        num_llm = t * (h // spatial) * (w // spatial)
        doc_img_tokens = _build_image_token_block(num_llm)
    return query_img_tokens, doc_img_tokens


def _prepare_score_inputs(
    cfg, tokenizer, score_model, args, image_grid_thw,
    query_img_tokens, doc_img_tokens,
    image_embeds_list, deepstack_list,
):
    """Tokenize prompt, pad, build position_ids, and assemble score model inputs."""
    messages = _build_prompt(
        args.instruction, args.query, args.doc, query_img_tokens, doc_img_tokens
    )
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="np",
    )
    input_ids = enc["input_ids"].astype(np.int64, copy=False)
    attention_mask = enc["attention_mask"].astype(np.int64, copy=False)

    image_token_id = int(cfg.image_token_id)
    mm_token_type_ids = (input_ids == image_token_id).astype(np.int64)

    fixed_len = _maybe_get_fixed_seq_len(score_model)
    if fixed_len is not None:
        pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        input_ids, attention_mask = _left_pad_to_len(
            input_ids, attention_mask, pad_id, int(fixed_len)
        )
        if mm_token_type_ids is not None:
            valid = int(mm_token_type_ids.shape[1]) if mm_token_type_ids.ndim == 2 else 0
            target = int(input_ids.shape[1])
            if 0 < valid < target:
                pad_len = target - valid
                mm_token_type_ids = np.concatenate(
                    [np.zeros((1, pad_len), dtype=mm_token_type_ids.dtype), mm_token_type_ids],
                    axis=1,
                )

    position_ids = _build_position_ids_np(
        cfg, input_ids, attention_mask, mm_token_type_ids, image_grid_thw
    )

    hidden_size = int(getattr(cfg, "hidden_size", 2048))
    num_deepstack = len(getattr(cfg.vision_config, "deepstack_visual_indexes", []))
    image_token_cnt = int((input_ids == image_token_id).sum())

    if image_token_cnt == 0:
        image_embeds = np.zeros((1, hidden_size), dtype=np.float16)
        deepstack_embeds = np.zeros((num_deepstack, 1, hidden_size), dtype=np.float16)
    else:
        image_embeds = np.concatenate(image_embeds_list, axis=0).astype(np.float16, copy=False)
        deepstack_embeds = np.concatenate(deepstack_list, axis=1).astype(np.float16, copy=False)
        if int(image_embeds.shape[0]) != int(image_token_cnt):
            grid_info = image_grid_thw.tolist() if image_grid_thw is not None else None
            raise RuntimeError(
                f"image_embeds length mismatch: "
                f"embeds={int(image_embeds.shape[0])} vs image_token_cnt={image_token_cnt}. "
                f"image_grid_thw={grid_info}"
            )

    score_feed = {
        "input_ids": input_ids.astype(np.int32, copy=False),
        "attention_mask": attention_mask.astype(np.int32, copy=False),
        "position_ids": position_ids.astype(np.int32, copy=False),
        "image_embeds": image_embeds.astype(np.float16, copy=False),
        "deepstack_embeds": deepstack_embeds.astype(np.float16, copy=False),
    }
    preferred_order = [
        "input_ids", "attention_mask", "position_ids",
        "image_embeds", "deepstack_embeds",
    ]
    return _build_mslite_inputs(score_model, score_feed, preferred_order=preferred_order)


def main():
    p = argparse.ArgumentParser(
        description="Qwen3-VL-Reranker-2B MindSpore Lite (Ascend) inference (vision + score)"
    )
    p.add_argument("--vision-model", type=str, required=True, help="Path to qwen3_vl_reranker_vision(.mindir)")
    p.add_argument("--score-model", type=str, required=True, help="Path to qwen3_vl_reranker_score(.mindir)")
    p.add_argument("--processor", type=str, default="Qwen/Qwen3-VL-Reranker-2B", help="Tokenizer/config path or HF id")
    p.add_argument("--instruction", type=str, default="Retrieve images or text relevant to the user's query.")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--doc", type=str, required=True)
    p.add_argument("--query-image", type=str, default=None)
    p.add_argument("--doc-image", type=str, default=None)
    p.add_argument("--image-size", type=int, default=128, help="Must match --vision-image-size used in export")
    p.add_argument("--device-id", type=int, default=0)
    args = p.parse_args()

    if mslite is None:
        print("Error: mindspore_lite not installed.")
        sys.exit(1)
    if AutoTokenizer is None or AutoConfig is None or AutoImageProcessor is None:
        print("Error: transformers not installed or incompatible.")
        sys.exit(1)

    cfg = AutoConfig.from_pretrained(args.processor, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.processor, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(
        args.processor, trust_remote_code=True, use_fast=False
    )

    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = int(args.device_id)

    print(f"Loading vision model from {args.vision_model}...")
    vision_model = mslite.Model()
    vision_model.build_from_file(args.vision_model, mslite.ModelType.MINDIR, context)
    print(f"Loading score model from {args.score_model}...")
    score_model = mslite.Model()
    score_model.build_from_file(args.score_model, mslite.ModelType.MINDIR, context)
    print(f"Loading processor from {args.processor}...")

    t_total_start = time.perf_counter()
    t_preprocess_start = time.perf_counter()

    image_embeds_list, deepstack_list, image_grid_thw, vision_total_ms = _run_vision_inference(
        args, image_processor, vision_model
    )

    query_img_tokens, doc_img_tokens = _build_image_tokens(args, cfg, image_grid_thw)

    score_inputs = _prepare_score_inputs(
        cfg, tokenizer, score_model, args, image_grid_thw,
        query_img_tokens, doc_img_tokens,
        image_embeds_list, deepstack_list,
    )

    t_preprocess_end = time.perf_counter()
    preprocess_ms = (t_preprocess_end - t_preprocess_start) * 1000.0

    t_score_start = time.perf_counter()
    score = _run_mslite(score_model, score_inputs)[0]
    t_score_end = time.perf_counter()
    score_ms = (t_score_end - t_score_start) * 1000.0

    t_total_end = time.perf_counter()
    total_ms = (t_total_end - t_total_start) * 1000.0

    raw_val = float(score.reshape(-1)[0])
    print("=" * 60)
    print(f"score: {raw_val:.6f}")
    print("=" * 60)
    print("\n--- Performance ---")
    if vision_total_ms > 0:
        print(f"  Vision inference:  {vision_total_ms:.2f} ms")
    print(f"  Preprocessing:     {preprocess_ms:.2f} ms")
    print(f"  Score inference:   {score_ms:.2f} ms")
    print(f"  Total:             {total_ms:.2f} ms")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
