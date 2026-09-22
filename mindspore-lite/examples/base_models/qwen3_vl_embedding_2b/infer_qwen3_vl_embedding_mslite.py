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
MindSpore Lite (Ascend/CPU) inference for Qwen3-VL-Embedding-2B (vision + text).

The script loads the two MindIR models converted from the ONNX chunks produced by
`export_qwen3_vl_embedding_image_onnx.py`:

  - vision model: `pixel_values [num_patches, 1536]`
                  -> `image_embeds`, `ds0`, `ds1`, `ds2` (each `[num_merged, 2048]`)
  - text model:   `input_ids`, `attention_mask`, `position_ids [3, batch, seq_len]`
                  plus the vision outputs -> `last_hidden_state`

Preprocessing only uses numpy/PIL and the HuggingFace tokenizer/image processor
(no torch). Inputs are wrapped with the instruction-aware prompt used by the
official model (`system{instruction}` + `user{image/text}` and
`add_generation_prompt=True`), and the embeddings are pooled on the last token and
L2-normalized on the host, which matches the official Qwen3-VL-Embedding usage.

Notes:
  - `--image-size` must match `--vision-image-size` used at export time (1024 by
    default), because the vision position/rotary tables are baked into the vision graph.
  - Text-only requests feed a zero `image_embeds`/`ds*` block with as many rows as the
    text model declares (1024 by default) because MindSpore Lite on Ascend does not
    support size-0 tensors; those rows are only read at `<|image_pad|>` positions.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from io import BytesIO
from typing import Dict, List, NamedTuple, Optional, Tuple

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


_NUM_DEEPSTACK = 3
_TEXT_FEATURE_ORDER = ["input_ids", "attention_mask", "position_ids", "image_embeds", "ds0", "ds1", "ds2"]
# Sequence-length buckets (动态分档) compiled into the text model by `ge.dynamicDims` in
# `configs/qwen3_vl_embedding_text_image.ini`. The Ascend runtime only accepts exactly these
# sequence lengths, so prompts are left-padded up to the next bucket; keep the two in sync.
_DEFAULT_SEQ_LEN_BUCKETS = (128, 512, 1024, 2048, 4096)
_MSLITE_NP_DTYPE = {
    "BOOL": np.bool_,
    "INT8": np.int8,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT64": np.int64,
    "UINT8": np.uint8,
    "UINT16": np.uint16,
    "UINT32": np.uint32,
    "UINT64": np.uint64,
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "FLOAT64": np.float64,
}


def _load_image(path_or_url: str) -> Image.Image:
    """Load an image from a local path or URL as RGB."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        with urllib.request.urlopen(path_or_url) as resp:
            data = resp.read()
        return Image.open(BytesIO(data)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def _pad_to_square(image: Image.Image) -> Image.Image:
    """Pad an image to a square canvas with black borders."""
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    out = Image.new("RGB", (side, side), (0, 0, 0))
    out.paste(image, ((side - width) // 2, (side - height) // 2))
    return out


def _pad_and_resize(image: Image.Image, image_size: int) -> Image.Image:
    """Pad an image to a square and resize it to image_size x image_size."""
    image = _pad_to_square(image)
    size = int(image_size)
    if size <= 0 or image.size == (size, size):
        return image
    resample = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)
    return image.resize((size, size), resample=resample)


def _ms_tensor(arr: np.ndarray) -> "mslite.Tensor":
    """Create a MindSpore Lite tensor from a contiguous numpy array."""
    return mslite.Tensor(np.ascontiguousarray(arr))


def _np_dtype_of(data_type) -> Optional[np.dtype]:
    """Map a MindSpore Lite DataType (or its string form) to a numpy dtype."""
    name = getattr(data_type, "name", None)
    if name is None:
        name = str(data_type).rsplit(".", maxsplit=1)[-1]
    return _MSLITE_NP_DTYPE.get(str(name).upper())


def _declared_dtype(tensor) -> Optional[np.dtype]:
    """Return the numpy dtype that a MindIR model declares for one input tensor.

    `mindspore_lite.Tensor` exposes the dtype as `dtype`; the older `data_type` name is
    still accepted. The Ascend backend rejects a tensor whose dtype does not match the
    model, e.g. int64 ids fed to a model that declares int32.
    """
    for attr in ("dtype", "data_type"):
        value = getattr(tensor, attr, None)
        if value is None:
            continue
        dtype = _np_dtype_of(value)
        if dtype is not None:
            return dtype
    return None


def _build_mslite_inputs(model: "mslite.Model", feed_dict: Dict[str, np.ndarray], preferred_order=None) -> list:
    """Build the input tensor list, matching by name and aligning the dtypes."""
    inputs = model.get_inputs()
    if not inputs:
        if preferred_order:
            return [_ms_tensor(feed_dict[name]) for name in preferred_order]
        return [_ms_tensor(value) for value in feed_dict.values()]
    aligned = {}
    for tensor in inputs:
        name = getattr(tensor, "name", None)
        if name is None or name not in feed_dict:
            continue
        arr = feed_dict[name]
        want = _declared_dtype(tensor)
        aligned[name] = arr if want is None or arr.dtype == want else arr.astype(want)
    if len(aligned) == len(inputs):
        return [_ms_tensor(aligned[getattr(t, "name")]) for t in inputs]
    if preferred_order:
        return [_ms_tensor(feed_dict[name]) for name in preferred_order]
    model_names = [getattr(t, "name", "") for t in inputs]
    raise RuntimeError(f"Input name mismatch. model_inputs={model_names} feed_keys={list(feed_dict.keys())}")


def _run_mslite(model: "mslite.Model", inputs: list) -> List[np.ndarray]:
    """Run one forward pass and return the outputs as numpy arrays."""
    if inputs and isinstance(inputs[0], np.ndarray):
        inputs = [_ms_tensor(x) for x in inputs]
    return [t.get_data_to_numpy() for t in model.predict(inputs)]


def _create_context(device: str, device_id: int) -> "mslite.Context":
    """Create the MindSpore Lite context for the requested device."""
    context = mslite.Context()
    context.target = [device]
    if device == "ascend":
        context.ascend.device_id = int(device_id)
    return context


def _load_mslite_model(model_path: str, context: "mslite.Context") -> "mslite.Model":
    """Build a MindIR model on the given context."""
    print(f"Loading model from {model_path}...")
    model = mslite.Model()
    model.build_from_file(model_path, mslite.ModelType.MINDIR, context)
    return model


def _print_model_io(model: "mslite.Model", title: str) -> None:
    """Print the input/output names, shapes and dtypes of a MindIR model."""
    print(f"{title} inputs:")
    for tensor in model.get_inputs():
        dtype = _declared_dtype(tensor)
        print(f"  - {getattr(tensor, 'name', '')}\tshape={getattr(tensor, 'shape', None)}"
              f"\tdtype={np.dtype(dtype).name if dtype is not None else 'unknown'}")


def _maybe_get_fixed_seq_len(text_model: "mslite.Model") -> Optional[int]:
    """Return the fixed sequence length of the text model, if it is static."""
    try:
        inputs = text_model.get_inputs()
    except Exception:
        return None
    for tensor in inputs or []:
        name = str(getattr(tensor, "name", "") or "")
        if "input_ids" not in name:
            continue
        shape = getattr(tensor, "shape", None)
        if shape and len(shape) >= 2:
            try:
                seq_len = int(shape[1])
            except Exception:
                continue
            if seq_len > 0:
                return seq_len
    return None


def _left_pad_to_len(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    pad_token_id: int,
    target_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Left-pad a single sequence to the fixed sequence length of the text model."""
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        return input_ids, attention_mask
    if int(input_ids.shape[1]) == int(target_len):
        return input_ids, attention_mask
    valid = int(attention_mask[0].sum())
    if valid > int(target_len):
        raise RuntimeError(
            f"Tokenized seq_len({valid}) > model fixed seq_len({int(target_len)}). "
            "Please shorten the input text."
        )
    pad_len = int(target_len) - valid
    out_ids = np.concatenate(
        [np.full((pad_len,), int(pad_token_id), dtype=input_ids.dtype), input_ids[0, :valid]], axis=0
    )[None, :]
    out_mask = np.concatenate(
        [np.zeros((pad_len,), dtype=attention_mask.dtype), np.ones((valid,), dtype=attention_mask.dtype)], axis=0
    )[None, :]
    return out_ids, out_mask


def _parse_seq_len_buckets(value: str) -> Tuple[int, ...]:
    """Parse the comma separated sequence-length buckets given on the command line."""
    buckets = [int(item) for item in str(value).replace(",", " ").split()]
    if not buckets or min(buckets) <= 0:
        raise ValueError(f"invalid --seq-len-buckets value: {value!r}")
    return tuple(sorted(set(buckets)))


def _align_to_bucket(valid_len: int, buckets: Tuple[int, ...]) -> int:
    """Return the smallest bucket that can hold `valid_len` tokens.

    The Ascend runtime only accepts the exact sequence lengths compiled into the model, so a
    prompt that does not fill a bucket has to be padded up to the next one.
    """
    for bucket in buckets:
        if int(valid_len) <= int(bucket):
            return int(bucket)
    raise RuntimeError(
        f"the prompt has {valid_len} tokens, which exceeds the largest sequence-length bucket "
        f"{int(buckets[-1])} compiled into the model; add a larger value to `ge.dynamicDims` in "
        "`configs/qwen3_vl_embedding_text_image.ini`, re-convert the text chunk and pass the new "
        "bucket list with --seq-len-buckets."
    )


def _build_image_token_block(num_llm_tokens: int) -> str:
    """Build the vision token block placed in the prompt for one image."""
    return "<|vision_start|>" + ("<|image_pad|>" * int(num_llm_tokens)) + "<|vision_end|>"


class _ImageContext(NamedTuple):
    """Vision outputs of one image, shared by every request that embeds that image."""

    image_tokens: str
    image_grid_thw: np.ndarray
    features: List[np.ndarray]
    vision_ms: float


class _EmbedRequest(NamedTuple):
    """One embedding request: a text plus its optional image context."""

    instruction: str
    text: str
    image_ctx: Optional[_ImageContext] = None


def _num_image_tokens(grid_thw: np.ndarray, spatial_merge_size: int) -> int:
    """Number of merged visual tokens produced by one image grid."""
    grid_t, grid_h, grid_w = [int(x) for x in grid_thw.tolist()]
    return grid_t * (grid_h // int(spatial_merge_size)) * (grid_w // int(spatial_merge_size))


def _vision_position_ids(start_position: int, grid_thw: np.ndarray, spatial_merge_size: int) -> np.ndarray:
    """Compute the 3D (temporal/height/width) position ids of one image grid."""
    grid_t, grid_h, grid_w = [int(x) for x in grid_thw.tolist()]
    llm_grid_h = grid_h // int(spatial_merge_size)
    llm_grid_w = grid_w // int(spatial_merge_size)
    image_seq_length = grid_t * llm_grid_h * llm_grid_w
    # Width varies fastest (np.tile), height is repeated per column (np.repeat); note that
    # torch's `repeat` tiles while numpy's `repeat` repeats each element, so this must not
    # be translated one to one from `get_vision_position_ids`.
    position_width = np.tile(
        np.arange(start_position, start_position + llm_grid_w, dtype=np.int64), llm_grid_h * grid_t
    )
    position_height = np.repeat(
        np.arange(start_position, start_position + llm_grid_h, dtype=np.int64), llm_grid_w * grid_t
    )
    position_temporal = np.repeat(
        np.arange(start_position, start_position + grid_t, dtype=np.int64), llm_grid_h * llm_grid_w
    )
    if position_temporal.shape[0] != image_seq_length:
        raise RuntimeError(f"unexpected vision position id count for grid_thw={grid_thw.tolist()}")
    return np.stack([position_temporal, position_height, position_width], axis=0)


def _build_mrope_position_ids(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    image_token_id: int,
    image_grid_thw: Optional[np.ndarray],
    spatial_merge_size: int,
) -> np.ndarray:
    """Build the 3-row (temporal/height/width) mrope position ids.

    Text tokens use the running text position and image tokens use the 3D grid
    positions starting at the current text position, matching Qwen3-VL's
    `get_rope_index`. One sequence with at most one contiguous image block.
    """
    batch_size, seq_len = input_ids.shape
    position_ids = np.zeros((3, int(batch_size), int(seq_len)), dtype=np.int64)
    for batch_idx in range(int(batch_size)):
        valid_pos = np.where(attention_mask[batch_idx] == 1)[0]
        ids = input_ids[batch_idx][valid_pos]
        image_mask = ids == int(image_token_id)
        image_cnt = int(image_mask.sum())
        if image_cnt == 0:
            position_ids[:, batch_idx, valid_pos] = np.arange(ids.shape[0], dtype=np.int64)[None, :]
            continue
        if image_grid_thw is None or int(image_grid_thw.shape[0]) != 1:
            raise RuntimeError("expected exactly one image_grid_thw row when the prompt has image tokens")
        image_pos = np.where(image_mask)[0]
        if image_cnt != int(image_pos[-1] - image_pos[0]) + 1:
            raise RuntimeError("the <|image_pad|> tokens of one image must be contiguous")
        grid_thw = image_grid_thw[0]
        num_llm_tokens = _num_image_tokens(grid_thw, spatial_merge_size)
        if num_llm_tokens != image_cnt:
            raise RuntimeError(
                f"image token count mismatch: prompt has {image_cnt} <|image_pad|> tokens but grid "
                f"{grid_thw.tolist()} (spatial_merge_size={int(spatial_merge_size)}) gives {num_llm_tokens}"
            )
        image_start = int(image_pos[0])
        image_end = image_start + image_cnt
        grid_h = int(grid_thw[1]) // int(spatial_merge_size)
        grid_w = int(grid_thw[2]) // int(spatial_merge_size)
        positions = np.arange(ids.shape[0], dtype=np.int64)
        positions[image_end:] = positions[image_end:] - image_cnt + max(grid_h, grid_w)
        cur_positions = np.repeat(positions[None, :], 3, axis=0)
        cur_positions[:, image_start:image_end] = _vision_position_ids(image_start, grid_thw, spatial_merge_size)
        position_ids[:, batch_idx, valid_pos] = cur_positions
    return position_ids


def _build_messages(instruction: str, text: str, image_tokens: Optional[str]) -> List[dict]:
    """Build the instruction-aware chat messages used by Qwen3-VL-Embedding."""
    content = []
    if image_tokens:
        content.append(image_tokens)
    if text:
        content.append(text)
    if not content:
        content.append("")
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": "\n".join(content)},
    ]


def _tokenize_prompt(
    tokenizer,
    text_model: "mslite.Model",
    request: _EmbedRequest,
    seq_len_buckets: Optional[Tuple[int, ...]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tokenize one request and left-pad it to a sequence length the model accepts.

    Static models use their fixed sequence length; dynamic ones are padded up to the next
    sequence-length bucket, because the Ascend runtime only accepts the exact lengths that
    were compiled into the model.
    """
    image_tokens = request.image_ctx.image_tokens if request.image_ctx is not None else None
    messages = _build_messages(request.instruction, request.text, image_tokens)
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="np",
    )
    input_ids = np.asarray(enc["input_ids"], dtype=np.int64)
    attention_mask = np.asarray(enc["attention_mask"], dtype=np.int64)
    fixed_len = _maybe_get_fixed_seq_len(text_model)
    if fixed_len is None:
        valid_len = int(attention_mask[0].sum()) if attention_mask.ndim == 2 else int(input_ids.shape[1])
        fixed_len = _align_to_bucket(valid_len, tuple(seq_len_buckets or _DEFAULT_SEQ_LEN_BUCKETS))
    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    input_ids, attention_mask = _left_pad_to_len(input_ids, attention_mask, pad_token_id, int(fixed_len))
    return input_ids, attention_mask


def _feature_input_rows(text_model: "mslite.Model", name: str = "image_embeds") -> Optional[int]:
    """Return the static row count declared for a feature input, or None when it is dynamic."""
    for model_input in text_model.get_inputs():
        if model_input.name != name:
            continue
        dims = [int(dim) for dim in model_input.shape]
        if len(dims) == 2 and dims[0] > 0:
            return dims[0]
        return None
    return None


def _build_feature_inputs(
    cfg,
    input_ids: np.ndarray,
    request: _EmbedRequest,
    text_model: "mslite.Model",
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return the vision features of one request, or a dummy block when it has no image.

    MindSpore Lite on Ascend rejects size-0 tensors, so a request without an image feeds an
    all-zero feature block whose row count matches the static shape of the model input. The
    dummy rows are only read at `<|image_pad|>` positions, which do not exist without an
    image, so they never influence the result.
    """
    image_token_id = int(cfg.image_token_id)
    hidden_size = int(getattr(cfg.text_config, "hidden_size", 2048))
    expected_rows = _feature_input_rows(text_model)
    image_token_cnt = int((input_ids == image_token_id).sum())
    if request.image_ctx is None:
        if image_token_cnt != 0:
            raise RuntimeError("the prompt contains image tokens but no vision features were computed")
        dummy = np.zeros((expected_rows or 1, hidden_size), dtype=np.float32)
        return dummy, [dummy.copy() for _ in range(_NUM_DEEPSTACK)]
    image_embeds = request.image_ctx.features[0]
    deepstack = list(request.image_ctx.features[1:1 + _NUM_DEEPSTACK])
    if int(image_embeds.shape[0]) != image_token_cnt:
        raise RuntimeError(
            f"image feature count mismatch: vision returned {int(image_embeds.shape[0])} rows but the "
            f"prompt has {image_token_cnt} <|image_pad|> tokens. Check that --image-size matches the "
            "--vision-image-size used at export time."
        )
    if expected_rows is not None and int(image_embeds.shape[0]) != expected_rows:
        raise RuntimeError(
            f"the text model expects {expected_rows} image feature rows but the vision model returned "
            f"{int(image_embeds.shape[0])}; export both chunks with the same --vision-image-size."
        )
    return image_embeds, deepstack


def _prepare_text_feed(
    cfg,
    tokenizer,
    text_model: "mslite.Model",
    request: _EmbedRequest,
    seq_len_buckets: Optional[Tuple[int, ...]] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Build the text model feed (ids, mask, mrope position ids and image features) of one request."""
    input_ids, attention_mask = _tokenize_prompt(tokenizer, text_model, request, seq_len_buckets)
    image_grid_thw = request.image_ctx.image_grid_thw if request.image_ctx is not None else None
    position_ids = _build_mrope_position_ids(
        input_ids,
        attention_mask,
        int(cfg.image_token_id),
        image_grid_thw,
        int(cfg.vision_config.spatial_merge_size),
    )
    image_embeds, deepstack = _build_feature_inputs(cfg, input_ids, request, text_model)
    feed = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "image_embeds": image_embeds,
        "ds0": deepstack[0],
        "ds1": deepstack[1],
        "ds2": deepstack[2],
    }
    return feed, attention_mask


def _last_token_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Pool the last non-padding token of each sequence.

    The prompts are left-padded, so the last token of a sequence sits at the last
    position and `attention_mask.sum(axis=1) - 1` (which only holds for right
    padding) would pool a padding position whose attention output is undefined.
    """
    mask = np.asarray(attention_mask)
    if mask.ndim != 2 or mask.shape[0] != last_hidden_state.shape[0]:
        raise RuntimeError(f"attention_mask {mask.shape} does not match hidden states {last_hidden_state.shape}")
    last_idx = mask.shape[1] - 1 - np.argmax(mask[:, ::-1] > 0, axis=1)
    rows = np.arange(mask.shape[0])
    if not np.all(mask[rows, last_idx] > 0):
        raise RuntimeError("every sequence must contain at least one non-padding token")
    return last_hidden_state[rows, last_idx]


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings on the host (kept out of the graph on purpose)."""
    denom = np.clip(np.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-12, None)
    return (embeddings / denom).astype(np.float32, copy=False)


def _run_vision(
    cfg,
    vision_model: "mslite.Model",
    image_processor,
    image_path: str,
    image_size: int,
) -> _ImageContext:
    """Run the vision model on one image and return its tokens, grid, features and latency."""
    image = _pad_and_resize(_load_image(image_path), int(image_size))
    feats = image_processor.preprocess(image, do_resize=False, return_tensors="np")
    grid_thw = np.asarray(feats["image_grid_thw"], dtype=np.int64)
    feed = {"pixel_values": np.asarray(feats["pixel_values"])}
    inputs = _build_mslite_inputs(vision_model, feed, preferred_order=["pixel_values"])
    start = time.perf_counter()
    features = _run_mslite(vision_model, inputs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    num_llm_tokens = _num_image_tokens(grid_thw[0], int(cfg.vision_config.spatial_merge_size))
    print(f"Image grid_thw={grid_thw[0].tolist()}, image tokens={num_llm_tokens}, "
          f"vision inference={elapsed_ms:.2f} ms")
    return _ImageContext(_build_image_token_block(num_llm_tokens), grid_thw, features, elapsed_ms)


def _encode_text(
    text_model: "mslite.Model",
    tokenizer,
    cfg,
    request: _EmbedRequest,
    seq_len_buckets: Optional[Tuple[int, ...]] = None,
) -> Tuple[np.ndarray, float]:
    """Embed one request and return the L2-normalized last-token embedding and its latency."""
    feed, attention_mask = _prepare_text_feed(cfg, tokenizer, text_model, request, seq_len_buckets)
    inputs = _build_mslite_inputs(text_model, feed, preferred_order=_TEXT_FEATURE_ORDER)
    start = time.perf_counter()
    hidden = _run_mslite(text_model, inputs)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    # One request is embedded per call, so the batch dimension is dropped here.
    return _normalize(_last_token_pool(hidden, attention_mask))[0], elapsed_ms


def _embed_requests(
    text_model: "mslite.Model",
    tokenizer,
    cfg,
    requests: List[_EmbedRequest],
    seq_len_buckets: Optional[Tuple[int, ...]] = None,
) -> Tuple[np.ndarray, float]:
    """Embed a list of requests and return the stacked embeddings and the total text latency."""
    embeddings = []
    total_ms = 0.0
    for request in requests:
        embedding, elapsed_ms = _encode_text(text_model, tokenizer, cfg, request, seq_len_buckets)
        embeddings.append(embedding)
        total_ms += elapsed_ms
    return np.stack(embeddings, axis=0), total_ms


def _print_similarity(embeddings: np.ndarray, texts: List[str], image_embedding: Optional[np.ndarray]) -> None:
    """Print the cosine similarity of the already normalized embeddings."""
    sim = embeddings @ embeddings.T
    print("\nText-to-text similarity (cosine):")
    for i in range(int(sim.shape[0])):
        for j in range(i + 1, int(sim.shape[0])):
            print(f"  text[{i}] vs text[{j}]: {float(sim[i, j]):.4f}")
    if image_embedding is None:
        return
    cross = embeddings @ image_embedding
    print("\nImage-to-text similarity (cosine):")
    for i, text in enumerate(texts):
        print(f"  image vs text[{i}] ({text[:40]!r}): {float(cross[i]):.4f}")


def _parse_args() -> argparse.Namespace:
    """Parse the inference CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-Embedding-2B MindSpore Lite inference (vision + text models)"
    )
    parser.add_argument("--vision-model", type=str, required=True,
                        help="Path to qwen3_vl_embedding_2b_vision(.mindir)")
    parser.add_argument("--text-model", type=str, required=True,
                        help="Path to qwen3_vl_embedding_2b_text_image(.mindir)")
    parser.add_argument("--processor", type=str, default="Qwen/Qwen3-VL-Embedding-2B",
                        help="Tokenizer/config/image processor path or HuggingFace id")
    parser.add_argument("--texts", type=str, nargs="+", default=["Hello world", "Hi there", "Good morning"],
                        help="Texts to embed; when --image is given every text is embedded together with that image")
    parser.add_argument("--image", type=str, default="https://hbr.org/resources/images/article_assets/2018/03/mar18_9_824179306.jpg",
                        help="Optional image path or URL; it is prepended to every text and is also embedded "
                             "on its own so that the image-text similarity can be printed")
    parser.add_argument("--instruction", type=str, default="Represent the user's input.",
                        help="Instruction placed in the system prompt")
    parser.add_argument("--image-size", type=int, default=1024,
                        help="Must match --vision-image-size used in the export script")
    parser.add_argument("--seq-len-buckets", type=_parse_seq_len_buckets, default=_DEFAULT_SEQ_LEN_BUCKETS,
                        help="Comma separated sequence-length buckets (动态分档) compiled into the text "
                             "model by `ge.dynamicDims`; prompts are left-padded up to the next bucket "
                             f"(default: {','.join(str(b) for b in _DEFAULT_SEQ_LEN_BUCKETS)})")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "ascend"],
                        help="Device used for inference")
    parser.add_argument("--device-id", type=int, default=0, help="Ascend device id (only for --device ascend)")
    parser.add_argument("--compute-similarity", action="store_true", help="Print the similarity matrices")
    return parser.parse_args()


def _check_runtime_dependencies() -> None:
    """Fail fast when mindspore_lite or transformers are not importable."""
    if mslite is None:
        print("Error: mindspore_lite not installed.")
        sys.exit(1)
    if AutoTokenizer is None or AutoConfig is None or AutoImageProcessor is None:
        print("Error: transformers not installed or incompatible.")
        sys.exit(1)


def _load_processor(processor_id: str):
    """Load the model config, the tokenizer and the slow image processor."""
    print(f"Loading processor from {processor_id}...")
    cfg = AutoConfig.from_pretrained(processor_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(processor_id, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(processor_id, trust_remote_code=True, use_fast=False)
    return cfg, tokenizer, image_processor


def main():
    """Load the vision/text MindIR models, embed the inputs and print the results."""
    args = _parse_args()
    _check_runtime_dependencies()
    cfg, tokenizer, image_processor = _load_processor(args.processor)

    context = _create_context(args.device, args.device_id)
    vision_model = _load_mslite_model(args.vision_model, context)
    _print_model_io(vision_model, "Vision model")
    text_model = _load_mslite_model(args.text_model, context)
    _print_model_io(text_model, "Text model")

    total_start = time.perf_counter()

    image_ctx = None
    if args.image:
        image_ctx = _run_vision(cfg, vision_model, image_processor, args.image, args.image_size)

    requests = [_EmbedRequest(args.instruction, text, image_ctx) for text in args.texts]
    embeddings, text_ms = _embed_requests(text_model, tokenizer, cfg, requests, args.seq_len_buckets)
    text_calls = len(requests)

    image_embedding = None
    if image_ctx is not None:
        image_request = _EmbedRequest(args.instruction, "", image_ctx)
        image_embedding, elapsed_ms = _encode_text(text_model, tokenizer, cfg, image_request, args.seq_len_buckets)
        text_ms += elapsed_ms
        text_calls += 1

    total_ms = (time.perf_counter() - total_start) * 1000.0
    print("=" * 60)
    print(f"embeddings shape: {tuple(embeddings.shape)}")
    if image_embedding is not None:
        print(f"image embedding shape: {tuple(image_embedding.shape)}")
    print("=" * 60)

    if args.compute_similarity:
        _print_similarity(embeddings, list(args.texts), image_embedding)

    print("\n--- Performance ---")
    if image_ctx is not None:
        print(f"  Vision inference:  {image_ctx.vision_ms:.2f} ms")
    # `text_ms` is the sum over every text-model call, so report the per-call average.
    print(f"  Text inference:    {text_ms / text_calls:.2f} ms (per call)")
    print(f"  Total:             {total_ms:.2f} ms")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
