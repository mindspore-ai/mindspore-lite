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
"""Graph-mode greedy decode for Qwen3.5-9B LiteBoost."""
from __future__ import annotations

# The graph path drives HuggingFace internals directly (StaticCache layer
# buffers, swapped forwards, per-layer overrides) and has to keep the upstream
# call signatures, so touching protected members is deliberate here.
# pylint: disable=protected-access,unused-argument

import logging
import time

import torch
import torch_npu  # noqa: F401
from transformers import StaticCache

logger = logging.getLogger(__name__)


def _make_static_cache(model, input_ids, max_cache_len):
    """Build a StaticCache pre-allocated to `max_cache_len`."""
    text_model = model.model if hasattr(model, "model") else model
    if hasattr(text_model, "language_model"):
        config = text_model.language_model.config
    else:
        config = text_model.config
    cache = StaticCache(config=config, max_batch_size=1, max_cache_len=max_cache_len)
    return cache


def prefill_warmup(model, input_ids, attention_mask, pixel_values=None,
                   image_grid_thw=None, mm_token_type_ids=None, device="npu:0"):
    """Run one throwaway prefill to absorb first-call lazy-init / aclnn compile."""
    extra = {}
    if pixel_values is not None:
        extra["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        extra["image_grid_thw"] = image_grid_thw
    if mm_token_type_ids is not None:
        extra["mm_token_type_ids"] = mm_token_type_ids
    with torch.inference_mode():
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **extra,
        )
    torch_npu.npu.synchronize()


def _get_inner_model(model):
    """Return the model that holds `rope_deltas` (Qwen3_5Model or Qwen3_5TextModel)."""
    inner = model.model if hasattr(model, "model") else model
    return inner


def _decode_position_ids(model, past_len, batch_size, device):
    """Build 3D mRoPE position_ids for a single decode step."""
    inner_model = _get_inner_model(model)
    rope_deltas = getattr(inner_model, "rope_deltas", None)
    if rope_deltas is None:
        rope_deltas = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    pos_2d = torch.arange(past_len, past_len + 1, dtype=torch.long, device=device)
    pos_2d = pos_2d.view(1, 1).expand(batch_size, 1)
    pos_2d = pos_2d + rope_deltas.to(device=device, dtype=torch.long)
    position_ids = pos_2d.unsqueeze(0).expand(3, batch_size, 1)
    return position_ids


def _snapshot_state(state):
    """Clone a linear-attention state container (per-state dict or tensor)."""
    if isinstance(state, dict):
        return {k: (v.clone() if torch.is_tensor(v) else None) for k, v in state.items()}
    return state.clone() if torch.is_tensor(state) else None


def _restore_state(state, saved):
    """Copy a snapshot back in place, keeping the graph-baked device addresses."""
    if saved is None:
        return
    if isinstance(state, dict):
        for k, v in saved.items():
            if v is None:
                continue
            if torch.is_tensor(state.get(k)):
                state[k].copy_(v)
            else:
                state[k] = v.clone()
    elif torch.is_tensor(state):
        state.copy_(saved)


def _save_cache_state(cache):
    """Snapshot per-layer cache state for a post-warmup rewind."""
    with torch.inference_mode():
        saved = ([], [])
        for lyr in cache.layers:
            cl = getattr(lyr, "cumulative_length", None)
            saved[0].append(cl.clone() if torch.is_tensor(cl) else None)
            # transformers 5.x stores the linear-attention states in per-state
            # dicts (`conv_states[0]`), so snapshot those too -- without them a
            # rewind leaves the GDN state as far advanced as the throwaway
            # warmup / capture steps.
            saved[1].append(
                (_snapshot_state(getattr(lyr, "conv_states", None)),
                 _snapshot_state(getattr(lyr, "recurrent_states", None))))
    return saved


def _rewind_cache_state(cache, saved):
    """Restore cache state saved by `_save_cache_state` (undo warmup effects)."""
    with torch.inference_mode():
        for lyr, cl in zip(cache.layers, saved[0]):
            if cl is not None:
                lyr.cumulative_length.copy_(cl)
        for lyr, (cs, rs) in zip(cache.layers, saved[1]):
            _restore_state(getattr(lyr, "conv_states", None), cs)
            _restore_state(getattr(lyr, "recurrent_states", None), rs)


def _reset_static_decode_mask(model, cache, valid_len, device):
    """Make the static decode mask cover exactly `valid_len` positions.

    Full-attention decode reads an additive `[batch, 1, max_cache_len]` mask that
    is created lazily and then updated in place, one position per step. Every
    forward that runs before the replay loop (prefill, decode warmup, graph
    capture) leaves extra rows unlocked, and the KV slots behind them hold
    whatever those throwaway steps wrote -- so the mask has to be rebuilt from
    scratch whenever the loop starts from a rewound state.
    """
    from lite_boost.model.qwen3_5_9b.model import _get_static_buffers

    text_model = model.model if hasattr(model, "model") else model
    inner = text_model.language_model if hasattr(text_model, "language_model") else text_model
    with torch.inference_mode():
        for i, layer in enumerate(inner.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            cache_layer = cache.layers[i] if hasattr(cache, "layers") else None
            max_cache_len = getattr(cache_layer, "max_cache_len", None)
            if max_cache_len is None:
                continue
            buf = _get_static_buffers(attn.layer_idx, 1, max_cache_len, device)
            buf["mask"].fill_(-1e4)
            if valid_len > 0:
                buf["mask"][:, :, :valid_len] = 0


def graph_generate(
    model,
    input_ids,
    attention_mask,
    pixel_values=None,
    image_grid_thw=None,
    mm_token_type_ids=None,
    max_new_tokens: int = 256,
    device: str = "npu:0",
    num_warmup: int = 3,
    num_priming: int = 1,
    capture_graph: bool = True,
    stream=None,
    timing: dict | None = None,
):
    """Greedy decode with optional `torch.npu.graph` capture."""
    def _sync():
        torch_npu.npu.synchronize()

    prompt_len = input_ids.shape[1]
    max_cache_len = prompt_len + max_new_tokens + 16
    cache = _make_static_cache(model, input_ids, max_cache_len)

    extra_prefill = {}
    if pixel_values is not None:
        extra_prefill["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        extra_prefill["image_grid_thw"] = image_grid_thw
    if mm_token_type_ids is not None:
        extra_prefill["mm_token_type_ids"] = mm_token_type_ids

    generated = []

    _sync()
    t_prefill_0 = time.time()
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            **extra_prefill,
        )
    _sync()
    t_prefill = time.time() - t_prefill_0
    if timing is not None:
        timing["prefill"] = t_prefill
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    generated.append(next_tok.item())
    # The prefill wrote `prompt_len` positions into the KV cache; the decode
    # mask is created lazily by the first decode step, so seed it with exactly
    # that window (and with the right length) here.
    _reset_static_decode_mask(model, cache, input_ids.shape[1], device)

    _sync()
    t_prime_0 = time.time()
    for _ in range(num_priming):
        past_len = int(cache.get_seq_length())
        position_ids = _decode_position_ids(model, past_len, 1, device)
        with torch.inference_mode():
            out = model(
                input_ids=next_tok,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_tok.item())
    _sync()
    t_prime = time.time() - t_prime_0
    if timing is not None:
        timing["prime"] = t_prime

    remaining = max_new_tokens - 1 - num_priming
    if remaining <= 0:
        return generated

    if not capture_graph:
        _sync()
        t_loop_0 = time.time()
        for _ in range(remaining):
            past_len = int(cache.get_seq_length())
            position_ids = _decode_position_ids(model, past_len, 1, device)
            with torch.inference_mode():
                out = model(
                    input_ids=next_tok,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            generated.append(next_tok.item())
        _sync()
        t_loop = time.time() - t_loop_0
        if timing is not None:
            timing["decode_loop"] = t_loop
            timing["decode_per_tok_ms"] = t_loop / remaining * 1000
        return generated

    static_input_ids = torch.zeros(1, 1, dtype=input_ids.dtype, device=device)
    static_input_ids.copy_(next_tok)

    static_position_ids = torch.zeros(3, 1, 1, dtype=torch.long, device=device)
    past_len = int(cache.get_seq_length())
    static_position_ids.copy_(_decode_position_ids(model, past_len, 1, device))

    capture_stream = stream or torch_npu.npu.Stream(device=device)

    saved_state = _save_cache_state(cache)

    _sync()
    t_warmup_0 = time.time()
    with torch_npu.npu.stream(capture_stream):
        for _ in range(num_warmup):
            with torch.inference_mode():
                _ = model(
                    input_ids=static_input_ids,
                    attention_mask=None,
                    position_ids=static_position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
    capture_stream.synchronize()
    _sync()
    t_warmup = time.time() - t_warmup_0
    if timing is not None:
        timing["warmup"] = t_warmup

    _rewind_cache_state(cache, saved_state)
    _sync()

    _sync()
    t_capture_0 = time.time()
    g = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(g, stream=capture_stream):
        with torch.inference_mode():
            static_out = model(
                input_ids=static_input_ids,
                attention_mask=None,
                position_ids=static_position_ids,
                past_key_values=cache,
                use_cache=True,
            )
    _sync()
    t_capture = time.time() - t_capture_0
    if timing is not None:
        timing["capture"] = t_capture

    # Capturing ran one real decode step on the cache; rewind it so the replay
    # loop feeds the same token at the same position, and rebuild the mask
    # because the warmup and capture steps unlocked rows past the sequence.
    _rewind_cache_state(cache, saved_state)
    replay_from = int(cache.get_seq_length())
    _reset_static_decode_mask(model, cache, replay_from, device)

    _sync()
    t_replay_0 = time.time()
    t_copy_total = 0.0
    t_replay_only = 0.0
    t_argmax_total = 0.0
    for _ in range(remaining):
        past_len = int(cache.get_seq_length())
        new_pos = _decode_position_ids(model, past_len, 1, device)
        t_c0 = time.time()
        static_input_ids.copy_(next_tok)
        static_position_ids.copy_(new_pos)
        t_copy_total += time.time() - t_c0
        t_r0 = time.time()
        g.replay()
        capture_stream.synchronize()
        t_replay_only += time.time() - t_r0
        t_a0 = time.time()
        next_tok = static_out.logits[:, -1:, :].argmax(dim=-1)
        t_argmax_total += time.time() - t_a0
        generated.append(next_tok.item())
    _sync()
    t_replay = time.time() - t_replay_0
    if timing is not None:
        timing["decode_replay"] = t_replay
        timing["decode_per_tok_ms"] = t_replay / remaining * 1000
        timing["decode_copy_ms"] = t_copy_total / remaining * 1000
        timing["decode_replay_only_ms"] = t_replay_only / remaining * 1000
        timing["decode_argmax_ms"] = t_argmax_total / remaining * 1000

    return generated


def _get_text_model_and_inner(model):
    """Return (text_model, inner_model) for Qwen3_5ForConditionalGeneration."""
    text_model = model.model if hasattr(model, "model") else model
    inner = text_model.language_model if hasattr(text_model, "language_model") else text_model
    return text_model, inner


def _prepare_prefill_inputs(
    model, input_ids, attention_mask, pixel_values, image_grid_thw,
    mm_token_type_ids, bucket, device, extra_token_embeds=None,
):
    """Eagerly run vision + 3D position_ids, then pad inputs_embeds to bucket."""
    import torch.nn.functional as F
    text_model = model.model if hasattr(model, "model") else model

    with torch.no_grad():
        inputs_embeds = text_model.get_input_embeddings()(input_ids)

        image_embeds = None
        if pixel_values is not None:
            image_outputs = text_model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True,
            )
            image_embeds_list = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds_list, dim=0).to(device)
        if extra_token_embeds is not None:
            if image_embeds is not None:
                raise ValueError(
                    "pass either pixel_values or extra_token_embeds, not both")
            image_embeds = extra_token_embeds.to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        if image_embeds is not None:
            image_mask, _ = text_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask, image_embeds.to(inputs_embeds.dtype),
            )

        position_ids = text_model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_type_ids,
        )
        if isinstance(position_ids, tuple):
            position_ids = position_ids[0]
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device,
            ).view(1, 1, -1).expand(3, 1, input_ids.shape[1]).contiguous()

        real_len = input_ids.shape[1]
        pad_len = bucket - real_len

        if pad_len > 0:
            inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, pad_len), value=0)
            attention_mask_padded = F.pad(attention_mask, (0, pad_len), value=0)
            position_ids_padded = F.pad(position_ids, (0, pad_len), value=0)
        else:
            attention_mask_padded = attention_mask
            position_ids_padded = position_ids

    return inputs_embeds, attention_mask_padded, position_ids_padded, real_len


def _build_causal_pad_mask(bucket, real_len, device):
    """Build [1, 1, bucket, bucket] int8 causal+pad mask for PFA."""
    causal = torch.triu(
        torch.full((bucket, bucket), -1e4, dtype=torch.float16, device=device),
        diagonal=1,
    )
    if real_len < bucket:
        pad_rows = torch.zeros(bucket, bucket, dtype=torch.float16, device=device)
        pad_rows[real_len:, :] = -1e4
        pad_rows[:, real_len:] = -1e4
        mask = torch.minimum(causal, pad_rows)
    else:
        mask = causal
    int8_mask = torch.where(
        mask == 0,
        torch.zeros((), dtype=torch.int8, device=device),
        torch.full((), -128, dtype=torch.int8, device=device),
    )
    return int8_mask.unsqueeze(0).unsqueeze(0)


def _reset_cache_for_prefill(model, cache, real_len, bucket, device):
    """Reset StaticCache + per-layer static buffers before prefill replay."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    text_model = model.model if hasattr(model, "model") else model
    inner = text_model.language_model if hasattr(text_model, "language_model") else text_model

    with torch.inference_mode():
        for layer_cache in getattr(cache, "layers", []):
            cl = getattr(layer_cache, "cumulative_length", None)
            if cl is not None and torch.is_tensor(cl):
                cl.fill_(0)
            elif hasattr(layer_cache, "cumulative_length"):
                layer_cache.cumulative_length = 0

    with torch.inference_mode():
        if not hasattr(model, "_prefill_mask_buffers"):
            model._prefill_mask_buffers = {}
        mask_key = (bucket, str(device))
        prefill_mask = model._prefill_mask_buffers.get(mask_key)
        fresh_mask = _build_causal_pad_mask(bucket, real_len, device)
        if prefill_mask is None:
            prefill_mask = fresh_mask
            model._prefill_mask_buffers[mask_key] = prefill_mask
        else:
            prefill_mask.copy_(fresh_mask)

        if not hasattr(model, "_lb_shared_reset_buffers"):
            model._lb_shared_reset_buffers = {}
        shared = model._lb_shared_reset_buffers.get(str(device))
        if shared is None:
            shared = {
                "seq_len": torch.zeros(8, dtype=torch.int32, device=device),
                "conv_idx": torch.zeros(4, dtype=torch.int64, device=device),
            }
            model._lb_shared_reset_buffers[str(device)] = shared
        shared["seq_len"].fill_(real_len)
        pad_key = (bucket, str(device))
        if not hasattr(model, "_lb_pad_row_buffers"):
            model._lb_pad_row_buffers = {}
        pad_bufs = model._lb_pad_row_buffers.get(pad_key)
        if pad_bufs is None:
            pad_bufs = {
                "pos": torch.arange(bucket, dtype=torch.int32, device=device),
                "pad_rows": torch.empty(bucket, dtype=torch.bool, device=device),
            }
            model._lb_pad_row_buffers[pad_key] = pad_bufs
        pad_bufs["pad_rows"].copy_(pad_bufs["pos"] >= shared["seq_len"][:1])
        first_la = None
        for la in inner.layers:
            if (isinstance(getattr(la, "linear_attn", None), Qwen3_5GatedDeltaNet)
                    and getattr(la.linear_attn, "conv_kernel_size", None) is not None):
                first_la = la
                break
        if first_la is not None:
            k_size = int(first_la.linear_attn.conv_kernel_size)
            conv_idx = torch.arange(
                max(real_len - k_size, 0), real_len,
                dtype=torch.int64, device=device,
            )
            cur = shared["conv_idx"]
            if cur.numel() != conv_idx.numel():
                shared["conv_idx"] = conv_idx
            else:
                cur.copy_(conv_idx)

        for _, layer in enumerate(inner.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None and getattr(attn, "_bucketed_prefill", False):
                attn._prefill_mask = prefill_mask
            la = getattr(layer, "linear_attn", None)
            if la is not None and isinstance(la, Qwen3_5GatedDeltaNet):
                la._static_actual_seq_len = shared["seq_len"][:1]
                la._static_seq_len8 = shared["seq_len"]
                la._static_prefill_conv_idx = shared["conv_idx"]
                la._static_pad_rows = pad_bufs["pad_rows"]
                la._lb_prefill_exact = real_len == bucket
                la._lb_bucketed_exec = True

    # The prefill writes the whole bucket into the KV cache, padding rows
    # included, so the decode mask must expose exactly the real prompt.
    _reset_static_decode_mask(model, cache, real_len, device)


def _build_multi_round_mask(new_bucket, past_len, new_real_len, max_cache_len, device):
    """Build [1, 1, new_bucket, max_cache_len] float16 mask for multi-round prefill."""
    mask = torch.zeros(new_bucket, max_cache_len, dtype=torch.float16, device=device)

    col_idx = torch.arange(max_cache_len, device=device)
    row_idx = torch.arange(new_bucket, device=device)
    causal_boundary = past_len + row_idx + 1
    future_new = (col_idx.unsqueeze(0) >= causal_boundary.unsqueeze(1)) & \
                 (col_idx.unsqueeze(0) < past_len + new_real_len)
    mask = torch.where(future_new, torch.full_like(mask, -1e4), mask)

    if past_len + new_real_len < max_cache_len:
        mask[:, past_len + new_real_len:] = -1e4

    if new_real_len < new_bucket:
        mask[new_real_len:, :] = -1e4

    return mask.unsqueeze(0).unsqueeze(0)


def _prepare_multi_round_prefill(
    model, input_ids, attention_mask, past_len, mm_token_type_ids,
    new_bucket, device,
):
    """Prepare inputs for multi-round prefill (no vision, just new text tokens)."""
    import torch.nn.functional as F
    text_model = model.model if hasattr(model, "model") else model

    inputs_embeds = text_model.get_input_embeddings()(input_ids)
    new_real_len = input_ids.shape[1]

    inner_model = _get_inner_model(model)
    rope_deltas = getattr(inner_model, "rope_deltas", None)
    if rope_deltas is None:
        rope_deltas_val = 0
    else:
        rope_deltas_val = int(rope_deltas.item())
    pos_text = torch.arange(
        past_len, past_len + new_real_len, dtype=torch.long, device=device,
    )
    pos_text = pos_text + rope_deltas_val
    position_ids = pos_text.view(1, 1, new_real_len).expand(3, 1, new_real_len).contiguous()

    pad_len = new_bucket - new_real_len
    if pad_len > 0:
        inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, pad_len), value=0)
        attention_mask_padded = F.pad(attention_mask, (0, pad_len), value=0)
        position_ids_padded = F.pad(position_ids, (0, pad_len), value=0)
    else:
        attention_mask_padded = attention_mask
        position_ids_padded = position_ids

    return inputs_embeds, attention_mask_padded, position_ids_padded, new_real_len


def _reset_cache_for_multi_round_prefill(
    model, cache, past_len, new_real_len, new_bucket, max_cache_len, device,
):
    """Prepare StaticCache + per-layer buffers for multi-round prefill replay."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    text_model = model.model if hasattr(model, "model") else model
    inner = text_model.language_model if hasattr(text_model, "language_model") else text_model

    with torch.inference_mode():
        for i, layer in enumerate(inner.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None and getattr(attn, "_bucketed_prefill", False):
                layer_cache = cache.layers[i] if hasattr(cache, "layers") else None
                if layer_cache is not None and hasattr(layer_cache, "cumulative_length"):
                    cl = layer_cache.cumulative_length
                    if torch.is_tensor(cl):
                        cl.fill_(past_len)
                    else:
                        layer_cache.cumulative_length = past_len

    mr_mask_key = ("mr_mask", new_bucket, max_cache_len, str(device))
    if not hasattr(model, "_mr_mask_buffers"):
        model._mr_mask_buffers = {}
    static_mr_mask = model._mr_mask_buffers.get(mr_mask_key)
    new_mask = _build_multi_round_mask(new_bucket, past_len, new_real_len, max_cache_len, device)
    if static_mr_mask is None:
        static_mr_mask = new_mask.clone()
        model._mr_mask_buffers[mr_mask_key] = static_mr_mask
    else:
        with torch.inference_mode():
            static_mr_mask.copy_(new_mask)

    with torch.inference_mode():
        for _, layer in enumerate(inner.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None and getattr(attn, "_bucketed_prefill", False):
                attn._prefill_mask = static_mr_mask
            la = getattr(layer, "linear_attn", None)
            if la is not None and isinstance(la, Qwen3_5GatedDeltaNet):
                if la._static_actual_seq_len is None or \
                   la._static_actual_seq_len.device.type != torch.device(device).type:
                    la._static_actual_seq_len = torch.tensor(
                        [new_real_len], dtype=torch.int32, device=device,
                    )
                else:
                    la._static_actual_seq_len.copy_(
                        torch.tensor([new_real_len], dtype=torch.int32, device=device)
                    )
                la._lb_prefill_exact = False
                la._multi_round_prefill = True
                la._static_pad_rows = None
                base = max(0, new_real_len - 4)
                idx = torch.tensor(
                    [base, base + 1, base + 2, base + 3],
                    dtype=torch.int64, device=device,
                )
                idx = idx.clamp(max=3 + new_bucket - 1)
                if not hasattr(la, "_static_conv_state_idx") or \
                   la._static_conv_state_idx is None or \
                   la._static_conv_state_idx.device.type != torch.device(device).type:
                    la._static_conv_state_idx = idx.clone()
                else:
                    la._static_conv_state_idx.copy_(idx)

    _reset_static_decode_mask(model, cache, past_len + new_real_len, device)


def _clear_multi_round_flags(model):
    """Clear `_multi_round_prefill` flag on linear-attn layers after round 2+."""
    text_model = model.model if hasattr(model, "model") else model
    inner = text_model.language_model if hasattr(text_model, "language_model") else text_model
    for layer in inner.layers:
        la = getattr(layer, "linear_attn", None)
        if la is not None:
            la._multi_round_prefill = False


def multi_round_graph_generate(
    model,
    input_ids,
    attention_mask,
    past_key_values,
    past_len,
    mm_token_type_ids=None,
    new_buckets=(128, 512, 1024, 1050, 2048),
    max_new_tokens: int = 256,
    device: str = "npu:0",
    num_warmup: int = 3,
    num_priming: int = 1,
    capture_graph: bool = True,
    stream=None,
    timing: dict | None = None,
    use_chunk: bool = True,
):
    """Multi-round decode for round 2+ (Opt 15: chunk kernel path)."""
    _ = input_ids.shape[1]

    if not use_chunk:
        return _multi_round_prefill_as_decode(
            model, input_ids, attention_mask, past_key_values, past_len,
            mm_token_type_ids, max_new_tokens, device,
            num_warmup, num_priming, capture_graph, stream, timing,
        )

    return _multi_round_chunk_generate(
        model, input_ids, attention_mask, past_key_values, past_len,
        mm_token_type_ids, new_buckets, max_new_tokens, device,
        num_warmup, num_priming, capture_graph, stream, timing,
    )


def _multi_round_chunk_generate(
    model, input_ids, attention_mask, past_key_values, past_len,
    mm_token_type_ids, new_buckets, max_new_tokens, device,
    num_warmup, num_priming, capture_graph, stream, timing,
):
    """Opt 15: multi-round chunk kernel prefill with conv context injection."""
    def _sync():
        torch_npu.npu.synchronize()

    new_real_len = input_ids.shape[1]

    if timing is not None:
        timing["new_real_len"] = new_real_len
        timing["past_len"] = past_len

    cache = past_key_values
    _, inner = _get_text_model_and_inner(model)

    inner_model = _get_inner_model(model)
    rope_deltas = getattr(inner_model, "rope_deltas", None)
    if rope_deltas is None:
        rope_deltas_val = 0
    else:
        rope_deltas_val = int(rope_deltas.item())

    eligible = [b for b in new_buckets if b >= new_real_len]
    if not eligible:
        if timing is not None:
            timing["new_bucket"] = -1
            timing["fallback"] = "prefill_as_decode"
        return _multi_round_prefill_as_decode(
            model, input_ids, attention_mask, past_key_values, past_len,
            mm_token_type_ids, max_new_tokens, device,
            num_warmup, num_priming, capture_graph, stream, timing,
        )
    new_bucket = min(eligible)
    if timing is not None:
        timing["new_bucket"] = new_bucket

    max_cache_len = 0
    for i, layer in enumerate(inner.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and getattr(attn, "_bucketed_prefill", False):
            layer_cache = cache.layers[i] if hasattr(cache, "layers") else None
            if layer_cache is not None:
                kc = getattr(layer_cache, "keys", None)
                if torch.is_tensor(kc) and kc.dim() >= 3:
                    max_cache_len = max(max_cache_len, kc.shape[-2])
                kc = getattr(layer_cache, "key_cache", None)
                if torch.is_tensor(kc) and kc.dim() >= 3:
                    max_cache_len = max(max_cache_len, kc.shape[-2])
            break
    if max_cache_len == 0:
        max_cache_len = past_len + new_bucket + max_new_tokens + 16

    _sync()
    t_prep_0 = time.time()
    inputs_embeds, attn_mask_padded, position_ids_padded, _ = \
        _prepare_multi_round_prefill(
            model, input_ids, attention_mask, past_len, mm_token_type_ids,
            new_bucket, device,
        )
    _sync()
    t_prep = time.time() - t_prep_0
    if timing is not None:
        timing["multi_round_prep"] = t_prep

    if not hasattr(model, "_mr_bucket_buffers"):
        model._mr_bucket_buffers = {}
    mr_buffers = model._mr_bucket_buffers.get(new_bucket)
    if mr_buffers is None:
        static_inputs_embeds = torch.zeros(
            1, new_bucket, inputs_embeds.shape[-1],
            dtype=inputs_embeds.dtype, device=device,
        )
        static_attn_mask = torch.zeros(1, new_bucket, dtype=attention_mask.dtype, device=device)
        static_position_ids = torch.zeros(3, 1, new_bucket, dtype=torch.long, device=device)
        mr_buffers = {
            "inputs_embeds": static_inputs_embeds,
            "attn_mask": static_attn_mask,
            "position_ids": static_position_ids,
        }
        model._mr_bucket_buffers[new_bucket] = mr_buffers
    static_inputs_embeds = mr_buffers["inputs_embeds"]
    static_attn_mask = mr_buffers["attn_mask"]
    static_position_ids = mr_buffers["position_ids"]

    if not hasattr(model, "_mr_prefill_graphs"):
        model._mr_prefill_graphs = {}
    g_prefill_mr = model._mr_prefill_graphs.get(new_bucket)
    capture_stream = stream or torch_npu.npu.Stream(device=device)

    if g_prefill_mr is None:
        _reset_cache_for_multi_round_prefill(
            model, cache, past_len, new_real_len, new_bucket, max_cache_len, device,
        )
        static_inputs_embeds.copy_(inputs_embeds)
        static_attn_mask.copy_(attn_mask_padded)
        static_position_ids.copy_(position_ids_padded)

        prev_internal_fmt = getattr(torch.npu.config, "allow_internal_format", True)
        try:
            torch.npu.config.allow_internal_format = False

            _sync()
            t_warmup_0 = time.time()
            with torch_npu.npu.stream(capture_stream):
                for _ in range(num_warmup):
                    _reset_cache_for_multi_round_prefill(
                        model, cache, past_len, new_real_len, new_bucket, max_cache_len, device,
                    )
                    with torch.inference_mode():
                        inner_out = inner(
                            inputs_embeds=static_inputs_embeds,
                            attention_mask=static_attn_mask,
                            position_ids=static_position_ids,
                            past_key_values=cache,
                            use_cache=True,
                        )
            capture_stream.synchronize()
            _sync()
            if timing is not None:
                timing["multi_round_warmup"] = time.time() - t_warmup_0

            _sync()
            t_capture_0 = time.time()
            g_prefill_mr = torch_npu.npu.NPUGraph()
            with torch_npu.npu.graph(g_prefill_mr, stream=capture_stream):
                with torch.inference_mode():
                    inner_out = inner(
                        inputs_embeds=static_inputs_embeds,
                        attention_mask=static_attn_mask,
                        position_ids=static_position_ids,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    static_mr_prefill_out = inner_out.last_hidden_state
            _sync()
            if timing is not None:
                timing["multi_round_capture"] = time.time() - t_capture_0
            model._mr_prefill_graphs[new_bucket] = g_prefill_mr
            model._mr_prefill_static_out = static_mr_prefill_out
        finally:
            torch.npu.config.allow_internal_format = prev_internal_fmt
    else:
        static_mr_prefill_out = model._mr_prefill_static_out

    _sync()
    t_prefill_0 = time.time()
    _reset_cache_for_multi_round_prefill(
        model, cache, past_len, new_real_len, new_bucket, max_cache_len, device,
    )
    static_inputs_embeds.copy_(inputs_embeds)
    static_attn_mask.copy_(attn_mask_padded)
    static_position_ids.copy_(position_ids_padded)
    g_prefill_mr.replay()
    capture_stream.synchronize()
    _sync()
    t_prefill = time.time() - t_prefill_0
    if timing is not None:
        timing["multi_round_prefill"] = t_prefill
        timing["multi_round_prefill_per_tok_ms"] = (
            t_prefill / max(1, new_real_len) * 1000
        )

    last_hidden = static_mr_prefill_out[:, new_real_len - 1:new_real_len, :]
    with torch.no_grad():
        logits = model.lm_head(last_hidden)
    next_tok = logits.argmax(dim=-1)
    generated = [next_tok.item()]

    with torch.inference_mode():
        for i, layer in enumerate(inner.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None and getattr(attn, "_bucketed_prefill", False):
                layer_cache = cache.layers[i] if hasattr(cache, "layers") else None
                if layer_cache is not None and hasattr(layer_cache, "cumulative_length"):
                    cl = layer_cache.cumulative_length
                    if torch.is_tensor(cl):
                        cl.fill_(past_len + new_real_len)
                    else:
                        layer_cache.cumulative_length = past_len + new_real_len

    _clear_multi_round_flags(model)

    _sync()
    t_prime_0 = time.time()
    for _ in range(num_priming):
        past_len_now = int(cache.get_seq_length())
        position_ids = _decode_position_ids(model, past_len_now, 1, device)
        with torch.inference_mode():
            out = model(
                input_ids=next_tok,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_tok.item())
    _sync()
    if timing is not None:
        timing["multi_round_prime"] = time.time() - t_prime_0

    remaining = max_new_tokens - 1 - num_priming
    if remaining <= 0:
        return generated[:max_new_tokens]

    if hasattr(model, "_decode_graph") and model._decode_graph is not None:
        g_decode = model._decode_graph
        static_input_ids = model._decode_static_input_ids
        static_decode_position_ids = model._decode_static_position_ids
        static_decode_out = model._decode_static_out
    else:
        static_input_ids = torch.zeros(1, 1, dtype=input_ids.dtype, device=device)
        static_input_ids.copy_(next_tok)
        static_decode_position_ids = torch.zeros(3, 1, 1, dtype=torch.long, device=device)
        past_len_now = int(cache.get_seq_length())
        static_decode_position_ids.copy_(
            _decode_position_ids(model, past_len_now, 1, device)
        )

        saved_state = _save_cache_state(cache)
        _sync()
        t_dwarmup_0 = time.time()
        with torch_npu.npu.stream(capture_stream):
            for _ in range(num_warmup):
                with torch.inference_mode():
                    _ = model(
                        input_ids=static_input_ids,
                        attention_mask=None,
                        position_ids=static_decode_position_ids,
                        past_key_values=cache,
                        use_cache=True,
                    )
        capture_stream.synchronize()
        _sync()
        if timing is not None:
            timing["multi_round_decode_warmup"] = time.time() - t_dwarmup_0
        _rewind_cache_state(cache, saved_state)
        _sync()

        _sync()
        t_dcapture_0 = time.time()
        g_decode = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(g_decode, stream=capture_stream):
            with torch.inference_mode():
                static_decode_out = model(
                    input_ids=static_input_ids,
                    attention_mask=None,
                    position_ids=static_decode_position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
        _sync()
        if timing is not None:
            timing["multi_round_decode_capture"] = time.time() - t_dcapture_0
        model._decode_graph = g_decode
        model._decode_static_input_ids = static_input_ids
        model._decode_static_position_ids = static_decode_position_ids
        model._decode_static_out = static_decode_out

    if not capture_graph:
        _sync()
        t_loop_0 = time.time()
        for _ in range(remaining):
            past_len_now = int(cache.get_seq_length())
            position_ids = _decode_position_ids(model, past_len_now, 1, device)
            with torch.inference_mode():
                out = model(
                    input_ids=next_tok,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            generated.append(next_tok.item())
        _sync()
        t_loop = time.time() - t_loop_0
        if timing is not None:
            timing["multi_round_decode_loop"] = t_loop
            timing["multi_round_decode_per_tok_ms"] = t_loop / remaining * 1000
        return generated

    _sync()
    t_replay_0 = time.time()
    past_len_host = int(cache.get_seq_length())
    static_pos_value = past_len_host + rope_deltas_val
    for _ in range(remaining):
        static_input_ids.copy_(next_tok)
        static_decode_position_ids.fill_(static_pos_value)
        g_decode.replay()
        capture_stream.synchronize()
        next_tok = static_decode_out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_tok.item())
        past_len_host += 1
        static_pos_value += 1
    _sync()
    t_replay = time.time() - t_replay_0
    if timing is not None:
        timing["multi_round_decode_replay"] = t_replay
        timing["multi_round_decode_per_tok_ms"] = t_replay / remaining * 1000

    return generated


def _multi_round_prefill_as_decode(
    model, input_ids, attention_mask, past_key_values, past_len,
    mm_token_type_ids, max_new_tokens, device,
    num_warmup, num_priming, capture_graph, stream, timing,
):
    """Opt 14 fallback: prefill-as-decode recurrent path (N ms/tok)."""
    def _sync():
        torch_npu.npu.synchronize()

    new_real_len = input_ids.shape[1]

    if timing is not None:
        timing["new_real_len"] = new_real_len
        timing["past_len"] = past_len

    cache = past_key_values

    inner_model = _get_inner_model(model)
    rope_deltas = getattr(inner_model, "rope_deltas", None)
    if rope_deltas is None:
        rope_deltas_val = 0
    else:
        rope_deltas_val = int(rope_deltas.item())

    _sync()
    t_prime_0 = time.time()
    next_tok = None
    prime_count = min(num_priming, new_real_len)
    for i in range(prime_count):
        cur_input = input_ids[:, i:i+1]
        past_len_now = past_len + i
        position_ids = _decode_position_ids(model, past_len_now, 1, device)
        with torch.inference_mode():
            out = model(
                input_ids=cur_input,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    _sync()
    t_prime = time.time() - t_prime_0
    if timing is not None:
        timing["multi_round_prime"] = t_prime
        timing["multi_round_prime_per_tok_ms"] = (
            t_prime / max(1, prime_count) * 1000
        )

    capture_stream = stream or torch_npu.npu.Stream(device=device)
    if hasattr(model, "_decode_graph") and model._decode_graph is not None:
        g_decode = model._decode_graph
        static_input_ids = model._decode_static_input_ids
        static_decode_position_ids = model._decode_static_position_ids
        static_decode_out = model._decode_static_out
    else:
        if prime_count < new_real_len:
            capture_tok = input_ids[:, prime_count:prime_count+1]
            capture_past_len = past_len + prime_count
        else:
            capture_tok = next_tok
            capture_past_len = past_len + prime_count
        static_input_ids = torch.zeros(1, 1, dtype=input_ids.dtype, device=device)
        static_input_ids.copy_(capture_tok)
        static_decode_position_ids = torch.zeros(3, 1, 1, dtype=torch.long, device=device)
        static_decode_position_ids.copy_(
            _decode_position_ids(model, capture_past_len, 1, device)
        )

        saved_state = _save_cache_state(cache)
        _sync()
        t_dwarmup_0 = time.time()
        with torch_npu.npu.stream(capture_stream):
            for _ in range(num_warmup):
                with torch.inference_mode():
                    _ = model(
                        input_ids=static_input_ids,
                        attention_mask=None,
                        position_ids=static_decode_position_ids,
                        past_key_values=cache,
                        use_cache=True,
                    )
        capture_stream.synchronize()
        _sync()
        if timing is not None:
            timing["multi_round_decode_warmup"] = time.time() - t_dwarmup_0
        _rewind_cache_state(cache, saved_state)
        _sync()

        _sync()
        t_dcapture_0 = time.time()
        g_decode = torch_npu.npu.NPUGraph()
        with torch_npu.npu.graph(g_decode, stream=capture_stream):
            with torch.inference_mode():
                static_decode_out = model(
                    input_ids=static_input_ids,
                    attention_mask=None,
                    position_ids=static_decode_position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
        _sync()
        if timing is not None:
            timing["multi_round_decode_capture"] = time.time() - t_dcapture_0
        model._decode_graph = g_decode
        model._decode_static_input_ids = static_input_ids
        model._decode_static_position_ids = static_decode_position_ids
        model._decode_static_out = static_decode_out

    remaining_new = new_real_len - prime_count
    generated = []

    if remaining_new > 0:
        if not capture_graph:
            _sync()
            t_loop_0 = time.time()
            for i in range(remaining_new):
                idx = prime_count + i
                cur_input = input_ids[:, idx:idx+1]
                past_len_now = past_len + idx
                position_ids = _decode_position_ids(model, past_len_now, 1, device)
                with torch.inference_mode():
                    out = model(
                        input_ids=cur_input,
                        attention_mask=None,
                        position_ids=position_ids,
                        past_key_values=cache,
                        use_cache=True,
                    )
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            _sync()
            t_loop = time.time() - t_loop_0
            if timing is not None:
                timing["multi_round_prefill_loop"] = t_loop
                timing["multi_round_prefill_per_tok_ms"] = (
                    t_loop / remaining_new * 1000
                )
        else:
            _sync()
            t_prefill_0 = time.time()
            past_len_host = past_len + prime_count
            static_pos_value = past_len_host + rope_deltas_val
            for i in range(remaining_new):
                idx = prime_count + i
                cur_input = input_ids[:, idx:idx+1]
                static_input_ids.copy_(cur_input)
                static_decode_position_ids.fill_(static_pos_value)
                g_decode.replay()
                capture_stream.synchronize()
                next_tok = static_decode_out.logits[:, -1:, :].argmax(dim=-1)
                past_len_host += 1
                static_pos_value += 1
            _sync()
            t_prefill = time.time() - t_prefill_0
            if timing is not None:
                timing["multi_round_prefill_loop"] = t_prefill
                timing["multi_round_prefill_per_tok_ms"] = (
                    t_prefill / remaining_new * 1000
                )

    generated.append(next_tok.item())

    remaining = max_new_tokens - 1
    if remaining <= 0:
        return generated[:max_new_tokens]

    if not capture_graph:
        _sync()
        t_loop_0 = time.time()
        for _ in range(remaining):
            past_len_now = int(cache.get_seq_length())
            position_ids = _decode_position_ids(model, past_len_now, 1, device)
            with torch.inference_mode():
                out = model(
                    input_ids=next_tok,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            generated.append(next_tok.item())
        _sync()
        t_loop = time.time() - t_loop_0
        if timing is not None:
            timing["multi_round_decode_loop"] = t_loop
            timing["multi_round_decode_per_tok_ms"] = t_loop / remaining * 1000
        return generated

    _sync()
    t_replay_0 = time.time()
    past_len_host = int(cache.get_seq_length())
    static_pos_value = past_len_host + rope_deltas_val
    for _ in range(remaining):
        static_input_ids.copy_(next_tok)
        static_decode_position_ids.fill_(static_pos_value)
        g_decode.replay()
        capture_stream.synchronize()
        next_tok = static_decode_out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_tok.item())
        past_len_host += 1
        static_pos_value += 1
    _sync()
    t_replay = time.time() - t_replay_0
    if timing is not None:
        timing["multi_round_decode_replay"] = t_replay
        timing["multi_round_decode_per_tok_ms"] = t_replay / remaining * 1000

    return generated


def init_bucket_pipelines(
    model,
    buckets=(128, 256, 512, 1024, 2048, 4096),
    device="npu:0",
    num_warmup: int = 3,
    stream=None,
    timing: dict | None = None,
    max_new_tokens: int = 0,
):
    """Init-time pooled prefill capture for continuous serving (prefill only).

    ``max_new_tokens`` only sizes the pooled cache, so the same cache can hold
    the decode steps of a request.  It does not change prefill computation or
    capture order; the default 0 keeps the original prefill-only capacity.
    """
    def _sync():
        torch_npu.npu.synchronize()

    buckets = tuple(sorted(int(b) for b in buckets))
    if not buckets:
        raise ValueError("buckets must be a non-empty sequence")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    _, inner = _get_text_model_and_inner(model)
    if not getattr(inner, "_bucketed_prefill", False):
        raise RuntimeError(
            "init_bucket_pipelines requires boost_qwen3_5_9b(..., "
            "graph_mode=True, bucketed_prefill=True)")

    max_cache_len = max(buckets) + max_new_tokens + 16
    cache = _make_static_cache(model, None, max_cache_len)
    model._pool_cache = cache
    model._pool_graphs = {}
    model._pool_outs = {}
    # A new cache/mask invalidates any decode graph captured against the old one.
    model._pool_decode_graph = None
    model._pool_decode_static_input_ids = None
    model._pool_decode_static_position_ids = None
    model._pool_decode_static_out = None
    capture_stream = stream or torch_npu.npu.Stream(device=device)
    model._pool_stream = capture_stream

    def _force_tail_fill():
        for layer in inner.layers:
            la = getattr(layer, "linear_attn", None)
            if la is not None:
                la._lb_prefill_exact = False

    _sync()
    t_init_0 = time.time()
    for bucket in buckets:
        dummy_ids = torch.zeros(1, bucket, dtype=torch.long, device=device)
        dummy_mask = torch.ones(1, bucket, dtype=torch.long, device=device)
        inputs_embeds, attn_mask_padded, position_ids_padded, _ = \
            _prepare_prefill_inputs(
                model, dummy_ids, dummy_mask, None, None, None, bucket, device)

        if not hasattr(model, "_bucket_buffers"):
            model._bucket_buffers = {}
        buffers = model._bucket_buffers.get(bucket)
        if buffers is None:
            buffers = {
                "inputs_embeds": torch.zeros(
                    1, bucket, inputs_embeds.shape[-1],
                    dtype=inputs_embeds.dtype, device=device),
                "attn_mask": torch.zeros(
                    1, bucket, dtype=attn_mask_padded.dtype, device=device),
                "position_ids": torch.zeros(
                    3, 1, bucket, dtype=torch.long, device=device),
            }
            model._bucket_buffers[bucket] = buffers
        static_inputs_embeds = buffers["inputs_embeds"]
        static_attn_mask = buffers["attn_mask"]
        static_position_ids = buffers["position_ids"]

        _reset_cache_for_prefill(model, cache, bucket, bucket, device)
        _force_tail_fill()
        static_inputs_embeds.copy_(inputs_embeds)
        static_attn_mask.copy_(attn_mask_padded)
        static_position_ids.copy_(position_ids_padded)

        prev_internal_fmt = getattr(
            torch.npu.config, "allow_internal_format", True)
        try:
            torch.npu.config.allow_internal_format = False

            _sync()
            t_warmup_0 = time.time()
            with torch_npu.npu.stream(capture_stream):
                for _ in range(num_warmup):
                    _reset_cache_for_prefill(
                        model, cache, bucket, bucket, device)
                    _force_tail_fill()
                    with torch.inference_mode():
                        inner(
                            inputs_embeds=static_inputs_embeds,
                            attention_mask=static_attn_mask,
                            position_ids=static_position_ids,
                            past_key_values=cache,
                            use_cache=True,
                        )
            capture_stream.synchronize()
            _sync()
            t_warmup = time.time() - t_warmup_0

            with torch_npu.npu.stream(capture_stream):
                _reset_cache_for_prefill(
                    model, cache, bucket, bucket, device)
                _force_tail_fill()
            capture_stream.synchronize()
            _sync()

            _sync()
            t_capture_0 = time.time()
            g_prefill = torch_npu.npu.NPUGraph()
            with torch_npu.npu.graph(g_prefill, stream=capture_stream):
                with torch.inference_mode():
                    inner_out = inner(
                        inputs_embeds=static_inputs_embeds,
                        attention_mask=static_attn_mask,
                        position_ids=static_position_ids,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    static_prefill_out = inner_out.last_hidden_state
            _sync()
            t_capture = time.time() - t_capture_0
        finally:
            torch.npu.config.allow_internal_format = prev_internal_fmt

        model._pool_graphs[bucket] = g_prefill
        model._pool_outs[bucket] = static_prefill_out
        if timing is not None:
            timing[f"init_warmup_{bucket}"] = t_warmup
            timing[f"init_capture_{bucket}"] = t_capture
        logger.info("%s", f"[pool-init] bucket={bucket} "
              f"warmup={t_warmup * 1000:.0f}ms "
              f"capture={t_capture * 1000:.0f}ms")

    _sync()
    if timing is not None:
        timing["init_total"] = time.time() - t_init_0
    return cache


def pooled_prefill(
    model,
    input_ids,
    attention_mask=None,
    pixel_values=None,
    image_grid_thw=None,
    mm_token_type_ids=None,
    extra_token_embeds=None,
    device="npu:0",
    timing: dict | None = None,
):
    """One prefill request on the pooled pipelines (prefill only, no decode)."""
    graphs = getattr(model, "_pool_graphs", None)
    if not graphs:
        raise RuntimeError(
            "pooled_prefill: init_bucket_pipelines(model, buckets, device) "
            "must run first (no pooled bucket graphs on this model)")

    real_len = input_ids.shape[1]
    eligible = [b for b in graphs if b >= real_len]
    if not eligible:
        raise ValueError(
            f"pooled_prefill: prompt length {real_len} exceeds the largest "
            f"pooled bucket {max(graphs)} — route to an overflow path "
            f"(e.g. graph_generate) instead")
    bucket = min(eligible)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    torch_npu.npu.synchronize()
    t_prep_0 = time.time()
    inputs_embeds, attn_mask_padded, position_ids_padded, real_len = \
        _prepare_prefill_inputs(
            model, input_ids, attention_mask, pixel_values, image_grid_thw,
            mm_token_type_ids, bucket, device,
            extra_token_embeds=extra_token_embeds)
    torch_npu.npu.synchronize()
    t_prep = time.time() - t_prep_0

    t_reset_0 = time.time()
    _reset_cache_for_prefill(model, model._pool_cache, real_len, bucket, device)
    _, inner = _get_text_model_and_inner(model)
    for layer in inner.layers:
        la = getattr(layer, "linear_attn", None)
        if la is not None:
            la._multi_round_prefill = False
    torch_npu.npu.synchronize()
    t_reset = time.time() - t_reset_0

    buffers = model._bucket_buffers[bucket]
    t_copy_0 = time.time()
    buffers["inputs_embeds"].copy_(inputs_embeds)
    buffers["attn_mask"].copy_(attn_mask_padded)
    buffers["position_ids"].copy_(position_ids_padded)
    t_copy = time.time() - t_copy_0

    t_replay_0 = time.time()
    graphs[bucket].replay()
    model._pool_stream.synchronize()
    torch_npu.npu.synchronize()
    t_replay = time.time() - t_replay_0

    for layer in inner.layers:
        la = getattr(layer, "linear_attn", None)
        if la is not None:
            la._lb_bucketed_exec = False

    t_lm_0 = time.time()
    prefill_out = model._pool_outs[bucket]
    last_hidden = prefill_out[:, real_len - 1:real_len, :]
    with torch.no_grad():
        logits = model.lm_head(last_hidden)
    next_tok = int(logits.argmax(dim=-1).item())
    t_lm = time.time() - t_lm_0

    if timing is not None:
        timing.update({
            "bucket": bucket, "real_len": real_len,
            "prep": t_prep, "reset": t_reset, "copy": t_copy,
            "replay": t_replay, "lm_head": t_lm,
            "prefill_total": (t_prep + t_reset + t_copy + t_replay + t_lm),
        })
    return next_tok, logits


def _set_cache_seq_length(cache, length):
    """Set the logical sequence length of every cache layer in place."""
    with torch.inference_mode():
        for layer_cache in getattr(cache, "layers", []):
            cl = getattr(layer_cache, "cumulative_length", None)
            if cl is not None and torch.is_tensor(cl):
                cl.fill_(length)
            elif hasattr(layer_cache, "cumulative_length"):
                layer_cache.cumulative_length = length


def init_decode_pipeline(model, device="npu:0", num_warmup: int = 3,
                         stream=None, timing: dict | None = None):
    """Capture the decode NPU graph once, on the post-prefill pooled cache.

    ``init_bucket_pipelines`` must run first: each bucket capture executes a
    real prefill, which allocates the KV / GDN state buffers the decode graph
    bakes in and leaves ``has_previous_state`` true so the trace takes the
    incremental linear-attention branch.  The capture executes one real decode
    step, so its cache mutation is rewound afterwards.
    """
    def _sync():
        torch_npu.npu.synchronize()

    cache = getattr(model, "_pool_cache", None)
    if cache is None or not getattr(model, "_pool_graphs", None):
        raise RuntimeError(
            "init_decode_pipeline: init_bucket_pipelines(model, buckets, device) "
            "must run first")
    if getattr(model, "_pool_decode_graph", None) is not None:
        return model._pool_decode_graph

    capture_stream = stream or getattr(model, "_pool_stream", None)
    if capture_stream is None:
        capture_stream = torch_npu.npu.Stream(device=device)

    _, inner = _get_text_model_and_inner(model)
    for layer_idx, layer in enumerate(inner.layers):
        if getattr(layer, "linear_attn", None) is None:
            continue
        state_flags = getattr(cache.layers[layer_idx], "has_previous_state", None)
        if not isinstance(state_flags, dict) or not all(state_flags.values()):
            raise RuntimeError(
                "init_decode_pipeline: linear-attention state is not initialized; "
                "run init_bucket_pipelines (its captures execute a real prefill) "
                "before capturing decode")
        # Prefill-only trace flags must not leak into the decode graph.
        layer.linear_attn._lb_bucketed_exec = False
        layer.linear_attn._multi_round_prefill = False

    _sync()
    t_init_0 = time.time()
    saved_state = _save_cache_state(cache)
    past_len = int(cache.get_seq_length())
    static_input_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
    static_decode_position_ids = torch.zeros(3, 1, 1, dtype=torch.long, device=device)
    static_decode_position_ids.copy_(_decode_position_ids(model, past_len, 1, device))

    t_warmup_0 = time.time()
    with torch_npu.npu.stream(capture_stream):
        for _ in range(num_warmup):
            with torch.inference_mode():
                _ = model(
                    input_ids=static_input_ids,
                    attention_mask=None,
                    position_ids=static_decode_position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
    capture_stream.synchronize()
    _sync()
    t_warmup = time.time() - t_warmup_0

    _rewind_cache_state(cache, saved_state)
    _sync()

    t_capture_0 = time.time()
    g_decode = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(g_decode, stream=capture_stream):
        with torch.inference_mode():
            static_decode_out = model(
                input_ids=static_input_ids,
                attention_mask=None,
                position_ids=static_decode_position_ids,
                past_key_values=cache,
                use_cache=True,
            )
    _sync()
    t_capture = time.time() - t_capture_0

    # Warmup and capture advanced the cache and unlocked extra mask rows; undo
    # both so the first request replays from its own post-prefill state.
    _rewind_cache_state(cache, saved_state)
    _reset_static_decode_mask(model, cache, int(cache.get_seq_length()), device)

    model._pool_decode_graph = g_decode
    model._pool_decode_static_input_ids = static_input_ids
    model._pool_decode_static_position_ids = static_decode_position_ids
    model._pool_decode_static_out = static_decode_out
    model._pool_decode_stream = capture_stream
    model._pool_decode_max_cache_len = cache.get_max_length()
    model._pool_decode_device = str(static_input_ids.device)

    if timing is not None:
        timing["decode_warmup"] = t_warmup
        timing["decode_capture"] = t_capture
        timing["decode_init_total"] = time.time() - t_init_0
    logger.info("%s", f"[decode-init] warmup={t_warmup * 1000:.0f}ms "
          f"capture={t_capture * 1000:.0f}ms")
    return g_decode


# Stop tokens are only checked once per block in the non-synchronizing loop;
# a block is at most this many steps, so a hit wastes at most the rest of it.
_STOP_CHECK_BLOCK = 8


def pooled_decode(model, first_token, real_len: int, max_new_tokens: int,
                  device="npu:0", timing: dict | None = None,
                  opt_no_decode_sync: bool = False, stop_token_ids=None):
    """Replay the startup-captured decode graph for one request.

    ``first_token`` comes from ``pooled_prefill``, so ``max_new_tokens`` counts
    it and the function runs up to ``max_new_tokens - 1`` graph steps.  When
    ``stop_token_ids`` is given (an int or an iterable of ints), generation
    ends as soon as one of those ids is produced and ``timing`` reports
    ``finish_reason="stop"`` (plus ``stop_token``); otherwise it runs the full
    budget and reports ``"length"``.  Per-token latencies average over the
    steps actually run, and ``timing["decode_steps"]`` counts them.  The
    pooled cache is left advanced; the next ``pooled_prefill`` resets it.
    """
    def _sync():
        torch_npu.npu.synchronize()

    g_decode = getattr(model, "_pool_decode_graph", None)
    if g_decode is None:
        raise RuntimeError(
            "pooled_decode: init_bucket_pipelines + init_decode_pipeline "
            "must run first")
    cache = model._pool_cache
    static_input_ids = model._pool_decode_static_input_ids
    static_decode_position_ids = model._pool_decode_static_position_ids
    static_decode_out = model._pool_decode_static_out
    capture_stream = getattr(model, "_pool_decode_stream", None)
    max_cache_len = getattr(model, "_pool_decode_max_cache_len", None)
    if max_cache_len is None or cache.get_max_length() != max_cache_len:
        raise RuntimeError(
            "pooled_decode: the pooled cache no longer matches the captured "
            "decode graph; do not mix pooled paths with graph_generate / "
            "multi_round in the same process")

    first_token = int(first_token)
    if stop_token_ids is None:
        stop_ids = frozenset()
    elif isinstance(stop_token_ids, int):
        stop_ids = frozenset((stop_token_ids,))
    else:
        stop_ids = frozenset(int(tok) for tok in stop_token_ids)
    remaining = int(max_new_tokens) - 1
    finish_reason = "length"
    stop_token = None
    if first_token in stop_ids:
        finish_reason = "stop"
        stop_token = first_token
        remaining = 0
    if remaining <= 0:
        if timing is not None:
            timing["decode_replay"] = 0.0
            timing["decode_per_tok_ms"] = 0.0
            timing["decode_copy_ms"] = 0.0
            timing["decode_replay_only_ms"] = 0.0
            timing["decode_argmax_ms"] = 0.0
            timing["decode_steps"] = 0
            timing["finish_reason"] = finish_reason
            timing["stop_token"] = stop_token
        return [first_token]

    if real_len + remaining > max_cache_len:
        raise ValueError(
            f"pooled_decode: real_len={real_len} + {remaining} decode steps "
            f"exceeds pooled cache capacity {max_cache_len}; call "
            f"init_bucket_pipelines(..., max_new_tokens={max_new_tokens})")

    # pooled_prefill leaves cumulative_length at the padded bucket; the decode
    # graph writes at the logical end of the real prompt.
    _set_cache_seq_length(cache, real_len)
    inner_model = _get_inner_model(model)
    rope_deltas = getattr(inner_model, "rope_deltas", None)
    rope_deltas_val = 0 if rope_deltas is None else int(rope_deltas.item())
    static_pos_value = real_len + rope_deltas_val

    next_tok = torch.tensor(
        [[first_token]], dtype=static_input_ids.dtype, device=static_input_ids.device)
    static_generated_npu = torch.zeros(
        remaining, dtype=torch.long, device=static_input_ids.device)
    executed = remaining
    t_replay_0 = time.time()
    t_copy_total = 0.0
    t_replay_only = 0.0
    t_argmax_total = 0.0
    if opt_no_decode_sync:
        with torch_npu.npu.stream(capture_stream):
            if stop_ids:
                # Without per-step synchronization the stop token is only
                # visible once a block completes, so check every
                # _STOP_CHECK_BLOCK steps.
                for block_0 in range(0, remaining, _STOP_CHECK_BLOCK):
                    block_len = min(_STOP_CHECK_BLOCK, remaining - block_0)
                    for step in range(block_0, block_0 + block_len):
                        static_input_ids.copy_(next_tok)
                        static_decode_position_ids.fill_(static_pos_value)
                        g_decode.replay()
                        next_tok = static_decode_out.logits[:, -1:, :].argmax(dim=-1)
                        static_generated_npu[step] = next_tok.view(-1)
                        static_pos_value += 1
                    capture_stream.synchronize()
                    block_tokens = static_generated_npu[
                        block_0:block_0 + block_len].cpu().tolist()
                    hit = next(
                        (idx for idx, tok in enumerate(block_tokens)
                         if tok in stop_ids), None)
                    if hit is not None:
                        executed = block_0 + hit + 1
                        stop_token = int(block_tokens[hit])
                        finish_reason = "stop"
                        break
                    executed = block_0 + block_len
            else:
                for step in range(remaining):
                    static_input_ids.copy_(next_tok)
                    static_decode_position_ids.fill_(static_pos_value)
                    g_decode.replay()
                    next_tok = static_decode_out.logits[:, -1:, :].argmax(dim=-1)
                    static_generated_npu[step] = next_tok.view(-1)
                    static_pos_value += 1
        _sync()
    else:
        for step in range(remaining):
            t_copy_0 = time.time()
            static_input_ids.copy_(next_tok)
            static_decode_position_ids.fill_(static_pos_value)
            t_copy_total += time.time() - t_copy_0
            t_replay_only_0 = time.time()
            g_decode.replay()
            if capture_stream is not None:
                capture_stream.synchronize()
            t_replay_only += time.time() - t_replay_only_0
            t_argmax_0 = time.time()
            next_tok = static_decode_out.logits[:, -1:, :].argmax(dim=-1)
            static_generated_npu[step] = next_tok.view(-1)
            t_argmax_total += time.time() - t_argmax_0
            static_pos_value += 1
            executed = step + 1
            if stop_ids:
                token = int(next_tok.item())
                if token in stop_ids:
                    stop_token = token
                    finish_reason = "stop"
                    break
        _sync()
    t_replay = time.time() - t_replay_0

    generated = [first_token]
    generated.extend(static_generated_npu[:executed].cpu().tolist())
    if timing is not None:
        timing["decode_replay"] = t_replay
        timing["decode_per_tok_ms"] = t_replay / executed * 1000
        timing["decode_copy_ms"] = t_copy_total / executed * 1000
        timing["decode_replay_only_ms"] = t_replay_only / executed * 1000
        timing["decode_argmax_ms"] = t_argmax_total / executed * 1000
        timing["decode_steps"] = executed
        timing["finish_reason"] = finish_reason
        timing["stop_token"] = stop_token
    return generated
