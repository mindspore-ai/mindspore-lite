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
Tensor-parallel (TP) forward replacements for BooguImage layers.

Each function here is a drop-in replacement for the original ``__call__`` /
``forward`` / ``decode`` method, adding all-reduce and GQA-overlap logic
when tensor parallelism is active, and falling back to the original method
otherwise. Weight sharding lives in ``shard_boogu_transformer`` (also here);
the SP counterpart is ``usp.py``.
"""

import math
import os
import types

import torch
import torch.nn.functional as F

from lite_boost.parallel import (
    is_distributed_active,
    get_rank as _get_tp_rank,
    get_world_size as _get_tp_ws,
    broadcast as _lb_broadcast,
    mm_all_reduce as _lb_mm_all_reduce,
    concat_all_reduce_split as _lb_concat_all_reduce_split,
    gqa_expand_kv_overlap as _lb_gqa_expand_kv_overlap,
    shard_colwise as _shard_colwise,
    shard_rowwise as _shard_rowwise,
    shard_kv_overlap as _shard_kv_overlap,
    get_overlap_kv_heads_per_rank as _get_overlap_kv_heads_per_rank,
)

from diffusers.utils.torch_utils import randn_tensor as _lb_randn_tensor


# ---------------------------------------------------------------------------
# Weight sharding
# ---------------------------------------------------------------------------

def _shard_colwise_rowwise(linear, rank, world_size):
    """Shard a Linear along both output and input dimensions in place for the given TP ``rank``."""
    if linear is None:
        return
    in_chunk = linear.in_features // world_size
    out_chunk = linear.out_features // world_size
    in_start = rank * in_chunk
    out_start = rank * out_chunk

    linear.weight.data = linear.weight.data[out_start:out_start + out_chunk, in_start:in_start + in_chunk]
    linear.in_features = in_chunk
    linear.out_features = out_chunk
    if linear.bias is not None:
        linear.bias.data = linear.bias.data[out_start:out_start + out_chunk]


def _shard_feed_forward(ffn, rank, world_size):
    """Shard a feed-forward block's three linears across TP ranks."""
    _shard_colwise(ffn.linear_1, rank, world_size)
    _shard_colwise(ffn.linear_3, rank, world_size)
    _shard_rowwise(ffn.linear_2, rank, world_size)


def _shard_attention_standard(attn, rank, world_size, num_attention_heads, num_kv_heads, head_dim):
    """Shard a standard attention module (Q col-wise, KV overlap, output row-wise)."""
    _shard_colwise(attn.to_q, rank, world_size)
    _shard_kv_overlap(attn.to_k, rank, world_size, num_kv_heads, head_dim)
    _shard_kv_overlap(attn.to_v, rank, world_size, num_kv_heads, head_dim)
    _shard_rowwise(attn.to_out[0], rank, world_size)

    heads_per_rank = num_attention_heads // world_size
    attn.heads = heads_per_rank
    attn.inner_dim = attn.to_q.out_features
    attn.inner_kv_dim = attn.to_k.out_features

    start_head, kv_heads_on_rank = _get_overlap_kv_heads_per_rank(rank, world_size, num_kv_heads)
    attn._tp_kv_heads = kv_heads_on_rank
    attn._tp_kv_start_head = start_head
    attn._tp_full_kv_heads = num_kv_heads


def _shard_double_stream_processor(processor, rank, world_size, num_attention_heads, num_kv_heads, head_dim):
    """Shard a double-stream attention processor (image + instruction branches)."""
    heads_per_rank = num_attention_heads // world_size

    _shard_colwise(processor.img_to_q, rank, world_size)
    _shard_colwise(processor.instruct_to_q, rank, world_size)

    _shard_kv_overlap(processor.img_to_k, rank, world_size, num_kv_heads, head_dim)
    _shard_kv_overlap(processor.img_to_v, rank, world_size, num_kv_heads, head_dim)

    _shard_kv_overlap(processor.instruct_to_k, rank, world_size, num_kv_heads, head_dim)
    _shard_kv_overlap(processor.instruct_to_v, rank, world_size, num_kv_heads, head_dim)

    _shard_rowwise(processor.instruct_out, rank, world_size)
    _shard_rowwise(processor.img_out, rank, world_size)

    processor.num_attention_heads = heads_per_rank

    start_head, kv_heads_on_rank = _get_overlap_kv_heads_per_rank(rank, world_size, num_kv_heads)
    processor._tp_kv_heads = kv_heads_on_rank
    processor._tp_kv_start_head = start_head
    processor._tp_full_kv_heads = num_kv_heads


def _shard_double_stream_block(block, rank, world_size, num_kv_heads, head_dim):
    """Shard a full double-stream transformer block (attention + dual FFN)."""
    num_attention_heads = block.num_attention_heads

    _shard_double_stream_processor(
        block.img_instruct_attn.processor, rank, world_size,
        num_attention_heads, num_kv_heads, head_dim,
    )

    _shard_rowwise(block.img_instruct_attn.to_out[0], rank, world_size)

    heads_per_rank = num_attention_heads // world_size
    block.img_instruct_attn.heads = heads_per_rank
    block.img_instruct_attn.inner_dim = heads_per_rank * head_dim
    _, kv_heads_on_rank = _get_overlap_kv_heads_per_rank(rank, world_size, num_kv_heads)

    block.img_instruct_attn.inner_kv_dim = kv_heads_on_rank * head_dim
    block.img_instruct_attn._tp_kv_heads = kv_heads_on_rank
    block.img_instruct_attn._tp_kv_start_head, _ = _get_overlap_kv_heads_per_rank(
        rank, world_size, num_kv_heads,
    )
    block.img_instruct_attn._tp_full_kv_heads = num_kv_heads

    _shard_attention_standard(block.img_self_attn, rank, world_size, num_attention_heads, num_kv_heads, head_dim)
    _shard_feed_forward(block.img_feed_forward, rank, world_size)
    _shard_feed_forward(block.instruct_feed_forward, rank, world_size)


def _shard_single_stream_block(block, rank, world_size, num_attention_heads, num_kv_heads, head_dim):
    """Shard a single-stream transformer block (standard attention + FFN)."""
    _shard_attention_standard(block.attn, rank, world_size, num_attention_heads, num_kv_heads, head_dim)
    _shard_feed_forward(block.feed_forward, rank, world_size)


def shard_boogu_transformer(model, rank=None, world_size=None):
    """Shard all BooguImage transformer sub-modules across TP ranks in place."""
    if rank is None:
        rank = _get_tp_rank()
    if world_size is None:
        world_size = _get_tp_ws()

    if world_size <= 1:
        return model

    num_attention_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_kv_heads
    head_dim = model.config.hidden_size // num_attention_heads

    if num_attention_heads % world_size != 0:
        raise ValueError(
            f"num_attention_heads ({num_attention_heads}) must be divisible by "
            f"world_size ({world_size})"
        )

    for block in model.noise_refiner:
        _shard_single_stream_block(block, rank, world_size, num_attention_heads, num_kv_heads, head_dim)

    for block in model.ref_image_refiner:
        _shard_single_stream_block(block, rank, world_size, num_attention_heads, num_kv_heads, head_dim)

    for block in model.context_refiner:
        _shard_single_stream_block(block, rank, world_size, num_attention_heads, num_kv_heads, head_dim)

    for block in model.double_stream_layers:
        _shard_double_stream_block(block, rank, world_size, num_kv_heads=num_kv_heads, head_dim=head_dim)

    for block in model.single_stream_layers:
        _shard_single_stream_block(block, rank, world_size, num_attention_heads, num_kv_heads, head_dim)
    return model


# ---------------------------------------------------------------------------
# Forward replacements
# ---------------------------------------------------------------------------

def _fp32_qkv_enabled():
    return os.environ.get("LB_TP_FP32_QKV")=="1"


def _fp32_linear(layer, x):
    out_dtype = x.dtype
    bias = layer.bias.float() if layer.bias is not None else None
    h = torch.nn.functional.linear(x.float(), layer.weight.float(), bias)
    return h.to(out_dtype)


def _tp_gqa_expand(key, value, attn_or_processor):
    """Expand per-rank K/V to match local query heads for GQA (TP-aware)."""
    num_q_heads = attn_or_processor.heads if hasattr(attn_or_processor, 'heads') \
        else attn_or_processor.num_attention_heads
    kv_heads = key.shape[1]

    if kv_heads >= num_q_heads:
        return key, value

    if not is_distributed_active():
        factor = num_q_heads // kv_heads
        return (
            key.repeat_interleave(factor, dim=1),
            value.repeat_interleave(factor, dim=1),
        )

    tp_kv_start_head = getattr(attn_or_processor, '_tp_kv_start_head', None)
    tp_full_kv_heads = getattr(attn_or_processor, '_tp_full_kv_heads', None)
    if tp_kv_start_head is None or tp_full_kv_heads is None:
        tp_ws = _get_tp_ws()
        full_q_heads = num_q_heads * tp_ws
        heads_per_kv_group = full_q_heads // kv_heads
        local_q = torch.arange(num_q_heads, device=key.device)
        global_q = _get_tp_rank() * num_q_heads + local_q
        kv_idx = global_q // heads_per_kv_group
        return key[:, kv_idx], value[:, kv_idx]

    return _lb_gqa_expand_kv_overlap(
        key, value, num_q_heads, tp_full_kv_heads,
        tp_kv_start_head, tp_full_kv_heads,
    )


def _compute_softmax_scale(sequence_length, base_sequence_length, scale):
    """Compute softmax scale with optional log-based scaling."""
    if base_sequence_length is not None:
        return math.sqrt(math.log(sequence_length, base_sequence_length)) * scale
    return scale


def _build_causal_attention_mask(attention_mask):
    """Build a causal attention mask from a 3-D validity mask."""
    _, seq_len, _ = attention_mask.shape
    diag_valid = torch.diagonal(attention_mask, dim1=-2, dim2=-1)
    lengths = diag_valid.sum(dim=-1)
    arange_l = torch.arange(seq_len, device=attention_mask.device)
    q_valid = arange_l.unsqueeze(0) < lengths.unsqueeze(1)
    k_valid = q_valid
    causal = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=attention_mask.device)
    )
    combined = causal & q_valid.unsqueeze(-1) & k_valid.unsqueeze(-2)
    return combined.unsqueeze(1)


def _prepare_single_stream_mask(attention_mask, batch_size):
    """Reshape or build causal attention mask for single-stream SDPA."""
    attention_mask = attention_mask.bool()
    if attention_mask.dim() == 2:
        return attention_mask.view(batch_size, 1, 1, -1)
    if attention_mask.dim() == 3:
        return _build_causal_attention_mask(attention_mask)
    raise ValueError(f"Invalid attention mask shape: {attention_mask.shape}")


def _apply_norm_and_rope(query, key, norm_q, norm_k, rotary_emb):
    """Apply Q/K layer norms and rotary positional embedding."""
    from boogu.models.embeddings import apply_rotary_emb
    if norm_q is not None:
        query = norm_q(query)
    if norm_k is not None:
        key = norm_k(key)
    if rotary_emb is not None:
        query = apply_rotary_emb(query, rotary_emb, use_real=False)
        key = apply_rotary_emb(key, rotary_emb, use_real=False)
    return query, key


def _project_single_stream_qkv(attn, hidden_states, encoder_hidden_states):
    """Project and reshape Q, K, V for single-stream attention."""
    if _fp32_qkv_enabled():
        query = _fp32_linear(attn.to_q, hidden_states)
        key = _fp32_linear(attn.to_k, encoder_hidden_states)
        value = _fp32_linear(attn.to_v, encoder_hidden_states)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

    head_dim = query.shape[-1] // attn.heads
    kv_heads = key.shape[-1] // head_dim
    batch_size = hidden_states.shape[0]

    query = query.view(batch_size, -1, attn.heads, head_dim)
    key = key.view(batch_size, -1, kv_heads, head_dim)
    value = value.view(batch_size, -1, kv_heads, head_dim)
    dtype = query.dtype
    return query, key, value, head_dim, dtype


def _align_layers_to_device(layers, device):
    """Move layers to the target device if they are not already there."""
    for layer in layers:
        if (layer.weight.device != device
                and str(layer.weight.device).lower() != "meta"
                and str(device).lower() not in {"meta", "auto"}):
            layer = layer.to(device)


def _project_double_stream_qkv(self, img_hs, instruct_hs, encoder_seq_lengths, seq_lengths, attn):
    """Project, concat, and reshape Q/K/V for double-stream attention."""
    if _fp32_qkv_enabled():
        img_query = _fp32_linear(self.img_to_q, img_hs)
        img_key = _fp32_linear(self.img_to_k, img_hs)
        img_value = _fp32_linear(self.img_to_v, img_hs)
        instruct_query = _fp32_linear(self.instruct_to_q, instruct_hs)
        instruct_key = _fp32_linear(self.instruct_to_k, instruct_hs)
        instruct_value = _fp32_linear(self.instruct_to_v, instruct_hs)
    else:
        img_query = self.img_to_q(img_hs)
        img_key = self.img_to_k(img_hs)
        img_value = self.img_to_v(img_hs)
        instruct_query = self.instruct_to_q(instruct_hs)
        instruct_key = self.instruct_to_k(instruct_hs)
        instruct_value = self.instruct_to_v(instruct_hs)
    img_list = [img_query, img_key, img_value]
    instruct_list = [instruct_query, instruct_key, instruct_value]
    query, key, value = self._concat_instruction_image_features(
        img_list, instruct_list, encoder_seq_lengths, seq_lengths,
    )

    batch_size = img_hs.shape[0]
    head_dim = query.shape[-1] // attn.heads
    kv_heads = key.shape[-1] // head_dim
    query = query.view(batch_size, -1, attn.heads, head_dim)
    key = key.view(batch_size, -1, kv_heads, head_dim)
    value = value.view(batch_size, -1, kv_heads, head_dim)
    dtype = query.dtype
    return query, key, value, head_dim, dtype


def _prepare_joint_attention_mask(joint_attention_mask, batch_size):
    """Reshape joint attention mask for double-stream SDPA."""
    joint_attention_mask = joint_attention_mask.bool()
    if joint_attention_mask.dim() == 2:
        return joint_attention_mask.view(batch_size, 1, 1, -1)
    if joint_attention_mask.dim() == 3:
        return joint_attention_mask.unsqueeze(1)
    raise ValueError(f"Invalid joint attention mask shape: {joint_attention_mask.shape}")


def _split_merge_project(self, hidden_states, encoder_seq_lengths, seq_lengths):
    """Split, project, and merge double-stream attention output."""
    split_results = self._split_instruction_image_features(
        [hidden_states], encoder_seq_lengths, seq_lengths,
    )
    instruct_hs, img_hs = split_results[0]
    instruct_projected, img_projected = _lb_concat_all_reduce_split(
        instruct_hs, self.instruct_out.weight, self.instruct_out.bias,
        img_hs, self.img_out.weight, self.img_out.bias,
        cat_dim=1, out_dim=-1,
    )
    merged = self._concat_instruction_image_features(
        [img_projected], [instruct_projected], encoder_seq_lengths, seq_lengths,
    )
    return merged[0]


def tp_single_stream_processor_call(
    self, attn, hidden_states, encoder_hidden_states,
    attention_mask=None, image_rotary_emb=None, base_sequence_length=None,
):
    """TP-aware replacement for single-stream attention processor ``__call__``."""
    if not is_distributed_active():
        return self._lb_original_call(
            attn, hidden_states, encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            base_sequence_length=base_sequence_length,
        )
    batch_size, sequence_length, _ = hidden_states.shape

    query, key, value, head_dim, dtype = _project_single_stream_qkv(
        attn, hidden_states, encoder_hidden_states,
    )

    query, key = _apply_norm_and_rope(query, key, attn.norm_q, attn.norm_k, image_rotary_emb)
    query, key = query.to(dtype), key.to(dtype)

    softmax_scale = _compute_softmax_scale(sequence_length, base_sequence_length, attn.scale)

    if attention_mask is not None:
        attention_mask = _prepare_single_stream_mask(attention_mask, batch_size)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    key, value = _tp_gqa_expand(key, value, attn)

    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, scale=softmax_scale,
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.type_as(query)

    hidden_states = _lb_mm_all_reduce(hidden_states, attn.to_out[0].weight, attn.to_out[0].bias)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


def tp_double_stream_processor_call(
    self, attn, img_hidden_states, instruct_hidden_states, joint_attention_mask=None,
    rotary_emb=None, encoder_seq_lengths=None, seq_lengths=None, base_sequence_length=None,
):
    """TP-aware replacement for double-stream attention processor ``__call__``."""
    if not is_distributed_active():
        return self._lb_original_call(
            attn, img_hidden_states, instruct_hidden_states,
            joint_attention_mask=joint_attention_mask,
            rotary_emb=rotary_emb,
            encoder_seq_lengths=encoder_seq_lengths,
            seq_lengths=seq_lengths,
            base_sequence_length=base_sequence_length,
        )
    batch_size = img_hidden_states.shape[0]

    device = img_hidden_states.device
    _align_layers_to_device(
        [self.img_to_q, self.img_to_k, self.img_to_v,
         self.instruct_to_q, self.instruct_to_k, self.instruct_to_v,
         self.instruct_out, self.img_out],
        device,
    )

    query, key, value, head_dim, dtype = _project_double_stream_qkv(
        self, img_hidden_states, instruct_hidden_states, encoder_seq_lengths, seq_lengths, attn,
    )

    sequence_length = max(seq_lengths)

    query, key = _apply_norm_and_rope(query, key, attn.norm_q, attn.norm_k, rotary_emb)
    query, key = query.to(dtype), key.to(dtype)

    softmax_scale = _compute_softmax_scale(sequence_length, base_sequence_length, attn.scale)

    if joint_attention_mask is not None:
        joint_attention_mask = _prepare_joint_attention_mask(joint_attention_mask, batch_size)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    key, value = _tp_gqa_expand(key, value, self)

    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=joint_attention_mask, scale=softmax_scale,
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.type_as(query)

    hidden_states = _split_merge_project(
        self, hidden_states, encoder_seq_lengths, seq_lengths,
    )

    hidden_states = _lb_mm_all_reduce(hidden_states, attn.to_out[0].weight, attn.to_out[0].bias)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


def tp_single_stream_processor_call_flash(self, attn, *args, **kwargs):
    """TP-aware placeholder for flash single-stream processor (delegates to original)."""
    if not is_distributed_active():
        return self._lb_original_call(attn, *args, **kwargs)
    return self._lb_original_call(attn, *args, **kwargs)


def tp_double_stream_processor_call_flash(self, attn, *args, **kwargs):
    """TP-aware placeholder for flash double-stream processor (delegates to original)."""
    if not is_distributed_active():
        return self._lb_original_call(attn, *args, **kwargs)
    return self._lb_original_call(attn, *args, **kwargs)


def tp_feed_forward_forward(self, x):
    """TP-aware replacement for feed-forward ``forward`` (SwiGLU + matmul-all-reduce)."""
    if not is_distributed_active():
        return self._lb_original_forward(x)
    if _fp32_qkv_enabled():
        h1,h2 = _fp32_linear(self.linear_1, x), _fp32_linear(self.linear_3, x)
    else:
        h1, h2 = self.linear_1(x), self.linear_3(x)
    swiglu_fun = self.swiglu
    activated = swiglu_fun(h1, h2)
    return _lb_mm_all_reduce(activated, self.linear_2.weight, self.linear_2.bias)


def tp_vae_decode(self, latents, *args, **kwargs):
    """TP-aware VAE decode: rank 0 decodes, other ranks return zero tensors."""
    if not is_distributed_active() or _get_tp_rank() == 0:
        return self._lb_original_decode(latents, *args, **kwargs)

    vae_scale_factor = getattr(self, "_lb_vae_scale_factor", 8)
    height = int(latents.shape[2] * vae_scale_factor)
    width = int(latents.shape[3] * vae_scale_factor)
    dtype = latents.dtype
    device = latents.device
    image = torch.zeros(1, 3, height, width, dtype=dtype, device=device)
    return (image,)


def _tp_encode_instruction_broadcast(pipe, kwargs, device):
    """Encode the instruction on rank 0 and broadcast embeds/mask to all ranks."""
    rank = _get_tp_rank()
    instruction = kwargs.get("instruction")
    negative_instruction = kwargs.get("negative_instruction", "")
    cache = getattr(pipe, '_lb_embeds_cache', None)
    if cache is None:
        cache={}
        pipe._lb_embeds_cache = cache
    cache_key = instruction if isinstance(instruction, str) else tuple(instruction)
    if cache_key in cache:
        return cache[cache_key]
    if rank==0:
        if getattr(pipe, "mllm", None) is not None:
            pipe.mllm.to(device)
        embeds, mask,_,_,_,_ = pipe.encode_instruction(
            instruction=instruction,
            negative_instruction=negative_instruction,
            do_classifier_free_guidance=False,
            device=device,
        )
        shape = torch.tensor(list(embeds.shape)+list(mask.shape), dtype=torch.long, device=device)
        _lb_broadcast(shape, src=0)
        _lb_broadcast(embeds, src=0)
        _lb_broadcast(mask, src=0)
    else:
        shape = torch.zeros(5, dtype=torch.long, device=device)
        _lb_broadcast(shape, src=0)
        batch, seq_len, embed_dim = shape[:3].tolist()
        mask_batch, mask_len = shape[3:5].tolist()
        embeds = torch.zeros(batch, seq_len, embed_dim, dtype=torch.bfloat16, device=device)
        mask = torch.zeros(mask_batch, mask_len, dtype=torch.int64, device=device)
        _lb_broadcast(embeds, src=0)
        _lb_broadcast(mask, src=0)
    cache[cache_key] = (embeds, mask)
    return embeds, mask


def _tp_prepare_latents_broadcast(pipe, kwargs, device):
    latent_channels = pipe.transformer.config.in_channels
    latent_h = kwargs['height']// pipe.vae_scale_factor
    latent_w = kwargs['width']// pipe.vae_scale_factor
    shape = (1,latent_channels,latent_h,latent_w)
    generator = kwargs.get("generator")
    latents = _lb_randn_tensor(shape, device=device, generator=generator, dtype=torch.bfloat16)
    return latents


def tp_pipeline_call(self, *args, **kwargs):
    """TP-aware placeholder for pipeline call (delegates to original)."""
    if not getattr(self, '_lite_boost_tp', False):
        return type(self)._lb_original_call(self, *args, **kwargs)
    device = kwargs.get("device")
    if device is None:
        if torch.npu.is_available():
            device = f"npu:{_get_tp_rank()}"
        else:
            device = f"cuda:{_get_tp_rank()}"
    if kwargs.get('instruction_embeds') is None and kwargs.get('instruction') is not None:
        embeds, mask = _tp_encode_instruction_broadcast(self, kwargs, device)
        kwargs['instruction_embeds'] = embeds
        kwargs['instruction_attention_mask'] = mask
    if kwargs.get('latents') is None:
        kwargs['latents'] = _tp_prepare_latents_broadcast(self, kwargs, device)
    return type(self)._lb_original_call(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# Forward patching (apply the TP replacements above to a transformer)
# ---------------------------------------------------------------------------

def patch_processor(processor):
    """Replace a processor's ``__call__`` with the TP-aware variant (idempotent)."""
    cls = processor.__class__
    cls_name = cls.__name__
    if cls_name.endswith("Flash2Varlen"):
        if "DoubleStream" in cls_name:
            new_call = tp_double_stream_processor_call_flash
        else:
            new_call = tp_single_stream_processor_call_flash
    else:
        if "DoubleStream" in cls_name:
            new_call = tp_double_stream_processor_call
        else:
            new_call = tp_single_stream_processor_call
    if not hasattr(processor, "_lb_original_call"):
        processor._lb_original_call = processor.__call__

    if not getattr(cls, "_lb_tp_patched", False):
        cls.__call__ = new_call
        cls._lb_tp_patched = True


def patch_ffn(ffn):
    """Replace an FFN module's ``forward`` with the TP-aware variant (idempotent)."""
    if ffn is None:
        return
    if not hasattr(ffn, "_lb_original_forward"):
        ffn._lb_original_forward = ffn.forward
    ffn.forward = types.MethodType(tp_feed_forward_forward, ffn)


def patch_single_stream_block(block):
    """Patch attention processor and feed-forward of a single-stream block."""
    patch_processor(block.attn.processor)
    patch_ffn(block.feed_forward)


def patch_double_stream_block(block):
    """Patch attention processor and dual feed-forwards of a double-stream block."""
    patch_processor(block.img_instruct_attn.processor)
    patch_ffn(block.img_feed_forward)
    patch_ffn(block.instruct_feed_forward)


def patch_transformer_forwards(transformer):
    """Patch forward methods of all blocks in the BooguImage transformer."""
    for block in transformer.noise_refiner:
        patch_single_stream_block(block)
    for block in transformer.ref_image_refiner:
        patch_single_stream_block(block)
    for block in transformer.context_refiner:
        patch_single_stream_block(block)
    for block in transformer.double_stream_layers:
        patch_double_stream_block(block)
    for block in transformer.single_stream_layers:
        patch_single_stream_block(block)
