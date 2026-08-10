#!/usr/bin/env python3
# Copyright 2025 Qwen-Image Team The HuggingFace Team
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
Qwen-Image-Edit Ulysses Sequence Parallel (USP) adapters.

Provides USPQwenDoubleStreamAttnProcessor and usp_dit_forward as drop-in
replacements for QwenDoubleStreamAttnProcessor2_0 and
QwenImageTransformer2DModel.forward respectively.

Design (joint attention over [text tokens | image latent tokens]):
  - text tokens are replicated on every rank with local head slicing
    (no communication, heads gathered back via all_gather after attention);
  - image latent tokens are split along the sequence (Ulysses all_to_all),
    then gathered back with the reverse all_to_all.
"""
import math
from typing import Optional

import torch
import torch.distributed as dist

from lite_boost.layers.attention import flash_attention
from lite_boost.parallel.context_parallel import all_to_all_4d


def _eager_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **_kwargs):
    """Eager attention fallback for 3D/4D [B, H, L, D] inputs (NPU fusion SDPA
    op is unsupported). Installed globally as an SDPA fallback covering calls
    outside the DiT joint attention, which uses lite_boost flash_attention."""
    if scale is None:
        scale = query.shape[-1] ** -0.5
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
    if is_causal:
        seq_len_q, seq_len_k = query.shape[-2], key.shape[-2]
        causal = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device).tril()
        attn_weights = attn_weights.masked_fill(~causal, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_weights = attn_weights + attn_mask
    attn_weights = torch.softmax(attn_weights, dim=-1)
    if dropout_p > 0:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)
    return torch.matmul(attn_weights, value)


def patch_eager_sdpa():
    """Globally replace F.scaled_dot_product_attention with an eager matmul
    implementation (idempotent), since the NPU fusion SDPA op is unsupported."""
    if not getattr(torch.nn.functional, "_lite_boost_eager_sdpa_patched", False):
        torch.nn.functional.scaled_dot_product_attention = _eager_sdpa
        setattr(torch.nn.functional, "_lite_boost_eager_sdpa_patched", True)


def _gather_heads(x: torch.Tensor) -> torch.Tensor:
    """All-gather the head dimension across ranks (text stream recovery).

    Uses the list-based all_gather + cat: torch_npu's
    all_gather_into_tensor does not enforce the output shape and concatenates
    along dim 0, which corrupts the head layout of [B, S, H, D] tensors.
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return x
    chunks = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(chunks, x.contiguous())
    return torch.cat(chunks, dim=2)


class USPQwenDoubleStreamAttnProcessor:
    """
    Ulysses sequence-parallel joint attention for Qwen-Image-Edit DiT blocks.

    Image tokens are split along the sequence (one chunk per rank) and
    exchanged with all_to_all inside the attention; text tokens are kept
    complete on every rank with H/world_size local heads.
    """

    def __init__(self):
        self.seq_pad = 0

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,  # image stream, local sequence chunk
        encoder_hidden_states: torch.FloatTensor = None,  # text stream, full sequence
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("USPQwenDoubleStreamAttnProcessor requires encoder_hidden_states (text stream)")
        if attention_mask is not None:
            raise ValueError(
                "USPQwenDoubleStreamAttnProcessor does not accept an external attention_mask. "
                "Pass encoder_hidden_states_mask to let the processor build the joint mask."
            )

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        heads = attn.heads
        heads_local = heads // world_size
        seq_txt = encoder_hidden_states.shape[1]

        # QKV projections
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (heads, -1))
        img_key = img_key.unflatten(-1, (heads, -1))
        img_value = img_value.unflatten(-1, (heads, -1))
        txt_query = txt_query.unflatten(-1, (heads, -1))
        txt_key = txt_key.unflatten(-1, (heads, -1))
        txt_value = txt_value.unflatten(-1, (heads, -1))

        # QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # RoPE: image tokens use the local slice of the image frequency table;
        # text tokens use the full text frequency table (replicated everywhere)
        img_freqs, txt_freqs = image_rotary_emb
        seq_img_local = hidden_states.shape[1]
        img_freqs_local = img_freqs[rank * seq_img_local: (rank + 1) * seq_img_local]
        img_query = _apply_rope(img_query, img_freqs_local)
        img_key = _apply_rope(img_key, img_freqs_local)
        txt_query = _apply_rope(txt_query, txt_freqs)
        txt_key = _apply_rope(txt_key, txt_freqs)

        # Ulysses forward: image chunk -> full image sequence with H/P heads
        img_query = all_to_all_4d(img_query)
        img_key = all_to_all_4d(img_key)
        img_value = all_to_all_4d(img_value)

        # Text stream: local head slicing, no communication needed
        txt_query = txt_query[:, :, rank * heads_local: (rank + 1) * heads_local]
        txt_key = txt_key[:, :, rank * heads_local: (rank + 1) * heads_local]
        txt_value = txt_value[:, :, rank * heads_local: (rank + 1) * heads_local]

        # Strip padding tokens added for torch.chunk alignment
        if self.seq_pad > 0:
            img_query = img_query[:, :-self.seq_pad]
            img_key = img_key[:, :-self.seq_pad]
            img_value = img_value[:, :-self.seq_pad]

        # Joint attention, order: [text, image]; flash_attention uses [B, L, H, D]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        seq_img = joint_query.shape[1] - seq_txt
        if encoder_hidden_states_mask is not None:
            # Text padding is dropped via varlen key lengths (equivalent to
            # masking those positions with -inf in the attention weights).
            text_lens = encoder_hidden_states_mask.sum(dim=1)
            q_lens = torch.full_like(text_lens, joint_query.shape[1])
            k_lens = text_lens + seq_img
        else:
            q_lens = k_lens = None

        joint_hidden_states = flash_attention(
            joint_query,
            joint_key,
            joint_value,
            q_lens=q_lens,
            k_lens=k_lens,
            causal=False,
        )

        # [B, L, H/P, D] -> [B, L, H/P*D]
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split outputs: text part / image part
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        # Text heads: all_gather to restore all heads
        txt_attn_output = txt_attn_output.unflatten(-1, (heads_local, -1))
        txt_attn_output = _gather_heads(txt_attn_output).flatten(2, 3)

        # Image: reverse all_to_all back to the local sequence chunk (all heads)
        img_attn_output = img_attn_output.unflatten(-1, (heads_local, -1))
        if self.seq_pad > 0:
            img_attn_output = torch.nn.functional.pad(img_attn_output, (0, 0, 0, 0, 0, self.seq_pad))
        img_attn_output = all_to_all_4d(img_attn_output, scatter_idx=1, gather_idx=2).flatten(2, 3)

        # Output projections
        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())

        return img_attn_output, txt_attn_output


def _apply_rope(x, freqs):
    """Real-number rotary embedding (NPU lacks complex dtype support).

    freqs: [S, D] = cat([cos, sin]) along the last dim, D = 2 * head_dim.
    """
    from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
    return apply_rotary_emb_qwen(x, freqs, use_real=False)


def usp_dit_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: Optional[list[tuple[int, int, int]]] = None,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[dict] = None,
    controlnet_block_samples=None,
    additional_t_cond=None,
    return_dict: bool = True,
) -> torch.Tensor:
    """Ulysses Sequence Parallel DiT forward for QwenImageTransformer2DModel."""
    from diffusers.models.transformers.transformer_qwenimage import compute_text_seq_len_from_mask

    hidden_states = self.img_in(hidden_states)

    timestep = timestep.to(hidden_states.dtype)

    if self.zero_cond_t:
        timestep = torch.cat([timestep, timestep * 0], dim=0)
        modulate_index = torch.tensor(
            [[0] * math.prod(sample[0]) + [1] * sum(math.prod(s) for s in sample[1:]) for sample in img_shapes],
            device=timestep.device,
            dtype=torch.int,
        )
    else:
        modulate_index = None

    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
        encoder_hidden_states, encoder_hidden_states_mask
    )

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, hidden_states, additional_t_cond)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
    )

    image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)

    # USP: pad and split the image sequence across ranks
    world_size = dist.get_world_size()
    seq_pad = (world_size - hidden_states.shape[1] % world_size) % world_size
    if seq_pad > 0:
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, seq_pad))
        img_freqs, txt_freqs = image_rotary_emb
        img_freqs = torch.nn.functional.pad(img_freqs, (0, 0, 0, seq_pad))
        image_rotary_emb = (img_freqs, txt_freqs)
        if modulate_index is not None:
            modulate_index = torch.nn.functional.pad(modulate_index, (0, seq_pad))
    for block in self.transformer_blocks:
        block.attn.processor.seq_pad = seq_pad

    hidden_states = torch.chunk(hidden_states, world_size, dim=1)[dist.get_rank()]
    if modulate_index is not None:
        modulate_index = torch.chunk(modulate_index, world_size, dim=1)[dist.get_rank()]

    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = getattr(self, "_gradient_checkpointing_func")(
                block,
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                temb,
                image_rotary_emb,
                attention_kwargs,
                modulate_index,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
                modulate_index=modulate_index,
            )

        # controlnet residual
        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            interval_control = int(math.ceil(interval_control))
            hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

    # USP: gather the image sequence back across ranks
    if world_size > 1:
        out_shape = list(hidden_states.shape)
        out_shape[1] *= world_size
        gathered = torch.empty(out_shape, device=hidden_states.device, dtype=hidden_states.dtype)
        dist.all_gather_into_tensor(gathered, hidden_states.contiguous())
        hidden_states = gathered
        if seq_pad > 0:
            hidden_states = hidden_states[:, :-seq_pad]

    if self.zero_cond_t:
        temb = temb.chunk(2, dim=0)[0]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)

    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    return Transformer2DModelOutput(sample=output)
