# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Qwen3 transformers monkey-patches (custom Ms* operator ONNX exports).

Replaces the transformers 4.57 Qwen3 forwards (attention / decoder layer /
model / causal LM / RMSNorm) with custom-operator implementations so
``torch.onnx.export`` produces the mslite-llm NNRT graph.

Same Ms* operator contract as the Qwen2.5 exporter (``MsRotaryPosEmb`` /
``MsScatterND`` / ``MsRmsNorm`` / ``MsGroupMatmul`` / ``MsAddSoftmax`` from the
custom_ops torch_custom layer), with two Qwen3-specific differences:

* per-head Q/K RMSNorm (``q_norm`` / ``k_norm``, weight shape ``[head_dim]``)
  applied on the reshaped ``(bsz, q_len, n_heads, head_dim)`` view before the
  head-dim transpose — they export as plain ``MsRmsNorm`` nodes;
* full head-dim RoPE (NeoX half-split over ``head_dim``, same as Qwen2.5;
  Qwen3 simply has no partial-rotary config) — ``MsRotaryPosEmb`` is reused
  unchanged.

Additionally patches transformers 4.57's GGUF config mapping: the dense
``qwen3`` entry is missing ``attention.key_length -> head_dim`` (only
``qwen3_moe`` has it), so GGUF checkpoints with ``head_dim != hidden // heads``
(e.g. MiniMind-3, head_dim=96) load with the wrong default 128.

Example:
    from models.qwen3.qwen3_patch import apply_qwen3_patch

    apply_qwen3_patch()  # requires transformers 4.57.x
"""

import logging
import math
import os
import sys
import warnings
from importlib.metadata import version as _pkg_version
from typing import List, Optional, Tuple, Union

import torch
import transformers
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

# custom_ops torch_custom layer (vendored under custom_ops/) is the single
# source of truth for the Ms* operator ONNX contracts (eager reference +
# ``custom::`` symbolic).  Import the adapters directly instead of keeping
# inline duplicates.
_CUSTOM_OPS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "custom_ops"
)
sys.path.insert(0, _CUSTOM_OPS_DIR)

# pylint: disable=wrong-import-position  # adapters resolve via _CUSTOM_OPS_DIR
from torch_custom.ms_rotary_pos_emb import MsRotaryPosEmb  # noqa: E402
from torch_custom.ms_scatter_nd import MsScatterND  # noqa: E402
from torch_custom.ms_rms_norm import MsRmsNorm  # noqa: E402
from torch_custom.ms_group_matmul import MsGroupMatmul  # noqa: E402
from torch_custom.ms_add_softmax import MsAddSoftmax  # noqa: E402

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Part 1: monkey-patches — replace transformers Qwen3 forward methods with
# custom-operator (Ms*) ONNX-export implementations.
# ─────────────────────────────────────────────────────────────────────────────


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def update_kvcache(past_key, past_value, key_states, value_states, current_pos):
    """Scatter-update the KV cache (``MsScatterND``, BNSD layout).

    NNRT contract: pure ``MsScatterND`` — the device kernel writes
    ``past[:, :, pos:pos+seq_len, :] = state``.  No mask input.
    """
    key_states = key_states.to(torch.float16)
    past_key = MsScatterND.apply(past_key, current_pos, key_states, "BNSD")

    value_states = value_states.to(torch.float16)
    past_value = MsScatterND.apply(past_value, current_pos, value_states, "BNSD")

    return past_key, past_value


def rms_forward(self, hidden):
    return MsRmsNorm.apply(hidden, self.weight, self.variance_epsilon)


def qwen3_attention_forward(  # pylint: disable=unused-argument
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    past_key_value: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    valid_seq_len: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Qwen3 attention forward with RoPE lifted out (rope_cos/rope_sin inputs).

    Qwen3-specific: per-head Q/K RMSNorm (``q_norm`` / ``k_norm``, shared
    ``[head_dim]`` weight over the head-dim view) applied before the head-dim
    transpose; projections are bias-free (``attention_bias=False``).
    """
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, -1, self.config.num_attention_heads, self.head_dim)
    key_states = key_states.view(bsz, -1, self.config.num_key_value_heads, self.head_dim)
    value_states = value_states.view(bsz, -1, self.config.num_key_value_heads, self.head_dim)

    # Qwen3 per-head Q/K RMSNorm (exports as MsRmsNorm with [head_dim] weight).
    query_states = self.q_norm(query_states).transpose(1, 2)
    key_states = self.k_norm(key_states).transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # custom_ops MsRotaryPosEmb consumes BNSD q/k and [B, S, D] cos/sin.
    query_states, key_states = MsRotaryPosEmb.apply(query_states, key_states, rope_cos, rope_sin)

    key_states, value_states = update_kvcache(
        past_key_value[0], past_key_value[1], key_states, value_states, valid_seq_len
    )

    attn_weights = MsGroupMatmul.apply(query_states, key_states, True)
    attn_weights = attn_weights / math.sqrt(self.head_dim)

    attn_weights = MsAddSoftmax.apply(attn_weights, attention_mask)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = MsGroupMatmul.apply(attn_weights, value_states, False)

    if attn_output.size() != (bsz, self.config.num_attention_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.config.num_attention_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, -1, self.config.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, [key_states, value_states]


def qwen3_decoder_layer_forward(  # pylint: disable=unused-argument
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    past_key_value: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    valid_seq_len: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Qwen3 decoder-layer forward patched to consume GGUF-side rope tables."""
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        valid_seq_len=valid_seq_len,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def qwen3_model_forward(  # pylint: disable=protected-access
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    past_key_values: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    valid_seq_len: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    """Qwen3 model forward patched for rope-table injection."""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    if input_ids is not None:
        _, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        _, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            warnings.warn("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

    max_seq_len = seq_length
    if seq_length == 1:
        position_ids = (valid_seq_len - 1).unsqueeze(0)
    else:
        position_ids = torch.arange(0, max_seq_len).unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    hidden_states = inputs_embeds

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = []
    layer_idx = 0

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                past_key_value=past_key_values[layer_idx],
                output_attentions=output_attentions,
                use_cache=use_cache,
                valid_seq_len=valid_seq_len,
            )

        hidden_states = layer_outputs[0]
        if use_cache:
            past_key_value = layer_outputs[2 if output_attentions else 1]
            next_decoder_cache.append(past_key_value)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)
        layer_idx += 1
    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def qwen3_causal_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    valid_seq_len: Optional[torch.LongTensor] = None,
    lmhead_idx: torch.LongTensor = None,
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.Tensor]] = None,
    position_ids: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    """Qwen3 causal-LM forward patched for GGUF-side weights and rope tables."""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        valid_seq_len=valid_seq_len,
    )

    hidden_states = outputs[0]

    # Select the logits row: last real token in the (right-padded) chunk.
    hidden_states = hidden_states[:, lmhead_idx]

    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: patching — version check + monkey-patch application
# ─────────────────────────────────────────────────────────────────────────────


def check_version():
    try:
        return _pkg_version("transformers")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("transformers not installed") from exc


def patch_gguf_config_mapping():
    """Add the missing ``attention.key_length -> head_dim`` qwen3 GGUF mapping.

    transformers 4.57's ``GGUF_CONFIG_MAPPING["qwen3"]`` omits key_length (only
    ``qwen3_moe`` carries it), so dense Qwen3 checkpoints with
    ``head_dim != hidden_size // num_attention_heads`` (MiniMind-3: 96 vs 768/8)
    load with the Qwen3 default head_dim=128 and fail on weight shapes.
    """
    mapping = transformers.integrations.ggml.GGUF_CONFIG_MAPPING["qwen3"]
    mapping.setdefault("attention.key_length", "head_dim")


def apply_qwen3_patch():
    """Monkey-patch transformers qwen3 modules with the custom-op forwards.

    Requires transformers 4.57.x: the patched forwards match the 4.57 Qwen3
    module attributes (per-head q_norm/k_norm RMSNorm, bias-free projections)
    and GGUF skeleton loading uses the 4.57 ``gguf_file=`` kwarg /
    ``modeling_gguf_pytorch_utils``.
    """
    transformers_version = check_version()
    if not transformers_version.startswith("4.57"):
        raise RuntimeError(
            f"Transformers version {transformers_version} is NOT compatible with the qwen3 exporter. "
            "The exporter requires Transformers 4.57.x."
        )

    patch_gguf_config_mapping()

    transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.forward = qwen3_causal_model_forward
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = qwen3_attention_forward
    transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer.forward = qwen3_decoder_layer_forward
    transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward = qwen3_model_forward
    transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm.forward = rms_forward
    logger.info("Applied qwen3 patches (transformers %s).", transformers_version)
