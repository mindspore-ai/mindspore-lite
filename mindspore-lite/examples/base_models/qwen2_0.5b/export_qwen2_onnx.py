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
Export Qwen2-0.5B model to ONNX format (prefill + decode split).
"""

import sys
import argparse
import gc
from pathlib import Path
import torch

try:
    import torch._dynamo

    torch._dynamo.disable()
except:
    pass

try:
    from transformers import AutoModelForCausalLM
except ImportError:
    print("Error: transformers package not found or version too low.")
    print(
        "Please install the latest version: pip install git+https://github.com/huggingface/transformers"
    )
    sys.exit(1)

try:
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
except Exception:
    apply_rotary_pos_emb = None


def _make_additive_causal_mask(attention_mask, q_len, k_len, past_len, dtype):
    """
    Build additive causal + padding mask compatible with ONNX export.
    """
    mask_value = torch.finfo(dtype).min
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    causal = causal.to(dtype) * mask_value
    causal = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len)
    padding = (1.0 - attention_mask.to(dtype)) * mask_value
    padding = padding[:, None, None, :]
    return causal + padding


def _text_attn_forward(
        attn_mod, hidden_states, position_embeddings, attention_mask, past_key, past_value
):
    """
    Custom attention forward for Qwen2. Supports GQA and KV-cache concatenation.
    Qwen2 has no q_norm/k_norm; hasattr guards keep this compatible with Qwen3 too.
    """
    input_shape = hidden_states.shape[:-1]
    head_dim = attn_mod.head_dim
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    hidden_shape = (*input_shape, -1, head_dim)

    query_states = attn_mod.q_proj(hidden_states).view(hidden_shape)
    key_states = attn_mod.k_proj(hidden_states).view(hidden_shape)
    # q_norm / k_norm present in Qwen3 but not Qwen2
    if hasattr(attn_mod, "q_norm"):
        query_states = attn_mod.q_norm(query_states)
    if hasattr(attn_mod, "k_norm"):
        key_states = attn_mod.k_norm(key_states)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    if apply_rotary_pos_emb is not None:
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

    if past_key is not None:
        key_states = torch.cat([past_key, key_states], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)

    key_states_for_attn = key_states
    value_states_for_attn = value_states

    if num_kv_heads < num_heads:
        key_states_for_attn = key_states.repeat_interleave(
            num_heads // num_kv_heads, dim=1
        )
        value_states_for_attn = value_states.repeat_interleave(
            num_heads // num_kv_heads, dim=1
        )

    scaling = getattr(attn_mod, "scaling", 1.0 / (head_dim ** 0.5))
    attn_weights = (
            torch.matmul(query_states, key_states_for_attn.transpose(2, 3)) * scaling
    )
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states_for_attn)
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
    attn_output = attn_mod.o_proj(attn_output)
    return attn_output, key_states, value_states


class Qwen2LlmPrefill(torch.nn.Module):
    """
    Qwen2-0.5B prefill stage: processes the full input prompt and returns
    logits plus stacked KV cache for all layers.
    """

    def __init__(self, model, lm_head):
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head
        self.num_hidden_layers = model.config.num_hidden_layers

    def forward(self, input_ids, attention_mask, position_ids):
        """
        Qwen2LlmPrefill forward function
        """
        inputs_embeds = self.model.embed_tokens(input_ids)
        q_len = input_ids.shape[1]

        position_embeddings = self.model.rotary_emb(inputs_embeds, position_ids)
        k_len = q_len
        attn_mask = _make_additive_causal_mask(
            attention_mask, q_len, k_len, 0, inputs_embeds.dtype
        )

        hidden_states = inputs_embeds
        present = []

        for layer in self.model.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn,
                hidden_states,
                position_embeddings,
                attn_mask,
                None,
                None,
            )
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)
            present.append(pk)
            present.append(pv)

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        present_kv = torch.stack(present, dim=0)
        return logits, present_kv


class Qwen2LlmDecode(torch.nn.Module):
    """
    Qwen2-0.5B decode stage: single-token autoregressive step with KV cache.
    """

    def __init__(self, model, lm_head):
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head
        self.num_hidden_layers = model.config.num_hidden_layers

    def forward(self, input_ids, attention_mask, position_ids, past_key_values):
        """
        Qwen2LlmDecode forward function
        """
        inputs_embeds = self.model.embed_tokens(input_ids)
        q_len = input_ids.shape[1]

        position_embeddings = self.model.rotary_emb(inputs_embeds, position_ids)
        past_key_0 = past_key_values[0]
        past_len = past_key_0.shape[2]
        k_len = past_len + q_len
        attn_mask = _make_additive_causal_mask(
            attention_mask, q_len, k_len, past_len, inputs_embeds.dtype
        )

        hidden_states = inputs_embeds
        present = []

        for i, layer in enumerate(self.model.layers):
            pk_in = past_key_values[2 * i]
            pv_in = past_key_values[2 * i + 1]
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn,
                hidden_states,
                position_embeddings,
                attn_mask,
                pk_in,
                pv_in,
            )
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)
            present.append(pk)
            present.append(pv)

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        present_kv = torch.stack(present, dim=0)
        return logits, present_kv


def _setup_models(model, lm_head, device):
    """Initialize prefill and decode models."""
    model.eval()
    lm_head.eval()
    model.to(device)
    lm_head.to(device)
    return Qwen2LlmPrefill(model, lm_head).to(device).eval(), Qwen2LlmDecode(model, lm_head).to(device).eval()


def _get_model_config(model):
    """Extract needed config values."""
    return {
        "num_layers": model.config.num_hidden_layers,
        "num_kv_heads": model.config.num_key_value_heads,
        "head_dim": getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads),
    }


def _export_prefill(prefill, dummy_input_ids, dummy_attention_mask, dummy_position_ids, output_path):
    """Export prefill model to ONNX."""
    print(f"Exporting LLM prefill to {output_path}...")
    input_names = ["input_ids", "attention_mask", "position_ids"]
    output_names = ["logits", "present_key_values"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "position_ids": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
        "present_key_values": {1: "batch", 3: "seq_len"},
    }
    with torch.no_grad():
        torch.onnx.export(
            prefill,
            (dummy_input_ids, dummy_attention_mask, dummy_position_ids),
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=True,
            verbose=True,
            do_constant_folding=False,
        )
    print("LLM prefill exported successfully.")


def _create_decode_dummy_inputs(device, num_layers, num_kv_heads, head_dim, dummy_seq):
    """Create dummy inputs for decode model."""
    dummy_step = 1
    dummy_past_len = dummy_seq
    return (
        torch.randint(0, 1000, (1, dummy_step), dtype=torch.int64, device=device),
        torch.ones(1, dummy_past_len + dummy_step, dtype=torch.int64, device=device),
        torch.tensor([[dummy_past_len]], dtype=torch.int64, device=device),
        torch.zeros(2 * num_layers, 1, num_kv_heads, dummy_past_len, head_dim, dtype=torch.float16, device=device),
    )


def _export_decode(decode, inputs, output_path):
    """Export decode model to ONNX."""
    print(f"Exporting LLM decode to {output_path}...")
    input_names = ["input_ids", "attention_mask", "position_ids", "past_key_values"]
    output_names = ["logits", "present_key_values"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "step"},
        "attention_mask": {0: "batch", 1: "total_seq_len"},
        "position_ids": {0: "batch", 1: "step"},
        "logits": {0: "batch", 1: "step"},
        "past_key_values": {1: "batch", 3: "past_seq_len"},
        "present_key_values": {1: "batch", 3: "total_seq_len"},
    }
    with torch.no_grad():
        torch.onnx.export(
            decode,
            inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=True,
            verbose=True,
            do_constant_folding=False,
        )
    print("LLM decode exported successfully.")


def export_llm_prefill_decode(model, output_dir, device="cpu"):
    """
    Export Qwen2-0.5B prefill and decode to ONNX (opset 17, dynamic shapes).
    """
    lm_head = model.lm_head
    prefill, decode = _setup_models(model, lm_head, device)
    cfg = _get_model_config(model)

    prefill_path = Path(output_dir) / "qwen2_llm_prefill.onnx"
    decode_path = Path(output_dir) / "qwen2_llm_decode.onnx"

    dummy_seq = 8
    dummy_input_ids = torch.randint(0, 1000, (1, dummy_seq), dtype=torch.int64, device=device)
    dummy_attention_mask = torch.ones(1, dummy_seq, dtype=torch.int64, device=device)
    dummy_position_ids = torch.arange(dummy_seq, device=device, dtype=torch.int64).view(1, -1)

    _export_prefill(prefill, dummy_input_ids, dummy_attention_mask, dummy_position_ids, prefill_path)

    decode_inputs = _create_decode_dummy_inputs(device, cfg["num_layers"], cfg["num_kv_heads"], cfg["head_dim"],
                                                dummy_seq)
    _export_decode(decode, decode_inputs, decode_path)


def main():
    parser = argparse.ArgumentParser(description="Export Qwen2-0.5B to ONNX")
    parser.add_argument(
        "--model-id",
        type=str,
        default="../../models",
        help="HuggingFace model ID or local path (default: ../../models)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./qwen2_onnx",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for export (cpu or cuda)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model from {args.model_id} in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.to(args.device)

    export_llm_prefill_decode(model, output_dir, args.device)

    print("Clearing memory after export...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nExport finished. Files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
