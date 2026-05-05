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
Export Qwen3-1.7B model to ONNX format.
"""

import argparse
import gc
import sys
from pathlib import Path

import torch

try:
    import torch._dynamo

    torch._dynamo.disable()
except Exception:
    pass

try:
    from transformers import AutoModelForCausalLM
except ImportError:
    print("Error: transformers package not found or version too low.")
    print("Please install the latest version: pip install transformers")
    sys.exit(1)

try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
except Exception:
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    except Exception:
        apply_rotary_pos_emb = None


def _make_additive_causal_mask(attention_mask, q_len, k_len, past_len, dtype):
    """
    Make additive causal mask for Qwen3-1.7B inference.
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
    Text attention forward function for Qwen3-1.7B inference.
    """
    input_shape = hidden_states.shape[:-1]
    head_dim = attn_mod.head_dim
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    hidden_shape = (*input_shape, -1, head_dim)

    query_states = attn_mod.q_proj(hidden_states).view(hidden_shape)
    key_states = attn_mod.k_proj(hidden_states).view(hidden_shape)
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

    scaling = getattr(attn_mod, "scaling", 1.0 / (head_dim**0.5))
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


class Qwen3LlmPrefill(torch.nn.Module):
    """Qwen3-1.7B LLM Prefill wrapper."""

    def __init__(self, model, lm_head):
        """
        Qwen3-1.7B LLM Prefill wrapper.
        """
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids):
        """
        Prefill forward function for Qwen3-1.7B inference.
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


class Qwen3LlmDecode(torch.nn.Module):
    """Qwen3-1.7B LLM Decode wrapper."""

    def __init__(self, model, lm_head):
        """
        Qwen3-1.7B LLM Decode wrapper.
        """
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids, past_key_values):
        """
        Decode forward function for Qwen3-1.7B inference.
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


def _prepare_llm_modules(model, device: str):
    """
    Prepare LLM modules for Qwen3-1.7B inference.
    """
    lm_head = model.lm_head
    model.eval()
    lm_head.eval()
    model.to(device)
    lm_head.to(device)
    prefill = Qwen3LlmPrefill(model, lm_head).to(device).eval()
    decode = Qwen3LlmDecode(model, lm_head).to(device).eval()
    return prefill, decode, lm_head


def _get_kv_cache_config(model):
    """
    Get KV cache configuration for Qwen3-1.7B inference.
    """
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )
    return num_layers, num_kv_heads, head_dim


def _prepare_output_paths(output_dir):
    """
    Prepare output paths for Qwen3-1.7B inference.
    """
    prefill_dir = Path(output_dir) / "prefill"
    decode_dir = Path(output_dir) / "decode"
    prefill_dir.mkdir(parents=True, exist_ok=True)
    decode_dir.mkdir(parents=True, exist_ok=True)
    prefill_path = prefill_dir / "qwen3_1_7b_llm_prefill.onnx"
    decode_path = decode_dir / "qwen3_1_7b_llm_decode.onnx"
    return prefill_path, decode_path


def _create_prefill_dummy_inputs(device: str, dummy_seq_len: int):
    """
    Create dummy inputs for Qwen3-1.7B prefill.
    """
    dummy_seq = int(dummy_seq_len)
    dummy_input_ids = torch.randint(
        0, 1000, (1, dummy_seq), dtype=torch.int64, device=device
    )
    dummy_attention_mask = torch.ones(1, dummy_seq, dtype=torch.int64, device=device)
    dummy_position_ids = torch.arange(dummy_seq, device=device, dtype=torch.int64).view(
        1, -1
    )
    return dummy_seq, dummy_input_ids, dummy_attention_mask, dummy_position_ids


def _export_prefill_onnx(prefill, prefill_path: Path, dummy_inputs, use_dynamo: bool):
    """
    Export LLM prefill to ONNX format.
    """
    print(f"Exporting LLM prefill to {prefill_path}...")
    with torch.no_grad():
        torch.onnx.export(
            prefill,
            dummy_inputs,
            str(prefill_path),
            input_names=["input_ids", "attention_mask", "position_ids"],
            output_names=["logits", "present_key_values"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=use_dynamo,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "position_ids": {0: "batch", 1: "seq_len"},
                "logits": {0: "batch", 1: "seq_len"},
                "present_key_values": {1: "batch", 3: "seq_len"},
            },
        )
    print("LLM prefill exported successfully.")


def _create_decode_dummy_inputs(
    device: str, dummy_seq: int, num_layers: int, num_kv_heads: int, head_dim: int
):
    """
    Create dummy inputs for Qwen3-1.7B decode.
    """
    dummy_step = 1
    dummy_past_len = int(dummy_seq)
    dummy_input_ids_step = torch.randint(
        0, 1000, (1, dummy_step), dtype=torch.int64, device=device
    )
    dummy_attention_mask_step = torch.ones(
        1, dummy_past_len + dummy_step, dtype=torch.int64, device=device
    )
    dummy_position_ids_step = torch.tensor(
        [[dummy_past_len]], dtype=torch.int64, device=device
    )
    dummy_past = torch.zeros(
        2 * num_layers,
        1,
        num_kv_heads,
        dummy_past_len,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    return (
        dummy_input_ids_step,
        dummy_attention_mask_step,
        dummy_position_ids_step,
        dummy_past,
    )


def _export_decode_onnx(decode, decode_path: Path, dummy_inputs, use_dynamo: bool):
    """
    Export LLM decode to ONNX format.
    """
    print(f"Exporting LLM decode to {decode_path}...")
    with torch.no_grad():
        torch.onnx.export(
            decode,
            dummy_inputs,
            str(decode_path),
            input_names=[
                "input_ids",
                "attention_mask",
                "position_ids",
                "past_key_values",
            ],
            output_names=["logits", "present_key_values"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=use_dynamo,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "step"},
                "attention_mask": {0: "batch", 1: "total_seq_len"},
                "position_ids": {0: "batch", 1: "step"},
                "logits": {0: "batch", 1: "step"},
                "past_key_values": {1: "batch", 3: "past_seq_len"},
                "present_key_values": {1: "batch", 3: "total_seq_len"},
            },
        )
    print("LLM decode exported successfully.")


def export_llm_prefill_decode(
    model, output_dir, device="cpu", dummy_seq_len=8, use_dynamo=False
):
    """
    Export Qwen3-1.7B model to ONNX format.
    """
    prefill, decode, _ = _prepare_llm_modules(model, device=device)
    num_layers, num_kv_heads, head_dim = _get_kv_cache_config(model)
    prefill_path, decode_path = _prepare_output_paths(output_dir)

    dummy_seq, dummy_input_ids, dummy_attention_mask, dummy_position_ids = (
        _create_prefill_dummy_inputs(device=device, dummy_seq_len=dummy_seq_len)
    )
    _export_prefill_onnx(
        prefill=prefill,
        prefill_path=prefill_path,
        dummy_inputs=(dummy_input_ids, dummy_attention_mask, dummy_position_ids),
        use_dynamo=use_dynamo,
    )

    decode_dummy_inputs = _create_decode_dummy_inputs(
        device=device,
        dummy_seq=dummy_seq,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    _export_decode_onnx(
        decode=decode,
        decode_path=decode_path,
        dummy_inputs=decode_dummy_inputs,
        use_dynamo=use_dynamo,
    )


def main():
    """
    Main function to export Qwen3-1.7B model to ONNX format.
    """
    parser = argparse.ArgumentParser(description="Export Qwen3-1.7B to ONNX")
    parser.add_argument(
        "--model-id",
        type=str,
        default="./Qwen3-1.7B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./qwen3_1_7b_onnx", help="Output directory"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for export (cpu or cuda)"
    )
    parser.add_argument(
        "--dummy-seq-len", type=int, default=8, help="Dummy sequence length for export"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Export dtype",
    )
    parser.add_argument(
        "--use-dynamo", action="store_true", help="Use torch dynamo exporter path"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model {args.model_id} in FP16 for export...")
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=args.device,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    export_llm_prefill_decode(
        model,
        output_dir,
        args.device,
        args.dummy_seq_len,
        args.use_dynamo,
    )

    print("Clearing memory after export...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nExport finished. Files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
