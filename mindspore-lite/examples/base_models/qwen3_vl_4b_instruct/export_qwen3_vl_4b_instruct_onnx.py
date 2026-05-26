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
Export Qwen3-VL-4B-Instruct to ONNX.
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
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    print("Error: transformers package not found or version too low.")
    print(
        "Please install the latest version: pip install git+https://github.com/huggingface/transformers"
    )
    sys.exit(1)


try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
except Exception:
    apply_rotary_pos_emb = None


class VisionTowerWrapper(torch.nn.Module):
    """
    Wrapper for Qwen3-VL Vision Tower to cache position embeddings.
    """

    def __init__(self, vision_tower, dummy_grid_thw):
        """
        Initialize Qwen3-VL Vision Tower wrapper.
        """
        super().__init__()
        self.vision_tower = vision_tower
        self.dummy_grid_thw = dummy_grid_thw
        with torch.no_grad():
            cached_pos_embeds = vision_tower.fast_pos_embed_interpolate(dummy_grid_thw)
            cached_rot_pos_emb = vision_tower.rot_pos_emb(dummy_grid_thw)
        vision_tower.fast_pos_embed_interpolate = lambda x: cached_pos_embeds
        vision_tower.rot_pos_emb = lambda x: cached_rot_pos_emb

    def forward(self, pixel_values):
        """
        Forward pass for Qwen3-VL Vision Tower wrapper.
        """
        outputs = self.vision_tower(
            pixel_values, grid_thw=self.dummy_grid_thw, return_dict=True
        )
        image_embeds = outputs.pooler_output
        deepstack = outputs.deepstack_features
        if isinstance(deepstack, (list, tuple)):
            if len(deepstack) == 0:
                deepstack = image_embeds.new_zeros(
                    (0, image_embeds.shape[0], image_embeds.shape[1])
                )
            else:
                deepstack = torch.stack(deepstack, dim=0)
        return image_embeds, deepstack


def export_vision_tower(model, output_path, device="cpu", vision_image_size=128):
    """
    Export Qwen3-VL Vision Tower to ONNX.
    """
    print(f"Exporting Vision Tower to {output_path}...")

    # Qwen3-VL uses a visual encoder (usually model.model.visual)
    vision_tower = model.model.visual
    vision_tower.eval()
    vision_tower.to(device)

    patch_size = model.config.vision_config.patch_size
    grid_h = int(vision_image_size) // int(patch_size)
    grid_w = int(vision_image_size) // int(patch_size)
    dummy_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.int64).to(device)
    dummy_seq_len = int(
        dummy_grid_thw[0, 0].item()
        * dummy_grid_thw[0, 1].item()
        * dummy_grid_thw[0, 2].item()
    )
    dummy_pixel_values = torch.randn(
        dummy_seq_len, 1536, device=device, dtype=torch.float16
    )

    wrapper = VisionTowerWrapper(vision_tower, dummy_grid_thw)

    try:
        print("Exporting Vision Tower with legacy exporter...")

        # Instead of patching tolist, we use the legacy exporter's internal function
        # which bypasses the new capture logic and handles the vision tower's
        # complex return types and symbolic shapes more gracefully.
        from torch.onnx import utils as onnx_utils

        vision_tower.eval()
        wrapper.eval()

        with torch.no_grad():
            onnx_utils.export(
                wrapper,
                (dummy_pixel_values,),
                output_path,
                input_names=["pixel_values"],
                output_names=["image_embeds", "deepstack_embeds"],
                opset_version=14,
                do_constant_folding=True,
            )
        print("Vision Tower exported successfully.")
    except Exception as e:
        print(f"Failed to export Vision Tower with trace, trying direct export: {e}")
        try:
            print(f"Exporting Vision Tower to {output_path}...")
            vision_tower.eval()
            wrapper.eval()
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    (dummy_pixel_values,),
                    output_path,
                    input_names=["pixel_values"],
                    output_names=["image_embeds", "deepstack_embeds"],
                    opset_version=18,
                    do_constant_folding=True,
                )
            print("Vision Tower exported successfully.")
        except Exception as e2:
            print(f"Failed to export Vision Tower: {e2}")
            import traceback

            traceback.print_exc()


class LLMWrapper(torch.nn.Module):
    """
    Wrapper for Qwen3-VL-4B-Instruct LLM to cache position embeddings.
    """

    def __init__(self, llm):
        """
        Initialize Qwen3-VL-4B-Instruct LLM wrapper.
        """
        super().__init__()
        self.llm = llm

    def forward(self, input_ids, attention_mask, position_ids):
        """
        Forward pass for Qwen3-VL-4B-Instruct LLM decode.
        """
        # Explicitly pass arguments to the LLM's forward method
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=False,
        )
        # return_dict=False returns a tuple
        if isinstance(outputs, (list, tuple)):
            return outputs[0]
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs


def _make_additive_causal_mask(attention_mask, q_len, k_len, past_len, dtype):
    """
    Create additive causal mask for Qwen3-VL-4B-Instruct text model.
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


def _text_position_ids(position_ids, batch, seq_len, device):
    """
    Process position ids for Qwen3-VL-4B-Instruct text model.
    """
    if position_ids is None:
        base = torch.arange(seq_len, device=device).view(1, -1).expand(batch, -1)
        position_ids = base
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        mm_position_ids = position_ids[1:]
        return text_position_ids, mm_position_ids
    return position_ids, None


def _text_attn_forward(
    attn_mod, hidden_states, position_embeddings, attention_mask, past_key, past_value
):
    """
    Forward pass for Qwen3-VL-4B-Instruct text attention.
    """
    input_shape = hidden_states.shape[:-1]
    head_dim = attn_mod.head_dim
    hidden_shape = (*input_shape, -1, head_dim)
    query_states = attn_mod.q_norm(
        attn_mod.q_proj(hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    key_states = attn_mod.k_norm(
        attn_mod.k_proj(hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    value_states = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    if apply_rotary_pos_emb is None:
        raise RuntimeError("apply_rotary_pos_emb not available")
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key is not None:
        key_states = torch.cat([past_key, key_states], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward

    interface = getattr(attn_mod.config, "_attn_implementation", "eager")
    attention_fn = ALL_ATTENTION_FUNCTIONS.get_interface(
        interface, eager_attention_forward
    )
    attn_output, _ = attention_fn(
        attn_mod,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=attn_mod.scaling,
        is_causal=True,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_mod.o_proj(attn_output)
    return attn_output, key_states, value_states


class Qwen3VLLlmPrefill(torch.nn.Module):
    """
    Qwen3-VL-4B-Instruct LLM prefill model.
    """

    def __init__(self, text_model, lm_head, image_token_id: int, num_deepstack: int):
        """
        Initialize Qwen3-VL-4B-Instruct LLM prefill model.
        """
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.num_hidden_layers = text_model.config.num_hidden_layers
        self.image_token_id = int(image_token_id)
        self.num_deepstack = int(num_deepstack)

    def forward(
        self, input_ids, attention_mask, position_ids, image_embeds, deepstack_embeds
    ):
        """
        Forward pass for Qwen3-VL-4B-Instruct LLM prefill.
        """
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        image_mask = input_ids == self.image_token_id
        image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask,
            image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        )
        deepstack_dense = []
        for i in range(self.num_deepstack):
            dense = inputs_embeds.new_zeros(inputs_embeds.shape)
            dense = dense.masked_scatter(
                image_mask,
                deepstack_embeds[i].to(
                    device=inputs_embeds.device, dtype=inputs_embeds.dtype
                ),
            )
            deepstack_dense.append(dense)
        bsz, q_len = input_ids.shape
        text_pos, mm_pos = _text_position_ids(
            position_ids, bsz, q_len, inputs_embeds.device
        )
        if mm_pos is None:
            mm_pos = text_pos[None, ...].expand(3, bsz, q_len)
        position_embeddings = self.text_model.rotary_emb(inputs_embeds, mm_pos)
        k_len = q_len
        attn_mask = _make_additive_causal_mask(
            attention_mask, q_len, k_len, 0, inputs_embeds.dtype
        )
        hidden_states = inputs_embeds
        present = []
        for layer_idx, layer in enumerate(self.text_model.layers):
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
            if layer_idx < self.num_deepstack:
                hidden_states = hidden_states + deepstack_dense[layer_idx]
            present.append(pk)
            present.append(pv)
        hidden_states = self.text_model.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        present_kv = torch.stack(present, dim=0)
        return logits, present_kv


class Qwen3VLLlmDecode(torch.nn.Module):
    """
    Qwen3-VL-4B-Instruct LLM decode model.
    """

    def __init__(self, text_model, lm_head):
        """
        Qwen3-VL-4B-Instruct LLM decode model.
        """
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.num_hidden_layers = text_model.config.num_hidden_layers

    def forward(self, input_ids, attention_mask, position_ids, past_key_values):
        """
        Forward pass for Qwen3-VL-4B-Instruct LLM decode.
        """
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        bsz, q_len = input_ids.shape
        text_pos, mm_pos = _text_position_ids(
            position_ids, bsz, q_len, inputs_embeds.device
        )
        if mm_pos is None:
            mm_pos = text_pos[None, ...].expand(3, bsz, q_len)
        position_embeddings = self.text_model.rotary_emb(inputs_embeds, mm_pos)
        past_key_0 = past_key_values[0]
        past_len = past_key_0.shape[2]
        k_len = past_len + q_len
        attn_mask = _make_additive_causal_mask(
            attention_mask, q_len, k_len, past_len, inputs_embeds.dtype
        )
        hidden_states = inputs_embeds
        present = []
        for i, layer in enumerate(self.text_model.layers):
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
        hidden_states = self.text_model.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        present_kv = torch.stack(present, dim=0)
        return logits, present_kv


def _get_llm_export_meta(model):
    """
    Get metadata for Qwen3-VL-4B-Instruct LLM export.
    """
    text_model = model.model.language_model
    lm_head = model.lm_head
    num_layers = text_model.config.num_hidden_layers
    num_kv_heads = text_model.config.num_key_value_heads
    head_dim = getattr(
        text_model.config,
        "head_dim",
        text_model.config.hidden_size // text_model.config.num_attention_heads,
    )
    image_token_id = model.config.image_token_id
    num_deepstack = len(
        getattr(model.config.vision_config, "deepstack_visual_indexes", [])
    )
    return (
        text_model,
        lm_head,
        num_layers,
        num_kv_heads,
        head_dim,
        image_token_id,
        num_deepstack,
    )


def _prepare_llm_modules(model, device):
    """
    Prepare Qwen3-VL-4B-Instruct LLM modules for export.
    """
    text_model, lm_head, *_ = _get_llm_export_meta(model)
    text_model.eval()
    lm_head.eval()
    text_model.to(device)
    lm_head.to(device)
    return text_model, lm_head


def _build_llm_wrappers(text_model, lm_head, image_token_id, num_deepstack, device):
    """
    Build Qwen3-VL-4B-Instruct LLM wrappers for export.
    """
    prefill = (
        Qwen3VLLlmPrefill(
            text_model,
            lm_head,
            image_token_id=image_token_id,
            num_deepstack=num_deepstack,
        )
        .to(device)
        .eval()
    )
    decode = Qwen3VLLlmDecode(text_model, lm_head).to(device).eval()
    return prefill, decode


def _make_prefill_dummy_inputs(
    text_model, num_deepstack, device, dummy_seq=8, dummy_num_img_tokens=16
):
    """
    Create dummy inputs for Qwen3-VL-4B-Instruct LLM prefill.
    """
    dummy_input_ids = torch.randint(
        0, 1000, (1, dummy_seq), dtype=torch.int64, device=device
    )
    dummy_attention_mask = torch.ones(1, dummy_seq, dtype=torch.int64, device=device)
    base_pos = torch.arange(dummy_seq, device=device, dtype=torch.int64).view(1, -1)
    dummy_position_ids = base_pos.unsqueeze(0).expand(4, 1, dummy_seq)
    dummy_image_embeds = torch.randn(
        dummy_num_img_tokens,
        text_model.config.hidden_size,
        device=device,
        dtype=torch.float16,
    )
    dummy_deepstack = torch.randn(
        num_deepstack,
        dummy_num_img_tokens,
        text_model.config.hidden_size,
        device=device,
        dtype=torch.float16,
    )
    return (
        dummy_input_ids,
        dummy_attention_mask,
        dummy_position_ids,
        dummy_image_embeds,
        dummy_deepstack,
    )


def _make_decode_dummy_inputs(
    num_layers, num_kv_heads, head_dim, device, dummy_past_len=8, dummy_step=1
):
    """
    Create dummy inputs for Qwen3-VL-4B-Instruct LLM decode.
    """
    dummy_input_ids_step = torch.randint(
        0, 1000, (1, dummy_step), dtype=torch.int64, device=device
    )
    dummy_attention_mask_step = torch.ones(
        1, dummy_past_len + dummy_step, dtype=torch.int64, device=device
    )
    step_pos = torch.tensor([[dummy_past_len]], dtype=torch.int64, device=device)
    dummy_position_ids_step = step_pos.unsqueeze(0).expand(4, 1, dummy_step)
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


def _export_onnx(
    prefill_or_decode, onnx_path, args, input_names, output_names, dynamic_axes
):
    """
    Export Qwen3-VL-4B-Instruct LLM prefill or decode model to ONNX.
    """
    print(f"Exporting to {onnx_path}...")
    with torch.no_grad():
        torch.onnx.export(
            prefill_or_decode,
            args,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=18,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )


def export_llm_prefill_decode(model, output_dir, device="cpu"):
    """
    Export Qwen3-VL-4B-Instruct LLM prefill and decode models to ONNX.
    """
    (
        text_model,
        lm_head,
        num_layers,
        num_kv_heads,
        head_dim,
        image_token_id,
        num_deepstack,
    ) = _get_llm_export_meta(model)
    _prepare_llm_modules(model, device)
    prefill, decode = _build_llm_wrappers(
        text_model, lm_head, image_token_id, num_deepstack, device
    )

    prefill_path = Path(output_dir) / "qwen3_vl_llm_prefill.onnx"
    decode_path = Path(output_dir) / "qwen3_vl_llm_decode.onnx"

    dummy_inputs = _make_prefill_dummy_inputs(text_model, num_deepstack, device)

    prefill_input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        "image_embeds",
        "deepstack_embeds",
    ]
    prefill_output_names = ["logits", "present_key_values"]
    prefill_dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "position_ids": {1: "batch", 2: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
        "present_key_values": {1: "batch", 3: "seq_len"},
        "image_embeds": {0: "num_image_tokens"},
        "deepstack_embeds": {1: "num_image_tokens"},
    }

    _export_onnx(
        prefill,
        prefill_path,
        dummy_inputs,
        prefill_input_names,
        prefill_output_names,
        prefill_dynamic_axes,
    )
    print("LLM prefill exported successfully.")

    (
        dummy_input_ids_step,
        dummy_attention_mask_step,
        dummy_position_ids_step,
        dummy_past,
    ) = _make_decode_dummy_inputs(
        num_layers,
        num_kv_heads,
        head_dim,
        device,
    )

    decode_input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values",
    ]
    decode_output_names = ["logits", "present_key_values"]

    decode_dynamic_axes = {
        "input_ids": {0: "batch", 1: "step"},
        "attention_mask": {0: "batch", 1: "total_seq_len"},
        "position_ids": {1: "batch", 2: "step"},
        "logits": {0: "batch", 1: "step"},
        "past_key_values": {1: "batch", 3: "past_seq_len"},
        "present_key_values": {1: "batch", 3: "total_seq_len"},
    }

    _export_onnx(
        decode,
        decode_path,
        (
            dummy_input_ids_step,
            dummy_attention_mask_step,
            dummy_position_ids_step,
            dummy_past,
        ),
        decode_input_names,
        decode_output_names,
        decode_dynamic_axes,
    )
    print("LLM decode exported successfully.")


def _clear_torch_cache():
    """
    Clear PyTorch cache to free memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _export_vision_step(model_id, output_dir, device, vision_image_size):
    """
    Export Qwen3-VL-4B-Instruct vision tower model to ONNX.
    """
    print("\nStep 1/2: Exporting Vision Tower...")
    print(f"Loading model {model_id} in FP16 for Vision export...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    try:
        del model.model.language_model
        del model.lm_head
    except Exception:
        pass
    _clear_torch_cache()
    vision_path = Path(output_dir) / "qwen3_vl_vision.onnx"
    export_vision_tower(model, str(vision_path), device, vision_image_size)
    del model
    _clear_torch_cache()
    return vision_path


def _export_llm_step(model_id, output_dir, device):
    """
    Export Qwen3-VL-4B-Instruct LLM prefill and decode models to ONNX.
    """
    print("\nStep 2/2: Exporting LLM...")
    print(f"Loading model {model_id} in FP16 for LLM export...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    try:
        del model.model.visual
    except Exception:
        pass
    _clear_torch_cache()
    export_llm_prefill_decode(model, output_dir, device)
    del model
    _clear_torch_cache()


def main():
    """
    Main function for Qwen3-VL-4B-Instruct export to ONNX.
    """
    parser = argparse.ArgumentParser(description="Export Qwen3-VL-4B-Instruct to ONNX")
    parser.add_argument(
        "--model-id",
        type=str,
        default="./Qwen/Qwen3-VL-4B-Instruct",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./qwen3_vl_4b_instruct_onnx", help="Output directory"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for export (cpu or cuda)"
    )
    parser.add_argument(
        "--vision-image-size",
        type=int,
        default=128,
        help="Vision export image size. Must be divisible by vision_config.patch_size.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _export_vision_step(args.model_id, output_dir, args.device, args.vision_image_size)
    _export_llm_step(args.model_id, output_dir, args.device)

    print(f"\nExport finished. Files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
