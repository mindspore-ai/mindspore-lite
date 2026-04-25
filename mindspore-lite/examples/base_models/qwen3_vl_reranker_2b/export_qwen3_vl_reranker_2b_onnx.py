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
Export Qwen3-VL-Reranker-2B to ONNX.

This exporter follows the split style used by `examples/base_models/qwen3_vl_2b`:
- `qwen3_vl_reranker_vision.onnx`: Vision tower (fixed grid_thw by --vision-image-size)
- `qwen3_vl_reranker_score.onnx`: Reranker scoring model (single forward, outputs sigmoid score)

The scoring ONNX takes `image_embeds`/`deepstack_embeds` as inputs, which can be produced by the
vision ONNX. For text-only rerank, you can pass empty `image_embeds` and `deepstack_embeds`.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    import torch._dynamo  # type: ignore

    torch._dynamo.disable()
except Exception:
    pass

try:
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
except Exception:
    AutoTokenizer = None
    Qwen3VLForConditionalGeneration = None

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
except Exception:
    apply_rotary_pos_emb = None


def _torch_dtype(dtype: str) -> torch.dtype:
    v = str(dtype).lower()
    if v in ("fp16", "float16", "half"):
        return torch.float16
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    if v in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported --dtype: {dtype}")


def _pick_device(device: str) -> str:
    device = str(device).lower()
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: --device cuda requested but CUDA is not available. Fallback to cpu.")
        return "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("--device must be cpu or cuda")
    return device


def _clear_torch_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_model(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype,
    attn_implementation: str = "eager",
):
    """Load Qwen3VLForConditionalGeneration model."""
    if Qwen3VLForConditionalGeneration is None:
        raise RuntimeError(
            "transformers is missing or too old. Please install transformers>=4.57.0."
        )
    kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": str(attn_implementation),
        "trust_remote_code": True,
    }
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, device_map=device, **kwargs
        )
    except Exception:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        model.to(device)
    model.eval()
    return model


class VisionTowerWrapper(torch.nn.Module):
    """
    Wrapper for Qwen3-VL vision tower to cache position embeddings.

    Exported vision ONNX takes only `pixel_values` as input, and fixes `grid_thw`
    to the value derived from `--vision-image-size`.
    """

    def __init__(self, vision_tower, dummy_grid_thw: torch.Tensor):
        super().__init__()
        self.vision_tower = vision_tower
        self.dummy_grid_thw = dummy_grid_thw
        with torch.no_grad():
            cached_pos_embeds = vision_tower.fast_pos_embed_interpolate(dummy_grid_thw)
            cached_rot_pos_emb = vision_tower.rot_pos_emb(dummy_grid_thw)
        vision_tower.fast_pos_embed_interpolate = lambda x: cached_pos_embeds
        vision_tower.rot_pos_emb = lambda x: cached_rot_pos_emb

    def forward(self, pixel_values: torch.Tensor):
        """Run vision tower forward and return (image_embeds, deepstack_embeds)."""
        outputs = self.vision_tower(
            pixel_values, grid_thw=self.dummy_grid_thw, return_dict=True
        )
        # Some transformers versions / export paths may return tuple even with return_dict=True.
        if isinstance(outputs, (list, tuple)):
            image_embeds = outputs[0]
            deepstack = outputs[1] if len(outputs) > 1 else ()
        else:
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


def export_vision_tower(
    model,
    output_path: Path,
    device: str,
    vision_image_size: int,
):
    """
    Export Qwen3-VL vision tower to ONNX.
    """
    print(f"Exporting vision tower to {output_path} ...")

    vision_tower = model.model.visual
    vision_tower.eval()
    vision_tower.to(device)

    patch_size = int(model.config.vision_config.patch_size)
    if int(vision_image_size) % patch_size != 0:
        raise ValueError(
            f"--vision-image-size must be divisible by patch_size={patch_size}, got {vision_image_size}"
        )

    grid_h = int(vision_image_size) // patch_size
    grid_w = int(vision_image_size) // patch_size
    dummy_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.int64, device=device)
    dummy_seq_len = int(dummy_grid_thw[0, 0] * dummy_grid_thw[0, 1] * dummy_grid_thw[0, 2])

    in_channels = int(getattr(model.config.vision_config, "in_channels", 3))
    temporal_patch_size = int(getattr(model.config.vision_config, "temporal_patch_size", 2))
    patch_dim = patch_size * patch_size * in_channels * temporal_patch_size

    try:
        vt_dtype = next(vision_tower.parameters()).dtype
    except Exception:
        vt_dtype = torch.float16

    dummy_pixel_values = torch.randn(
        dummy_seq_len, patch_dim, device=device, dtype=vt_dtype
    )
    wrapper = VisionTowerWrapper(vision_tower, dummy_grid_thw).to(device).eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print("Exporting vision tower with legacy exporter (opset 14) ...")
        from torch.onnx import utils as onnx_utils  # pylint: disable=import-error

        with torch.no_grad():
            onnx_utils.export(
                wrapper,
                (dummy_pixel_values,),
                str(output_path),
                input_names=["pixel_values"],
                output_names=["image_embeds", "deepstack_embeds"],
                opset_version=14,
                do_constant_folding=True,
            )
        print("Vision tower exported successfully.")
    except Exception as e:
        print(f"Legacy export (opset 14) failed ({e}), retrying legacy exporter (opset 18) ...")
        from torch.onnx import utils as onnx_utils  # pylint: disable=import-error
        with torch.no_grad():
            onnx_utils.export(
                wrapper,
                (dummy_pixel_values,),
                str(output_path),
                input_names=["pixel_values"],
                output_names=["image_embeds", "deepstack_embeds"],
                opset_version=18,
                do_constant_folding=True,
            )
        print("Vision tower exported successfully.")


def _make_additive_causal_mask(
    attention_mask: torch.Tensor, q_len: int, k_len: int, past_len: int, dtype: torch.dtype
) -> torch.Tensor:
    """Build additive causal + padding mask for attention."""
    mask_value = torch.finfo(dtype).min
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    causal = causal.to(dtype) * mask_value
    causal = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len)
    padding = (1.0 - attention_mask.to(dtype)) * mask_value
    padding = padding[:, None, None, :]
    return causal + padding


def _text_position_ids(
    position_ids: Optional[torch.Tensor], batch: int, seq_len: int, device: torch.device
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Split 4D position_ids into text and multimodal parts."""
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
    attn_mod,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask: torch.Tensor,
    past_key: Optional[torch.Tensor],
    past_value: Optional[torch.Tensor],
):
    """Custom attention forward with RoPE and optional KV cache."""
    input_shape = hidden_states.shape[:-1]
    head_dim = attn_mod.head_dim
    hidden_shape = (*input_shape, -1, head_dim)
    query_states = attn_mod.q_norm(
        attn_mod.q_proj(hidden_states).view(hidden_shape)
    ).transpose(1, 2)
    key_states = attn_mod.k_norm(attn_mod.k_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    value_states = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    if apply_rotary_pos_emb is None:
        raise RuntimeError("apply_rotary_pos_emb not available; transformers too old?")
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key is not None:
        key_states = torch.cat([past_key, key_states], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward

    interface = getattr(attn_mod.config, "_attn_implementation", "eager")
    # transformers>=4.57: ALL_ATTENTION_FUNCTIONS is an AttentionInterface (Mapping-like) without get_interface.
    if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
        attention_fn = ALL_ATTENTION_FUNCTIONS.get_interface(interface, eager_attention_forward)
    else:
        attention_fn = ALL_ATTENTION_FUNCTIONS.get(interface, eager_attention_forward)
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
    return attn_output


class Qwen3VLRerankerScore(torch.nn.Module):
    """
    Qwen3-VL-Reranker scoring model (single forward).

    Inputs:
      - input_ids:       (batch, seq_len) int64
      - attention_mask:  (batch, seq_len) int64
      - position_ids:    (4, batch, seq_len) int64
      - image_embeds:    (num_image_tokens, hidden_size) fp16/bf16
      - deepstack_embeds:(num_deepstack, num_image_tokens, hidden_size) fp16/bf16
    Output:
      - score: (batch,) fp16/bf16  (sigmoid probability of "yes")
    """

    def __init__(
        self,
        text_model,
        score_linear: torch.nn.Linear,
        image_token_id: int,
        num_deepstack: int,
    ):
        super().__init__()
        self.text_model = text_model
        self.score_linear = score_linear
        self.image_token_id = int(image_token_id)
        self.num_deepstack = int(num_deepstack)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        image_embeds: torch.Tensor,
        deepstack_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Run score model forward and return sigmoid probability."""
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
        text_pos, mm_pos = _text_position_ids(position_ids, bsz, q_len, inputs_embeds.device)
        if mm_pos is None:
            mm_pos = text_pos[None, ...].expand(3, bsz, q_len)

        position_embeddings = self.text_model.rotary_emb(inputs_embeds, mm_pos)
        attn_mask = _make_additive_causal_mask(
            attention_mask, q_len, q_len, 0, inputs_embeds.dtype
        )

        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.text_model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_out = _text_attn_forward(
                layer.self_attn, hidden_states, position_embeddings, attn_mask, None, None
            )
            hidden_states = residual + attn_out

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)

            if layer_idx < self.num_deepstack:
                hidden_states = hidden_states + deepstack_dense[layer_idx]

        hidden_states = self.text_model.norm(hidden_states)
        last = hidden_states[:, -1]
        score = self.score_linear(last)
        score = torch.sigmoid(score).squeeze(-1)
        return score


def _build_score_linear(model, tokenizer, device: str) -> torch.nn.Linear:
    """Build score linear layer from lm_head weights (yes - no)."""
    token_yes = tokenizer.get_vocab().get("yes", None)
    token_no = tokenizer.get_vocab().get("no", None)
    if token_yes is None or token_no is None:
        raise RuntimeError('Tokenizer vocab does not contain required tokens: "yes"/"no".')

    lm_head_w = model.lm_head.weight.data
    w = lm_head_w[int(token_yes)] - lm_head_w[int(token_no)]
    lin = torch.nn.Linear(int(w.shape[0]), 1, bias=False).to(device)
    with torch.no_grad():
        lin.weight.copy_(w.view(1, -1))
    return lin


def _build_score_wrapper_and_dummies(model, tokenizer, device, dummy_seq_len, vision_image_size):
    """Build Qwen3VLRerankerScore wrapper and dummy inputs for ONNX export."""
    text_model = model.model.language_model
    text_model.eval()
    text_model.to(device)

    score_linear = _build_score_linear(model, tokenizer, device=device)
    score_linear.eval()
    score_linear.to(device).to(dtype=text_model.embed_tokens.weight.dtype)

    image_token_id = int(model.config.image_token_id)
    num_deepstack = len(getattr(model.config.vision_config, "deepstack_visual_indexes", []))
    hidden_size = int(model.config.hidden_size)

    patch_size = int(model.config.vision_config.patch_size)
    spatial_merge_size = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
    grid_h = int(vision_image_size) // patch_size
    grid_w = int(vision_image_size) // patch_size
    num_img_tokens = (grid_h // spatial_merge_size) * (grid_w // spatial_merge_size) * 1

    if dummy_seq_len <= num_img_tokens + 8:
        raise ValueError(
            f"--dummy-seq-len too small for vision_image_size={vision_image_size}. "
            f"Need > num_img_tokens({num_img_tokens}) + 8, got {dummy_seq_len}."
        )

    wrapper = Qwen3VLRerankerScore(
        text_model=text_model,
        score_linear=score_linear,
        image_token_id=image_token_id,
        num_deepstack=num_deepstack,
    ).to(device).eval()

    dummy_input_ids = torch.randint(
        low=0,
        high=int(getattr(model.config.text_config, "vocab_size", 151936)),
        size=(1, int(dummy_seq_len)),
        dtype=torch.int64,
        device=device,
    )
    dummy_input_ids[0, 1 : 1 + int(num_img_tokens)] = int(image_token_id)
    dummy_attention_mask = torch.ones(
        1, int(dummy_seq_len), dtype=torch.int64, device=device
    )
    base_pos = torch.arange(int(dummy_seq_len), device=device, dtype=torch.int64).view(1, -1)
    dummy_position_ids = base_pos.unsqueeze(0).expand(4, 1, int(dummy_seq_len))

    act_dtype = text_model.embed_tokens.weight.dtype
    dummy_image_embeds = torch.randn(
        int(num_img_tokens), hidden_size, device=device, dtype=act_dtype,
    )
    dummy_deepstack = torch.randn(
        max(int(num_deepstack), 1), int(num_img_tokens), hidden_size,
        device=device, dtype=act_dtype,
    )
    if int(num_deepstack) == 0:
        dummy_deepstack = dummy_deepstack[:0]

    return wrapper, (
        dummy_input_ids, dummy_attention_mask, dummy_position_ids,
        dummy_image_embeds, dummy_deepstack,
    )


def export_score_model(
    model_id: str,
    output_path: Path,
    device: str,
    torch_dtype: torch.dtype,
    opset: int,
    dummy_seq_len: int,
    vision_image_size: int,
):
    """Export Qwen3-VL-Reranker score model to ONNX."""
    if AutoTokenizer is None:
        raise RuntimeError("transformers is missing or too old. Please install transformers>=4.57.0.")

    print(f"Loading reranker model for score export: {model_id}")
    model = _load_model(model_id, device=device, torch_dtype=torch_dtype, attn_implementation="eager")
    try:
        del model.model.visual
    except Exception:
        pass
    _clear_torch_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    wrapper, dummy_inputs = _build_score_wrapper_and_dummies(
        model, tokenizer, device, dummy_seq_len, vision_image_size
    )

    input_names = [
        "input_ids", "attention_mask", "position_ids",
        "image_embeds", "deepstack_embeds",
    ]
    output_names = ["score"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "position_ids": {1: "batch", 2: "seq_len"},
        "image_embeds": {0: "num_image_tokens"},
        "deepstack_embeds": {1: "num_image_tokens"},
        "score": {0: "batch"},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting score model to {output_path} (opset {opset}) ...")
    export_kwargs = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": int(opset),
        "do_constant_folding": True,
        "dynamic_axes": dynamic_axes,
    }
    with torch.no_grad():
        from torch.onnx import utils as onnx_utils  # pylint: disable=import-error

        onnx_utils.export(
            wrapper,
            dummy_inputs,
            str(output_path),
            **export_kwargs,
        )
    print("Score model exported successfully.")

    del model
    _clear_torch_cache()


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-VL-Reranker-2B to ONNX (vision + score).")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-VL-Reranker-2B",
        help="HuggingFace model ID or local model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./qwen3_vl_reranker_onnx",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for export",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Model dtype: fp16/bf16/fp32. Default: fp16 on cuda, bf16 on cpu.",
    )
    parser.add_argument(
        "--opset", type=int, default=18, help="ONNX opset version for score model export"
    )
    parser.add_argument(
        "--vision-image-size",
        type=int,
        default=128,
        help="Vision export image size (pixels). Must be divisible by vision_config.patch_size.",
    )
    parser.add_argument(
        "--dummy-seq-len",
        type=int,
        default=128,
        help="Dummy seq len for score model export (controls tracing shape only).",
    )
    args = parser.parse_args()

    device = _pick_device(args.device)
    if args.dtype is None:
        dtype = "fp16" if device == "cuda" else "bf16"
    else:
        dtype = args.dtype
    torch_dtype = _torch_dtype(dtype)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vision_onnx = out_dir / "qwen3_vl_reranker_vision.onnx"
    score_onnx = out_dir / "qwen3_vl_reranker_score.onnx"

    print(f"Export config: device={device} dtype={torch_dtype} opset={int(args.opset)}")
    print(f"  model_id: {args.model_id}")
    print(f"  output_dir: {out_dir}")
    print(f"  vision_image_size: {int(args.vision_image_size)}  dummy_seq_len: {int(args.dummy_seq_len)}")

    print("\nStep 1/2: exporting vision tower ...")
    vision_model = _load_model(
        args.model_id, device=device, torch_dtype=torch_dtype, attn_implementation="eager"
    )
    try:
        del vision_model.model.language_model
        del vision_model.lm_head
    except Exception:
        pass
    _clear_torch_cache()
    export_vision_tower(
        vision_model,
        output_path=vision_onnx,
        device=device,
        vision_image_size=int(args.vision_image_size),
    )
    del vision_model
    _clear_torch_cache()

    print("\nStep 2/2: exporting score model ...")
    export_score_model(
        model_id=args.model_id,
        output_path=score_onnx,
        device=device,
        torch_dtype=torch_dtype,
        opset=int(args.opset),
        dummy_seq_len=int(args.dummy_seq_len),
        vision_image_size=int(args.vision_image_size),
    )

    print("\nExport finished.")
    print(f"  vision: {vision_onnx}")
    print(f"  score:  {score_onnx}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
