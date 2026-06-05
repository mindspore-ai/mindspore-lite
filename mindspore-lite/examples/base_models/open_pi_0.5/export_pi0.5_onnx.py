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
Export pi0.5 base PyTorch model to two ONNX models in float16 precision:
  1. prefix_encoder.onnx - SigLIP + PaliGemma LLM -> KV cache (float16)
  2. denoise_step.onnx   - Action Expert single denoising step -> velocity (float16)

Usage:
  python export_pi0.5_onnx.py \
    --checkpoint_dir ./pi05_base \
    --output_dir ./onnx_output_fp16
"""

import argparse
import dataclasses
import logging
import os
import sys
import types

import numpy as np
import safetensors.torch
import torch
from torch import nn
from torch.autograd import Function

class _NullTypecheckCtx:
    """No-op context manager used to stub out openpi typechecking at import time."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def _identity(fn):
    """Identity decorator used to stub openpi's typecheck decorator."""
    return fn


# Add project src to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Mock heavy dependencies that are only needed for JAX training, not PyTorch export.
def _mock_module(name, attrs=None):
    """Create a lightweight stub module and register it in sys.modules."""
    import importlib
    mod = types.ModuleType(name)
    mod.__path__ = []
    mod.__package__ = name.rsplit(".", 1)[0] if "." in name else name
    mod.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    return mod

# Create mock modules in dependency order
_mock_module("flax")
_mock_module("flax.nnx", {"Nothing": object, "Param": object, "filterlib": types.ModuleType("filterlib")})
_mock_module("flax.linen", {"Module": object})
_mock_module("flax.traverse_util")
_mock_module("jax")
_mock_module("jax.numpy", {"float32": np.float32, "int32": np.int32, "bool_": np.bool_, "uint8": np.uint8})
_mock_module("jax.random", {"key": lambda x: x, "split": lambda x: (x, x)})
_mock_module("jaxlib")
_mock_module("jaxtyping")
_mock_module("chex")
_mock_module("optax")
_mock_module("orbax")
_mock_module("orbax.checkpoint")
_mock_module("ml_collections", {"FieldReference": lambda x: x})
_mock_module("openpi.shared")
_mock_module("openpi.shared.array_typing", {
    "KeyArrayLike": object,
    "disable_typechecking": _NullTypecheckCtx,
    "typecheck": _identity,
})
_mock_module("openpi.shared.nnx_utils", {"PathRegex": lambda x: x})
_mock_module("openpi.shared.sharding", {})
_mock_module("openpi.training", {})
_mock_module("openpi.training.sharding", {})
_mock_module("openpi.models.lora", {"LoRAConfig": type("LoRAConfig", (), {"rank": 0, "alpha": 1.0})})

# Mock openpi.models.gemma with inline configs (avoids importing flax.linen)
@dataclasses.dataclass(frozen=True)
class _GemmaConfig:
    width: int = 2048
    depth: int = 18
    mlp_dim: int = 16384
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256

def _get_config(variant):
    configs = {
        "gemma_2b": _GemmaConfig(width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256),
        "gemma_300m": _GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256),
    }
    return configs.get(variant, configs["gemma_2b"])

# Pre-create the mock for openpi.models.gemma BEFORE adding src to path
_mock_module("openpi.models", {})
_mock_module("openpi.models.gemma", {
    "Config": _GemmaConfig,
    "get_config": _get_config,
    "Variant": str,
})

# Mock image_tools with torch-only resize
def _resize_with_pad_torch(image, height, width):
    """Resize an NHWC image tensor with aspect-ratio-preserving zero padding."""
    import torch.nn.functional as F
    b, h, w, c = image.shape
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_nchw = image.permute(0, 3, 1, 2)
    resized = F.interpolate(img_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
    result = torch.zeros(b, c, height, width, dtype=image.dtype, device=image.device)
    pad_h, pad_w = height - new_h, width - new_w
    top, left = pad_h // 2, pad_w // 2
    result[:, :, top:top+new_h, left:left+new_w] = resized
    return result.permute(0, 2, 3, 1)

_mock_module("openpi.shared.image_tools", {"resize_with_pad_torch": _resize_with_pad_torch})
# Now import the PyTorch model - it will use the mocked JAX/Flax modules
from openpi.models_pytorch import pi0_pytorch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_onnx")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_LLM_LAYERS = 18
HEAD_DIM = 256
SIGLIP_PATCH_SIZE = 14
PATCHES_PER_IMAGE = (224 // SIGLIP_PATCH_SIZE) ** 2  # 256


# ---------------------------------------------------------------------------
# ONNX-compatible attention mask helper
# ---------------------------------------------------------------------------
def make_att_2d_masks_onnx(pad_masks, att_masks):
    """ONNX-compatible version: cast bool to int32 before cumsum."""
    cumsum = torch.cumsum(att_masks.to(torch.int32), dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


# ---------------------------------------------------------------------------
# Simple KV cache for ONNX tracing
# ---------------------------------------------------------------------------
class SimpleCache:
    """Minimal KV cache compatible with GemmaAttention's access pattern."""

    def __init__(self, key_value_pairs):
        self._pairs = key_value_pairs

    def __getitem__(self, idx):
        return self._pairs[idx]

    def get_seq_length(self, layer_idx=0):
        return self._pairs[layer_idx][0].shape[2]


# ---------------------------------------------------------------------------
# RotaryMul: Custom ONNX operator for CANN-native RoPE fusion (denoise only)
# ---------------------------------------------------------------------------
class RotaryMulONNX(Function):
    """Wraps RoPE as a Custom ONNX node that maps to CANN RotaryMul op.

    CANN RotaryMul computes:
      y = (x * r1) + (rotate_half(x) * r2)

    This version handles concatenated [q, k] input for efficiency.

    Inputs:
      - x: concatenated [q, k] tensor (B, num_heads + num_kv_heads, seq_len, head_dim)
      - cos: cosine embedding (1, 1, seq_len, head_dim)
      - sin: sine embedding (1, 1, seq_len, head_dim)

    Output:
      - y: rotated concatenated tensor, same shape as x
    """

    @staticmethod
    def forward(_ctx, x, cos, sin):
        """Compute fused rotary embedding rotation for concatenated [q, k]."""
        def rotate_half(t):
            x1 = t[..., : t.shape[-1] // 2]
            x2 = t[..., t.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotate_half(x) * sin)

    @staticmethod
    def symbolic(g, x, cos, sin):
        return g.op(
            "Custom",
            x,
            cos,
            sin,
            type_s="RotaryMul",
            input_names_s=["x", "r1", "r2"],
            output_names_s=["y"],
            output_num_i=1,
            input_index_i=[0, 1, 2],
        )


class FusedGemmaAttention(nn.Module):
    """Attention module that uses CANN RotaryMul for RoPE.

    Optimized: concatenates [q, k] before RotaryMul, then splits result.
    Reduces Custom nodes from 2 per layer to 1 per layer.
    """

    def __init__(self, original_attn: nn.Module):
        super().__init__()
        self.config = original_attn.config
        self.layer_idx = original_attn.layer_idx
        self.head_dim = original_attn.head_dim
        self.num_key_value_groups = original_attn.num_key_value_groups
        self.scaling = original_attn.scaling
        self.attention_dropout = original_attn.attention_dropout
        self.is_causal = original_attn.is_causal
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_value=None, cache_position=None, use_cache=False, **_kwargs):
        """Run self-attention with CANN RotaryMul fusion on the concatenated [q, k] pair."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)  # (1, S, H) -> (1, 1, S, H)
        sin = sin.unsqueeze(1)
        n_q = query_states.shape[1]
        n_k = key_states.shape[1]

        # Concatenate q and k for single RotaryMul operation
        qk_concat = torch.cat([query_states, key_states], dim=1)
        qk_rotated = RotaryMulONNX.apply(qk_concat, cos, sin)
        query_states, key_states = qk_rotated.split([n_q, n_k], dim=1)

        if past_key_value is not None:
            if use_cache:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                key_states = torch.cat([past_key_value[self.layer_idx][0], key_states], dim=2)
                value_states = torch.cat([past_key_value[self.layer_idx][1], value_states], dim=2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def patch_denoise_expert_for_rope_fusion(model):
    """Patch ONLY the Action Expert (gemma_expert) attention modules with RotaryMul fusion.

    PaliGemma (paligemma) attention modules are left untouched.
    Returns the count of patched attention modules.
    """
    patch_count = 0
    expert = model.paligemma_with_expert.gemma_expert

    for name, module in expert.named_modules():
        if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj') \
           and hasattr(module, 'o_proj') and hasattr(module, 'layer_idx') \
           and not isinstance(module, FusedGemmaAttention):

            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[-1]

            if parent_name:
                parent = dict(expert.named_modules())[parent_name]
            else:
                parent = expert

            fused = FusedGemmaAttention(module)
            setattr(parent, child_name, fused)
            patch_count += 1

    return patch_count


# ---------------------------------------------------------------------------
# Prefix Encoder Wrapper
# ---------------------------------------------------------------------------
class PrefixEncoderWrapper(nn.Module):
    """Wraps PI0Pytorch to export: images + lang -> KV cache tensors."""

    def __init__(self, model: pi0_pytorch.PI0Pytorch, num_layers: int = 18):
        super().__init__()
        self.model = model
        self.num_layers = num_layers

    def forward(
        self,
        image_0: torch.Tensor,
        image_1: torch.Tensor,
        image_2: torch.Tensor,
        img_mask_0: torch.Tensor,
        img_mask_1: torch.Tensor,
        img_mask_2: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ):
        """Run prefix encoder: images + language tokens -> per-layer KV cache tensors."""
        images = [image_0, image_1, image_2]
        img_masks = [img_mask_0, img_mask_1, img_mask_2]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        prefix_att_2d_masks = make_att_2d_masks_onnx(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks.to(torch.int32), dim=1) - 1
        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)

        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        outputs = [prefix_pad_masks]
        for i in range(self.num_layers):
            kv = past_key_values[i]
            outputs.append(kv[0])
            outputs.append(kv[1])
        return tuple(outputs)


# ---------------------------------------------------------------------------
# Denoise Step Wrapper
# ---------------------------------------------------------------------------
class DenoiseStepWrapper(nn.Module):
    """Wraps PI0Pytorch to export: state + x_t + timestep + KV cache -> v_t."""

    def __init__(self, model: pi0_pytorch.PI0Pytorch, num_layers: int = 18):
        super().__init__()
        self.model = model
        self.num_layers = num_layers

    def forward(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        *kv_tensors: torch.Tensor,
    ):
        """Run a single denoise step: state + x_t + timestep + KV cache -> velocity v_t."""
        kv_pairs = []
        for i in range(self.num_layers):
            key = kv_tensors[2 * i]
            value = kv_tensors[2 * i + 1]
            kv_pairs.append((key, value))

        cache = SimpleCache(kv_pairs)

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.model.embed_suffix(
            state, x_t, timestep
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks_onnx(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks.to(torch.int32), dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks.to(torch.int32), dim=1) - 1
        full_att_2d_masks_4d = self.model._prepare_attention_masks_4d(full_att_2d_masks)

        self.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        outputs_embeds, _ = self.model.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=cache,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.model.config.action_horizon :]
        v_t = self.model.action_out_proj(suffix_out)
        return v_t


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
class SimpleConfig:
    """Minimal config for PI0Pytorch that avoids importing flax/jax."""
    pi05 = True
    action_dim = 32
    action_horizon = 50
    max_token_len = 200
    paligemma_variant = "gemma_2b"
    action_expert_variant = "gemma_300m"
    dtype = "float32"
    pytorch_compile_mode = None
    discrete_state_input = True


def load_model(checkpoint_dir: str) -> pi0_pytorch.PI0Pytorch:
    """Load PI0Pytorch model from safetensors checkpoint and convert to float16."""
    model_config = SimpleConfig()

    logger.info("Creating model: pi05=%s, action_dim=%s, action_horizon=%s, max_token_len=%s",
                model_config.pi05, model_config.action_dim,
                model_config.action_horizon, model_config.max_token_len)

    model = pi0_pytorch.PI0Pytorch(model_config)

    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    logger.info("Loading weights from %s", weight_path)
    safetensors.torch.load_model(model, weight_path)

    # Convert selected bfloat16 params to float32 first, then whole model to float16 for ONNX export
    logger.info("Converting model to float16 for ONNX export")
    model.paligemma_with_expert.to_bfloat16_for_selected_params("float32")
    model.half()
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %.2fB parameters", param_count / 1e9)
    return model


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------
def export_prefix_encoder(model, output_path, config):
    """Export the prefix encoder (SigLIP + PaliGemma LLM) to ONNX in float16."""
    logger.info("Exporting prefix encoder (float16)...")

    wrapper = PrefixEncoderWrapper(model, num_layers=NUM_LLM_LAYERS)
    wrapper.eval()

    dummy_inputs = (
        torch.randn(1, 3, 224, 224).half(),
        torch.randn(1, 3, 224, 224).half(),
        torch.randn(1, 3, 224, 224).half(),
        torch.ones(1, dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
        torch.zeros(1, config.max_token_len, dtype=torch.long),
        torch.ones(1, config.max_token_len, dtype=torch.bool),
    )

    input_names = [
        "image_0", "image_1", "image_2",
        "img_mask_0", "img_mask_1", "img_mask_2",
        "lang_tokens", "lang_masks",
    ]
    output_names = ["prefix_pad_masks"] + [
        f"kv_{t}_{i}" for i in range(NUM_LLM_LAYERS) for t in ["key", "val"]
    ]

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
            verbose=False,
            dynamo=False,
        )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Prefix encoder exported: %s (%.1f MB)", output_path, size_mb)


def export_denoise_step(model, output_path, config, prefix_seq_len):
    """Export a single denoising step (Action Expert) to ONNX in float16."""
    logger.info("Exporting denoise step (float16)...")

    wrapper = DenoiseStepWrapper(model, num_layers=NUM_LLM_LAYERS)
    wrapper.eval()

    dummy_inputs = [
        torch.randn(1, config.action_dim).half(),
        torch.randn(1, config.action_horizon, config.action_dim).half(),
        torch.tensor([1.0]).half(),
        torch.ones(1, prefix_seq_len, dtype=torch.bool),
    ]
    # Interleaved: key_0, val_0, key_1, val_1, ... (matches forward's kv_tensors[2*i]/[2*i+1])
    for i in range(NUM_LLM_LAYERS):
        dummy_inputs.append(torch.randn(1, 1, prefix_seq_len, HEAD_DIM).half())
        dummy_inputs.append(torch.randn(1, 1, prefix_seq_len, HEAD_DIM).half())

    input_names = ["state", "x_t", "timestep", "prefix_pad_masks"]
    for i in range(NUM_LLM_LAYERS):
        input_names.append(f"kv_key_{i}")
        input_names.append(f"kv_val_{i}")
    output_names = ["v_t"]

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            tuple(dummy_inputs),
            output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
            verbose=False,
            dynamo=False,
        )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Denoise step exported: %s (%.1f MB)", output_path, size_mb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Export pi0.5 base model to float16 ONNX")
    parser.add_argument(
        "--checkpoint_dir",
        default="./pi05_base",
        help="Path to model checkpoint directory (with model.safetensors)",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "onnx_output_0611_fp16"),
        help="Output directory for ONNX files",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint_dir)
    config = model.config

    # Compute prefix_seq_len: 3 images x 256 patches + max_token_len
    prefix_seq_len = 3 * PATCHES_PER_IMAGE + config.max_token_len  # 3*256+200=968

    logger.info("Model config: action_dim=%s, action_horizon=%s, max_token_len=%s, prefix_seq_len=%s",
                config.action_dim, config.action_horizon,
                config.max_token_len, prefix_seq_len)

    prefix_path = os.path.join(args.output_dir, "prefix_encoder.onnx")
    denoise_path = os.path.join(args.output_dir, "denoise_step.onnx")

    # Export prefix encoder (no RotaryMul fusion — uses original attention)
    export_prefix_encoder(model, prefix_path, config)

    # Patch ONLY Action Expert attention with RotaryMul fusion for denoise step
    logger.info("Patching Action Expert attention -> FusedGemmaAttention for CANN RotaryMul fusion")
    patch_count = patch_denoise_expert_for_rope_fusion(model)
    logger.info("RotaryMul fusion applied to Action Expert: %s attention modules patched", patch_count)

    # Export denoise step (with RotaryMul fusion)
    export_denoise_step(model, denoise_path, config, prefix_seq_len)

    # Note: ONNX export may create external data files (model.*) in the output directory.
    # These are required for the ONNX models to load. Do NOT delete them.
    # They will be embedded into the MindIR during conversion.

    prefix_mb = os.path.getsize(prefix_path) / (1024 * 1024)
    denoise_mb = os.path.getsize(denoise_path) / (1024 * 1024)
    logger.info("=" * 60)
    logger.info("ONNX Export Complete (float16)!")
    logger.info("  Prefix encoder: %s (%.1f MB) [float16, no RotaryMul]",
                prefix_path, prefix_mb)
    logger.info("  Denoise step:   %s (%.1f MB) [float16, RotaryMul x%s]",
                denoise_path, denoise_mb, patch_count)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
