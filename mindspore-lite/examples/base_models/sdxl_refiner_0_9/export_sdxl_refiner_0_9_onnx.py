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
"""Export stabilityai/stable-diffusion-xl-refiner-0.9 to ONNX for MindSpore Lite.

SDXL refiner 0.9 is the second-stage of the SDXL two-stage pipeline. It takes
base latents (denoised to a high timestep by the base UNet) and refines them
towards the noise-free image. The refiner is exported as three ONNX sub-graphs
so the heavy compute runs as MindIR on Ascend:

  1. text_encoder_2: CLIPTextModelWithProjection (CLIP-G), the ONLY text
                     encoder the refiner uses (the base concatenates CLIP-L
                     +CLIP-G to 2048; the refiner uses CLIP-G alone).
                     input_ids[1,77] -> last_hidden_state[1,77,1280]
                     (penultimate hidden layer) + text_embeds[1,1280] (pooled).
  2. unet          : SDXLUNet2DConditionModel (refiner config), sample
                     [1,4,128,128] + timestep[1] + encoder_hidden_states
                     [1,77,1280] (CLIP-G penultimate, NOT the 2048 concat the
                     base uses) + added_cond_kwargs{text_embeds[1,1280],
                     time_ids[1,5]} -> noise_pred[1,4,128,128].
                     Self+cross attention is exported as the CANN
                     ``PromptFlashAttention`` Custom op (full bidirectional, no
                     mask) by monkeypatching ``torch.nn.functional
                     .scaled_dot_product_attention`` during tracing.
  3. vae_decoder   : AutoencoderKL.decode, latents[1,4,128,128] -> image
                     [1,3,1024,1024].

Refiner-specific conditioning: ``time_ids`` is a 5-tuple
``(orig_h, orig_w, crop_top, crop_left, aesthetic_score)`` (the refiner sets
``requires_aesthetics_score=True``), NOT the base's 6-tuple
``(orig_h, orig_w, crop_top, crop_left, target_h, target_w)``. The default
aesthetic score is 6.0 (positive) / 2.5 (negative), per diffusers.

Fixed shapes: 1024x1024 -> 4-ch 128x128 latent; CLIP sequence length 77.
"""

import argparse
import gc
import math
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from diffusers import (
        AutoencoderKL,
        UNet2DConditionModel,
    )
    from transformers import (
        CLIPTextModelWithProjection,
    )
except ImportError as exc:  # pragma: no cover
    print(f"Error: missing dependency: {exc}")
    print("Install: pip install torch diffusers transformers onnx")
    raise SystemExit(1)


_OPSET = 17


# ---------------------------------------------------------------------------
# Custom CANN operators (exported as ONNX ``Custom`` nodes for Ascend).
# ---------------------------------------------------------------------------


def _as_list_str(items):
    """Convert items to a list of string representations for ONNX attributes."""
    return [str(x) for x in items]


class _CustomPromptFlashAttentionFull(torch.autograd.Function):
    """CANN PromptFlashAttention (full bidirectional, no mask) for ONNX export.

    SDXL refiner UNet self-attention (over the 16384 latent tokens) and
    cross-attention (over the 77 CLIP-G tokens) are both full (no causal / no
    padding mask), so the Custom node omits ``atten_mask`` (sparse_mode=0 +
    no mask == attend-to-all). Inputs q/k/v are in BNSD layout (batch,
    num_heads, seq, head_dim) -- which is exactly what diffusers'
    ``AttnProcessor2_0`` passes to ``F.scaled_dot_product_attention``.
    """

    @staticmethod
    def forward(ctx, query, key, value, num_heads_i, num_kv_heads_i,
                scale_value_f, input_layout_s):
        """Cheap shape-preserving stub used during tracing.

        The exported ONNX uses the :meth:`symbolic` Custom node (the real CANN
        op), NOT this forward, so the returned values are irrelevant -- only
        the output shape matters for tracing downstream layers. Returning
        ``query`` (same shape, finite values) avoids materialising the
        O(seq**2) score matrix during the trace of self-attention over the
        16384 latent tokens.
        """
        del ctx, key, value, num_heads_i, num_kv_heads_i
        del scale_value_f, input_layout_s
        return query

    @staticmethod
    def symbolic(g, query, key, value, num_heads_i, num_kv_heads_i,
                 scale_value_f, input_layout_s):
        """Export a Custom node for prompt flash attention (no mask)."""
        y = g.op(
            "Custom", query, key, value,
            type_s="PromptFlashAttention",
            num_heads_i=int(num_heads_i),
            num_key_value_heads_i=int(num_kv_heads_i),
            scale_value_f=float(scale_value_f),
            input_layout_s=str(input_layout_s),
            pre_tokens_i=2147483647,
            next_tokens_i=0,
            sparse_mode_i=0,
            inner_precise_i=1,
            input_names_s=_as_list_str(["query", "key", "value"]),
            output_names_s=_as_list_str(["attention_out"]),
        )
        y.setType(query.type())
        return y


def _patch_sdxl_attention():
    """Replace ``F.scaled_dot_product_attention`` with the CANN Custom op.

    SDXL refiner UNet's ``AttnProcessor2_0`` projects q/k/v, reshapes to
    ``(batch, num_heads, seq, head_dim)`` (BNSD) and calls
    ``F.scaled_dot_product_attention(query, key, value, attn_mask=None,
    dropout_p=0.0, is_causal=False)``. We intercept that call so the q/k/v are
    already in BNSD (no transpose needed), emit the Custom op, and return its
    output. The q/k/v projections, group-norm, LayerNorm (norm_q/norm_k),
    residual add and output projection stay as the original (standard)
    diffusers code.

    Only the refiner UNet path is affected during tracing; the VAE uses a
    separate processor that does not go through SDPA, and the text encoder
    does not call SDPA with the 4D q/k/v either.
    """
    _orig_sdpa = F.scaled_dot_product_attention

    def _custom_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None, enable_gqa=False):
        del attn_mask, dropout_p, is_causal, enable_gqa
        # query layout here is (batch, num_heads, seq, head_dim) [BNSD].
        num_heads = int(query.shape[1])
        head_dim = int(query.shape[-1])
        scale_val = float(scale) if scale is not None \
            else float(1.0 / math.sqrt(head_dim))
        return _CustomPromptFlashAttentionFull.apply(
            query, key, value, num_heads, num_heads, scale_val, "BNSD")

    F.scaled_dot_product_attention = _custom_sdpa
    return _orig_sdpa


def _restore_attention(_orig_sdpa):
    """Restore the original ``F.scaled_dot_product_attention``."""
    F.scaled_dot_product_attention = _orig_sdpa


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _clear_cache():
    """Release torch caches and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _export_onnx(module, dummy_inputs, input_names, output_names, out_path):
    """Trace a module to ONNX with the legacy exporter (opset 17, float32)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module, tuple(dummy_inputs), str(out_path),
            input_names=input_names, output_names=output_names,
            opset_version=_OPSET,
            # Disable constant folding: with it on, the tracer materialises
            # large folded constants for the self-attention over the 16384
            # latent tokens and OOMs on CPU. The converter does its own
            # ascend_oriented optimisation, so export-time folding is
            # unnecessary.
            do_constant_folding=False,
            dynamo=False)
    _clear_cache()
    print(f"[export] saved {out_path}")


# ---------------------------------------------------------------------------
# Sub-module wrappers.
# ---------------------------------------------------------------------------


class _TextEncoder2Wrapper(torch.nn.Module):
    """CLIPTextModelWithProjection: input_ids -> (last_hidden, pooled).

    The refiner uses only CLIP-G (text_encoder_2). Following the diffusers
    ``encode_prompt`` convention, ``last_hidden_state`` is the *penultimate*
    hidden layer (``hidden_states[-2]``), which is the cross-attention context
    [1,77,1280]. ``text_embeds`` is the pooled CLIP-G output [1,1280], used as
    ``added_cond_kwargs.text_embeds``.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids):
        """Return (last_hidden_state[1,77,1280], text_embeds[1,1280])."""
        out = self.encoder(input_ids, output_hidden_states=True)
        last_hidden = out.hidden_states[-2]
        pooled = out.text_embeds
        return last_hidden, pooled


class _UnetWrapper(torch.nn.Module):
    """SDXL refiner UNet2DConditionModel.

    ``added_cond_kwargs`` carries the refiner micro-conditioning:
    ``text_embeds`` (pooled CLIP-G output, [1,1280]) and ``time_ids`` -- a
    5-value tuple ``[orig_h, orig_w, crop_top, crop_left, aesthetic_score]``
    because the refiner sets ``requires_aesthetics_score=True`` (NOT the
    base's 6-value ``[..., target_h, target_w]``).
    """

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states,
                text_embeds, time_ids):
        """Return the predicted noise [1,4,128,128]."""
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]


class _VaeWrapper(torch.nn.Module):
    """AutoencoderKL.decode: latents -> image (latents already denormalised)."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        """Return the decoded image [1,3,1024,1024]."""
        return self.vae.decode(latents, return_dict=False)[0]


# ---------------------------------------------------------------------------
# Export entry points (one component at a time to keep host memory bounded).
# ---------------------------------------------------------------------------


def export_text_encoder_2(model_dir, output_dir, dtype):
    """Export CLIP-G text encoder 2 to ONNX (last_hidden + pooled).

    The refiner uses only CLIP-G; its ``cross_attention_dim`` is 1280 (the
    CLIP-G penultimate hidden size), NOT the base's 2048 concat.
    """
    print("\n[text_encoder_2] loading CLIPTextModelWithProjection ...")
    encoder = CLIPTextModelWithProjection.from_pretrained(
        Path(model_dir) / "text_encoder_2", torch_dtype=dtype,
        low_cpu_mem_usage=True).eval()
    wrapper = _TextEncoder2Wrapper(encoder).to("cpu").eval()
    dummy_ids = torch.zeros((1, 77), dtype=torch.int64)
    _export_onnx(
        wrapper, (dummy_ids,),
        input_names=["input_ids"],
        output_names=["last_hidden_state", "text_embeds"],
        out_path=Path(output_dir) / "sdxl_text_encoder_2.onnx")
    del encoder, wrapper
    _clear_cache()


def export_unet(model_dir, output_dir, height, width, dtype, use_custom_op,
                aesthetic_score=6.0):
    """Export the SDXL refiner UNet to ONNX (attention -> CANN Custom op)."""
    print("\n[unet] loading UNet2DConditionModel (refiner config) ...")
    unet = UNet2DConditionModel.from_pretrained(
        Path(model_dir) / "unet", torch_dtype=dtype,
        low_cpu_mem_usage=True).eval()
    print(f"[unet] config: cross_attention_dim="
          f"{unet.config.cross_attention_dim}, in_channels="
          f"{unet.config.in_channels}, out_channels="
          f"{unet.config.out_channels}, addition_time_embed_dim="
          f"{unet.config.addition_time_embed_dim}")
    # Sanity: the refiner must use CLIP-G (1280) as the cross-attn context.
    assert unet.config.cross_attention_dim == 1280, (
        "Expected SDXL refiner cross_attention_dim=1280 (CLIP-G only), "
        f"got {unet.config.cross_attention_dim}.")
    assert unet.config.in_channels == 4, (
        "Expected SDXL refiner in_channels=4 (latent denoiser), "
        f"got {unet.config.in_channels}.")

    wrapper = _UnetWrapper(unet).to("cpu").eval()

    orig_sdpa = None
    if use_custom_op:
        orig_sdpa = _patch_sdxl_attention()

    latent_h, latent_w = height // 8, width // 8  # 128
    sample = torch.randn(1, 4, latent_h, latent_w, dtype=dtype)
    timestep = torch.tensor([999.0], dtype=dtype)
    # Refiner uses CLIP-G penultimate hidden ONLY -> 1280 (no CLIP-L concat).
    encoder_hidden_states = torch.randn(1, 77, 1280, dtype=dtype)
    text_embeds = torch.randn(1, 1280, dtype=dtype)
    # Refiner micro-conditioning (requires_aesthetics_score=True):
    # (orig_h, orig_w, crop_top, crop_left, aesthetic_score) -- 5 values.
    time_ids = torch.tensor(
        [[float(height), float(width), 0.0, 0.0, float(aesthetic_score)]],
        dtype=dtype)

    try:
        _export_onnx(
            wrapper, (sample, timestep, encoder_hidden_states, text_embeds,
                      time_ids),
            input_names=["sample", "timestep", "encoder_hidden_states",
                         "text_embeds", "time_ids"],
            output_names=["noise_pred"],
            out_path=Path(output_dir) / "sdxl_unet.onnx")
    finally:
        if orig_sdpa is not None:
            _restore_attention(orig_sdpa)
    del unet, wrapper
    _clear_cache()


def export_vae(model_dir, output_dir, height, width, dtype):
    """Export the SDXL refiner VAE decoder (AutoencoderKL.decode) to ONNX.

    The refiner ships its own VAE (architecturally identical to the base
    SDXL VAE); we export it from the refiner weights so the example is fully
    self-contained.
    """
    print("\n[vae] loading AutoencoderKL ...")
    vae = AutoencoderKL.from_pretrained(
        Path(model_dir) / "vae", torch_dtype=dtype,
        low_cpu_mem_usage=True).eval()
    wrapper = _VaeWrapper(vae).to("cpu").eval()
    latent_h, latent_w = height // 8, width // 8  # 128
    dummy_z = torch.randn(1, 4, latent_h, latent_w, dtype=dtype)
    _export_onnx(
        wrapper, (dummy_z,),
        input_names=["latents"], output_names=["image"],
        out_path=Path(output_dir) / "sdxl_vae_decoder.onnx")
    del vae, wrapper
    _clear_cache()


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export stable-diffusion-xl-refiner-0.9 components to ONNX")
    parser.add_argument("--model-id", type=str,
                        default="./stable-diffusion-xl-refiner-0.9",
                        help="Local diffusers weights directory (or HF id).")
    parser.add_argument("--output-dir", type=str,
                        default="./sdxl_refiner_0_9_onnx",
                        help="Output directory for ONNX files.")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1024, 1024],
                        metavar=("HEIGHT", "WIDTH"),
                        help="Output image resolution (fixed for SDXL refiner).")
    parser.add_argument("--dtype", type=str, default="fp32",
                        choices=["fp16", "fp32"],
                        help="Export dtype (fp32 recommended for converter).")
    parser.add_argument("--no-custom-op", action="store_true",
                        help="Do NOT replace attention with the CANN "
                             "PromptFlashAttention op.")
    parser.add_argument("--aesthetic-score", type=float, default=6.0,
                        help="Aesthetic score baked into time_ids "
                             "(refiner positive conditioning).")
    parser.add_argument("--components", type=str,
                        default="text_encoder_2,unet,vae",
                        help="Comma-separated subset to export.")
    return parser.parse_args()


def main():
    """Parse arguments and export the requested SDXL refiner components."""
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    height, width = args.resolution
    wanted = {c.strip() for c in args.components.split(",") if c.strip()}
    use_custom_op = not bool(args.no_custom_op)

    print(f"SDXL refiner 0.9 ONNX export: resolution={height}x{width}, "
          f"dtype={args.dtype}, custom_op={use_custom_op}, "
          f"aesthetic_score={args.aesthetic_score}")

    if "text_encoder_2" in wanted:
        export_text_encoder_2(args.model_id, out, dtype)
    if "unet" in wanted:
        export_unet(args.model_id, out, height, width, dtype, use_custom_op,
                    aesthetic_score=args.aesthetic_score)
    if "vae" in wanted:
        export_vae(args.model_id, out, height, width, dtype)

    print(f"\nAll requested components exported to {out}")


if __name__ == "__main__":
    main()
