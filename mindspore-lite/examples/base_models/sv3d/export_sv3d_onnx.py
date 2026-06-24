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
"""Export stabilityai/sv3d to ONNX for MindSpore Lite.

SV3D is an image-to-video diffusion model (fixed 1024x576, 25 frames). The
pipeline is exported as three ONNX sub-graphs so the heavy compute runs as
MindIR on Ascend:

  1. image_encoder : CLIPVisionModelWithProjection (CLIP-ViT-H, frozen),
                     pixel_values[1,3,224,224] -> image_embeds[1,1024]. The
                     embeds become the UNet cross-attention conditioning
                     (encoder_hidden_states, seq_len=1, cross_attention_dim=1024).
  2. unet          : UNetSpatioTemporalConditionModel, sample[2,25,8,72,128]
                     (8 = 4 noise + 4 VAE image latent concat on channel dim,
                     2 = CFG uncond+cond batch) + timestep[2]
                     + encoder_hidden_states[2,1,1024] (CLIP image_embeds)
                     + added_time_ids[2,3] (fps-1, motion_bucket_id,
                     noise_aug_strength) -> noise_pred[2,25,4,72,128].
                     Self+cross attention is exported as the CANN
                     ``PromptFlashAttention`` Custom op (full bidirectional, no
                     mask) by monkeypatching ``F.scaled_dot_product_attention``
                     during tracing.
  3. vae_decoder   : AutoencoderKLTemporalDecoder.decode, latents[1,4,72,128]
                     -> frame image[1,3,576,1024] (num_frames fixed at export
                     time, applied per decode-chunk).

VAE *encoder* is NOT exported -- it runs once on CPU torch to produce the
first-frame conditioning latent. The EulerDiscreteScheduler also runs on CPU.

Fixed geometry: 1024x576 -> 4-ch 72x128 latent (H/8=72, W/8=128); 25 frames
(the SV3D default, ``unet.config.num_frames``).
"""

import argparse
import gc
import math
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from diffusers import (
        AutoencoderKLTemporalDecoder,
        UNetSpatioTemporalConditionModel,
    )
    from transformers import CLIPVisionModelWithProjection
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

    SVD UNet's spatio-temporal attention (self-attention over the flattened
    frame tokens, and cross-attention over the single CLIP image-embed token)
    are both full (no causal / no padding mask), so the Custom node omits
    ``atten_mask`` (sparse_mode=0 + no mask == attend-to-all). Inputs q/k/v are
    in BNSD layout (batch, num_heads, seq, head_dim) -- which is exactly what
    diffusers' ``AttnProcessor2_0`` passes to
    ``F.scaled_dot_product_attention``.
    """

    @staticmethod
    def forward(ctx, query, key, value, num_heads_i, num_kv_heads_i,
                scale_value_f, input_layout_s):
        """Cheap shape-preserving stub used during tracing.

        The exported ONNX uses the :meth:`symbolic` Custom node (the real CANN
        op), NOT this forward, so the returned values are irrelevant -- only
        the output shape matters for tracing downstream layers. Returning
        ``query`` (same shape, finite values) avoids materialising the
        O(seq**2) score matrix during the trace of the spatio-temporal
        self-attention over the many frame tokens.
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


def _patch_svd_attention():
    """Replace ``F.scaled_dot_product_attention`` with the CANN Custom op.

    SVD UNet's ``AttnProcessor2_0`` projects q/k/v, reshapes to
    ``(batch, num_heads, seq, head_dim)`` (BNSD) and calls
    ``F.scaled_dot_product_attention(query, key, value, attn_mask=None,
    dropout_p=0.0, is_causal=False)``. We intercept that call so the q/k/v are
    already in BNSD (no transpose needed), emit the Custom op, and return its
    output. The q/k/v projections, group-norm, residual add and output
    projection stay as the original (standard) diffusers code.

    Only the SVD UNet path is affected during tracing; the temporal VAE decoder
    uses a separate processor that does not go through SDPA with 4D q/k/v, and
    the CLIP image encoder does not call SDPA with 4D q/k/v either.
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
            # large folded constants for the spatio-temporal self-attention
            # and OOMs on CPU. The converter does its own ascend_oriented
            # optimisation, so export-time folding is unnecessary.
            do_constant_folding=False,
            dynamo=False)
    _clear_cache()
    print(f"[export] saved {out_path}")


# ---------------------------------------------------------------------------
# Sub-module wrappers.
# ---------------------------------------------------------------------------


class _ImageEncoderWrapper(torch.nn.Module):
    """CLIPVisionModelWithProjection: pixel_values -> image_embeds[1,1024].

    The SVD pipeline takes only the projected ``image_embeds`` (CLIP-ViT-H
    projection_dim=1024) as conditioning; ``last_hidden_state`` is unused.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values):
        """Return the projected image embedding [1,1024]."""
        out = self.encoder(pixel_values)
        return out.image_embeds


class _UnetWrapper(torch.nn.Module):
    """UNetSpatioTemporalConditionModel: sample/timestep/embeds/added_time.

    ``sample`` is the channel-concatenated [noise(4) | image_latent(4)] over
    dim=2 (channels), shape ``[batch, num_frames, 8, h, w]``. The UNet itself
    flattens batch*frames internally and re-applies the per-frame embeddings.

    ``added_time_ids`` is the 3-value SVD micro-conditioning (fps-1,
    motion_bucket_id, noise_aug_strength), encoded via the sinusoidal
    ``add_time_proj`` and ``add_embedding`` (projection_class_embeddings_input
    _dim=768 = 3 * 256).
    """

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states,
                added_time_ids):
        """Return the predicted noise [batch, num_frames, 4, h, w]."""
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
            return_dict=False,
        )[0]


class _VaeDecoderWrapper(torch.nn.Module):
    """AutoencoderKLTemporalDecoder.decode for a fixed chunk of frames.

    The temporal decoder runs a Conv3d over the frame axis and therefore needs
    ``num_frames`` fixed at export time. We export a single-frame chunk
    (``num_frames=1``): the host then decodes each frame independently and the
    temporal conv reduces to a per-frame conv (kernel (3,1,1) over a 1-frame
    window), which matches the diffusers ``decode_chunk_size=1`` decode path.
    """

    def __init__(self, vae, num_frames=1):
        super().__init__()
        self.vae = vae
        self.num_frames = int(num_frames)

    def forward(self, latents):
        """Decode one frame-chunk of latents -> image [n,3,H,W]."""
        return self.vae.decode(
            latents, num_frames=self.num_frames, return_dict=False)[0]


# ---------------------------------------------------------------------------
# Export entry points (one component at a time to keep host memory bounded).
# ---------------------------------------------------------------------------


def export_image_encoder(model_dir, output_dir, dtype):
    """Export CLIP vision encoder (image_embeds) to ONNX."""
    print("\n[image_encoder] loading CLIPVisionModelWithProjection ...")
    encoder = CLIPVisionModelWithProjection.from_pretrained(
        Path(model_dir) / "image_encoder", torch_dtype=dtype,
        low_cpu_mem_usage=True).eval()
    wrapper = _ImageEncoderWrapper(encoder).to("cpu").eval()
    # CLIP-ViT-H image input is 224x224x3 (the SVD pipeline resizes the
    # conditioning image to 224x224 before the feature_extractor normalises).
    dummy_pixels = torch.randn(1, 3, 224, 224, dtype=dtype)
    _export_onnx(
        wrapper, (dummy_pixels,),
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        out_path=Path(output_dir) / "svd_image_encoder.onnx")
    del encoder, wrapper
    _clear_cache()


def export_unet(model_dir, output_dir, height, width, num_frames, dtype,
                use_custom_op):
    """Export SVD UNet to ONNX (attention -> CANN Custom op).

    Batch dim is fixed at 2 to support classifier-free guidance (the
    uncond+cond pair is run in a single forward). 8 input channels = 4 noise
    + 4 VAE image latent (channel-concatenated by the pipeline on dim=2).
    """
    print("\n[unet] loading UNetSpatioTemporalConditionModel ...")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        Path(model_dir) / "unet", torch_dtype=dtype,
        low_cpu_mem_usage=True).eval()
    wrapper = _UnetWrapper(unet).to("cpu").eval()

    orig_sdpa = None
    if use_custom_op:
        orig_sdpa = _patch_svd_attention()

    latent_h, latent_w = height // 8, width // 8  # 72, 128
    in_channels = unet.config.in_channels  # 8 = 4 + 4
    batch = 2  # CFG x2 (uncond + cond)
    sample = torch.randn(batch, num_frames, in_channels, latent_h, latent_w,
                         dtype=dtype)
    timestep = torch.tensor([999.0], dtype=dtype).expand(batch)
    # CLIP image_embeds projected to 1024, unsqueezed to seq_len=1 by pipeline.
    encoder_hidden_states = torch.randn(batch, 1, 1024, dtype=dtype)
    # SVD added_time_ids: (fps-1=6, motion_bucket_id=127, noise_aug=0.02).
    added_time_ids = torch.tensor(
        [[6.0, 127.0, 0.02]], dtype=dtype).repeat(batch, 1)

    try:
        _export_onnx(
            wrapper,
            (sample, timestep, encoder_hidden_states, added_time_ids),
            input_names=["sample", "timestep", "encoder_hidden_states",
                         "added_time_ids"],
            output_names=["noise_pred"],
            out_path=Path(output_dir) / "svd_unet.onnx")
    finally:
        if orig_sdpa is not None:
            _restore_attention(orig_sdpa)
    del unet, wrapper
    _clear_cache()


def export_vae_decoder(model_dir, output_dir, height, width, dtype):
    """Export SVD temporal VAE decoder (single-frame chunk) to ONNX."""
    print("\n[vae] loading AutoencoderKLTemporalDecoder ...")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        Path(model_dir) / "vae", torch_dtype=dtype,
        low_cpu_mem_usage=True).eval()
    wrapper = _VaeDecoderWrapper(vae, num_frames=1).to("cpu").eval()
    latent_h, latent_w = height // 8, width // 8  # 72, 128
    # Single-frame chunk: [1, 4, 72, 128] -> [1, 3, 576, 1024].
    dummy_z = torch.randn(1, 4, latent_h, latent_w, dtype=dtype)
    _export_onnx(
        wrapper, (dummy_z,),
        input_names=["latents"], output_names=["image"],
        out_path=Path(output_dir) / "svd_vae_decoder.onnx")
    del vae, wrapper
    _clear_cache()


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export sv3d components "
                    "to ONNX")
    parser.add_argument("--model-id", type=str,
                        default="./sv3d",
                        help="Local diffusers weights directory (or HF id).")
    parser.add_argument("--output-dir", type=str,
                        default="./sv3d_onnx",
                        help="Output directory for ONNX files.")
    parser.add_argument("--resolution", type=int, nargs=2,
                        default=[576, 1024], metavar=("HEIGHT", "WIDTH"),
                        help="Output video frame resolution (fixed for SVD). "
                             "SV3D default is 576 1024 (16:9).")
    parser.add_argument("--num-frames", type=int, default=25,
                        help="Number of video frames (SV3D default 25).")
    parser.add_argument("--dtype", type=str, default="fp32",
                        choices=["fp16", "fp32"],
                        help="Export dtype (fp32 recommended for converter).")
    parser.add_argument("--no-custom-op", action="store_true",
                        help="Do NOT replace attention with the CANN "
                             "PromptFlashAttention op.")
    parser.add_argument("--components", type=str,
                        default="image_encoder,unet,vae_decoder",
                        help="Comma-separated subset to export.")
    return parser.parse_args()


def main():
    """Parse arguments and export the requested SVD components."""
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    height, width = args.resolution
    num_frames = int(args.num_frames)
    wanted = {c.strip() for c in args.components.split(",") if c.strip()}
    use_custom_op = not bool(args.no_custom_op)

    print(f"SV3D ONNX export: resolution={height}x{width}, "
          f"frames={num_frames}, dtype={args.dtype}, "
          f"custom_op={use_custom_op}")

    if "image_encoder" in wanted:
        export_image_encoder(args.model_id, out, dtype)
    if "unet" in wanted:
        export_unet(args.model_id, out, height, width, num_frames, dtype,
                    use_custom_op)
    if "vae_decoder" in wanted:
        export_vae_decoder(args.model_id, out, height, width, dtype)

    print(f"\nAll requested components exported to {out}")


if __name__ == "__main__":
    main()
