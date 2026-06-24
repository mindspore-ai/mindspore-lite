"""Export Wan2.2-I2V-A14B-Diffusers submodules to ONNX for MindSpore Lite.

Wan2.2-I2V-A14B is an image-to-video MoE model. Crucially the Wan2.2-A14B
"Mixture-of-Experts" is **NOT** a per-token top-k router MoE -- it is a
**two-expert dense-conditional** design specialised for diffusion denoising
(see the Wan2.2 model card / paper):

  * a **high-noise expert** (the Wan2.2 transformer, shipped as
    ``transformer/``) that handles the early (high-noise) denoising steps;
  * a **low-noise expert** (shipped as ``transformer_2/``) that handles the
    late (low-noise) denoising steps.

The two experts are switched at a fixed timestep threshold
``boundary_ratio * num_train_timesteps`` (the Wan2.2 ``t_moe`` SNR threshold).
**The switching happens in the scheduler loop (Python), not inside the traced
graph**, so each expert is a plain dense ``WanTransformer3DModel`` and there is
no data-dependent control flow to JIT-trace. This makes the export fully
trace-safe -- no monkeypatch of routing, no gather-instead-of-boolean-indexing
tricks are needed; both experts export as independent dense DiTs.

In diffusers this maps to ``WanImageToVideoPipeline`` with
``boundary_ratio`` set and both ``transformer`` and ``transformer_2`` populated.

Wan2.2-I2V-A14B also uses the Wan2.2 ``expand_timesteps`` schedule (per-token
timestep) and the Wan2.2 **mask-mix 16-channel** image conditioning (NOT the
Wan2.1-I2V 36-channel channel-concat scheme):

  * The transformer ``hidden_states`` is 16-channel and built every step as
    ``latent_model_input = (1 - first_frame_mask) * condition
                            + first_frame_mask * latents``
    where ``condition`` is the VAE-encoded image-as-video latent and
    ``first_frame_mask`` is 0 on the first latent frame (carrying the
    conditioning image) and 1 elsewhere.
  * The per-token timestep is
    ``(first_frame_mask[:, ::2, ::2] * t).flatten()`` expanded across batch,
    i.e. one timestep per spatiotemporal token (the conditioning frame is
    driven to ``t=0``).

The pipeline is split into five fixed-shape sub-models:

  * text encoder       : UMT5-XXL, input_ids[1,S] + attention_mask[1,S]
                         -> last_hidden_state[1,S,4096]
  * clip image encoder : CLIPVisionModel (ViT-H/14), pixel_values[1,3,224,224]
                         -> image_embeds[1,257,1280] (penultimate hidden state)
  * transformer (high-noise expert, DiT 14B, in_channels=16). Inputs:
                           hidden_states[1,16,F',H',W']
                           timestep[1, T]              (per-token)
                           encoder_hidden_states[1,S,4096]
                           encoder_hidden_states_image[1,257,1280]
                         -> noise_pred[1,16,F',H',W']
  * transformer_2 (low-noise expert, DiT 14B, in_channels=16). Same I/O shape
    as the high-noise expert.
  * vae decoder        : AutoencoderKLWan.decode, latents[1,16,F',H',W']
                         -> video[1,3,F,H,W]

The DiT self/cross attention (full bidirectional, no mask) is exported as the
CANN ``PromptFlashAttention`` Custom op by monkeypatching diffusers' attention
dispatch (``transformer_wan.dispatch_attention_fn``), so the ~33k-token
spatiotemporal attention does not materialise the full score matrix on Ascend.
The text / CLIP / VAE are standard-op graphs.

All sub-models are loaded in float32 and exported with the legacy exporter
(``torch.onnx.utils.export``) at opset 17; fixed shapes (ascend_oriented
friendly). ``do_constant_folding=False`` to avoid OOM on the long-sequence
graph. The VAE encoder is NOT exported (it runs once on CPU in the inferencer
to encode the conditioning image into ``condition``).
"""

import argparse
import gc
import math
import os

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from transformers import CLIPVisionModel, UMT5EncoderModel

import diffusers.models.transformers.transformer_wan as transformer_wan

_VAE_SCALE_FACTOR_TEMPORAL = 4
_VAE_SCALE_FACTOR_SPATIAL = 8
_LATENT_CHANNELS = 16
_OPSET = 17

# CLIP ViT-H/14 image encoder (Wan2.2 I2V / TI2V / Wan2.1 I2V): 224x224 input,
# 257 output tokens (256 patches + CLS), 1280 hidden dim.
_CLIP_IMAGE_SIZE = 224
_CLIP_NUM_TOKENS = 257
_CLIP_HIDDEN_DIM = 1280

# Wan2.2 I2V A14B uses the mask-mix 16-channel conditioning (expand_timesteps
# path), so the DiT in_channels = 16 (NOT the Wan2.1-I2V 36-channel concat).
_DIT_IN_CHANNELS = _LATENT_CHANNELS


# ---------------------------------------------------------------------------
# Custom CANN operators (exported as ONNX ``Custom`` nodes for Ascend).
# ---------------------------------------------------------------------------


def _as_list_str(items):
    """Convert items to a list of string representations for ONNX attributes."""
    return [str(x) for x in items]


class _CustomPromptFlashAttentionFull(torch.autograd.Function):
    """CANN PromptFlashAttention (full bidirectional, no mask) for ONNX export.

    Wan self-attention, cross-attention and image cross-attention are all full
    (no causal / no padding mask), so the Custom node omits ``atten_mask``
    (sparse_mode=0 + no mask == attend-to-all). Inputs q/k/v are in BNSD layout
    (batch, num_heads, seq, head_dim).
    """

    @staticmethod
    def forward(ctx, query, key, value, num_heads_i, num_kv_heads_i,
                scale_value_f, input_layout_s):
        """Cheap shape-preserving stub used during tracing.

        The exported ONNX uses the :meth:`symbolic` Custom node (the real CANN
        op), NOT this forward, so the returned values are irrelevant -- only the
        output shape matters for tracing downstream layers. Returning ``query``
        (same shape, finite values) avoids materialising the O(seq**2) score
        matrix during the trace of long-sequence attention (e.g. Wan's ~33k
        spatiotemporal tokens).
        """
        del ctx, key, value, num_heads_i, num_kv_heads_i, scale_value_f, input_layout_s
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


def _patch_rmsnorm():
    """Replace ``torch.nn.RMSNorm.forward`` with a standard-op decomposition.

    Wan uses ``torch.nn.RMSNorm`` for q/k-norms; the legacy ONNX exporter has no
    symbolic for ``aten::rms_norm``. The decomposition (fp32 variance -> rsqrt
    -> mul weight) is numerically identical and traces to standard ONNX ops.
    """

    def _forward(self, hidden_states):
        """Decomposed RMSNorm forward: fp32 variance, rsqrt, scale by weight."""
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if getattr(self, "weight", None) is not None:
            hidden_states = hidden_states.to(self.weight.dtype) * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)
        return hidden_states

    torch.nn.RMSNorm.forward = _forward


def _patch_wan_attention():
    """Replace Wan's attention dispatch with the CANN Custom op.

    The Wan processor calls ``dispatch_attention_fn(q, k, v, ...)`` with q/k/v
    in layout (batch, seq, num_heads, head_dim) [BSHD]. We transpose to BNSD,
    run the Custom op, and transpose back. q/k/v projections, RMSNorm
    (norm_q/norm_k/norm_added_k) and RoPE stay as the original (standard)
    diffusers code. This covers both self-attention and the (image) cross
    attention paths. Both experts share the same patched dispatch.
    """

    def _custom_dispatch(query, key, value, attn_mask=None, dropout_p=0.0,
                         is_causal=False, scale=None, enable_gqa=False,
                         attention_kwargs=None, *, backend=None,
                         parallel_config=None):
        """Dispatch q/k/v (BSHD) to the CANN PromptFlashAttention Custom op."""
        del attn_mask, dropout_p, is_causal, enable_gqa, attention_kwargs
        del backend, parallel_config
        # value layout: (batch, seq, num_heads, head_dim)
        num_heads = int(value.shape[-2])
        head_dim = int(value.shape[-1])
        scale_val = float(scale) if scale is not None else float(1.0 / math.sqrt(head_dim))
        q = query.transpose(1, 2)  # BNSD
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = _CustomPromptFlashAttentionFull.apply(
            q, k, v, num_heads, num_heads, scale_val, "BNSD")
        return out.transpose(1, 2)  # back to BSHD

    transformer_wan.dispatch_attention_fn = _custom_dispatch


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _clear_cache():
    """Release torch caches and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _latent_shape(num_frames, height, width):
    """Compute the fixed latent shape (B, C, F', H', W') for the given video size."""
    num_latent_frames = (num_frames - 1) // _VAE_SCALE_FACTOR_TEMPORAL + 1
    latent_h = height // _VAE_SCALE_FACTOR_SPATIAL
    latent_w = width // _VAE_SCALE_FACTOR_SPATIAL
    return (1, _LATENT_CHANNELS, num_latent_frames, latent_h, latent_w)


def _per_token_timestep_len(num_frames, height, width):
    """Number of per-token timesteps for the Wan2.2 ``expand_timesteps`` path.

    The pipeline computes ``temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t)
    .flatten()``. ``first_frame_mask`` has latent shape
    ``[1, 1, F', H', W']``; the ``[:, ::2, ::2]`` strides are over the latent
    spatial dims, giving a per-token length of ``F' * (H' // 2) * (W' // 2)``.
    With patch_size (1,2,2) this equals the DiT token sequence length.
    """
    num_latent_frames = (num_frames - 1) // _VAE_SCALE_FACTOR_TEMPORAL + 1
    latent_h = height // _VAE_SCALE_FACTOR_SPATIAL
    latent_w = width // _VAE_SCALE_FACTOR_SPATIAL
    return num_latent_frames * (latent_h // 2) * (latent_w // 2)


def _export_onnx(wrapper, dummy_inputs, input_names, output_names, out_path,
                 dynamic_axes=None):
    """Trace a wrapper to ONNX with the legacy exporter (opset 17, float32)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.onnx.utils.export(
        wrapper,
        tuple(dummy_inputs),
        out_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=_OPSET,
        # Disable constant folding: with it on, the tracer materialises huge
        # folded constants for long-sequence graphs (Wan's ~33k spatiotemporal
        # tokens) and OOMs on CPU. The converter does its own ascend_oriented
        # optimisation, so export-time folding is unnecessary.
        do_constant_folding=False,
    )
    _clear_cache()
    print(f"[export] saved {out_path}")


# ---------------------------------------------------------------------------
# Sub-module wrappers.
# ---------------------------------------------------------------------------


class _TextEncoderWrapper(nn.Module):
    """Wrap UMT5EncoderModel to expose input_ids/attention_mask -> last_hidden_state."""

    def __init__(self, encoder):
        """Store the UMT5 encoder."""
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        """Return the encoder last hidden state [1, seq_len, 4096]."""
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class _ClipImageEncoderWrapper(nn.Module):
    """Wrap CLIPVisionModel to expose pixel_values -> penultimate hidden state.

    Wan I2V/TI2V consumes the penultimate hidden state (``hidden_states[-2]``)
    of the CLIP ViT image encoder as ``encoder_hidden_states_image``.
    """

    def __init__(self, encoder):
        """Store the CLIP vision encoder."""
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values):
        """Return the CLIP penultimate hidden state [1, 257, 1280]."""
        out = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
        return out.hidden_states[-2]


class _TransformerWrapper(nn.Module):
    """Wrap WanTransformer3DModel to expose hidden_states/timestep/embeds -> noise_pred.

    Used for BOTH the high-noise (``transformer``) and low-noise
    (``transformer_2``) experts; their graph shape is identical.
    """

    def __init__(self, model):
        """Store the Wan DiT expert model."""
        super().__init__()
        self.model = model

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_hidden_states_image):
        """Return the predicted noise [1, 16, F', H', W'].

        ``hidden_states`` is the 16-channel mask-mix tensor
        ``(1-mask)*condition + mask*latents`` built by the pipeline (NOT the
        bare latent); ``timestep`` is the per-token timestep tensor [1, T]
        (Wan2.2 ``expand_timesteps``); ``encoder_hidden_states_image`` is the
        CLIP image embedding [1, 257, 1280].
        """
        return self.model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=False,
        )[0]


class _VaeWrapper(nn.Module):
    """Wrap AutoencoderKLWan.decode to expose latents -> video."""

    def __init__(self, vae):
        """Store the Wan VAE decoder."""
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        """Return the decoded video [1, 3, F, H, W] (latents already denormalised)."""
        return self.vae.decode(latents, return_dict=False)[0]


# ---------------------------------------------------------------------------
# Export entry points.
# ---------------------------------------------------------------------------


def export_text_encoder(model_dir, output_dir, max_seq_len, dtype):
    """Export the UMT5-XXL text encoder to ONNX."""
    encoder = UMT5EncoderModel.from_pretrained(
        os.path.join(model_dir, "text_encoder"), torch_dtype=dtype
    ).eval()
    wrapper = _TextEncoderWrapper(encoder)
    input_ids = torch.zeros((1, max_seq_len), dtype=torch.int64)
    attention_mask = torch.ones((1, max_seq_len), dtype=torch.int64)
    out_path = os.path.join(output_dir, "wan_text_encoder.onnx")
    _export_onnx(
        wrapper, (input_ids, attention_mask),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"], out_path=out_path,
    )
    del encoder, wrapper
    _clear_cache()


def export_clip_image_encoder(model_dir, output_dir, dtype):
    """Export the CLIP ViT-H/14 image encoder to ONNX."""
    encoder = CLIPVisionModel.from_pretrained(
        os.path.join(model_dir, "image_encoder"), torch_dtype=dtype
    ).eval()
    wrapper = _ClipImageEncoderWrapper(encoder)
    pixel_values = torch.zeros((1, 3, _CLIP_IMAGE_SIZE, _CLIP_IMAGE_SIZE), dtype=dtype)
    out_path = os.path.join(output_dir, "wan_clip_image_encoder.onnx")
    _export_onnx(
        wrapper, (pixel_values,),
        input_names=["pixel_values"],
        output_names=["image_embeds"], out_path=out_path,
    )
    del encoder, wrapper
    _clear_cache()


def _export_one_expert(model_dir, output_dir, height, width, num_frames,
                       max_seq_len, dtype, use_custom_op, subfolder, out_name):
    """Export a single Wan DiT expert (dense) to ONNX.

    ``subfolder`` selects which expert to load from the model dir
    (``transformer`` = high-noise, ``transformer_2`` = low-noise). Both experts
    are dense ``WanTransformer3DModel`` instances with identical I/O shapes; the
    MoE switching is done in the scheduler loop, not in the graph.
    """
    model = WanTransformer3DModel.from_pretrained(
        os.path.join(model_dir, subfolder), torch_dtype=dtype
    ).eval()
    text_dim = model.config.text_dim
    image_dim = model.config.image_dim
    latent_shape = _latent_shape(num_frames, height, width)
    ts_len = _per_token_timestep_len(num_frames, height, width)
    wrapper = _TransformerWrapper(model)
    hidden_states = torch.zeros(latent_shape, dtype=dtype)
    # Wan2.2 expand_timesteps: per-token timesteps of shape [batch, seq_len].
    timestep = torch.zeros((1, ts_len), dtype=dtype)
    encoder_hidden_states = torch.zeros((1, max_seq_len, text_dim), dtype=dtype)
    encoder_hidden_states_image = torch.zeros(
        (1, _CLIP_NUM_TOKENS, image_dim if image_dim is not None else _CLIP_HIDDEN_DIM),
        dtype=dtype,
    )
    out_path = os.path.join(output_dir, out_name)
    _export_onnx(
        wrapper,
        (hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image),
        input_names=["hidden_states", "timestep", "encoder_hidden_states",
                     "encoder_hidden_states_image"],
        output_names=["noise_pred"], out_path=out_path,
    )
    del model, wrapper
    _clear_cache()


def export_transformer(model_dir, output_dir, height, width, num_frames,
                       max_seq_len, dtype, use_custom_op):
    """Export BOTH Wan DiT experts (high-noise + low-noise) to ONNX.

    The Wan2.2-A14B MoE = two dense experts; the attention dispatch is patched
    once (shared by both experts) to emit the CANN PromptFlashAttention Custom
    op. Both experts share identical I/O shapes (16-ch mask-mix input,
    per-token timestep). Memory is freed between the two exports so a single
    host can export both ~14B experts sequentially.
    """
    if use_custom_op:
        _patch_rmsnorm()
        _patch_wan_attention()
    print("[export] Wan DiT expert 1/2 (high-noise, transformer/) ...")
    _export_one_expert(model_dir, output_dir, height, width, num_frames,
                       max_seq_len, dtype, use_custom_op, "transformer",
                       "wan_transformer_high_noise.onnx")
    print("[export] Wan DiT expert 2/2 (low-noise, transformer_2/) ...")
    _export_one_expert(model_dir, output_dir, height, width, num_frames,
                       max_seq_len, dtype, use_custom_op, "transformer_2",
                       "wan_transformer_low_noise.onnx")


def export_vae(model_dir, output_dir, height, width, num_frames, dtype):
    """Export the Wan VAE decoder to ONNX."""
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_dir, "vae"), torch_dtype=dtype
    ).eval()
    latent_shape = _latent_shape(num_frames, height, width)
    wrapper = _VaeWrapper(vae)
    latents = torch.zeros(latent_shape, dtype=dtype)
    out_path = os.path.join(output_dir, "wan_vae_decoder.onnx")
    _export_onnx(
        wrapper, (latents,),
        input_names=["latents"], output_names=["video"], out_path=out_path,
    )
    del vae, wrapper
    _clear_cache()


def _dtype_of(name):
    """Map a dtype name string to a torch dtype."""
    return {"float32": torch.float32, "float16": torch.float16}[name]


def main():
    """Parse arguments and export the requested Wan2.2-I2V-A14B sub-modules."""
    parser = argparse.ArgumentParser(description="Export Wan2.2-I2V-A14B submodules to ONNX")
    parser.add_argument("--model-dir", required=True, help="Wan2.2-I2V-A14B-Diffusers weight dir")
    parser.add_argument("--output-dir", default="./wan2_2_i2v_a14b_onnx", help="ONNX output dir")
    parser.add_argument("--parts", default="text,clip,transformer,vae",
                        help="comma list: text,clip,transformer,vae")
    parser.add_argument("--height", type=int, default=480, help="video height (multiple of 16)")
    parser.add_argument("--width", type=int, default=832, help="video width (multiple of 16)")
    parser.add_argument("--num-frames", type=int, default=81, help="number of video frames")
    parser.add_argument("--max-seq-len", type=int, default=512, help="UMT5 max sequence length")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                        help="export dtype (float32 recommended for converter compatibility)")
    parser.add_argument("--no-custom-op", action="store_true",
                        help="Do NOT replace attention with the CANN PromptFlashAttention op.")
    args = parser.parse_args()

    if args.height % 16 or args.width % 16:
        raise ValueError("height/width must be multiples of 16")
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = _dtype_of(args.dtype)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]

    if "text" in parts:
        print("[export] UMT5 text encoder ...")
        export_text_encoder(args.model_dir, args.output_dir, args.max_seq_len, dtype)
    if "clip" in parts:
        print("[export] CLIP image encoder ...")
        export_clip_image_encoder(args.model_dir, args.output_dir, dtype)
    if "transformer" in parts:
        print("[export] Wan DiT MoE (high-noise + low-noise experts) ...")
        export_transformer(args.model_dir, args.output_dir, args.height, args.width,
                           args.num_frames, args.max_seq_len, dtype, not args.no_custom_op)
    if "vae" in parts:
        print("[export] Wan VAE decoder ...")
        export_vae(args.model_dir, args.output_dir, args.height, args.width,
                   args.num_frames, dtype)

    print("[export] done ->", args.output_dir)


if __name__ == "__main__":
    main()
