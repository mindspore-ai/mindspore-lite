#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Export kandinskylab/Kandinsky-5.0-I2V-Lite-5s to ONNX for MindSpore Lite Ascend.

Kandinsky-5.0-I2V-Lite-5s is the image-to-video variant of the ~6B flow-matching
DiT. Its upstream layout (verified against the kandinsky-5 repo + diffusers
``Kandinsky5I2VPipeline``) is identical to the T2I Lite sibling except for the
I2I conditioning mechanism:

  * text encoder  : Qwen2.5-VL-7B-Instruct (NOT MT5/T5). The I2I pipeline wraps
                    the prompt + a downscaled copy of the source image in a
                    chat template whose vision placeholder ``<|image_pad|>``
                    lets Qwen read the source image. The hidden state is sliced
                    at ``[:, 55:]`` (55 template tokens, vs 41 for T2I).
  * pooled encoder: CLIP text (openai/clip-vit-large-patch14) -> pooler_output.
                    NOTE: for I2I the CLIP prompt is the plain text prompt
                    (the source image does NOT go through CLIP).
  * transformer   : ``Kandinsky5Transformer3DModel`` with ``visual_cond=True``.
                    Its patchify input dim is ``2*in_visual_dim + 1 = 33``
                    (noise + image latent + mask), all concatenated along the
                    channel axis. The transformer operates in channels-LAST
                    layout ``[B, T=1, H', W', 33]`` and returns the noise
                    prediction ``[B, T=1, H', W', 16]``.
  * image decoder : the **FLUX.1-dev AutoencoderKL** (8x spatial, 16 latent
                    channels). The source image is VAE-ENCODED on CPU (torch)
                    in the inference script; only the VAE DECODER is exported
                    here (same as T2I).

I2I conditioning mechanism (img2img + channel concat, NOT partial-noise SDE):
the latents tensor is built as
``torch.cat([noise(16), image_latents(16), mask(1)], dim=-1)`` -> ``[..., 33]``.
Only the first 16 channels are updated by the scheduler each step; the image
and mask channels are carried through unchanged and re-fed every step (the
transformer sees them at inference but does not modify them).

The pipeline is exported as four ONNX sub-graphs so the heavy compute runs as
MindIR on Ascend. Attention in the transformer is full bidirectional (no causal
/ no padding mask for I2I either), so it is exported as the CANN
``PromptFlashAttention`` Custom op by monkeypatching diffusers'
``dispatch_attention_fn``. RoPE / RMSNorm / AdaLN / projections stay as the
original diffusers ops.

All sub-models are loaded in float32 and exported with the legacy exporter
(``torch.onnx.utils.export``) at opset 17, fixed shapes
(ascend_oriented friendly).
"""

import argparse
import gc
import math
import os

import torch
import torch.nn as nn

try:
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel
except ImportError as exc:  # pragma: no cover
    print(f"Error: missing dependency: {exc}")
    print("Install: pip install torch diffusers transformers onnx")
    raise SystemExit(1)

# K5 transformer + Qwen2.5-VL live in diffusers/transformers main. Import lazily
# inside the export routines so a missing nightly diffusers does not break the
# VAE/CLIP exports.

_OPSET = 17
_LATENT_CHANNELS = 16  # base VAE latent channels (in_visual_dim of the DiT)
_VAE_SCALE_SPATIAL = 8
_PATCH_H = 2  # K5 patchify (T=1, H=2, W=2)
_PATCH_W = 2
# Number of template tokens the K5 *I2I* pipeline drops from the Qwen hidden
# state. The I2I chat template embeds the source image and is longer than the
# T2I template (55 vs 41). See pipeline_kandinsky_i2i.py prompt_template.
_PROMPT_TEMPLATE_ENCODE_START_IDX = 55


# ---------------------------------------------------------------------------
# Custom CANN operators (exported as ONNX ``Custom`` nodes for Ascend).
# ---------------------------------------------------------------------------


def _as_list_str(items):
    """Convert items to a list of string representations for ONNX attributes."""
    return [str(x) for x in items]


class _CustomPromptFlashAttentionFull(torch.autograd.Function):
    """CANN PromptFlashAttention (full bidirectional, no mask) for ONNX export.

    K5 I2I self-attention (visual + text) and cross-attention (visual attends to
    text) are both full (no causal / no padding mask), so the Custom node omits
    ``atten_mask`` (sparse_mode=0 + no mask == attend-to-all). Inputs q/k/v are
    in BNSD layout (batch, num_heads, seq, head_dim).
    """

    @staticmethod
    def forward(ctx, query, key, value, num_heads_i, num_kv_heads_i,
                scale_value_f, input_layout_s):
        """Cheap shape-preserving stub used during tracing.

        The exported ONNX uses :meth:`symbolic` (the real CANN op), NOT this
        forward, so the returned values are irrelevant -- only the output shape
        matters for tracing downstream layers. Returning ``query`` (same shape,
        finite values) avoids materialising the O(seq**2) score matrix during
        the trace of the 4096-token visual self-attention.
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


def _patch_rmsnorm():
    """Replace ``torch.nn.RMSNorm.forward`` with a standard-op decomposition.

    K5 uses ``torch.nn.RMSNorm`` for q/k-norms; the legacy ONNX exporter has no
    symbolic for ``aten::rms_norm``. The decomposition (fp32 variance -> rsqrt
    -> mul weight) is numerically identical and traces to standard ONNX ops.
    """

    def _forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if getattr(self, "weight", None) is not None:
            hidden_states = hidden_states.to(self.weight.dtype) * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)
        return hidden_states

    torch.nn.RMSNorm.forward = _forward


def _patch_kandinsky_attention():
    """Replace the K5 attention dispatch with the CANN Custom op.

    The K5 ``Kandinsky5AttnProcessor`` calls ``dispatch_attention_fn(q, k, v,
    ...)`` with q/k/v in layout (batch, seq, num_heads, head_dim) [BSHD]
    (same dispatcher helper as FLUX/Wan). We transpose to BNSD, run the Custom
    op, and transpose back. q/k/v projections, RMSNorm (norm_q/norm_k) and the
    3D RoPE application stay as the original (standard) diffusers code.
    """
    import diffusers.models.transformers.transformer_kandinsky as k5

    def _custom_dispatch(query, key, value, attn_mask=None, dropout_p=0.0,
                         is_causal=False, scale=None, enable_gqa=False,
                         attention_kwargs=None, *, backend=None,
                         parallel_config=None):
        del attn_mask, dropout_p, is_causal, enable_gqa, attention_kwargs
        del backend, parallel_config
        # value layout: (batch, seq, num_heads, head_dim)
        num_heads = int(value.shape[-2])
        head_dim = int(value.shape[-1])
        scale_val = float(scale) if scale is not None else float(
            1.0 / math.sqrt(head_dim))
        q = query.transpose(1, 2)  # BNSD
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = _CustomPromptFlashAttentionFull.apply(
            q, k, v, num_heads, num_heads, scale_val, "BNSD")
        return out.transpose(1, 2)  # back to BSHD

    k5.dispatch_attention_fn = _custom_dispatch


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _clear_cache():
    """Release torch caches and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _latent_shape(height, width):
    """Fixed spatial latent shape for a 2D image of (height, width).

    Returns (H', W') = (height // 8, width // 8). The full I2I latent tensor
    is channels-last [B, 1, H', W', 33] (built by the inference script).
    """
    return (height // _VAE_SCALE_SPATIAL, width // _VAE_SCALE_SPATIAL)


def _export_onnx(wrapper, dummy_inputs, input_names, output_names, out_path):
    """Trace a wrapper to ONNX with the legacy exporter (opset 17, float32)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.onnx.utils.export(
        wrapper,
        tuple(dummy_inputs),
        out_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=_OPSET,
        # Disable constant folding: with it on, the tracer materialises huge
        # folded constants for the 4096-token visual attention graph and OOMs on
        # CPU. The converter does its own ascend_oriented optimisation, so
        # export-time folding is unnecessary.
        do_constant_folding=False,
    )
    _clear_cache()
    print(f"[export] saved {out_path}")


# ---------------------------------------------------------------------------
# Sub-module wrappers.
# ---------------------------------------------------------------------------


class _QwenEncoderWrapper(nn.Module):
    """Wrap Qwen2.5-VL to expose input_ids/attention_mask -> last_hidden_state.

    The K5 I2I pipeline feeds the DiT the Qwen hidden state sliced at index
    ``_PROMPT_TEMPLATE_ENCODE_START_IDX`` (drops the I2I system-template prefix
    which now also carries the source image tokens). We do that slice here so
    the exported graph emits exactly the tokens the DiT consumes
    (sequence length = max_seq_len - 55).

    NOTE: the source image is encoded into the Qwen sequence by the VL
    tokenizer at *inference* time (it expands the ``<|image_pad|>`` placeholder
    into image tokens). The exported graph itself only consumes the resulting
    ``input_ids`` / ``attention_mask`` -- it is image-agnostic.
    """

    def __init__(self, encoder, start_idx):
        super().__init__()
        self.encoder = encoder
        self.start_idx = int(start_idx)

    def forward(self, input_ids, attention_mask):
        """Return the sliced last hidden state [1, seq_len-55, 3584]."""
        out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[-1]
        return hidden[:, self.start_idx:]


class _ClipPoolWrapper(nn.Module):
    """Wrap CLIPTextModel to expose input_ids/attention_mask -> pooler_output.

    For I2I the CLIP branch sees the PLAIN text prompt (the source image does
    NOT pass through CLIP -- only Qwen reads the image). pooler_output is added
    to the DiT timestep embedding.
    """

    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, input_ids, attention_mask):
        """Return the CLIP pooled embedding (added to the DiT timestep embed)."""
        return self.clip(
            input_ids=input_ids, attention_mask=attention_mask,
            return_dict=True).pooler_output


class _I2ITransformerWrapper(nn.Module):
    """Wrap Kandinsky5Transformer3DModel for I2I (channels-last, 33-ch input).

    The K5 I2I pipeline builds the visual input by concatenating along the
    channel axis: ``cat([noise(16), image_latents(16), mask(1)], dim=-1)`` ->
    ``[B, T=1, H', W', 33]``. To keep the ONNX input list explicit and avoid a
    python-side ``torch.cat`` in the graph, this wrapper takes the three
    channel groups as SEPARATE inputs and concatenates them ON-DEVICE before
    calling the DiT. The DiT output is the noise prediction
    ``[B, T=1, H', W', 16]`` (channels-last).

    ``visual_rope_pos`` is a 3-tuple of 1D LongTensors (T, H, W arange) and
    ``text_rope_pos`` is a 1D LongTensor; ONNX cannot carry python tuples as
    inputs, so they are taken as separate flat inputs and re-packed.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, noise, image_latents, mask, encoder_hidden_states,
                timestep, pooled_projections, visual_rope_h, visual_rope_w,
                text_rope):
        """Return the predicted velocity [1, 1, H', W', 16] (channels-last)."""
        # Concat along channel axis (last). noise/image are [B,1,H,W,16], mask
        # is [B,1,H,W,1] -> hidden_states [B,1,H,W,33].
        hidden_states = torch.cat([noise, image_latents, mask], dim=-1)
        # T-axis has length 1 for I2I -> arange(1) is a constant; reconstruct it
        # on-device so the graph has no python-side dependency on batch.
        t_pos = torch.arange(1, dtype=visual_rope_h.dtype,
                             device=visual_rope_h.device)
        visual_rope_pos = (t_pos, visual_rope_h, visual_rope_w)
        return self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            pooled_projections=pooled_projections,
            visual_rope_pos=visual_rope_pos,
            text_rope_pos=text_rope,
            return_dict=False,
        )[0]


class _VaeWrapper(nn.Module):
    """Wrap AutoencoderKL.decode to expose latents -> image.

    The DiT predicts latents in channels-last ``[B, 1, H', W', 16]``; before
    VAE decode the inference script permutes to NCHW ``[B, 16, H', W']`` and
    drops the T=1 axis, so this wrapper takes plain NCHW (same as T2I).
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        """Return the decoded image [1, 3, H, W] (latents already denormalised)."""
        return self.vae.decode(latents, return_dict=False)[0]


# ---------------------------------------------------------------------------
# Export entry points.
# ---------------------------------------------------------------------------


def export_text_encoder(qwen_dir, output_dir, max_seq_len, dtype):
    """Export the Qwen2.5-VL text encoder (last_hidden_state, sliced) to ONNX.

    The I2I prompt template is longer than T2I (it carries the source image),
    so a larger ``max_seq_len`` is recommended (e.g. 768). The graph slices
    ``[:, 55:]`` internally.
    """
    from transformers import AutoModelForCausalLM
    print(f"[export] loading Qwen2.5-VL text encoder from {qwen_dir} ({dtype}) ...")
    # K5 uses Qwen2.5-VL as a text encoder; load the causal LM and read
    # ``hidden_states[-1]`` (the transformer stack output, before the LM head).
    encoder = AutoModelForCausalLM.from_pretrained(
        qwen_dir, torch_dtype=dtype, low_cpu_mem_usage=True).eval()
    wrapper = _QwenEncoderWrapper(
        encoder, _PROMPT_TEMPLATE_ENCODE_START_IDX).eval()
    seq_after_slice = max_seq_len - _PROMPT_TEMPLATE_ENCODE_START_IDX
    input_ids = torch.zeros((1, max_seq_len), dtype=torch.int64)
    attention_mask = torch.ones((1, max_seq_len), dtype=torch.int64)
    out_path = os.path.join(output_dir, "kandinsky_text_encoder.onnx")
    print(f"[export] Qwen seq_in={max_seq_len} -> seq_out={seq_after_slice}")
    _export_onnx(
        wrapper, (input_ids, attention_mask),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"], out_path=out_path,
    )
    del encoder, wrapper
    _clear_cache()


def export_clip_encoder(clip_dir, output_dir, dtype):
    """Export the CLIP text encoder (pooler_output) to ONNX.

    CLIP sees the PLAIN text prompt for I2I (no source image through CLIP).
    """
    print(f"[export] loading CLIP text encoder from {clip_dir} ({dtype}) ...")
    clip = CLIPTextModel.from_pretrained(
        clip_dir, torch_dtype=dtype, low_cpu_mem_usage=True).eval()
    wrapper = _ClipPoolWrapper(clip).eval()
    max_len = 77
    input_ids = torch.zeros((1, max_len), dtype=torch.int64)
    attention_mask = torch.ones((1, max_len), dtype=torch.int64)
    out_path = os.path.join(output_dir, "kandinsky_clip_encoder.onnx")
    _export_onnx(
        wrapper, (input_ids, attention_mask),
        input_names=["input_ids", "attention_mask"],
        output_names=["pooled_embeds"], out_path=out_path,
    )
    del clip, wrapper
    _clear_cache()


def export_transformer(model_path, output_dir, height, width, max_seq_len,
                       dtype, use_custom_op):
    """Export the K5 I2I DiT transformer to ONNX (attention -> CANN Custom op).

    Inputs (channels-last, I2I): noise[1,1,H',W',16], image_latents[1,1,H',W',
    16], mask[1,1,H',W',1] (concatenated to 33 ch on-device), plus the standard
    text / timestep / RoPE inputs. Output: noise_pred[1,1,H',W',16].
    """
    from diffusers import Kandinsky5Transformer3DModel
    if use_custom_op:
        _patch_rmsnorm()
        _patch_kandinsky_attention()
    print(f"[export] loading Kandinsky5Transformer3DModel (I2I) from {model_path} ...")
    # K5 I2I ships as a diffusers dir (Kandinsky-5.0-I2V-Lite-5s-sft-Diffusers) OR
    # a single-file checkpoint; from_pretrained / from_single_file both work and
    # read ``visual_cond=True`` + ``in_visual_dim=16`` from the config.
    if os.path.isdir(model_path) and os.path.exists(
            os.path.join(model_path, "config.json")):
        model = Kandinsky5Transformer3DModel.from_pretrained(
            model_path, torch_dtype=dtype).eval()
    else:
        model = Kandinsky5Transformer3DModel.from_single_file(
            model_path, torch_dtype=dtype).eval()
    if not bool(model.config.visual_cond):
        raise ValueError(
            "Kandinsky5Transformer3DModel loaded from %s has visual_cond=False "
            "(T2I variant). I2I-Lite requires visual_cond=True (in_visual_dim=16"
            " -> patchify dim 33)." % model_path)
    wrapper = _I2ITransformerWrapper(model).eval()

    latent_h, latent_w = _latent_shape(height, width)
    seq_after_slice = max_seq_len - _PROMPT_TEMPLATE_ENCODE_START_IDX
    text_dim = int(model.config.in_text_dim)  # 3584
    pooled_dim = int(model.config.in_text_dim2)  # 768
    in_visual_dim = int(model.config.in_visual_dim)  # 16 (base latent channels)

    # Channels-last I2I latents.
    noise = torch.zeros((1, 1, latent_h, latent_w, in_visual_dim), dtype=dtype)
    image_latents = torch.zeros(
        (1, 1, latent_h, latent_w, in_visual_dim), dtype=dtype)
    mask = torch.ones((1, 1, latent_h, latent_w, 1), dtype=dtype)
    encoder_hidden_states = torch.zeros(
        (1, seq_after_slice, text_dim), dtype=dtype)
    timestep = torch.tensor([950.0], dtype=dtype)
    pooled_projections = torch.zeros((1, pooled_dim), dtype=dtype)
    # RoPE positions are integer arange over the post-patchify grid (latent_h/2
    # x latent_w/2 for visual; seq_after_slice for text). dtype int64.
    visual_rope_h = torch.arange(latent_h // _PATCH_H, dtype=torch.int64)
    visual_rope_w = torch.arange(latent_w // _PATCH_W, dtype=torch.int64)
    text_rope = torch.arange(seq_after_slice, dtype=torch.int64)

    out_path = os.path.join(output_dir, "kandinsky_transformer.onnx")
    print(f"[export] latent spatial ({latent_h},{latent_w}) visual_rope "
          f"({visual_rope_h.shape[0]}, {visual_rope_w.shape[0]}) "
          f"text_rope {text_rope.shape[0]} in_visual_dim={in_visual_dim}")
    _export_onnx(
        wrapper,
        (noise, image_latents, mask, encoder_hidden_states, timestep,
         pooled_projections, visual_rope_h, visual_rope_w, text_rope),
        input_names=["noise", "image_latents", "mask",
                     "encoder_hidden_states", "timestep", "pooled_projections",
                     "visual_rope_h", "visual_rope_w", "text_rope"],
        output_names=["noise_pred"], out_path=out_path,
    )
    del model, wrapper
    _clear_cache()


def export_vae(vae_dir, output_dir, height, width, dtype):
    """Export the FLUX.1-dev VAE decoder (AutoencoderKL.decode) to ONNX.

    Identical to the T2I export: NCHW ``[1, 16, H', W']`` -> ``[1, 3, H, W]``.
    The VAE *encoder* (used to encode the source image) is NOT exported here;
    it runs on CPU torch in the inference script.
    """
    print(f"[export] loading AutoencoderKL (FLUX VAE) from {vae_dir} ({dtype}) ...")
    vae = AutoencoderKL.from_pretrained(
        vae_dir, torch_dtype=dtype, low_cpu_mem_usage=True).eval()
    wrapper = _VaeWrapper(vae).eval()
    latent_h, latent_w = _latent_shape(height, width)
    latents = torch.zeros((1, _LATENT_CHANNELS, latent_h, latent_w), dtype=dtype)
    out_path = os.path.join(output_dir, "kandinsky_dcae_decoder.onnx")
    _export_onnx(
        wrapper, (latents,),
        input_names=["latents"], output_names=["image"], out_path=out_path,
    )
    del vae, wrapper
    _clear_cache()


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _dtype_of(name):
    """Map a dtype name string to a torch dtype."""
    return {"float32": torch.float32, "float16": torch.float16}[name]


def _parse_args():
    """Parse command-line arguments for the K5 I2I Lite ONNX export."""
    parser = argparse.ArgumentParser(
        description="Export Kandinsky-5.0-I2V-Lite-5s submodules to ONNX")
    parser.add_argument(
        "--k5-model", required=True,
        help="Path to the K5 I2I Lite checkpoint: a diffusers dir "
             "(Kandinsky-5.0-I2V-Lite-5s-sft-Diffusers/transformer) OR a "
             "single-file safetensors checkpoint.")
    parser.add_argument(
        "--qwen-dir", required=True,
        help="Qwen2.5-VL-7B-Instruct weights dir (text encoder).")
    parser.add_argument(
        "--clip-dir", required=True,
        help="openai/clip-vit-large-patch14 weights dir (pooled encoder).")
    parser.add_argument(
        "--vae-dir", required=True,
        help="FLUX.1-dev vae/ dir (image decoder + source-image encoder). "
             "K5 reuses the FLUX VAE.")
    parser.add_argument(
        "--output-dir", default="./kandinsky5_i2v_lite_5s_onnx",
        help="ONNX output directory.")
    parser.add_argument(
        "--parts", default="text,clip,transformer,vae",
        help="Comma list: text,clip,transformer,vae.")
    parser.add_argument(
        "--height", type=int, default=1024, help="image height (multiple of 16).")
    parser.add_argument(
        "--width", type=int, default=1024, help="image width (multiple of 16).")
    parser.add_argument(
        "--max-seq-len", type=int, default=768,
        help="Qwen max sequence length (< 1024; 55 I2I template tokens dropped). "
             "I2I prompts are longer than T2I because of the image tokens.")
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float16"],
        help="export dtype (float32 recommended for converter compatibility).")
    parser.add_argument(
        "--no-custom-op", action="store_true",
        help="Do NOT replace attention with the CANN PromptFlashAttention op.")
    return parser.parse_args()


def main():
    """Parse arguments and export the requested K5 I2I Lite sub-modules."""
    args = _parse_args()
    if args.height % 16 or args.width % 16:
        raise ValueError("height/width must be multiples of 16")
    if args.max_seq_len <= _PROMPT_TEMPLATE_ENCODE_START_IDX:
        raise ValueError(
            f"max_seq_len must be > {_PROMPT_TEMPLATE_ENCODE_START_IDX}")
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = _dtype_of(args.dtype)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]

    print(f"K5 I2I Lite ONNX export: {args.height}x{args.width}, "
          f"max_seq={args.max_seq_len}, dtype={args.dtype}, "
          f"custom_op={not args.no_custom_op}")

    if "text" in parts:
        export_text_encoder(args.qwen_dir, args.output_dir,
                            args.max_seq_len, dtype)
    if "clip" in parts:
        export_clip_encoder(args.clip_dir, args.output_dir, dtype)
    if "transformer" in parts:
        export_transformer(args.k5_model, args.output_dir, args.height,
                           args.width, args.max_seq_len, dtype,
                           not args.no_custom_op)
    if "vae" in parts:
        export_vae(args.vae_dir, args.output_dir, args.height, args.width, dtype)

    print(f"[export] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
