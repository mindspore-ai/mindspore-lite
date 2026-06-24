"""Export Wan2.1-T2V-1.3B-Diffusers submodules to ONNX for MindSpore Lite cloud-side inference.

The pipeline is split into three fixed-shape sub-models (mirrors the flux1_dev
example, which is the closest reference in this repo):

  * text encoder  : UMT5-XXL, input_ids[1,512] + attention_mask[1,512]
                    -> last_hidden_state[1,512,4096]
  * transformer   : WanTransformer3DModel (DiT, 1.3B), hidden_states[1,16,F',H',W'] +
                    timestep[1] + encoder_hidden_states[1,512,4096] -> noise_pred
  * vae decoder   : AutoencoderKLWan.decode, latents[1,16,F',H',W'] -> video[1,3,F,H,W]

The transformer attention (full bidirectional, no mask) is exported as the CANN
``PromptFlashAttention`` Custom op by monkeypatching diffusers' attention dispatch
(``transformer_wan.dispatch_attention_fn``), so the 32k-token spatiotemporal
attention does not materialise the full score matrix on Ascend. The T5 / VAE are
standard-op graphs and run in ONNX Runtime.

All sub-models are loaded in float32 and exported with the legacy exporter
(``torch.onnx.utils.export``) at opset 17; fixed shapes (ascend_oriented friendly).
"""

import argparse
import gc
import math
import os

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from transformers import UMT5EncoderModel

import diffusers.models.transformers.transformer_wan as transformer_wan

_VAE_SCALE_FACTOR_TEMPORAL = 4
_VAE_SCALE_FACTOR_SPATIAL = 8
_LATENT_CHANNELS = 16
_OPSET = 17


# ---------------------------------------------------------------------------
# Custom CANN operators (exported as ONNX ``Custom`` nodes for Ascend).
# ---------------------------------------------------------------------------


def _as_list_str(items):
    """Convert items to a list of string representations for ONNX attributes."""
    return [str(x) for x in items]


class _CustomPromptFlashAttentionFull(torch.autograd.Function):
    """CANN PromptFlashAttention (full bidirectional, no mask) for ONNX export.

    Wan self-attention and cross-attention are both full (no causal / no padding
    mask), so the Custom node omits ``atten_mask`` (sparse_mode=0 + no mask ==
    attend-to-all). Inputs q/k/v are in BNSD layout (batch, num_heads, seq,
    head_dim).
    """

    @staticmethod
    def forward(ctx, query, key, value, num_heads_i, num_kv_heads_i,
                scale_value_f, input_layout_s):
        """Cheap shape-preserving stub used during tracing.

        The exported ONNX uses the :meth:`symbolic` Custom node (the real CANN
        op), NOT this forward, so the returned values are irrelevant -- only the
        output shape matters for tracing downstream layers. Returning ``query``
        (same shape, finite values) avoids materialising the O(seq**2) score
        matrix during the trace of long-sequence attention (e.g. Wan's 32k
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
    symbolic for ``aten::rms_norm``. The decomposition (fp32 variance -> rsqrt ->
    mul weight) is numerically identical and traces to standard ONNX ops.
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


def _patch_wan_attention():
    """Replace Wan's attention dispatch with the CANN Custom op.

    The Wan processor calls ``dispatch_attention_fn(q, k, v, ...)`` with q/k/v in
    layout (batch, seq, num_heads, head_dim) [BSHD]. We transpose to BNSD, run
    the Custom op, and transpose back. q/k/v projections, RMSNorm (norm_q/norm_k)
    and RoPE stay as the original (standard) diffusers code.
    """

    def _custom_dispatch(query, key, value, attn_mask=None, dropout_p=0.0,
                         is_causal=False, scale=None, enable_gqa=False,
                         attention_kwargs=None, *, backend=None,
                         parallel_config=None):
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


def _export_onnx(wrapper, dummy_inputs, input_names, output_names, out_path,
                 dynamic_axes=None, opset=_OPSET):
    """Trace a wrapper to ONNX with the legacy exporter (float32, no folding)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.onnx.utils.export(
        wrapper,
        tuple(dummy_inputs),
        out_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        # Disable constant folding: with it on, the tracer materialises huge
        # folded constants for long-sequence graphs (Wan's 32k spatiotemporal
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
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        """Return the encoder last hidden state [1, seq_len, 4096]."""
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class _TransformerWrapper(nn.Module):
    """Wrap WanTransformer3DModel to expose hidden_states/timestep/embeds -> noise_pred."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        """Return the predicted noise [1, 16, F', H', W']."""
        return self.model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


class _VaeWrapper(nn.Module):
    """Wrap AutoencoderKLWan.decode to expose latents -> video."""

    def __init__(self, vae):
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


def export_transformer(model_dir, output_dir, height, width, num_frames,
                       max_seq_len, dtype, use_custom_op):
    """Export the Wan DiT transformer to ONNX (attention -> CANN Custom op)."""
    if use_custom_op:
        _patch_rmsnorm()
        _patch_wan_attention()
    model = WanTransformer3DModel.from_pretrained(
        os.path.join(model_dir, "transformer"), torch_dtype=dtype
    ).eval()
    text_dim = model.config.text_dim
    latent_shape = _latent_shape(num_frames, height, width)
    wrapper = _TransformerWrapper(model)
    hidden_states = torch.zeros(latent_shape, dtype=dtype)
    timestep = torch.tensor([950.0], dtype=dtype)
    encoder_hidden_states = torch.zeros((1, max_seq_len, text_dim), dtype=dtype)
    out_path = os.path.join(output_dir, "wan_transformer.onnx")
    _export_onnx(
        wrapper, (hidden_states, timestep, encoder_hidden_states),
        input_names=["hidden_states", "timestep", "encoder_hidden_states"],
        output_names=["noise_pred"], out_path=out_path,
    )
    del model, wrapper
    _clear_cache()


def _patch_wan_vae_singlepass():
    """Make ``AutoencoderKLWan._decode`` a single full-sequence pass.

    The stock ``_decode`` loops over latent frames with a streaming
    ``feat_cache``; under JIT trace this unrolls into N decoder forwards,
    which is extremely slow (and memory-heavy) to trace. With ``feat_cache``
    all-None (the state ``clear_cache`` produces), each ``WanCausalConv3d``
    runs a full causal conv over the whole sequence, which is numerically
    identical to the streamed loop — so a single ``decoder(x)`` call suffices.
    """
    from diffusers.models.autoencoders import autoencoder_kl_wan as _mod

    def _decode_singlepass(self, z, return_dict=True):
        self.clear_cache()
        x = self.post_quant_conv(z)
        self._conv_idx = [0]
        out = self.decoder(x, feat_cache=self._feat_map, feat_idx=self._conv_idx, first_chunk=True)
        if getattr(self.config, "patch_size", None) is not None:
            out = _mod.unpatchify(out, patch_size=self.config.patch_size)
        out = torch.clamp(out, min=-1.0, max=1.0)
        self.clear_cache()
        if not return_dict:
            return (out,)
        return _mod.DecoderOutput(sample=out)

    AutoencoderKLWan._decode = _decode_singlepass


def export_vae(model_dir, output_dir, height, width, num_frames, dtype):
    """Export the Wan VAE decoder to ONNX."""
    _patch_wan_vae_singlepass()
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
        opset=19,  # Wan VAE uses _upsample_nearest_exact2d (needs opset >=18)
    )
    del vae, wrapper
    _clear_cache()


def _dtype_of(name):
    """Map a dtype name string to a torch dtype."""
    return {"float32": torch.float32, "float16": torch.float16}[name]


def main():
    """Parse arguments and export the requested Wan2.1-T2V-1.3B sub-modules."""
    parser = argparse.ArgumentParser(description="Export Wan2.1-T2V-1.3B submodules to ONNX")
    parser.add_argument("--model-dir", required=True, help="Wan2.1-T2V-1.3B-Diffusers weight dir")
    parser.add_argument("--output-dir", default="./wan2_1_t2v_1_3b_onnx", help="ONNX output dir")
    parser.add_argument("--parts", default="text,transformer,vae",
                        help="comma list: text,transformer,vae")
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
    if "transformer" in parts:
        print("[export] Wan transformer (DiT) ...")
        export_transformer(args.model_dir, args.output_dir, args.height, args.width,
                           args.num_frames, args.max_seq_len, dtype, not args.no_custom_op)
    if "vae" in parts:
        print("[export] Wan VAE decoder ...")
        export_vae(args.model_dir, args.output_dir, args.height, args.width, args.num_frames, dtype)

    print("[export] done ->", args.output_dir)


if __name__ == "__main__":
    main()
