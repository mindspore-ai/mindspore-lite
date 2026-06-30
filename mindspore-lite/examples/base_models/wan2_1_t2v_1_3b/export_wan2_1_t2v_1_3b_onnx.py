"""Export Wan2.1-T2V-1.3B-Diffusers submodules to ONNX for MindSpore Lite cloud-side inference.

The pipeline is split into three fixed-shape sub-models (mirrors the flux1_dev
example, which is the closest reference in this repo):

  * text encoder  : UMT5-XXL, input_ids[1,512] + attention_mask[1,512]
                    -> last_hidden_state[1,512,4096]
  * transformer   : WanTransformer3DModel (DiT, 1.3B), hidden_states[1,16,F',H',W'] +
                    timestep[1] + encoder_hidden_states[1,512,4096] -> noise_pred
  * vae decoder   : AutoencoderKLWan.decode, latents[1,16,F',H',W'] -> video[1,3,F,H,W]

The transformer attention (full bidirectional, no mask) is rewritten as standard
``BatchMatMul + Softmax + BatchMatMul`` ops by monkeypatching diffusers' attention
dispatch (``transformer_wan.dispatch_attention_fn``). The CANN ``PromptFlashAttention``
Custom-op path does NOT convert on Atlas 300I Duo (310P3): the GE build cannot infer
the Custom node's output shape ("context is a null pointer"). Wan's attention is full
(no mask), so standard-op attention is numerically identical and is feasible for the
fixed 13-frame / 480x832 demo shape. The T5 / VAE are also standard-op graphs.

All sub-models are loaded in float32 and exported at fixed shapes
(ascend_oriented friendly). The text encoder uses the legacy exporter
(``torch.onnx.utils.export``) at opset 17; the transformer and VAE use the
dynamo exporter (``torch.onnx.export(dynamo=True)``) which produces fully
static graphs (0 symbolic dims) and is far faster for large 3D-conv models.
"""

import argparse
import gc
import math
import os

import torch
from torch import nn
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers.models.transformers import transformer_wan
from transformers import UMT5EncoderModel

_VAE_SCALE_FACTOR_TEMPORAL = 4
_VAE_SCALE_FACTOR_SPATIAL = 8
_LATENT_CHANNELS = 16
_OPSET = 17


# ---------------------------------------------------------------------------
# diffusers monkey-patches (standard-op decompositions for clean conversion).
# ---------------------------------------------------------------------------


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
    """Replace Wan's attention dispatch with standard-op full attention.

    The Wan processor calls ``dispatch_attention_fn(q, k, v, ...)`` with q/k/v in
    layout (batch, seq, num_heads, head_dim) [BSHD]. We transpose to BNSD and run
    explicit ``BatchMatMul + Softmax + BatchMatMul`` attention, then transpose back.
    Wan's self/cross attention are full (no causal / no padding mask), so this is
    numerically identical to diffusers' SDPA. q/k/v projections, RMSNorm
    (norm_q/norm_k) and RoPE stay as the original diffusers code.
    """

    def _custom_dispatch(query, key, value, attn_mask=None, dropout_p=0.0,
                         is_causal=False, scale=None, enable_gqa=False,
                         attention_kwargs=None, *, backend=None,
                         parallel_config=None):
        del attn_mask, dropout_p, is_causal, enable_gqa, attention_kwargs
        del backend, parallel_config
        # value layout: (batch, seq, num_heads, head_dim)
        head_dim = int(value.shape[-1])
        scale_val = float(scale) if scale is not None else float(1.0 / math.sqrt(head_dim))
        q = query.transpose(1, 2)  # BNSD
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        # Explicit full attention (BatchMatMul + Softmax + BatchMatMul). The CANN
        # PromptFlashAttention Custom-op plugin path does NOT convert on Atlas 300I
        # Duo (310P3): the GE build cannot infer the Custom node's output shape and
        # aborts with "context is a null pointer" on the downstream Transpose. Wan's
        # self/cross attention are both full (no mask), so standard-op attention is
        # numerically identical; softmax runs in fp32 for stability then casts back.
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale_val   # [B,N,Sq,Sk]
        attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)                               # [B,N,Sq,D]
        return out.transpose(1, 2)  # back to BSHD

    transformer_wan.dispatch_attention_fn = _custom_dispatch


def _patch_wan_layernorm():
    """Replace ``FP32LayerNorm.forward`` with a standard-op decomposition.

    Wan uses ``FP32LayerNorm`` (an ``nn.LayerNorm`` subclass) for the per-block
    norm1/norm2/norm3 and the output norm. The dynamo ONNX exporter fuses
    ``F.layer_norm`` into a single ``LayerNormalization`` op whose output shape the
    Ascend GE build cannot infer ("context is a null pointer"). Decomposing into
    Mean/Sub/Pow/Mean/Sqrt/Div/Mul/Add (all standard ops, computed in fp32) is
    numerically identical and keeps the graph fully static + standard-op.
    """

    from diffusers.models.normalization import FP32LayerNorm

    def _forward(self, inputs):
        origin_dtype = inputs.dtype
        x = inputs.float()
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) * (x - mean)).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if getattr(self, "weight", None) is not None:
            x = x * self.weight.float()
        if getattr(self, "bias", None) is not None:
            x = x + self.bias.float()
        return x.to(origin_dtype)

    FP32LayerNorm.forward = _forward


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
                 dynamic_axes=None, opset=_OPSET, constant_folding=False,
                 use_dynamo=False):
    """Trace a wrapper to ONNX (float32) and resolve shapes for the converter.

    Two exporters:

    * **legacy** (``use_dynamo=False``, default) — ``torch.onnx.utils.export``. The
      Wan VAE uses this; a post-pass runs ONNX shape inference with ``data_prop``
      to turn the ``.size()``-driven symbolic dims into concrete shapes.
    * **dynamo** (``use_dynamo=True``) — ``torch.onnx.export(dynamo=True)``. The Wan
      transformer uses this: the legacy exporter leaves thousands of symbolic dims
      that onnxsim would normally fold, but the transformer is >2 GiB so onnxsim
      cannot run. dynamo produces a fully static graph directly.

    The ``ascend_oriented`` GE build aborts ("context is a null pointer") if any
    intermediate shape stays symbolic, so we must hand it a fully static graph.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if use_dynamo:
        torch.onnx.export(
            wrapper,
            tuple(dummy_inputs),
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamo=True,
            external_data=True,
        )
    else:
        torch.onnx.utils.export(
            wrapper,
            tuple(dummy_inputs),
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=constant_folding,
        )
        # Resolve symbolic dims -> concrete shapes for the ascend_oriented converter.
        # data_prop shape inference alone is not enough for the Wan VAE (dynamic
        # Shape/Gather/Concat->Reshape chains); onnxsim constant-folds them, the way
        # dynamo does for the transformer. onnxsim only fits models <2 GiB (the VAE /
        # text encoder), so it is best-effort and falls back to data_prop on failure.
        import onnx
        from onnx import shape_inference
        model = onnx.load(out_path, load_external_data=False)
        try:
            import onnxsim
            model, _ = onnxsim.simplify(model)
        except Exception as exc:  # noqa: BLE001 -- fall back to data_prop only
            print(f"[export] onnxsim skipped ({type(exc).__name__}); using data_prop")
            model = shape_inference.infer_shapes(model, data_prop=True)
        onnx.save(model, out_path)
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
    """Wrap AutoencoderKLWan decode to expose latents -> video.

    Calls the decode steps directly (post_quant_conv -> decoder -> clamp) instead of
    ``vae.decode``/``_decode``: the streaming ``_decode`` mutates instance state
    (``self._feat_map`` via ``clear_cache``) which ``torch.export`` (dynamo)
    mis-traces and collapses the spatial output. The conv/resample patches already
    ignore the cache, so calling ``decoder(x, feat_cache=None)`` is equivalent and
    gives dynamo a clean, stateless graph.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        """Return the decoded video [1, 3, F, H, W] (latents already denormalised)."""
        vae = self.vae
        x = vae.post_quant_conv(latents)
        out = vae.decoder(x, feat_cache=None, feat_idx=[0], first_chunk=True)
        patch_size = getattr(vae.config, "patch_size", None)
        if patch_size is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
            out = unpatchify(out, patch_size=patch_size)
        return torch.clamp(out, min=-1.0, max=1.0)


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
                       max_seq_len, dtype, use_custom_op=True):
    """Export the Wan DiT transformer to ONNX (attention -> standard-op full attn).

    Always applies the RMSNorm + attention standard-op patches: ``aten::rms_norm``
    has no legacy-exporter symbolic, and ``F.scaled_dot_product_attention`` lowers to
    an ONNX ``If`` the converter cannot handle, so both must be decomposed.
    ``use_custom_op`` is retained for CLI compatibility but the CANN Custom-op path
    is not used (it does not convert on Atlas 300I Duo / 310P3).
    """
    del use_custom_op
    _patch_rmsnorm()
    _patch_wan_layernorm()
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
        # dynamo exporter: produces a fully static graph (0 symbolic dims). The
        # legacy exporter leaves thousands of symbolic dims that onnxsim would
        # fold, but the transformer is >2 GiB so onnxsim cannot run.
        use_dynamo=True,
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


def _patch_wan_resample_singlepass():
    """Make ``WanResample`` do the full temporal upsample in a single pass.

    The stock ``upsample3d`` path only runs the temporal 2x upsample
    (``time_conv`` + interleave-reshape) on the *second* call of its streaming
    cache protocol; on the first call it just marks the cache ("Rep") and returns
    ``x`` unchanged. ``_patch_wan_vae_singlepass`` makes the decoder run a single
    full-sequence pass, so that first call is also the only call -- and without
    this patch the temporal axis is never upsampled (a 4-latent-frame input decodes
    to 4 frames instead of 13). We replace ``forward`` so ``upsample3d`` always does
    ``time_conv`` (full causal conv over the whole sequence) + the 2x interleave,
    which is exactly what the streaming "Rep" branch computes for a full sequence.
    """

    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResample

    def _forward(self, x):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            x = self.time_conv(x)                       # [b, c*2, t, h, w]
            x = x.reshape(b, 2, c, t, h, w)
            x = torch.stack((x[:, 0], x[:, 1]), 3)      # [b, c, t, 2, h, w]
            x = x.reshape(b, c, t * 2, h, w)            # temporal 2x interleave
            # diffusers' streaming decode feeds latent frames one at a time, so the
            # first frame is not duplicated -> 2t-1 output frames per upsample (4->7->13
            # for a 4-latent-frame / 13-video-frame clip). The full-sequence single pass
            # interleaves 2t; drop the g1_0 slot (temporal index 1) to match. The causal
            # time_conv sees the whole sequence, so the kept frames are numerically
            # identical to the streaming decode.
            x = torch.cat([x[:, :, :1], x[:, :, 2:]], dim=2)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)                            # spatial 2x upsample
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
        return x

    WanResample.forward = _forward


def _patch_wan_vae_no_cache():
    """Strip the streaming feat_cache/feat_idx logic from the VAE conv blocks.

    The stock ``WanCausalConv3d`` / ``WanResidualBlock`` forwards mutate a Python
    list (``feat_cache``) and an index counter (``feat_idx``) passed through the
    decoder. ``torch.export`` (dynamo) mis-traces that mutable state and the
    exported VAE collapses to a spatially-flat output (each channel comes out
    uniform). For a single full-sequence pass the cache is never read (every conv
    is called once with ``cache_x=None`` -> full causal conv), so removing the
    cache branches changes nothing numerically but gives dynamo a clean graph.
    """

    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    def _cc_forward(self, x, cache_x=None):
        del cache_x
        # WanCausalConv3d extends nn.Conv3d; pad (asymmetric, causal) then run the conv.
        return torch.nn.Conv3d.forward(self, torch.nn.functional.pad(x, self._padding))

    WanCausalConv3d.forward = _cc_forward

    def _res_forward(self, x, **_kwargs):
        h = self.conv_shortcut(x)
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + h

    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualBlock
    WanResidualBlock.forward = _res_forward


def _patch_wan_vae_attention():
    """Replace SDPA in the VAE mid-block attention with explicit ops.

    The stock ``WanAttentionBlock`` calls ``F.scaled_dot_product_attention``, which the
    legacy ONNX exporter lowers to an ONNX ``If`` node (flash-vs-math backend selection).
    The ``ascend_oriented`` converter cannot lower that ``If``
    ("i: 3 out of range ... ValueNode<If>", conversion aborts). The mid-block attention
    is single-head over the HxW spatial tokens, so spelling it out as
    BatchMatMul + Softmax + BatchMatMul (1/sqrt(C) scaling) is numerically identical and
    traces to standard ops with no control flow.
    """

    from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock

    def _forward(self, x):
        identity = x
        batch_size, channels, time, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1).permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        # explicit single-head attention (replaces F.scaled_dot_product_attention)
        scale = float(1.0 / float(channels) ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)
        # x is [B*T, 1, H*W, C]; index the singleton head axis with [:, 0] instead of
        # squeeze(1) -- torch.squeeze emits a conditional ONNX If ("squeeze only if the
        # dim is size 1") that the ascend_oriented converter cannot lower.
        x = x[:, 0].permute(0, 2, 1).reshape(batch_size * time, channels, height, width)
        x = self.proj(x)
        x = x.view(batch_size, time, channels, height, width).permute(0, 2, 1, 3, 4)
        return x + identity

    WanAttentionBlock.forward = _forward


def _patch_wan_upsample():
    """Replace ``WanUpsample.forward`` with a standard-op 2x nearest-exact decomposition.

    diffusers' WanUpsample delegates to ``nn.Upsample(mode="nearest-exact",
    scale_factor=(2, 2))``, which traces to ``aten::_upsample_nearest_exact2d`` -- an
    internal op with no symbolic in the legacy ONNX exporter (export aborts with
    ``UnsupportedOperatorError`` at opset 17/19). WanUpsample is only ever built with a
    2x2 factor over a 4D ``[N, C, H, W]`` tensor (WanResample reshapes the 5D video to
    4D before upsampling, then back to 5D). For an integer 2x factor, nearest-exact is
    exactly "duplicate each pixel 2x2" -- i.e. ``repeat_interleave(2)`` on H and W --
    which traces to standard ONNX ops and needs no upsample symbolic.
    """

    from diffusers.models.autoencoders.autoencoder_kl_wan import WanUpsample

    def _forward(x):
        dtype = x.dtype
        xf = x.float()
        xf = xf.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        return xf.to(dtype)

    WanUpsample.forward = staticmethod(_forward)


def export_vae(model_dir, output_dir, height, width, num_frames, dtype):
    """Export the Wan VAE decoder to ONNX."""
    _patch_wan_vae_singlepass()
    _patch_wan_vae_no_cache()
    _patch_wan_resample_singlepass()
    _patch_wan_vae_attention()
    _patch_wan_upsample()
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
        # dynamo exporter: the legacy JIT trace of the 13-frame VAE decoder (full-res
        # 3D causal convs after temporal upsample) takes ~1h on CPU, while dynamo's
        # torch.export is far faster and also yields a fully static graph.
        use_dynamo=True,
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
                        help="(legacy) kept for CLI compatibility; attention is always "
                             "exported as standard-op full attention (BatchMatMul+Softmax).")
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
                           args.num_frames, args.max_seq_len, dtype)
    if "vae" in parts:
        print("[export] Wan VAE decoder ...")
        export_vae(args.model_dir, args.output_dir, args.height, args.width, args.num_frames, dtype)

    print("[export] done ->", args.output_dir)


if __name__ == "__main__":
    main()
