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
# WITHOUT WARRANTIES OR WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Export black-forest-labs/FLUX.1-dev to ONNX for MindSpore Lite Ascend inference.

FLUX.1-dev is a ~12B Rectified-Flow MMDiT (19 double-stream + 38 single-stream
blocks, inner_dim 3072). The pipeline (text-to-image) is exported as four ONNX
sub-graphs so the heavy compute runs as MindIR on Ascend:

  1. transformer : packed latents + T5/CLIP embeds + timestep/guidance + ids
                   -> noise prediction (the denoiser, run 28x in the loop).
                   Attention is exported as the CANN ``PromptFlashAttention``
                   Custom op (full bidirectional, no mask) by monkeypatching
                   diffusers' attention dispatch; everything else (AdaLN, RoPE,
                   projections) is traced as standard ops.
  2. vae_decoder : 16-ch latent -> RGB image (AutoencoderKL.decode).
  3. t5_encoder  : input_ids -> last_hidden_state (T5EncoderModel, XXL).
  4. clip_encoder: input_ids -> pooled embedding (CLIPTextModel).

Fixed shapes: 1024x1024 -> 4096 latent tokens; T5 sequence length 256
(FLUX.1-dev standard; pass --t5-seq-len 512 for longer prompts).
"""

import argparse
import gc
import math
from pathlib import Path

import torch

try:
    import torch._dynamo

    torch._dynamo.disable()
except Exception:
    pass

try:
    from diffusers import FluxTransformer2DModel, AutoencoderKL
    from diffusers.models import transformers as _  # noqa: F401 (ensure import side effects)
    from diffusers.models.transformers import transformer_flux
    from transformers import T5EncoderModel, CLIPTextModel
except ImportError as exc:  # pragma: no cover
    print(f"Error: missing dependency: {exc}")
    print("Install: pip install torch diffusers transformers onnx")
    raise SystemExit(1)


_USE_CUSTOM_OP = True


# ---------------------------------------------------------------------------
# Custom CANN operators (exported as ONNX ``Custom`` nodes for Ascend).
# ---------------------------------------------------------------------------


def _as_list_str(items):
    """Convert items to a list of string representations for ONNX attr."""
    return [str(x) for x in items]


class _CustomPromptFlashAttention(torch.autograd.Function):
    """CANN PromptFlashAttention Custom op for ONNX export.

    FLUX attention is full bidirectional (no causal / no padding mask). We still
    pass an ``atten_mask`` tensor (all-False == attend-to-all) because torch 2.9's
    legacy ONNX exporter only invokes an ``autograd.Function.symbolic`` when the
    op has a 4th tensor input; a no-op shared mask keeps the graph small
    (deduplicated to a single constant). q/k/v are in BNSD layout.
    """

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads_i,
                num_kv_heads_i, scale_value_f, input_layout_s):
        """Fallback full attention used during tracing / shape inference."""
        del ctx, atten_mask, num_heads_i, num_kv_heads_i, input_layout_s
        scores = torch.matmul(query, key.transpose(-2, -1)) * float(scale_value_f)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(probs, value)

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads_i,
                 num_kv_heads_i, scale_value_f, input_layout_s):
        """Export a Custom node for prompt flash attention."""
        y = g.op(
            "Custom", query, key, value, atten_mask,
            type_s="PromptFlashAttention",
            num_heads_i=int(num_heads_i),
            num_key_value_heads_i=int(num_kv_heads_i),
            scale_value_f=float(scale_value_f),
            input_layout_s=str(input_layout_s),
            pre_tokens_i=2147483647,
            next_tokens_i=0,
            sparse_mode_i=0,
            inner_precise_i=1,
            input_names_s=_as_list_str(["query", "key", "value", "atten_mask"]),
            optional_input_names_s=_as_list_str(["atten_mask"]),
            output_names_s=_as_list_str(["attention_out"]),
        )
        y.setType(query.type())
        return y


_NOOP_MASK_CACHE = {}


def _noop_mask(bsz, q_len, kv_len):
    """Cached all-False bool mask [bsz, 1, q_len, kv_len] (== attend-to-all).

    Cached by shape so all attention blocks in one forward share one constant
    (torch.onnx deduplicates the shared tensor -> no graph bloat).
    """
    key = (int(bsz), int(q_len), int(kv_len))
    m = _NOOP_MASK_CACHE.get(key)
    if m is None:
        m = torch.zeros(bsz, 1, q_len, kv_len, dtype=torch.bool)
        _NOOP_MASK_CACHE[key] = m
    return m


def _patch_rmsnorm():
    """Replace ``torch.nn.RMSNorm.forward`` with a standard-op decomposition.

    FLUX uses ``torch.nn.RMSNorm`` for q/k-norms; the legacy ONNX exporter has no
    symbolic for ``aten::rms_norm``. A 2-input ``autograd.Function`` symbolic is
    not reliably invoked by torch 2.9, so we decompose into standard ops
    (Pow/ReduceMean/Add/Rsqrt/Mul -- fp32 accumulator, identical to the native
    op). The converter maps these to Ascend ops directly.
    """
    def _forward(self, hidden_states):
        eps = float(getattr(self, "eps", 1e-6))
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        weight = getattr(self, "weight", None)
        if weight is not None:
            hidden_states = hidden_states.to(weight.dtype) * weight
        return hidden_states

    torch.nn.RMSNorm.forward = _forward


def _patch_flux_attention():
    """Replace diffusers attention dispatch with the CANN Custom op.

    The diffusers FLUX processor calls ``dispatch_attention_fn(q, k, v, ...)``
    with q/k/v in layout (batch, seq, num_heads, head_dim) [BSHD]. We transpose
    to BNSD, run the Custom op, and transpose back. q/k/v projections, RMSNorm
    (norm_q/norm_k) and RoPE stay as the original (standard) diffusers code.
    """
    def _custom_dispatch(query, key, value, attn_mask=None, dropout_p=0.0,
                         is_causal=False, scale=None, enable_gqa=False,
                         attention_kwargs=None, *, backend=None,
                         parallel_config=None):
        del dropout_p, is_causal, enable_gqa, attention_kwargs, backend, parallel_config
        # value layout: (batch, seq, num_heads, head_dim) [BSHD]
        bsz = int(value.shape[0])
        q_len = int(query.shape[1])
        kv_len = int(value.shape[1])
        num_heads = int(value.shape[-2])
        head_dim = int(value.shape[-1])
        scale_val = float(scale) if scale is not None else float(1.0 / math.sqrt(head_dim))
        q = query.transpose(1, 2)  # BNSD
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        mask = _noop_mask(bsz, q_len, kv_len)  # all-False == attend-to-all
        out = _CustomPromptFlashAttention.apply(
            q, k, v, mask, num_heads, num_heads, scale_val, "BNSD")
        return out.transpose(1, 2)  # back to BSHD

    transformer_flux.dispatch_attention_fn = _custom_dispatch


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _clear_cache():
    """Release torch caches and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _freeze(model):
    """Set requires_grad=False on all parameters (fixes torch 2.9 Conv2d trace error).

    torch 2.9's legacy ONNX exporter raises
    ``RuntimeError: Cannot insert a Tensor that requires grad as a constant``
    when tracing Conv2d whose weight Parameter still has requires_grad=True.
    Freezing params lets the tracer fold weights as constants (desired for
    inference export). Applied to all components for uniformity.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _latent_token_count(height, width, vae_scale=8, pack=2):
    """Number of packed latent tokens for an (height, width) image."""
    return (height // (vae_scale * pack)) * (width // (vae_scale * pack))


def _make_latent_image_ids(num_tokens, h_tokens, w_tokens, device, dtype):
    """Reproduce FluxPipeline._prepare_latent_image_ids (h_tokens x w_tokens)."""
    ids = torch.zeros(h_tokens, w_tokens, 3, device=device, dtype=dtype)
    ids[..., 1] = ids[..., 1] + torch.arange(h_tokens, device=device, dtype=dtype)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w_tokens, device=device, dtype=dtype)[None, :]
    return ids.reshape(h_tokens * w_tokens, 3)


# ---------------------------------------------------------------------------
# Export routines (one component at a time to keep host memory bounded).
# ---------------------------------------------------------------------------


def export_transformer(model_dir, output_dir, dtype, height, width, t5_seq_len, dynamo=True):
    """Export the FLUX.1 transformer (denoiser) to ONNX."""
    print(f"\n[transformer] loading FluxTransformer2DModel ({dtype}) ...")
    model = FluxTransformer2DModel.from_pretrained(
        model_dir, subfolder="transformer", torch_dtype=dtype, low_cpu_mem_usage=True)
    model.eval()
    _freeze(model)
    _patch_rmsnorm()
    # The PromptFlashAttention Custom op only works with the legacy (dynamo=False)
    # exporter. With dynamo (default, much faster) attention traces as standard
    # SDPA ops that the Ascend converter maps directly.
    if not dynamo:
        _patch_flux_attention()

    device = torch.device("cpu")
    h_tok = height // 16  # 1024 -> 64
    w_tok = width // 16
    num_img_tokens = h_tok * w_tok  # 4096
    in_channels = 64
    inner = int(model.config.num_attention_heads)  # 24

    dummy_hidden = torch.randn(1, num_img_tokens, in_channels, device=device, dtype=dtype)
    dummy_enc = torch.randn(1, t5_seq_len, 4096, device=device, dtype=dtype)
    dummy_pooled = torch.randn(1, 768, device=device, dtype=dtype)
    dummy_t = torch.tensor([0.5], device=device, dtype=dtype)        # timestep/1000
    dummy_guidance = torch.tensor([3.5], device=device, dtype=dtype)
    txt_ids = torch.zeros(t5_seq_len, 3, device=device, dtype=dtype)
    img_ids = _make_latent_image_ids(num_img_tokens, h_tok, w_tok, device, dtype)

    out_path = Path(output_dir) / "flux1_transformer.onnx"
    print(f"[transformer] exporting -> {out_path} "
          f"(img_tokens={num_img_tokens}, t5_seq={t5_seq_len}, heads={inner})")
    # NOTE: args must follow the FluxTransformer2DModel.forward positional order
    # (hidden_states, encoder_hidden_states, pooled_projections, timestep,
    #  img_ids, txt_ids, guidance) -- torch.onnx binds by position, not name.
    with torch.no_grad():
        torch.onnx.export(
            model, (
                dummy_hidden,        # hidden_states
                dummy_enc,           # encoder_hidden_states
                dummy_pooled,        # pooled_projections
                dummy_t,             # timestep
                img_ids,             # img_ids
                txt_ids,             # txt_ids
                dummy_guidance,      # guidance
            ), str(out_path),
            input_names=["hidden_states", "encoder_hidden_states", "pooled_projections",
                         "timestep", "img_ids", "txt_ids", "guidance"],
            output_names=["noise_pred"],
            opset_version=18, dynamo=dynamo)
    print("[transformer] done.")
    del model
    _clear_cache()
    return out_path


def export_vae(model_dir, output_dir, dtype, height, width, dynamo=True):
    """Export the FLUX.1 VAE decoder (AutoencoderKL.decode) to ONNX."""
    print(f"\n[vae] loading AutoencoderKL ({dtype}) ...")
    vae = AutoencoderKL.from_pretrained(
        model_dir, subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True)
    vae.eval()
    _freeze(vae)
    device = torch.device("cpu")
    latent_h, latent_w = height // 8, width // 8  # 128

    class _DecodeWrapper(torch.nn.Module):
        def __init__(self, decode_fn):
            super().__init__()
            self._decode = decode_fn

        def forward(self, z):
            return self._decode(z)[0]

    wrapper = _DecodeWrapper(vae.decode).to(device).eval()
    dummy_z = torch.randn(1, 16, latent_h, latent_w, device=device, dtype=dtype)
    out_path = Path(output_dir) / "flux1_vae_decoder.onnx"
    print(f"[vae] exporting -> {out_path} (latent {latent_h}x{latent_w})")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy_z,), str(out_path),
            input_names=["latents"], output_names=["image"],
            opset_version=18, dynamo=dynamo)
    print("[vae] done.")
    del vae, wrapper
    _clear_cache()
    return out_path


def export_t5(model_dir, output_dir, dtype, t5_seq_len, dynamo=True):
    """Export the T5-XXL encoder to ONNX (input_ids -> last_hidden_state)."""
    print(f"\n[t5] loading T5EncoderModel ({dtype}) ...")
    t5 = T5EncoderModel.from_pretrained(
        model_dir, subfolder="text_encoder_2", torch_dtype=dtype, low_cpu_mem_usage=True)
    t5.eval()
    _freeze(t5)
    device = torch.device("cpu")
    dummy_ids = torch.randint(0, 32000, (1, t5_seq_len), device=device, dtype=torch.int64)

    out_path = Path(output_dir) / "flux1_t5_encoder.onnx"
    print(f"[t5] exporting -> {out_path} (seq_len={t5_seq_len})")
    with torch.no_grad():
        torch.onnx.export(
            t5, (dummy_ids,), str(out_path),
            input_names=["input_ids"], output_names=["last_hidden_state"],
            opset_version=18, dynamo=dynamo)
    print("[t5] done.")
    del t5
    _clear_cache()
    return out_path


def export_clip(model_dir, output_dir, dtype, dynamo=True):
    """Export the CLIP text encoder to ONNX (input_ids -> pooled embedding)."""
    print(f"\n[clip] loading CLIPTextModel ({dtype}) ...")
    clip = CLIPTextModel.from_pretrained(
        model_dir, subfolder="text_encoder", torch_dtype=dtype, low_cpu_mem_usage=True)
    clip.eval()
    _freeze(clip)
    device = torch.device("cpu")
    max_len = 77
    dummy_ids = torch.randint(0, 49408, (1, max_len), device=device, dtype=torch.int64)

    class _ClipPool(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self._m = m

        def forward(self, input_ids):
            return self._m(input_ids).pooler_output

    wrapper = _ClipPool(clip).to(device).eval()
    out_path = Path(output_dir) / "flux1_clip_encoder.onnx"
    print(f"[clip] exporting -> {out_path} (seq_len={max_len})")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy_ids,), str(out_path),
            input_names=["input_ids"], output_names=["pooled_embeds"],
            opset_version=18, dynamo=dynamo)
    print("[clip] done.")
    del clip, wrapper
    _clear_cache()
    return out_path


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(description="Export FLUX.1-dev components to ONNX")
    parser.add_argument("--model-id", type=str, default="./FLUX.1-dev",
                        help="Local diffusers weights directory (or HF id).")
    parser.add_argument("--output-dir", type=str, default="./flux1_dev_onnx",
                        help="Output directory for ONNX files.")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1024, 1024],
                        metavar=("HEIGHT", "WIDTH"), help="Output image resolution.")
    parser.add_argument("--t5-seq-len", type=int, default=256,
                        help="Fixed T5 sequence length (256 standard, 512 for long prompts).")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "fp32"],
                        help="Export dtype. fp32 is REQUIRED for the legacy exporter on CPU: "
                             "aarch64 CPU fp16 Conv2d is pathologically slow (VAE hangs). The "
                             "converter's force_fp16 config still yields an fp16 MindIR.")
    parser.add_argument("--legacy", action="store_true",
                        help="Use the slow legacy TorchScript exporter (dynamo=False). "
                             "Only needed to emit the PromptFlashAttention Custom node.")
    parser.add_argument("--no-custom-op", action="store_true",
                        help="Do NOT replace attention with the CANN PromptFlashAttention op "
                             "(legacy exporter only).")
    parser.add_argument("--components", type=str,
                        default="transformer,vae,t5,clip",
                        help="Comma-separated subset to export.")
    return parser.parse_args()


def main():
    args = _parse_args()
    global _USE_CUSTOM_OP
    _USE_CUSTOM_OP = not bool(args.no_custom_op)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    height, width = args.resolution
    wanted = {c.strip() for c in args.components.split(",") if c.strip()}
    dynamo = not bool(args.legacy)
    # `--no-custom-op` forces dynamo (Custom op only works with the legacy exporter)
    use_dyn = dynamo or bool(args.no_custom_op)

    print(f"FLUX.1-dev ONNX export: resolution={height}x{width}, "
          f"t5_seq={args.t5_seq_len}, dtype={args.dtype}, dynamo={use_dyn}, "
          f"custom_op={(not args.no_custom_op) and (not use_dyn)}")

    if "transformer" in wanted:
        export_transformer(args.model_id, out, dtype, height, width, args.t5_seq_len, dynamo=use_dyn)
    if "vae" in wanted:
        export_vae(args.model_id, out, dtype, height, width, dynamo=use_dyn)
    if "t5" in wanted:
        export_t5(args.model_id, out, dtype, args.t5_seq_len, dynamo=use_dyn)
    if "clip" in wanted:
        export_clip(args.model_id, out, dtype, dynamo=use_dyn)

    print(f"\nAll requested components exported to {out}")


if __name__ == "__main__":
    main()
