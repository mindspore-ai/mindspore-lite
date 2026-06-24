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
Export black-forest-labs/FLUX.2-dev to ONNX for MindSpore Lite Ascend inference.

FLUX.2-dev is a ~32B MMDiT (8 double-stream + 48 single-stream blocks, inner_dim
6144, in_channels 128) whose transformer weights are 64 GB (bf16) -- too large
for a single 300I Duo chip (44 GB). The transformer is therefore split at export
time into two balanced ONNX sub-graphs for pipeline-parallel inference across two
chips (~32 GB each):

  - transformer_part0 : x/context embedders + temb/mods + RoPE + 8 double blocks
                        + first K single blocks.  hidden -> concatenated stream.
  - transformer_part1 : recompute temb/mods/RoPE + remaining single blocks +
                        drop text tokens + norm_out + proj_out -> noise_pred.

Splitting at export (not convert) keeps each ONNX ~32 GB so host-side
ONNX->MindIR conversion fits in memory. part1 recomputes temb/modulation/RoPE
from the original inputs (cheap, numerically identical, no weight duplication
beyond a few small modulation MLPs) so the per-step cross-chip handoff is a
single ``hidden_states`` tensor.

Also exports the VAE decoder (AutoencoderKLFlux2.decode). The Mistral3 text
encoder (~24B, 48 GB) is intentionally NOT exported here -- it runs on CPU once
per prompt in the inference script (see README).

Attention is exported as the CANN ``PromptFlashAttention`` Custom op (full
bidirectional, no mask) and q/k-norms as the CANN ``RmsNorm`` Custom op, by
monkeypatching diffusers' attention dispatch + ``torch.nn.RMSNorm``.
"""

import argparse
import gc
import math
from pathlib import Path

import torch
import torch.nn as nn

try:
    import torch._dynamo

    torch._dynamo.disable()
except Exception:
    pass

try:
    from diffusers import Flux2Transformer2DModel, AutoencoderKLFlux2
    from diffusers.models.transformers import transformer_flux2
except ImportError as exc:  # pragma: no cover
    print(f"Error: missing dependency: {exc}")
    print("Install: pip install torch diffusers transformers onnx")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Custom CANN operators (Custom ONNX nodes for Ascend).
# ---------------------------------------------------------------------------


def _as_list_str(items):
    return [str(x) for x in items]


class _CustomPromptFlashAttention(torch.autograd.Function):
    """CANN PromptFlashAttention Custom op for ONNX export (full bidirectional).

    See ``flux1_dev/export_flux1_dev_onnx.py`` for details. A 4th tensor input
    (all-False ``atten_mask`` == attend-to-all) is required because torch 2.9's
    legacy ONNX exporter only invokes an ``autograd.Function.symbolic`` when the
    op has a 4th tensor input; the shared mask is deduplicated (no graph bloat).
    """

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads_i,
                num_kv_heads_i, scale_value_f, input_layout_s):
        del ctx, atten_mask, num_heads_i, num_kv_heads_i, input_layout_s
        scores = torch.matmul(query, key.transpose(-2, -1)) * float(scale_value_f)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(probs, value)

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads_i,
                 num_kv_heads_i, scale_value_f, input_layout_s):
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
    """Cached all-False bool mask [bsz, 1, q_len, kv_len] (== attend-to-all)."""
    key = (int(bsz), int(q_len), int(kv_len))
    m = _NOOP_MASK_CACHE.get(key)
    if m is None:
        m = torch.zeros(bsz, 1, q_len, kv_len, dtype=torch.bool)
        _NOOP_MASK_CACHE[key] = m
    return m


def _patch_rmsnorm():
    """Decompose ``torch.nn.RMSNorm`` (q/k-norms) into standard ONNX ops.

    torch 2.9's legacy exporter has no symbolic for ``aten::rms_norm`` and does
    not reliably invoke a 2-input ``autograd.Function.symbolic``; the standard-op
    decomposition (fp32 variance -> rsqrt -> mul weight) is identical in math and
    maps to Ascend ops via the converter.
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


def _patch_flux2_attention():
    """Replace diffusers FLUX.2 attention dispatch with the CANN Custom op."""
    def _custom_dispatch(query, key, value, attn_mask=None, dropout_p=0.0,
                         is_causal=False, scale=None, enable_gqa=False,
                         attention_kwargs=None, *, backend=None,
                         parallel_config=None):
        del dropout_p, is_causal, enable_gqa, attention_kwargs, backend, parallel_config
        bsz = int(value.shape[0])
        q_len = int(query.shape[1])
        kv_len = int(value.shape[1])
        num_heads = int(value.shape[-2])
        head_dim = int(value.shape[-1])
        scale_val = float(scale) if scale is not None else float(1.0 / math.sqrt(head_dim))
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        mask = _noop_mask(bsz, q_len, kv_len)
        out = _CustomPromptFlashAttention.apply(
            q, k, v, mask, num_heads, num_heads, scale_val, "BNSD")
        return out.transpose(1, 2)

    transformer_flux2.dispatch_attention_fn = _custom_dispatch


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _latent_image_ids_4d(h_tokens, w_tokens):
    """FLUX.2 4D (T,H,W,L) latent position ids -> [h*w, 4]. Mirror _prepare_latent_ids."""
    t = torch.arange(1)
    h = torch.arange(h_tokens)
    w = torch.arange(w_tokens)
    l = torch.arange(1)
    return torch.cartesian_prod(t, h, w, l)  # [h*w, 4]


def _text_ids_4d(seq_len):
    """FLUX.2 4D text position ids -> [seq, 4]. Mirror _prepare_text_ids."""
    t = torch.arange(1); h = torch.arange(1); w = torch.arange(1); l = torch.arange(seq_len)
    return torch.cartesian_prod(t, h, w, l)  # [seq, 4]


def _rope(m, img_ids, txt_ids):
    """Reproduce FLUX.2 concat RoPE from separate img/txt ids."""
    image_rotary_emb = m.pos_embed(img_ids)
    text_rotary_emb = m.pos_embed(txt_ids)
    return (torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0))


# ---------------------------------------------------------------------------
# Split wrappers.
# ---------------------------------------------------------------------------


class Flux2Part0(nn.Module):
    """Embedders + temb/mods + RoPE + 8 double blocks + first K single blocks."""

    def __init__(self, m, k_single):
        super().__init__()
        self.x_embedder = m.x_embedder
        self.context_embedder = m.context_embedder
        self.time_guidance_embed = m.time_guidance_embed
        self.double_stream_modulation_img = m.double_stream_modulation_img
        self.double_stream_modulation_txt = m.double_stream_modulation_txt
        self.single_stream_modulation = m.single_stream_modulation
        self.pos_embed = m.pos_embed
        self.transformer_blocks = m.transformer_blocks
        self.single_transformer_blocks = nn.ModuleList(
            list(m.single_transformer_blocks)[:k_single])

    def forward(self, hidden_states, encoder_hidden_states, timestep, guidance,
                img_ids, txt_ids):
        timestep = timestep.to(hidden_states.dtype) * 1000
        guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)
        dmod_img = self.double_stream_modulation_img(temb)
        dmod_txt = self.double_stream_modulation_txt(temb)
        smod = self.single_stream_modulation(temb)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        concat_rotary_emb = _rope(self, img_ids, txt_ids)
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states, encoder_hidden_states, dmod_img, dmod_txt,
                concat_rotary_emb, {})
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for block in self.single_transformer_blocks:
            hidden_states = block(hidden_states, None, smod, concat_rotary_emb, {})
        return hidden_states


class Flux2Part1(nn.Module):
    """Remaining single blocks + norm_out + proj_out (recomputes temb/mods/RoPE)."""

    def __init__(self, m, k_single):
        super().__init__()
        self.time_guidance_embed = m.time_guidance_embed
        self.single_stream_modulation = m.single_stream_modulation
        self.pos_embed = m.pos_embed
        self.single_transformer_blocks = nn.ModuleList(
            list(m.single_transformer_blocks)[k_single:])
        self.norm_out = m.norm_out
        self.proj_out = m.proj_out

    def forward(self, hidden_states, timestep, guidance, img_ids, txt_ids):
        timestep = timestep.to(hidden_states.dtype) * 1000
        guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)
        smod = self.single_stream_modulation(temb)
        concat_rotary_emb = _rope(self, img_ids, txt_ids)
        for block in self.single_transformer_blocks:
            hidden_states = block(hidden_states, None, smod, concat_rotary_emb, {})
        num_txt = int(txt_ids.shape[0])
        hidden_states = hidden_states[:, num_txt:, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)


# ---------------------------------------------------------------------------
# Export routines.
# ---------------------------------------------------------------------------


def export_transformer(model_dir, output_dir, dtype, height, width, seq_len, k_single):
    """Export the FLUX.2 transformer as two pipeline-parallel ONNX halves."""
    print(f"\n[transformer] loading Flux2Transformer2DModel ({dtype}) ...")
    model = Flux2Transformer2DModel.from_pretrained(
        model_dir, subfolder="transformer", torch_dtype=dtype, low_cpu_mem_usage=True)
    model.eval()
    _patch_rmsnorm()
    _patch_flux2_attention()

    device = torch.device("cpu")
    h_tok = height // 16  # 1024 -> 64
    w_tok = width // 16
    num_img = h_tok * w_tok  # 4096
    in_ch = 128
    joint = int(model.config.joint_attention_dim)  # 15360

    img_ids = _latent_image_ids_4d(h_tok, w_tok).to(device=device, dtype=dtype)
    txt_ids = _text_ids_4d(seq_len).to(device=device, dtype=dtype)

    # ---- part0 ----
    part0 = Flux2Part0(model, k_single).to(device).eval()
    d_hidden = torch.randn(1, num_img, in_ch, device=device, dtype=dtype)
    d_enc = torch.randn(1, seq_len, joint, device=device, dtype=dtype)
    d_t = torch.tensor([0.5], device=device, dtype=dtype)
    d_g = torch.tensor([3.5], device=device, dtype=dtype)
    with torch.no_grad():
        mid = part0(d_hidden, d_enc, d_t, d_g, img_ids, txt_ids)
    print(f"[transformer] part0 mid hidden shape: {tuple(mid.shape)}")
    p0 = Path(output_dir) / "flux2_transformer_part0.onnx"
    print(f"[transformer] exporting part0 -> {p0}")
    with torch.no_grad():
        torch.onnx.export(
            part0, (d_hidden, d_enc, d_t, d_g, img_ids, txt_ids), str(p0),
            input_names=["hidden_states", "encoder_hidden_states", "timestep",
                         "guidance", "img_ids", "txt_ids"],
            output_names=["hidden_mid"],
            opset_version=18, do_constant_folding=False, dynamo=False)
    del part0, mid
    _clear_cache()

    # ---- part1 ----
    part1 = Flux2Part1(model, k_single).to(device).eval()
    d_mid = torch.randn(1, num_img + seq_len, int(model.config.num_attention_heads)
                        * int(model.config.attention_head_dim), device=device, dtype=dtype)
    p1 = Path(output_dir) / "flux2_transformer_part1.onnx"
    print(f"[transformer] exporting part1 -> {p1}")
    with torch.no_grad():
        torch.onnx.export(
            part1, (d_mid, d_t, d_g, img_ids, txt_ids), str(p1),
            input_names=["hidden_mid", "timestep", "guidance", "img_ids", "txt_ids"],
            output_names=["noise_pred"],
            opset_version=18, do_constant_folding=False, dynamo=False)
    del part1, model
    _clear_cache()
    return p0, p1


def export_vae(model_dir, output_dir, dtype, height, width):
    """Export the FLUX.2 VAE decoder (AutoencoderKLFlux2.decode) to ONNX."""
    print(f"\n[vae] loading AutoencoderKLFlux2 ({dtype}) ...")
    vae = AutoencoderKLFlux2.from_pretrained(
        model_dir, subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True)
    vae.eval()
    device = torch.device("cpu")
    latent_h, latent_w = height // 8, width // 8  # 128

    class _Wrap(nn.Module):
        def __init__(self, dec):
            super().__init__()
            self._dec = dec

        def forward(self, z):
            return self._dec(z)[0]

    wrapper = _Wrap(vae.decode).to(device).eval()
    dummy_z = torch.randn(1, 32, latent_h, latent_w, device=device, dtype=dtype)
    out_path = Path(output_dir) / "flux2_vae_decoder.onnx"
    print(f"[vae] exporting -> {out_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy_z,), str(out_path),
            input_names=["latents"], output_names=["image"],
            opset_version=18, do_constant_folding=False, dynamo=False)
    del vae, wrapper
    _clear_cache()
    return out_path


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(description="Export FLUX.2-dev transformer(split)+VAE to ONNX")
    p.add_argument("--model-id", default="./FLUX.2-dev")
    p.add_argument("--output-dir", default="./flux2_dev_onnx")
    p.add_argument("--resolution", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W"))
    p.add_argument("--seq-len", type=int, default=512, help="Mistral3 token sequence length.")
    p.add_argument("--split-single", type=int, default=16,
                   help="Number of single blocks in part0 (rest go to part1). 16 balances ~32GB each.")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--components", default="transformer,vae")
    return p.parse_args()


def main():
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    h, w = args.resolution
    wanted = {c.strip() for c in args.components.split(",") if c.strip()}
    print(f"FLUX.2-dev ONNX export: {h}x{w}, seq={args.seq_len}, split_single={args.split_single}, "
          f"dtype={args.dtype}")
    if "transformer" in wanted:
        export_transformer(args.model_id, out, dtype, h, w, args.seq_len, args.split_single)
    if "vae" in wanted:
        export_vae(args.model_id, out, dtype, h, w)
    print(f"\nAll requested components exported to {out}")


if __name__ == "__main__":
    main()
