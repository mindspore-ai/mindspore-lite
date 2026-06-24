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
"""Export stabilityai/stable-audio-3-small-music-base to ONNX for MindSpore Lite.

Stable Audio 3 (small) is a latent diffusion transformer for text-to-audio
generation, built with the ``stable-audio-tools`` package (NOT diffusers). The
pipeline is split into three fixed-shape sub-models so the heavy compute runs as
MindIR on Ascend:

  1. text_encoder      : T5 encoder -> last_hidden_state (text conditioning).
  2. dit (transformer) : latent DiT denoiser. Inputs = noisy latents + a global
                         conditioning vector (timestep + seconds embedded) +
                         T5 text embeds. Outputs = velocity / noise prediction.
                         Attention is exported as the CANN
                         ``PromptFlashAttention`` Custom op (full bidirectional,
                         no mask) by monkeypatching stable-audio-tools'
                         ``F.scaled_dot_product_attention`` call.
  3. audio_decoder     : latent autoencoder decoder. Maps denoised latents
                         [1, 64, latent_frames] -> stereo waveform
                         [1, 2, audio_samples] at 32 kHz.

All sub-models are loaded in float32 and exported with the legacy exporter
(``torch.onnx.utils.export``) at opset 17; fixed shapes (ascend_oriented
friendly). ``do_constant_folding=False`` (avoids materialising huge folded
constants for the long-sequence DiT attention graph and OOMs on CPU).

Architecture assumptions (stable-audio-tools is the source of truth; if your
checkpoint's ``model_config.json`` differs, override with --latent-channels /
--latent-downsampling / --audio-channels / --text-dim / --dit-hidden / --dit-
depth accordingly). See the README "FAQ / assumptions" section.
"""

import argparse
import gc
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# stable-audio-tools is the canonical loader. The package is pip-installable:
#   pip install stable-audio-tools
# It exposes create_model_from_config / create_autoencoder_from_config /
# create_conditioner_from_config which build the three sub-models from a
# model_config.json. If a different stable-audio-tools API is in use, point
# --model-dir at the unzipped checkpoint dir that contains model_config.json
# and the .safetensors / .pt weights.
try:
    import stable_audio_tools
    from stable_audio_tools import models as sat_models
    from stable_audio_tools import factories
except ImportError as exc:  # pragma: no cover
    print(f"Error: stable-audio-tools not found: {exc}")
    print("Install: pip install stable-audio-tools  (git+https://"
          "github.com/Stability-AI/stable-audio-tools)")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Defaults (match stable_audio_3_small music_base config; overridable via CLI).
# ---------------------------------------------------------------------------
_OPSET = 17

# Latent space: 64 latent channels, each latent frame ~ 1024 audio samples.
_DEFAULT_LATENT_CHANNELS = 64
_DEFAULT_LATENT_DOWNSAMPLING = 1024
_DEFAULT_AUDIO_CHANNELS = 2          # stereo
_DEFAULT_SAMPLE_RATE = 32000

# Text encoder (T5) -- Stable Audio 3 small uses a distilled T5 text encoder.
_DEFAULT_TEXT_DIM = 768
_DEFAULT_TEXT_SEQ_LEN = 256

# DiT -- stable_audio_3_small (music base): depth 24, hidden 1536 (24x64).
_DEFAULT_DIT_HIDDEN = 1536
_DEFAULT_DIT_DEPTH = 24
_DEFAULT_DIT_HEADS = 24
# The global conditioning vector carries (timestep, seconds_start, seconds_total)
# embedded through a small MLP; its dim equals the DiT hidden size.
_DEFAULT_GLOBAL_COND_DIM = _DEFAULT_DIT_HIDDEN


# ---------------------------------------------------------------------------
# Custom CANN operators (exported as ONNX ``Custom`` nodes for Ascend).
# ---------------------------------------------------------------------------


def _as_list_str(items):
    """Convert items to a list of string representations for ONNX attributes."""
    return [str(x) for x in items]


class _CustomPromptFlashAttentionFull(torch.autograd.Function):
    """CANN PromptFlashAttention (full bidirectional, no mask) for ONNX export.

    Stable Audio 3 DiT attention (self-attention over latent tokens and
    cross-attention to T5 text tokens) is full -- no causal / no padding mask --
    so the Custom node omits ``atten_mask`` (sparse_mode=0 + no mask ==
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
        matrix during the trace of long-sequence attention.
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


def _patch_scaled_dot_product_attention(num_heads_from_shape):
    """Replace ``F.scaled_dot_product_attention`` with the CANN Custom op.

    stable-audio-tools' DiT blocks call ``F.scaled_dot_product_attention`` (or a
    thin wrapper around it) with q/k/v in layout (batch, heads, seq, head_dim)
    [BHSD] (or BSHD -- we accept both by sniffing the seq/head dims). We
    transpose to BNSD, run the Custom op, and transpose back.

    ``num_heads_from_shape`` is a callable that, given q/k/v tensors, returns
    the number of attention heads (so we can pass it as a Custom op attribute
    regardless of the layout).
    """

    def _custom_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None, enable_gqa=False,
                     need_weights=False):
        del attn_mask, dropout_p, is_causal, enable_gqa, need_weights
        num_heads = int(num_heads_from_shape(query, key, value))
        head_dim = int(query.shape[-1])
        scale_val = float(scale) if scale is not None else float(
            1.0 / math.sqrt(head_dim))

        # Normalise to BNSD. F.scaled_dot_product_attention receives q/k/v in
        # (batch, heads, seq, head_dim); we keep that layout and call directly.
        out = _CustomPromptFlashAttentionFull.apply(
            query, key, value, num_heads, num_heads, scale_val, "BNSD")
        return out

    F.scaled_dot_product_attention = _custom_sdpa


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _clear_cache():
    """Release torch caches and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _latent_frames(seconds, sample_rate, downsampling):
    """Number of latent frames for ``seconds`` of audio (ceil-div)."""
    audio_samples = int(round(seconds * sample_rate))
    return int(math.ceil(audio_samples / float(downsampling)))


def _export_onnx(wrapper, dummy_inputs, input_names, output_names, out_path):
    """Trace a wrapper to ONNX with the legacy exporter (opset 17, float32)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.onnx.utils.export(
        wrapper,
        tuple(dummy_inputs),
        out_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=_OPSET,
        # Disable constant folding: with it on, the tracer materialises huge
        # folded constants for the long-sequence DiT attention graph and OOMs
        # on CPU. The converter does its own ascend_oriented optimisation, so
        # export-time folding is unnecessary.
        do_constant_folding=False,
    )
    _clear_cache()
    print(f"[export] saved {out_path}")


def _load_submodels(model_dir):
    """Build the three stable-audio-tools sub-models from model_config.json.

    Returns ``(dit, autoencoder, conditioner)``. The conditioner holds the T5
    text encoder (and any other embedding networks). All sub-models are loaded
    in float32 (the converter will quantise to fp16 via force_fp16).
    """
    config_path = os.path.join(model_dir, "model_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"model_config.json not found at {config_path}. Point --model-dir "
            "at the unzipped stable-audio-3-small-music-base checkpoint dir.")
    import json
    with open(config_path, "r") as f:
        config = json.load(f)

    # stable-audio-tools factory: build the DiT, autoencoder, and conditioning.
    dit = sat_models.create_model_from_config(config["model"]["model"])
    autoencoder = sat_models.create_autoencoder_from_config(
        config["model"]["autoencoder"])
    conditioner = factories.create_conditioner_from_config(
        config["model"]["conditioning"])

    # Load weights if a checkpoint is present.
    weight_candidates = [
        os.path.join(model_dir, "model.safetensors"),
        os.path.join(model_dir, "model.pt"),
    ]
    weight_path = next((p for p in weight_candidates if os.path.isfile(p)), None)
    if weight_path is not None:
        print(f"[load] loading weights from {weight_path}")
        if weight_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(weight_path)
        else:
            state = torch.load(weight_path, map_location="cpu")
        # stable-audio-tools stores all sub-models under a flat state dict with
        # prefixes "model." (DiT), "autoencoder." and "conditioner.".
        dit_state = {k[len("model."):]: v for k, v in state.items()
                     if k.startswith("model.")}
        ae_state = {k[len("autoencoder."):]: v for k, v in state.items()
                    if k.startswith("autoencoder.")}
        cond_state = {k[len("conditioner."):]: v for k, v in state.items()
                      if k.startswith("conditioner.")}
        dit.load_state_dict(dit_state, strict=False)
        autoencoder.load_state_dict(ae_state, strict=False)
        conditioner.load_state_dict(cond_state, strict=False)

    dit.eval()
    autoencoder.eval()
    conditioner.eval()
    return dit, autoencoder, conditioner


# ---------------------------------------------------------------------------
# Sub-module wrappers.
# ---------------------------------------------------------------------------


class _TextEncoderWrapper(nn.Module):
    """Wrap the T5 text encoder inside the stable-audio-tools conditioner.

    Exposes ``input_ids [1, seq_len] -> last_hidden_state [1, seq_len, text_dim]``.
    """

    def __init__(self, t5_encoder, text_dim):
        super().__init__()
        self.encoder = t5_encoder
        self.text_dim = int(text_dim)

    def forward(self, input_ids, attention_mask):
        """Return the T5 last hidden state [1, seq_len, text_dim]."""
        out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            return_dict=True)
        hidden = out.last_hidden_state
        # Project / pad to text_dim if the T5 hidden size differs.
        if int(hidden.shape[-1]) != self.text_dim:
            hidden = F.pad(hidden, (0, self.text_dim - int(hidden.shape[-1])))
        return hidden


class _DitWrapper(nn.Module):
    """Wrap the stable-audio-tools DiT to expose fixed ONNX inputs.

    Stable Audio 3's DiT.forward signature (per stable-audio-tools) is roughly::

        dit(x, t, cross_attn_cond, cross_attn_masks=None,
            global_cond=None, cfg_dropout=0.0)

    where ``x`` is the noisy latent [B, C, T], ``t`` is the diffusion timestep
    [B], ``cross_attn_cond`` is the T5 text embeds [B, S, D], and
    ``global_cond`` is the (timestep, seconds_start, seconds_total) conditioning
    vector [B, G]. We expose exactly those four inputs so the rest of the
    scheduler math (t embedding, seconds embedding) is computed by the ONNX graph
    as in the original code.

    NOTE: stable-audio-tools internally embeds ``t`` and the seconds conditioning
    into ``global_cond`` before calling the DiT. To keep the export tractable and
    the on-device math identical, we pass the pre-built ``global_cond`` vector
    directly (built by the inference script on CPU) and a zero ``t`` (its
    information is already inside global_cond). See README FAQ.
    """

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, x, t, cross_attn_cond, global_cond):
        """Return the velocity prediction [B, C, T_latent]."""
        out = self.dit(
            x=x, t=t, cross_attn_cond=cross_attn_cond,
            cross_attn_masks=None, global_cond=global_cond,
            cfg_dropout=0.0)
        if isinstance(out, tuple):
            out = out[0]
        return out


class _AudioDecoderWrapper(nn.Module):
    """Wrap the stable-audio-tools autoencoder decoder.

    Exposes ``latents [1, C, T_latent] -> audio [1, audio_channels, T_audio]``.
    The wrapper runs the decoder on already-denoised latents (the encoder half
    of the autoencoder is not used at inference time).
    """

    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, latents):
        """Decode latents to a stereo waveform [1, audio_channels, T_audio]."""
        # stable-audio-tools Autoencoder.decode(latents) -> (audio, *_)
        out = self.autoencoder.decode(latents)
        if isinstance(out, tuple):
            out = out[0]
        return out


# ---------------------------------------------------------------------------
# Export entry points.
# ---------------------------------------------------------------------------


def export_text_encoder(model_dir, output_dir, dit, autoencoder, conditioner,
                        text_seq_len, text_dim):
    """Export the T5 text encoder to ONNX (input_ids/mask -> last_hidden_state)."""
    del dit, autoencoder
    # The T5 encoder lives under conditioner.conditioners["text_t5"].model (the
    # exact key depends on the model_config.json; we look it up generically).
    t5_encoder, t5_tokenizer = _find_t5(conditioner)
    wrapper = _TextEncoderWrapper(t5_encoder, text_dim).eval()

    device = torch.device("cpu")
    input_ids = torch.zeros((1, text_seq_len), dtype=torch.int64, device=device)
    attention_mask = torch.ones((1, text_seq_len), dtype=torch.int64,
                                device=device)
    out_path = os.path.join(output_dir, "stable_audio_text_encoder.onnx")
    print(f"[export] T5 text encoder (seq_len={text_seq_len}, "
          f"text_dim={text_dim}) -> {out_path}")
    _export_onnx(
        wrapper, (input_ids, attention_mask),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"], out_path=out_path)
    del wrapper, t5_encoder
    _clear_cache()
    return t5_tokenizer


def export_dit(model_dir, output_dir, dit, latent_channels, latent_frames,
               text_seq_len, text_dim, global_cond_dim, use_custom_op,
               dit_hidden, dit_heads):
    """Export the stable-audio-tools DiT (denoiser) to ONNX.

    Attention is replaced with the CANN ``PromptFlashAttention`` Custom op when
    ``use_custom_op`` is True (default).
    """
    if use_custom_op:
        # The DiT calls F.scaled_dot_product_attention with q/k/v in
        # (batch, heads, seq, head_dim). num_heads is read off the q tensor.
        def _heads_from_shape(query, key, value):  # noqa: ARG001
            # q is BHSD: shape[1] == heads.
            return int(query.shape[1])
        _patch_scaled_dot_product_attention(_heads_from_shape)

    wrapper = _DitWrapper(dit).eval()
    device = torch.device("cpu")
    dtype = next(dit.parameters()).dtype
    x = torch.zeros((1, latent_channels, latent_frames), dtype=dtype,
                    device=device)
    # t is zero here (its information is already inside global_cond, per the
    # wrapper docstring). The graph reads the real timestep through global_cond.
    t = torch.zeros((1,), dtype=dtype, device=device)
    cross_attn_cond = torch.zeros((1, text_seq_len, text_dim), dtype=dtype,
                                  device=device)
    global_cond = torch.zeros((1, global_cond_dim), dtype=dtype, device=device)
    out_path = os.path.join(output_dir, "stable_audio_dit.onnx")
    print(f"[export] DiT (latent=[1,{latent_channels},{latent_frames}], "
          f"text=[1,{text_seq_len},{text_dim}], hidden={dit_hidden}, "
          f"heads={dit_heads}, depth~{0}) -> {out_path}")
    _export_onnx(
        wrapper, (x, t, cross_attn_cond, global_cond),
        input_names=["x", "t", "cross_attn_cond", "global_cond"],
        output_names=["velocity_pred"], out_path=out_path)
    del wrapper
    _clear_cache()


def export_audio_decoder(model_dir, output_dir, autoencoder, latent_channels,
                         latent_frames, audio_channels):
    """Export the autoencoder decoder to ONNX (latents -> waveform)."""
    wrapper = _AudioDecoderWrapper(autoencoder).eval()
    device = torch.device("cpu")
    dtype = next(autoencoder.parameters()).dtype
    latents = torch.zeros((1, latent_channels, latent_frames), dtype=dtype,
                          device=device)
    out_path = os.path.join(output_dir, "stable_audio_audio_decoder.onnx")
    print(f"[export] audio decoder (latents=[1,{latent_channels},"
          f"{latent_frames}] -> audio=[1,{audio_channels},~]) -> {out_path}")
    _export_onnx(
        wrapper, (latents,),
        input_names=["latents"], output_names=["audio"],
        out_path=out_path)
    del wrapper
    _clear_cache()


# ---------------------------------------------------------------------------
# Conditioner helpers (locate the T5 encoder + tokenizer).
# ---------------------------------------------------------------------------


def _find_t5(conditioner):
    """Locate the T5 encoder module + tokenizer inside a stable-audio-tools
    ``MultiConditioner``.

    The conditioner exposes its sub-conditioners via ``conditioner.conditioners``
    (a dict keyed by name, e.g. ``"text_t5"``). Each text conditioner wraps a
    ``transformers`` model + tokenizer. Returns ``(t5_encoder, tokenizer)``.
    """
    conds = getattr(conditioner, "conditioners", None)
    if conds is None:
        raise AttributeError(
            "conditioner has no .conditioners dict; is this a stable-audio-tools "
            "MultiConditioner?")
    for name, cond in conds.items():
        # The T5 text conditioner holds a ``model`` (T5EncoderModel) and a
        # ``tokenizer`` (or ``tok``). We accept either attribute name.
        model = getattr(cond, "model", None)
        tok = getattr(cond, "tokenizer", None) or getattr(cond, "tok", None)
        if model is None:
            continue
        # Heuristic: T5 encoders expose config.model_type == "t5".
        cfg = getattr(model, "config", None)
        mt = getattr(cfg, "model_type", "") if cfg is not None else ""
        if "t5" in str(mt).lower() or "t5" in str(name).lower():
            return model, tok
    # Fallback: return the first sub-conditioner that has a .model + tokenizer.
    for cond in conds.values():
        model = getattr(cond, "model", None)
        tok = getattr(cond, "tokenizer", None) or getattr(cond, "tok", None)
        if model is not None and tok is not None:
            return model, tok
    raise RuntimeError(
        f"could not locate a T5 encoder in conditioner.conditioners="
        f"{list(conds)}; set --text-encoder-name explicitly.")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _dtype_of(name):
    """Map a dtype name string to a torch dtype."""
    return {"float32": torch.float32, "float16": torch.float16}[name]


def _parse_args():
    """Parse CLI arguments for the ONNX export."""
    parser = argparse.ArgumentParser(
        description="Export stable-audio-3-small-music-base submodules to ONNX")
    parser.add_argument("--model-dir", required=True,
                        help="stable-audio-3-small-music-base checkpoint dir "
                        "(must contain model_config.json + weights).")
    parser.add_argument("--output-dir", default="./stable_audio_onnx",
                        help="ONNX output directory.")
    parser.add_argument("--parts", default="text,dit,decoder",
                        help="comma list: text,dit,decoder")
    # Audio geometry.
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="seconds of audio to generate (fixed). 10s default "
                        "keeps the example tractable; the model supports up to ~47s.")
    parser.add_argument("--sample-rate", type=int, default=_DEFAULT_SAMPLE_RATE)
    parser.add_argument("--audio-channels", type=int, default=_DEFAULT_AUDIO_CHANNELS)
    # Latent geometry.
    parser.add_argument("--latent-channels", type=int,
                        default=_DEFAULT_LATENT_CHANNELS)
    parser.add_argument("--latent-downsampling", type=int,
                        default=_DEFAULT_LATENT_DOWNSAMPLING,
                        help="audio samples per latent frame.")
    # Text geometry.
    parser.add_argument("--text-seq-len", type=int, default=_DEFAULT_TEXT_SEQ_LEN)
    parser.add_argument("--text-dim", type=int, default=_DEFAULT_TEXT_DIM)
    # DiT geometry.
    parser.add_argument("--dit-hidden", type=int, default=_DEFAULT_DIT_HIDDEN)
    parser.add_argument("--dit-heads", type=int, default=_DEFAULT_DIT_HEADS)
    parser.add_argument("--dit-depth", type=int, default=_DEFAULT_DIT_DEPTH)
    parser.add_argument("--global-cond-dim", type=int,
                        default=_DEFAULT_GLOBAL_COND_DIM)
    # Misc.
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float16"],
                        help="export dtype (float32 recommended for converter "
                        "compatibility; force_fp16 happens at converter time).")
    parser.add_argument("--no-custom-op", action="store_true",
                        help="Do NOT replace attention with the CANN "
                        "PromptFlashAttention op.")
    return parser.parse_args()


def main():
    """Parse arguments and export the requested stable-audio sub-modules."""
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = _dtype_of(args.dtype)

    # Cast all sub-models to the export dtype before tracing.
    dit, autoencoder, conditioner = _load_submodels(args.model_dir)
    dit = dit.to(dtype)
    autoencoder = autoencoder.to(dtype)
    conditioner = conditioner.to(dtype)

    latent_frames = _latent_frames(
        args.seconds, args.sample_rate, args.latent_downsampling)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]

    print(f"[export] stable-audio-3-small-music-base: seconds={args.seconds}, "
          f"latent_frames={latent_frames} (downsampling="
          f"{args.latent_downsampling}), dtype={args.dtype}, "
          f"custom_op={not args.no_custom_op}")

    if "text" in parts:
        print("[export] T5 text encoder ...")
        export_text_encoder(
            args.model_dir, args.output_dir, dit, autoencoder, conditioner,
            args.text_seq_len, args.text_dim)
    if "dit" in parts:
        print("[export] DiT (transformer) ...")
        export_dit(
            args.model_dir, args.output_dir, dit, args.latent_channels,
            latent_frames, args.text_seq_len, args.text_dim,
            args.global_cond_dim, not args.no_custom_op,
            args.dit_hidden, args.dit_heads)
    if "decoder" in parts:
        print("[export] audio decoder ...")
        export_audio_decoder(
            args.model_dir, args.output_dir, autoencoder,
            args.latent_channels, latent_frames, args.audio_channels)

    print(f"[export] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
