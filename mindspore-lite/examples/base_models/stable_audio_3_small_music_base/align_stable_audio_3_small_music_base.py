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
"""Accuracy alignment for stable-audio-3-small-music-base: HF vs MindSpore Lite.

Runs each component once on identical fixed inputs (shared seeded latents /
text ids / global_cond) and reports the numerical gap between the
stable-audio-tools baseline (CPU) and the MindIR inference (Ascend). This is the
fast, rigorous check (1 forward per component, not the full 100-step loop):

  - text encoder : tokenize a fixed prompt, compare last_hidden_state.
  - DiT          : one denoiser forward on a fixed latent+embeds+global_cond,
                   compare velocity_pred.
  - audio decoder: one decode on a fixed latent, compare the waveform.

The HF baseline uses stable-audio-tools directly (CPU, float32). The MindIR
side uses the converted MindIR graphs (Ascend, force_fp16).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Import the inferencer helpers (shared scheduler / global_cond / model runners).
from infer_stable_audio_3_small_music_base_mslite import (
    StableAudioInferencer, _build_global_cond, _build_model, _run_model)


def _stats(name, a, b):
    """Print max/mean abs + max rel diff between two arrays."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        print(f"  {name:24s} SHAPE MISMATCH a={a.shape} b={b.shape} -- skipping")
        return
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-6)
    print(f"  {name:24s} shape={str(a.shape):22s} "
          f"max_abs={diff.max():.6e}  mean_abs={diff.mean():.6e}  "
          f"max_rel={(diff / denom).max():.4e}")


def _load_hf_submodels(model_dir):
    """Build the three stable-audio-tools sub-models on CPU (float32)."""
    # Re-use the export script's loader so the HF baseline uses the same
    # model_config.json + checkpoint the ONNX was exported from.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from export_stable_audio_3_small_music_base_onnx import (
        _load_submodels, _find_t5)
    dit, autoencoder, conditioner = _load_submodels(model_dir)
    t5_encoder, tokenizer = _find_t5(conditioner)
    return dit, autoencoder, t5_encoder, tokenizer


def main():
    """Run the three-component parity check and print diff stats."""
    p = argparse.ArgumentParser(
        description="stable-audio-3-small alignment: HF (CPU) vs MindSpore Lite")
    p.add_argument("--model-dir", required=True,
                   help="stable-audio-3-small-music-base checkpoint dir.")
    p.add_argument("--mindir-dir", required=True,
                   help="dir with the 3 *_graph.mindir files.")
    p.add_argument("--prompt", default="128 BPM tech house drum loop")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text-seq-len", type=int, default=256)
    p.add_argument("--text-dim", type=int, default=768)
    p.add_argument("--global-cond-dim", type=int, default=1536)
    p.add_argument("--latent-channels", type=int, default=64)
    p.add_argument("--latent-downsampling", type=int, default=1024)
    p.add_argument("--sample-rate", type=int, default=32000)
    p.add_argument("--dit-device", type=int, default=0)
    p.add_argument("--text-device", type=int, default=1)
    p.add_argument("--decoder-device", type=int, default=0)
    args = p.parse_args()

    print("=" * 70)
    print("stable-audio-3-small-music-base component parity: "
          "stable-audio-tools (CPU) vs MindSpore Lite (Ascend)")
    print("=" * 70)

    import torch
    import math

    latent_frames = int(math.ceil(
        args.seconds * args.sample_rate / float(args.latent_downsampling)))
    dtype = torch.float32

    # ---- load HF sub-models (CPU baseline) ----
    print("\n[HF] loading stable-audio-tools sub-models on CPU (float32) ...")
    dit_hf, ae_hf, t5_hf, tokenizer = _load_hf_submodels(args.model_dir)
    dit_hf = dit_hf.to(dtype).eval()
    ae_hf = ae_hf.to(dtype).eval()
    t5_hf = t5_hf.to(dtype).eval()

    # ---- fixed inputs ----
    toks = tokenizer(
        args.prompt, padding="max_length", max_length=args.text_seq_len,
        truncation=True, add_special_tokens=True, return_attention_mask=True,
        return_tensors="pt")
    input_ids = toks["input_ids"].to(torch.int64)
    attention_mask = toks["attention_mask"].to(torch.int64)

    rng = np.random.RandomState(args.seed)
    latent_init = rng.standard_normal(
        (1, args.latent_channels, latent_frames)).astype(np.float32)
    # HF side: latents start at sigma_max (matches the inference script).
    latent_init_hf = torch.from_numpy(latent_init * 1000.0)

    sigma = 1000.0  # initial sigma
    global_cond_np = _build_global_cond(
        sigma, 0.0, args.seconds, args.global_cond_dim, 0).astype(np.float32)
    global_cond_hf = torch.from_numpy(global_cond_np)

    # ---- HF forwards ----
    with torch.no_grad():
        hf_text = t5_hf(
            input_ids=input_ids, attention_mask=attention_mask,
            return_dict=True).last_hidden_state.numpy()
        if hf_text.shape[-1] != args.text_dim:
            pad = args.text_dim - hf_text.shape[-1]
            if pad > 0:
                hf_text = np.pad(hf_text, ((0, 0), (0, 0), (0, pad)))
            else:
                hf_text = hf_text[..., :args.text_dim]

        hf_v = dit_hf(
            x=latent_init_hf, t=torch.tensor([sigma], dtype=dtype),
            cross_attn_cond=torch.from_numpy(hf_text),
            cross_attn_masks=None, global_cond=global_cond_hf,
            cfg_dropout=0.0)
        if isinstance(hf_v, tuple):
            hf_v = hf_v[0]
        hf_v = hf_v.numpy()

        hf_audio = ae_hf.decode(latent_init_hf)
        if isinstance(hf_audio, tuple):
            hf_audio = hf_audio[0]
        hf_audio = hf_audio.numpy()

    # ---- MindIR forwards ----
    print("\n[MindIR] loading MindIR sub-models ...")
    text_m = _build_model(
        Path(args.mindir_dir) / "stable_audio_text_encoder_graph.mindir",
        args.text_device)
    dit_m = _build_model(
        Path(args.mindir_dir) / "stable_audio_dit_graph.mindir",
        args.dit_device)
    dec_m = _build_model(
        Path(args.mindir_dir) / "stable_audio_audio_decoder_graph.mindir",
        args.decoder_device)

    ms_text = _run_model(
        text_m,
        {"input_ids": input_ids.numpy().astype(np.int64),
         "attention_mask": attention_mask.numpy().astype(np.int64)},
        ["input_ids", "attention_mask"])[0]
    ms_v = _run_model(
        dit_m,
        {"x": (latent_init * 1000.0).astype(np.float32),
         "t": np.array([sigma], dtype=np.float32),
         "cross_attn_cond": hf_text.astype(np.float32),
         "global_cond": global_cond_np},
        ["x", "t", "cross_attn_cond", "global_cond"])[0]
    ms_audio = _run_model(
        dec_m,
        {"latents": (latent_init * 1000.0).astype(np.float32)},
        ["latents"])[0]

    # ---- stats ----
    print("\n--- text encoder parity ---")
    _stats("T5 last_hidden_state", hf_text, ms_text)

    print("\n--- DiT forward parity (1 step, sigma=1000.0) ---")
    _stats("velocity_pred", hf_v, ms_v)

    print("\n--- audio decoder parity ---")
    _stats("audio waveform", hf_audio, ms_audio)

    print("\n" + "=" * 70)
    print("Parity check complete. fp16 on Ascend -> expect max_abs ~1e-2 to "
          "1e-1 on velocity_pred/audio (long-sequence attention + fp16).")
    print("If max_rel on velocity_pred is large, see README FAQ "
          "(force_fp32 / assumptions).")
    print("=" * 70)


if __name__ == "__main__":
    main()
