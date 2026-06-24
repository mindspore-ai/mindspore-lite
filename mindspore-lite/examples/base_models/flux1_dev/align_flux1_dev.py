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
Accuracy alignment for FLUX.1-dev: HuggingFace (CPU) vs MindSpore Lite (Ascend).

Runs each component once on identical fixed inputs and reports the numerical
gap. This is the fast, rigorous check (1 forward per component, not the full
28-step loop):

  - CLIP / T5 : tokenize a fixed prompt, compare pooled / last_hidden_state.
  - transformer: one denoiser forward on a fixed latent+embeds, compare noise_pred.
  - VAE       : one decode on a fixed latent, compare the image.

Optionally also runs the full HF pipeline once on CPU and saves a baseline
image for visual/PSNR comparison with the MindIR output (--full-baseline).
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from infer_flux1_dev_mslite import (
    Flux1Inferencer, _build_model, _pack_latents, _unpack_latents,
    _latent_image_ids, _build_inputs)


def _stats(name, a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-6)
    print(f"  {name:28s} shape={str(a.shape):22s} "
          f"max_abs={diff.max():.6e}  mean_abs={diff.mean():.6e}  "
          f"max_rel={(diff/denom).max():.4e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="./FLUX.1-dev")
    p.add_argument("--onnx-dir", default="./flux1_dev_onnx")
    p.add_argument("--mindir-dir", default="./flux1_dev_onnx")
    p.add_argument("--prompt", default="A cat holding a sign that says hello world")
    p.add_argument("--t5-seq-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--transformer-device", type=int, default=0)
    p.add_argument("--text-device", type=int, default=1)
    p.add_argument("--full-baseline", action="store_true",
                   help="Also run the full HF pipeline on CPU (slow) and save a baseline image.")
    p.add_argument("--baseline-out", default="./flux1_baseline.png")
    args = p.parse_args()

    md = Path(args.model_dir)
    mindir = Path(args.mindir_dir)
    height = width = 1024
    h_tok = w_tok = 64
    num_img_tokens = 4096
    dtype = torch.float16

    print("=" * 70)
    print("FLUX.1-dev component parity: HuggingFace (CPU fp16) vs MindSpore Lite")
    print("=" * 70)

    # ---- fixed inputs ----
    from transformers import AutoTokenizer
    clip_tok = AutoTokenizer.from_pretrained(md / "tokenizer")
    t5_tok = AutoTokenizer.from_pretrained(md / "tokenizer_2")
    clip_ids = clip_tok([args.prompt], padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt").input_ids
    t5_ids = t5_tok([args.prompt], padding="max_length", max_length=args.t5_seq_len,
                    truncation=True, return_tensors="pt").input_ids
    rng = np.random.default_rng(args.seed)
    noise = rng.standard_normal((1, 16, 128, 128)).astype(np.float32)
    latents_packed = _pack_latents(noise, 16, 128, 128).astype(np.float16)
    img_ids = _latent_image_ids(h_tok, w_tok).astype(np.float16)
    txt_ids = np.zeros((args.t5_seq_len, 3), dtype=np.float16)

    # ---- HF components on CPU ----
    from diffusers import FluxTransformer2DModel, AutoencoderKL
    from transformers import T5EncoderModel, CLIPTextModel

    print("\n[HF] loading CLIP / T5 / VAE / transformer on CPU (fp16) ...")
    clip = CLIPTextModel.from_pretrained(md / "text_encoder", torch_dtype=dtype).eval()
    t5 = T5EncoderModel.from_pretrained(md / "text_encoder_2", torch_dtype=dtype).eval()
    vae = AutoencoderKL.from_pretrained(md / "vae", torch_dtype=dtype).eval()

    with torch.no_grad():
        hf_pooled = clip(clip_ids).pooler_output.numpy()
        hf_enc = t5(t5_ids)[0].numpy()

    # fixed latent for VAE-decode parity (re-use the init noise as a stand-in latent)
    latent_dec = (noise / 0.3611 + 0.1159).astype(np.float16)

    # ---- MindIR text encoders ----
    print("\n[MindIR] loading CLIP/T5/transformer/VAE MindIR ...")
    clip_m = _build_model(mindir / "flux1_clip_encoder_graph.mindir", "ascend", args.text_device)
    t5_m = _build_model(mindir / "flux1_t5_encoder_graph.mindir", "ascend", args.text_device)
    tx_m = _build_model(mindir / "flux1_transformer_graph.mindir", "ascend", args.transformer_device)
    vae_m = _build_model(mindir / "flux1_vae_decoder_graph.mindir", "ascend", args.transformer_device)

    ms_pooled = clip_m.predict(_build_inputs(
        clip_m, {"input_ids": clip_ids.numpy().astype(np.int64)}, ["input_ids"]))[0].get_data_to_numpy()
    ms_enc = t5_m.predict(_build_inputs(
        t5_m, {"input_ids": t5_ids.numpy().astype(np.int64)}, ["input_ids"]))[0].get_data_to_numpy()

    print("\n--- text encoder parity ---")
    _stats("CLIP pooled", hf_pooled, ms_pooled)
    _stats("T5 last_hidden_state", hf_enc, ms_enc)

    # ---- transformer forward parity (one step) ----
    from diffusers.models.transformers import transformer_flux
    from export_flux1_dev_onnx import _patch_flux_attention  # noqa: F401 (keeps HF math native)
    transformer = FluxTransformer2DModel.from_pretrained(
        md / "transformer", torch_dtype=dtype).eval()
    with torch.no_grad():
        hf_noise = transformer(
            hidden_states=torch.from_numpy(latents_packed),
            encoder_hidden_states=torch.from_numpy(hf_enc.astype(np.float16)),
            pooled_projections=torch.from_numpy(hf_pooled.astype(np.float16)),
            timestep=torch.tensor([0.5], dtype=dtype),
            guidance=torch.tensor([3.5], dtype=dtype),
            img_ids=torch.from_numpy(img_ids), txt_ids=torch.from_numpy(txt_ids),
            return_dict=False)[0].numpy()

    feed = {
        "hidden_states": latents_packed,
        "encoder_hidden_states": hf_enc.astype(np.float16),
        "pooled_projections": hf_pooled.astype(np.float16),
        "timestep": np.array([0.5], dtype=np.float16),
        "guidance": np.array([3.5], dtype=np.float16),
        "img_ids": img_ids, "txt_ids": txt_ids,
    }
    ms_noise = tx_m.predict(_build_inputs(
        tx_m, feed, ["hidden_states", "encoder_hidden_states", "pooled_projections",
                     "timestep", "guidance", "img_ids", "txt_ids"]))[0].get_data_to_numpy()
    print("\n--- transformer forward parity (1 step, t=0.5, guidance=3.5) ---")
    _stats("noise_pred", hf_noise, ms_noise)

    # ---- VAE decode parity ----
    with torch.no_grad():
        hf_img = vae.decode(torch.from_numpy(latent_dec), return_dict=False)[0].numpy()
    ms_img = vae_m.predict(_build_inputs(
        vae_m, {"latents": latent_dec}, ["latents"]))[0].get_data_to_numpy()
    print("\n--- VAE decode parity ---")
    _stats("vae image", hf_img, ms_img)

    print("\n" + "=" * 70)
    print("Parity check complete. max_abs <~1e-2 (fp16) on noise_pred/image is expected.")
    print("=" * 70)

    # ---- optional full HF baseline image ----
    if args.full_baseline:
        print("\n[HF] running full FluxPipeline on CPU (slow, 1 image) ...")
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(md, torch_dtype=dtype)
        init = torch.from_numpy(noise)
        img = pipe(prompt=args.prompt, num_inference_steps=28, guidance_scale=3.5,
                   height=1024, width=1024, latents=init, max_sequence_length=args.t5_seq_len,
                   generator=torch.Generator("cpu").manual_seed(args.seed)).images[0]
        img.save(args.baseline_out)
        print(f"[HF] baseline image saved -> {args.baseline_out}")


if __name__ == "__main__":
    main()
