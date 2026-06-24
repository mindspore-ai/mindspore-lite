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
"""
Accuracy alignment for FLUX.2-dev: HuggingFace (CPU) vs MindSpore Lite (Ascend).

Validates the two things that differ from the baseline:
  - transformer split : part0(dev0) + part1(dev1) MindIR vs the full HF model
                        (one forward on fixed inputs -> compare noise_pred).
  - VAE decode        : MindIR vs HF AutoencoderKLFlux2.decode.

The Mistral3 text encoder is identical on both sides (it runs on CPU in both the
baseline and the MindIR pipeline), so it is not compared here.

Loading the full 32B transformer on CPU needs ~64 GB host RAM.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from infer_flux2_dev_mslite import (
    Flux2Inferencer, _build_model, _build_inputs, _pack_latents,
    _unpack_latents_rowmajor, _unpatchify_latents)


def _stats(name, a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    diff = np.abs(a - b); denom = np.maximum(np.abs(a), 1e-6)
    print(f"  {name:26s} shape={str(a.shape):22s} max_abs={diff.max():.6e} "
          f"mean_abs={diff.mean():.6e} max_rel={(diff/denom).max():.4e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="./FLUX.2-dev")
    p.add_argument("--mindir-dir", default="./flux2_dev_onnx")
    p.add_argument("--prompt", default="A cat holding a sign that says hello world")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--part0-device", type=int, default=0)
    p.add_argument("--part1-device", type=int, default=1)
    args = p.parse_args()

    md = Path(args.model_dir); mindir = Path(args.mindir_dir)
    height = width = 1024
    h_tok = w_tok = 64
    dtype = torch.float16

    print("=" * 70)
    print("FLUX.2-dev parity: HuggingFace (CPU fp16) vs MindSpore Lite (pipeline-parallel)")
    print("=" * 70)

    # ---- fixed inputs: Mistral3 embeds (CPU, shared by both sides) ----
    from transformers import AutoProcessor, Mistral3ForConditionalGeneration
    from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
    print("\n[Mistral3] encoding prompt on CPU ...")
    te = Mistral3ForConditionalGeneration.from_pretrained(
        md / "text_encoder", torch_dtype=dtype, low_cpu_mem_usage=True).eval()
    tok = AutoProcessor.from_pretrained(md / "tokenizer")
    with torch.no_grad():
        embeds = Flux2Pipeline._get_mistral_3_small_prompt_embeds(
            text_encoder=te, tokenizer=tok, prompt=args.prompt, dtype=dtype,
            device="cpu", max_sequence_length=args.seq_len)
        text_ids = Flux2Pipeline._prepare_text_ids(embeds).to(dtype)
    enc = embeds.numpy().astype(np.float16)
    txt_ids = text_ids.numpy().astype(np.float16)
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    del te  # free ~48GB before loading the transformer
    import gc; gc.collect()

    rng = np.random.default_rng(args.seed)
    noise = rng.standard_normal((1, 128, h_tok, w_tok)).astype(np.float32)
    latents_packed = _pack_latents(noise).astype(np.float16)
    img_ids = Flux2Inferencer._latent_ids_np(h_tok, w_tok).astype(np.float16)

    # ---- HF full transformer (CPU) ----
    print("\n[HF] loading full Flux2Transformer2DModel on CPU (~64GB) ...")
    from diffusers import Flux2Transformer2DModel
    tx = Flux2Transformer2DModel.from_pretrained(
        md / "transformer", torch_dtype=dtype, low_cpu_mem_usage=True).eval()
    t_t = torch.tensor([0.5], dtype=dtype); t_g = torch.tensor([3.5], dtype=dtype)
    with torch.no_grad():
        hf_noise = tx(hidden_states=torch.from_numpy(latents_packed),
                      encoder_hidden_states=torch.from_numpy(enc),
                      timestep=t_t, guidance=t_g,
                      img_ids=torch.from_numpy(img_ids), txt_ids=torch.from_numpy(txt_ids),
                      return_dict=False)[0].numpy()
    del tx; gc.collect()

    # ---- MindIR pipeline-parallel transformer ----
    print("\n[MindIR] loading transformer part0/part1 ...")
    p0 = _build_model(mindir / "flux2_transformer_part0_graph.mindir", "ascend", args.part0_device)
    p1 = _build_model(mindir / "flux2_transformer_part1_graph.mindir", "ascend", args.part1_device)
    t_arr = np.array([0.5], dtype=np.float16); g_arr = np.array([3.5], dtype=np.float16)
    feed0 = {"hidden_states": latents_packed, "encoder_hidden_states": enc, "timestep": t_arr,
             "guidance": g_arr, "img_ids": img_ids, "txt_ids": txt_ids}
    mid = p0.predict(_build_inputs(p0, feed0,
        ["hidden_states", "encoder_hidden_states", "timestep", "guidance", "img_ids", "txt_ids"]))[0].get_data_to_numpy()
    feed1 = {"hidden_mid": mid, "timestep": t_arr, "guidance": g_arr, "img_ids": img_ids, "txt_ids": txt_ids}
    ms_noise = p1.predict(_build_inputs(p1, feed1,
        ["hidden_mid", "timestep", "guidance", "img_ids", "txt_ids"]))[0].get_data_to_numpy()

    print("\n--- transformer split parity (1 step, t=0.5, guidance=3.5) ---")
    _stats("noise_pred", hf_noise, ms_noise)

    # ---- VAE decode parity ----
    print("\n[VAE] loading HF AutoencoderKLFlux2 + MindIR VAE ...")
    from diffusers import AutoencoderKLFlux2
    vae = AutoencoderKLFlux2.from_pretrained(md / "vae", torch_dtype=dtype, low_cpu_mem_usage=True).eval()
    # build a fixed decoded latent (unpack + bn denorm + unpatchify) for both sides
    latent_dec = _unpatchify_latents(_unpack_latents_rowmajor(latents_packed, h_tok, w_tok))
    eps = float(vae.config.batch_norm_eps)
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(torch.float32).numpy()
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + eps).to(torch.float32).numpy()
    z = (latent_dec * bn_std + bn_mean).astype(np.float16)
    with torch.no_grad():
        hf_img = vae.decode(torch.from_numpy(z), return_dict=False)[0].numpy()
    vae_m = _build_model(mindir / "flux2_vae_decoder_graph.mindir", "ascend", args.part0_device)
    ms_img = vae_m.predict(_build_inputs(vae_m, {"latents": z}, ["latents"]))[0].get_data_to_numpy()
    print("\n--- VAE decode parity ---")
    _stats("vae image", hf_img, ms_img)

    print("\n" + "=" * 70)
    print("Parity complete. fp16 max_abs <~1e-2 on noise_pred/image is expected.")
    print("=" * 70)


if __name__ == "__main__":
    main()
