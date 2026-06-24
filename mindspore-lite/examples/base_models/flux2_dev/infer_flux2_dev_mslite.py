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
End-to-end text-to-image inference for FLUX.2-dev on MindSpore Lite (Ascend).

The FLUX.2-dev transformer is 64 GB (bf16), too large for one 300I Duo chip, so
it runs as two pipeline-parallel MindIR halves: part0 on dev0, part1 on dev1,
with the intermediate ``hidden_states`` copied through host each step. The VAE
decoder runs as a MindIR on Ascend. The Mistral3 text encoder (~24B, 48 GB) is
run once per prompt on CPU (it does not fit a single Ascend chip); the resulting
``encoder_hidden_states`` feed part0.

Pipeline: Mistral3(CPU) -> noise -> [part0(dev0) -> host -> part1(dev1)] x N ->
scheduler.step -> unpack + BN-denorm + unpatchify -> VAE(dev) -> image.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mindspore_lite as mslite
    import torch
    from transformers import AutoProcessor, Mistral3ForConditionalGeneration
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
except ImportError as exc:  # pragma: no cover
    print(f"Error: missing dependency: {exc}")
    sys.exit(1)

from PIL import Image


# ---------------------------------------------------------------------------
# mslite helpers.
# ---------------------------------------------------------------------------


def _build_model(path, device, device_id):
    ctx = mslite.Context()
    ctx.target = [device]
    if device == "ascend":
        ctx.ascend.device_id = int(device_id)
        ctx.ascend.precision_mode = "force_fp16"
    model = mslite.Model()
    model.build_from_file(path, mslite.ModelType.MINDIR, ctx)
    return model


def _build_inputs(model, feed_dict, preferred_order):
    inputs = model.get_inputs()
    if not inputs:
        return [mslite.Tensor(v) for v in feed_dict.values()]
    if all(getattr(t, "name", None) in feed_dict for t in inputs):
        return [mslite.Tensor(feed_dict[t.name]) for t in inputs]
    return [mslite.Tensor(feed_dict[k]) for k in preferred_order]


# ---------------------------------------------------------------------------
# numpy latent pack / unpack / unpatchify (mirror Flux2Pipeline).
# ---------------------------------------------------------------------------


def _pack_latents(latents):
    """[1,C,H,W] -> [1, H*W, C]."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).transpose(0, 2, 1)


def _unpack_latents_rowmajor(latents_packed, h_tokens, w_tokens):
    """[1, h*w, C] -> [1, C, h, w] (row-major ids <=> identity scatter)."""
    b, n, c = latents_packed.shape
    return latents_packed.reshape(b, h_tokens, w_tokens, c).transpose(0, 3, 1, 2)


def _unpatchify_latents(latents):
    """[1, 128, h, w] -> [1, 32, h*2, w*2] (reverse of _patchify_latents)."""
    b, c, h, w = latents.shape
    lat = latents.reshape(b, c // 4, 2, 2, h, w)
    lat = lat.transpose(0, 1, 4, 2, 5, 3)
    return lat.reshape(b, c // 4, h * 2, w * 2)


# ---------------------------------------------------------------------------
# Inference.
# ---------------------------------------------------------------------------


class Flux2Inferencer:
    """FLUX.2-dev text-to-image: pipeline-parallel transformer + CPU Mistral3."""

    def __init__(self, part0_path, part1_path, vae_path, model_dir,
                 device="ascend", dev0=0, dev1=1, vae_device=None,
                 height=1024, width=1024, seq_len=512):
        self.height = int(height)
        self.width = int(width)
        self.seq_len = int(seq_len)
        self.h_tok = self.height // 16
        self.w_tok = self.width // 16
        self.vae_device = int(vae_device) if vae_device is not None else int(dev0)

        print(f"Loading transformer part0 MindIR (dev{dev0}) ...")
        self.part0 = _build_model(part0_path, device, dev0)
        print(f"Loading transformer part1 MindIR (dev{dev1}) ...")
        self.part1 = _build_model(part1_path, device, dev1)
        print(f"Loading VAE MindIR (dev{self.vae_device}) ...")
        self.vae = _build_model(vae_path, device, self.vae_device)

        print("Loading Mistral3 text encoder + tokenizer on CPU (one-time, ~48GB) ...")
        dtype = torch.float16
        self.text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            Path(model_dir) / "text_encoder", torch_dtype=dtype, low_cpu_mem_usage=True).eval()
        self.tokenizer = AutoProcessor.from_pretrained(Path(model_dir) / "tokenizer")

        # VAE batch-norm stats for the denorm step (decode itself runs on Ascend).
        vae = AutoencoderKLFlux2.from_pretrained(
            Path(model_dir) / "vae", torch_dtype=dtype, low_cpu_mem_usage=True).eval()
        bn = vae.bn
        eps = float(vae.config.batch_norm_eps)
        self.bn_mean = bn.running_mean.view(1, -1, 1, 1).to(torch.float32).numpy()
        self.bn_std = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + eps).to(torch.float32).numpy()
        del vae

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")

    def _encode_prompt(self, prompt):
        """Run Mistral3 on CPU -> encoder_hidden_states + text_ids (numpy fp16)."""
        t0 = time.perf_counter()
        with torch.no_grad():
            embeds = Flux2Pipeline._get_mistral_3_small_prompt_embeds(
                text_encoder=self.text_encoder, tokenizer=self.tokenizer,
                prompt=prompt, dtype=self.text_encoder.dtype, device="cpu",
                max_sequence_length=self.seq_len)
            text_ids = Flux2Pipeline._prepare_text_ids(embeds).to(embeds.dtype)
        enc = embeds.numpy().astype(np.float16)
        txt_ids = text_ids.numpy().astype(np.float16)
        # drop batch dim for ids ([1, L, 4] -> [L, 4])
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        return enc, txt_ids, (time.perf_counter() - t0) * 1000.0

    def infer(self, prompt, seed=0, num_inference_steps=28, guidance=3.5):
        t_start = time.perf_counter()
        encoder_hidden_states, txt_ids, t_text = self._encode_prompt(prompt)
        seq_len = encoder_hidden_states.shape[1]

        # noise in patchified 128-ch space, then pack
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal((1, 128, self.h_tok, self.w_tok)).astype(np.float32)
        latents = _pack_latents(noise).astype(np.float16)  # [1, h*w, 128]

        # 4D position ids
        img_ids = self._latent_ids_np(self.h_tok, self.w_tok).astype(np.float16)

        # scheduler (same family as FLUX.1)
        image_seq_len = latents.shape[1]
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15))
        timesteps, _ = retrieve_timesteps(self.scheduler, num_inference_steps, "cpu", sigmas=None, mu=mu)
        self.scheduler.set_begin_index(0)

        latents_t = torch.from_numpy(latents)
        guidance_arr = np.array([guidance], dtype=np.float16)

        total_ms = 0.0
        for _, t in enumerate(timesteps):
            t_arr = np.array([float(t) / 1000.0], dtype=np.float16)
            feed0 = {"hidden_states": latents_t.numpy(), "encoder_hidden_states": encoder_hidden_states,
                     "timestep": t_arr, "guidance": guidance_arr, "img_ids": img_ids, "txt_ids": txt_ids}
            ti = time.perf_counter()
            mid = self.part0.predict(_build_inputs(
                self.part0, feed0,
                ["hidden_states", "encoder_hidden_states", "timestep", "guidance", "img_ids", "txt_ids"]))[0]
            hidden_mid = mid.get_data_to_numpy()  # dev0 -> host
            feed1 = {"hidden_mid": hidden_mid, "timestep": t_arr, "guidance": guidance_arr,
                     "img_ids": img_ids, "txt_ids": txt_ids}
            out = self.part1.predict(_build_inputs(
                self.part1, feed1,
                ["hidden_mid", "timestep", "guidance", "img_ids", "txt_ids"]))[0]
            noise_pred = out.get_data_to_numpy()
            total_ms += (time.perf_counter() - ti) * 1000.0
            latents_t = self.scheduler.step(
                torch.from_numpy(noise_pred[:, :latents_t.size(1)]), t, latents_t, return_dict=False)[0]

        # decode
        latents = latents_t.float().numpy()
        latents = _unpack_latents_rowmajor(latents, self.h_tok, self.w_tok)  # [1,128,h,w]
        latents = latents * self.bn_std + self.bn_mean
        latents = _unpatchify_latents(latents)  # [1,32,128,128]
        latents = latents.astype(np.float16)
        td = time.perf_counter()
        image = self.vae.predict(_build_inputs(
            self.vae, {"latents": latents}, ["latents"]))[0].get_data_to_numpy()
        t_vae = (time.perf_counter() - td) * 1000.0

        image = (image / 2 + 0.5).clip(0, 1)
        image = (image[0].transpose(1, 2, 0) * 255).round().astype(np.uint8)
        t_e2e = (time.perf_counter() - t_start) * 1000.0
        timing = {"text_encode": t_text, "transformer_total": total_ms,
                  "transformer_avg": total_ms / max(1, len(timesteps)), "steps": len(timesteps),
                  "vae": t_vae, "e2e": t_e2e}
        return image, timing

    @staticmethod
    def _latent_ids_np(h_tokens, w_tokens):
        """4D (T,H,W,L) latent ids -> [h*w, 4]."""
        t = np.arange(1)
        h = np.arange(h_tokens)
        w = np.arange(w_tokens)
        l = np.arange(1)
        return np.array(np.meshgrid(t, h, w, l, indexing="ij")).reshape(4, -1).T


def main():
    p = argparse.ArgumentParser(description="FLUX.2-dev MindSpore Lite inference (pipeline-parallel)")
    p.add_argument("--transformer-part0", required=True)
    p.add_argument("--transformer-part1", required=True)
    p.add_argument("--vae-model", required=True)
    p.add_argument("--model-dir", default="./FLUX.2-dev")
    p.add_argument("--prompt", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--device", default="ascend", choices=["ascend", "cpu"])
    p.add_argument("--part0-device", type=int, default=0)
    p.add_argument("--part1-device", type=int, default=1)
    p.add_argument("--vae-device", type=int, default=0)
    p.add_argument("--output", default="./flux2_output.png")
    args = p.parse_args()

    inf = Flux2Inferencer(
        args.transformer_part0, args.transformer_part1, args.vae_model, args.model_dir,
        device=args.device, dev0=args.part0_device, dev1=args.part1_device,
        vae_device=args.vae_device, height=args.height, width=args.width, seq_len=args.seq_len)
    image, timing = inf.infer(args.prompt, seed=args.seed,
                              num_inference_steps=args.steps, guidance=args.guidance)
    Image.fromarray(image).save(args.output)
    print("\n--- Performance ---")
    print(f"  Text encode (Mistral3, CPU): {timing['text_encode']:.2f} ms")
    print(f"  Transformer total (2-chip):   {timing['transformer_total']:.2f} ms")
    print(f"  Transformer avg/step:         {timing['transformer_avg']:.2f} ms ({timing['steps']} steps)")
    print(f"  VAE decode:                   {timing['vae']:.2f} ms")
    print(f"  End-to-end:                   {timing['e2e']:.2f} ms")
    print(f"  Saved image -> {args.output}")


if __name__ == "__main__":
    main()
