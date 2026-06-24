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
"""End-to-end inference for SDXL refiner 0.9 on MindSpore Lite (Ascend).

The refiner is the SDXL two-stage pipeline's second stage. In production it
runs after the base UNet has denoised the latents to a high timestep; here,
for a self-contained example, it is demonstrated standalone by running its
own denoising loop from pure noise (the same EulerDiscreteScheduler as the
base, with the refiner's full timestep range). The 2-stage (base->refiner)
usage is documented in the README.

Loads three MindIR sub-models (CLIP-G text encoder on dev1, UNet + VAE on
dev0), encodes the prompt, runs the refiner UNet denoising loop
(``EulerDiscreteScheduler`` on CPU, UNet on Ascend with classifier-free
guidance), and decodes the latent to an image (VAE on Ascend). The only torch
usage is the (numpy-backed) diffusers scheduler and the tokenizer; no torch
tensor ever touches the Ascend models.

Component split on the 300I Duo: text encoder -> dev1, UNet + VAE -> dev0.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mindspore_lite as mslite
    import torch
    from transformers import AutoTokenizer
    from diffusers import EulerDiscreteScheduler
except ImportError as exc:  # pragma: no cover
    print(f"Error: missing dependency: {exc}")
    print("Install: pip install mindspore-lite transformers diffusers torch")
    sys.exit(1)

from PIL import Image


# ---------------------------------------------------------------------------
# mslite helpers.
# ---------------------------------------------------------------------------


_DTYPE_MAP = {
    np.dtype("float16"): mslite.DataType.FLOAT16,
    np.dtype("float32"): mslite.DataType.FLOAT32,
    np.dtype("int64"): mslite.DataType.INT64,
    np.dtype("int32"): mslite.DataType.INT32,
}


def _np_to_input(tensor_info, array):
    """Cast a numpy array to the dtype expected by a model input tensor."""
    target = _DTYPE_MAP.get(np.dtype(tensor_info.dtype))
    return array.astype(target) if target is not None else array


def _build_model(mindir_path, device_id):
    """Build an mslite Model from a ``*_graph.mindir`` on a given Ascend device."""
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = int(device_id)
    context.ascend.precision_mode = "force_fp16"
    model = mslite.Model()
    model.build_from_file(str(mindir_path), mslite.ModelType.MINDIR, context)
    return model


def _run_model(model, feed_dict, preferred_order):
    """Run a model by matching input names, falling back to a preferred order.

    Casts each feed array to the dtype the model input tensor expects.
    """
    inputs = model.get_inputs()
    if all(getattr(t, "name", None) in feed_dict for t in inputs):
        tensors = [mslite.Tensor(_np_to_input(t, feed_dict[t.name])) for t in inputs]
    else:
        tensors = [mslite.Tensor(_np_to_input(t, feed_dict[k]))
                   for t, k in zip(inputs, preferred_order)]
    outputs = model.predict(tensors)
    return [o.get_data_to_numpy() for o in outputs]


# ---------------------------------------------------------------------------
# Inference.
# ---------------------------------------------------------------------------


class SdxlRefinerInferencer:
    """SDXL refiner 0.9 standalone denoising on MindSpore Lite.

    Self-contained variant: the refiner runs its own EulerDiscreteScheduler
    loop from seeded noise (mirrors the base UNet), instead of consuming
    base latents. The refiner's CLIP-G conditioning (encoder_hidden_states
    1280-dim, time_ids 5-tuple with aesthetic_score) is honoured.
    """

    def __init__(self, mindir_dir, model_dir, unet_device=0, vae_device=0,
                 text_device=1, height=1024, width=1024,
                 aesthetic_score=6.0, negative_aesthetic_score=2.5):
        """Load sub-models, tokenizer and scheduler; record image geometry."""
        mindir_dir = Path(mindir_dir)
        self.height = int(height)
        self.width = int(width)
        self.latent_h = self.height // 8
        self.latent_w = self.width // 8

        print(f"Loading text encoder 2 MindIR (dev{text_device}) ...")
        self.text_encoder_2 = _build_model(
            mindir_dir / "sdxl_text_encoder_2_graph.mindir", text_device)
        print(f"Loading refiner UNet MindIR (dev{unet_device}) ...")
        self.unet = _build_model(mindir_dir / "sdxl_unet_graph.mindir", unet_device)
        print(f"Loading VAE MindIR (dev{vae_device}) ...")
        self.vae = _build_model(
            mindir_dir / "sdxl_vae_decoder_graph.mindir", vae_device)

        # The refiner ships tokenizer_2 only (CLIP-G); its tokenizer dir is
        # named ``tokenizer_2`` in the diffusers layout, matching the base.
        self.tokenizer_2 = AutoTokenizer.from_pretrained(
            Path(model_dir) / "tokenizer_2")
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            model_dir, subfolder="scheduler")

        from json import load as _load
        with open(Path(model_dir) / "vae" / "config.json") as fh:
            vae_cfg = _load(fh)
        self.scaling_factor = float(vae_cfg.get("scaling_factor", 0.13025))

        # Refiner micro-conditioning (requires_aesthetics_score=True):
        # (orig_h, orig_w, crop_top, crop_left, aesthetic_score) -- 5 values.
        # Positive and negative use their own aesthetic_score.
        self.add_time_ids_pos = np.array(
            [[float(self.height), float(self.width), 0.0, 0.0,
              float(aesthetic_score)]], dtype=np.float32)
        self.add_time_ids_neg = np.array(
            [[float(self.height), float(self.width), 0.0, 0.0,
              float(negative_aesthetic_score)]], dtype=np.float32)

    def _encode_prompt(self, prompt):
        """Tokenize + run CLIP-G -> (embeds[1,77,1280], pooled[1,1280]).

        The refiner uses CLIP-G only (penultimate hidden = 1280), NOT the
        base's 2048 concat.
        """
        t0 = time.perf_counter()
        ids_2 = self.tokenizer_2(
            [prompt], padding="max_length", max_length=77, truncation=True,
            return_tensors="np").input_ids.astype(np.int64)
        enc2_out = _run_model(self.text_encoder_2, {"input_ids": ids_2},
                              ["input_ids"])
        embeds, pooled = enc2_out[0], enc2_out[1]
        embeds = embeds.astype(np.float32)
        pooled = pooled.astype(np.float32)
        return embeds, pooled, (time.perf_counter() - t0) * 1000.0

    def _unet_forward(self, sample, timestep, embeds, pooled, time_ids):
        """Run one refiner UNet forward (returns noise_pred as numpy float32)."""
        feed = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": embeds,
            "text_embeds": pooled,
            "time_ids": np.broadcast_to(
                time_ids, (sample.shape[0], 5)).astype(np.float32),
        }
        noise_pred = _run_model(
            self.unet, feed,
            ["sample", "timestep", "encoder_hidden_states",
             "text_embeds", "time_ids"])[0]
        return noise_pred.astype(np.float32)

    def _denoise(self, latents, prompt_embeds, prompt_pooled,
                 negative_embeds, negative_pooled, num_steps, guidance_scale):
        """Run the CFG Euler denoising loop, returning denoised latents."""
        self.scheduler.set_timesteps(num_steps, device="cpu")
        self.scheduler.set_begin_index(0)
        timesteps = self.scheduler.timesteps

        # CFG: stack [uncond, cond] and run the UNet once per step. Use the
        # negative time_ids (with negative_aesthetic_score) for the uncond
        # branch and the positive time_ids for the cond branch.
        latents = np.concatenate([latents, latents], axis=0)
        embeds = np.concatenate([negative_embeds, prompt_embeds], axis=0)
        pooled = np.concatenate([negative_pooled, prompt_pooled], axis=0)
        time_ids = np.concatenate(
            [self.add_time_ids_neg, self.add_time_ids_pos], axis=0)

        latents_t = torch.from_numpy(latents)
        total_unet_ms = 0.0
        for i, t in enumerate(timesteps):
            latent_model_input = self.scheduler.scale_model_input(
                latents_t, t)
            sample = latent_model_input.numpy().astype(np.float32)
            timestep = np.array([float(t)], dtype=np.float32)
            ti = time.perf_counter()
            noise_pred = self._unet_forward(
                sample, timestep, embeds, pooled, time_ids)
            total_unet_ms += (time.perf_counter() - ti) * 1000.0

            noise_uncond, noise_cond = noise_pred[0:1], noise_pred[1:2]
            noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            latents_t = self.scheduler.step(
                torch.from_numpy(noise), t, latents_t[0:1],
                return_dict=False)[0]
            latents_t = torch.cat([latents_t, latents_t], dim=0)
        return latents_t[0:1].numpy(), total_unet_ms, len(timesteps)

    def infer(self, prompt, negative_prompt="", seed=0, num_inference_steps=30,
              guidance_scale=5.0):
        """Run the full refiner pipeline. Returns (image_uint8, timing_dict)."""
        t_start = time.perf_counter()
        prompt_embeds, prompt_pooled, t_text = self._encode_prompt(prompt)
        negative_embeds, negative_pooled, _ = self._encode_prompt(
            negative_prompt or "")

        # Initial noise (numpy, fixed seed). EulerDiscreteScheduler multiplies
        # the init noise by init_noise_sigma = max(sigma), applied below.
        rng = np.random.RandomState(seed)
        latents = rng.standard_normal(
            (1, 4, self.latent_h, self.latent_w)).astype(np.float32)
        init_noise_sigma = float(self.scheduler.init_noise_sigma)
        latents = latents * init_noise_sigma

        latents, total_unet_ms, steps = self._denoise(
            latents, prompt_embeds, prompt_pooled, negative_embeds,
            negative_pooled, num_inference_steps, guidance_scale)

        # Decode: unscale latents, then VAE.
        latents = latents / self.scaling_factor
        latents = latents.astype(np.float16)
        td = time.perf_counter()
        image = _run_model(self.vae, {"latents": latents}, ["latents"])[0]
        t_vae = (time.perf_counter() - td) * 1000.0

        image = (image / 2 + 0.5).clip(0, 1)
        image = (image[0].transpose(1, 2, 0) * 255).round().astype(np.uint8)
        t_e2e = (time.perf_counter() - t_start) * 1000.0
        avg_step = total_unet_ms / max(1, steps)
        timing = {
            "text_encode": t_text, "unet_total": total_unet_ms,
            "unet_avg_step": avg_step, "steps": steps,
            "vae": t_vae, "e2e": t_e2e,
        }
        return image, timing


def main():
    """Parse arguments and run SDXL refiner 0.9 inference on Ascend."""
    parser = argparse.ArgumentParser(
        description="SDXL refiner 0.9 MindSpore Lite inference")
    parser.add_argument("--mindir-dir", required=True,
                        help="dir with the 3 *_graph.mindir")
    parser.add_argument("--model-dir", required=True,
                        help="stable-diffusion-xl-refiner-0.9 diffusers weights dir")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt",
                        default="lowres, blurry, worst quality, low quality")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--aesthetic-score", type=float, default=6.0)
    parser.add_argument("--negative-aesthetic-score", type=float, default=2.5)
    parser.add_argument("--unet-device", type=int, default=0)
    parser.add_argument("--vae-device", type=int, default=0)
    parser.add_argument("--text-device", type=int, default=1)
    parser.add_argument("--output", default="./sdxl_refiner_output.png")
    args = parser.parse_args()

    inferencer = SdxlRefinerInferencer(
        args.mindir_dir, args.model_dir, unet_device=args.unet_device,
        vae_device=args.vae_device, text_device=args.text_device,
        height=args.height, width=args.width,
        aesthetic_score=args.aesthetic_score,
        negative_aesthetic_score=args.negative_aesthetic_score)
    image, timing = inferencer.infer(
        args.prompt, negative_prompt=args.negative_prompt, seed=args.seed,
        num_inference_steps=args.steps, guidance_scale=args.guidance)

    Image.fromarray(image).save(args.output)
    print("\n--- Performance ---")
    print(f"  Text encode (CLIP-G, dev{args.text_device}): "
          f"{timing['text_encode']:.2f} ms")
    print(f"  Refiner UNet total ({timing['steps']} steps, CFG x2): "
          f"{timing['unet_total']:.2f} ms")
    print(f"  Refiner UNet avg/step:               {timing['unet_avg_step']:.2f} ms")
    print(f"  VAE decode (dev{args.vae_device}): {timing['vae']:.2f} ms")
    print(f"  End-to-end:                  {timing['e2e']:.2f} ms")
    print(f"  Saved image -> {args.output}")


if __name__ == "__main__":
    main()
