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
"""End-to-end text-to-image inference for SDXL base 1.0 on MindSpore Lite (Ascend).

Loads four MindIR sub-models (two CLIP text encoders on dev1, UNet + VAE on
dev0), encodes the prompt, runs the UNet denoising loop
(``EulerDiscreteScheduler`` on CPU, UNet on Ascend with classifier-free
guidance), and decodes the latent to an image (VAE on Ascend). The only torch
usage is the (numpy-backed) diffusers scheduler and the tokenizers; no torch
tensor ever touches the Ascend models.

Component split on the 300I Duo: text encoders -> dev1, UNet + VAE -> dev0.
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


class SdxlInferencer:
    """SDXL base 1.0 text-to-image on MindSpore Lite (UNet+VAE / text encoders)."""

    def __init__(self, mindir_dir, model_dir, unet_device=0, vae_device=0,
                 text_device=1, height=1024, width=1024):
        """Load sub-models, tokenizers and scheduler; record image geometry."""
        mindir_dir = Path(mindir_dir)
        self.height = int(height)
        self.width = int(width)
        self.latent_h = self.height // 8
        self.latent_w = self.width // 8

        print(f"Loading text encoder 1 MindIR (dev{text_device}) ...")
        self.text_encoder_1 = _build_model(
            mindir_dir / "sdxl_text_encoder_graph.mindir", text_device)
        print(f"Loading text encoder 2 MindIR (dev{text_device}) ...")
        self.text_encoder_2 = _build_model(
            mindir_dir / "sdxl_text_encoder_2_graph.mindir", text_device)
        print(f"Loading UNet MindIR (dev{unet_device}) ...")
        self.unet = _build_model(mindir_dir / "sdxl_unet_graph.mindir", unet_device)
        print(f"Loading VAE MindIR (dev{vae_device}) ...")
        self.vae = _build_model(
            mindir_dir / "sdxl_vae_decoder_graph.mindir", vae_device)

        self.tokenizer = AutoTokenizer.from_pretrained(Path(model_dir) / "tokenizer")
        self.tokenizer_2 = AutoTokenizer.from_pretrained(
            Path(model_dir) / "tokenizer_2")
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            model_dir, subfolder="scheduler")

        from json import load as _load
        with open(Path(model_dir) / "vae" / "config.json") as fh:
            vae_cfg = _load(fh)
        self.scaling_factor = float(vae_cfg.get("scaling_factor", 0.13025))

        # SDXL micro-conditioning: (orig_h, orig_w, crop_top, crop_left,
        # target_h, target_w).
        self.add_time_ids = np.array(
            [[float(self.height), float(self.width), 0.0, 0.0,
              float(self.height), float(self.width)]], dtype=np.float32)

    def _encode_prompt(self, prompt):
        """Tokenize + run both CLIP encoders -> (embeds[1,77,2048], pooled)."""
        t0 = time.perf_counter()
        ids_1 = self.tokenizer(
            [prompt], padding="max_length", max_length=77, truncation=True,
            return_tensors="np").input_ids.astype(np.int64)
        ids_2 = self.tokenizer_2(
            [prompt], padding="max_length", max_length=77, truncation=True,
            return_tensors="np").input_ids.astype(np.int64)

        enc1 = _run_model(self.text_encoder_1, {"input_ids": ids_1},
                          ["input_ids"])[0]
        enc2_out = _run_model(self.text_encoder_2, {"input_ids": ids_2},
                              ["input_ids"])
        enc2, pooled = enc2_out[0], enc2_out[1]

        embeds = np.concatenate([enc1, enc2], axis=-1).astype(np.float32)
        pooled = pooled.astype(np.float32)
        return embeds, pooled, (time.perf_counter() - t0) * 1000.0

    def _unet_forward(self, sample, timestep, embeds, pooled):
        """Run one UNet forward (returns noise_pred as numpy float32)."""
        feed = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": embeds,
            "text_embeds": pooled,
            "time_ids": np.broadcast_to(
                self.add_time_ids, (sample.shape[0], 6)).astype(np.float32),
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

        # CFG: stack [uncond, cond] and run the UNet once per step.
        latents = np.concatenate([latents, latents], axis=0)
        embeds = np.concatenate([negative_embeds, prompt_embeds], axis=0)
        pooled = np.concatenate([negative_pooled, prompt_pooled], axis=0)

        latents_t = torch.from_numpy(latents)
        total_unet_ms = 0.0
        for i, t in enumerate(timesteps):
            latent_model_input = self.scheduler.scale_model_input(
                latents_t, t)
            sample = latent_model_input.numpy().astype(np.float32)
            timestep = np.array([float(t)], dtype=np.float32)
            ti = time.perf_counter()
            noise_pred = self._unet_forward(sample, timestep, embeds, pooled)
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
        """Run the full SDXL pipeline. Returns (image_uint8, timing_dict)."""
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
    """Parse arguments and run SDXL base 1.0 text-to-image inference."""
    parser = argparse.ArgumentParser(
        description="SDXL base 1.0 MindSpore Lite inference")
    parser.add_argument("--mindir-dir", required=True,
                        help="dir with the 4 *_graph.mindir")
    parser.add_argument("--model-dir", required=True,
                        help="stable-diffusion-xl-base-1.0 diffusers weights dir")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt",
                        default="lowres, blurry, worst quality, low quality")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--unet-device", type=int, default=0)
    parser.add_argument("--vae-device", type=int, default=0)
    parser.add_argument("--text-device", type=int, default=1)
    parser.add_argument("--output", default="./sdxl_output.png")
    args = parser.parse_args()

    inferencer = SdxlInferencer(
        args.mindir_dir, args.model_dir, unet_device=args.unet_device,
        vae_device=args.vae_device, text_device=args.text_device,
        height=args.height, width=args.width)
    image, timing = inferencer.infer(
        args.prompt, negative_prompt=args.negative_prompt, seed=args.seed,
        num_inference_steps=args.steps, guidance_scale=args.guidance)

    Image.fromarray(image).save(args.output)
    print("\n--- Performance ---")
    print(f"  Text encode (CLIP-L+CLIP-G, dev{args.text_device}): "
          f"{timing['text_encode']:.2f} ms")
    print(f"  UNet total ({timing['steps']} steps, CFG x2): "
          f"{timing['unet_total']:.2f} ms")
    print(f"  UNet avg/step:               {timing['unet_avg_step']:.2f} ms")
    print(f"  VAE decode (dev{args.vae_device}): {timing['vae']:.2f} ms")
    print(f"  End-to-end:                  {timing['e2e']:.2f} ms")
    print(f"  Saved image -> {args.output}")


if __name__ == "__main__":
    main()
