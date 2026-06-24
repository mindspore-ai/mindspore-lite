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
"""End-to-end accuracy alignment for SDXL base 1.0: HF diffusers vs MindSpore Lite.

Generates the SAME image twice from identical inputs (prompt, fixed initial
latents, scheduler settings):

  1. HF ``StableDiffusionXLPipeline`` on CPU (ground truth, float32).
  2. MindSpore Lite pipeline on Ascend (infer_sdxl_base_1_0_mslite internals,
     driven here so the shared seeded latents are fed to both pipelines).

then compares the two images (max abs / mean abs / PSNR). The shared initial
latents are produced once with a seeded torch generator, so the only source of
difference is Ascend vs CPU numerics.

NOTE: the HF CPU baseline for a full 1024x1024 / 30-step run is slow; pass a
smaller ``--steps`` for a faster (still end-to-end) comparison. Both pipelines
always use the same settings.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from infer_sdxl_base_1_0_mslite import (
    SdxlInferencer,
    _run_model,
)


def _make_latents(shape, seed, path):
    """Create seeded latents, save to npy, and return the torch tensor for HF."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    latents = torch.randn(shape, generator=gen, dtype=torch.float32)
    np.save(path, latents.numpy().astype(np.float32))
    return latents


def _hf_baseline(model_dir, prompt, negative_prompt, latents, height, width,
                 num_steps, guidance_scale):
    """Run the HF StableDiffusionXLPipeline on CPU; return image in [0,1]."""
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_dir, torch_dtype=torch.float32)
    out = pipe(
        prompt=prompt, negative_prompt=negative_prompt or None,
        height=height, width=width, num_inference_steps=num_steps,
        guidance_scale=guidance_scale, latents=latents, output_type="np",
    )
    return np.asarray(out.images[0])  # (H, W, 3) in [0,1]


def _mslite_image(inferencer, prompt, negative_prompt, latents_np,
                  num_steps, guidance_scale):
    """Drive the MSLite pipeline internals with the shared seeded latents.

    Mirrors ``SdxlInferencer.infer`` but feeds ``latents_np`` (the same tensor
    the HF baseline used) instead of freshly sampled noise, so both pipelines
    denoise from byte-identical initial latents.
    """
    prompt_embeds, prompt_pooled, _ = inferencer._encode_prompt(prompt)  # noqa: SLF001
    neg_embeds, neg_pooled, _ = inferencer._encode_prompt(negative_prompt or "")

    init_noise_sigma = float(inferencer.scheduler.init_noise_sigma)
    latents = latents_np * init_noise_sigma

    latents, _, _ = inferencer._denoise(  # noqa: SLF001
        latents, prompt_embeds, prompt_pooled, neg_embeds, neg_pooled,
        num_steps, guidance_scale)

    latents = latents / inferencer.scaling_factor  # noqa: SLF001
    latents = latents.astype(np.float16)
    t0 = time.perf_counter()
    image = _run_model(inferencer.vae, {"latents": latents}, ["latents"])[0]
    t_vae = (time.perf_counter() - t0) * 1000.0
    print(f"[align] MSLite VAE decode: {t_vae:.2f} ms")

    image = (image / 2 + 0.5).clip(0, 1)
    return (image[0].transpose(1, 2, 0) * 255).round().astype(np.uint8)


def _compare(a, b):
    """Return max_abs, mean_abs, PSNR between two [0,1] images."""
    diff = np.abs(a - b)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))
    return max_abs, mean_abs, psnr


def main():
    """Run HF and MSLite pipelines on shared latents and report the gap."""
    parser = argparse.ArgumentParser(
        description="SDXL base 1.0 HF vs MSLite alignment")
    parser.add_argument("--mindir-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt",
                        default="A cat holding a sign that says hello world, "
                                "highly detailed, 4k")
    parser.add_argument("--negative-prompt",
                        default="lowres, blurry, worst quality, low quality")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-device", type=int, default=1)
    parser.add_argument("--unet-device", type=int, default=0)
    parser.add_argument("--vae-device", type=int, default=0)
    args = parser.parse_args()

    shape = (1, 4, args.height // 8, args.width // 8)
    latents_npy = Path(args.mindir_dir) / "_align_latents.npy"
    latents = _make_latents(shape, args.seed, str(latents_npy))

    print("[align] running HF baseline on CPU ...")
    hf = _hf_baseline(args.model_dir, args.prompt, args.negative_prompt, latents,
                      args.height, args.width, args.steps, args.guidance)
    print(f"[align] HF image: {hf.shape}, "
          f"range [{hf.min():.3f}, {hf.max():.3f}]")

    print("[align] running MSLite pipeline on Ascend ...")
    inferencer = SdxlInferencer(
        args.mindir_dir, args.model_dir, unet_device=args.unet_device,
        vae_device=args.vae_device, text_device=args.text_device,
        height=args.height, width=args.width)
    shared_latents = np.load(str(latents_npy)).astype(np.float32)
    ms_uint8 = _mslite_image(
        inferencer, args.prompt, args.negative_prompt, shared_latents,
        args.steps, args.guidance)
    ms = ms_uint8.astype(np.float32) / 255.0
    print(f"[align] MSLite image: {ms.shape}, "
          f"range [{ms.min():.3f}, {ms.max():.3f}]")

    max_abs, mean_abs, psnr = _compare(hf, ms)
    print("\n--- Alignment (HF vs MSLite) ---")
    print(f"  image shape    : {hf.shape}")
    print(f"  max  abs error : {max_abs:.6f}")
    print(f"  mean abs error : {mean_abs:.6f}")
    print(f"  PSNR (dB)      : {psnr:.2f}")


if __name__ == "__main__":
    main()
