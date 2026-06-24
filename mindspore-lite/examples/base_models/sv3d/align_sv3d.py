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
"""End-to-end accuracy alignment for SV3D: HF diffusers vs MindSpore Lite.

Generates the SAME video twice from identical inputs (conditioning image, fixed
initial latents, scheduler/guidance settings):

  1. HF ``StableVideoDiffusionPipeline`` on CPU (ground truth, float32).
  2. MindSpore Lite pipeline on Ascend (the internals of
     ``infer_sv3d_mslite.SvdInferencer``, driven here so the shared
     seeded latents are fed to both pipelines).

then compares the two videos frame-wise (max abs / mean abs / PSNR per frame,
plus aggregate). The shared initial latents are produced once with a seeded
torch generator, so the only source of difference is Ascend vs CPU numerics.

NOTE: the HF CPU baseline for a full 576x1024 / 25-frame / 25-step run is very
slow; pass a smaller ``--steps`` for a faster (still end-to-end) comparison.
Both pipelines always use the same settings.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from infer_sv3d_mslite import SvdInferencer


def _make_latents(shape, seed, path):
    """Create seeded latents, save to npy, return the torch tensor for HF.

    ``shape`` is ``(1, num_frames, 4, latent_h, latent_w)`` (the SVD latent
    layout). The HF pipeline takes a [B,F,C,H,W] ``latents`` argument directly.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    latents = torch.randn(shape, generator=gen, dtype=torch.float32)
    np.save(path, latents.numpy().astype(np.float32))
    return latents


def _hf_baseline(model_dir, image_pil, latents, height, width, num_frames,
                 num_steps, fps, motion_bucket_id, noise_aug_strength,
                 min_gs, max_gs):
    """Run the HF StableVideoDiffusionPipeline on CPU; return frames in [0,1].

    Returns an array of shape ``(num_frames, H, W, 3)`` in [0,1].
    """
    from diffusers import StableVideoDiffusionPipeline
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_dir, torch_dtype=torch.float32)
    out = pipe(
        image=image_pil, height=height, width=width,
        num_frames=num_frames, num_inference_steps=num_steps, fps=fps,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        min_guidance_scale=min_gs, max_guidance_scale=max_gs,
        latents=latents, decode_chunk_size=8, output_type="np",
    )
    vid = np.asarray(out.frames[0])  # (F, H, W, 3) in [0,1]
    return vid


def _mslite_video(inferencer, image_pil, vae_torch, latents_np, num_steps,
                  fps, motion_bucket_id, noise_aug_strength, min_gs, max_gs):
    """Drive the MSLite pipeline internals with the shared seeded latents.

    Mirrors ``SvdInferencer.infer`` but feeds ``latents_np`` (the same tensor
    the HF baseline used) instead of freshly sampled noise, so both pipelines
    denoise from byte-identical initial latents. Returns frames uint8
    [F,H,W,3].
    """
    # Reuse the public infer path but inject the shared latents by
    # monkeypatching the RandomState draw: simplest is to call the internal
    # steps directly with latents_np * init_noise_sigma.
    init_noise_sigma = float(inferencer.scheduler.init_noise_sigma)
    latents = latents_np * init_noise_sigma

    image_embeds, _ = inferencer._encode_image(image_pil)  # noqa: SLF001
    image_embeds_cfg = np.concatenate(
        [np.zeros_like(image_embeds[None]), image_embeds[None]], axis=0)

    gen = torch.Generator()
    gen.manual_seed(0)  # noise_aug noise is not the alignment-dominant term
    image_latents = inferencer._encode_vae_image(  # noqa: SLF001
        image_pil, vae_torch, noise_aug_strength, gen)
    image_latents_frames = np.broadcast_to(
        image_latents[None],
        (1, inferencer.num_frames, 4, inferencer.latent_h,
         inferencer.latent_w)).astype(np.float32)

    add_ids = np.array(
        [[float(fps) - 1.0, float(motion_bucket_id),
          float(noise_aug_strength)]], dtype=np.float32)
    add_ids_cfg = np.concatenate([add_ids, add_ids], axis=0)

    latents_out, _, _ = inferencer._denoise(  # noqa: SLF001
        latents, image_latents_frames, image_embeds_cfg, add_ids_cfg,
        min_gs, max_gs, num_steps)

    frames, _ = inferencer._decode_frames(latents_out[0])  # noqa: SLF001
    frames = (frames / 2 + 0.5).clip(0, 1)
    frames = (frames * 255).round().astype(np.uint8)
    return frames.transpose(0, 2, 3, 1)  # NCHW -> NHWC


def _compare(a, b):
    """Return per-frame and aggregate (max_abs, mean_abs, PSNR) for [0,1] vid.

    ``a`` and ``b`` are [F,H,W,3] in [0,1].
    """
    diff = np.abs(a - b)
    per_max = diff.max(axis=(1, 2, 3))
    per_mean = diff.mean(axis=(1, 2, 3))
    mse = np.mean((a - b) ** 2, axis=(1, 2, 3))
    eps = 1e-12
    per_psnr = 10.0 * np.log10(1.0 / np.maximum(mse, eps))
    agg_mse = float(np.mean((a - b) ** 2))
    agg_psnr = float("inf") if agg_mse == 0 else float(
        10.0 * np.log10(1.0 / agg_mse))
    return {
        "per_frame_max": per_max, "per_frame_mean": per_mean,
        "per_frame_psnr": per_psnr,
        "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
        "psnr": agg_psnr,
    }


def main():
    """Run HF and MSLite SVD pipelines on shared latents and report the gap."""
    parser = argparse.ArgumentParser(
        description="SV3D HF vs MSLite alignment")
    parser.add_argument("--mindir-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True,
                        help="conditioning image path.")
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--motion-bucket-id", type=int, default=127)
    parser.add_argument("--noise-aug-strength", type=float, default=0.02)
    parser.add_argument("--min-guidance", type=float, default=1.0)
    parser.add_argument("--max-guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-device", type=int, default=1)
    parser.add_argument("--unet-device", type=int, default=0)
    parser.add_argument("--vae-device", type=int, default=0)
    args = parser.parse_args()

    from diffusers import AutoencoderKLTemporalDecoder
    from diffusers.utils import load_image

    latent_h, latent_w = args.height // 8, args.width // 8
    shape = (1, args.num_frames, 4, latent_h, latent_w)
    latents_npy = Path(args.mindir_dir) / "_align_latents.npy"
    latents = _make_latents(shape, args.seed, str(latents_npy))

    image_pil = load_image(args.image)
    vae_torch = AutoencoderKLTemporalDecoder.from_pretrained(
        args.model_dir, subfolder="vae", torch_dtype=torch.float32).eval()

    print("[align] running HF baseline on CPU ...")
    hf = _hf_baseline(
        args.model_dir, image_pil, latents, args.height, args.width,
        args.num_frames, args.steps, args.fps, args.motion_bucket_id,
        args.noise_aug_strength, args.min_guidance, args.max_guidance)
    print(f"[align] HF video: {hf.shape}, "
          f"range [{hf.min():.3f}, {hf.max():.3f}]")

    print("[align] running MSLite pipeline on Ascend ...")
    inferencer = SvdInferencer(
        args.mindir_dir, args.model_dir, image_device=args.image_device,
        unet_device=args.unet_device, vae_device=args.vae_device,
        height=args.height, width=args.width, num_frames=args.num_frames)
    shared_latents = np.load(str(latents_npy)).astype(np.float32)
    ms_uint8 = _mslite_video(
        inferencer, image_pil, vae_torch, shared_latents, args.steps,
        args.fps, args.motion_bucket_id, args.noise_aug_strength,
        args.min_guidance, args.max_guidance)
    ms = ms_uint8.astype(np.float32) / 255.0
    print(f"[align] MSLite video: {ms.shape}, "
          f"range [{ms.min():.3f}, {ms.max():.3f}]")

    cmp = _compare(hf, ms)
    print("\n--- Alignment (HF vs MSLite) ---")
    print(f"  video shape      : {hf.shape}")
    print(f"  aggregate max abs: {cmp['max_abs']:.6f}")
    print(f"  aggregate mean abs: {cmp['mean_abs']:.6f}")
    print(f"  aggregate PSNR (dB): {cmp['psnr']:.2f}")
    print(f"  per-frame max abs (min/mean/max): "
          f"{cmp['per_frame_max'].min():.6f} / "
          f"{cmp['per_frame_max'].mean():.6f} / "
          f"{cmp['per_frame_max'].max():.6f}")
    print(f"  per-frame mean abs (min/mean/max): "
          f"{cmp['per_frame_mean'].min():.6f} / "
          f"{cmp['per_frame_mean'].mean():.6f} / "
          f"{cmp['per_frame_mean'].max():.6f}")
    print(f"  per-frame PSNR (dB, min/mean/max): "
          f"{cmp['per_frame_psnr'].min():.2f} / "
          f"{cmp['per_frame_psnr'].mean():.2f} / "
          f"{cmp['per_frame_psnr'].max():.2f}")


if __name__ == "__main__":
    main()
