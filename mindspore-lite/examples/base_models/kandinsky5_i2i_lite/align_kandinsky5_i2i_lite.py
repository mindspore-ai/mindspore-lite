"""End-to-end accuracy alignment for Kandinsky-5.0-I2I-Lite: HF vs MindSpore Lite.

Generates the SAME image-to-image output twice from identical inputs (prompt,
source image, and fixed initial noise):

  1. HF diffusers ``Kandinsky5I2IPipeline`` on CPU (ground truth, float32).
  2. MindSpore Lite pipeline on Ascend (``infer_kandinsky5_i2i_lite_mslite``).

The shared initial noise is produced once with a seeded numpy generator and fed
to both pipelines, so the only source of difference is Ascend fp16 vs CPU fp32
numerics. Compares the two output images (max abs / mean abs / PSNR).

NOTE: the HF CPU baseline for a full 1024x1024 / 50-step I2I run is slow. Pass a
smaller ``--height/--width`` or fewer ``--num-inference-steps`` for a faster
end-to-end check. The exact noise-shape mapping between the HF pipeline's
``latents`` argument and the MSLite channels-last ``[1,1,H',W',16]`` noise is
handled in :func:`_hf_latents`; verify the reshape against the real checkpoint
at run time (flagged in the README FAQ).
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from diffusers import Kandinsky5I2IPipeline

from infer_kandinsky5_i2i_lite_mslite import Kandinsky5I2IInferencer

_VAE_SCALE_SPATIAL = 8
_LATENT_CHANNELS = 16


def _seeded_noise(latent_h, latent_w, seed, path):
    """Create seeded channels-last noise [1,1,H',W',16], save to npy."""
    rng = np.random.RandomState(seed)
    noise = rng.standard_normal((1, 1, latent_h, latent_w, _LATENT_CHANNELS)).astype(np.float32)
    np.save(path, noise)
    return noise


def _hf_latents(noise_cl, height, width):
    """Reshape MSLite channels-last noise to the HF pipeline's expected shape.

    The HF ``Kandinsky5I2IPipeline`` consumes initial noise in NCDHW-ish latent
    layout; the MSLite inferencer uses channels-last ``[1,1,H',W',16]``. This
    permutes back to ``[1,16,H',W']`` (H'=H//8). Verify against the checkpoint.
    """
    latent_h, latent_w = height // _VAE_SCALE_SPATIAL, width // _VAE_SCALE_SPATIAL
    return torch.from_numpy(noise_cl.reshape(latent_h, latent_w, _LATENT_CHANNELS).transpose(2, 0, 1)).unsqueeze(0)


def _hf_baseline(pipe, prompt, negative_prompt, source_image, noise, height, width,
                 num_steps, guidance_scale):
    """Run the HF Kandinsky5I2IPipeline on CPU, return image in [0,1] (H,W,3)."""
    out = pipe(
        prompt=prompt, negative_prompt=negative_prompt, image=source_image,
        height=height, width=width, num_inference_steps=num_steps,
        guidance_scale=guidance_scale, generator=torch.Generator().manual_seed(0),
    )
    img = np.asarray(out.images[0]).astype(np.float32) / 255.0
    return img


def _mslite_image(inferencer, prompt, negative_prompt, source_image, noise_npy, out_png):
    """Run the MSLite pipeline, reload the saved PNG, return image in [0,1]."""
    inferencer.generate(prompt, negative_prompt, source_image, out_png, latents_npy=noise_npy)
    return np.asarray(Image.open(out_png).convert("RGB")).astype(np.float32) / 255.0


def _compare(a, b):
    """Return max_abs, mean_abs, PSNR between two [0,1] images."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    diff = np.abs(a - b)
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))
    return float(diff.max()), float(diff.mean()), psnr


def main():
    """Run HF and MSLite I2I pipelines on shared noise + source image; report diff."""
    parser = argparse.ArgumentParser(description="Kandinsky-5.0-I2I-Lite HF vs MSLite alignment")
    parser.add_argument("--mindir-dir", required=True)
    parser.add_argument("--qwen-dir", required=True)
    parser.add_argument("--clip-dir", required=True)
    parser.add_argument("--vae-dir", required=True)
    parser.add_argument("--k5-model", required=True, help="K5 I2I model dir (for HF pipeline)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    latent_h, latent_w = args.height // _VAE_SCALE_SPATIAL, args.width // _VAE_SCALE_SPATIAL
    noise_npy = str(Path(args.mindir_dir) / "_align_noise.npy")
    noise = _seeded_noise(latent_h, latent_w, args.seed, noise_npy)
    source_image = Image.open(args.source_image).convert("RGB")

    print("[align] running HF Kandinsky5I2IPipeline on CPU ...")
    pipe = Kandinsky5I2IPipeline.from_pretrained(args.k5_model, torch_dtype=torch.float32)
    hf = _hf_baseline(pipe, args.prompt, args.negative_prompt, source_image, noise,
                      args.height, args.width, args.num_inference_steps, args.guidance_scale)
    print(f"[align] HF image: shape={hf.shape}, range [{hf.min():.3f}, {hf.max():.3f}]")

    print("[align] running MSLite pipeline on Ascend ...")
    inferencer = Kandinsky5I2IInferencer(
        args.mindir_dir, args.qwen_dir, args.clip_dir, args.vae_dir,
        height=args.height, width=args.width, num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale)
    out_png = str(Path(args.mindir_dir) / "_align_mslite.png")
    ms = _mslite_image(inferencer, args.prompt, args.negative_prompt, source_image, noise_npy, out_png)
    print(f"[align] MSLite image: shape={ms.shape}, range [{ms.min():.3f}, {ms.max():.3f}]")

    max_abs, mean_abs, psnr = _compare(hf, ms)
    print("\n--- Alignment (HF vs MSLite) ---")
    print(f"  max  abs error : {max_abs:.6f}")
    print(f"  mean abs error : {mean_abs:.6f}")
    print(f"  PSNR (dB)      : {psnr:.2f}")


if __name__ == "__main__":
    main()
