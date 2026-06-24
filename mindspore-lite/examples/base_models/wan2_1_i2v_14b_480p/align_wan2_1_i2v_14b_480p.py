"""End-to-end accuracy alignment for Wan2.1-I2V-14B-480P: HF diffusers vs MindSpore Lite.

Generates the SAME video twice from identical inputs (prompt, conditioning
image, fixed initial latents, scheduler settings):

  1. HF ``WanImageToVideoPipeline`` on CPU (ground truth, float32). The pipeline
     runs with the Wan2.1 schedule (``expand_timesteps=False``): scalar timestep
     and 36-channel channel-wise image conditioning.
  2. MindSpore Lite pipeline on Ascend (infer_wan2_1_i2v_14b_480p_mslite).

then compares the two videos frame-by-frame (max abs / mean abs / PSNR). The
shared initial latents are produced once with a seeded torch generator and fed
to both pipelines, so the only source of difference is the Ascend vs CPU
numerics. The conditioning image is also shared.

NOTE: the HF CPU baseline for a full 480x832x81 / 50-step run is slow; pass a
smaller ``--num-frames`` / ``--num-inference-steps`` for a faster (still
end-to-end) comparison. Both pipelines always use the same settings.
"""

import argparse
from pathlib import Path

import numpy as np

import torch
from diffusers import WanImageToVideoPipeline

from infer_wan2_1_i2v_14b_480p_mslite import WanI2VInferencer


def _make_latents(shape, seed, path):
    """Create seeded latents, save to npy, and return the torch tensor for HF."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    latents = torch.randn(shape, generator=gen, dtype=torch.float32)
    np.save(path, latents.numpy().astype(np.float32))
    return latents


def _hf_baseline(model_dir, image, prompt, negative_prompt, latents, height, width,
                 num_frames, num_steps, guidance_scale, max_seq_len):
    """Run the HF WanImageToVideoPipeline (Wan2.1 schedule) on CPU.

    Returns video frames in [0,1] with shape (F, H, W, 3).
    """
    pipe = WanImageToVideoPipeline.from_pretrained(model_dir, torch_dtype=torch.float32)
    # Wan2.1 I2V uses the scalar-timestep schedule (expand_timesteps=False);
    # force it off explicitly for clarity (it is normally set from config.json).
    pipe.config.expand_timesteps = False
    out = pipe(
        image=image, prompt=prompt, negative_prompt=negative_prompt,
        height=height, width=width, num_frames=num_frames,
        num_inference_steps=num_steps, guidance_scale=guidance_scale,
        latents=latents, max_sequence_length=max_seq_len, output_type="np",
    )
    return np.asarray(out.frames[0])  # (F, H, W, 3) in [0,1]


def _mslite_video(inferencer, prompt, image, negative_prompt, latents_npy):
    """Run the MSLite pipeline and return frames in [0,1] (F, H, W, 3)."""
    _, video = inferencer.generate(prompt, image, negative_prompt, "_align_mslite.mp4",
                                   latents_npy=latents_npy)
    frames = ((video[0].transpose(1, 2, 3, 0) / 2 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    return frames.astype(np.float32) / 255.0


def _compare(a, b):
    """Return max_abs, mean_abs, mean PSNR between two [0,1] frame stacks."""
    diff = np.abs(a - b)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    mse = np.mean((a - b) ** 2)
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))
    return max_abs, mean_abs, psnr


def _load_image(path):
    """Load an image as a PIL.Image (RGB)."""
    from PIL import Image
    return Image.open(path).convert("RGB")


def main():
    """Run HF and MSLite pipelines on shared latents+image and report the difference."""
    parser = argparse.ArgumentParser(description="Wan2.1-I2V-14B-480P HF vs MSLite alignment")
    parser.add_argument("--mindir-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True, help="conditioning image (first frame)")
    parser.add_argument("--prompt", default="A cat walking on a beach, cinematic, 4k.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=21)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    num_latent_frames = (args.num_frames - 1) // 4 + 1
    shape = (1, 16, num_latent_frames, args.height // 8, args.width // 8)
    latents_npy = str(Path(args.mindir_dir) / "_align_latents.npy")
    latents = _make_latents(shape, args.seed, latents_npy)
    image = _load_image(args.image)

    print("[align] running HF baseline (WanImageToVideoPipeline, Wan2.1 schedule) on CPU ...")
    hf = _hf_baseline(args.model_dir, image, args.prompt, args.negative_prompt, latents,
                      args.height, args.width, args.num_frames, args.num_inference_steps,
                      args.guidance_scale, args.max_seq_len)
    print(f"[align] HF frames: {hf.shape}, range [{hf.min():.3f}, {hf.max():.3f}]")

    print("[align] running MSLite pipeline on Ascend ...")
    inferencer = WanI2VInferencer(
        args.mindir_dir, args.model_dir, height=args.height, width=args.width,
        num_frames=args.num_frames, max_seq_len=args.max_seq_len,
        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale)
    ms = _mslite_video(inferencer, args.prompt, image, args.negative_prompt, latents_npy)
    print(f"[align] MSLite frames: {ms.shape}, range [{ms.min():.3f}, {ms.max():.3f}]")

    n = min(hf.shape[0], ms.shape[0])
    max_abs, mean_abs, psnr = _compare(hf[:n], ms[:n])
    print("\n--- Alignment (HF vs MSLite) ---")
    print(f"  frames compared: {n}")
    print(f"  max  abs error : {max_abs:.6f}")
    print(f"  mean abs error : {mean_abs:.6f}")
    print(f"  PSNR (dB)      : {psnr:.2f}")


if __name__ == "__main__":
    main()
