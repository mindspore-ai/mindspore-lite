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
"""End-to-end image-to-video inference for SVD-XT-1-1 on MindSpore Lite (Ascend).

SVD-XT-1-1 is the continued-training refinement of SVD-XT with identical
architecture and pipeline hyperparameters; only the checkpoint weights differ
(improved motion/quality). The defaults below match diffusers 0.38.0
``StableVideoDiffusionPipeline.__call__`` for the 1-1 release.

Loads three MindIR sub-models (CLIP image_encoder on dev1; UNet + VAE on dev0),
runs the SVD denoising loop and decodes a video.

The CPU side (torch) does the parts that do not benefit from Ascend and that
diffusers implements in numpy/torch:

  * image preprocessing (resize-with-antialiasing + CLIP normalisation, VAE
    preprocess of the conditioning frame) and VAE *encode* of the conditioning
    frame (run once);
  * the ``EulerDiscreteScheduler`` (timesteps, scale_model_input, step) and
    the per-frame classifier-free-guidance schedule
    (``linspace(min_gs, max_gs, num_frames)``);
  * the final postprocess (clamp -> uint8 frames) and mp4/PNG export.

The Ascend side (MindIR) does the heavy compute:

  * CLIP image_encoder -> image_embeds (conditioning);
  * UNet denoising (25 steps, CFG x2 per step);
  * VAE temporal decoder (single-frame chunks).

Component split on the 300I Duo: image_encoder -> dev1, unet + vae -> dev0.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mindspore_lite as mslite
    import torch
    import torch.nn.functional as F
    from diffusers import EulerDiscreteScheduler
    from diffusers.utils import load_image
    from transformers import CLIPImageProcessor
except ImportError as exc:  # pragma: no cover
    print(f"Error: missing dependency: {exc}")
    print("Install: pip install mindspore-lite diffusers transformers torch")
    sys.exit(1)


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
        tensors = [mslite.Tensor(_np_to_input(t, feed_dict[t.name]))
                   for t in inputs]
    else:
        tensors = [mslite.Tensor(_np_to_input(t, feed_dict[k]))
                   for t, k in zip(inputs, preferred_order)]
    outputs = model.predict(tensors)
    return [o.get_data_to_numpy() for o in outputs]


# ---------------------------------------------------------------------------
# Image preprocessing (mirrors StableVideoDiffusionPipeline._encode_image and
# the VAE encode path, but on CPU torch).
# ---------------------------------------------------------------------------


def _resize_with_antialiasing(image, size, interpolation="bicubic",
                              align_corners=True):
    """Reproduce diffusers' antialiasing downscale to ``size=(224,224)``.

    ``image`` is a torch tensor [N,3,H,W] in [0,1] (post-video_processor
    un-normalisation). Used only for the CLIP image_encoder input.
    """
    h, w = image.shape[-2:]
    factors = (h / size[0], w / size[1])
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]
    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1
    image = _gaussian_blur2d(image, ks, sigmas)
    return F.interpolate(image, size=size, mode=interpolation,
                         align_corners=align_corners)


def _compute_padding(kernel_size):
    """Compute reflect padding tuple for a 2D kernel (mirrors diffusers)."""
    computed = [k - 1 for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]
    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        out_padding[2 * i + 0] = computed_tmp // 2
        out_padding[2 * i + 1] = computed_tmp - computed_tmp // 2
    return out_padding


def _filter2d(image, kernel):
    """Apply a separable 2D filter (reflect padding) -- mirrors diffusers."""
    b, c, h, w = image.shape
    tmp_kernel = kernel[:, None, ...].to(device=image.device, dtype=image.dtype)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]
    pad = _compute_padding([height, width])
    image = F.pad(image, pad, mode="reflect")
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    image = image.view(-1, tmp_kernel.size(0), image.size(-2), image.size(-1))
    out = F.conv2d(image, tmp_kernel, groups=tmp_kernel.size(0), padding=0)
    return out.view(b, c, h, w)


def _gaussian(window_size, sigma):
    """1D gaussian kernel (mirrors diffusers)."""
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])
    bs = sigma.shape[0]
    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype)
         - window_size // 2).expand(bs, -1)
    if window_size % 2 == 0:
        x = x + 0.5
    g = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))
    return g / g.sum(-1, keepdim=True)


def _gaussian_blur2d(image, kernel_size, sigma):
    """Separable gaussian blur (mirrors diffusers)."""
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=image.dtype)
    else:
        sigma = sigma.to(dtype=image.dtype)
    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(image, kernel_x[..., None, :])
    return _filter2d(out_x, kernel_y[..., None])


def _prepare_clip_input(image_pil, height, width, feature_extractor):
    """Reproduce SVD ``_encode_image`` up to the CLIP input tensor.

    The diffusers pipeline does: video_processor.numpy_to_pt -> normalise
    [-1,1] -> antialias resize to (224,224) -> un-normalise [0,1] ->
    feature_extractor(do_normalize=True, no resize/rescale) -> CLIP.
    We accept a PIL image and resize it to (width,height) first (matching
    ``pipe(image, height=576, width=1024)`` which resizes the conditioning
    image to the output resolution before preprocessing).
    """
    image = image_pil.convert("RGB").resize((width, height))
    arr = np.asarray(image, dtype=np.float32) / 255.0  # (H,W,3)
    pt = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    pt = pt * 2.0 - 1.0  # normalise to [-1,1]
    pt = _resize_with_antialiasing(pt, (224, 224))
    pt = (pt + 1.0) / 2.0  # back to [0,1]
    # feature_extractor normalises with CLIP mean/std, no resize/rescale.
    clip_in = feature_extractor(
        images=pt, do_normalize=True, do_center_crop=False, do_resize=False,
        do_rescale=False, return_tensors="pt").pixel_values
    return clip_in.numpy().astype(np.float32)  # (1,3,224,224)


def _prepare_vae_input(image_pil, height, width):
    """Reproduce video_processor.preprocess for the conditioning frame.

    Resize to (height,width), normalise to [-1,1], shape (1,3,H,W).
    """
    image = image_pil.convert("RGB").resize((width, height))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    pt = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return (pt * 2.0 - 1.0)  # (1,3,H,W) in [-1,1]


# ---------------------------------------------------------------------------
# Inference.
# ---------------------------------------------------------------------------


class SvdInferencer:
    """SVD-XT-1-1 image-to-video on MindSpore Lite (image_encoder / unet+vae)."""

    def __init__(self, mindir_dir, model_dir, image_device=1,
                 unet_device=0, vae_device=0, height=576, width=1024,
                 num_frames=25):
        """Load sub-models, feature_extractor, scheduler; record geometry."""
        mindir_dir = Path(mindir_dir)
        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.latent_h = self.height // 8
        self.latent_w = self.width // 8

        print(f"Loading CLIP image_encoder MindIR (dev{image_device}) ...")
        self.image_encoder = _build_model(
            mindir_dir / "svd_image_encoder_graph.mindir", image_device)
        print(f"Loading UNet MindIR (dev{unet_device}) ...")
        self.unet = _build_model(mindir_dir / "svd_unet_graph.mindir",
                                 unet_device)
        print(f"Loading VAE decoder MindIR (dev{vae_device}) ...")
        self.vae = _build_model(mindir_dir / "svd_vae_decoder_graph.mindir",
                                vae_device)

        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            Path(model_dir) / "feature_extractor")
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            model_dir, subfolder="scheduler")

        from json import load as _load
        with open(Path(model_dir) / "vae" / "config.json") as fh:
            vae_cfg = _load(fh)
        self.scaling_factor = float(vae_cfg.get("scaling_factor", 0.18215))
        # SVD vae force_upcast is False for fp16 -> no upcast needed here.

    # -- conditioning ------------------------------------------------------

    def _encode_image(self, image_pil):
        """CLIP image_embeds [1,1024] via Ascend image_encoder.

        Returns image_embeds and the wall-clock time (ms).
        """
        t0 = time.perf_counter()
        clip_in = _prepare_clip_input(
            image_pil, self.height, self.width, self.feature_extractor)
        embeds = _run_model(
            self.image_encoder, {"pixel_values": clip_in},
            ["pixel_values"])[0]
        return embeds.astype(np.float32), (time.perf_counter() - t0) * 1000.0

    def _encode_vae_image(self, image_pil, vae_torch, noise_aug_strength,
                          generator):
        """VAE-encode the conditioning frame on CPU torch -> image_latents.

        Mirrors ``_encode_vae_image``: encode -> latent_dist.mode(), add
        noise_aug_strength * noise, return [1,4,h,w]. ``vae_torch`` is the
        diffusers AutoencoderKLTemporalDecoder loaded on CPU for this one
        forward.
        """
        image = _prepare_vae_input(image_pil, self.height, self.width)
        noise = torch.randn(image.shape, generator=generator)
        image = image + noise_aug_strength * noise
        with torch.no_grad():
            image_latents = vae_torch.encode(image).latent_dist.mode()
        return image_latents.numpy().astype(np.float32)  # (1,4,h,w)

    # -- UNet forward ------------------------------------------------------

    def _unet_forward(self, sample, timestep, image_embeds, added_time_ids):
        """Run one UNet forward (returns noise_pred as numpy float32).

        ``sample`` already has the image_latents concatenated on dim=2
        (channels) and is batch=2 (CFG). ``image_embeds`` and
        ``added_time_ids`` are likewise batch=2.
        """
        feed = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": image_embeds,
            "added_time_ids": added_time_ids,
        }
        noise_pred = _run_model(
            self.unet, feed,
            ["sample", "timestep", "encoder_hidden_states",
             "added_time_ids"])[0]
        return noise_pred.astype(np.float32)

    def _denoise(self, latents, image_latents_frames, image_embeds,
                 added_time_ids, min_gs, max_gs, num_steps):
        """Run the SVD CFG Euler denoising loop.

        SVD uses a per-frame guidance scale (``linspace(min_gs, max_gs,
        num_frames)``) applied after the UNet forward: the cond/uncond split
        is along the batch dim, and guidance_scale broadcasts over
        (batch, frames, 1, 1, 1). Returns denoised latents
        [1, num_frames, 4, h, w] and UNet timing.
        """
        self.scheduler.set_timesteps(num_steps, device="cpu")
        self.scheduler.set_begin_index(0)
        timesteps = self.scheduler.timesteps

        # Per-frame guidance schedule [1, F, 1, 1, 1] (broadcast over frames).
        gs = torch.linspace(min_gs, max_gs, self.num_frames).unsqueeze(0)
        gs = gs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (1,F,1,1,1)

        latents_t = torch.from_numpy(latents)
        image_latents_t = torch.from_numpy(image_latents_frames)
        embeds_t = torch.from_numpy(image_embeds)
        add_ids = torch.from_numpy(added_time_ids)

        total_unet_ms = 0.0
        for i, t in enumerate(timesteps):
            # CFG: stack [uncond, cond] along batch.
            latent_model_input = torch.cat([latents_t, latents_t], dim=0)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            # Concat image_latents over channels (dim=2). image_latents is
            # already (1,F,4,h,w) -> expand to batch=2.
            il = image_latents_t.expand(2, -1, -1, -1, -1)
            sample = torch.cat([latent_model_input, il], dim=2)

            ti = time.perf_counter()
            noise_pred = self._unet_forward(
                sample.numpy().astype(np.float32),
                np.array([float(t), float(t)], dtype=np.float32),
                embeds_t.numpy().astype(np.float32),
                add_ids.numpy().astype(np.float32))
            total_unet_ms += (time.perf_counter() - ti) * 1000.0

            noise_pred_t = torch.from_numpy(noise_pred)
            noise_uncond, noise_cond = noise_pred_t.chunk(2)
            noise = noise_uncond + gs * (noise_cond - noise_uncond)
            latents_t = self.scheduler.step(
                noise, t, latents_t, return_dict=False)[0]
        return latents_t.numpy(), total_unet_ms, len(timesteps)

    # -- decode ------------------------------------------------------------

    def _decode_frames(self, latents):
        """Decode latents [F,4,h,w] -> frames [F,3,H,W] via Ascend VAE.

        The exported VAE decoder runs a single-frame chunk (num_frames=1),
        so we decode each frame independently. Latents are first unscaled by
        ``1/scaling_factor`` (the pipeline divides by scaling_factor before
        decode).
        """
        latents = latents / self.scaling_factor
        frames = []
        td = time.perf_counter()
        for f in range(latents.shape[0]):
            chunk = latents[f:f + 1].astype(np.float16)
            img = _run_model(self.vae, {"latents": chunk}, ["latents"])[0]
            frames.append(img.astype(np.float32))
        t_vae = (time.perf_counter() - td) * 1000.0
        return np.concatenate(frames, axis=0), t_vae

    # -- entry point -------------------------------------------------------

    def infer(self, image_pil, vae_torch, seed=0, num_inference_steps=25,
              fps=7, motion_bucket_id=127, noise_aug_strength=0.02,
              min_guidance_scale=1.0, max_guidance_scale=3.0):
        """Run the full SVD-XT-1-1 pipeline. Returns (frames_uint8, timing).

        Defaults match diffusers 0.38.0 ``StableVideoDiffusionPipeline`` for
        the 1-1 release (identical to SVD-XT): fps=7 (internal fps-1=6),
        motion_bucket_id=127, noise_aug_strength=0.02, min/max gs 1.0/3.0.

        ``frames_uint8`` is [F, H, W, 3] in [0,255]. ``vae_torch`` is the CPU
        diffusers VAE used only for the one-shot conditioning encode.
        """
        t_start = time.perf_counter()
        # 1. CLIP image_embeds.
        image_embeds, t_clip = self._encode_image(image_pil)
        # image_embeds [1,1024] -> [1,1,1024]; CFG -> [2,1,1024].
        image_embeds_cfg = np.concatenate(
            [np.zeros_like(image_embeds[None]), image_embeds[None]], axis=0)

        # 2. VAE-encode conditioning frame (CPU torch, once) -> image_latents.
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        image_latents = self._encode_vae_image(
            image_pil, vae_torch, noise_aug_strength, gen)  # (1,4,h,w)
        # Repeat for each frame -> (1,F,4,h,w); CFG handled in _denoise.
        image_latents_frames = np.broadcast_to(
            image_latents[None],
            (1, self.num_frames, 4, self.latent_h, self.latent_w
             )).astype(np.float32)

        # 3. added_time_ids (fps-1, motion_bucket_id, noise_aug_strength).
        add_ids = np.array(
            [[float(fps) - 1.0, float(motion_bucket_id),
              float(noise_aug_strength)]], dtype=np.float32)
        add_ids_cfg = np.concatenate([add_ids, add_ids], axis=0)

        # 4. initial noise latents [1, F, 4, h, w].
        rng = np.random.RandomState(seed)
        latents = rng.standard_normal(
            (1, self.num_frames, 4, self.latent_h, self.latent_w
             )).astype(np.float32)
        latents = latents * float(self.scheduler.init_noise_sigma)

        # 5. denoise.
        latents, total_unet_ms, steps = self._denoise(
            latents, image_latents_frames, image_embeds_cfg, add_ids_cfg,
            min_guidance_scale, max_guidance_scale, num_inference_steps)

        # 6. decode (latents are [1,F,4,h,w] -> [F,4,h,w]).
        latents_flat = latents[0]
        frames, t_vae = self._decode_frames(latents_flat)

        # 7. postprocess -> uint8 [F,H,W,3].
        frames = (frames / 2 + 0.5).clip(0, 1)
        frames = (frames * 255).round().astype(np.uint8)
        frames = frames.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        t_e2e = (time.perf_counter() - t_start) * 1000.0
        avg_step = total_unet_ms / max(1, steps)
        timing = {
            "image_encode": t_clip, "unet_total": total_unet_ms,
            "unet_avg_step": avg_step, "steps": steps,
            "vae": t_vae, "e2e": t_e2e,
        }
        return frames, timing


# ---------------------------------------------------------------------------
# Video export.
# ---------------------------------------------------------------------------


def _save_video(frames_uint8, output_path, fps=7):
    """Save [F,H,W,3] uint8 frames to mp4 (imageio) or fall back to PNGs."""
    output_path = str(output_path)
    try:
        import imageio
        imageio.mimsave(output_path, list(frames_uint8), fps=fps,
                        codec="libx264", quality=8)
        print(f"[export] saved video -> {output_path} ({len(frames_uint8)} "
              f"frames, {fps} fps)")
        return
    except Exception as exc:  # pragma: no cover
        print(f"[export] imageio mp4 failed ({exc}); saving PNGs instead")
    base = output_path.rsplit(".", 1)[0]
    from PIL import Image
    for i, fr in enumerate(frames_uint8):
        Image.fromarray(fr).save(f"{base}_frame{i:03d}.png")
    print(f"[export] saved {len(frames_uint8)} frames -> {base}_frame*.png")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main():
    """Parse arguments and run SVD-XT-1-1 image-to-video inference."""
    parser = argparse.ArgumentParser(
        description="SVD-XT-1-1 MindSpore Lite inference")
    parser.add_argument("--mindir-dir", required=True,
                        help="dir with the 3 *_graph.mindir")
    parser.add_argument("--model-dir", required=True,
                        help="stable-video-diffusion-img2vid-xt-1-1 diffusers "
                             "weights dir (for vae/feature_extractor/scheduler)")
    parser.add_argument("--image", required=True,
                        help="conditioning image path (PIL-loadable).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--motion-bucket-id", type=int, default=127)
    parser.add_argument("--noise-aug-strength", type=float, default=0.02)
    parser.add_argument("--min-guidance", type=float, default=1.0)
    parser.add_argument("--max-guidance", type=float, default=3.0)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--image-device", type=int, default=1)
    parser.add_argument("--unet-device", type=int, default=0)
    parser.add_argument("--vae-device", type=int, default=0)
    parser.add_argument("--output", default="./svd_output.mp4")
    args = parser.parse_args()

    inferencer = SvdInferencer(
        args.mindir_dir, args.model_dir, image_device=args.image_device,
        unet_device=args.unet_device, vae_device=args.vae_device,
        height=args.height, width=args.width, num_frames=args.num_frames)

    # CPU VAE for the one-shot conditioning encode.
    from diffusers import AutoencoderKLTemporalDecoder
    vae_torch = AutoencoderKLTemporalDecoder.from_pretrained(
        args.model_dir, subfolder="vae", torch_dtype=torch.float32).eval()

    image_pil = load_image(args.image)
    frames, timing = inferencer.infer(
        image_pil, vae_torch, seed=args.seed,
        num_inference_steps=args.steps, fps=args.fps,
        motion_bucket_id=args.motion_bucket_id,
        noise_aug_strength=args.noise_aug_strength,
        min_guidance_scale=args.min_guidance,
        max_guidance_scale=args.max_guidance)

    _save_video(frames, args.output, fps=args.fps)
    print("\n--- Performance ---")
    print(f"  Image encode (CLIP, dev{args.image_device}): "
          f"{timing['image_encode']:.2f} ms")
    print(f"  UNet total ({timing['steps']} steps, CFG x2, per-frame gs): "
          f"{timing['unet_total']:.2f} ms")
    print(f"  UNet avg/step:               {timing['unet_avg_step']:.2f} ms")
    print(f"  VAE decode ({args.num_frames} frames, dev{args.vae_device}): "
          f"{timing['vae']:.2f} ms")
    print(f"  End-to-end:                  {timing['e2e']:.2f} ms")


if __name__ == "__main__":
    main()
