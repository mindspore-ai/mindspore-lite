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
"""End-to-end image-to-image inference for Kandinsky-5.0-I2I-Lite on MindSpore Lite.

Loads four MindIR models (Qwen2.5-VL text encoder + CLIP pooled on dev1,
K5 I2I DiT transformer + FLUX VAE decoder on dev0), encodes the prompt and the
source image, runs the flow-matching denoising loop (FlowMatchEulerDiscrete
scheduler on CPU, transformer on Ascend with classifier-free guidance), and
decodes the latent to an image (FLUX VAE on Ascend).

Model inference (text encoder / CLIP / transformer / VAE) is pure
``mindspore_lite`` + ``numpy``. ``torch`` is imported ONLY for:

  * the (numpy-backed) diffusers FlowMatchEuler scheduler that runs on CPU;
  * the Qwen2.5-VL processor + CLIP tokenizer;
  * the FLUX VAE *encoder* of the source image (CPU only -- the encoder is not
    exported to MindIR; only the VAE *decoder* runs on Ascend).

I2I conditioning (verified against pipeline_kandinsky_i2i.py): the visual input
to the DiT is built as ``cat([noise(16), image_latents(16), mask(1)], -1)`` ->
``[1, 1, H', W', 33]`` (channels-last). Only the first 16 channels (the noise)
are updated by the scheduler each step; the image_latents and mask channels are
carried through unchanged and re-fed every step. The DiT predicts
``[1, 1, H', W', 16]``.
"""

import argparse
import time
from pathlib import Path

import numpy as np

import mindspore_lite as mslite

# torch is used ONLY for: (1) the CPU/numpy-backed diffusers FlowMatchEuler
# scheduler; (2) the Qwen2.5-VL processor + CLIP tokenizer; (3) the FLUX VAE
# *encoder* of the source image. The Ascend model inference is pure mslite +
# numpy (see module docstring).
import torch  # noqa: E402  (scheduler + tokenizer + VAE-encode only)
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402

_LATENT_CHANNELS = 16
_VAE_SCALE_SPATIAL = 8
_PATCH_H = 2
_PATCH_W = 2
_PROMPT_TEMPLATE_ENCODE_START_IDX = 55
# K5 I2I wraps the prompt + source image in a fixed chat template before Qwen
# tokenisation. The vision placeholder <|image_pad|> is expanded by the VL
# processor into image tokens. "promt" typo is intentional (upstream).
_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a promt engineer. Based on the provided source image (first image) "
    "and target image (second image), create an interesting text prompt that "
    "can be used together with the source image to create the target image:"
    "<|im_end|><|im_start|>user{}<|vision_start|><|image_pad|><|vision_end|>"
    "<|im_end|>")


def _build_model(mindir_path, device_id):
    """Build an mslite Model from a ``*_graph.mindir`` on a given Ascend device."""
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = int(device_id)
    context.ascend.precision_mode = "force_fp16"
    model = mslite.Model()
    model.build_from_file(str(mindir_path), mslite.ModelType.MINDIR, context)
    return model


def _np_to_input(tensor_info, array):
    """Cast a numpy array to the dtype expected by a model input tensor."""
    dtype_map = {np.dtype("float16"): np.float16, np.dtype("float32"): np.float32,
                 np.dtype("int32"): np.int32, np.dtype("int64"): np.int64}
    target = dtype_map.get(np.dtype(tensor_info.dtype))
    return array.astype(target) if target is not None else array


def _run_model(model, feed_dict, preferred_order):
    """Run a model by matching input names, falling back to a preferred order."""
    inputs = model.get_inputs()
    if all(getattr(t, "name", None) in feed_dict for t in inputs):
        tensors = [mslite.Tensor(_np_to_input(t, feed_dict[t.name])) for t in inputs]
    else:
        tensors = [mslite.Tensor(_np_to_input(t, feed_dict[k]))
                   for t, k in zip(inputs, preferred_order)]
    outputs = model.predict(tensors)
    return [o.get_data_to_numpy() for o in outputs]


class Kandinsky5I2IInferencer:
    """End-to-end Kandinsky-5.0-I2I-Lite inferencer over MindSpore Lite."""

    def __init__(self, mindir_dir, qwen_dir, clip_dir, vae_dir,
                 text_device=1, transformer_device=0, vae_device=0,
                 height=1024, width=1024, max_seq_len=768,
                 num_inference_steps=50, guidance_scale=3.5, shift=1.0,
                 vae_scaling_factor=0.3611):
        """Load sub-models, tokenizers, scheduler and VAE encoder/decoder.

        ``vae_dir`` is the FLUX.1-dev ``vae/`` dir used BOTH for the CPU
        VAE-encoder (source image) and the Ascend VAE-decoder MindIR.
        """
        mindir_dir = Path(mindir_dir)
        self.qwen_dir = Path(qwen_dir)
        self.clip_dir = Path(clip_dir)
        self.vae_dir = Path(vae_dir)
        self.height = height
        self.width = width
        self.max_seq_len = max_seq_len
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.vae_scaling_factor = float(vae_scaling_factor)
        self.latent_h = height // _VAE_SCALE_SPATIAL
        self.latent_w = width // _VAE_SCALE_SPATIAL
        self.seq_after_slice = max_seq_len - _PROMPT_TEMPLATE_ENCODE_START_IDX

        print(f"[infer] loading Qwen/CLIP MindIR (dev{text_device}) ...")
        self.text_model = _build_model(
            mindir_dir / "kandinsky_text_encoder_graph.mindir", text_device)
        self.clip_model = _build_model(
            mindir_dir / "kandinsky_clip_encoder_graph.mindir", text_device)
        print(f"[infer] loading transformer/VAE MindIR (dev{transformer_device}) ...")
        self.transformer = _build_model(
            mindir_dir / "kandinsky_transformer_graph.mindir", transformer_device)
        self.vae = _build_model(
            mindir_dir / "kandinsky_dcae_decoder_graph.mindir", vae_device)

        # Qwen2.5-VL processor (text + image tokens) and CLIP tokenizer.
        from transformers import AutoTokenizer, AutoProcessor
        self.qwen_proc = AutoProcessor.from_pretrained(self.qwen_dir)
        self.clip_tok = AutoTokenizer.from_pretrained(self.clip_dir)
        # FLUX VAE encoder (CPU torch) for the source image. Lazy import to keep
        # the module importable without a VAE checkpoint present.
        from diffusers import AutoencoderKL
        self.vae_encoder = AutoencoderKL.from_pretrained(
            self.vae_dir, torch_dtype=torch.float32).eval()
        # FlowMatchEulerDiscreteScheduler with the K5 time-shift.
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)

    # -- source-image encoding (CPU torch) ----------------------------------

    def _vae_encode_image(self, image):
        """VAE-encode the source image to latents [1,1,H',W',16] (channels-last).

        Mirrors ``Kandinsky5I2IPipeline.prepare_latents``: preprocess to
        ``[-1,1]``, ``vae.encode(...).latent_dist.sample()``, unsqueeze T=1,
        multiply by ``scaling_factor``, permute to channels-last. ``image`` is a
        PIL.Image (RGB).
        """
        from torchvision import transforms as _T  # CPU-only preprocess
        preprocess = _T.Compose([
            _T.Resize((self.height, self.width)),
            _T.ToTensor(),
            _T.Normalize([0.5], [0.5]),
        ])
        pixel_values = preprocess(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            moment = self.vae_encoder.encode(pixel_values).latent_dist
            image_latents = moment.sample().to(torch.float32)
        # image_latents: [1, 16, H', W'] -> scale -> [1, 1, H', W', 16]
        image_latents = image_latents * self.vae_scaling_factor
        image_latents = image_latents.unsqueeze(2)  # add T=1 -> [1,16,1,H',W']
        image_latents = image_latents.permute(0, 2, 3, 4, 1)  # NHWDC
        return image_latents.numpy().astype(np.float32)

    # -- prompt encoding (Ascend) -------------------------------------------

    def _encode_prompt(self, prompt, image):
        """Tokenise with the K5 I2I chat template (+ source image) and run Qwen.

        Returns (encoder_hidden_states[1, seq-55, 3584], pooled[1, 768]) in
        float32. The Qwen [:, 55:] slice is baked into the exported graph, so we
        feed the full-length tokenised template+image. CLIP sees the plain
        prompt (no image).

        ``image`` is the source PIL.Image (the VL processor resizes it to half
        resolution internally, matching upstream ``image.resize((w//2, h//2))``).
        """
        full_text = _PROMPT_TEMPLATE.format(prompt)
        # Upstream resizes the source image to half size before tokenisation.
        half_img = image.resize(
            (image.size[0] // 2, image.size[1] // 2))
        max_allowed_len = _PROMPT_TEMPLATE_ENCODE_START_IDX + self.max_seq_len
        qwen_inputs = self.qwen_proc(
            text=[full_text], images=[half_img], videos=None,
            max_length=max_allowed_len, truncation=True, return_tensors="np",
            padding=True)
        qwen_ids = np.asarray(qwen_inputs["input_ids"], dtype=np.int64)
        qwen_mask = np.asarray(qwen_inputs["attention_mask"], dtype=np.int64)
        # If the VL processor produced a sequence longer than max_seq_len (image
        # tokens expand the placeholder), truncate to the graph's fixed length.
        if qwen_ids.shape[1] > self.max_seq_len:
            qwen_ids = qwen_ids[:, :self.max_seq_len]
            qwen_mask = qwen_mask[:, :self.max_seq_len]
        elif qwen_ids.shape[1] < self.max_seq_len:
            pad = self.max_seq_len - qwen_ids.shape[1]
            qwen_ids = np.pad(qwen_ids, ((0, 0), (0, pad)), constant_values=0)
            qwen_mask = np.pad(qwen_mask, ((0, 0), (0, pad)), constant_values=0)
        enc = _run_model(self.text_model,
                         {"input_ids": qwen_ids, "attention_mask": qwen_mask},
                         ["input_ids", "attention_mask"])[0].astype(np.float32)

        clip_inputs = self.clip_tok(
            prompt, padding="max_length", max_length=77, truncation=True,
            add_special_tokens=True, return_tensors="np")
        clip_ids = np.asarray(clip_inputs["input_ids"], dtype=np.int64)
        clip_mask = np.asarray(clip_inputs["attention_mask"], dtype=np.int64)
        pooled = _run_model(self.clip_model,
                            {"input_ids": clip_ids, "attention_mask": clip_mask},
                            ["input_ids", "attention_mask"])[0].astype(np.float32)
        return enc, pooled

    def _rope_inputs(self):
        """Build the RoPE position tensors (int64) matching the export."""
        visual_rope_h = np.arange(self.latent_h // _PATCH_H, dtype=np.int64)
        visual_rope_w = np.arange(self.latent_w // _PATCH_W, dtype=np.int64)
        text_rope = np.arange(self.seq_after_slice, dtype=np.int64)
        return visual_rope_h, visual_rope_w, text_rope

    # -- denoising loop (Ascend transformer, CPU scheduler) -----------------

    def _denoise(self, noise_latents, image_latents, mask, cond_enc,
                 cond_pooled, uncond_enc, uncond_pooled):
        """Run the CFG flow-matching I2I denoising loop; return denoised latents.

        ``noise_latents`` is channels-last ``[1,1,H',W',16]`` (the part updated
        by the scheduler). ``image_latents``/``mask`` are the fixed conditioning
        channels re-fed every step. The scheduler operates on the noise channels
        only (matching upstream ``latents[:,:,:,:,:16] = step(...)``).
        """
        self.scheduler.set_timesteps(self.num_inference_steps, "cpu")
        self.scheduler.set_begin_index(0)
        timesteps = self.scheduler.timesteps
        visual_rope_h, visual_rope_w, text_rope = self._rope_inputs()
        # The DiT consumes channels-last [1,1,H',W',16]; the diffusers
        # FlowMatchEuler step is layout-agnostic as long as pred / latent shapes
        # match. We keep the scheduler-state noise in NCHW [1,16,1,H',W'] and
        # permute to channels-last only when feeding the DiT.
        order = ["noise", "image_latents", "mask", "encoder_hidden_states",
                 "timestep", "pooled_projections", "visual_rope_h",
                 "visual_rope_w", "text_rope"]
        # noise_latents is [1,1,H',W',16]; permute to [1,16,1,H',W'] for sched.
        latents_t = torch.from_numpy(np.transpose(noise_latents, (0, 4, 1, 2, 3)))
        for t in timesteps:
            timestep = np.array([float(t)], dtype=np.float32)
            # Current noise as channels-last [1,1,H',W',16] for the DiT.
            cur_noise = np.transpose(
                latents_t.numpy().astype(np.float32), (0, 2, 3, 4, 1))
            base_feed = {
                "noise": cur_noise,
                "image_latents": image_latents,
                "mask": mask,
                "timestep": timestep,
                "visual_rope_h": visual_rope_h,
                "visual_rope_w": visual_rope_w,
                "text_rope": text_rope,
            }
            cond_feed = {**base_feed,
                         "encoder_hidden_states": cond_enc,
                         "pooled_projections": cond_pooled}
            uncond_feed = {**base_feed,
                           "encoder_hidden_states": uncond_enc,
                           "pooled_projections": uncond_pooled}
            # DiT returns noise_pred in channels-last [1,1,H',W',16]; permute to
            # [1,16,1,H',W'] for the scheduler.
            noise_cond = _run_model(self.transformer, cond_feed, order)[0]
            noise_uncond = _run_model(self.transformer, uncond_feed, order)[0]
            noise_cond = np.transpose(noise_cond, (0, 4, 1, 2, 3))
            noise_uncond = np.transpose(noise_uncond, (0, 4, 1, 2, 3))
            noise_pred = noise_uncond + self.guidance_scale * (
                noise_cond - noise_uncond)
            latents_t = self.scheduler.step(
                torch.from_numpy(noise_pred), t, latents_t,
                return_dict=False)[0]
        # Back to channels-last [1,1,H',W',16].
        return np.transpose(latents_t.numpy().astype(np.float32),
                            (0, 2, 3, 4, 1))

    def _vae_decode(self, latents):
        """Denormalise latents (undo VAE scaling) and decode to an RGB image.

        ``latents`` is channels-last ``[1,1,H',W',16]``; we permute to NCHW
        ``[1,16,H',W']``, drop T=1, divide by scaling_factor, then call the FLUX
        VAE decoder (which takes ``[B, C, H, W]``).
        """
        latents = latents / self.vae_scaling_factor
        latents_2d = latents[0, 0]  # [H', W', 16]
        latents_2d = np.transpose(latents_2d, (2, 0, 1))[None]  # [1,16,H',W']
        image = _run_model(self.vae, {"latents": latents_2d.astype(np.float32)},
                           ["latents"])[0]
        return image

    def generate(self, prompt, negative_prompt, source_image, output_path,
                 seed=42, latents_npy=None):
        """Run the full I2I pipeline with stage timing and save the PNG.

        ``source_image`` is a PIL.Image (RGB). Returns the timing dict; the
        image is saved to ``output_path``.
        """
        timing = {}

        t0 = time.perf_counter()
        image_latents = self._vae_encode_image(source_image)
        timing["vae_encode_ms"] = (time.perf_counter() - t0) * 1000

        # Fixed mask = ones, channels-last [1,1,H',W',1] (matches upstream
        # ``torch.ones_like(latents[...,:1])``).
        mask = np.ones((1, 1, self.latent_h, self.latent_w, 1),
                       dtype=np.float32)

        t0 = time.perf_counter()
        cond_enc, cond_pooled = self._encode_prompt(prompt, source_image)
        uncond_enc, uncond_pooled = self._encode_prompt(
            negative_prompt or "", source_image)
        timing["text_encode_ms"] = (time.perf_counter() - t0) * 1000

        # Initial noise (channels-last [1,1,H',W',16]); NOT pre-scaled by the
        # VAE scaling factor here because the image_latents channels are already
        # scaled and the scheduler must see the noise in the same space. The
        # scheduler step (FlowMatchEuler) updates the noise channels directly.
        if latents_npy:
            noise_latents = np.load(latents_npy).astype(np.float32)
        else:
            rng = np.random.RandomState(seed)
            noise_latents = rng.standard_normal(
                (1, 1, self.latent_h, self.latent_w, _LATENT_CHANNELS)
            ).astype(np.float32)

        t0 = time.perf_counter()
        final_noise = self._denoise(
            noise_latents, image_latents, mask, cond_enc, cond_pooled,
            uncond_enc, uncond_pooled)
        timing["transformer_total_ms"] = (time.perf_counter() - t0) * 1000
        timing["transformer_avg_step_ms"] = (
            timing["transformer_total_ms"] / self.num_inference_steps)

        t0 = time.perf_counter()
        image = self._vae_decode(final_noise)
        timing["vae_decode_ms"] = (time.perf_counter() - t0) * 1000

        _save_png(image, output_path)
        timing["e2e_ms"] = (timing["vae_encode_ms"]
                            + timing["text_encode_ms"]
                            + timing["transformer_total_ms"]
                            + timing["vae_decode_ms"])
        return timing


def _save_png(image_chw, path):
    """Save a [1,3,H,W] float image (range [-1,1]) as a PNG via PIL."""
    from PIL import Image
    img = (image_chw[0].transpose(1, 2, 0) / 2 + 0.5).clip(0, 1)
    img = (img * 255).round().astype(np.uint8)
    Image.fromarray(img).save(str(path))
    print(f"[infer] saved image -> {path}")


def _parse_args():
    """Parse command-line arguments for K5 I2I Lite MindSpore Lite inference."""
    parser = argparse.ArgumentParser(
        description="Kandinsky-5.0-I2I-Lite MindSpore Lite inference")
    parser.add_argument("--mindir-dir", required=True,
                        help="dir with the 4 *_graph.mindir files.")
    parser.add_argument("--qwen-dir", required=True,
                        help="Qwen2.5-VL-7B-Instruct weights dir (processor).")
    parser.add_argument("--clip-dir", required=True,
                        help="openai/clip-vit-large-patch14 dir (tokenizer).")
    parser.add_argument("--vae-dir", required=True,
                        help="FLUX.1-dev vae/ dir (CPU encoder + Ascend decoder).")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--source-image", required=True,
                        help="Path to the source (conditioning) image (PNG/JPG).")
    parser.add_argument("--output", default="./kandinsky5_i2i_output.png")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--shift", type=float, default=1.0,
                        help="FlowMatchEuler shift (= upstream scheduler_scale).")
    parser.add_argument("--vae-scaling-factor", type=float, default=0.3611,
                        help="FLUX VAE scaling factor (0.3611).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latents-npy", default=None,
                        help="pre-generated noise latents (for alignment).")
    parser.add_argument("--text-device", type=int, default=1)
    parser.add_argument("--transformer-device", type=int, default=0)
    parser.add_argument("--vae-device", type=int, default=0)
    return parser.parse_args()


def main():
    """Parse args and run Kandinsky-5.0-I2I-Lite image-to-image inference."""
    args = _parse_args()
    if args.height % 16 or args.width % 16:
        raise ValueError("height/width must be multiples of 16")
    from PIL import Image
    source_image = Image.open(args.source_image).convert("RGB")
    inferencer = Kandinsky5I2IInferencer(
        args.mindir_dir, args.qwen_dir, args.clip_dir, args.vae_dir,
        text_device=args.text_device, transformer_device=args.transformer_device,
        vae_device=args.vae_device, height=args.height, width=args.width,
        max_seq_len=args.max_seq_len,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale, shift=args.shift,
        vae_scaling_factor=args.vae_scaling_factor)
    timing = inferencer.generate(
        args.prompt, args.negative_prompt, source_image, args.output,
        args.seed, args.latents_npy)

    print("\n--- Performance ---")
    print(f"  VAE encode (source, CPU): {timing['vae_encode_ms']:.2f} ms")
    print(f"  Text encode (Qwen+CLIP):  {timing['text_encode_ms']:.2f} ms")
    print(f"  Transformer total:        {timing['transformer_total_ms']:.2f} ms")
    print(f"  Transformer avg/step:     {timing['transformer_avg_step_ms']:.2f} ms "
          f"({args.num_inference_steps} steps, CFG x2)")
    print(f"  VAE decode:               {timing['vae_decode_ms']:.2f} ms")
    print(f"  End-to-end:               {timing['e2e_ms']:.2f} ms")


if __name__ == "__main__":
    main()
