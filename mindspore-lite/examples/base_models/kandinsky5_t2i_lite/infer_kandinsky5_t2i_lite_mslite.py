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
"""End-to-end text-to-image inference for Kandinsky-5.0-T2I-Lite on MindSpore Lite.

Loads four MindIR models (Qwen2.5-VL text encoder + CLIP pooled on dev1,
K5 DiT transformer + FLUX VAE decoder on dev0), encodes the prompt, runs the
flow-matching denoising loop (FlowMatchEulerDiscreteScheduler on CPU,
transformer on Ascend with classifier-free guidance), and decodes the latent to
an image (FLUX VAE on Ascend).

Model inference (text encoder / CLIP / transformer / VAE) is pure
``mindspore_lite`` + ``numpy``. ``torch`` is imported ONLY for the
(numpy-backed) diffusers scheduler that runs on CPU, exactly as in the
flux1_dev / wan examples; no torch tensor ever touches the Ascend models.
"""

import argparse
import time
from pathlib import Path

import numpy as np

import mindspore_lite as mslite

# torch is used ONLY for the CPU/numpy-backed diffusers FlowMatchEuler scheduler
# and for the Qwen / CLIP tokenizers; the Ascend model inference is pure mslite
# + numpy (see module docstring).
import torch  # noqa: E402  (scheduler + tokenizer only)
from transformers import AutoTokenizer  # noqa: E402
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402

_LATENT_CHANNELS = 16
_VAE_SCALE_SPATIAL = 8
_PATCH_H = 2
_PATCH_W = 2
_PROMPT_TEMPLATE_ENCODE_START_IDX = 41
# K5 wraps the prompt in a fixed chat template before Qwen tokenisation. The
# typo "promt" is intentional (matches upstream pipeline_kandinsky_t2i.py).
_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a promt engineer. Describe the image by detailing the color, "
    "shape, size, texture, quantity, text, spatial relationships of the "
    "objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n")


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


class Kandinsky5T2IInferencer:
    """End-to-end Kandinsky-5.0-T2I-Lite inferencer over MindSpore Lite."""

    def __init__(self, mindir_dir, qwen_dir, clip_dir,
                 text_device=1, transformer_device=0, vae_device=0,
                 height=1024, width=1024, max_seq_len=512,
                 num_inference_steps=50, guidance_scale=3.5, shift=1.0,
                 vae_scaling_factor=0.3611):
        """Load sub-models, tokenizers, scheduler and VAE latent statistics."""
        mindir_dir = Path(mindir_dir)
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

        self.qwen_tok = AutoTokenizer.from_pretrained(Path(qwen_dir))
        self.clip_tok = AutoTokenizer.from_pretrained(Path(clip_dir))
        # FlowMatchEulerDiscreteScheduler with the K5 time-shift (shift absorbs
        # the upstream scheduler_scale; default 1.0 = linear schedule).
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)

    def _latent_shape(self):
        """Return the fixed latent shape (B, C, T=1, H', W') for an image."""
        return (1, _LATENT_CHANNELS, 1, self.latent_h, self.latent_w)

    def _encode_prompt(self, prompt):
        """Tokenise with the K5 chat template and run Qwen + CLIP on Ascend.

        Returns (encoder_hidden_states[1, seq-41, 3584], pooled[1, 768]) in
        float32. The Qwen [:, 41:] slice is baked into the exported graph, so we
        feed the full-length tokenised template.
        """
        full_text = _PROMPT_TEMPLATE.format(prompt)
        qwen_inputs = self.qwen_tok(
            full_text, padding="max_length", max_length=self.max_seq_len,
            truncation=True, add_special_tokens=False, return_tensors="np")
        qwen_ids = np.asarray(qwen_inputs["input_ids"], dtype=np.int64)
        qwen_mask = np.asarray(qwen_inputs["attention_mask"], dtype=np.int64)
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

    def _denoise(self, latents, cond_enc, cond_pooled, uncond_enc, uncond_pooled):
        """Run the CFG flow-matching denoising loop, returning denoised latents."""
        self.scheduler.set_timesteps(self.num_inference_steps, "cpu")
        self.scheduler.set_begin_index(0)
        timesteps = self.scheduler.timesteps
        visual_rope_h, visual_rope_w, text_rope = self._rope_inputs()
        latents_t = torch.from_numpy(latents)
        order = ["hidden_states", "encoder_hidden_states", "timestep",
                 "pooled_projections", "visual_rope_h", "visual_rope_w",
                 "text_rope"]
        for t in timesteps:
            timestep = np.array([float(t)], dtype=np.float32)
            base_feed = {
                "hidden_states": latents,
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
            noise_cond = _run_model(self.transformer, cond_feed, order)[0]
            noise_uncond = _run_model(self.transformer, uncond_feed, order)[0]
            noise_pred = noise_uncond + self.guidance_scale * (
                noise_cond - noise_uncond)
            latents_t = self.scheduler.step(
                torch.from_numpy(noise_pred), t, latents_t,
                return_dict=False)[0]
            latents = latents_t.numpy().astype(np.float32)
        return latents

    def _vae_decode(self, latents):
        """Denormalise latents (undo VAE scaling) and decode to an RGB image.

        K5 applies ``data *= vae.config.scaling_factor`` before the DiT and
        divides after; here we divide to undo it, then drop the T=1 axis before
        calling the FLUX VAE decoder (which takes [B, C, H, W]).
        """
        latents = latents / self.vae_scaling_factor
        latents_2d = latents[0, :, 0, :, :]  # [16, H', W']
        latents_2d = latents_2d[None, ...].astype(np.float32)  # [1,16,H',W']
        image = _run_model(self.vae, {"latents": latents_2d}, ["latents"])[0]
        return image

    def generate(self, prompt, negative_prompt, output_path, seed=42,
                 latents_npy=None):
        """Run the full T2I pipeline with stage timing and save the PNG.

        Returns the timing dict. The image is saved to ``output_path``.
        """
        timing = {}

        t0 = time.perf_counter()
        cond_enc, cond_pooled = self._encode_prompt(prompt)
        uncond_enc, uncond_pooled = self._encode_prompt(negative_prompt or "")
        timing["text_encode_ms"] = (time.perf_counter() - t0) * 1000

        if latents_npy:
            latents = np.load(latents_npy).astype(np.float32)
        else:
            rng = np.random.RandomState(seed)
            latents = rng.standard_normal(self._latent_shape()).astype(np.float32)
        # K5 scales the latents before the DiT (data *= scaling_factor).
        latents = latents * self.vae_scaling_factor

        t0 = time.perf_counter()
        latents = self._denoise(latents, cond_enc, cond_pooled,
                                uncond_enc, uncond_pooled)
        timing["transformer_total_ms"] = (time.perf_counter() - t0) * 1000
        timing["transformer_avg_step_ms"] = (
            timing["transformer_total_ms"] / self.num_inference_steps)

        t0 = time.perf_counter()
        image = self._vae_decode(latents)
        timing["vae_decode_ms"] = (time.perf_counter() - t0) * 1000

        _save_png(image, output_path)
        timing["e2e_ms"] = (timing["text_encode_ms"]
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
    """Parse command-line arguments for K5 Lite MindSpore Lite inference."""
    parser = argparse.ArgumentParser(
        description="Kandinsky-5.0-T2I-Lite MindSpore Lite inference")
    parser.add_argument("--mindir-dir", required=True,
                        help="dir with the 4 *_graph.mindir files.")
    parser.add_argument("--qwen-dir", required=True,
                        help="Qwen2.5-VL-7B-Instruct weights dir (tokenizer).")
    parser.add_argument("--clip-dir", required=True,
                        help="openai/clip-vit-large-patch14 dir (tokenizer).")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", default="./kandinsky5_output.png")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--shift", type=float, default=1.0,
                        help="FlowMatchEuler shift (= upstream scheduler_scale).")
    parser.add_argument("--vae-scaling-factor", type=float, default=0.3611,
                        help="FLUX VAE scaling factor (0.3611).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latents-npy", default=None,
                        help="pre-generated latents (for alignment).")
    parser.add_argument("--text-device", type=int, default=1)
    parser.add_argument("--transformer-device", type=int, default=0)
    parser.add_argument("--vae-device", type=int, default=0)
    return parser.parse_args()


def main():
    """Parse arguments and run Kandinsky-5.0-T2I-Lite text-to-image inference."""
    args = _parse_args()
    if args.height % 16 or args.width % 16:
        raise ValueError("height/width must be multiples of 16")
    inferencer = Kandinsky5T2IInferencer(
        args.mindir_dir, args.qwen_dir, args.clip_dir,
        text_device=args.text_device, transformer_device=args.transformer_device,
        vae_device=args.vae_device, height=args.height, width=args.width,
        max_seq_len=args.max_seq_len,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale, shift=args.shift,
        vae_scaling_factor=args.vae_scaling_factor)
    timing = inferencer.generate(
        args.prompt, args.negative_prompt, args.output, args.seed,
        args.latents_npy)

    print("\n--- Performance ---")
    print(f"  Text encode (Qwen+CLIP):  {timing['text_encode_ms']:.2f} ms")
    print(f"  Transformer total:        {timing['transformer_total_ms']:.2f} ms")
    print(f"  Transformer avg/step:     {timing['transformer_avg_step_ms']:.2f} ms "
          f"({args.num_inference_steps} steps, CFG x2)")
    print(f"  VAE decode:               {timing['vae_decode_ms']:.2f} ms")
    print(f"  End-to-end:               {timing['e2e_ms']:.2f} ms")


if __name__ == "__main__":
    main()
