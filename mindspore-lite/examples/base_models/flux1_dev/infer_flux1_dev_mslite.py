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
End-to-end text-to-image inference for FLUX.1-dev on MindSpore Lite (Ascend).

Loads four MindIR models (transformer + VAE on dev0, T5-XXL + CLIP on dev1),
encodes the prompt, runs the rectified-flow denoising loop (FlowMatchEuler
scheduler on CPU, transformer on Ascend), and decodes the latent to an image
(VAE on Ascend). The only torch usage is the (numpy-backed) diffusers scheduler
and the tokenizers.
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
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
except ImportError as exc:  # pragma: no cover
    print(f"Error: missing dependency: {exc}")
    print("Install: pip install mindspore-lite transformers diffusers torch")
    sys.exit(1)

from PIL import Image


# ---------------------------------------------------------------------------
# mslite helpers.
# ---------------------------------------------------------------------------


def _np_dtype_to_mslite(dtype):
    dt = np.dtype(dtype)
    if dt == np.float16:
        return mslite.DataType.FLOAT16
    if dt == np.float32:
        return mslite.DataType.FLOAT32
    if dt == np.int64:
        return mslite.DataType.INT64
    if dt == np.int32:
        return mslite.DataType.INT32
    raise TypeError(f"unsupported dtype for mslite.Tensor: {dt}")


def _build_inputs(model, feed_dict, preferred_order):
    """Build mslite input tensors by name (fallback to preferred order)."""
    inputs = model.get_inputs()
    if not inputs:
        return [mslite.Tensor(v) for v in feed_dict.values()]
    if all(getattr(t, "name", None) in feed_dict for t in inputs):
        return [mslite.Tensor(feed_dict[t.name]) for t in inputs]
    return [mslite.Tensor(feed_dict[k]) for k in preferred_order]


def _build_model(path, device, device_id):
    ctx = mslite.Context()
    ctx.target = [device]
    if device == "ascend":
        ctx.ascend.device_id = int(device_id)
        # allow the large transformer weights to load (Ascend graph compile)
        ctx.ascend.precision_mode = "force_fp16" if device == "ascend" else "enforce_fp32"
    model = mslite.Model()
    model.build_from_file(path, mslite.ModelType.MINDIR, ctx)
    return model


# ---------------------------------------------------------------------------
# numpy pack / unpack / ids (mirror diffusers FluxPipeline).
# ---------------------------------------------------------------------------


def _pack_latents(latents, num_channels, height, width):
    """[1,C,H,W] -> [1, (H/2)*(W/2), C*4]."""
    latents = latents.reshape(1, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.transpose(0, 2, 4, 1, 3, 5)
    return latents.reshape(1, (height // 2) * (width // 2), num_channels * 4)


def _unpack_latents(latents, height, width, vae_scale=8):
    """[1, (H/2)*(W/2), C*4] -> [1, C, H, W]."""
    batch, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale * 2))
    width = 2 * (int(width) // (vae_scale * 2))
    latents = latents.reshape(batch, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.transpose(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch, channels // (2 * 2), height, width)


def _latent_image_ids(h_tokens, w_tokens):
    """Reproduce FluxPipeline._prepare_latent_image_ids -> [h*w, 3]."""
    ids = np.zeros((h_tokens, w_tokens, 3), dtype=np.float32)
    ids[..., 1] = ids[..., 1] + np.arange(h_tokens)[:, None]
    ids[..., 2] = ids[..., 2] + np.arange(w_tokens)[None, :]
    return ids.reshape(h_tokens * w_tokens, 3)


# ---------------------------------------------------------------------------
# Inference.
# ---------------------------------------------------------------------------


class Flux1Inferencer:
    """FLUX.1-dev text-to-image on MindSpore Lite (transformer+VAE / T5+CLIP)."""

    def __init__(self, transformer_path, vae_path, t5_path, clip_path, model_dir,
                 device="ascend", transformer_device=0, text_device=1,
                 height=1024, width=1024, t5_seq_len=256):
        self.height = int(height)
        self.width = int(width)
        self.t5_seq_len = int(t5_seq_len)
        self.h_tok = self.height // 16
        self.w_tok = self.width // 16
        self.num_img_tokens = self.h_tok * self.w_tok

        print(f"Loading transformer MindIR (dev{transformer_device}) ...")
        self.transformer = _build_model(transformer_path, device, transformer_device)
        print(f"Loading VAE MindIR (dev{transformer_device}) ...")
        self.vae = _build_model(vae_path, device, transformer_device)
        print(f"Loading T5 MindIR (dev{text_device}) ...")
        self.t5 = _build_model(t5_path, device, text_device)
        print(f"Loading CLIP MindIR (dev{text_device}) ...")
        self.clip = _build_model(clip_path, device, text_device)

        self.clip_tok = AutoTokenizer.from_pretrained(Path(model_dir) / "tokenizer")
        self.t5_tok = AutoTokenizer.from_pretrained(Path(model_dir) / "tokenizer_2")

        cfg = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")
        self.scheduler = cfg
        self.scaling_factor = float(cfg.config.get("scaling_factor", 0.3611))
        self.shift_factor = float(cfg.config.get("shift_factor", 0.1159))

    def _encode_prompt(self, prompt):
        """Run CLIP + T5 on Ascend -> (encoder_hidden_states, pooled), both fp16."""
        t0 = time.perf_counter()
        clip_inputs = self.clip_tok(
            [prompt], padding="max_length", max_length=77,
            truncation=True, return_tensors="np").input_ids.astype(np.int64)
        pooled = self.clip.predict(_build_inputs(
            self.clip, {"input_ids": clip_inputs}, ["input_ids"]))[0].get_data_to_numpy()

        t5_inputs = self.t5_tok(
            [prompt], padding="max_length", max_length=self.t5_seq_len,
            truncation=True, return_tensors="np").input_ids.astype(np.int64)
        enc = self.t5.predict(_build_inputs(
            self.t5, {"input_ids": t5_inputs}, ["input_ids"]))[0].get_data_to_numpy()

        if pooled.dtype != np.float16:
            pooled = pooled.astype(np.float16)
        if enc.dtype != np.float16:
            enc = enc.astype(np.float16)
        return enc, pooled, (time.perf_counter() - t0) * 1000.0

    def infer(self, prompt, seed=0, num_inference_steps=28, guidance=3.5):
        """Run the full FLUX.1-dev pipeline. Returns (image_uint8, timing_dict)."""
        t_start = time.perf_counter()
        encoder_hidden_states, pooled, t_text = self._encode_prompt(prompt)

        # initial noise (numpy, fixed seed) -> packed latents
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(
            (1, 16, self.height // 8, self.width // 8)).astype(np.float32)
        latents_np = _pack_latents(noise, 16, self.height // 8, self.width // 8).astype(np.float16)

        txt_ids = np.zeros((self.t5_seq_len, 3), dtype=np.float16)
        img_ids = _latent_image_ids(self.h_tok, self.w_tok).astype(np.float16)

        # scheduler: same shift schedule as FluxPipeline (mu from image_seq_len)
        image_seq_len = self.num_img_tokens
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler, num_inference_steps, "cpu", sigmas=None, mu=mu)
        self.scheduler.set_begin_index(0)

        latents = torch.from_numpy(latents_np)  # fp16, packed
        guidance_arr = np.array([guidance], dtype=np.float16)

        total_transformer_ms = 0.0
        for _, t in enumerate(timesteps):
            timestep = (t / 1000.0)
            t_arr = np.array([float(timestep)], dtype=np.float16)
            feed = {
                "hidden_states": latents.numpy(),
                "encoder_hidden_states": encoder_hidden_states,
                "pooled_projections": pooled,
                "timestep": t_arr,
                "guidance": guidance_arr,
                "img_ids": img_ids,
                "txt_ids": txt_ids,
            }
            ti = time.perf_counter()
            out = self.transformer.predict(_build_inputs(
                self.transformer, feed,
                ["hidden_states", "encoder_hidden_states", "pooled_projections",
                 "timestep", "guidance", "img_ids", "txt_ids"]))
            noise_pred = out[0].get_data_to_numpy()
            total_transformer_ms += (time.perf_counter() - ti) * 1000.0

            latents = self.scheduler.step(
                torch.from_numpy(noise_pred), t, latents, return_dict=False)[0]

        # decode
        latents = latents.float().numpy()
        latents = _unpack_latents(latents, self.height, self.width)
        latents = (latents / self.scaling_factor) + self.shift_factor
        latents = latents.astype(np.float16)
        td = time.perf_counter()
        image = self.vae.predict(_build_inputs(
            self.vae, {"latents": latents}, ["latents"]))[0].get_data_to_numpy()
        t_vae = (time.perf_counter() - td) * 1000.0

        image = (image / 2 + 0.5).clip(0, 1)
        image = (image[0].transpose(1, 2, 0) * 255).round().astype(np.uint8)
        t_e2e = (time.perf_counter() - t_start) * 1000.0
        avg_step = total_transformer_ms / max(1, len(timesteps))
        timing = {
            "text_encode": t_text, "transformer_total": total_transformer_ms,
            "transformer_avg": avg_step, "steps": len(timesteps),
            "vae": t_vae, "e2e": t_e2e,
        }
        return image, timing


def main():
    parser = argparse.ArgumentParser(description="FLUX.1-dev MindSpore Lite inference")
    parser.add_argument("--transformer-model", required=True)
    parser.add_argument("--vae-model", required=True)
    parser.add_argument("--t5-model", required=True)
    parser.add_argument("--clip-model", required=True)
    parser.add_argument("--model-dir", default="./FLUX.1-dev",
                        help="Diffusers weights dir (for tokenizers + scheduler).")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--t5-seq-len", type=int, default=256)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--device", default="ascend", choices=["ascend", "cpu"])
    parser.add_argument("--transformer-device", type=int, default=0)
    parser.add_argument("--text-device", type=int, default=1)
    parser.add_argument("--output", default="./flux1_output.png")
    args = parser.parse_args()

    inferencer = Flux1Inferencer(
        args.transformer_model, args.vae_model, args.t5_model, args.clip_model,
        args.model_dir, device=args.device, transformer_device=args.transformer_device,
        text_device=args.text_device, height=args.height, width=args.width,
        t5_seq_len=args.t5_seq_len)

    image, timing = inferencer.infer(
        args.prompt, seed=args.seed, num_inference_steps=args.steps, guidance=args.guidance)

    Image.fromarray(image).save(args.output)
    print("\n--- Performance ---")
    print(f"  Text encode (CLIP+T5): {timing['text_encode']:.2f} ms")
    print(f"  Transformer total:     {timing['transformer_total']:.2f} ms")
    print(f"  Transformer avg/step:  {timing['transformer_avg']:.2f} ms ({timing['steps']} steps)")
    print(f"  VAE decode:            {timing['vae']:.2f} ms")
    print(f"  End-to-end:            {timing['e2e']:.2f} ms")
    print(f"  Saved image -> {args.output}")


if __name__ == "__main__":
    main()
