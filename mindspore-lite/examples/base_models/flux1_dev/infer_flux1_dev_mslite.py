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
"""Run FLUX.1-dev text-to-image inference with MindSpore Lite on Ascend."""

import argparse
import time
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

import mindspore_lite as mslite
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from flux1_utils import (
    CLIP_SEQUENCE_LENGTH,
    FluxShape,
    build_flow_schedule,
    euler_step,
    load_scheduler_config,
    load_vae_scaling,
    make_image_ids,
    make_initial_latents,
    make_text_ids,
    pack_latents,
    postprocess_image,
    unpack_latents,
)


@dataclass(frozen=True)
class RuntimeConfig:
    """Model locations, fixed shape, and component placement."""

    mindir_dir: Path
    model_dir: Path
    shape: FluxShape
    transformer_config: Path
    transformer_device: int = 0
    text_device: int = 1
    vae_device: int = 0


@dataclass(frozen=True)
class GenerationConfig:
    """Inputs for one deterministic image generation."""

    prompt: str
    output: Path
    seed: int = 42
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    latents_npy: Path = None


def _resolve_mindir(directory, stem):
    """Resolve external-weight and single-file converter output names."""
    candidates = [directory / f"{stem}_graph.mindir", directory / f"{stem}.mindir"]
    chosen = next((path for path in candidates if path.exists()), None)
    if chosen is None:
        names = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"MindIR not found; checked: {names}")
    return chosen


def _build_model(mindir_path, device_id, config_path=None, online_ge=False):
    """Build one MindIR model on an Ascend device."""
    mindir_path = mindir_path.resolve()
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = int(device_id)
    if online_ge:
        context.ascend.provider = "ge"
        context.ascend.precision_mode = "preferred_optimal"
    model = mslite.Model()
    if config_path is None:
        model.build_from_file(str(mindir_path), mslite.ModelType.MINDIR, context)
    else:
        config_path = config_path.resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"runtime config not found: {config_path}")
        parser = ConfigParser(interpolation=None)
        parser.read(config_path, encoding="utf-8")
        mixlist = parser.get(
            "ascend_context", "mixprecision_list_path", fallback="",
        ).strip().strip('"')
        config_dict = None
        if mixlist:
            mixlist_path = Path(mixlist)
            if not mixlist_path.is_absolute():
                mixlist_path = config_path.parent / mixlist_path
            mixlist_path = mixlist_path.resolve()
            if not mixlist_path.is_file():
                raise FileNotFoundError(f"mix-precision list not found: {mixlist_path}")
            config_dict = {
                "ascend_context": {"mixprecision_list_path": str(mixlist_path)},
            }
        model.build_from_file(
            str(mindir_path),
            mslite.ModelType.MINDIR,
            context,
            config_path=str(config_path),
            config_dict=config_dict,
        )
    return model


def _np_to_input(tensor_info, array):
    """Cast an array to the exact dtype expected by a MindIR input."""
    dtype_map = {
        mslite.DataType.FLOAT16: np.float16,
        mslite.DataType.FLOAT32: np.float32,
        mslite.DataType.INT32: np.int32,
        mslite.DataType.INT64: np.int64,
        mslite.DataType.BOOL: np.bool_,
    }
    dtype = dtype_map.get(tensor_info.dtype)
    return np.asarray(array).astype(dtype) if dtype is not None else np.asarray(array)


def _run_model(model, feed_dict, preferred_order):
    """Match model inputs by name, with export order as a compatibility fallback."""
    inputs = model.get_inputs()
    if all(getattr(item, "name", None) in feed_dict for item in inputs):
        arrays = [feed_dict[item.name] for item in inputs]
    else:
        arrays = [feed_dict[name] for name in preferred_order]
    tensors = [mslite.Tensor(_np_to_input(info, array))
               for info, array in zip(inputs, arrays)]
    return [output.get_data_to_numpy() for output in model.predict(tensors)]


class Flux1Inferencer:
    """Four-component FLUX.1-dev pipeline backed by MindSpore Lite."""

    def __init__(self, config):
        """Load MindIR components, tokenizers, and Diffusers JSON configuration."""
        self.config = config
        directory = config.mindir_dir
        print("[load] transformer")
        self.transformer = _build_model(
            _resolve_mindir(directory, "flux1_transformer"), config.transformer_device,
            config.transformer_config, online_ge=True,
        )
        print("[load] VAE decoder")
        self.vae = _build_model(
            _resolve_mindir(directory, "flux1_vae_decoder"), config.vae_device,
        )
        print("[load] T5 encoder")
        self.t5 = _build_model(
            _resolve_mindir(directory, "flux1_t5_encoder"), config.text_device,
        )
        print("[load] CLIP encoder")
        self.clip = _build_model(
            _resolve_mindir(directory, "flux1_clip_encoder"), config.text_device,
        )
        self.t5_tokenizer = AutoTokenizer.from_pretrained(config.model_dir / "tokenizer_2")
        self.clip_tokenizer = AutoTokenizer.from_pretrained(config.model_dir / "tokenizer")
        self.scheduler_config = load_scheduler_config(config.model_dir)
        self.vae_scaling, self.vae_shift = load_vae_scaling(config.model_dir)

    def _encode_prompt(self, prompt):
        """Run the T5-XXL and CLIP-L prompt encoders."""
        shape = self.config.shape
        t5_tokens = self.t5_tokenizer(
            prompt,
            padding="max_length",
            max_length=shape.t5_sequence_length,
            truncation=True,
            return_tensors="np",
        )
        clip_tokens = self.clip_tokenizer(
            prompt,
            padding="max_length",
            max_length=CLIP_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="np",
        )
        t5_ids = np.asarray(t5_tokens["input_ids"], dtype=np.int64)
        clip_ids = np.asarray(clip_tokens["input_ids"], dtype=np.int64)
        prompt_embeds = _run_model(
            self.t5, {"input_ids": t5_ids}, ["input_ids"],
        )[0].astype(np.float32)
        pooled = _run_model(
            self.clip, {"input_ids": clip_ids}, ["input_ids"],
        )[0].astype(np.float32)
        return prompt_embeds, pooled

    def _load_latents(self, request):
        """Load reproducible latents or generate Gaussian packed latents."""
        shape = self.config.shape
        if request.latents_npy is None:
            return make_initial_latents(shape, request.seed)
        latents = np.load(request.latents_npy).astype(np.float32)
        unpacked_shape = (1, 16, shape.latent_height, shape.latent_width)
        packed_shape = (1, shape.image_sequence_length, 64)
        if latents.shape == unpacked_shape:
            return pack_latents(latents)
        if latents.shape != packed_shape:
            raise ValueError(f"latents shape must be {unpacked_shape} or {packed_shape}")
        return latents

    def _denoise(self, latents, prompt_embeds, pooled, request):
        """Run deterministic FlowMatchEuler denoising with a float32 CPU state."""
        shape = self.config.shape
        timesteps, sigmas = build_flow_schedule(
            request.num_inference_steps,
            shape.image_sequence_length,
            self.scheduler_config,
        )
        image_ids = make_image_ids(shape)
        text_ids = make_text_ids(shape)
        guidance = np.array([request.guidance_scale], dtype=np.float32)
        total_ms = 0.0
        order = ["hidden_states", "encoder_hidden_states", "pooled_projections",
                 "timestep", "img_ids", "txt_ids", "guidance"]
        for index, timestep in enumerate(timesteps):
            feed = {
                "hidden_states": latents,
                "encoder_hidden_states": prompt_embeds,
                "pooled_projections": pooled,
                "timestep": np.array([timestep / 1000.0], dtype=np.float32),
                "img_ids": image_ids,
                "txt_ids": text_ids,
                "guidance": guidance,
            }
            start = time.perf_counter()
            noise_pred = _run_model(self.transformer, feed, order)[0]
            step_ms = (time.perf_counter() - start) * 1000.0
            total_ms += step_ms
            latents = euler_step(latents, noise_pred, sigmas[index], sigmas[index + 1])
            print(f"[infer] step {index + 1:02d}/{len(timesteps)}: {step_ms:.2f} ms")
        return latents, total_ms

    def generate(self, request):
        """Generate and save one image, returning stage timings."""
        total_start = time.perf_counter()
        start = time.perf_counter()
        prompt_embeds, pooled = self._encode_prompt(request.prompt)
        text_ms = (time.perf_counter() - start) * 1000.0
        latents = self._load_latents(request)
        latents, transformer_ms = self._denoise(latents, prompt_embeds, pooled, request)
        latents = unpack_latents(latents, self.config.shape)
        latents = latents / self.vae_scaling + self.vae_shift
        start = time.perf_counter()
        image = _run_model(self.vae, {"latents": latents}, ["latents"])[0]
        vae_ms = (time.perf_counter() - start) * 1000.0
        image = postprocess_image(image)
        request.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(request.output)
        return {
            "text_encode_ms": text_ms,
            "transformer_total_ms": transformer_ms,
            "transformer_avg_step_ms": transformer_ms / request.num_inference_steps,
            "vae_decode_ms": vae_ms,
            "end_to_end_ms": (time.perf_counter() - total_start) * 1000.0,
        }


def _parse_args():
    """Parse inference options."""
    parser = argparse.ArgumentParser(description="FLUX.1-dev MindSpore Lite inference")
    parser.add_argument(
        "--mindir-dir", required=True, help="directory containing the four MindIR models",
    )
    parser.add_argument("--model-dir", required=True, help="local FLUX.1-dev Diffusers directory")
    parser.add_argument("--prompt", default="A cat holding a sign that says MindSpore Lite")
    parser.add_argument("--output", default="./flux1_output.png")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--t5-seq-len", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latents-npy", type=Path, default=None)
    parser.add_argument(
        "--transformer-config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "flux1_transformer_runtime.config",
        help="online GE runtime config for the Transformer",
    )
    parser.add_argument("--transformer-device", type=int, default=0)
    parser.add_argument("--text-device", type=int, default=1)
    parser.add_argument("--vae-device", type=int, default=0)
    return parser.parse_args()


def main():
    """Build the pipeline, generate one image, and print measured timings."""
    args = _parse_args()
    runtime = RuntimeConfig(
        mindir_dir=Path(args.mindir_dir),
        model_dir=Path(args.model_dir),
        shape=FluxShape(args.height, args.width, args.t5_seq_len),
        transformer_config=args.transformer_config,
        transformer_device=args.transformer_device,
        text_device=args.text_device,
        vae_device=args.vae_device,
    )
    request = GenerationConfig(
        args.prompt,
        Path(args.output),
        args.seed,
        args.num_inference_steps,
        args.guidance_scale,
        args.latents_npy,
    )
    timing = Flux1Inferencer(runtime).generate(request)
    print("\n--- Performance ---")
    print(f"  Text encode:          {timing['text_encode_ms']:.2f} ms")
    print(f"  Transformer total:    {timing['transformer_total_ms']:.2f} ms")
    print(f"  Transformer avg/step: {timing['transformer_avg_step_ms']:.2f} ms")
    print(f"  VAE decode:           {timing['vae_decode_ms']:.2f} ms")
    print(f"  End-to-end:           {timing['end_to_end_ms']:.2f} ms")
    print(f"  Saved image:          {request.output}")


if __name__ == "__main__":
    main()
