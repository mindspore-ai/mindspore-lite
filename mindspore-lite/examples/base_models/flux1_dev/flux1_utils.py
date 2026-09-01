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
"""Shared NumPy utilities for FLUX.1-dev ONNX and MindSpore Lite inference."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


VAE_SCALE_FACTOR = 8
LATENT_PACK_FACTOR = 2
LATENT_CHANNELS = 16
CLIP_SEQUENCE_LENGTH = 77
DEFAULT_T5_SEQUENCE_LENGTH = 256
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512


@dataclass(frozen=True)
class FluxShape:
    """Fixed input shapes used for export, conversion, and inference."""

    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    t5_sequence_length: int = DEFAULT_T5_SEQUENCE_LENGTH

    def __post_init__(self):
        """Validate dimensions required by latent packing."""
        if self.height <= 0 or self.width <= 0 or self.t5_sequence_length <= 0:
            raise ValueError("height, width, and t5_sequence_length must be positive")
        divisor = VAE_SCALE_FACTOR * LATENT_PACK_FACTOR
        if self.height % divisor or self.width % divisor:
            raise ValueError(f"height and width must be divisible by {divisor}")

    @property
    def latent_height(self):
        """Return the unpacked latent height."""
        return self.height // VAE_SCALE_FACTOR

    @property
    def latent_width(self):
        """Return the unpacked latent width."""
        return self.width // VAE_SCALE_FACTOR

    @property
    def token_height(self):
        """Return the packed latent token-grid height."""
        return self.latent_height // LATENT_PACK_FACTOR

    @property
    def token_width(self):
        """Return the packed latent token-grid width."""
        return self.latent_width // LATENT_PACK_FACTOR

    @property
    def image_sequence_length(self):
        """Return the number of packed image tokens."""
        return self.token_height * self.token_width


def load_json(path):
    """Load a UTF-8 JSON object from path."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_scheduler_config(model_dir):
    """Load the FlowMatch scheduler configuration from a Diffusers model."""
    path = Path(model_dir) / "scheduler" / "scheduler_config.json"
    return load_json(path)


def load_vae_scaling(model_dir):
    """Return the VAE scaling and shift factors from its configuration."""
    config = load_json(Path(model_dir) / "vae" / "config.json")
    scaling = float(config.get("scaling_factor", 0.3611))
    shift = float(config.get("shift_factor", 0.1159))
    return scaling, shift


def calculate_shift(image_sequence_length, config):
    """Calculate the resolution-dependent FLUX timestep shift."""
    base_length = int(config.get("base_image_seq_len", 256))
    max_length = int(config.get("max_image_seq_len", 4096))
    base_shift = float(config.get("base_shift", 0.5))
    max_shift = float(config.get("max_shift", 1.15))
    slope = (max_shift - base_shift) / (max_length - base_length)
    intercept = base_shift - slope * base_length
    return image_sequence_length * slope + intercept


def _shift_sigmas(sigmas, image_sequence_length, config):
    """Apply the scheduler's dynamic or constant sigma shift."""
    if not bool(config.get("use_dynamic_shifting", False)):
        shift = float(config.get("shift", 1.0))
        return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    mu = calculate_shift(image_sequence_length, config)
    shift_type = config.get("time_shift_type", "exponential")
    numerator = np.exp(mu) if shift_type == "exponential" else mu
    if shift_type not in ("exponential", "linear"):
        raise ValueError(f"unsupported time_shift_type: {shift_type}")
    return numerator / (numerator + (1.0 / sigmas - 1.0))


def _stretch_to_terminal(sigmas, terminal):
    """Stretch a sigma schedule to end at a configured terminal value."""
    one_minus_sigma = 1.0 - sigmas
    scale = one_minus_sigma[-1] / (1.0 - terminal)
    return 1.0 - one_minus_sigma / scale


def build_flow_schedule(num_steps, image_sequence_length, config):
    """Build FLUX FlowMatchEuler timesteps and sigmas without PyTorch."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    unsupported = (
        "use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas", "stochastic_sampling",
    )
    enabled = [name for name in unsupported if bool(config.get(name, False))]
    if enabled:
        raise ValueError(f"unsupported scheduler options: {', '.join(enabled)}")
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps, dtype=np.float32)
    sigmas = _shift_sigmas(sigmas, image_sequence_length, config)
    terminal = config.get("shift_terminal")
    if terminal is not None:
        sigmas = _stretch_to_terminal(sigmas, float(terminal))
    train_steps = float(config.get("num_train_timesteps", 1000))
    timesteps = (sigmas * train_steps).astype(np.float32)
    sigmas = np.concatenate((sigmas, np.zeros(1, dtype=np.float32))).astype(np.float32)
    return timesteps, sigmas


def euler_step(sample, model_output, sigma, next_sigma):
    """Apply one deterministic FlowMatchEuler update in float32."""
    sample_fp32 = np.asarray(sample, dtype=np.float32)
    output_fp32 = np.asarray(model_output, dtype=np.float32)
    return sample_fp32 + (float(next_sigma) - float(sigma)) * output_fp32


def pack_latents(latents):
    """Pack [B,C,H,W] latents into [B,H/2*W/2,C*4] tokens."""
    batch, channels, height, width = latents.shape
    packed = latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
    packed = packed.transpose(0, 2, 4, 1, 3, 5)
    return packed.reshape(batch, (height // 2) * (width // 2), channels * 4)


def unpack_latents(latents, shape):
    """Unpack latent tokens into [B,C,H/8,W/8] for VAE decoding."""
    batch, _, channels = latents.shape
    unpacked = latents.reshape(batch, shape.token_height, shape.token_width, channels // 4, 2, 2)
    unpacked = unpacked.transpose(0, 3, 1, 4, 2, 5)
    return unpacked.reshape(batch, channels // 4, shape.latent_height, shape.latent_width)


def make_image_ids(shape, dtype=np.float32):
    """Create FLUX packed image position IDs with shape [image_seq, 3]."""
    ids = np.zeros((shape.token_height, shape.token_width, 3), dtype=dtype)
    ids[..., 1] = np.arange(shape.token_height, dtype=dtype)[:, None]
    ids[..., 2] = np.arange(shape.token_width, dtype=dtype)[None, :]
    return ids.reshape(shape.image_sequence_length, 3)


def make_text_ids(shape, dtype=np.float32):
    """Create zero FLUX text position IDs with shape [text_seq, 3]."""
    return np.zeros((shape.t5_sequence_length, 3), dtype=dtype)


def make_initial_latents(shape, seed):
    """Generate deterministic Gaussian latents and pack them for the transformer."""
    rng = np.random.default_rng(seed)
    size = (1, LATENT_CHANNELS, shape.latent_height, shape.latent_width)
    return pack_latents(rng.standard_normal(size).astype(np.float32))


def postprocess_image(image):
    """Convert a decoded image tensor [1,3,H,W] to uint8 HWC."""
    image = (np.asarray(image, dtype=np.float32) / 2.0 + 0.5).clip(0.0, 1.0)
    return np.rint(image[0].transpose(1, 2, 0) * 255.0).astype(np.uint8)
