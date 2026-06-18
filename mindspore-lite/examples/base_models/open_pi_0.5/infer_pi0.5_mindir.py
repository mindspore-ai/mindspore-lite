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
"""
MindSpore Lite inference for pi0.5 base — optimized with zero-copy KV cache.

This is the float16 version: works with fp16 MINDIR models exported by
export_pi0.5_onnx.py. KV cache device Tensors are allocated as FLOAT16
to match the model's output dtype.

Key optimization: KV cache remains on Ascend device between denoise steps,
avoiding ~750MB Host↔Device copy per step (fp16: 36 tensors × 2 bytes × 10 steps
≈ 7.5GB total, half of the fp32 case).

Other inputs/outputs use standard numpy copy (minimal overhead).

Usage:
  python infer_pi0.5_mindir.py \
    --prefix_model ./mindir_output/prefix_encoder_graph.mindir \
    --denoise_model ./mindir_output/denoise_step.mindir \
    --device Ascend \
    --prompt "pick up the cup"
"""

import argparse
import json
import logging
import os
import time
from typing import Optional

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mindir_nocopy")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_LLM_LAYERS = 18
HEAD_DIM = 256
SIGLIP_PATCH_SIZE = 14
PATCHES_PER_IMAGE = (224 // SIGLIP_PATCH_SIZE) ** 2  # 256
IMAGE_RESOLUTION = (224, 224)

# pi0.5 base model defaults
ACTION_DIM = 32
ACTION_HORIZON = 50
MAX_TOKEN_LEN = 200
NUM_DENOISE_STEPS = 10


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------
def cast_numpy_to_tensor_dtype(arr: np.ndarray, dtype_str: str) -> np.ndarray:
    """Cast numpy array to match the target mslite Tensor dtype string.

    dtype_str is the str() of mslite DataType, e.g. 'DataType.FLOAT16'.
    """
    if "INT32" in dtype_str and arr.dtype != np.int32:
        return arr.astype(np.int32)
    if "INT64" in dtype_str and arr.dtype != np.int64:
        return arr.astype(np.int64)
    if "BOOL" in dtype_str and arr.dtype != np.bool_:
        return arr.astype(np.bool_)
    if "FLOAT16" in dtype_str and arr.dtype != np.float16:
        return arr.astype(np.float16)
    if "FLOAT32" in dtype_str and arr.dtype != np.float32:
        return arr.astype(np.float32)
    return arr


def dtype_bytes(dtype_str: str) -> int:
    """Bytes per element for a given mslite DataType string."""
    if "FLOAT16" in dtype_str:
        return 2
    if "INT32" in dtype_str or "FLOAT32" in dtype_str:
        return 4
    if "INT64" in dtype_str:
        return 8
    return 1  # BOOL, INT8, etc.


# ---------------------------------------------------------------------------
# Lightweight PaliGemma Tokenizer (pure sentencepiece, no JAX/PyTorch)
# ---------------------------------------------------------------------------
class LiteTokenizer:
    """PaliGemma tokenizer for pi0.5 — pure sentencepiece."""

    def __init__(self, max_len: int = 200, tokenizer_path: Optional[str] = None):
        self._max_len = max_len
        if tokenizer_path is None:
            tokenizer_path = self._find_tokenizer()
        if tokenizer_path is None:
            raise FileNotFoundError(
                "PaliGemma tokenizer not found.\n"
                "Download it with:\n"
                "  pip install gcsfs\n"
                "  python -c \"import gcsfs; gcsfs.GCSFileSystem(token='anon')\n"
                "    .get('gs://big_vision/paligemma_tokenizer.model',\n"
                "          'paligemma_tokenizer.model')\"\n"
                "Or use the openpi download utility, or set --tokenizer_path explicitly."
            )
        import sentencepiece
        self._sp = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path)
        logger.info("Loaded tokenizer from %s", tokenizer_path)

    @staticmethod
    def _find_tokenizer():
        """Search common cache locations for the PaliGemma tokenizer model file."""
        candidates = [
            os.path.expanduser("~/.cache/openpi/big_vision/paligemma_tokenizer.model"),
            os.path.join(os.path.dirname(__file__), "paligemma_tokenizer.model"),
            "/tmp/paligemma_tokenizer.model",
            "paligemma_tokenizer.model",
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def tokenize(self, prompt: str, state: Optional[np.ndarray] = None):
        """Tokenize prompt with optional discretized state for pi0.5."""
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")

        if state is not None:
            discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized.astype(int)))
            full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
            tokens = self._sp.encode(full_prompt, add_bos=True)
        else:
            tokens = self._sp.encode(cleaned, add_bos=True) + self._sp.encode("\n")

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            mask = [True] * tokens_len + [False] * (self._max_len - tokens_len)
            tokens = tokens + [0] * (self._max_len - tokens_len)
        else:
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.array(tokens, dtype=np.int64), np.array(mask, dtype=bool)


# ---------------------------------------------------------------------------
# Image Preprocessing (pure OpenCV)
# ---------------------------------------------------------------------------
def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize image with padding to maintain aspect ratio."""
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = height - new_h
    pad_w = width - new_w
    top, left = pad_h // 2, pad_w // 2

    result = np.zeros((height, width, 3), dtype=image.dtype)
    result[top:top + new_h, left:left + new_w] = resized
    return result


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image: uint8 HWC -> float32 NCHW normalized to [-1, 1].

    Output stays float32; cast to the tensor dtype before set_data_from_numpy.
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0 * 2.0 - 1.0
    elif image.max() > 1.0:
        image = image.astype(np.float32) / 255.0 * 2.0 - 1.0

    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # CHW -> HWC

    resized = resize_with_pad(image, IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1])
    # HWC -> NCHW
    nchw = np.transpose(resized, (2, 0, 1))[np.newaxis, ...]
    return nchw.astype(np.float32)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def load_norm_stats(checkpoint_dir: str, asset_id: Optional[str] = None):
    """Load normalization stats from checkpoint assets."""
    stats_path = os.path.join(checkpoint_dir, "assets")
    if asset_id:
        stats_path = os.path.join(stats_path, asset_id)
    stats_file = os.path.join(stats_path, "norm_stats.json")

    if not os.path.exists(stats_file):
        logger.warning("Norm stats not found at %s, using identity normalization", stats_file)
        return None

    with open(stats_file, encoding="utf-8") as f:
        raw = json.load(f)

    norm_stats = {}
    for key, val in raw.items():
        norm_stats[key] = {
            "mean": np.array(val["mean"]),
            "std": np.array(val["std"]),
            "q01": np.array(val["q01"]) if val.get("q01") is not None else None,
            "q99": np.array(val["q99"]) if val.get("q99") is not None else None,
        }
    return norm_stats


def normalize_quantile(x: np.ndarray, stats: dict) -> np.ndarray:
    """Quantile normalization: map to [-1, 1]."""
    q01 = stats["q01"][..., :x.shape[-1]]
    q99 = stats["q99"][..., :x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def unnormalize_quantile(x: np.ndarray, stats: dict) -> np.ndarray:
    """Reverse quantile normalization."""
    q01 = stats["q01"]
    q99 = stats["q99"]
    dim = q01.shape[-1]
    if dim < x.shape[-1]:
        front = (x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        return np.concatenate([front, x[..., dim:]], axis=-1)
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def pad_to_dim(x: np.ndarray, target_dim: int) -> np.ndarray:
    """Zero-pad last dimension to target_dim."""
    if x.shape[-1] < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[-1] = (0, target_dim - x.shape[-1])
        return np.pad(x, pad_width)
    return x


# ---------------------------------------------------------------------------
# MindSpore Lite Model Wrapper (with device Tensor support for KV cache)
# ---------------------------------------------------------------------------
class MindIRModel:
    """Wrapper for a MindSpore Lite MINDIR model.

    dtype-aware: numpy inputs are cast to the target mslite Tensor dtype,
    including FLOAT16 for fp16 models.
    """

    def __init__(self, model_path: str, device: str = "Ascend"):
        import mindspore_lite as msl

        self.msl = msl
        self.model = msl.Model()
        context = msl.Context()
        if device == "Ascend":
            context.target = ["ascend"]
            context.ascend.device_id = 5
            self.device_str = "ascend:5"
        elif device == "GPU":
            context.target = ["gpu"]
            self.device_str = "gpu"
        else:
            context.target = ["cpu"]
            self.device_str = "cpu"

        self.model.build_from_file(model_path, msl.ModelType.MINDIR, context)
        logger.info("Loaded MINDIR model: %s", model_path)

    def _fill_inputs(self, inputs: list[np.ndarray]):
        """Cast each numpy array to the target tensor dtype and load it."""
        ms_inputs = self.model.get_inputs()
        for i, arr in enumerate(inputs):
            target = ms_inputs[i]
            arr = cast_numpy_to_tensor_dtype(arr, str(target.dtype))
            target.set_data_from_numpy(arr)
        return ms_inputs

    def predict(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        """Run inference with numpy arrays."""
        ms_inputs = self._fill_inputs(inputs)
        outputs = self.model.predict(ms_inputs)
        return [o.get_data_to_numpy() for o in outputs]

    def predict_with_outputs(self, inputs: list[np.ndarray], output_tensors: list) -> list:
        """Run inference with pre-allocated output Tensors."""
        ms_inputs = self._fill_inputs(inputs)
        outputs = self.model.predict(ms_inputs, outputs=output_tensors)
        return outputs


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
class Timer:
    """Context manager that records elapsed time in milliseconds."""

    def __init__(self, label: str):
        self.label = label
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


# ---------------------------------------------------------------------------
# Zero-Copy KV Cache Inference Pipeline
# ---------------------------------------------------------------------------
class ZeroCopyKVCachePolicy:
    """pi0.5 base inference with zero-copy KV cache optimization.

    KV cache is kept on Ascend device as device Tensors, avoiding
    Host↔Device copy between denoise steps. Tensor dtype (FLOAT16/FLOAT32)
    is derived from the model's output template, so this works for both
    fp16 and fp32 models.
    """

    def __init__(
        self,
        prefix_model_path: str,
        denoise_model_path: str,
        action_dim: int = ACTION_DIM,
        action_horizon: int = ACTION_HORIZON,
        max_token_len: int = MAX_TOKEN_LEN,
        num_denoise_steps: int = NUM_DENOISE_STEPS,
        norm_stats: Optional[dict] = None,
        device: str = "Ascend",
        tokenizer_path: Optional[str] = None,
    ):
        self.prefix_model = MindIRModel(prefix_model_path, device)
        self.denoise_model = MindIRModel(denoise_model_path, device)
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.max_token_len = max_token_len
        self.num_denoise_steps = num_denoise_steps
        self.prefix_seq_len = 3 * PATCHES_PER_IMAGE + max_token_len  # 968
        self.norm_stats = norm_stats
        self.tokenizer = LiteTokenizer(max_token_len, tokenizer_path)

    def preprocess(self, obs: dict) -> dict:
        """Preprocess raw observation into model inputs."""
        state = np.asarray(obs.get("state", np.zeros(self.action_dim, dtype=np.float32)))
        if state.ndim == 1:
            state = state.astype(np.float32)

        if self.norm_stats and "state" in self.norm_stats:
            state = normalize_quantile(state, self.norm_stats["state"])
        state = pad_to_dim(state, self.action_dim)

        images = {}
        image_masks = {}
        for key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
            raw = obs.get(f"image/{key}")
            if raw is not None:
                images[key] = preprocess_image(np.asarray(raw))
                image_masks[key] = np.array([True])
            else:
                images[key] = np.zeros((1, 3, 224, 224), dtype=np.float32)
                image_masks[key] = np.array([False])

        prompt = obs.get("prompt", "")
        lang_tokens, lang_masks = self.tokenizer.tokenize(prompt, state)

        return {
            "state": state,
            "images": images,
            "image_masks": image_masks,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
        }

    def run_prefix_encoder(self, preprocessed: dict):
        """Run prefix encoder -> KV cache on device (zero-copy to denoise model).

        Output device Tensors are allocated with the dtype reported by the
        model's output template (FLOAT16 for fp16 models).
        """
        inputs = [
            preprocessed["images"]["base_0_rgb"],
            preprocessed["images"]["left_wrist_0_rgb"],
            preprocessed["images"]["right_wrist_0_rgb"],
            preprocessed["image_masks"]["base_0_rgb"],
            preprocessed["image_masks"]["left_wrist_0_rgb"],
            preprocessed["image_masks"]["right_wrist_0_rgb"],
            preprocessed["lang_tokens"].reshape(1, -1),
            preprocessed["lang_masks"].reshape(1, -1),
        ]

        msl = self.prefix_model.msl
        ms_inputs = self.prefix_model.model.get_inputs()
        for i, arr in enumerate(inputs):
            target = ms_inputs[i]
            arr = cast_numpy_to_tensor_dtype(arr, str(target.dtype))
            target.set_data_from_numpy(arr)

        # Pre-allocate output tensors on device, dtype derived from template
        ms_outputs = self.prefix_model.model.get_outputs()
        kv_device_outputs = []
        for out_template in ms_outputs:
            dtype_str = str(out_template.dtype)
            if "BOOL" in dtype_str:
                dt = msl.DataType.BOOL
            elif "FLOAT16" in dtype_str:
                dt = msl.DataType.FLOAT16
            else:
                dt = msl.DataType.FLOAT32

            kv_tensor = msl.Tensor(
                shape=out_template.shape,
                dtype=dt,
                device=self.prefix_model.device_str,
            )
            kv_device_outputs.append(kv_tensor)

        # Run with device output tensors
        outputs = self.prefix_model.model.predict(ms_inputs, outputs=kv_device_outputs)

        # First output is prefix_pad_masks (small, copy to host)
        prefix_pad_masks = outputs[0].get_data_to_numpy()

        # Remaining 36 outputs are KV cache (stay on device)
        kv_device = outputs[1:]

        total_mb = sum(
            int(np.prod(t.shape)) * dtype_bytes(str(t.dtype)) / 1024 / 1024
            for t in kv_device
        )
        logger.info("Prefix encoder: KV cache on device directly "
                    "(%s tensors, ~%.1f MB, dtype=%s)",
                    len(kv_device), total_mb, str(kv_device[0].dtype))
        return prefix_pad_masks, kv_device

    def create_kv_cache_device_tensors(self, kv_cache_numpy: list[np.ndarray]) -> list:
        """Create device Tensors for KV cache — keeps them on Ascend device.

        dtype follows the denoise model's KV input template, so fp16 models
        get FLOAT16 tensors.
        """
        msl = self.denoise_model.msl
        ms_inputs = self.denoise_model.model.get_inputs()
        # KV inputs start at index 3 (after x_t, timestep, prefix_pad_masks)
        kv_device = []
        for i, kv_np in enumerate(kv_cache_numpy):
            target = ms_inputs[3 + i]
            dtype_str = str(target.dtype)
            if "FLOAT16" in dtype_str:
                dt = msl.DataType.FLOAT16
            else:
                dt = msl.DataType.FLOAT32
            kv_tensor = msl.Tensor(
                shape=kv_np.shape,
                dtype=dt,
                device=self.denoise_model.device_str,
            )
            kv_tensor.set_data_from_numpy(cast_numpy_to_tensor_dtype(kv_np, dtype_str))
            kv_device.append(kv_tensor)

        total_mb = sum(
            int(np.prod(t.shape)) * dtype_bytes(str(t.dtype)) / 1024 / 1024
            for t in kv_device
        )
        logger.info("Created %s KV cache device Tensors "
                    "(~%.1f MB on device, dtype=%s)",
                    len(kv_device), total_mb, str(kv_device[0].dtype))
        return kv_device

    def run_denoise_step_with_device_kv(self, x_t, timestep, prefix_pad_masks, kv_device):
        """Run single denoising step using device KV cache Tensors.

        KV cache stays on device — no Host↔Device copy.
        Only x_t, timestep, prefix_pad_masks are sent from host (small overhead).
        dtype of host inputs is cast to the denoise model's input template.
        """
        ms_inputs = self.denoise_model.model.get_inputs()

        # Set numpy inputs (x_t, timestep, prefix_pad_masks) with proper dtype
        host_inputs = [x_t, timestep.reshape(1), prefix_pad_masks]
        for i, arr in enumerate(host_inputs):
            target = ms_inputs[i]
            arr = cast_numpy_to_tensor_dtype(arr, str(target.dtype))
            target.set_data_from_numpy(arr)

        # Use device KV tensors directly (no copy)
        denoise_inputs = ms_inputs[:3] + kv_device

        outputs = self.denoise_model.model.predict(denoise_inputs)
        return outputs[0].get_data_to_numpy()  # v_t

    def infer(self, obs: dict, noise=None) -> dict:
        """Full inference pipeline with zero-copy KV cache.

        x_t accumulation stays in float32 for numerical stability; v_t (float16
        from fp16 model output) is upcast on each Euler update.
        """
        timings = {}

        # --- Preprocess ---
        with Timer("preprocess") as t:
            preprocessed = self.preprocess(obs)
        timings["preprocess"] = t.elapsed_ms

        # --- Prefix Encoder (outputs KV cache directly to device) ---
        with Timer("prefix_encoder") as t:
            prefix_pad_masks, kv_device = self.run_prefix_encoder(preprocessed)
        timings["prefix_encoder"] = t.elapsed_ms
        logger.info("  Prefix encoder: %.1f ms", t.elapsed_ms)

        # --- Denoising Loop (KV cache stays on device) ---
        denoise_total_ms = 0.0
        action_shape = (1, self.action_horizon, self.action_dim)
        if noise is None:
            noise = np.random.normal(0.0, 1.0, action_shape).astype(np.float32)
        x_t = noise.copy()
        dt = -1.0 / self.num_denoise_steps
        time_val = 1.0

        for step in range(self.num_denoise_steps):
            timestep = np.array([time_val], dtype=np.float32)
            with Timer(f"denoise_step_{step}") as t:
                v_t = self.run_denoise_step_with_device_kv(x_t, timestep, prefix_pad_masks, kv_device)
            denoise_total_ms += t.elapsed_ms
            x_t = x_t + dt * v_t
            time_val += dt
            logger.info(
                "  Step %s/%s: %.1f ms, v_t range=[%.4f, %.4f]",
                step, self.num_denoise_steps, t.elapsed_ms, v_t.min(), v_t.max()
            )

        timings["denoise_total"] = denoise_total_ms
        timings["denoise_avg"] = denoise_total_ms / self.num_denoise_steps
        actions = x_t

        # --- Postprocess ---
        with Timer("postprocess") as t:
            actions = self.postprocess(actions)
        timings["postprocess"] = t.elapsed_ms

        # --- Totals ---
        model_total = timings["prefix_encoder"] + timings["denoise_total"]
        total = timings["preprocess"] + model_total + timings["postprocess"]

        logger.info("=" * 60)
        logger.info("TIMING SUMMARY")
        logger.info("=" * 60)
        logger.info("  Preprocess:                  %8.1f ms", timings['preprocess'])
        logger.info("  Prefix Encoder:              %8.1f ms", timings['prefix_encoder'])
        logger.info("  Denoise Loop (total):        %8.1f ms  (%s steps x %.1f ms/step)",
                    timings['denoise_total'], self.num_denoise_steps, timings['denoise_avg'])
        logger.info("  Postprocess:                 %8.1f ms", timings['postprocess'])
        logger.info("-" * 60)
        logger.info("  Model inference total:       %8.1f ms", model_total)
        logger.info("  End-to-end total:            %8.1f ms", total)
        logger.info("=" * 60)

        return {"actions": actions, "timings": timings}

    def postprocess(self, actions: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        if self.norm_stats and "actions" in self.norm_stats:
            actions = unnormalize_quantile(actions, self.norm_stats["actions"])
        return actions


# ---------------------------------------------------------------------------
# Test observation generator
# ---------------------------------------------------------------------------
def create_test_observation(seed: int = 42) -> dict:
    """Create a synthetic observation for testing with fixed random seed."""
    rng = np.random.RandomState(seed)
    return {
        "image/base_0_rgb": rng.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "image/left_wrist_0_rgb": rng.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "image/right_wrist_0_rgb": rng.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "state": rng.uniform(-1, 1, 32).astype(np.float32),
        "prompt": "pick up the cup",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run pi0.5 base inference with zero-copy KV cache (float16)")
    parser.add_argument("--prefix_model",
                        default=os.path.join(os.path.dirname(__file__),
                                             "mindir_output_fp16", "prefix_encoder_graph.mindir"),
                        help="Path to prefix_encoder MindIR")
    parser.add_argument("--denoise_model",
                        default=os.path.join(os.path.dirname(__file__),
                                             "mindir_output_fp16", "denoise_step.mindir"),
                        help="Path to denoise_step MindIR")
    parser.add_argument("--device", default="Ascend", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument("--prompt", default="pick up the cup", help="Task prompt")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tokenizer_path", default=None, help="Path to paligemma_tokenizer.model")
    parser.add_argument("--output", default="mindir_nocopy_result_fp16.npy", help="Output file")
    args = parser.parse_args()

    np.random.seed(args.seed)

    policy = ZeroCopyKVCachePolicy(
        prefix_model_path=args.prefix_model,
        denoise_model_path=args.denoise_model,
        num_denoise_steps=args.num_steps,
        device=args.device,
        tokenizer_path=args.tokenizer_path,
    )

    obs = create_test_observation(seed=args.seed)
    obs["prompt"] = args.prompt

    logger.info("Starting MindIR inference with zero-copy KV cache (float16)...")
    result = policy.infer(obs)

    actions = result["actions"]
    logger.info("Actions shape: %s", actions.shape)
    logger.info("Actions sample (first 3 steps, first 8 dims):\n%s", actions[0, :3, :8])
    logger.info("Actions range: [%.4f, %.4f]", actions.min(), actions.max())

    np.save(args.output, actions)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
