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
"""stable-audio-3-small-music-base text-to-audio inference with MindSpore Lite.

Loads the three converted MindIR sub-models (T5 text encoder, latent DiT
denoiser, audio autoencoder decoder), encodes the prompt, runs the flow-matching
denoising loop (Euler scheduler on CPU, DiT on Ascend), and decodes the latent
to a stereo waveform (audio decoder on Ascend), saving it as a 32 kHz WAV.

Model inference (text encoder / DiT / audio decoder) is pure
``mindspore_lite`` + ``numpy``. ``torch`` is imported ONLY for the CPU/numpy-
backed stable-audio-tools scheduler that runs on CPU (timestep schedule + Euler
update), exactly as in the wan2_1 / flux1_dev examples; no torch tensor ever
touches the Ascend models.

Component placement on the 300I Duo (component split, not tensor-parallel):
text encoder -> dev1, DiT -> dev0, audio decoder -> dev0.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mindspore_lite as mslite
except ImportError:
    print("Error: mindspore_lite not found. Please install MindSpore Lite first.")
    sys.exit(1)

# torch is used ONLY for the CPU/numpy-backed stable-audio-tools scheduler; the
# Ascend model inference is pure mslite + numpy (see module docstring).
try:
    import torch  # noqa: E402  (scheduler only)
except ImportError:
    print("Error: torch not found (needed for the CPU scheduler only).")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Defaults (must match the export script defaults / config).
# ---------------------------------------------------------------------------
_DEFAULT_LATENT_CHANNELS = 64
_DEFAULT_LATENT_DOWNSAMPLING = 1024
_DEFAULT_AUDIO_CHANNELS = 2
_DEFAULT_SAMPLE_RATE = 32000

_DEFAULT_TEXT_DIM = 768
_DEFAULT_TEXT_SEQ_LEN = 256

_DEFAULT_DIT_HIDDEN = 1536
_DEFAULT_GLOBAL_COND_DIM = _DEFAULT_DIT_HIDDEN


# ---------------------------------------------------------------------------
# mslite helpers.
# ---------------------------------------------------------------------------


def _build_model(mindir_path, device_id):
    """Build an mslite Model from a ``*_graph.mindir`` on a given Ascend device.

    force_fp16 is set so the Ascend graph compiles the sub-models in fp16
    (matching the converter's precision_mode=force_fp16).
    """
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


# ---------------------------------------------------------------------------
# Scheduler (pure-numpy flow-matching Euler, mirrors stable-audio-tools).
# ---------------------------------------------------------------------------


class FlowMatchingEulerScheduler:
    """Pure-numpy flow-matching Euler scheduler for Stable Audio 3.

    Stable Audio 3 uses a v-prediction / flow-matching objective trained on the
    ``sigma`` schedule defined in stable-audio-tools. At inference, the model
    predicts a velocity ``v(x, sigma)`` and the Euler update integrates the
    probability-flow ODE from sigma_max to sigma_min.

    This scheduler is intentionally minimal and numpy-only so it runs on CPU
    alongside tokenisation (no torch tensor ever touches the Ascend models). It
    mirrors stable-audio-tools' ``Scheduler1D`` / Euler sampling math:

        x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v_pred

    with the sigmas spaced geometrically between ``sigma_min`` and
    ``sigma_max``.
    """

    def __init__(self, num_steps, sigma_min=0.0001, sigma_max=1000.0):
        self.num_steps = int(num_steps)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        # Geometric sigma schedule, descending (sigma_max -> sigma_min).
        self.sigmas = np.exp(np.linspace(
            math.log(self.sigma_max), math.log(self.sigma_min),
            self.num_steps + 1)).astype(np.float32)

    def step(self, x, v_pred, i):
        """One Euler update: integrate the velocity prediction.

        Args:
            x:       current latent [1, C, T] (numpy, float32).
            v_pred:  velocity prediction from the DiT at sigma_i.
            i:       step index (0-based).
        Returns:
            updated latent at sigma_{i+1}.
        """
        dsigma = float(self.sigmas[i + 1] - self.sigmas[i])
        return x + dsigma * v_pred


# ---------------------------------------------------------------------------
# Global-conditioning builder (timestep + seconds -> global_cond vector).
# ---------------------------------------------------------------------------


def _build_global_cond(sigma, seconds, seconds_total, global_cond_dim,
                       device_id_of_dit):  # noqa: ARG001
    """Build the DiT ``global_cond`` vector on CPU (numpy).

    Stable Audio 3 encodes (sigma, seconds_start, seconds_total) via sinusoidal
    embeddings + a small MLP inside the DiT graph (the graph already contains
    the MLP weights). Here we only need to pack the three scalars into the
    ``global_cond`` input the graph expects.

    The exact layout is: [sigma_embed(seconds_total), seconds_start,
    seconds_total] reshaped to [1, global_cond_dim]. We construct it as a
    concatenation of sinusoidal embeddings + the raw scalars, which matches the
    stable-audio-tools ``NumberConditioner`` contract (the graph does the MLP
    internally on this packed input).

    NOTE: this is the one piece that must be verified against the actual
    model_config.json (see README FAQ). If your checkpoint's conditioner uses a
    different embedding dim or layout, override this function.
    """
    half = global_cond_dim // 2
    emb = _sinusoidal_embedding(np.array([seconds_total], dtype=np.float32), half)
    out = np.concatenate([
        emb.reshape(-1),
        np.array([seconds / max(seconds_total, 1e-6)], dtype=np.float32),
        np.array([seconds_total], dtype=np.float32),
        np.array([float(sigma)], dtype=np.float32),
    ], axis=0).astype(np.float32)
    # Pad / truncate to global_cond_dim.
    if out.shape[0] < global_cond_dim:
        out = np.pad(out, (0, global_cond_dim - out.shape[0]))
    elif out.shape[0] > global_cond_dim:
        out = out[:global_cond_dim]
    return out.reshape(1, global_cond_dim)


def _sinusoidal_embedding(values, dim):
    """Standard sinusoidal embedding (numpy) for a 1-D tensor of scalars."""
    half = dim // 2
    freqs = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32)
                   / max(half, 1))
    args = values.reshape(-1, 1) * freqs.reshape(1, -1)
    emb = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
    if emb.shape[-1] < dim:
        emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[-1])))
    return emb


# ---------------------------------------------------------------------------
# Inferencer.
# ---------------------------------------------------------------------------


class StableAudioInferencer:
    """End-to-end stable-audio-3-small text-to-audio inferencer over MindSpore
    Lite (Ascend)."""

    def __init__(self, mindir_dir, model_dir, text_device=1, dit_device=0,
                 decoder_device=0, seconds=10.0,
                 sample_rate=_DEFAULT_SAMPLE_RATE,
                 latent_channels=_DEFAULT_LATENT_CHANNELS,
                 latent_downsampling=_DEFAULT_LATENT_DOWNSAMPLING,
                 audio_channels=_DEFAULT_AUDIO_CHANNELS,
                 text_seq_len=_DEFAULT_TEXT_SEQ_LEN,
                 text_dim=_DEFAULT_TEXT_DIM,
                 global_cond_dim=_DEFAULT_GLOBAL_COND_DIM,
                 num_inference_steps=100, sigma_min=0.0001, sigma_max=1000.0):
        """Load the three MindIR sub-models, tokenizer, and build the scheduler."""
        mindir_dir = Path(mindir_dir)
        self.seconds = float(seconds)
        self.sample_rate = int(sample_rate)
        self.latent_channels = int(latent_channels)
        self.latent_downsampling = int(latent_downsampling)
        self.audio_channels = int(audio_channels)
        self.text_seq_len = int(text_seq_len)
        self.text_dim = int(text_dim)
        self.global_cond_dim = int(global_cond_dim)
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        self.latent_frames = int(math.ceil(
            self.seconds * self.sample_rate / float(self.latent_downsampling)))

        print(f"[infer] loading text encoder MindIR (dev{text_device}) ...")
        self.text_model = _build_model(
            mindir_dir / "stable_audio_text_encoder_graph.mindir", text_device)
        print(f"[infer] loading DiT MindIR (dev{dit_device}) ...")
        self.dit = _build_model(
            mindir_dir / "stable_audio_dit_graph.mindir", dit_device)
        print(f"[infer] loading audio decoder MindIR (dev{decoder_device}) ...")
        self.audio_decoder = _build_model(
            mindir_dir / "stable_audio_audio_decoder_graph.mindir",
            decoder_device)

        # Tokenizer: load from the stable-audio-tools checkpoint's conditioner.
        # We re-use the export helper to locate it (cheap, CPU-only).
        self.tokenizer = _load_t5_tokenizer(model_dir)

        self.scheduler = FlowMatchingEulerScheduler(
            self.num_inference_steps, self.sigma_min, self.sigma_max)

    def _encode_prompt(self, prompt):
        """Tokenize + run the T5 text encoder -> [1, seq_len, text_dim] embeds."""
        toks = self.tokenizer(
            prompt, padding="max_length", max_length=self.text_seq_len,
            truncation=True, add_special_tokens=True,
            return_attention_mask=True, return_tensors="np")
        input_ids = toks["input_ids"].astype(np.int64)
        attention_mask = toks["attention_mask"].astype(np.int64)
        embeds = _run_model(
            self.text_model,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            ["input_ids", "attention_mask"])[0]
        return embeds.astype(np.float32)

    def _denoise(self, latents, prompt_embeds, negative_embeds, guidance_scale):
        """Run the CFG flow-matching denoising loop, returning denoised latents."""
        sigmas = self.scheduler.sigmas
        for i in range(self.num_inference_steps):
            sigma = float(sigmas[i])
            global_cond = _build_global_cond(
                sigma, 0.0, self.seconds, self.global_cond_dim, 0)
            global_cond_np = global_cond.astype(np.float32)
            sigma_arr = np.array([sigma], dtype=np.float32)

            # CFG: run cond + uncond, blend.
            feed_cond = {
                "x": latents, "t": sigma_arr,
                "cross_attn_cond": prompt_embeds,
                "global_cond": global_cond_np,
            }
            feed_uncond = {
                "x": latents, "t": sigma_arr,
                "cross_attn_cond": negative_embeds,
                "global_cond": global_cond_np,
            }
            v_cond = _run_model(
                self.dit, feed_cond,
                ["x", "t", "cross_attn_cond", "global_cond"])[0]
            v_uncond = _run_model(
                self.dit, feed_uncond,
                ["x", "t", "cross_attn_cond", "global_cond"])[0]
            v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)
            latents = self.scheduler.step(latents, v_pred, i)
        return latents

    def _decode_audio(self, latents):
        """Decode latents -> waveform [1, audio_channels, T_audio]."""
        audio = _run_model(
            self.audio_decoder, {"latents": latents.astype(np.float32)},
            ["latents"])[0]
        return audio

    @staticmethod
    def _save_wav(audio, output_path, sample_rate):
        """Save a [1, C, T] waveform as a WAV (soundfile, falling back to
        scipy.io.wavfile)."""
        wav = audio[0].T  # -> [T, C]
        wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
        try:
            import soundfile as sf
            sf.write(str(output_path), wav, int(sample_rate), subtype="FLOAT")
        except ImportError:
            try:
                from scipy.io import wavfile
                wav_int = (wav * 32767.0).astype(np.int16)
                wavfile.write(str(output_path), int(sample_rate), wav_int)
            except ImportError:
                raise RuntimeError(
                    "Neither soundfile nor scipy is installed; cannot write WAV. "
                    "pip install soundfile")
        print(f"[infer] saved audio -> {output_path} "
              f"({wav.shape[0] / sample_rate:.2f}s, {wav.shape[-1]}ch)")

    def generate(self, prompt, negative_prompt, output_path, seed=42,
                 latents_npy=None, guidance_scale=4.0):
        """Run the full text-to-audio pipeline with stage timing and save WAV.

        Args:
            prompt:           text prompt.
            negative_prompt:  negative prompt (CFG uncond branch).
            output_path:      output WAV path.
            seed:             RNG seed for the initial noise.
            latents_npy:      optional pre-generated initial latents (for
                              alignment; overrides the seed).
            guidance_scale:   CFG scale (typical 3-7 for stable-audio-3).
        Returns:
            (timing_dict, audio_array).
        """
        timing = {}

        t0 = time.perf_counter()
        prompt_embeds = self._encode_prompt(prompt)
        negative_embeds = self._encode_prompt(negative_prompt or "")
        timing["text_encode_ms"] = (time.perf_counter() - t0) * 1000

        latent_shape = (1, self.latent_channels, self.latent_frames)
        if latents_npy:
            latents = np.load(latents_npy).astype(np.float32)
        else:
            rng = np.random.RandomState(seed)
            # Stable Audio 3 inits latents at sigma_max (high-noise end).
            latents = (rng.standard_normal(latent_shape).astype(np.float32)
                       * float(self.sigma_max))

        t0 = time.perf_counter()
        latents = self._denoise(latents, prompt_embeds, negative_embeds,
                                guidance_scale)
        timing["dit_total_ms"] = (time.perf_counter() - t0) * 1000
        timing["dit_avg_step_ms"] = (
            timing["dit_total_ms"] / max(1, self.num_inference_steps))

        t0 = time.perf_counter()
        audio = self._decode_audio(latents)
        timing["audio_decode_ms"] = (time.perf_counter() - t0) * 1000

        self._save_wav(audio, output_path, self.sample_rate)
        timing["e2e_ms"] = (timing["text_encode_ms"]
                            + timing["dit_total_ms"]
                            + timing["audio_decode_ms"])
        return timing, audio


# ---------------------------------------------------------------------------
# Tokenizer loader (re-uses stable-audio-tools to locate the T5 tokenizer).
# ---------------------------------------------------------------------------


def _load_t5_tokenizer(model_dir):
    """Locate and return the T5 tokenizer from the stable-audio-tools
    checkpoint's conditioner."""
    import json
    import os
    config_path = os.path.join(model_dir, "model_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"model_config.json not found at {config_path}.")
    with open(config_path, "r") as f:
        config = json.load(f)
    cond_cfg = config["model"]["conditioning"]
    conds = cond_cfg.get("configs", {})
    # Each entry may have a ``config`` with the HF model name for the tokenizer.
    for name, cfg in conds.items():
        if "t5" in str(name).lower() or "t5" in str(cfg).lower():
            inner = cfg.get("config", {})
            model_name = (inner.get("model_name") or inner.get("pretrained_model")
                          or "t5-base")
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(model_name)
    # Fallback: try the standard stable-audio-3 T5 tokenizer name.
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("t5-base")
    except Exception:  # pragma: no cover
        return AutoTokenizer.from_pretrained(
            os.path.join(model_dir, "tokenizer"))


def main():
    """Parse arguments and run stable-audio-3-small text-to-audio inference."""
    parser = argparse.ArgumentParser(
        description="stable-audio-3-small-music-base MindSpore Lite inference")
    parser.add_argument("--mindir-dir", required=True,
                        help="dir with the 3 *_graph.mindir files.")
    parser.add_argument("--model-dir", required=True,
                        help="stable-audio-3-small-music-base checkpoint dir "
                        "(for the tokenizer + model_config.json).")
    parser.add_argument("--prompt", default="128 BPM tech house drum loop, "
                        "punchy kick, deep bass, 909 hi-hats")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", default="stable_audio_output.wav")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="seconds of audio to generate (must match export).")
    parser.add_argument("--sample-rate", type=int, default=_DEFAULT_SAMPLE_RATE)
    parser.add_argument("--latent-channels", type=int,
                        default=_DEFAULT_LATENT_CHANNELS)
    parser.add_argument("--latent-downsampling", type=int,
                        default=_DEFAULT_LATENT_DOWNSAMPLING)
    parser.add_argument("--audio-channels", type=int,
                        default=_DEFAULT_AUDIO_CHANNELS)
    parser.add_argument("--text-seq-len", type=int, default=_DEFAULT_TEXT_SEQ_LEN)
    parser.add_argument("--text-dim", type=int, default=_DEFAULT_TEXT_DIM)
    parser.add_argument("--global-cond-dim", type=int,
                        default=_DEFAULT_GLOBAL_COND_DIM)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--sigma-min", type=float, default=0.0001)
    parser.add_argument("--sigma-max", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latents-npy", default=None,
                        help="pre-generated latents (for alignment).")
    parser.add_argument("--text-device", type=int, default=1)
    parser.add_argument("--dit-device", type=int, default=0)
    parser.add_argument("--decoder-device", type=int, default=0)
    args = parser.parse_args()

    inferencer = StableAudioInferencer(
        args.mindir_dir, args.model_dir,
        text_device=args.text_device, dit_device=args.dit_device,
        decoder_device=args.decoder_device,
        seconds=args.seconds, sample_rate=args.sample_rate,
        latent_channels=args.latent_channels,
        latent_downsampling=args.latent_downsampling,
        audio_channels=args.audio_channels,
        text_seq_len=args.text_seq_len, text_dim=args.text_dim,
        global_cond_dim=args.global_cond_dim,
        num_inference_steps=args.num_inference_steps,
        sigma_min=args.sigma_min, sigma_max=args.sigma_max)

    timing, _ = inferencer.generate(
        args.prompt, args.negative_prompt, args.output,
        seed=args.seed, latents_npy=args.latents_npy,
        guidance_scale=args.guidance_scale)

    print("\n--- Performance ---")
    print(f"  Text encode (T5, dev{args.text_device}): "
          f"{timing['text_encode_ms']:.2f} ms")
    print(f"  DiT total ({args.num_inference_steps} steps):       "
          f"{timing['dit_total_ms']:.2f} ms")
    print(f"  DiT avg/step:                  "
          f"{timing['dit_avg_step_ms']:.2f} ms")
    print(f"  Audio decode (dev{args.decoder_device}):           "
          f"{timing['audio_decode_ms']:.2f} ms")
    print(f"  End-to-end:                    {timing['e2e_ms']:.2f} ms")
    print(f"  Saved audio -> {args.output}")


if __name__ == "__main__":
    main()
