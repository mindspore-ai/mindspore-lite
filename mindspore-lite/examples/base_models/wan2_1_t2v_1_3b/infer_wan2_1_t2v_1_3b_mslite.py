"""Wan2.1-T2V-1.3B text-to-video inference with MindSpore Lite (Ascend 300I Duo).

Loads the three converted MindIR sub-models (UMT5 text encoder, Wan DiT
transformer, Wan VAE decoder), encodes the prompt, runs the flow-matching
denoising loop (UniPC scheduler on CPU, transformer on Ascend), and decodes the
latent to a video (VAE on Ascend).

Model inference (text encoder / transformer / VAE) is pure ``mindspore_lite`` +
``numpy``. ``torch`` is imported ONLY for the (numpy-backed) diffusers UniPC
scheduler that runs on CPU, exactly as in the flux1_dev example; no torch tensor
ever touches the Ascend models.

Component placement on the 300I Duo (component split, not tensor-parallel):
text encoder -> dev1, transformer -> dev0, VAE decoder -> dev0.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import mindspore_lite as mslite

# torch is used ONLY for the CPU/numpy-backed diffusers UniPC scheduler; the
# Ascend model inference is pure mslite + numpy (see module docstring).
import torch  # noqa: E402  (scheduler only)
from transformers import AutoTokenizer  # noqa: E402
from diffusers import UniPCMultistepScheduler  # noqa: E402

_VAE_SCALE_FACTOR_TEMPORAL = 4
_VAE_SCALE_FACTOR_SPATIAL = 8
_LATENT_CHANNELS = 16


def _build_model(mindir_path, device_id):
    """Build an mslite Model from a ``*_graph.mindir`` on a given Ascend device."""
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = int(device_id)
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


class WanT2VInferencer:
    """End-to-end Wan2.1-T2V-1.3B text-to-video inferencer over MindSpore Lite."""

    def __init__(self, mindir_dir, model_dir, text_device=1, transformer_device=0,
                 vae_device=0, height=480, width=832, num_frames=81,
                 max_seq_len=512, num_inference_steps=50, guidance_scale=5.0):
        """Load sub-models, tokenizer, scheduler and VAE latent statistics."""
        mindir_dir = Path(mindir_dir)
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.max_seq_len = max_seq_len
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        self.text_model = _build_model(mindir_dir / "wan_text_encoder_graph.mindir", text_device)
        self.transformer = _build_model(mindir_dir / "wan_transformer_graph.mindir", transformer_device)
        self.vae = _build_model(mindir_dir / "wan_vae_decoder_graph.mindir", vae_device)

        self.tokenizer = AutoTokenizer.from_pretrained(Path(model_dir) / "tokenizer")
        self.scheduler = UniPCMultistepScheduler.from_pretrained(model_dir, subfolder="scheduler")
        vae_cfg = json.loads((Path(model_dir) / "vae" / "config.json").read_text())
        self.latents_mean = np.array(vae_cfg["latents_mean"], dtype=np.float32).reshape(1, 16, 1, 1, 1)
        self.latents_std = np.array(vae_cfg["latents_std"], dtype=np.float32).reshape(1, 16, 1, 1, 1)

        self.num_latent_frames = (num_frames - 1) // _VAE_SCALE_FACTOR_TEMPORAL + 1
        self.latent_h = height // _VAE_SCALE_FACTOR_SPATIAL
        self.latent_w = width // _VAE_SCALE_FACTOR_SPATIAL

    def _latent_shape(self):
        """Return the fixed latent shape (B, C, F', H', W')."""
        return (1, _LATENT_CHANNELS, self.num_latent_frames, self.latent_h, self.latent_w)

    def _encode_prompt(self, prompt):
        """Tokenize + run the UMT5 text encoder -> [1, seq_len, 4096] embeds."""
        toks = self.tokenizer(prompt, padding="max_length", max_length=self.max_seq_len,
                              truncation=True, add_special_tokens=True,
                              return_attention_mask=True, return_tensors="np")
        input_ids = toks["input_ids"].astype(np.int64)
        attention_mask = toks["attention_mask"].astype(np.int64)
        embeds = _run_model(self.text_model,
                            {"input_ids": input_ids, "attention_mask": attention_mask},
                            ["input_ids", "attention_mask"])[0]
        return embeds.astype(np.float32)

    def _denoise(self, latents, prompt_embeds, negative_embeds):
        """Run the CFG flow-matching denoising loop, returning denoised latents."""
        self.scheduler.set_timesteps(self.num_inference_steps)
        self.scheduler.set_begin_index(0)
        timesteps = self.scheduler.timesteps
        latents_t = torch.from_numpy(latents)
        for t in timesteps:
            timestep = np.array([float(t)], dtype=np.float32)
            noise_cond = _run_model(self.transformer,
                                    {"hidden_states": latents, "timestep": timestep,
                                     "encoder_hidden_states": prompt_embeds},
                                    ["hidden_states", "timestep", "encoder_hidden_states"])[0]
            noise_uncond = _run_model(self.transformer,
                                      {"hidden_states": latents, "timestep": timestep,
                                       "encoder_hidden_states": negative_embeds},
                                      ["hidden_states", "timestep", "encoder_hidden_states"])[0]
            noise_pred = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
            latents_t = self.scheduler.step(torch.from_numpy(noise_pred), t, latents_t,
                                            return_dict=False)[0]
            latents = latents_t.numpy().astype(np.float32)
        return latents

    def _vae_decode(self, latents):
        """Denormalise latents and decode to a video array [1, 3, F, H, W]."""
        latents = latents / self.latents_std + self.latents_mean
        video = _run_model(self.vae, {"latents": latents.astype(np.float32)}, ["latents"])[0]
        return video

    @staticmethod
    def _save_video(video, output_path, fps=16):
        """Save the decoded video as an mp4 (imageio) or per-frame PNGs."""
        frames = ((video[0].transpose(1, 2, 3, 0) / 2 + 0.5).clip(0, 1) * 255).astype(np.uint8)
        output_path = str(output_path)
        try:
            import imageio
            imageio.mimsave(output_path, [frames[i] for i in range(frames.shape[0])], fps=fps)
            print(f"[infer] saved video -> {output_path}")
        except ImportError:
            stem = Path(output_path).with_suffix("")
            for i in (0, frames.shape[0] // 2, frames.shape[0] - 1):
                _save_png(frames[i], f"{stem}_frame{i}.png")
            print(f"[infer] imageio not installed; saved key frames under {stem}_frame*.png")

    def generate(self, prompt, negative_prompt, output_path, seed=42, latents_npy=None):
        """Run the full T2V pipeline with stage timing and save the video."""
        timing = {}

        t0 = time.perf_counter()
        prompt_embeds = self._encode_prompt(prompt)
        negative_embeds = self._encode_prompt(negative_prompt or "")
        timing["text_encode_ms"] = (time.perf_counter() - t0) * 1000

        if latents_npy:
            latents = np.load(latents_npy).astype(np.float32)
        else:
            rng = np.random.RandomState(seed)
            latents = rng.standard_normal(self._latent_shape()).astype(np.float32)

        t0 = time.perf_counter()
        latents = self._denoise(latents, prompt_embeds, negative_embeds)
        timing["transformer_total_ms"] = (time.perf_counter() - t0) * 1000
        timing["transformer_avg_step_ms"] = timing["transformer_total_ms"] / self.num_inference_steps

        t0 = time.perf_counter()
        video = self._vae_decode(latents)
        timing["vae_decode_ms"] = (time.perf_counter() - t0) * 1000

        self._save_video(video, output_path)
        timing["e2e_ms"] = (timing["text_encode_ms"] + timing["transformer_total_ms"]
                            + timing["vae_decode_ms"])
        return timing, video


def _save_png(frame_rgb, path):
    """Save an HxWx3 uint8 frame as a PNG using PIL (keeps deps minimal)."""
    from PIL import Image
    Image.fromarray(frame_rgb).save(path)


def main():
    """Parse arguments and run Wan2.1-T2V-1.3B text-to-video inference."""
    parser = argparse.ArgumentParser(description="Wan2.1-T2V-1.3B MindSpore Lite inference")
    parser.add_argument("--mindir-dir", required=True, help="dir with the 3 *_graph.mindir")
    parser.add_argument("--model-dir", required=True, help="Wan2.1-T2V-1.3B weights dir")
    parser.add_argument("--prompt", default="A cat walking on a beach, cinematic, 4k.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", default="wan_output.mp4")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=13)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latents-npy", default=None, help="pre-generated latents (for alignment)")
    parser.add_argument("--text-device", type=int, default=1)
    parser.add_argument("--transformer-device", type=int, default=0)
    parser.add_argument("--vae-device", type=int, default=0)
    args = parser.parse_args()

    inferencer = WanT2VInferencer(
        args.mindir_dir, args.model_dir, args.text_device, args.transformer_device,
        args.vae_device, args.height, args.width, args.num_frames,
        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale)
    timing, _ = inferencer.generate(
        args.prompt, args.negative_prompt, args.output, args.seed, args.latents_npy)

    print("\n--- Performance ---")
    print(f"  Text encode (UMT5):      {timing['text_encode_ms']:.2f} ms")
    print(f"  Transformer total:       {timing['transformer_total_ms']:.2f} ms")
    print(f"  Transformer avg/step:    {timing['transformer_avg_step_ms']:.2f} ms "
          f"({args.num_inference_steps} steps, CFG x2)")
    print(f"  VAE decode:              {timing['vae_decode_ms']:.2f} ms")
    print(f"  End-to-end:              {timing['e2e_ms']:.2f} ms")


if __name__ == "__main__":
    main()
