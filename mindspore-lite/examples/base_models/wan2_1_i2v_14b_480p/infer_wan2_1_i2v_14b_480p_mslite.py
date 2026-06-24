"""Wan2.1-I2V-14B-480P image-to-video inference with MindSpore Lite (Ascend 300I Duo).

Loads the four converted MindIR sub-models (UMT5 text encoder, CLIP image
encoder, Wan DiT transformer, Wan VAE decoder), encodes the prompt + the
conditioning image, builds the Wan2.1 I2V channel-wise conditioning, runs the
flow-matching denoising loop (UniPC scheduler on CPU, transformer on Ascend),
and decodes the latent to a video (VAE on Ascend).

Model inference (text / CLIP / transformer / VAE) is pure ``mindspore_lite`` +
``numpy``. ``torch`` is imported ONLY for the (numpy-backed) diffusers UniPC
scheduler that runs on CPU, exactly as in the wan2_1_t2v example; no torch
tensor ever touches the Ascend models.

Wan2.1 I2V details handled in numpy (mirrors diffusers'
``pipeline_wan_i2v.py`` WITHOUT ``expand_timesteps`` -- the Wan2.1 schedule):

  * ``video_condition`` = the conditioning image (frame 0) concatenated with
    zero frames for the remaining ``num_frames-1`` slots; encoded by the VAE
    encoder (CPU, argmax mode) then denormalised by latents_mean/latents_std.
  * ``mask_lat_size`` is a per-latent-frame mask of shape
    ``[1, 4, F', H', W']`` (4 = vae_scale_factor_temporal): 1 on the first
    latent frame, 0 elsewhere (built by repeat_interleave + view + transpose
    exactly as in ``prepare_latents``).
  * The transformer input is the 36-channel concatenation
    ``cat([latents(16), mask_lat_size(4), latent_condition(16)], dim=1)``.
  * The timestep is the Wan2.1 scalar ``[1]`` (NOT per-token).

Component placement on the 300I Duo (component split across 3 chips, not
tensor-parallel): the 14B DiT (~28GB fp16) needs a whole 44GB chip to itself,
so text+CLIP image encoders -> dev1, transformer -> dev0, VAE decoder -> dev2.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import mindspore_lite as mslite

# torch is used ONLY for the CPU/numpy-backed diffusers UniPC scheduler and the
# one-shot VAE encode of the conditioning image; the Ascend model inference is
# pure mslite + numpy (see module docstring).
import torch  # noqa: E402  (scheduler + vae encode only)
from transformers import AutoTokenizer, CLIPImageProcessor  # noqa: E402
from diffusers import UniPCMultistepScheduler, AutoencoderKLWan  # noqa: E402

_VAE_SCALE_FACTOR_TEMPORAL = 4
_VAE_SCALE_FACTOR_SPATIAL = 8
_LATENT_CHANNELS = 16

# Wan2.1 I2V channel-wise conditioning channel counts (see module docstring).
_MASK_CHANNELS = _VAE_SCALE_FACTOR_TEMPORAL


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


class WanI2VInferencer:
    """End-to-end Wan2.1-I2V-14B-480P image-to-video inferencer over MindSpore Lite."""

    def __init__(self, mindir_dir, model_dir, text_device=1, clip_device=1,
                 transformer_device=0, vae_device=2, height=480, width=832,
                 num_frames=81, max_seq_len=512, num_inference_steps=50,
                 guidance_scale=5.0):
        """Load sub-models, tokenizers, scheduler and VAE latent statistics."""
        mindir_dir = Path(mindir_dir)
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.max_seq_len = max_seq_len
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        self.text_model = _build_model(
            mindir_dir / "wan_text_encoder_graph.mindir", text_device)
        self.clip_model = _build_model(
            mindir_dir / "wan_clip_image_encoder_graph.mindir", clip_device)
        self.transformer = _build_model(
            mindir_dir / "wan_transformer_graph.mindir", transformer_device)
        self.vae = _build_model(mindir_dir / "wan_vae_decoder_graph.mindir", vae_device)

        self.tokenizer = AutoTokenizer.from_pretrained(Path(model_dir) / "tokenizer")
        self.image_processor = CLIPImageProcessor.from_pretrained(
            Path(model_dir) / "feature_extractor")
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            model_dir, subfolder="scheduler")
        vae_cfg = json.loads((Path(model_dir) / "vae" / "config.json").read_text())
        self.z_dim = int(vae_cfg["z_dim"])
        self.latents_mean = np.array(
            vae_cfg["latents_mean"], dtype=np.float32).reshape(1, self.z_dim, 1, 1, 1)
        # diffusers stores latents_std as the std itself; prepare_latents uses
        # 1.0 / latents_std as the denormalisation multiplier. We keep the raw
        # std here and divide where needed.
        self.latents_std = np.array(
            vae_cfg["latents_std"], dtype=np.float32).reshape(1, self.z_dim, 1, 1, 1)
        # VAE encode is a small CPU-side torch model (we only run it once for the
        # conditioning image); loaded lazily to keep the import-time light.
        self._vae_encoder = None
        self._vae_encoder_dir = Path(model_dir) / "vae"

        self.num_latent_frames = (num_frames - 1) // _VAE_SCALE_FACTOR_TEMPORAL + 1
        self.latent_h = height // _VAE_SCALE_FACTOR_SPATIAL
        self.latent_w = width // _VAE_SCALE_FACTOR_SPATIAL
        # Precompute the Wan2.1 I2V per-latent-frame mask_lat_size
        # [1, 4, F', H', W']: 1 on the first latent frame, 0 elsewhere.
        self.mask_lat_size = self._build_mask_lat_size()

    def _build_mask_lat_size(self):
        """Build the Wan2.1 I2V mask_lat_size [1, 4, F', H', W'].

        Mirrors ``pipeline_wan_i2v.prepare_latents`` (non-expand branch): a
        per-frame mask of shape ``[B, 1, num_frames, H', W']`` that is 1 on frame
        0 and 0 on frames 1..num_frames-1, then collapsed to latent frames via
        ``view(B, -1, vae_scale_factor_temporal, H', W').transpose(1, 2)``.
        """
        mask = np.ones((1, 1, self.num_frames, self.latent_h, self.latent_w),
                       dtype=np.float32)
        mask[:, :, list(range(1, self.num_frames))] = 0
        first_frame_mask = mask[:, :, 0:1]
        first_frame_mask = np.repeat(
            first_frame_mask, _VAE_SCALE_FACTOR_TEMPORAL, axis=2)
        mask = np.concatenate([first_frame_mask, mask[:, :, 1:, :]], axis=2)
        # mask is now (1, 1, num_frames, H', W') with num_frames latent-frame groups
        mask = mask.reshape(1, -1, _VAE_SCALE_FACTOR_TEMPORAL, self.latent_h,
                            self.latent_w)
        mask = mask.transpose(0, 2, 1, 3, 4)  # (1, 4, F', H', W')
        return mask.astype(np.float32)

    def _latent_shape(self):
        """Return the fixed latent shape (B, C, F', H', W')."""
        return (1, _LATENT_CHANNELS, self.num_latent_frames, self.latent_h, self.latent_w)

    def _load_vae_encoder(self):
        """Load the VAE encoder on CPU (only used to encode the conditioning image)."""
        if self._vae_encoder is None:
            self._vae_encoder = AutoencoderKLWan.from_pretrained(
                str(self._vae_encoder_dir), torch_dtype=torch.float32).eval()
        return self._vae_encoder

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

    def _encode_image_clip(self, image):
        """Preprocess the image (CLIPImageProcessor) + run CLIP -> [1, 257, 1280]."""
        proc = self.image_processor(images=image, return_tensors="np")
        pixel_values = proc["pixel_values"].astype(np.float32)
        embeds = _run_model(self.clip_model, {"pixel_values": pixel_values},
                            ["pixel_values"])[0]
        return embeds.astype(np.float32)

    def _encode_condition_latent(self, image):
        """VAE-encode the conditioning image into the 16-channel condition latent.

        Mirrors diffusers' ``prepare_latents`` (non-expand branch, last_image is
        None): ``video_condition`` is the single image frame concatenated with
        ``num_frames-1`` zero frames, encoded by the VAE encoder in argmax mode,
        then re-normalised to the latent space with latents_mean / (1/latents_std).
        Runs on CPU (torch) because the VAE encoder is not part of the Ascend
        MindIR set (only the VAE decoder is exported).
        """
        vae = self._load_vae_encoder()
        # image: HxWx3 uint8 (PIL or ndarray). Build a full-length "video"
        # [1, 3, num_frames, H, W] in [-1, 1] as diffusers' VideoProcessor does:
        # frame 0 = image, frames 1..num_frames-1 = zeros.
        arr = np.asarray(image).astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 3:  # HWC -> CHW
            arr = arr.transpose(2, 0, 1)
        arr = arr / 127.5 - 1.0
        video_cond = np.zeros((1, 3, self.num_frames, self.height, self.width),
                              dtype=np.float32)
        video_cond[:, :, 0] = arr
        video_cond_t = torch.from_numpy(video_cond)
        with torch.no_grad():
            enc = vae.encode(video_cond_t)
            latent_cond = enc.latent_dist.mode()  # argmax mode
        latent_cond = latent_cond.numpy().astype(np.float32)
        # prepare_latents: latent_condition = (latent_condition - mean) * (1/std)
        latent_cond = (latent_cond - self.latents_mean) * (1.0 / self.latents_std)
        return latent_cond

    def _denoise(self, latents, latent_condition, prompt_embeds, negative_embeds,
                 image_embeds):
        """Run the CFG flow-matching denoising loop, returning denoised latents.

        Wan2.1 I2V keeps the CLIP image embedding fixed for both the cond and
        uncond branches (only the text embed differs), matching the HF pipeline.
        The 36-channel DiT input is rebuilt every step from the current latents.
        """
        self.scheduler.set_timesteps(self.num_inference_steps)
        self.scheduler.set_begin_index(0)
        timesteps = self.scheduler.timesteps
        latents_t = torch.from_numpy(latents)
        for t in timesteps:
            # Wan2.1 I2V: scalar timestep [1]; 36-ch input = cat(latents, mask, cond).
            timestep = np.array([float(t)], dtype=np.float32)
            latent_model_input = np.concatenate(
                [latents, self.mask_lat_size, latent_condition], axis=1
            ).astype(np.float32)
            noise_cond = _run_model(self.transformer,
                                    {"hidden_states": latent_model_input,
                                     "timestep": timestep,
                                     "encoder_hidden_states": prompt_embeds,
                                     "encoder_hidden_states_image": image_embeds},
                                    ["hidden_states", "timestep", "encoder_hidden_states",
                                     "encoder_hidden_states_image"])[0]
            noise_uncond = _run_model(self.transformer,
                                      {"hidden_states": latent_model_input,
                                       "timestep": timestep,
                                       "encoder_hidden_states": negative_embeds,
                                       "encoder_hidden_states_image": image_embeds},
                                      ["hidden_states", "timestep", "encoder_hidden_states",
                                       "encoder_hidden_states_image"])[0]
            noise_pred = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
            latents_t = self.scheduler.step(
                torch.from_numpy(noise_pred), t, latents_t, return_dict=False)[0]
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

    def generate(self, prompt, image, negative_prompt, output_path, seed=42,
                 latents_npy=None):
        """Run the full I2V pipeline with stage timing and save the video.

        ``image`` is a PIL.Image or numpy HxWx3 uint8 array used as the first
        frame. ``negative_prompt`` drives the uncond text branch; the CLIP image
        embedding is shared by cond/uncond.
        """
        timing = {}

        t0 = time.perf_counter()
        prompt_embeds = self._encode_prompt(prompt)
        negative_embeds = self._encode_prompt(negative_prompt or "")
        timing["text_encode_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        image_embeds = self._encode_image_clip(image)
        timing["clip_encode_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        latent_condition = self._encode_condition_latent(image)
        timing["vae_encode_ms"] = (time.perf_counter() - t0) * 1000

        if latents_npy:
            latents = np.load(latents_npy).astype(np.float32)
        else:
            rng = np.random.RandomState(seed)
            latents = rng.standard_normal(self._latent_shape()).astype(np.float32)

        t0 = time.perf_counter()
        latents = self._denoise(latents, latent_condition, prompt_embeds, negative_embeds,
                                image_embeds)
        timing["transformer_total_ms"] = (time.perf_counter() - t0) * 1000
        timing["transformer_avg_step_ms"] = timing["transformer_total_ms"] / self.num_inference_steps

        t0 = time.perf_counter()
        video = self._vae_decode(latents)
        timing["vae_decode_ms"] = (time.perf_counter() - t0) * 1000

        self._save_video(video, output_path)
        timing["e2e_ms"] = (timing["text_encode_ms"] + timing["clip_encode_ms"]
                            + timing["vae_encode_ms"] + timing["transformer_total_ms"]
                            + timing["vae_decode_ms"])
        return timing, video


def _save_png(frame_rgb, path):
    """Save an HxWx3 uint8 frame as a PNG using PIL (keeps deps minimal)."""
    from PIL import Image
    Image.fromarray(frame_rgb).save(path)


def _load_image(path):
    """Load an image as a PIL.Image (resizing is done by the CLIP image processor)."""
    from PIL import Image
    return Image.open(path).convert("RGB")


def main():
    """Parse arguments and run Wan2.1-I2V-14B-480P image-to-video inference."""
    parser = argparse.ArgumentParser(description="Wan2.1-I2V-14B-480P MindSpore Lite inference")
    parser.add_argument("--mindir-dir", required=True, help="dir with the 4 *_graph.mindir")
    parser.add_argument("--model-dir", required=True, help="Wan2.1-I2V-14B-480P weights dir")
    parser.add_argument("--image", required=True, help="conditioning image (first frame)")
    parser.add_argument("--prompt", default="A cat walking on a beach, cinematic, 4k.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", default="wan_output.mp4")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latents-npy", default=None, help="pre-generated latents (for alignment)")
    parser.add_argument("--text-device", type=int, default=1)
    parser.add_argument("--clip-device", type=int, default=1)
    parser.add_argument("--transformer-device", type=int, default=0)
    parser.add_argument("--vae-device", type=int, default=2)
    args = parser.parse_args()

    inferencer = WanI2VInferencer(
        args.mindir_dir, args.model_dir, args.text_device, args.clip_device,
        args.transformer_device, args.vae_device, args.height, args.width,
        args.num_frames, num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale)
    image = _load_image(args.image)
    timing, _ = inferencer.generate(
        args.prompt, image, args.negative_prompt, args.output, args.seed,
        args.latents_npy)

    print("\n--- Performance ---")
    print(f"  Text encode (UMT5, dev1):     {timing['text_encode_ms']:.2f} ms")
    print(f"  CLIP encode (dev1):           {timing['clip_encode_ms']:.2f} ms")
    print(f"  VAE encode (CPU):             {timing['vae_encode_ms']:.2f} ms")
    print(f"  Transformer total:            {timing['transformer_total_ms']:.2f} ms")
    print(f"  Transformer avg/step:         {timing['transformer_avg_step_ms']:.2f} ms "
          f"({args.num_inference_steps} steps, CFG x2)")
    print(f"  VAE decode (dev2):            {timing['vae_decode_ms']:.2f} ms")
    print(f"  End-to-end:                   {timing['e2e_ms']:.2f} ms")


if __name__ == "__main__":
    main()
