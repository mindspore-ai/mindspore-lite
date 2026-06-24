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
"""Accuracy alignment for Kandinsky-5.0-T2I-Lite: HF (CPU) vs MindSpore Lite.

Two modes (run sequentially):

  * Component parity (fast, default): one forward per component on identical
    fixed inputs. Reports max_abs / mean_abs / max_rel for the Qwen last hidden
    state, CLIP pooled, transformer velocity (1 step), and VAE image.

  * Full-pipeline image parity (``--full-baseline``): runs the full HF
    ``Kandinsky5T2IPipeline`` once on CPU and the full MSLite pipeline on
    Ascend, both on the SAME seeded initial latents, then compares the two
    images (max abs / mean abs / PSNR). Slow (50 CFG steps on CPU).

For component parity the HF reference re-implements the K5 prompt encoding
(chat template + Qwen last_hidden_state[:, 41:] + CLIP pooler_output) so the
math matches the exported graphs exactly; the transformer forward uses the same
diffusers model with NATIVE attention (the CANN Custom op is only used at
MindIR time), so any diff is purely CPU-fp32 vs Ascend-fp16 numerics.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from infer_kandinsky5_t2i_lite_mslite import (
    Kandinsky5T2IInferencer, _build_model, _run_model, _save_png,
    _PROMPT_TEMPLATE, _PROMPT_TEMPLATE_ENCODE_START_IDX)


def _stats(name, a, b):
    """Print max_abs / mean_abs / max_rel between two numpy arrays."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-6)
    print(f"  {name:28s} shape={str(a.shape):24s} "
          f"max_abs={diff.max():.6e}  mean_abs={diff.mean():.6e}  "
          f"max_rel={(diff / denom).max():.4e}")


def _compare_image(a, b):
    """Return max_abs, mean_abs, PSNR between two [0,1] images."""
    diff = np.abs(a - b)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))
    return max_abs, mean_abs, psnr


def _hf_text_encode(qwen_model, clip_model, qwen_tok, clip_tok, prompt, max_seq_len):
    """Re-implement the K5 prompt encoding in HF (CPU) to match the export.

    Returns (qwen_hidden_sliced[1, seq-41, 3584], clip_pooled[1, 768]) as numpy
    fp32. Mirrors pipeline_kandinsky_t2i.py: chat template -> Qwen tokenise
    (no special tokens) -> last hidden state -> [:, 41:]; CLIP tokenise
    (special tokens, max_len 77) -> pooler_output.
    """
    full_text = _PROMPT_TEMPLATE.format(prompt)
    qwen_inputs = qwen_tok(
        full_text, padding="max_length", max_length=max_seq_len,
        truncation=True, add_special_tokens=False, return_tensors="pt").to(
        qwen_model.device)
    clip_inputs = clip_tok(
        prompt, padding="max_length", max_length=77, truncation=True,
        add_special_tokens=True, return_tensors="pt").to(clip_model.device)
    with torch.no_grad():
        qwen_hidden = qwen_model(
            input_ids=qwen_inputs.input_ids,
            attention_mask=qwen_inputs.attention_mask,
            output_hidden_states=True, return_dict=True).hidden_states[-1]
        clip_pooled = clip_model(
            input_ids=clip_inputs.input_ids,
            attention_mask=clip_inputs.attention_mask,
            return_dict=True).pooler_output
    return (qwen_hidden[:, _PROMPT_TEMPLATE_ENCODE_START_IDX:].cpu().numpy(),
            clip_pooled.cpu().numpy())


def _parse_args():
    """Parse command-line arguments for the K5 Lite alignment script."""
    p = argparse.ArgumentParser(
        description="Kandinsky-5.0-T2I-Lite HF vs MSLite alignment")
    p.add_argument("--mindir-dir", required=True)
    p.add_argument("--qwen-dir", required=True, help="Qwen2.5-VL weights dir.")
    p.add_argument("--clip-dir", required=True, help="CLIP weights dir.")
    p.add_argument("--k5-model", required=True,
                   help="K5 Lite single-file checkpoint (for HF transformer).")
    p.add_argument("--vae-dir", required=True, help="FLUX.1-dev vae/ dir.")
    p.add_argument("--prompt", default="A cat in a red hat holding a sign HELLO")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--shift", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--transformer-device", type=int, default=0)
    p.add_argument("--text-device", type=int, default=1)
    p.add_argument("--full-baseline", action="store_true",
                   help="Also run the full HF pipeline on CPU and compare image.")
    p.add_argument("--baseline-out", default="./kandinsky5_baseline.png")
    p.add_argument("--mslite-out", default="./kandinsky5_align_mslite.png")
    return p.parse_args()


def _text_encoder_parity(args, qwen_tok, clip_tok):
    """Run Qwen+CLIP on HF (CPU) and MindIR (Ascend) and compare (enc, pooled)."""
    from transformers import AutoModelForCausalLM, CLIPTextModel
    print("\n[HF] loading Qwen2.5-VL / CLIP on CPU (fp32) ...")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        args.qwen_dir, torch_dtype=torch.float32).eval()
    clip_model = CLIPTextModel.from_pretrained(
        args.clip_dir, torch_dtype=torch.float32).eval()
    hf_enc, hf_pooled = _hf_text_encode(
        qwen_model, clip_model, qwen_tok, clip_tok, args.prompt, args.max_seq_len)
    del qwen_model, clip_model

    print("\n[MindIR] loading Qwen/CLIP MindIR ...")
    mindir = Path(args.mindir_dir)
    qwen_m = _build_model(mindir / "kandinsky_text_encoder_graph.mindir",
                          args.text_device)
    clip_m = _build_model(mindir / "kandinsky_clip_encoder_graph.mindir",
                          args.text_device)
    full_text = _PROMPT_TEMPLATE.format(args.prompt)
    qwen_inputs = qwen_tok(
        full_text, padding="max_length", max_length=args.max_seq_len,
        truncation=True, add_special_tokens=False, return_tensors="np")
    clip_inputs = clip_tok(
        args.prompt, padding="max_length", max_length=77, truncation=True,
        add_special_tokens=True, return_tensors="np")
    order = ["input_ids", "attention_mask"]
    ms_enc = _run_model(
        qwen_m,
        {"input_ids": np.asarray(qwen_inputs.input_ids, dtype=np.int64),
         "attention_mask": np.asarray(qwen_inputs.attention_mask, dtype=np.int64)},
        order)[0]
    ms_pooled = _run_model(
        clip_m,
        {"input_ids": np.asarray(clip_inputs.input_ids, dtype=np.int64),
         "attention_mask": np.asarray(clip_inputs.attention_mask, dtype=np.int64)},
        order)[0]
    print("\n--- text encoder parity ---")
    _stats("Qwen last_hidden_state", hf_enc, ms_enc)
    _stats("CLIP pooled_embeds", hf_pooled, ms_pooled)
    return hf_enc, hf_pooled


def _transformer_vae_parity(args, hf_enc, hf_pooled):
    """Run K5 transformer + FLUX VAE on HF (CPU) and MindIR (Ascend), compare.

    Returns the shared seeded noise latent (used by the optional full-baseline
    run so both pipelines see identical inputs).
    """
    from diffusers import Kandinsky5Transformer3DModel, AutoencoderKL
    print("\n[HF] loading K5 transformer + FLUX VAE on CPU (fp32) ...")
    transformer = Kandinsky5Transformer3DModel.from_single_file(
        args.k5_model, torch_dtype=torch.float32).eval()
    vae = AutoencoderKL.from_pretrained(
        args.vae_dir, torch_dtype=torch.float32).eval()

    latent_h = args.height // 8
    latent_w = args.width // 8
    rng = np.random.RandomState(args.seed)
    noise = rng.standard_normal(
        (1, 16, 1, latent_h, latent_w)).astype(np.float32)
    seq_after_slice = hf_enc.shape[1]
    visual_rope_h = torch.arange(latent_h // 2, dtype=torch.int64)
    visual_rope_w = torch.arange(latent_w // 2, dtype=torch.int64)
    text_rope = torch.arange(seq_after_slice, dtype=torch.int64)
    t_pos = torch.arange(1, dtype=torch.int64)
    with torch.no_grad():
        hf_noise = transformer(
            hidden_states=torch.from_numpy(noise),
            encoder_hidden_states=torch.from_numpy(hf_enc),
            timestep=torch.tensor([950.0]),
            pooled_projections=torch.from_numpy(hf_pooled),
            visual_rope_pos=(t_pos, visual_rope_h, visual_rope_w),
            text_rope_pos=text_rope, return_dict=False)[0].numpy()

    print("\n[MindIR] loading transformer/VAE MindIR ...")
    mindir = Path(args.mindir_dir)
    tx_m = _build_model(mindir / "kandinsky_transformer_graph.mindir",
                        args.transformer_device)
    vae_m = _build_model(mindir / "kandinsky_dcae_decoder_graph.mindir",
                         args.transformer_device)
    feed = {
        "hidden_states": noise,
        "encoder_hidden_states": hf_enc.astype(np.float32),
        "timestep": np.array([950.0], dtype=np.float32),
        "pooled_projections": hf_pooled.astype(np.float32),
        "visual_rope_h": visual_rope_h.numpy(),
        "visual_rope_w": visual_rope_w.numpy(),
        "text_rope": text_rope.numpy(),
    }
    order = ["hidden_states", "encoder_hidden_states", "timestep",
             "pooled_projections", "visual_rope_h", "visual_rope_w", "text_rope"]
    ms_noise = _run_model(tx_m, feed, order)[0]
    print("\n--- transformer forward parity (1 step, t=950) ---")
    _stats("velocity (noise_pred)", hf_noise, ms_noise)

    latent_dec = (noise[0, :, 0] / 0.3611)[None]  # undo scaling, [1,16,H',W']
    with torch.no_grad():
        hf_img = vae.decode(torch.from_numpy(latent_dec), return_dict=False)[
            0].numpy()
    ms_img = _run_model(vae_m, {"latents": latent_dec.astype(np.float32)},
                        ["latents"])[0]
    print("\n--- VAE decode parity ---")
    _stats("vae image", hf_img, ms_img)
    return noise


def main():
    """Run HF vs MSLite alignment (component parity + optional full image)."""
    args = _parse_args()
    print("=" * 70)
    print("Kandinsky-5.0-T2I-Lite component parity: HF (CPU fp32) vs MindSpore Lite")
    print("=" * 70)

    from transformers import AutoTokenizer
    qwen_tok = AutoTokenizer.from_pretrained(args.qwen_dir)
    clip_tok = AutoTokenizer.from_pretrained(args.clip_dir)

    hf_enc, hf_pooled = _text_encoder_parity(args, qwen_tok, clip_tok)
    noise = _transformer_vae_parity(args, hf_enc, hf_pooled)

    print("\n" + "=" * 70)
    print("Component parity complete. fp16 noise/image max_abs < ~1e-1 expected.")
    print("=" * 70)

    if args.full_baseline:
        _full_baseline(args, noise, qwen_tok, clip_tok)


def _full_baseline(args, latents_np, qwen_tok, clip_tok):
    """Run the full HF + MSLite pipelines on shared latents and compare images."""
    from diffusers import Kandinsky5T2IPipeline
    print("\n[HF] running full Kandinsky5T2IPipeline on CPU (slow) ...")
    pipe = Kandinsky5T2IPipeline.from_pretrained(
        args.k5_model, torch_dtype=torch.float32)
    init = torch.from_numpy(latents_np[:, :, 0])  # [1,16,H',W'] for HF 2D path
    hf_img = pipe(
        prompt=args.prompt, negative_prompt=args.negative_prompt,
        height=args.height, width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale, max_sequence_length=args.max_seq_len,
        latents=init, output_type="np").images[0]
    hf_img = np.asarray(hf_img)  # [H, W, 3] in [0,1]
    print(f"[HF] baseline image saved -> {args.baseline_out}")
    _save_png(hf_img[None].transpose(0, 3, 1, 2) * 2 - 1, args.baseline_out)

    print("[MSLite] running full pipeline on Ascend ...")
    inferencer = Kandinsky5T2IInferencer(
        args.mindir_dir, args.qwen_dir, args.clip_dir,
        text_device=args.text_device, transformer_device=args.transformer_device,
        height=args.height, width=args.width, max_seq_len=args.max_seq_len,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale, shift=args.shift)
    latents_npy = str(Path(args.mindir_dir) / "_align_latents.npy")
    np.save(latents_npy, latents_np.astype(np.float32))
    inferencer.generate(args.prompt, args.negative_prompt, args.mslite_out,
                        seed=args.seed, latents_npy=latents_npy)

    from PIL import Image
    ms_img = np.asarray(Image.open(args.mslite_out).convert("RGB"),
                        dtype=np.float32) / 255.0
    max_abs, mean_abs, psnr = _compare_image(hf_img, ms_img)
    print("\n--- full-pipeline image alignment (HF vs MSLite) ---")
    print(f"  max  abs error : {max_abs:.6f}")
    print(f"  mean abs error : {mean_abs:.6f}")
    print(f"  PSNR (dB)      : {psnr:.2f}")


if __name__ == "__main__":
    main()
