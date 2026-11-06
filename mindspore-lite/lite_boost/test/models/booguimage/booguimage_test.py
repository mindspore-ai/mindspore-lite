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
"""BooguImage test script: performance by default, accuracy with ``--baseline``.

Both modes drive the fixed test request (4-step DMD, square ``--resolution``,
default 512) through the three boost forms checked in next to this script:

- ``sp2``: two-card SP (= CP-2) on both regions with communication-compute
  overlap enabled everywhere (``cc_overlap``); BSND stays off, its SP path fails
  in ``npu_fusion_attention`` on the master branch (see the README);
- ``tp2``: two-card TP on both regions (weights sharded by tensor dimension);
- ``1p``: single card plus the BSND fused attention.

The mllm encoder is never moved by the script: it shares the DiT's card
(single card / rank 0 -- on the two-card path the master branch's
``tp_pipeline_call`` places it there natively), so one card must hold the
DiT and the encoder together. ``--offload`` relaxes that footprint: the
components commute between host memory and the card around their calls --
each encode swaps the transformer out and the encoder in, the decode does
the same for the VAE (on the multi-card path only the decode rank, rank 0,
ever fetches it) -- so a dev card that could never hold all the weights
at once still runs the case. Timings then include those transfers, the
generated images are unaffected.

Performance mode (default) -- each form runs in a fresh ``torchrun``
subprocess (warmup then measured runs), prints ``RESULT`` lines and
aggregates ``<out-dir>/summary.txt``::

    python booguimage_test.py --resolution 512

Accuracy mode (``--baseline`` given) -- runs 1p, sp2 and tp2 once each and judges
the images against the plain PyTorch model's output for the same request,
PSNR >= 20 (``--psnr-min``), exit non-zero on failure. The reference comes
from an ``.npz`` holding one image array under key ``image`` (same prompt,
seed, steps and resolution, NPU generator -- see the README for the
generation snippet). Every case draws with the rank's NPU generator, so the
reference and every form share one random stream; the NPU RNG draws the same
stream for the same seed on every card::

    python booguimage_test.py --resolution 1024 --baseline reference.npz

All configuration is command-line arguments with defaults (``--model``
``--out-dir`` ``--prompt`` ``--steps`` ``--seed`` ``--warmup`` ``--runs``
``--modes`` ``--psnr-min``); see ``--help``.
"""
import argparse
import logging
import math
import os
import socket
import subprocess
import sys
import time

import torch

# The test prompt (fixed across every case so images stay comparable).
DEFAULT_PROMPT = (
    "生成一张街拍的照片，要有垃圾桶，红绿灯。画面中心有一个拾荒的老人，"
    "满脸沧桑。照片要有摄影质感，以及徕卡相机的街拍质感。"
)

# Relative checkpoint dir the official download command fills in
# (huggingface-cli download Boogu/Boogu-Image-0.1-Turbo --local-dir ...).
DEFAULT_MODEL_DIR = "models/Boogu-Image-0.1-Turbo"

# Boost forms under test live as static YAML files next to this script
# (``booguimage_sp2.yaml`` / ``booguimage_tp2.yaml`` / ``booguimage_1p.yaml``),
# so the checked-in configs are exactly what the cases run.

PSNR_MIN_DB = 20.0


def load_torch_npu():
    """Import torch_npu and its contrib device patches; return the module.

    The imports are side-effect only (backend registration + the
    device-mapping patches the boogu pipeline and lite_boost rely on), so the
    module is returned to keep them honest.
    """
    import torch_npu
    import torch_npu.contrib.transfer_to_npu
    return torch_npu


def boost_config_path(mode):
    """Checked-in boost YAML for ``mode`` (``sp2`` / ``tp2`` / ``1p``), next to the script."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f"booguimage_{mode}.yaml")


def save_reference_png(npz_path, out_png):
    """Extract the PyTorch-model reference image from an ``.npz`` file.

    The npz is expected to hold one image array -- the plain-PyTorch
    pipeline's output for the same request (prompt, seed, steps, resolution)
    as the boosted cases. Accepted layouts: ``(H, W, 3)`` (the documented
    contract, ``np.savez(..., image=np.asarray(pil_image))``), also
    ``(1, H, W, 3)`` / ``(3, H, W)`` / ``(1, 3, H, W)``; uint8 0-255 or float
    0-1 (rescaled by 255). The array is looked up under key ``image``
    (``images``/``img``/``arr_0``/the only array also accepted).

    Saves the reference as a PNG (so it can be eyeballed next to the outputs)
    and returns its size as ``(width, height)``.
    """
    import numpy as np
    from PIL import Image

    with np.load(npz_path) as data:
        preferred = ("image", "images", "img", "arr_0")
        key = next((name for name in preferred if name in data.files), None)
        if key is None and data.files:
            key = data.files[0]
        if key is None:
            raise ValueError(f"{npz_path}: empty npz, no array to use as reference")
        arr = np.asarray(data[key])
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] != 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] != 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)  # CHW -> HWC
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim not in (2, 3):
        raise ValueError(
            f"{npz_path}: array {key!r} has shape {arr.shape}, not an image "
            f"(expected (H, W, 3) or equivalent)"
        )
    if np.issubdtype(arr.dtype, np.floating) and float(arr.max()) <= 1.5:
        arr = arr * 255.0
    image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")
    image.save(out_png)
    logging.info("reference %s[%s] -> %s", npz_path, key, out_png)
    return image.size


def turbo_call_kwargs(args, height, width, device=None, rank=0):
    """Keyword arguments for one Turbo T2I request (DMD few-step, ``args.seed``).

    Mirrors the official ``inference_npu.py``: without
    ``max_input_image_pixels`` / ``max_input_image_side_length`` the pipeline
    falls back to its own pixel budget and the text conditioning shifts.
    ``device`` is only passed through for single-process calls -- the sp2
    path derives it per rank (a foreign ``device`` would break the HCCL
    broadcast on the other ranks).

    The generator lives on the rank's NPU like the official inference script
    does. The NPU RNG draws the same stream for the same seed on every card,
    so the npz reference (npu:0), 1p and both sp2 ranks prepare identical
    initial-latent and DMD-renoise streams -- that is what keeps the PSNR
    comparison meaningful.
    """
    generator = torch.Generator(f"npu:{rank}").manual_seed(args.seed)
    kwargs = {
        "instruction": [args.prompt],
        "negative_instruction": "",
        "empty_instruction": "",
        "height": height,
        "width": width,
        "max_input_image_pixels": height * width,
        "max_input_image_side_length": 2 * max(height, width),
        "num_inference_steps": args.steps,
        "text_guidance_scale": 1.0,
        "image_guidance_scale": 1.0,
        "empty_instruction_guidance_scale": 0.0,
        "use_dmd_student_inference": True,
        "dmd_conditioning_sigma": 0.001,
        "generator": generator,
    }
    if device is not None:
        kwargs["device"] = device
    return kwargs


def compute_psnr(path_a, path_b):
    """Peak SNR in dB between two PNGs; ``inf`` when they are identical."""
    import numpy as np
    from PIL import Image

    with Image.open(path_a) as raw_a, Image.open(path_b) as raw_b:
        img_a = np.asarray(raw_a.convert("RGB"), dtype=np.float64)
        img_b = np.asarray(raw_b.convert("RGB"), dtype=np.float64)
    if img_a.shape != img_b.shape:
        raise ValueError(f"image shapes differ: {img_a.shape} vs {img_b.shape}")
    mse = float(np.mean((img_a - img_b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(255.0 ** 2 / mse)


def free_port():
    """Grab a free TCP port for the next torchrun master."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def generate_and_save(pipe, args, resolution, out_path, device=None, rank=0):
    """One T2I request; rank 0 saves the PNG. Returns the wall seconds."""
    from PIL import Image

    with torch.inference_mode():
        start = time.perf_counter()
        image = pipe(
            **turbo_call_kwargs(args, resolution, resolution, device, rank)
        ).images[0]
        elapsed = time.perf_counter() - start
    if rank == 0:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        image.save(out_path)
        logging.info("rank 0: wrote %s in %.3fs", out_path, elapsed)
    return elapsed


def build_boosted_pipe(mode, rank, world, model_dir):
    """Load the pipeline and apply the ``mode`` boost (checked-in YAML)."""
    from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline
    from lite_boost import BoostManager

    pipe = BooguImageTurboPipeline.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    pipe = BoostManager()(pipe, config=boost_config_path(mode))
    logging.info("rank %d/%d: boosted as %s", rank, world, mode)
    return pipe


def install_mllm_offload(pipe, device):
    """Wrap ``encode_instruction`` to swap the encoder in for the encode.

    With ``--offload`` the card cannot hold DiT and encoder weights next to
    each other (that is the point of the switch on a dev card), so each encode
    is a swap: the transformer moves to host memory, the encoder moves in and
    encodes, moves back out, and the transformer is fetched back before the
    denoise loop. Both call paths go through the pipeline's
    ``encode_instruction`` (the two-card path's master-side tp encode calls it
    on cache miss), so wrapping it here covers 1p and sp2. The two-card path
    caches embeds and encodes once, single card re-encodes per request and
    pays the round trips every time -- that cost is why this is a dev-card
    switch, not a default. Pairs with ``install_vae_offload``.
    """
    original_encode = pipe.encode_instruction

    def encode_with_offload(*f_args, **f_kwargs):
        # The pipeline hands the precomputed embeds back into every internal
        # encode_instruction call (the encoder then skips them): forwarding
        # those untouched keeps the swaps to calls that actually run the mllm.
        guidance = (
            f_args[1] if len(f_args) > 1
            else f_kwargs.get("do_classifier_free_guidance", True)
        )
        runs_encoder = f_kwargs.get("instruction_embeds") is None or (
            guidance
            and (
                f_kwargs.get("negative_instruction_embeds") is None
                or f_kwargs.get("empty_instruction_embeds") is None
            )
        )
        if not runs_encoder:
            return original_encode(*f_args, **f_kwargs)
        transformer = pipe.transformer
        if next(transformer.parameters()).device.type != "cpu":
            transformer.to("cpu")
            torch.npu.empty_cache()
        if next(pipe.mllm.parameters()).device.type == "cpu":
            pipe.mllm.to(device)
        embeds = original_encode(*f_args, **f_kwargs)
        pipe.mllm.to("cpu")
        torch.npu.empty_cache()
        transformer.to(device)
        return embeds

    pipe.encode_instruction = encode_with_offload


def install_vae_offload(pipe, device):
    """Wrap ``vae.decode`` so the VAE is only resident around the decode.

    Same trade as ``install_mllm_offload``: encode needs the encoder on the
    card, decode needs the VAE, and a card that DiT + encoder + VAE together
    would fill keeps only one of the two peripherals resident at a time. On
    the multi-card path only rank 0 decodes (``tp_vae_decode`` returns zeros
    elsewhere), so the other ranks never fetch the VAE at all.
    """
    original_decode = pipe.vae.decode
    rank = int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))
    world = int(os.getenv("WORLD_SIZE", "1"))

    def decode_with_offload(*f_args, **f_kwargs):
        if world > 1 and rank != 0:
            return original_decode(*f_args, **f_kwargs)
        if next(pipe.vae.parameters()).device.type == "cpu":
            pipe.vae.to(device)
        images = original_decode(*f_args, **f_kwargs)
        pipe.vae.to("cpu")
        torch.npu.empty_cache()
        return images

    pipe.vae.decode = decode_with_offload


def boost_and_place(mode, rank, world, device, model_dir, offload=False):
    """Boost the pipeline and place it; single card holds everything.

    The whole pipeline (DiT + VAE + mllm encoder) lives on one card for 1p;
    on the two-card path ``tp_pipeline_call`` places the components per rank
    natively (encoder on rank 0), no test-script placement needed. With
    ``offload`` the encoder and VAE never stay resident (the two-card SP form
    replicates the full transformer on every rank, the TP form shards it):
    the encode rank swaps the transformer out for the encoder and the decode
    rank (0) swaps the VAE in, so a card that the three together would fill
    still runs the case.
    """
    from lite_boost.parallel import initialize_usp

    # Must precede BoostManager: the boost validates the configured world_size
    # against the distributed world size.
    initialize_usp()
    pipe = build_boosted_pipe(mode, rank, world, model_dir)
    if not offload:
        if world == 1:
            pipe.to(device)
        return pipe
    install_mllm_offload(pipe, device)
    install_vae_offload(pipe, device)
    if world == 1:
        # Nothing placed yet: the wrappers bring each component in when its
        # phase needs it (the first encode fetches the transformer too).
        return pipe
    # The boost placed the (replicated) transformer and the VAE per rank.
    # Send the VAE back everywhere and the transformer back on the encode
    # rank -- its encode cannot fit next to the transformer on a dev card,
    # and the master's tp encode would move the encoder in while it is.
    pipe.vae.to("cpu")
    if rank == 0:
        pipe.transformer.to("cpu")
    torch.npu.empty_cache()
    return pipe


def default_visible_devices(nproc):
    """Device list for a case: just the ranks."""
    return ",".join(str(idx) for idx in range(nproc))


def merge_result(result, line):
    """Fold one ``RESULT key=value ...`` line into the case result dict."""
    for token in line.split()[1:]:
        key, _, value = token.partition("=")
        if key:
            result[key] = value


def run_worker(args):
    """One boosted case inside a torchrun worker process."""
    load_torch_npu()
    logging.basicConfig(level=logging.INFO)
    rank = int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))
    world = int(os.getenv("WORLD_SIZE", "1"))
    device = f"npu:{rank}"
    # Mirrors the official inference_npu.py: boogu picks its NPU-fused kernels
    # from this env var while the modules are imported (flash-2-varlen
    # attention processors, NpuRMSNorm, npu_swiglu, float32 RoPE tables) --
    # unset, every module silently falls back to the plain implementations,
    # so it must be set before the first boogu import further down.
    os.environ["device"] = device

    pipe = boost_and_place(args.mode, rank, world, device, args.model, args.offload)
    out_path = args.out or os.path.join(
        args.out_dir, f"booguimage_{args.mode}_{args.resolution}.png"
    )
    call_device = None if world > 1 else device
    for _ in range(args.warmup):
        generate_and_save(pipe, args, args.resolution, out_path, call_device, rank)
    if rank == 0:
        torch.npu.reset_peak_memory_stats()

    times = [
        generate_and_save(pipe, args, args.resolution, out_path, call_device, rank)
        for _ in range(args.runs)
    ]
    if rank == 0:
        mean = sum(times) / len(times)
        all_times = ",".join(f"{t:.3f}" for t in times)
        print(
            f"RESULT mode={args.mode} res={args.resolution} runs={len(times)} "
            f"mean_s={mean:.3f} per_step_s={mean / args.steps:.4f} all=[{all_times}]",
            flush=True,
        )
        peak = torch.npu.max_memory_allocated() / 2 ** 30
        print(f"RESULT mode={args.mode} res={args.resolution} peak_gib={peak:.2f}", flush=True)


def launch_form(mode, args, stem, warmup, runs):
    """Run one boosted case in a fresh torchrun subprocess; return its RESULTs.

    ``stem`` names the output PNG and log inside ``args.out_dir``; ``warmup``
    and ``runs`` override the perf defaults (accuracy passes 0 and 1).
    """
    nproc = 2 if mode != "1p" else 1
    log_path = os.path.join(args.out_dir, f"{stem}.log")
    png_path = os.path.join(args.out_dir, f"{stem}.png")
    child_env = dict(os.environ)
    child_env.setdefault("ASCEND_RT_VISIBLE_DEVICES", default_visible_devices(nproc))
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--master_addr", "127.0.0.1",
        f"--master_port={free_port()}",
        os.path.abspath(__file__),
        "--stage", "worker", "--mode", mode, "--resolution", str(args.resolution),
        "--model", args.model, "--prompt", args.prompt,
        "--steps", str(args.steps), "--seed", str(args.seed),
        "--out", png_path, "--warmup", str(warmup), "--runs", str(runs),
    ]
    if args.offload:
        cmd.append("--offload")
    print(f"\n=== case {stem} ===\n{' '.join(cmd)}", flush=True)

    result = {}
    with open(log_path, "w", encoding="utf-8") as log, subprocess.Popen(
        cmd, env=child_env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    ) as proc:
        for line in proc.stdout:
            log.write(line)
            if line.startswith("RESULT"):
                print(line.rstrip(), flush=True)
                merge_result(result, line)
    result.update(mode=mode, res=str(args.resolution), rc=str(proc.returncode))
    return result


def print_summary(results, args):
    """Print the aggregated table and write it to ``<out_dir>/summary.txt``."""
    lines = [
        f"model={args.model} prompt=({'custom' if args.prompt != DEFAULT_PROMPT else 'default'})",
        "case                 runs  mean_s   s/step  peak_gib  rc",
    ]
    for res in results:
        name = f"{res['mode']}@{res['res']}"
        lines.append(
            f"{name:<20} {res.get('runs', '-'):>4}  "
            f"{res.get('mean_s', '-'):>6}  {res.get('per_step_s', '-'):>6}  "
            f"{res.get('peak_gib', '-'):>8}  {res.get('rc', '-')}"
        )
    summary = "\n".join(lines)
    print(f"\n{summary}", flush=True)
    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(summary + "\n")


def run_perf_driver(args):
    """Run every requested boost form at the resolution and summarise."""
    os.makedirs(args.out_dir, exist_ok=True)
    modes = [m for m in args.modes.split(",") if m]
    if not modes:
        logging.error(
            "--modes %r resolves to no case; pass comma separated forms e.g. sp2,1p",
            args.modes,
        )
        return 1
    results = [
        launch_form(mode, args, f"booguimage_{mode}_{args.resolution}", args.warmup, args.runs)
        for mode in modes
    ]
    print_summary(results, args)
    failed = [res for res in results if res.get("rc") != "0"]
    if failed:
        logging.error(
            "failed cases: %s", ", ".join(f"{r['mode']}@{r['res']}" for r in failed)
        )
    return 1 if failed else 0


def judge_psnr(reference_png, boosted_png, label, threshold):
    """Print the PSNR verdict for one boosted image; True on pass."""
    psnr = compute_psnr(reference_png, boosted_png)
    verdict = "PASS" if psnr >= threshold else "FAIL"
    print(
        f"RESULT psnr {label}={psnr:.2f}dB threshold={threshold} {verdict}",
        flush=True,
    )
    return psnr >= threshold


def run_acc_driver(args):
    """Run each form once and judge all against the npz reference."""
    os.makedirs(args.out_dir, exist_ok=True)
    reference = os.path.join(args.out_dir, f"acc_reference_{args.resolution}.png")
    width, height = save_reference_png(args.baseline, reference)
    if (width, height) != (args.resolution, args.resolution):
        logging.error(
            "reference %s is %dx%d but --resolution=%d; rerun with "
            "--resolution %d or regenerate the reference",
            args.baseline, width, height, args.resolution, width,
        )
        return 1

    acc_modes = ("1p", "sp2", "tp2")
    png = {
        mode: os.path.join(args.out_dir, f"acc_{mode}_{args.resolution}.png")
        for mode in acc_modes
    }
    for mode in acc_modes:
        result = launch_form(mode, args, f"acc_{mode}_{args.resolution}", 0, 1)
        if result["rc"] != "0":
            logging.error("case %s@%d failed, aborting before PSNR", mode, args.resolution)
            return 1

    ok = {
        mode: judge_psnr(reference, png[mode], f"reference_vs_{mode}", args.psnr_min)
        for mode in acc_modes
    }
    if all(ok.values()):
        print("RESULT accuracy ALL_PASS", flush=True)
        return 0
    logging.error(
        "accuracy gate failed (%s)",
        ", ".join(f"{mode}={passed}" for mode, passed in ok.items()),
    )
    return 1


def main():
    """Entry point: performance by default, accuracy with ``--baseline``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--resolution", type=int, default=512, help="square image resolution"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR, help="checkpoint dir")
    parser.add_argument(
        "--out-dir", default="output", help="output dir for PNGs, logs and summary"
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="test prompt")
    parser.add_argument("--steps", type=int, default=4, help="DMD inference steps")
    parser.add_argument("--seed", type=int, default=42, help="generator seed")
    parser.add_argument(
        "--offload", action="store_true",
        help="offload the mllm encoder to host memory after each text encode "
             "and fetch it back for the next one, so DiT and mllm can share "
             "one card that the weights alone would fill (dev-card validation; "
             "timings include the transfers, images are unaffected)",
    )
    parser.add_argument(
        "--modes", default="sp2,1p",
        help="performance mode only: boost forms to run, comma separated",
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="performance mode only: warmup runs per case"
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="performance mode only: measured runs per case"
    )
    parser.add_argument(
        "--psnr-min", type=float, default=PSNR_MIN_DB,
        help="accuracy mode only: PSNR threshold in dB",
    )
    parser.add_argument(
        "--baseline", default=None,
        help="npz holding the plain-PyTorch model's output image; "
             "given, the run is an accuracy comparison instead of performance",
    )
    parser.add_argument(
        "--stage", choices=("driver", "worker"), default="driver", help=argparse.SUPPRESS
    )
    parser.add_argument("--mode", default="sp2", help=argparse.SUPPRESS)
    parser.add_argument("--out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runs < 1 or args.steps < 1:
        parser.error("--runs and --steps must be >= 1")

    if args.stage == "worker":
        run_worker(args)
        return 0
    if args.baseline:
        return run_acc_driver(args)
    return run_perf_driver(args)


if __name__ == "__main__":
    sys.exit(main())
