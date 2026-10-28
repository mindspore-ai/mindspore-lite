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

"""Wan2.1-T2V-1.3B end-to-end inference on Ascend 300I Duo.

Any resolution (multiples of 16) and frame count (4k+1) can be given via
``--height/--width/--num-frames`` and must match the exported/converted
sub-models. Defaults are the official spec: 480x832, 81 frames (4k+1, 5s
@16fps), UniPC 50 steps, CFG 5.0 -- identical to the upstream diffusers
``WanPipeline`` semantics (the UniPC scheduler and its flow_shift are loaded
from the model dir's own config).

Long videos on 310P3:
  * transformer: CANN built-in PromptFlashAttention (injected by the export
    script) -- the flash kernel never materialises the [32760 x 32760] attention
    matrix that OOMs the stock path at conversion time;
  * VAE: single-pass decode of >13 latent frames fails GE tiling, so such
    videos are decoded in overlapping 13-latent segments (e.g. the 81-frame
    spec: latent[0:13] -> 49 frames, head-trimmed to 41; latent[8:21] -> 49
    frames, tail-trimmed to 40; the 3-latent overlap keeps the segment
    boundary at intra-segment smoothness) which reconstructs exactly 81 frames;
  * CFG runs on two chips concurrently (mslite predict releases the GIL, so a
    single process with two threads suffices).

Optional TP2 x CFG (experimental, ``--tp-size 2``): 4 chips form 2 independent
2-rank HCCL worlds (cond / uncond), each running a rank-sharded transformer
(export with ``--tp-size 2 --rank {0,1}``; the rank graphs carry
Custom(AllReduce) nodes and are compiled online by GE at build time). The
AllReduce reduction order makes the output very close to, but not bit-exact
with, the default path. The 2 chips of each world must sit on the same card
(HCCL goes over the intra-card HCCS link).

Run (after export + convert):
    python infer_wan2_1_t2v_1_3b_mslite.py \
        --mindir-dir ./wan2_1_t2v_1_3b_onnx --model-dir ./Wan2.1-T2V-1.3B-Diffusers \
        --device-a 4 --device-b 5 --prompt "A cat walking on a beach, cinematic, 4k."
"""

import argparse
import json
import multiprocessing as mp
import threading
import time
from pathlib import Path

import numpy as np

import mindspore_lite as mslite
# torch is used ONLY for the CPU/numpy-backed diffusers UniPC scheduler; the
# Ascend model inference is pure mslite + numpy (same as the flux1_dev example).
import torch  # noqa: E402  (scheduler only)
from transformers import AutoTokenizer  # noqa: E402
from diffusers import UniPCMultistepScheduler  # noqa: E402

_VAE_T = 4          # temporal downscale
_VAE_S = 8          # spatial downscale
# VAE segment decode for videos longer than this many latent frames
# (segment length 13, 3-latent overlap; see _vae_decode_and_save)
_VAE_SEG_LATENTS = 13


def _resolve_mindir(mindir_path):
    """Resolve a model path to the existing ``_graph.mindir`` or single ``.mindir``.

    A sub-model may be either ``<name>_graph.mindir`` (weights externalised into
    a sibling ``<name>_variables/`` dir, e.g. the transformer) or a single
    ``<name>.mindir`` (weights inlined, e.g. the VAE).
    """
    mindir_path = Path(mindir_path)
    candidates = [mindir_path,
                  mindir_path.with_name(mindir_path.stem.replace("_graph", "") + ".mindir")]
    return next((p for p in candidates if p.exists()), mindir_path)


def _build_model(mindir_path, device_id):
    """Build an mslite Model on a given Ascend device (graph+variables or single file)."""
    chosen = _resolve_mindir(mindir_path)
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = int(device_id)
    model = mslite.Model()
    model.build_from_file(str(chosen), mslite.ModelType.MINDIR, context)
    return model


def _np_to_input(tensor_info, array):
    """Cast a numpy array to the dtype expected by a model input tensor.

    ``tensor_info.dtype`` is a ``mindspore_lite.DataType`` enum (the converter casts
    the exported int64 input_ids/attention_mask to int32), so map via the enum
    rather than ``np.dtype``.
    """
    dtype_map = {mslite.DataType.FLOAT16: np.float16, mslite.DataType.FLOAT32: np.float32,
                 mslite.DataType.INT32: np.int32, mslite.DataType.INT64: np.int64,
                 mslite.DataType.BOOL: np.bool_}
    target = dtype_map.get(tensor_info.dtype)
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


def _encode_prompt(text_model, tokenizer, prompt, max_seq_len=512):
    """Encode one prompt and zero the embeddings past the attention-mask length.

    Matches diffusers' WanPipeline._get_t5_prompt_embeds: the UMT5 encoder
    produces non-zero values even for padded tokens, and without zeroing them the
    garbage corrupts the transformer's cross-attention.
    """
    toks = tokenizer(prompt, padding="max_length", max_length=max_seq_len,
                     truncation=True, add_special_tokens=True,
                     return_attention_mask=True, return_tensors="np")
    ids = toks["input_ids"].astype(np.int64)
    am = toks["attention_mask"].astype(np.int64)
    e = _run_model(text_model, {"input_ids": ids, "attention_mask": am},
                   ["input_ids", "attention_mask"])[0].astype(np.float32)
    e[:, int(am[0].sum()):, :] = 0.0
    return e


# ---------------------------------------------------------------------------
# Tensor-parallel (TP=2 x CFG) inference: 2 independent 2-rank HCCL worlds
# (cond / uncond), one worker process per rank, driven by this main process.
# The rank graphs carry Custom(AllReduce) nodes (export-time surgery); the two
# worlds are separate rank_tables so cond and uncond reduce within their own
# pair and both worlds run concurrently.
# ---------------------------------------------------------------------------

def _tp_write_rank_table(device_ids, run_dir):
    """Write rank_table.json for one 2-rank world; return its path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    devs_json = ",".join(
        f'{{"device_id":"{d}","rank_id":"{i}"}}' for i, d in enumerate(device_ids))
    table = ('{"version":"1.0","server_count":"1","server_list":['
             '{"server_id":"127.0.0.1","device":[' + devs_json + '],'
             '"host_nic_ip":"reserve"}],"status":"completed"}')
    path = run_dir / "rank_table.json"
    path.write_text(table)
    return str(path)


def _tp_write_config(rank_table_path, run_dir, cache_dir):
    """Write the HCCL config_file.ini used at build_from_file time.

    The TP rank graphs are plain MindIR (converted with optimize=none): GE
    compiles them online here, so the file also carries the session tuning
    (force_fp16) and a rank-private compile-cache dir -- two ranks
    cold-compiling into one directory race and kill each other.
    """
    run_dir = Path(run_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config_file.ini"
    path.write_text("[ascend_context]\n"
                    f"rank_table_file={rank_table_path}\n"
                    "plugin_custom_ops=All\n"
                    "\n[ge_session_options]\n"
                    "ge.constLifecycle=session\n"
                    "ge.exec.atomicCleanPolicy=1\n"
                    "ge.event=notify\n"
                    "ge.exec.staticMemoryPolicy=2\n"
                    "ge.exec.precision_mode=force_fp16\n"
                    "ge.externalWeight=2\n"
                    f"ge.graph_compiler_cache_dir={cache_dir}\n")
    return str(path)


def _tp_worker(group, rank, device, mindir, ini, task_q, result_q, noise_q):
    """One TP rank: build the rank graph in its HCCL world, serve latents.

    AllReduce syncs the two ranks of a world inside predict, so only rank 0
    ships its (already complete) noise_pred -- onto the shared ``noise_q`` --
    back to the main process (``result_q`` is this worker's private handshake
    queue). The encoder_hidden_states (text embeddings) are fixed across steps
    and arrive once in the "init" message. Builds are serialized by the main
    process ("go" handshake) so 4 concurrent online GE compilations don't
    oversubscribe the host CPUs.
    """
    ctx = mslite.Context()
    ctx.target = ["ascend"]
    ctx.ascend.device_id = int(device)
    ctx.ascend.rank_id = int(rank)
    ctx.ascend.provider = "ge"
    model = mslite.Model()
    result_q.put(("building", group, rank, None))
    task_q.get()   # wait for the main process to release this rank's build slot
    model.build_from_file(str(mindir), mslite.ModelType.MINDIR, ctx, str(ini))
    result_q.put(("ready", group, rank, None))
    ehs = None
    while True:
        msg = task_q.get()
        if msg is None:
            break
        if msg[0] == "init":
            ehs = msg[1]
            continue
        _, latent, timestep = msg
        ins = model.get_inputs()
        feeds = {"hidden_states": latent, "timestep": np.array([timestep], np.float32),
                 "encoder_hidden_states": ehs}
        tensors = [mslite.Tensor(_np_to_input(t, feeds[t.name])) for t in ins]
        outs = model.predict(tensors)
        if rank == 0:
            noise_q.put((group, outs[0].get_data_to_numpy()))
    result_q.put(("exit", group, rank, None))


def _vae_decode_and_save(models_dir, lat, lat_mean, lat_std,
                         dev_a, dev_b, output,
                         num_steps, dt_load, dt_text, dt_denoise):
    """Decode latents to video (segmented for long videos, dual-chip) + mp4 save."""
    t0 = time.perf_counter()
    lat_dn = lat * lat_std + lat_mean
    num_latents = lat_dn.shape[2]
    if num_latents <= _VAE_SEG_LATENTS:
        vae = _build_model(Path(models_dir) / "wan_vae_decoder_graph.mindir", dev_a)
        video = _run_model(vae, {"latents": lat_dn}, ["latents"])[0]
        mode = "single-pass"
    else:
        # Overlapping 13-latent segments: stride 10 with a 3-latent overlap so the
        # segment boundary stays at intra-segment smoothness (measured 0.325 vs
        # 0.337 intra on noise latents; a 1-latent overlap gave 0.368). The last
        # segment is anchored at the tail; segments alternate between both chips.
        seg_len, stride = _VAE_SEG_LATENTS, _VAE_SEG_LATENTS - 3
        starts = list(range(0, num_latents - seg_len + 1, stride))
        if starts[-1] != num_latents - seg_len:
            starts.append(num_latents - seg_len)
        vaes = [_build_model(Path(models_dir) / "wan_vae_decoder_13lat_graph.mindir", d)
                for d in (dev_a, dev_b)]
        seg_outs = [None] * len(starts)

        def run_jobs(chip, jobs):
            for idx, seg_start in jobs:
                seg_outs[idx] = _run_model(
                    vaes[chip], {"latents": lat_dn[:, :, seg_start:seg_start + seg_len]},
                    ["latents"])[0]

        jobs = [[], []]
        for idx, seg_start in enumerate(starts):
            jobs[idx % 2].append((idx, seg_start))
        threads = [threading.Thread(target=run_jobs, args=(chip, jobs[chip]))
                   for chip in (0, 1) if jobs[chip]]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        # A segment starting at latent s decodes 4*seg_len-3 = 49 frames covering
        # global frames [4s, 4s+49). Each segment "owns" global frames from the
        # next segment's 3-latent warmup boundary on: segment j keeps frames
        # [4*s_{j+1}+9 - 4*s_j:] locally (9 = 4*3-3, the frames the 3-latent
        # overlap re-decodes with fuller causal context), the first segment
        # keeps from 0 and the last keeps to the end.
        segs = []
        for idx, seg_start in enumerate(starts):
            out = seg_outs[idx]
            lo = 0 if idx == 0 else 9
            hi = (4 * starts[idx + 1] + 9 - 4 * seg_start) if idx < len(starts) - 1 else None
            segs.append(out[:, :, lo:hi])
        video = np.concatenate(segs, axis=2)
        mode = f"{len(starts)}-seg decode (dual-chip)"
    dt_vae = time.perf_counter() - t0
    print(f"[vae] {mode} {dt_vae:.1f}s -> {video.shape[2]} frames", flush=True)

    frames = ((video[0].transpose(1, 2, 3, 0) / 2 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    import imageio
    imageio.mimsave(output, [frames[i] for i in range(frames.shape[0])], fps=16)
    dt_e2e = dt_load + dt_text + dt_denoise + dt_vae
    print(f"[infer] saved {video.shape[2]} frames @16fps -> {output} "
          f"({video.shape[2]/16:.2f}s, {frames.shape[1]}x{frames.shape[2]})", flush=True)
    print("--- Performance ---")
    print(f"  Model load:              {dt_load:>12.1f} s")
    print(f"  Text encode (UMT5):      {dt_text:>12.2f} s")
    print(f"  Denoise ({num_steps} steps): {dt_denoise:>9.1f} s")
    print(f"  VAE decode:              {dt_vae:>12.1f} s")
    print(f"  End-to-end:              {dt_e2e:>12.1f} s ({dt_e2e/60:.1f} min)")


def _main_tp(args, models_dir, tokenizer, scheduler, lat_mean, lat_std):
    """TP=2 x CFG path: 4 rank worker processes in 2 independent HCCL worlds.

    cond world = devices[0:2], uncond world = devices[2:4]; each world reduces
    its own Custom(AllReduce) nodes (export-time surgery). The main process
    feeds latents, collects the two noise predictions (rank 0 of each world),
    runs the CFG merge + UniPC step on CPU, and broadcasts the next latent.
    """
    devices = [int(d) for d in str(args.tp_devices).split(",")]
    if len(devices) != 4:
        raise ValueError("--tp-devices needs 4 chips: cond_r0,cond_r1,uncond_r0,uncond_r1")
    run_dir = Path("tp_run")
    ctx_mp = mp.get_context("spawn")

    t0 = time.perf_counter()
    text_model = _build_model(models_dir / "wan_text_encoder_graph.mindir", devices[0])
    dt_load = time.perf_counter() - t0
    print(f"[load] text encoder built in {dt_load:.0f}s", flush=True)

    t0 = time.perf_counter()
    pe = _encode_prompt(text_model, tokenizer, args.prompt)
    ne = _encode_prompt(text_model, tokenizer, args.negative_prompt or "")
    dt_text = time.perf_counter() - t0
    print(f"[text] {dt_text:.2f}s", flush=True)
    del text_model   # free the 15 GB before workers allocate their ranks

    t0 = time.perf_counter()
    workers = []   # (group, rank)
    tasks, results = [], []
    noise_q = ctx_mp.Queue()
    for group in (0, 1):
        table = _tp_write_rank_table(devices[group * 2:group * 2 + 2], run_dir / f"group{group}")
        for rank in (0, 1):
            ini = _tp_write_config(table, run_dir / f"group{group}",
                                   run_dir / "cache" / f"g{group}r{rank}")
            tq = ctx_mp.Queue()
            tasks.append(tq)
            results.append(ctx_mp.Queue())
            rank_mindir = _resolve_mindir(
                models_dir / f"wan_transformer_pfa_rank{rank}_graph.mindir")
            p = ctx_mp.Process(target=_tp_worker,
                               args=(group, rank, devices[group * 2 + rank], rank_mindir,
                                     ini, tq, results[-1], noise_q),
                               daemon=True)
            p.start()
            workers.append((group, rank))
    # serialize the 4 online GE builds (each compile saturates the host CPUs;
    # running 4 at once thrashes the machine): release one rank at a time and
    # wait for its "ready"
    for i, (group, rank) in enumerate(workers):
        tasks[i].put(("go",))
        while True:
            msg = results[i].get()
            if msg[0] == "ready":
                break
        print(f"[load] TP rank built ({i+1}/4): group{group} rank{rank}", flush=True)
    dt_load += time.perf_counter() - t0

    num_latents = (args.num_frames - 1) // _VAE_T + 1
    latent_hw = (args.height // _VAE_S, args.width // _VAE_S)
    rng = np.random.RandomState(args.seed)
    lat = rng.standard_normal((1, 16, num_latents, *latent_hw)).astype(np.float32)
    scheduler.set_timesteps(args.num_inference_steps)
    scheduler.set_begin_index(0)
    lat_t = torch.from_numpy(lat)
    # ship the (fixed) text embeddings once per worker
    for group in (0, 1):
        ehs = pe if group == 0 else ne
        for r in (0, 1):
            tasks[group * 2 + r].put(("init", ehs))
    t0 = time.perf_counter()
    for i, t in enumerate(scheduler.timesteps):
        tt = float(t)
        for w in tasks:
            w.put(("step", lat, tt))
        noises = {}
        for _ in range(2):
            group, noise = noise_q.get()
            noises[group] = noise
        noise = (noises[1] + args.guidance_scale * (noises[0] - noises[1])).astype(np.float32)
        lat_t = scheduler.step(torch.from_numpy(noise), t, lat_t, return_dict=False)[0]
        lat = lat_t.numpy().astype(np.float32)
        if (i + 1) % 10 == 0 or i == 0:
            el = time.perf_counter() - t0
            print(f"[denoise] {i+1}/{args.num_inference_steps} ({el:.0f}s elapsed)",
                  flush=True)
    dt_denoise = time.perf_counter() - t0
    print(f"[denoise] total {dt_denoise:.0f}s", flush=True)
    for tq in tasks:
        tq.put(None)
    for q in results:
        q.get()   # exit ack (workers release their chips before the VAE builds)

    _vae_decode_and_save(models_dir, lat, lat_mean, lat_std,
                         devices[0], devices[1], args.output,
                         args.num_inference_steps, dt_load, dt_text, dt_denoise)


def main():
    """Parse arguments and run the Wan2.1-T2V-1.3B end-to-end pipeline."""
    parser = argparse.ArgumentParser(
        description="Wan2.1-T2V-1.3B MindSpore Lite inference (defaults: official 81-frame spec)")
    parser.add_argument("--mindir-dir", required=True,
                        help="dir with the 3 converted MindIR sub-models")
    parser.add_argument("--model-dir", required=True,
                        help="Wan2.1-T2V-1.3B-Diffusers weight dir (tokenizer/scheduler/vae config)")
    parser.add_argument("--prompt", default="A cat walking on a beach, cinematic, 4k.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", default="wan_output.mp4")
    parser.add_argument("--height", type=int, default=480,
                        help="video height; must match the exported/converted models")
    parser.add_argument("--width", type=int, default=832,
                        help="video width; must match the exported/converted models")
    parser.add_argument("--num-frames", type=int, default=81,
                        help="number of video frames (4k+1); must match the exported models")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-a", type=int, default=0,
                        help="chip A: text encoder + transformer cond + VAE")
    parser.add_argument("--device-b", type=int, default=1,
                        help="chip B: transformer uncond (CFG parallel)")
    parser.add_argument("--tp-size", type=int, default=1, choices=[1, 2],
                        help="2 = TP2 x CFG on 4 chips, experimental (needs the "
                             "per-rank MindIR from export --tp-size 2; output is "
                             "close to but not bit-exact with the default path)")
    parser.add_argument("--tp-devices", default="0,1,2,3",
                        help="4 chips for --tp-size 2: cond_rank0,cond_rank1,"
                             "uncond_rank0,uncond_rank1; each pair must sit on "
                             "the same card (intra-card HCCS link)")
    args = parser.parse_args()

    if args.height % 16 or args.width % 16:
        raise ValueError("height/width must be multiples of 16")
    if (args.num_frames - 1) % 4:
        raise ValueError("num-frames must be 4k+1 (official spec: 81)")

    models_dir = Path(args.mindir_dir)
    tokenizer = AutoTokenizer.from_pretrained(Path(args.model_dir) / "tokenizer")
    scheduler = UniPCMultistepScheduler.from_pretrained(
        Path(args.model_dir) / "scheduler")
    vae_cfg = json.loads((Path(args.model_dir) / "vae" / "config.json").read_text())
    lat_mean = np.array(vae_cfg["latents_mean"], dtype=np.float32).reshape((1, 16, 1, 1, 1))
    lat_std = np.array(vae_cfg["latents_std"], dtype=np.float32).reshape((1, 16, 1, 1, 1))

    if args.tp_size > 1:
        _main_tp(args, models_dir, tokenizer, scheduler, lat_mean, lat_std)
        return

    t0 = time.perf_counter()
    text_model = _build_model(models_dir / "wan_text_encoder_graph.mindir", args.device_a)
    ta = _build_model(models_dir / "wan_transformer_pfa_graph.mindir", args.device_a)
    tb = _build_model(models_dir / "wan_transformer_pfa_graph.mindir", args.device_b)
    dt_load = time.perf_counter() - t0
    print(f"[load] models built in {dt_load:.0f}s", flush=True)

    # ---- text encode ----
    t0 = time.perf_counter()
    pe = _encode_prompt(text_model, tokenizer, args.prompt)
    ne = _encode_prompt(text_model, tokenizer, args.negative_prompt or "")
    dt_text = time.perf_counter() - t0
    print(f"[text] {dt_text:.2f}s", flush=True)
    del text_model

    # ---- denoise (CFG dual-chip) ----
    num_latents = (args.num_frames - 1) // _VAE_T + 1
    latent_hw = (args.height // _VAE_S, args.width // _VAE_S)
    rng = np.random.RandomState(args.seed)
    lat = rng.standard_normal((1, 16, num_latents, *latent_hw)).astype(np.float32)
    scheduler.set_timesteps(args.num_inference_steps)
    scheduler.set_begin_index(0)
    lat_t = torch.from_numpy(lat)
    outs = {}
    t0 = time.perf_counter()
    for i, t in enumerate(scheduler.timesteps):
        tt = np.array([float(t)], dtype=np.float32)

        def work(model, embeds, key, timestep):
            outs[key] = _run_model(
                model, {"hidden_states": lat, "timestep": timestep,
                        "encoder_hidden_states": embeds},
                ["hidden_states", "timestep", "encoder_hidden_states"])[0]

        th1 = threading.Thread(target=work, args=(ta, pe, "cond", tt))
        th2 = threading.Thread(target=work, args=(tb, ne, "uncond", tt))
        th1.start()
        th2.start()
        th1.join()
        th2.join()
        noise = outs["uncond"] + args.guidance_scale * (outs["cond"] - outs["uncond"])
        lat_t = scheduler.step(torch.from_numpy(noise), t, lat_t, return_dict=False)[0]
        lat = lat_t.numpy().astype(np.float32)
        if (i + 1) % 10 == 0 or i == 0:
            el = time.perf_counter() - t0
            print(f"[denoise] {i+1}/{args.num_inference_steps} ({el:.0f}s elapsed)",
                  flush=True)
    dt_denoise = time.perf_counter() - t0
    print(f"[denoise] total {dt_denoise:.0f}s", flush=True)
    del ta, tb

    _vae_decode_and_save(models_dir, lat, lat_mean, lat_std,
                         args.device_a, args.device_b, args.output,
                         args.num_inference_steps, dt_load, dt_text, dt_denoise)


if __name__ == "__main__":
    main()
