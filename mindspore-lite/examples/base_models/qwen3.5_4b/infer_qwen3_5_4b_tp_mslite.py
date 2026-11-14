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
"""Qwen3.5-4B TP=2 inference (prefill + decode, image or text-only).

Two processes (one rank per chip, provider=ge + HCCL AllReduce). Vision is a
separate 1p model executed once on rank0's chip; with --no-image the vision
tower is skipped and a 16-row dummy image embedding is used.

Prefill uses the dynamic-seq graph (bucket = next of 256/512/1024/2048/4096).
Decode uses the static cap=--max-seq-len graph. Generated tokens are produced
by rank0 (lm_head is replicated) and broadcast to rank1 every step.
"""

import os

os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "21500-21600")

# The HCCL port range must be set before mindspore_lite loads HCCL.
# pylint: disable=wrong-import-position
import argparse
import gc
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import numpy as np
EX = Path(__file__).resolve().parent
sys.path.insert(0, str(EX))

import mindspore_lite as mslite  # noqa: E402
from infer_qwen3_5_4b_mslite import (  # noqa: E402
    Qwen354BInferencer, _build_mslite_inputs, _build_position_ids,
)
# pylint: enable=wrong-import-position

BUCKETS = (256, 512, 1024, 2048, 4096)
PF_ORDER = ["input_ids", "position_ids", "image_embeds", "attn_mask"]
DC_ORDER = ["input_ids", "attention_mask", "position_ids",
            "past_conv_states", "past_recurrent_states", "past_kv_cache"]


def pick_bucket(n):
    for b in BUCKETS:
        if b >= n:
            return b
    raise ValueError(f"sequence {n} exceeds max bucket {BUCKETS[-1]}")


def write_rank_table(device_ids, run_dir):
    devs = ",".join(f'{{"device_id":"{d}","rank_id":"{i}"}}'
                    for i, d in enumerate(device_ids))
    table = ('{"version":"1.0","server_count":"1","server_list":['
             '{"server_id":"127.0.0.1","device":[' + devs + '],'
             '"host_nic_ip":"reserve"}],"status":"completed"}')
    p = Path(run_dir) / "rank_table.json"
    p.write_text(table)
    return str(p)


def write_rank_config(run_dir, rank, table, buckets, image_tokens):
    """Write the per-rank GE config for the dynamic-seq prefill graph."""
    cache = Path(run_dir) / f"ge_cache_rank{rank}"
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    # GE requires >= 2 distinct dynamic-dims profiles; always compile the full
    # bucket set (Model.resize then selects the prompt's bucket).
    profiles = list(buckets)
    if len(set(profiles)) < 2:
        profiles = list(BUCKETS)
    entries = [f"{b},{b},{b},{b}" for b in profiles]
    cfg = Path(run_dir) / f"config_rank{rank}.ini"
    cfg.write_text(
        "[ascend_context]\n"
        f"rank_table_file={table}\n"
        "plugin_custom_ops=All\n"
        "\n"
        "[ge_session_options]\n"
        "ge.constLifecycle=session\n"
        "ge.exec.atomicCleanPolicy=1\n"
        "ge.event=notify\n"
        "ge.exec.staticMemoryPolicy=2\n"
        "ge.exec.precision_mode=force_fp16\n"
        "ge.externalWeight=2\n"
        f"ge.graph_compiler_cache_dir={cache}\n"
        "\n"
        "[ge_graph_options]\n"
        f"ge.inputShape=input_ids:1,-1;position_ids:4,1,-1;"
        f"image_embeds:{image_tokens},2560;attn_mask:1,1,-1,-1\n"
        f"ge.dynamicDims={';'.join(entries)}\n"
        "ge.dynamicNodeType=1\n")
    return str(cfg)


def write_decode_config(run_dir, rank, table):
    """Write the per-rank GE config for the decode graph."""
    cache = Path(run_dir) / f"ge_cache_decode_rank{rank}"
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    cfg = Path(run_dir) / f"config_decode_rank{rank}.ini"
    cfg.write_text(
        "[ascend_context]\n"
        f"rank_table_file={table}\n"
        "plugin_custom_ops=All\n"
        "\n"
        "[ge_session_options]\n"
        "ge.constLifecycle=session\n"
        "ge.exec.atomicCleanPolicy=1\n"
        "ge.event=notify\n"
        "ge.exec.staticMemoryPolicy=2\n"
        "ge.exec.precision_mode=force_fp16\n"
        "ge.externalWeight=2\n"
        f"ge.graph_compiler_cache_dir={cache}\n")
    return str(cfg)


def build_feed(args):
    """Tokenize the prompt, run the vision tower, and build the prefill feed."""
    inf = Qwen354BInferencer(
        vision_model_path="", prefill_model_path="", decode_model_path="",
        processor_id=args.processor, device="cpu", device_id=0,
        image_size=args.image_size, device_resident_states=False,
        no_image=args.no_image)
    if args.no_image:
        ids, mask, types, _, grid = inf._prepare_inputs("", args.prompt)
        image_tokens = 16
        embeds = np.zeros((16, 2560), dtype=np.float16)
        vision_ms = 0.0
    else:
        ids, mask, types, pix, grid = inf._prepare_inputs(args.image, args.prompt)
        image_tokens = int((ids == int(inf.cfg.image_token_id)).sum())
        ctx = mslite.Context()
        ctx.target = ["ascend"]
        ctx.ascend.device_id = int(args.device_ids.split(",")[0])
        vm = mslite.Model()
        vm.build_from_file(args.vision_model, mslite.ModelType.MINDIR, ctx)
        t0 = time.perf_counter()
        out = vm.predict(_build_mslite_inputs(
            vm, {"pixel_values": pix}, preferred_order=["pixel_values"]))
        vision_ms = (time.perf_counter() - t0) * 1000
        embeds = out[0].get_data_to_numpy().astype(np.float16)
        del out, vm
        gc.collect()
        if embeds.shape[0] != image_tokens:
            raise RuntimeError(f"image_embeds {embeds.shape[0]} != {image_tokens}")

    base = int(ids.shape[1])
    bucket = pick_bucket(base)
    pad = bucket - base
    ids = np.concatenate([ids[:, :-1], np.full((1, pad), 220, np.int64),
                          ids[:, -1:]], axis=1)
    types = np.concatenate([types[:, :-1], np.zeros((1, pad), np.int64),
                            types[:, -1:]], axis=1)
    mask = np.ones((1, bucket), np.int64)
    pos, _ = _build_position_ids(inf.cfg, ids, mask, types, grid)
    neg = np.float16(-65504.0)
    ar = np.arange(bucket)
    am = np.where(ar[None, :] > ar[:, None], neg, np.float16(0.0))[None, None]
    feed = {
        "input_ids": ids.astype(np.int32),
        "position_ids": pos.astype(np.int32),
        "image_embeds": embeds,
        "attn_mask": am.astype(np.float16),
    }
    return feed, bucket, image_tokens, vision_ms


def _prefill_forward(rank, ctx, dev, prefill_path, cfg, feed, bucket,
                     image_tokens, barrier, token_q):
    """Build one rank's prefill model, run it, and return the states."""
    pf = mslite.Model()
    t0 = time.perf_counter()
    pf.build_from_file(prefill_path, mslite.ModelType.MINDIR, ctx, cfg)
    pf_build = (time.perf_counter() - t0) * 1000
    ins = pf.get_inputs()
    names = [t.name for t in ins]
    shapes = {"input_ids": [1, bucket], "position_ids": [4, 1, bucket],
              "image_embeds": [image_tokens, 2560],
              "attn_mask": [1, 1, bucket, bucket]}
    pf.resize(ins, [shapes[n] for n in names])
    pf_inputs = _build_mslite_inputs(pf, feed, preferred_order=PF_ORDER)
    # GE checks output sizes against the largest compiled bucket.
    max_seq = max(BUCKETS)
    tpl = pf.get_outputs()
    pf_bufs = [
        _make_buf(tpl[0], [1, max_seq, 248320], mslite.DataType.FLOAT16, dev),
        _make_buf(tpl[1], [24, 1, 4096, 3], mslite.DataType.FLOAT16, dev),
        _make_buf(tpl[2], [24, 1, 16, 128, 128], mslite.DataType.FLOAT32, dev),
        _make_buf(tpl[3], [16, 1, 2, max_seq, 256], mslite.DataType.FLOAT16, dev),
    ]
    barrier.wait()
    pf.predict(pf_inputs, pf_bufs)  # warmup (compiles all buckets)
    barrier.wait()
    t0 = time.perf_counter()
    out = pf.predict(pf_inputs, pf_bufs)
    pf_ms = (time.perf_counter() - t0) * 1000
    logits = out[0].get_data_to_numpy().astype(np.float32)
    conv = out[1].get_data_to_numpy().astype(np.float16)
    rec = out[2].get_data_to_numpy().astype(np.float32)
    kv = out[3].get_data_to_numpy().astype(np.float16)[:, :, :, :bucket]
    del out
    token = int(np.argmax(logits[0, bucket - 1])) if rank == 0 else None
    if rank == 0:
        token_q.put(token)
    else:
        token = token_q.get()
    barrier.wait()
    return pf_ms, pf_build, conv, rec, kv, token


def _make_buf(tpl, shape, dtype, dev):
    tensor = mslite.Tensor(shape=list(shape), dtype=dtype, device=dev)
    tensor.name = tpl.name
    tensor.format = tpl.format
    return tensor


def _build_decode_model(ctx, decode_path, dc_cfg):
    dc = mslite.Model()
    t0 = time.perf_counter()
    dc.build_from_file(decode_path, mslite.ModelType.MINDIR, ctx, dc_cfg)
    dc_build = (time.perf_counter() - t0) * 1000
    return dc, dc_build


def _make_decode_states(bucket, cap, conv, rec, kv):
    past_conv = np.zeros((24, 1, 4096, 3), dtype=np.float16)
    past_conv[...] = conv
    past_rec = np.zeros((24, 1, 16, 128, 128), dtype=np.float32)
    past_rec[...] = rec
    past_kv = np.zeros((16, 1, 2, cap, 256), dtype=np.float16)
    past_kv[:, :, :, :bucket] = kv
    attn = np.zeros((1, cap), dtype=np.int32)
    attn[0, :bucket + 1] = 1
    return past_conv, past_rec, past_kv, attn


def _decode_loop(rank, dc, past_conv, past_rec, past_kv, attn, token, bucket,
                 cap, max_new, barrier, token_q):
    """Run the synchronized decode loop and return tokens and per-step latency."""
    del cap
    generated = [token]
    step_ms = []
    warm_feed = {
        "input_ids": np.array([[token]], dtype=np.int32),
        "attention_mask": attn.copy(),
        "position_ids": np.full((4, 1, 1), bucket, dtype=np.int32),
        "past_conv_states": past_conv,
        "past_recurrent_states": past_rec,
        "past_kv_cache": past_kv,
    }
    barrier.wait()
    dc.predict(_build_mslite_inputs(dc, warm_feed, preferred_order=DC_ORDER))
    barrier.wait()
    for step in range(max_new - 1):
        pos_val = bucket + step
        dc_feed = {
            "input_ids": np.array([[token]], dtype=np.int32),
            "attention_mask": attn,
            "position_ids": np.full((4, 1, 1), pos_val, dtype=np.int32),
            "past_conv_states": past_conv,
            "past_recurrent_states": past_rec,
            "past_kv_cache": past_kv,
        }
        dc_inputs = _build_mslite_inputs(dc, dc_feed, preferred_order=DC_ORDER)
        barrier.wait()
        t0 = time.perf_counter()
        dout = dc.predict(dc_inputs)
        step_ms.append((time.perf_counter() - t0) * 1000)
        lg = dout[0].get_data_to_numpy().astype(np.float32)
        past_conv = dout[1].get_data_to_numpy().astype(np.float16)
        past_rec = dout[2].get_data_to_numpy().astype(np.float32)
        past_kv = dout[3].get_data_to_numpy().astype(np.float16)
        del dout
        nxt = int(np.argmax(lg[0, 0])) if rank == 0 else None
        if rank == 0:
            token_q.put(nxt)
        else:
            nxt = token_q.get()
        barrier.wait()
        token = nxt
        generated.append(token)
        attn[0, bucket + step + 1] = 1
    return generated, step_ms


def worker(rank, device_id, prefill_path, decode_path, cfg, dc_cfg, feed, bucket,
           image_tokens, cap, max_new, barrier, token_q, result_q):
    """Run one rank's process: prefill, build decode, and generate tokens."""
    os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "21500-21600"
    ctx = mslite.Context()
    ctx.target = ["ascend"]
    ctx.ascend.device_id = int(device_id)
    ctx.ascend.rank_id = int(rank)
    ctx.ascend.provider = "ge"
    dev = f"ascend:{device_id}"

    pf_ms, pf_build, conv, rec, kv, token = _prefill_forward(
        rank, ctx, dev, prefill_path, cfg, feed, bucket, image_tokens,
        barrier, token_q)
    dc, dc_build = _build_decode_model(ctx, decode_path, dc_cfg)
    past_conv, past_rec, past_kv, attn = _make_decode_states(
        bucket, cap, conv, rec, kv)
    generated, step_ms = _decode_loop(
        rank, dc, past_conv, past_rec, past_kv, attn, token, bucket, cap,
        max_new, barrier, token_q)

    if rank == 0:
        result_q.put({"tokens": generated, "prefill_ms": pf_ms,
                      "decode_ms": step_ms, "pf_build": pf_build,
                      "dc_build": dc_build})
    else:
        result_q.put({"rank": 1})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill-dir", type=str, required=True,
                    help="dir with rank0/rank1 qwen3_5_llm_prefill_rank*_graph.mindir")
    ap.add_argument("--decode-dir", type=str, required=True,
                    help="dir with rank0/rank1 qwen3_5_llm_decode_rank*_graph.mindir")
    ap.add_argument("--processor", type=str, default=str(EX / "Qwen3.5-4B"))
    ap.add_argument("--vision-model", type=str, default="")
    ap.add_argument("--image", type=str, default="./demo.jpeg")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--prompt", type=str, default="Describe this image.")
    ap.add_argument("--no-image", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=4608)
    ap.add_argument("--device-ids", type=str, default="0,1")
    ap.add_argument("--work-dir", type=str, default="./qwen3_5_4b_tp_run",
                    help="directory for rank_table.json and per-rank GE cache")
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    for stale in list(EX.glob("model_build_cache_*")) + list(EX.glob("kernel_meta*")):
        shutil.rmtree(stale, ignore_errors=True)

    if not args.no_image and not args.vision_model:
        raise ValueError("--vision-model is required unless --no-image is set")
    feed, bucket, image_tokens, vision_ms = build_feed(args)
    print(f"prompt tokens={bucket} image_tokens={image_tokens} "
          f"vision={vision_ms:.2f} ms", flush=True)

    devices = [int(x) for x in args.device_ids.split(",")]
    table = write_rank_table(devices, work)
    cfgs = [write_rank_config(work, r, table, BUCKETS, image_tokens)
            for r in (0, 1)]
    dc_cfgs = [write_decode_config(work, r, table) for r in (0, 1)]
    ctx_mp = mp.get_context("spawn")
    barrier = ctx_mp.Barrier(2)
    token_q = ctx_mp.Queue()
    result_q = ctx_mp.Queue()
    procs = []
    for r in (1, 0):
        pf = Path(args.prefill_dir) / f"rank{r}" / f"qwen3_5_llm_prefill_rank{r}_graph.mindir"
        dc = Path(args.decode_dir) / f"rank{r}" / f"qwen3_5_llm_decode_rank{r}_graph.mindir"
        p = ctx_mp.Process(target=worker, args=(
            r, devices[r], str(pf), str(dc), cfgs[r], dc_cfgs[r], feed, bucket,
            image_tokens, args.max_seq_len, args.max_new_tokens,
            barrier, token_q, result_q))
        p.start()
        procs.append(p)
        if r == 1:
            time.sleep(10)
    res = []
    deadline = time.time() + 3600
    while len(res) < len(procs) and time.time() < deadline:
        try:
            res.append(result_q.get(timeout=5))
        except Exception:
            if all(not p.is_alive() for p in procs):
                break
    for p in procs:
        p.join(timeout=60)
    gen_list = [r for r in res if "tokens" in r]
    if not gen_list:
        print("TP inference workers failed; see tracebacks above.", flush=True)
        return
    gen = gen_list[0]
    toks = gen["tokens"]
    print(f"prefill {gen['prefill_ms']:.1f} ms, decode "
          f"{np.mean(gen['decode_ms']):.1f} ms/token (first token included)")
    text = Qwen354BInferencer(
        "", "", "", args.processor, device="cpu", image_size=args.image_size
    ).processor.tokenizer.decode(toks, skip_special_tokens=True)
    print("Generated:", text)


if __name__ == "__main__":
    main()
