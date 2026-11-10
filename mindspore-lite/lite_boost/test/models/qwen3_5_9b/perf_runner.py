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
"""Qwen3.5-9B LiteBoost prefill/decode benchmark (1p / TP / CP).

Usage (single card), from the directory holding this file::

    QWEN35_9B_MODEL_DIR=/path/Qwen3.5-9B python perf_runner.py

or under torchrun for multi-card, ``--parallel-mode`` picking the scheme::

    torchrun --nproc_per_node=2 perf_runner.py --model-dir /path/Qwen3.5-9B \
        [--parallel-mode tp|cp]

The script loads a bf16 checkpoint through the lite_boost adapter, applies the
selected optimization preset (``--opts``, the same names the accuracy runner
uses), then sweeps the prefill buckets 128/256/512/1024/2048 with one
``Qwen35GraphRunner``: startup captures every bucket's prefill graph and the
single decode graph once, and each request only replays them. Per shape it
reports

* ``prefill_ms`` -- cache reset + static-input copy + prefill graph replay,
  the same split the adapter's 403 baseline reports;
* ``first_token_ms`` -- TTFT, which adds input preparation and ``lm_head``;
* ``decode_ms_per_tok`` -- steady-state replay of the decode graph, first token
  excluded (it comes from prefill).

It lives next to the model tests because it *is* their performance harness:
``test_qwen3_5_9b.py --mode perf`` launches it once through torchrun and checks
the numbers it reports, so the benchmark and the case that guards it cannot
drift apart. ``python/model/qwen3_5_9b/`` keeps only the model adapter itself.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import  (npu backend)

logger = logging.getLogger(__name__)

DEFAULT_BUCKETS = (128, 256, 512, 1024, 2048)
DEFAULT_MODEL_ENV = "QWEN35_9B_MODEL_DIR"
SEED_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Artificial intelligence systems reason over long contexts, and prefilling "
    "those contexts dominates the end-to-end latency of large language models. "
)

NEW_OPS = {
    "opt_fused_conv_silu": True,
    "opt_full_attn_proj_fuse": True,
    "opt_chunk_qk_l2norm_sink": True,
    "opt_chunk_out_zero_pad": True,
}
OPTS = {
    "none": {},
    "new": NEW_OPS,
    "mid": {
        **NEW_OPS,
        "fuse_rms_norm": True,
        "fuse_rms_norm_gated": True,
        "fuse_inproj_ba": True,
        "opt_a_log_precompute": True,
    },
}


def _setup_device(device_id: int):
    """Initialize HCCL when launched under torchrun; return (rank, world, device)."""
    if dist.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from lite_boost.parallel import initialize_usp

        initialize_usp()
        rank = dist.get_rank()
        return rank, dist.get_world_size(), f"npu:{rank}"
    torch.npu.set_device(device_id)
    return 0, 1, f"npu:{device_id}"


def _parse_ints(text: str):
    """Parse a comma-separated integer list."""
    return tuple(int(part) for part in text.split(",") if part.strip())


def _make_prompt(tokenizer, length: int, device: str):
    """Build a deterministic prompt of exactly ``length`` token ids."""
    seed_ids = tokenizer(SEED_TEXT, add_special_tokens=False).input_ids
    if not seed_ids:
        raise ValueError("tokenizer produced an empty seed prompt")
    repeats = (length + len(seed_ids) - 1) // len(seed_ids)
    ids = (seed_ids * repeats)[:length]
    return torch.tensor([ids], dtype=torch.long, device=device)


def _load_model(model_dir: str, world_size: int, opts: str, device: str,
                parallel_mode: str = "tp"):
    """Load the bf16 checkpoint, boost it, and place it on the device."""
    from transformers import AutoModelForImageTextToText

    from lite_boost.model.qwen3_5_9b import boost_qwen3_5_9b

    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    logger.info("%s", f"  load (CPU): {time.time() - t0:.1f}s")

    t0 = time.time()
    boost_qwen3_5_9b(
        model, world_size=world_size, mode=parallel_mode,
        graph_mode=True, bucketed_prefill=True, **OPTS[opts],
    )
    logger.info("%s", f"  boost: {time.time() - t0:.1f}s")

    t0 = time.time()
    model.to(device).eval()
    torch_npu.npu.synchronize()
    logger.info("%s", f"  to({device}): {time.time() - t0:.1f}s")
    return model


def _bench_length(runner, tokenizer, length: int, max_new_tokens: int, device: str):
    """Replay one request and report its prefill and decode latency."""
    input_ids = _make_prompt(tokenizer, length, device)
    timing = {}
    out_ids = runner.generate(
        input_ids, attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens, timing=timing,
    )
    decode_ms = timing.get("decode_per_tok_ms", 0.0)
    return {
        "prefill_tokens": length,
        "prefill_bucket": timing.get("bucket", length),
        "prefill_ms": (timing.get("reset", 0.0) + timing.get("copy", 0.0)
                       + timing.get("replay", 0.0)) * 1000.0,
        "first_token_ms": timing.get("prefill_total", 0.0) * 1000.0,
        "decode_tokens": len(out_ids),
        "decode_ms_per_tok": decode_ms,
        "decode_tok_per_s": 1000.0 / decode_ms if decode_ms else 0.0,
        "output_ids": out_ids,
    }


def _log_results(rows, startup, rank: int):
    """Print the prefill/decode table and the startup numbers on rank 0."""
    if rank != 0:
        return
    logger.info("%s", "=" * 92)
    logger.info("%s", f"{'prefill':>8} {'bucket':>7} {'prefill(ms)':>12} "
                      f"{'TTFT(ms)':>10} {'decode(ms/tok)':>15} {'decode(tok/s)':>14}")
    for row in rows:
        logger.info("%s", f"{row['prefill_tokens']:>8} {row['prefill_bucket']:>7} "
                          f"{row['prefill_ms']:>12.2f} {row['first_token_ms']:>10.2f} "
                          f"{row['decode_ms_per_tok']:>15.2f} "
                          f"{row['decode_tok_per_s']:>14.2f}")
    logger.info("%s", "=" * 92)
    capture = {key.rsplit("_", 1)[-1]: round(val, 2)
               for key, val in startup.items() if key.startswith("init_capture_")}
    logger.info("%s", f"startup: prefill capture {capture} s, "
                      f"decode capture {startup.get('decode_capture', 0.0):.2f} s")


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Qwen3.5-9B LiteBoost prefill/decode benchmark.")
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get(DEFAULT_MODEL_ENV),
                        help=f"Hugging Face bf16 checkpoint ({DEFAULT_MODEL_ENV}).")
    parser.add_argument("--device-id", type=int, default=0,
                        help="NPU device id for 1p runs.")
    parser.add_argument("--lengths", type=str,
                        default=",".join(map(str, DEFAULT_BUCKETS)),
                        help="Comma-separated prefill token counts to sweep.")
    parser.add_argument("--buckets", type=str,
                        default=",".join(map(str, DEFAULT_BUCKETS)),
                        help="Comma-separated prefill buckets to capture.")
    parser.add_argument("--opts", default="mid", choices=sorted(OPTS),
                        help="Optimization preset.")
    parser.add_argument("--parallel-mode", default="tp", choices=("tp", "cp"),
                        help="Multi-card scheme used when WORLD_SIZE > 1.")
    parser.add_argument("--max-new-tokens", type=int, default=32,
                        help="Tokens generated per request after prefill.")
    parser.add_argument("--num-warmup", type=int, default=2,
                        help="Eager warmup steps before each graph capture.")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Write the benchmark dict to this JSON file.")
    return parser.parse_args()


def main(argv=None):
    """Load the boosted model and run the prefill/decode sweep."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = _parse_args() if argv is None else _parse_args()
    if not args.model_dir:
        logger.error("Set --model-dir or %s.", DEFAULT_MODEL_ENV)
        return 2
    lengths = _parse_ints(args.lengths)
    buckets = _parse_ints(args.buckets)
    missing = [length for length in lengths if length > max(buckets)]
    if missing:
        raise ValueError(f"lengths {missing} exceed the largest bucket {max(buckets)}")

    rank, world_size, device = _setup_device(args.device_id)
    if rank == 0:
        logger.info("%s", f"Model dir: {args.model_dir}")
        logger.info("%s", f"Device: {device}, world_size={world_size}, "
                          f"parallel_mode={args.parallel_mode}, opts={args.opts}, "
                          f"lengths={list(lengths)}")
    model = _load_model(args.model_dir, world_size, args.opts, device,
                        args.parallel_mode)

    from transformers import AutoProcessor

    from lite_boost.model.qwen3_5_9b.runner import Qwen35GraphRunner

    processor = AutoProcessor.from_pretrained(args.model_dir)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    # The sweep compares a fixed token count, so the checkpoint EOS stop set is
    # disabled here; production serving keeps the checkpoint default.
    runner = Qwen35GraphRunner(
        model, buckets=buckets, max_new_tokens=args.max_new_tokens,
        device=device, num_warmup=args.num_warmup, stop_token_ids=(),
    )
    startup = {}
    t0 = time.time()
    runner.warmup(timing=startup)
    startup["warmup_wall_s"] = time.time() - t0
    if rank == 0:
        logger.info("STARTUP: prefill_init=%.2fs decode_init=%.2fs wall=%.2fs",
                    startup.get("init_total", 0.0),
                    startup.get("decode_init_total", 0.0),
                    startup["warmup_wall_s"])

    rows = []
    for length in lengths:
        if rank == 0:
            logger.info("%s", f"[bench] prefill {length} tokens + "
                              f"{args.max_new_tokens} decode tokens ...")
        row = _bench_length(runner, tokenizer, length, args.max_new_tokens, device)
        if rank == 0:
            text = tokenizer.decode(row.pop("output_ids"), skip_special_tokens=True)
            logger.info("%s", f"  prefill {row['prefill_ms']:.2f} ms, "
                              f"decode {row['decode_ms_per_tok']:.2f} ms/tok, "
                              f"reply={text[:60]!r}")
        rows.append(row)

    _log_results(rows, startup, rank)
    payload = {"lengths": list(lengths), "buckets": list(buckets),
               "world_size": world_size, "opts": args.opts,
               "parallel_mode": args.parallel_mode,
               "max_new_tokens": args.max_new_tokens,
               "startup": {key: val for key, val in startup.items()
                           if isinstance(val, float)},
               "results": rows}
    if rank == 0 and args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("%s", f"Wrote {args.json_out}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
