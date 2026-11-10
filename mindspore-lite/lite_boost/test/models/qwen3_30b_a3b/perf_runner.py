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
"""Qwen3-30B-A3B LiteBoost inference and prefill/decode benchmark (1p / TP).

Usage (TP=2, two chips on one card), from the directory holding this file:
    torchrun --nproc_per_node=2 perf_runner.py \
        --model-dir /path/Qwen3-30B-A3B --graph

The script loads a bf16 Hugging Face checkpoint through the lite_boost adapter,
applies every Qwen3-30B-A3B boost (fused RMSNorm/FFN/MoE/router), then sweeps
the prefill buckets 128/256/512/1024/2048 and reports the prefill replay latency
and the steady-state decode latency per token, following the prefill + decode
split used by the LLM examples under examples/base_models.

It lives next to the model tests because it *is* their performance harness:
``test_qwen3_30b_a3b.py --mode perf`` launches it once through torchrun and
checks the numbers it reports, so the benchmark and the case that guards it
cannot drift apart. ``python/model/qwen3_30B_A3B/`` keeps only the model
adapter itself. The ``*_runner.py`` naming (this file for ``--mode perf``,
``accuracy_runner.py`` for ``--mode accuracy``) is the convention every model
directory follows.
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
import torch_npu

logger = logging.getLogger(__name__)

DEFAULT_LENGTHS = (128, 256, 512, 1024, 2048)
SEED_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Artificial intelligence systems reason over long contexts, and prefilling "
    "those contexts dominates the end-to-end latency of large language models. "
)


def _setup_env(graph: bool):
    """Set the lite_boost env switches before any model/graph code reads them."""
    os.environ["LB_PREFILL_GRAPH"] = "1" if graph else "0"
    if graph:
        os.environ.setdefault("LB_IFA_MASK", "2")
        os.environ.setdefault("LB_TP_SEPARATE_MM_AR", "1")


def _setup_device(device_id: int):
    """Initialize HCCL when launched under torchrun; return (rank, world, device)."""
    if dist.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from lite_boost.parallel import initialize_usp
        initialize_usp()
        rank = dist.get_rank()
        return rank, dist.get_world_size(), f"npu:{rank}"
    torch.npu.set_device(device_id)
    return 0, 1, f"npu:{device_id}"


def _parse_lengths(text: str):
    """Parse a comma-separated bucket list."""
    return tuple(int(part) for part in text.split(",") if part.strip())


def _make_prompt(tokenizer, length: int, device: str, offset: int = 0):
    """Build a deterministic prompt of exactly ``length`` token ids."""
    seed_ids = tokenizer(SEED_TEXT, add_special_tokens=False).input_ids
    if not seed_ids:
        raise ValueError("tokenizer produced an empty seed prompt")
    rotated = seed_ids[offset:] + seed_ids[:offset]
    repeats = (length + len(rotated) - 1) // len(rotated)
    ids = (rotated * repeats)[:length]
    return torch.tensor([ids], dtype=torch.long, device=device)


def _load_model(model_dir: str, tokenizer_dir: str, world_size: int, device: str):
    """Load the bf16 checkpoint, boost it, and place it on the device."""
    from lite_boost.model.qwen3_30B_A3B import (
        boost_qwen3_30B_A3B,
        load_qwen3_moe_bf16,
    )
    from transformers import AutoTokenizer

    t0 = time.time()
    model = load_qwen3_moe_bf16(model_dir, attn_implementation="eager")
    logger.info("%s", f"  load (CPU): {time.time() - t0:.1f}s")

    t0 = time.time()
    boost_qwen3_30B_A3B(
        model, world_size=world_size, mode="tp",
        fuse_rms_norm=True, fuse_ffn=True, fuse_moe=True,
        fuse_moe_router=True, use_eager=False,
    )
    logger.info("%s", f"  boost: {time.time() - t0:.1f}s")

    t0 = time.time()
    model.to(device)
    model.eval()
    torch_npu.npu.synchronize()
    logger.info("%s", f"  to({device}): {time.time() - t0:.1f}s")
    return model, AutoTokenizer.from_pretrained(tokenizer_dir)


def _bench_length(model, tokenizer, length: int, args, device: str):
    """Capture and replay one bucket; report its prefill and decode latency."""
    from lite_boost.model.qwen3_30B_A3B.graph_decode import graph_generate

    input_ids = _make_prompt(tokenizer, length, device)
    timing = {}
    out_ids = graph_generate(
        model, input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
        max_new_tokens=args.max_new_tokens, device=device,
        capture_graph=args.graph, timing=timing,
    )
    decode_ms = timing.get("decode_per_tok_ms", 0.0)
    return {
        "prefill_tokens": length,
        "prefill_ms": timing.get("prefill", 0.0) * 1000.0,
        "prefill_bucket": timing.get("prefill_bucket", length),
        "prefill_capture_s": timing.get("prefill_capture", 0.0),
        "decode_tokens": len(out_ids),
        "decode_ms_per_tok": decode_ms,
        "decode_tok_per_s": 1000.0 / decode_ms if decode_ms else 0.0,
        "output_ids": out_ids,
    }


def _bench_session(model, tokenizer, prefix_len: int, args, device: str):
    """Prefill a shared prefix once, then serve one suffix request."""
    from lite_boost.model.qwen3_30B_A3B.graph_decode import CommonPrefixSession

    prefix_ids = _make_prompt(tokenizer, prefix_len, device)
    t0 = time.time()
    session = CommonPrefixSession(
        model, prefix_ids, max_new_tokens=args.max_new_tokens, device=device,
    )
    session_setup_s = time.time() - t0
    # Start the suffix a few tokens into the seed so it does not extend the
    # common prefix beyond the intended shared part.
    suffix_ids = _make_prompt(tokenizer, args.lengths[-1], device, offset=8)
    request_ids = torch.cat([prefix_ids, suffix_ids], dim=1)
    timing = {}
    try:
        out_ids = session.generate(
            request_ids, torch.ones_like(request_ids), timing=timing)
    finally:
        session.close()
    return {
        "prefix_tokens": prefix_len,
        "suffix_prefill_tokens": suffix_ids.shape[1],
        "session_setup_s": session_setup_s,
        "suffix_prefill_ms": timing.get("prefill", 0.0) * 1000.0,
        "decode_ms_per_tok": timing.get("decode_per_tok_ms", 0.0),
        "decode_graph_reused": timing.get("decode_graph_reused", False),
        "decode_tokens": len(out_ids),
    }


def _log_results(results, session_result, rank: int):
    """Print the prefill/decode table and the session numbers on rank 0."""
    if rank != 0:
        return
    logger.info("%s", "=" * 78)
    logger.info("%s", f"{'prefill':>8} {'bucket':>7} {'prefill(ms)':>12} "
                      f"{'decode(ms/tok)':>15} {'decode(tok/s)':>14}")
    for row in results:
        logger.info("%s", f"{row['prefill_tokens']:>8} {row['prefill_bucket']:>7} "
                          f"{row['prefill_ms']:>12.2f} "
                          f"{row['decode_ms_per_tok']:>15.2f} "
                          f"{row['decode_tok_per_s']:>14.2f}")
    logger.info("%s", "=" * 78)
    if session_result is not None:
        logger.info("%s", f"common prefix {session_result['prefix_tokens']} tokens, "
                          f"suffix {session_result['suffix_prefill_tokens']} tokens: "
                          f"setup {session_result['session_setup_s']:.2f}s, "
                          f"suffix prefill {session_result['suffix_prefill_ms']:.2f} ms, "
                          f"decode {session_result['decode_ms_per_tok']:.2f} ms/tok")


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Qwen3-30B-A3B LiteBoost inference / prefill+decode benchmark.")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Hugging Face bf16 checkpoint directory.")
    parser.add_argument("--tokenizer-dir", type=str, default=None,
                        help="Tokenizer directory (defaults to --model-dir).")
    parser.add_argument("--device-id", type=int, default=0,
                        help="NPU device id for 1p runs.")
    parser.add_argument("--lengths", type=str, default=",".join(map(str, DEFAULT_LENGTHS)),
                        help="Comma-separated prefill token counts to sweep.")
    parser.add_argument("--max-new-tokens", type=int, default=32,
                        help="Tokens generated per request after prefill.")
    parser.add_argument("--graph", action="store_true",
                        help="Capture bucketed prefill + decode NPU graphs.")
    parser.add_argument("--session", action="store_true",
                        help="Also exercise the common-prefix session path.")
    parser.add_argument("--prefix-len", type=int, default=512,
                        help="Shared prefix length for --session.")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Write the benchmark dict to this JSON file.")
    return parser.parse_args()


def main():
    """Load the boosted model and run the prefill/decode sweep."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = _parse_args()
    args.lengths = _parse_lengths(args.lengths)
    tokenizer_dir = args.tokenizer_dir or args.model_dir
    _setup_env(args.graph)
    rank, world_size, device = _setup_device(args.device_id)

    if rank == 0:
        logger.info("%s", f"Model dir: {args.model_dir}")
        logger.info("%s", f"Device: {device}, world_size={world_size}, "
                          f"graph={args.graph}, lengths={list(args.lengths)}")
    model, tokenizer = _load_model(args.model_dir, tokenizer_dir, world_size, device)

    results = []
    for length in args.lengths:
        if rank == 0:
            logger.info("%s", f"[bench] prefill {length} tokens + "
                              f"{args.max_new_tokens} decode tokens ...")
        row = _bench_length(model, tokenizer, length, args, device)
        if rank == 0:
            text = tokenizer.decode(row.pop("output_ids"), skip_special_tokens=True)
            logger.info("%s", f"  prefill {row['prefill_ms']:.2f} ms, "
                              f"decode {row['decode_ms_per_tok']:.2f} ms/tok, "
                              f"reply={text[:60]!r}")
        results.append(row)

    session_result = None
    if args.session:
        if rank == 0:
            logger.info("%s", f"[session] shared prefix {args.prefix_len} tokens ...")
        session_result = _bench_session(
            model, tokenizer, args.prefix_len, args, device)

    _log_results(results, session_result, rank)
    payload = {"lengths": list(args.lengths), "world_size": world_size,
               "graph": args.graph, "results": results, "session": session_result}
    if rank == 0 and args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("%s", f"Wrote {args.json_out}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
