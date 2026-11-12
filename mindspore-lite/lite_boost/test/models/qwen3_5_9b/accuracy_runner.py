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
"""Produce or regenerate the accuracy fixture, or dump the fused logits.

``--mode reference`` is an offline tool, not something the test runs: it loads
the untouched checkpoint from transformers in float32 on the host (no
LiteBoost import, no sharding, no patched forward) and writes
``accuracy_reference.npz`` next to the test.  That file is the pre-LiteBoost
baseline, so a plain Python forward never has to run inside the test and the
checkpoint is loaded once per mode.

    accuracy_runner.py --mode reference --model-dir <bf16 dir> \
        --out accuracy_reference.npz

``--mode fused`` runs the boosted model on the NPUs through the same pooled
prefill graph a request uses, and writes logits at the fixture's positions plus
a top-1 id for every prompt position.  It reuses the fixture's ``input_ids`` so
both sides see one prompt, independent of the tokenizer version.

    torchrun --nproc_per_node=2 accuracy_runner.py --mode fused \
        --model-dir <bf16 dir> --input-ids accuracy_reference.npz --out x.npz \
        [--parallel-mode tp|cp]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

# The fused side reads the pooled prefill output buffer, like the request path.
# pylint: disable=protected-access

SEED_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Artificial intelligence systems reason over long contexts, and prefilling "
    "those contexts dominates the end-to-end latency of large language models. "
)

# Positions whose full logits are part of the fixture, evenly spread over the
# 64-token prompt and ending on the last token.  The fixture stays small (four
# vocab-sized vectors) while still covering early and late context.
REFERENCE_POSITIONS = (15, 31, 47, 63)
PROMPT_LEN = 64


def _parse_args():
    """Parse the command-line arguments for either mode."""
    parser = argparse.ArgumentParser(description="Qwen3.5-9B logits reference.")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--mode", choices=("reference", "fused"), required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--prompt-len", type=int, default=PROMPT_LEN)
    parser.add_argument("--input-ids", type=str, default=None,
                        help="npz fixture whose input_ids to reuse (fused mode)")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"),
                        default="float32",
                        help="reference dtype; float32 is the ground truth")
    parser.add_argument("--buckets", type=str, default="128",
                        help="Buckets to capture for the pooled prefill.")
    parser.add_argument("--opts", default="mid",
                        choices=("none", "new", "mid"),
                        help="Optimization preset applied before the graph run.")
    parser.add_argument("--parallel-mode", default="tp", choices=("tp", "cp"),
                        help="Multi-card scheme used when WORLD_SIZE > 1.")
    parser.add_argument("--num-warmup", type=int, default=2)
    parser.add_argument("--device-id", type=int, default=0,
                        help="NPU device id for 1p runs.")
    return parser.parse_args()


def _make_prompt(tokenizer, length: int):
    seed = tokenizer(SEED_TEXT, add_special_tokens=False).input_ids
    repeats = (length + len(seed) - 1) // len(seed)
    return torch.tensor([(seed * repeats)[:length]], dtype=torch.long)


def _load_input_ids(args):
    """Reuse the fixture prompt, or tokenize the fixed seed text."""
    if args.input_ids:
        with np.load(args.input_ids) as data:
            ids = torch.from_numpy(np.asarray(data["input_ids"], dtype=np.int64))
        # The fixture stores the prompt as a flat [prompt_len] vector.
        return ids.unsqueeze(0) if ids.dim() == 1 else ids
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    return _make_prompt(tokenizer, args.prompt_len)


def _write_reference(args):
    """Plain transformers on CPU: no LiteBoost import, no sharding, no patches."""
    from transformers import AutoModelForImageTextToText

    dtype = getattr(torch, args.dtype)
    start = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, dtype=dtype, attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"reference: loaded on CPU in {time.time() - start:.1f}s "
          f"({args.dtype})", flush=True)

    input_ids = _load_input_ids(args)
    start = time.time()
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits.float()
    print(f"reference: prefill {input_ids.shape[1]} tokens in "
          f"{time.time() - start:.1f}s", flush=True)

    positions = torch.tensor(REFERENCE_POSITIONS)
    np.savez(
        args.out,
        input_ids=input_ids[0].numpy(),
        positions=positions.numpy(),
        logits=logits[0].index_select(0, positions).numpy(),
        top1=logits[0].argmax(-1).numpy().astype(np.int64),
        model=np.asarray(args.model_dir),
        dtype=np.asarray(args.dtype),
    )
    print(f"wrote {args.out}: logits {tuple(logits.shape)} at positions "
          f"{list(REFERENCE_POSITIONS)}, top-1 for all positions", flush=True)


def _boost_presets():
    """Optimization presets, matching the benchmark's ``--opts`` names."""
    new_ops = {
        "opt_fused_conv_silu": True,
        "opt_full_attn_proj_fuse": True,
        "opt_chunk_qk_l2norm_sink": True,
        "opt_chunk_out_zero_pad": True,
    }
    return {
        "none": {},
        "new": new_ops,
        "mid": {
            **new_ops,
            "fuse_rms_norm": True,
            "fuse_rms_norm_gated": True,
            "fuse_inproj_ba": True,
            "opt_a_log_precompute": True,
        },
    }


def _setup_device(device_id: int):
    """Initialize HCCL when launched under torchrun; return (rank, world, device)."""
    import torch.distributed as dist

    if dist.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from lite_boost.parallel import initialize_usp

        initialize_usp()
        rank = dist.get_rank()
        return rank, dist.get_world_size(), f"npu:{rank}"
    torch.npu.set_device(device_id)
    return 0, 1, f"npu:{device_id}"


def _run_fused(args):
    """LiteBoost sharding + fused ops + the pooled prefill graph, on the NPUs."""
    # Multi-card accuracy must be reproducible run to run: pin HCCL to its
    # deterministic algorithm set before torch_npu initializes the comm domain.
    # Only the accuracy path sets this; the perf sweep leaves HCCL at default.
    os.environ.setdefault("HCCL_DETERMINISTIC", "true")
    import torch_npu  # noqa: F401  # pylint: disable=unused-import  (npu backend)
    from transformers import AutoModelForImageTextToText

    from lite_boost.model.qwen3_5_9b import boost_qwen3_5_9b
    from lite_boost.model.qwen3_5_9b.graph_decode import (
        init_bucket_pipelines,
        pooled_prefill,
    )

    rank, world, device = _setup_device(args.device_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    boost_qwen3_5_9b(
        model, world_size=world, mode=args.parallel_mode,
        graph_mode=True, bucketed_prefill=True, **(_boost_presets()[args.opts]),
    )
    model.to(device).eval()

    buckets = tuple(int(part) for part in args.buckets.split(",") if part.strip())
    init_bucket_pipelines(model, buckets=buckets, device=device,
                          num_warmup=args.num_warmup)
    input_ids = _load_input_ids(args).to(device)
    timing = {}
    pooled_prefill(model, input_ids, device=device, timing=timing)
    real_len = input_ids.shape[1]
    bucket = timing["bucket"]
    positions = torch.tensor(REFERENCE_POSITIONS)

    with torch.no_grad():
        hidden = model._pool_outs[bucket][:, :real_len, :]
        all_logits = model.lm_head(hidden)[0].float().cpu()
    if rank == 0:
        np.savez(
            args.out,
            logits=all_logits[positions].numpy(),
            top1=all_logits.argmax(-1).numpy().astype(np.int64),
        )
        print(f"saved logits {tuple(all_logits.shape)} at positions "
              f"{list(REFERENCE_POSITIONS)} from mode=fused to {args.out}",
              flush=True)
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main():
    args = _parse_args()
    if args.mode == "reference":
        _write_reference(args)
        return
    _run_fused(args)


if __name__ == "__main__":
    main()
