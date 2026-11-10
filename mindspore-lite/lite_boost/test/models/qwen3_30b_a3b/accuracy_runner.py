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
the untouched ``Qwen3MoeForCausalLM`` from transformers in float32 on the host
(no sharding, no patched forward) and writes ``accuracy_reference.npz``. That
file is the pre-LiteBoost baseline checked in next to the test. Two reasons the
baseline cannot be produced at test time: the 61 GB checkpoint does not fit on
one 32 GB card, and the only multi-card layout available is LiteBoost's own TP
sharding, which would already be the thing under test.

    accuracy_runner.py --mode reference --model-dir <bf16 dir> \
        --out accuracy_reference.npz

``--mode fused`` runs the fully boosted TP model on the NPUs and writes the
last prefill logits plus the top-1 ids of every measured round; the test then
compares them with the fixture. It reuses the fixture's ``input_ids`` so both
sides see one prompt, independent of the tokenizer version. The NPU path is
reduction-order nondeterministic in bf16, so a single forward is not enough to
tell a stable decision from a rounding flip: only repeating the forward under
``HCCL_DETERMINISTIC=true`` and checking that top-1 is unchanged makes the
agreement number trustworthy.

    torchrun --nproc_per_node=4 accuracy_runner.py --mode fused \
        --model-dir <bf16 dir> --input-ids accuracy_reference.npz \
        --rounds 3 --out x.npz
"""
import argparse
import time

import numpy as np
import torch

# Longer than PROMPT_LEN so the prompt is varied prose rather than a repeated
# phrase: repetition makes the next token nearly certain and hides routing
# mistakes behind an easy prediction.
SEED_TEXT = (
    "Large language models have moved from research demonstrations to "
    "production systems that answer questions, summarize documents, and write "
    "software. Their cost is dominated by two phases: prefill, which processes "
    "the whole prompt in parallel, and decode, which emits one token at a time "
    "while reusing a growing key-value cache. Prefill is compute bound and "
    "benefits from large matrix multiplications and fused kernels, whereas "
    "decode is memory bound and depends on how quickly weights and caches can "
    "be streamed from high bandwidth memory. Mixture-of-experts architectures "
    "add a router that selects a small subset of experts for each token, so a "
    "model can hold many more parameters than it evaluates at once. Routing "
    "decisions are made per layer and per token, which makes the computation "
    "irregular: some experts receive hundreds of tokens while others receive a "
    "handful. Implementations therefore sort tokens by expert, run grouped "
    "matrix multiplications, and scatter the results back to their original "
    "positions. Tensor parallelism splits each weight matrix across devices "
    "and inserts collective communication after the row-parallel projections. "
    "Sequence parallelism instead divides the context, trading attention work "
    "for all-to-all exchanges. Both approaches are sensitive to the bandwidth "
    "and latency of the interconnect, and both can introduce small numerical "
    "differences because floating point addition is not associative. In "
    "practice, an implementation is considered correct when its logits match a "
    "reference within a tolerance that reflects the precision of the deployed "
    "data type, and when every decisive top-1 prediction is preserved. Testing "
    "such a system requires a fixed prompt, a saved reference produced without "
    "the optimization under test, and enough repeated measurements to separate "
    "a genuine regression from rounding noise. Engineers should also record "
    "the exact software stack, because a change in the collective "
    "communication library can move a near tie from one token id to another "
    "even when the model and the weights are unchanged. Careful reporting of "
    "margins, not just accuracy percentages, makes those cases visible."
)

# Positions whose full logits are part of the fixture, evenly spread over the
# 256-token prompt and ending on the last token. The fixture stays small (four
# vocab-sized vectors) while still covering early and late context.
REFERENCE_POSITIONS = (63, 127, 191, 255)
PROMPT_LEN = 256


def _parse_args():
    """Parse the command line for either mode."""
    parser = argparse.ArgumentParser(description="Qwen3-30B-A3B logits reference.")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--mode", choices=("reference", "fused"), required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--prompt-len", type=int, default=PROMPT_LEN)
    parser.add_argument("--input-ids", type=str, default=None,
                        help="npz fixture whose input_ids to reuse (fused mode)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="measured forwards whose top-1 is saved (repeatability)")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"),
                        default="float32",
                        help="reference dtype; float32 is the ground truth")
    return parser.parse_args()


def _make_prompt(tokenizer, length: int):
    seed = tokenizer(SEED_TEXT, add_special_tokens=False).input_ids
    repeats = (length + len(seed) - 1) // len(seed)
    return torch.tensor([(seed * repeats)[:length]], dtype=torch.long)


def _load_input_ids(args):
    """Reuse the fixture prompt, or tokenize the fixed seed text."""
    if args.input_ids:
        with np.load(args.input_ids) as data:
            ids = np.asarray(data["input_ids"], dtype=np.int64)
        return torch.from_numpy(ids).reshape(1, -1)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    return _make_prompt(tokenizer, args.prompt_len)


def _write_reference(args):
    """Plain transformers on CPU: no LiteBoost import, no sharding, no patches."""
    from transformers import Qwen3MoeForCausalLM

    dtype = getattr(torch, args.dtype)
    start = time.time()
    model = Qwen3MoeForCausalLM.from_pretrained(
        args.model_dir,
        dtype=dtype,
        attn_implementation="eager",
        experts_implementation="eager",
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
    top2 = logits[0].topk(2, dim=-1).values
    np.savez(
        args.out,
        input_ids=input_ids[0].numpy(),
        positions=positions.numpy(),
        logits=logits[0].index_select(0, positions).numpy(),
        top1=logits[0].argmax(-1).numpy().astype(np.int64),
        margin=(top2[:, 0] - top2[:, 1]).numpy().astype(np.float32),
        model=np.asarray(args.model_dir),
        dtype=np.asarray(args.dtype),
    )
    print(f"wrote {args.out}: logits {tuple(logits.shape)} at positions "
          f"{list(REFERENCE_POSITIONS)}, top-1 for all positions", flush=True)


def _run_fused(args):
    """LiteBoost TP sharding + all fused ops, on the NPUs."""
    from lite_boost.model.qwen3_30B_A3B import (
        boost_qwen3_30B_A3B,
        load_qwen3_moe_bf16,
    )
    from lite_boost.parallel import initialize_usp

    initialize_usp()
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    device = f"npu:{rank}"

    model = load_qwen3_moe_bf16(args.model_dir)
    boost_qwen3_30B_A3B(
        model, world_size=world, mode="tp",
        fuse_rms_norm=True, fuse_ffn=True, fuse_moe=True,
        fuse_moe_router=True, use_eager=False,
    )
    model.to(device)

    input_ids = _load_input_ids(args).to(device)
    with torch.no_grad():
        round_top1 = []
        logits = None
        for _ in range(args.rounds):
            logits = model(input_ids=input_ids, use_cache=False).logits
            round_top1.append(logits.argmax(-1)[0])
    if rank == 0:
        np.savez(
            args.out,
            logits=logits.float().cpu().numpy()[0],
            round_top1=torch.stack([t.cpu() for t in round_top1]).numpy(),
        )
        print(f"saved logits {tuple(logits.shape)} and top-1 of "
              f"{args.rounds} round(s) from mode=fused to {args.out}", flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def main():
    args = _parse_args()
    if args.mode == "reference":
        _write_reference(args)
        return
    _run_fused(args)


if __name__ == "__main__":
    main()
