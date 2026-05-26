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
"""
ParallelManager — one-line model parallelization.

Usage:
    model = WanModel.from_pretrained(ckpt_dir)
    model = ParallelManager(model)
    # model is modified in-place: forward → usp_dit_forward
    # all other methods (.to, .cpu, .eval, etc.) work as normal
"""
import os

import torch
import torch.distributed as dist
import torch_npu


def initialize_usp():
    """Initialize HCCL distributed environment for Ulysses Sequence Parallel."""
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)

    local_rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = str(os.getenv("MASTER_ADDR", "127.0.0.1"))
    port = int(os.getenv("MASTER_PORT", "29502"))

    torch.set_num_threads(int(os.getenv("NUM_THREADS", "24")))

    if not dist.is_initialized():
        dist.init_process_group(
            backend="hccl",
            init_method=f"tcp://{master_addr}:{port}",
            world_size=world_size,
            rank=local_rank,
        )
    torch_npu.npu.set_device(local_rank)


class ParallelManager:
    """
    Modify a supported model in-place for Ulysses Sequence Parallel inference.

    Usage:
        model = WanModel.from_pretrained(ckpt_dir)
        model = ParallelManager(model)   # model modified in-place, returned as-is
        output = model(x, t, context, seq_len)
    """

    def __new__(cls, target):
        from lite_boost.model import setup_model
        setup_model(target)
        return target
