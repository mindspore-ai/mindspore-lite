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
r"""
Distributed-environment initializer for parallel inference.

This module provides the :func:`initialize_usp` function, which sets up the
HCCL distributed environment required before constructing
:class:`lite_boost.BoostManager`:

Usage:
    >>> import os
    >>> os.environ["RANK"] = "0"
    >>> os.environ["WORLD_SIZE"] = "1"
    >>> from lite_boost.parallel import initialize_usp
    >>> initialize_usp()
"""
import os

import torch
import torch.distributed as dist
import torch_npu


def get_world_size() -> int:
    """Return the distributed world size, or 1 when not initialized."""
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Return the rank of the current process, or 0 when not initialized."""
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_distributed_active():
    """Return True when the distributed process group is initialized (world size > 1)."""
    return dist is not None and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def all_reduce(x, op=dist.ReduceOp.SUM):
    """All-reduce ``x`` in place across ranks; return ``x`` unchanged when not initialized."""
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=op)
    return x


def broadcast(x, src=0):
    """Broadcast ``x`` from rank ``src``; return ``x`` unchanged when not initialized."""
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.broadcast(x, src=src)
    return x


def initialize_usp():
    r"""
    Initialize the HCCL distributed environment for parallel inference.

    This function configures the NPU runtime settings and initializes the HCCL
    distributed process group by reading the following environment variables:

    - ``RANK``: Local rank of the current process. Default: ``0``.
    - ``WORLD_SIZE``: Total number of distributed processes. Default: ``1``.
    - ``MASTER_ADDR``: IP address of the master node. Default: ``"127.0.0.1"``.
    - ``MASTER_PORT``: Port of the master node. Default: ``29502``.
    - ``NUM_THREADS``: Number of CPU threads per process. Default: ``24``.

    If the distributed process group has not been initialized, this function will
    initialize it with the ``hccl`` backend. After initialization, the NPU device
    corresponding to ``RANK`` is set as the active device.

    Note:
        This function must be called before constructing :class:`BoostManager`.
        It is typically invoked at the entry point of a distributed training or
        inference script.

    Raises:
        RuntimeError: If HCCL process group initialization fails.

    Examples:
        >>> import os
        >>> os.environ["RANK"] = "0"
        >>> os.environ["WORLD_SIZE"] = "1"
        >>> from lite_boost.parallel import initialize_usp
        >>> initialize_usp()
    """
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
