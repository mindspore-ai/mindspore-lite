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
ParallelManager — one-line model parallelization for distributed inference.

This module provides the :func:`initialize_usp` function and the
:class:`ParallelManager` class, which together enable distributed inference
on supported models with minimal code changes. Two parallelism strategies
are applied automatically based on the model type:

- **Ulysses Sequence Parallel (USP)** for DiT models — sequence-dimension
  parallelism via ``all_to_all`` communication around attention layers.
- **Data Parallel (DP) temporal tiling** for VAE models — temporal-dimension
  slicing with overlap, distributed across devices.

Usage:
    >>> import os
    >>> os.environ["RANK"] = "0"
    >>> os.environ["WORLD_SIZE"] = "1"
    >>> from lite_boost.parallel import initialize_usp, ParallelManager
    >>> from wan.textimage2video import WanTI2V
    >>> initialize_usp()
    >>> pipe = WanTI2V(config=cfg, checkpoint_dir=ckpt_dir, ...)
    >>> ParallelManager(pipe)
"""
import os

import torch
import torch.distributed as dist
import torch_npu


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
        This function must be called before constructing :class:`ParallelManager`.
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


class ParallelManager:
    r"""
    Modify a supported model in-place for distributed parallel inference.

    :class:`ParallelManager` wraps a supported model or pipeline and patches
    it in-place for multi-NPU parallel inference. Two parallelism strategies
    are applied automatically based on the detected model components:

    - **Ulysses Sequence Parallel (USP)** for DiT models — patches the
      ``forward`` method and attention layers to enable sequence-dimension
      parallelism via ``all_to_all`` communication. Each device holds full
      model weights and operates on a slice of the sequence.
    - **Data Parallel (DP) temporal tiling** for VAE models — replaces
      ``vae.encode`` and ``vae.decode`` with DP temporal slicing versions
      that split the video along the temporal dimension into overlapping
      chunks, distribute them across devices, and gather results.

    When a pipeline object (e.g., ``WanT2V``) is passed, both strategies
    are applied: USP for the DiT model and DP for the VAE. When a raw
    ``WanModel`` is passed, only USP is applied.

    The model is modified in-place and returned as-is, so all existing
    attributes and methods (``.to``, ``.cpu``, ``.eval``, etc.) continue
    to work normally.

    Args:
        target (object): A supported pipeline object to be parallelized.
            Supported classes include ``WanT2V`` and ``WanTI2V``.

    Returns:
        object, the same instance modified in-place with USP-patched
        forward and attention methods (for DiT) and DP-patched encode/decode
        methods (for VAE).

    Raises:
        RuntimeError: If the model type is not supported by lite_boost.

    Examples:
        >>> import os
        >>> os.environ["RANK"] = "0"
        >>> os.environ["WORLD_SIZE"] = "1"
        >>> from lite_boost.parallel import initialize_usp, ParallelManager
        >>> from wan.textimage2video import WanTI2V
        >>> initialize_usp()
        >>> pipe = WanTI2V(config=cfg, checkpoint_dir=ckpt_dir, ...)
        >>> ParallelManager(pipe)
    """

    def __new__(cls, target):
        from lite_boost.model import setup_model
        setup_model(target)
        return target
