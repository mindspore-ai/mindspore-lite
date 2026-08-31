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
BoostManager — one-line model parallelization for distributed inference.

This module provides the :class:`BoostManager` class, which enables
distributed inference on supported models with minimal code changes. Call
:func:`lite_boost.parallel.initialize_usp` first to set up the HCCL distributed
environment (implementation lives in ``lite_boost/parallel/_initializer.py``).
Two parallelism strategies are applied automatically based on the model type:

- **Ulysses Sequence Parallel (USP)** for DiT models — sequence-dimension
  parallelism via ``all_to_all`` communication around attention layers.
- **Data Parallel (DP) temporal tiling** for VAE models — temporal-dimension
  slicing with overlap, distributed across devices.

Usage:
    >>> import os
    >>> os.environ["RANK"] = "0"
    >>> os.environ["WORLD_SIZE"] = "1"
    >>> from lite_boost import BoostManager
    >>> from lite_boost.parallel import initialize_usp
    >>> from wan.textimage2video import WanTI2V
    >>> initialize_usp()
    >>> pipe = WanTI2V(config=cfg, checkpoint_dir=ckpt_dir, ...)
    >>> boost_manager = BoostManager()
    >>> pipe = boost_manager(pipe)
    >>> # Optionally select per-module optimizations via a YAML config file:
    >>> pipe = boost_manager(pipe, config="boost.yaml")
"""


def _load_config(config):
    """Parse the YAML config file into a dict (``None`` passes through).

    The dict is forwarded to the detected model's boost function, which
    reads the sections it supports (e.g. ``Parallel.dit`` / ``Parallel.vae``
    for Qwen-Image-Edit) and ignores the rest.
    """
    if config is None:
        return None
    import yaml  # deferred so importing this module stays dependency-free

    with open(config, "r", encoding="utf-8") as f:
        parsed = yaml.safe_load(f)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config}")
    return parsed


class BoostManager:
    r"""
    Modify a supported model in-place for distributed parallel inference.

    :class:`BoostManager` wraps a supported model or pipeline and patches
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
        target (object, optional): A supported pipeline object to be
            parallelized. Supported classes include ``WanT2V`` and
            ``WanTI2V``. When omitted, a :class:`BoostManager` instance is
            returned for later use.
        config (str, optional): Path to a YAML file that lets the user
            select which module optimizations to enable, parsed into a
            dict and forwarded to the model's boost function.

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
        >>> from lite_boost import BoostManager
        >>> from lite_boost.parallel import initialize_usp
        >>> from wan.textimage2video import WanTI2V
        >>> initialize_usp()
        >>> pipe = WanTI2V(config=cfg, checkpoint_dir=ckpt_dir, ...)
        >>> boost_manager = BoostManager()
        >>> pipe = boost_manager(pipe)
        >>> # or with a per-module optimization config:
        >>> pipe = boost_manager(pipe, config="boost.yaml")
    """

    def __new__(cls, target=None, config=None):
        # One-step shortcut: BoostManager(pipe) patches and returns the model.
        if target is not None:
            from lite_boost.model import setup_model
            setup_model(target, config=_load_config(config))
            return target
        # Two-step usage: boost_manager = BoostManager(); pipe = boost_manager(pipe).
        return super().__new__(cls)

    def __call__(self, target, config=None):
        from lite_boost.model import setup_model
        setup_model(target, config=_load_config(config))
        return target
