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
BooguImage boost entry point.

``boost_booguimage(pipe, config)`` is called by ``model.setup_model`` with
the dict parsed from the user's YAML (see ``booguimage.yaml``). It applies
the NPU compatibility patches (device validation, RoPE frequency gathering,
rotary embedding, SwiGLU, SDPA masks — wiring the kernels from
``npu_kernels.py`` into the boogu package), moves the modules to the rank
device, and dispatches to the parallel algorithm selected per transformer
region:

- TP (tensor parallel, ``tp.py``): shard the transformer weights across
  ranks and patch attention/FFN forwards with all-reduce based kernels.
- SP (Ulysses sequence parallel, ``usp.py``): keep full weights, shard the
  padded sequence; attention exchanges heads/sequence via all_to_all.

Missing config sections keep the default ``TP`` at the distributed world
size.
"""

import logging
import re
import types

import torch

from lite_boost.parallel import get_rank, get_world_size

from lite_boost.model.booguimage.npu_kernels import (
    npu_apply_rotary_emb,
    npu_swiglu,
    patch_sdpa_mask,
    patch_swiglu_attr,
)
from lite_boost.model.booguimage.tp import (
    shard_boogu_transformer,
    patch_transformer_forwards,
    tp_vae_decode,
    tp_pipeline_call,
)
from lite_boost.model.booguimage.usp import boost_sp_single_stream

_PIPELINE_CLASS_NAMES = frozenset({"BooguImagePipeline", "BooguImageTurboPipeline"})

# Parallel algorithms selectable per transformer region (Parallel.dit.<region>).
_DIT_PARALLEL_ALGS = ('TP', 'SP')
# Regions that currently support SP; TP is available everywhere.
_SP_SUPPORTED_REGIONS = ('single',)

_NPU_DEVICE_REGEX = r"^(cpu|cuda|cuda:\d+|npu|npu:\d+)$"

logger = logging.getLogger(__name__)

_BOOGU_INSTALL_HINT = (
    "The 'boogu' package is required for the BooguImage NPU compatibility "
    "patches. Install Boogu-Image (branch 'npu') from "
    "https://github.com/boogu-project/Boogu-Image; NPU patches are skipped "
    "until it is available."
)


# ---------------------------------------------------------------------------
# NPU compatibility wiring (boogu package patches)
# ---------------------------------------------------------------------------

def _npu_get_device_validator(additional_types=None):
    """Device validator that accepts ``npu`` / ``npu:N`` in addition to the
    CUDA-style strings Boogu-Image originally allows."""
    if additional_types is None:
        additional_types = []

    def validate(value):
        if not value:
            return None
        value = value.lower() if isinstance(value, str) else value
        if not isinstance(value, str):
            return value
        if re.match(_NPU_DEVICE_REGEX, value):
            return value
        if value in additional_types:
            return value
        return None
    return validate


def patch_device_validator():
    """Patch Boogu-Image's device validator to accept ``npu`` devices (idempotent)."""
    try:
        import boogu.utils.validator_utils as vu
    except ImportError:
        logger.warning(_BOOGU_INSTALL_HINT)
        return
    if getattr(vu, "_lb_npu_patched", False):
        return

    vu.get_device_validator = _npu_get_device_validator
    vu._lb_npu_patched = True

    try:
        import boogu.pipelines.boogu.pipeline_boogu as pb
        if not getattr(pb, "_lb_npu_validator_patched", False):
            pb.get_device_validator = _npu_get_device_validator
            pb._lb_npu_validator_patched = True
    except ImportError:
        logger.warning(_BOOGU_INSTALL_HINT)


def _npu_safe_gather_freqs(self, freqs_cis, ids):
    """NPU-safe replacement for RoPE ``_get_freqs_cis`` (avoids complex gather on NPU)."""
    device = ids.device
    if ids.device.type == "mps":
        ids = ids.to("cpu")

    result = []
    for i in range(len(self.axes_dim)):
        freqs = freqs_cis[i].to(ids.device)
        index = ids[:, :, i:i + 1].repeat(1, 1, freqs.shape[-1]).contiguous().to(torch.int64)
        freqs_expanded = freqs.unsqueeze(0).repeat(index.shape[0], 1, 1).contiguous()
        if freqs_expanded.is_complex():
            gathered_real = torch.gather(freqs_expanded.real.to(torch.float32), dim=1, index=index)
            gathered_imag = torch.gather(freqs_expanded.imag.to(torch.float32), dim=1, index=index)
            result.append(torch.complex(gathered_real, gathered_imag).to(torch.complex64))
        else:
            result.append(torch.gather(freqs_expanded, dim=1, index=index))
    return torch.cat(result, dim=-1).contiguous().to(device)


def patch_rope_gather():
    """Patch RoPE classes to use NPU-safe frequency gathering (idempotent)."""
    try:
        from boogu.models.transformers import rope as rope_mod
    except ImportError:
        return

    for cls_name in ("BooguImageRotaryPosEmbed", "BooguImageDoubleStreamRotaryPosEmbed"):
        cls = getattr(rope_mod, cls_name, None)
        if cls is None:
            continue
        if getattr(cls, "_lb_npu_patched", False):
            continue
        cls._get_freqs_cis = _npu_safe_gather_freqs
        cls._lb_npu_patched = True


def patch_rotary_emb():
    """Patch boogu's ``apply_rotary_emb`` with the NPU-safe variant
    (complex64 preferred, fused fp32 fallback; idempotent). Patching the
    ``boogu.models.embeddings`` definition covers all importers that did
    ``from .embeddings import apply_rotary_emb`` only if they import the
    module lazily; the direct importers are patched too."""
    import boogu.models.embeddings as emb
    if not getattr(emb, "_lb_npu_patched", False):
        emb.apply_rotary_emb = npu_apply_rotary_emb
        emb._lb_npu_patched = True

    try:
        import boogu.models.attention_processor as ap
        if not getattr(ap, "_lb_npu_patched", False):
            ap.apply_rotary_emb = npu_apply_rotary_emb
            ap._lb_npu_patched = True
    except ImportError:
        pass


def patch_swiglu():
    """Patch boogu's SwiGLU with the fused NPU variant (idempotent).

    ``LuminaFeedForward.forward`` calls ``self.swiglu`` bound at construction
    (a module-level function reference), so each FFN instance's attribute is
    swapped individually; the module-level functions are patched too for any
    code that calls them directly.
    """
    try:
        import boogu.models.transformers.block_lumina2 as bl
        import boogu.models.transformers.components as comp
    except ImportError:
        return
    patch_swiglu_attr(bl)
    patch_swiglu_attr(comp)


def patch_swiglu_instances(transformer):
    """Swap every ``LuminaFeedForward.swiglu`` instance attribute for the fused
    NPU variant. Needed because the attribute is bound at __init__ time, so
    patching the module afterwards misses already-constructed instances."""
    for _, module in transformer.named_modules():
        if hasattr(module, "swiglu") and callable(module.swiglu):
            module.swiglu = npu_swiglu


def patch_clean_boogu_for_npu(transformer=None):
    """Apply all Boogu-Image NPU compatibility patches in order."""
    patch_device_validator()
    patch_rope_gather()
    patch_rotary_emb()
    patch_swiglu()
    patch_sdpa_mask()
    if transformer is not None:
        patch_swiglu_instances(transformer)


# ---------------------------------------------------------------------------
# Pipeline-level TP patches
# ---------------------------------------------------------------------------

def _patch_vae_decode(pipe):
    """Patch the VAE ``decode`` so only rank 0 runs it and others return zeros."""
    vae = getattr(pipe, "vae", None)

    if vae is None or getattr(vae, "_lb_patched", False):
        return
    if not hasattr(vae, "_lb_original_decode"):
        vae._lb_original_decode = vae.decode
    vae._lb_vae_scale_factor = getattr(pipe, "vae_scale_factor", 8)
    vae.decode = types.MethodType(tp_vae_decode, vae)
    vae._lb_patched = True


def _patch_pipeline_call(pipe):
    """Patch the pipeline ``__call__`` with the TP-aware variant (idempotent)."""
    cls = type(pipe)
    if getattr(cls, "_lb_tp_call_patched", False):
        return
    cls._lb_original_call = cls.__call__
    cls.__call__ = tp_pipeline_call
    cls._lb_tp_call_patched = True


# ---------------------------------------------------------------------------
# Config parsing and dispatch
# ---------------------------------------------------------------------------

def _parse_parallel_config(config):
    """Read the ``Parallel.dit`` section of the boost config.

    Returns ``(algs, world_size)`` where ``algs`` maps each transformer
    region to its parallel algorithm. Missing keys fall back to ``TP``;
    world size defaults to the distributed world size. The yaml schema (see
    ``booguimage.yaml``) is::

        Parallel:
          dit:
            double:         # double-stream blocks: TP | SP (SP not yet supported)
              alg: TP
            single:         # single-stream blocks
              alg: TP
            world_size: 2
    """
    dist_world_size = get_world_size()
    dit = (config or {}).get('Parallel', {}).get('dit') or {}
    algs = {}
    for region in ('double', 'single'):
        section = dit.get(region) or {}
        if not isinstance(section, dict):
            raise ValueError(
                f"Parallel.dit.{region} must be a mapping with an 'alg' key "
                f"(e.g. {region}: {{alg: TP}}), got {section!r}"
            )
        alg = section.get('alg', 'TP')
        if alg not in _DIT_PARALLEL_ALGS:
            raise ValueError(
                f"Parallel.dit.{region}.alg {alg!r} is unsupported; expected "
                f"one of {_DIT_PARALLEL_ALGS}"
            )
        if alg == 'SP' and region not in _SP_SUPPORTED_REGIONS:
            raise ValueError(
                f"Parallel.dit.{region}.alg: SP is not supported for the "
                f"{region}-stream blocks yet; expected one of "
                f"{_SP_SUPPORTED_REGIONS}"
            )
        algs[region] = alg
    world_size = dit.get('world_size') or dist_world_size
    if world_size != dist_world_size:
        raise ValueError(
            f"Parallel.dit.world_size ({world_size}) must match the "
            f"distributed world size ({dist_world_size})"
        )
    return algs, world_size


def boost_booguimage(pipe, config=None):
    """Apply the configured parallel boost to a BooguImage pipeline in place.

    ``config`` is the dict parsed from the boost YAML (see
    ``booguimage.yaml``); ``None`` keeps the default TP everywhere at the
    distributed world size.
    """
    cls_name = pipe.__class__.__name__
    if cls_name not in _PIPELINE_CLASS_NAMES:
        raise ValueError(f"pipe class {cls_name} is not supported")

    algs, world_size = _parse_parallel_config(config)

    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        raise ValueError("transformer is None")

    patch_clean_boogu_for_npu(transformer)

    if world_size <= 1:
        pipe._lite_boost_tp = False
        return pipe

    rank = get_rank()
    device = f"npu:{rank}" if torch.npu.is_available() else f"cuda:{rank}"
    transformer.to(device)
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.to(device)

    if algs['single'] == 'SP':
        boost_sp_single_stream(transformer, world_size=world_size)
    else:
        shard_boogu_transformer(transformer, rank=rank, world_size=world_size)
        transformer.config.num_attention_heads //= world_size
        patch_transformer_forwards(transformer)

    _patch_vae_decode(pipe)
    _patch_pipeline_call(pipe)

    pipe._lite_boost_tp = True
    pipe._lite_boost_tp_rank = rank
    pipe._lite_boost_tp_world_size = world_size
    return pipe
