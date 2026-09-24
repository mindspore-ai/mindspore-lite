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
- SP (Ulysses sequence parallel, ``usp_single_stream.py`` /
  ``usp_double_stream.py``): keep full
  weights, shard the padded sequence; attention exchanges heads/sequence
  via all_to_all. Optional per-region ``cc_overlap`` hides the SP
  collectives behind FFN / projection matmuls (``comm_compute_overlap.py``).
- ``bsnd`` (``attention_bsnd.py``): fused-attention layout optimization (BNSD
  transposes removed via ``npu_fusion_attention`` in BSND layout); composes
  with 1P / TP (processor patch) and with SP / cc_overlap (backend axis of
  the shared kernels in ``attention_common.py``).

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
from lite_boost.model.booguimage.attention_common import attention_bsnd
from lite_boost.model.booguimage.usp_single_stream import (
    _sp_attention_bsnd,
    boost_sp_single_stream,
    set_sp_attention,
)
from lite_boost.model.booguimage.usp_double_stream import boost_sp_double_stream
from lite_boost.model.booguimage.attention_bsnd import install_bsnd_processors

_PIPELINE_CLASS_NAMES = frozenset({"BooguImagePipeline", "BooguImageTurboPipeline"})

# Parallel algorithms selectable per transformer region (Parallel.dit.<region>).
_DIT_PARALLEL_ALGS = ('TP', 'SP')
# Regions that currently support SP; TP is available everywhere.
_SP_SUPPORTED_REGIONS = ('single', 'double')

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
    """Read the ``Parallel`` section of the boost config.

    Returns ``(algs, world_size, options)`` where ``algs`` maps each
    transformer region to its parallel algorithm and ``options`` holds the
    per-optimization flags:

    - ``options['cc_overlap'][region]`` — hide SP collectives behind
      independent matmuls (SP only, needs world_size > 1).
    - ``options['bsnd']`` — BSND fused-attention backend for the attention
      kernels (any world size; on 1P/TP it patches the processors, under SP
      / cc_overlap it binds the SP / comm-compute kernels to the fused
      backend — the double-stream processor stays owned by SP there).

    Missing keys fall back to TP / disabled. The yaml schema (see
    ``booguimage.yaml``) is::

        Parallel:
          dit:
            double:         # double-stream blocks: TP | SP
              alg: TP
              cc_overlap: false   # SP only: overlap a2a with FFN matmuls
            single:         # single-stream blocks
              alg: TP
              cc_overlap: false
            world_size: 2
          bsnd: true        # BSND fused attention (composes with TP/SP/CC)
    """
    dist_world_size = get_world_size()
    root = (config or {}).get('Parallel', {})
    dit = root.get('dit') or {}
    algs = {}
    cc_overlap = {}
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
        cc = bool(section.get('cc_overlap', False))
        if cc and alg != 'SP':
            raise ValueError(
                f"Parallel.dit.{region}.cc_overlap requires alg: SP "
                f"(it overlaps the SP all_to_all with FFN matmuls)"
            )
        if cc and region == 'double':
            logger.warning(
                "Parallel.dit.double.cc_overlap: double-stream SP assumes "
                "batch_size == 1 and fixed sequence layout per forward "
                "(see usp_double_stream.py); results are only correct under those "
                "assumptions."
            )
        algs[region] = alg
        cc_overlap[region] = cc

    bsnd = bool((config or {}).get('bsnd', False))

    world_size = dit.get('world_size') or dist_world_size
    if world_size != dist_world_size:
        raise ValueError(
            f"Parallel.dit.world_size ({world_size}) must match the "
            f"distributed world size ({dist_world_size})"
        )
    return algs, world_size, {'cc_overlap': cc_overlap, 'bsnd': bsnd}


def boost_booguimage(pipe, config=None):
    """Apply the configured parallel boost to a BooguImage pipeline in place.

    ``config`` is the dict parsed from the boost YAML (see
    ``booguimage.yaml``); ``None`` keeps the default TP everywhere at the
    distributed world size.
    """
    cls_name = pipe.__class__.__name__
    if cls_name not in _PIPELINE_CLASS_NAMES:
        raise ValueError(f"pipe class {cls_name} is not supported")

    algs, world_size, options = _parse_parallel_config(config)

    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        raise ValueError("transformer is None")

    patch_clean_boogu_for_npu(transformer)

    if world_size <= 1:
        if options['bsnd']:
            install_bsnd_processors()
        pipe._lite_boost_tp = False
        return pipe

    rank = get_rank()
    device = f"npu:{rank}" if torch.npu.is_available() else f"cuda:{rank}"
    transformer.to(device)
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.to(device)

    if algs['single'] == 'SP':
        if options['cc_overlap']['single']:
            from lite_boost.model.booguimage.comm_compute_overlap import sp_attention_overlap

            def cc_attention(attn, hidden_local, mask_sdpa, rope_local,
                        base_sequence_length=None, backend=None):
                """Comm-compute overlap kernel with the bsnd option folded in."""
                return sp_attention_overlap(
                    attn, hidden_local, mask_sdpa, rope_local,
                    base_sequence_length,
                    backend=attention_bsnd if options['bsnd'] else backend,
                )
            set_sp_attention(cc_attention)
        elif options['bsnd']:
            set_sp_attention(_sp_attention_bsnd)
        boost_sp_single_stream(transformer, world_size=world_size)
        if algs['double'] == 'SP':
            boost_sp_double_stream(
                transformer, world_size=world_size,
                cc_overlap=options['cc_overlap']['double'],
                bsnd=options['bsnd'],
            )
    else:
        shard_boogu_transformer(transformer, rank=rank, world_size=world_size)
        transformer.config.num_attention_heads //= world_size
        patch_transformer_forwards(transformer)

    # On SP the double-stream joint attention is owned by usp_double_stream, so bsnd
    # only patches the single-stream processor.
    if options['bsnd']:
        install_bsnd_processors(patch_double_stream=algs['double'] != 'SP')

    _patch_vae_decode(pipe)
    _patch_pipeline_call(pipe)

    pipe._lite_boost_tp = True
    pipe._lite_boost_tp_rank = rank
    pipe._lite_boost_tp_world_size = world_size
    return pipe
