# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Common DP tiling primitives — model-agnostic, reusable across VAE implementations.

Each DP rank processes a contiguous chunk of frames at full spatial
resolution.  Overlapping frames ensure causal convolution caches are
correctly initialized at chunk boundaries.  Results are gathered and
concatenated — no sorting needed (contiguous chunk assignment preserves
global order).

Sections:
  1. 1D Chunking Geometry  — :class:`Chunk`, :func:`compute_1d_chunks`
     Pure-math partition of a 1D domain into uniform overlapping chunks.
     Reusable for any 1D tiling (temporal, spatial, etc.).

  2. Output Length Helpers  — :func:`compute_compress_len`, :func:`compute_expand_len`
     Pure-math helpers for encoder (compress) and decoder (expand) output
     length calculation given a temporal stride.

  3. Blending  — :func:`blend_along_axis`
     Linear cross-fade along an axis.  Requires torch but no dist.
     Reusable for any overlapping tile blending.

  4. DP Distribution  — :func:`scatter_evenly`, :func:`gather_and_concat`
     Contiguous chunk assignment across ranks with all_gather collection.
     Requires torch.distributed.  Reusable for any DP tiling pipeline.

  5. DP Temporal Orchestrator  — :func:`dp_temporal_process`
     Full tiling loop: partition → scatter → process → gather → crop.
     Accepts a model-specific ``chunk_fn`` callable.  The only section
     that a new VAE adapter typically needs to call directly.

Usage::

    from lite_boost.parallel.data_parallel import dp_temporal_process

    def encode_chunk(chunk):
        with torch.amp.autocast("npu", dtype=vae.dtype):
            return vae.model.encode(chunk.unsqueeze(0), vae.scale).float().squeeze(0)

    result = dp_temporal_process(
        video, encode_chunk, t_dim=1,
        chunk_frames=12, overlap_frames=8, temporal_stride=4,
        world_size=ws, rank=rank, device=device,
    )
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
import torch.distributed as dist

# ============================================================================
# Section 1: 1D Chunking Geometry (pure math, no torch needed)
# ============================================================================


@dataclass
class Chunk:
    """A contiguous range [start, end) along a 1D axis."""
    global_idx: int
    start: int
    end: int
    is_padding: bool = False


def compute_1d_chunks(
    total: int, chunk_size: int, overlap: int
) -> Tuple[List[Chunk], int]:
    """Partition [0, padded_total) into uniform overlapping chunks.

    The input domain is padded so every chunk has exactly *chunk_size*
    elements and every interior boundary has exactly *overlap* shared
    elements.

    Args:
        total:      Actual number of elements to cover.
        chunk_size: Chunk size including overlap.
        overlap:    Overlap between adjacent chunks.

    Returns:
        (chunks, pad) where *chunks* are in global order and *pad* is
        the number of extra elements appended beyond *total* for alignment.
    """
    if total <= 0:
        return [], 0

    stride = chunk_size - overlap
    if stride <= 0:
        raise ValueError(
            f"chunk_size ({chunk_size}) must exceed overlap ({overlap})")

    n = max(1, math.ceil(max(0, total - overlap) / stride))
    padded_total = (n - 1) * stride + chunk_size
    pad = padded_total - total

    chunks = []
    for i in range(n):
        s = i * stride
        e = s + chunk_size
        chunks.append(Chunk(global_idx=i, start=s, end=e))
    return chunks, pad


# ============================================================================
# Section 2: Output Length Helpers (pure math)
# ============================================================================


def compute_compress_len(input_len: int, stride: int) -> int:
    """Encode direction: sample frames → latent frames.

    Used by VAE encoders.  Example: 17 rgb frames at stride 4 → 5 latent frames.
    """
    return (input_len - 1) // stride + 1


def compute_expand_len(input_len: int, stride: int) -> int:
    """Decode direction: latent frames → sample frames.

    Used by VAE decoders.  Example: 5 latent frames at stride 4 → 17 rgb frames.
    """
    return (input_len - 1) * stride + 1


# ============================================================================
# Section 3: Blending (torch tensor ops, no dist)
# ============================================================================


def blend_along_axis(
    a: torch.Tensor, b: torch.Tensor, axis: int, extent: int
) -> torch.Tensor:
    """Linear cross-fade from end of *a* to start of *b* along *axis*.

    Mutates *b* in-place and returns it.

    Args:
        a:      Preceding tile (overlap region taken from its end).
        b:      Following tile (overlap region at its start is blended).
        axis:   Dimension to blend along.
                -3 = temporal, -2 = vertical (H), -1 = horizontal (W).
        extent: Number of elements to blend (capped to the overlap available).

    Returns:
        *b* with the overlap region linearly cross-faded.
    """
    extent = min(a.shape[axis], b.shape[axis], extent)
    if extent <= 0:
        return b

    ndim = a.ndim
    for i in range(extent):
        w = 1.0 - i / extent

        sl_a = [slice(None)] * ndim
        sl_a[axis] = slice(a.shape[axis] - extent + i, a.shape[axis] - extent + i + 1)

        sl_b = [slice(None)] * ndim
        sl_b[axis] = slice(i, i + 1)

        b[tuple(sl_b)] = a[tuple(sl_a)] * w + b[tuple(sl_b)] * (1.0 - w)

    return b


# ============================================================================
# Section 4: DP Distribution (requires torch.distributed)
# ============================================================================


def scatter_evenly(
    items: List, world_size: int, rank: int
) -> Tuple[List, int, int]:
    """Distribute items across ranks in contiguous blocks, pad to uniform count.

    Returns (local_items, max_count, n_total) where:
      - *local_items* has exactly *max_count* elements (padded with
        ``is_padding=True`` chunks).
      - *n_total* is the original number of items.

    All three values are computable locally without any communication.
    """
    n_items = len(items)
    max_count = math.ceil(n_items / world_size)

    base = n_items // world_size
    rem = n_items % world_size
    if rank < rem:
        start = rank * (base + 1)
        actual = base + 1
    else:
        start = rem * (base + 1) + (rank - rem) * base
        actual = base

    local = list(items[start:start + actual])
    while len(local) < max_count:
        local.append(Chunk(global_idx=-1, start=0, end=0, is_padding=True))

    return local, max_count, n_items


def gather_and_concat(
    local_results: List[torch.Tensor],
    local_chunks: List[Chunk],
    n_total: int,
    overlap: int,
    concat_dim: int,
    device: torch.device,
    world_size: int,
) -> torch.Tensor:
    """Gather chunk results from all ranks, strip overlap, concatenate.

    Uses ``dist.all_gather`` (tensor-based, no pickling).  Results are
    naturally ordered by ``(rank, local_index)`` which equals global
    chunk order due to contiguous chunk assignment — no sorting needed.

    Args:
        local_results: Per-chunk tensors (padded to uniform count, uniform shape).
        local_chunks:  Chunk specs (padded, same length as *local_results*).
        n_total:       Total number of real chunks across all ranks.
        overlap:       Overlap frames to strip from start of each chunk
                       (except the very first chunk, which keeps all frames).
        concat_dim:    Axis to concatenate along (typically the T axis).
        device:        Device for metadata tensors.
        world_size:    Total number of DP ranks.

    Returns:
        Concatenated tensor with overlap frames removed.
    """
    num_local = len(local_results)

    # Gather results: [num_local, ...] → world_size × [num_local, ...]
    result_batch = torch.stack(local_results)
    gathered_results = [torch.empty_like(result_batch) for _ in range(world_size)]
    dist.all_gather(gathered_results, result_batch)

    # Gather chunk metadata: [num_local, 2] with (global_idx, is_padding)
    meta = torch.zeros(num_local, 2, dtype=torch.int64, device=device)
    for i, ch in enumerate(local_chunks):
        meta[i, 0] = ch.global_idx
        meta[i, 1] = 1 if ch.is_padding else 0
    gathered_meta = [torch.empty_like(meta) for _ in range(world_size)]
    dist.all_gather(gathered_meta, meta)

    # Deterministic per-rank real chunk count (contiguous assignment)
    base = n_total // world_size
    rem = n_total % world_size
    actual_counts = [base + 1 if r < rem else base for r in range(world_size)]

    # Collect interior slices in natural (rank, index) order —
    # this IS the global chunk order because chunks were assigned contiguously.
    pieces = []
    for r in range(world_size):
        for m in range(actual_counts[r]):
            gid = int(gathered_meta[r][m, 0].item())
            is_pad = int(gathered_meta[r][m, 1].item()) != 0
            if is_pad:
                continue

            # Only strip overlap_start: those frames have cold causal
            # cache in this chunk.  overlap_end (warm cache) is kept.
            ov_s = overlap if gid > 0 else 0

            t = gathered_results[r][m]
            pieces.append(t.narrow(concat_dim, ov_s,
                                   t.shape[concat_dim] - ov_s))

    return torch.cat(pieces, dim=concat_dim)


# ============================================================================
# Section 5: DP Temporal Orchestrator (model-agnostic)
# ============================================================================


def dp_temporal_process(
    tensor: torch.Tensor,
    chunk_fn: Callable[[torch.Tensor], torch.Tensor],
    t_dim: int,
    chunk_frames: int,
    overlap_frames: int,
    temporal_stride: int = 1,
    world_size: int = 1,
    rank: int = 0,
    device: torch.device = None,
) -> torch.Tensor:
    """Process a tensor with DP tiling along one axis.

    The input is padded so every chunk has exactly *chunk_frames*.  Overlap
    in output space and the target output length are derived from the first
    chunk processed via *chunk_fn*.

    When *world_size* is 1 the distribution steps are no-ops and all chunks
    are processed sequentially on the single rank.

    Args:
        tensor:          Input tensor.
        chunk_fn:        Model-specific ``f(chunk) → output_chunk``.
        t_dim:           Axis to tile along.
        chunk_frames:    Chunk size (in input units along t_dim).
        overlap_frames:  Overlap between chunks (in input units).
        temporal_stride: Temporal compression factor (``vae_stride[0]``).
                         Used to compute exact target output length.
                         Default 1 = no compression.
        world_size:      Number of DP ranks.
        rank:            Current DP rank.
        device:          Target device for processing.

    Returns:
        Processed tensor with overlap frames removed and cropped to exact
        target length.
    """
    n_frames = tensor.shape[t_dim]
    if n_frames <= chunk_frames:
        return chunk_fn(tensor.to(device))

    # 1. Partition (pad input to align with chunk stride)
    chunks, pad = compute_1d_chunks(n_frames, chunk_frames, overlap_frames)

    if pad > 0:
        pad_shape = list(tensor.shape)
        pad_shape[t_dim] = pad
        last = tensor.narrow(t_dim, n_frames - 1, 1)
        tensor = torch.cat([tensor, last.expand(pad_shape)], dim=t_dim)

    tensor = tensor.to(device)

    # 2. Scatter
    local_chunks, max_count, n_total = scatter_evenly(chunks, world_size, rank)

    # 3. Process local chunks — first real chunk determines output shape
    local_results: list = []
    output_shape = None
    out_per_chunk = None

    for ch in local_chunks:
        if ch.is_padding:
            if output_shape is not None:
                local_results.append(
                    torch.zeros(output_shape, dtype=torch.float32, device=device))
            continue

        slc = [slice(None)] * tensor.dim()
        slc[t_dim] = slice(ch.start, ch.end)
        out = chunk_fn(tensor[tuple(slc)])

        if output_shape is None:
            output_shape = tuple(out.shape)
            out_per_chunk = out.shape[t_dim]

        local_results.append(out)

    if output_shape is None:
        return chunk_fn(tensor)

    while len(local_results) < max_count:
        local_results.append(
            torch.zeros(output_shape, dtype=torch.float32, device=device))

    # 4. Derive overlap_out and target_len from first chunk
    overlap_out = overlap_frames * out_per_chunk // chunk_frames
    if out_per_chunk > chunk_frames:
        # Expansion (decode): T_latent → T_rgb
        target_len = compute_expand_len(n_frames, temporal_stride)
    else:
        # Compression (encode): T_rgb → T_latent
        target_len = compute_compress_len(n_frames, temporal_stride)

    # 5. Gather + strip overlap_start + concat
    result = gather_and_concat(
        local_results, local_chunks, n_total, overlap_out,
        t_dim, device, world_size,
    )

    # 6. Crop to exact target length
    if result.shape[t_dim] > target_len:
        result = result.narrow(t_dim, 0, target_len)

    return result
