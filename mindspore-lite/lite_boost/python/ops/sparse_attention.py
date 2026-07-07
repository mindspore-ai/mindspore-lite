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
"""
Sparse flash attention (rf_v2).

Implements block-sparse attention with token rearrangement, block-wise pooling,
top-k sparse mask generation, and inverse rearrangement for video generation.
"""

import math
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# avgpool — block-wise mean pooling with aligned fast path
# ---------------------------------------------------------------------------
def avgpool(input_tensor, pool_size=128, input_layout='BNSD'):
    """Average pool input_tensor in blocks of pool_size along the sequence dim.

    Args:
        input_tensor: BSND [B,S,N,D] or BNSD [B,N,S,D]
        pool_size:    block size (default 128)
        input_layout: 'BSND' or 'BNSD'

    Returns:
        Pooled tensor with same layout, sequence dim reduced by factor pool_size.
    """
    if input_layout == "BSND":
        batch, seqlen, headnum, dim = input_tensor.shape
        num_full_blocks = seqlen // pool_size
        tail_size = seqlen % pool_size

        # Fast path: aligned sequence (no tail)
        if tail_size == 0:
            return input_tensor.view(
                batch, num_full_blocks, pool_size, headnum, dim
            ).mean(dim=2)

        # Slow path: handle tail
        full_blocks = input_tensor[:, :num_full_blocks * pool_size, :, :]
        full_pooled = full_blocks.view(
            batch, num_full_blocks, pool_size, headnum, dim
        ).mean(dim=2)

        tail_block = input_tensor[:, num_full_blocks * pool_size:, :, :]
        tail_pooled = tail_block.view(
            batch, 1, tail_size, headnum, dim
        ).mean(dim=2)

        return torch.cat([full_pooled, tail_pooled], dim=1)

    batch, headnum, seqlen, dim = input_tensor.shape
    num_full_blocks = seqlen // pool_size
    tail_size = seqlen % pool_size

    # Fast path: aligned sequence
    if tail_size == 0:
        return input_tensor.view(
            batch, headnum, num_full_blocks, pool_size, dim
        ).mean(dim=3)

    # Slow path: handle tail
    full_blocks = input_tensor[:, :, :num_full_blocks * pool_size, :]
    full_pooled = full_blocks.view(
        batch, headnum, num_full_blocks, pool_size, dim
    ).mean(dim=3)

    tail_block = input_tensor[:, :, num_full_blocks * pool_size:, :]
    tail_pooled = tail_block.view(
        batch, headnum, 1, tail_size, dim
    ).mean(dim=3)

    return torch.cat([full_pooled, tail_pooled], dim=2)


# ---------------------------------------------------------------------------
# get_mask_index — convert binary mask to select_idx format for rain_fusion op
# ---------------------------------------------------------------------------
def get_mask_index(mask):
    """Convert boolean block-sparse mask [B,N,S,S] to index matrix [B,N,S,S].

    For each row, true positions are sorted to the front; remaining filled with -1.
    """
    b, n, s, _ = mask.shape
    device = mask.device

    mask_reshaped = mask.reshape(-1, s, s)
    batch_size = mask_reshaped.shape[0]

    # Row indices [batch_size, s, s]
    row_indices = torch.arange(s, device=device).expand(batch_size, s, -1)

    # Replace False with large sentinel, sort to push true indices to front
    sorted_vals = torch.where(mask_reshaped, row_indices, 1e9).to(torch.float32)
    sorted_vals, _ = torch.sort(sorted_vals, dim=-1)

    # Mask out excess positions
    valid_count = mask_reshaped.sum(dim=-1, keepdim=True)
    keep_mask = row_indices < valid_count
    result = torch.where(keep_mask, sorted_vals, -1)

    return result.reshape(b, n, s, s).to(torch.int64)


# ---------------------------------------------------------------------------
# get_blockwise_mask — generate block-sparse mask from pooled QKV
# ---------------------------------------------------------------------------
def get_blockwise_mask(  # pylint: disable=unused-argument
    qkv_pool,
    txt_len, sparsity, scale, pool_size,
    latent_shape_q, latent_shape_k=None,
    input_layout=None, return_binary=False,
    protect_first_frame=True):  # pylint: disable=too-many-locals
    """Generate block-sparse attention mask via top-k thresholding on softmax scores.

    Args:
        qkv_pool:     concatenated pooled Q,K,V along dim=0
        txt_len:      number of text prefix tokens
        sparsity:     fraction of KV blocks to prune [0, 1)
        scale:        attention scale factor
        pool_size:    block size used for pooling
        latent_shape_q: (t, h, w) for query
        input_layout: 'BSND' or 'BNSD'
        return_binary: if True, return int8 mask; else return select_idx/select_num_idx
        protect_first_frame: whether to protect first frame blocks

    Returns:
        If return_binary: int8 mask [B, N, q_blocks, kv_blocks]
        Else: (select_idx, select_num_idx) for rain_fusion_attention op
    """
    _, hq, wq = latent_shape_q
    first_frame_len = hq * wq

    query_pool, key_pool, _ = torch.chunk(qkv_pool, 3, dim=0)

    # Compute attention scores on pooled tokens
    if input_layout == "BSND":
        attn_scores_head = torch.einsum("blnd,bsnd->bnls", query_pool, key_pool) * scale
    else:
        attn_scores_head = torch.einsum("bnld,bnsd->bnls", query_pool, key_pool) * scale

    score_matrix = torch.nn.functional.softmax(attn_scores_head, dim=-1)
    cols = score_matrix.shape[-1]

    # Top-k thresholding for sparsity
    keep_len = math.ceil(cols * (1 - sparsity))
    topk_values, _ = torch.topk(score_matrix, k=keep_len, dim=-1)
    thresholds = topk_values[..., -1:]
    mask = score_matrix >= thresholds

    # Protect text blocks (bidirectional attention with text)
    text_block_num = (txt_len + pool_size - 1) // pool_size
    if text_block_num > 0:
        mask[:, :, -text_block_num:, :] = True
        mask[:, :, :, -text_block_num:] = True

    # Protect first frame blocks
    if protect_first_frame:
        firstframe_block_num = (first_frame_len + pool_size - 1) // pool_size
        if firstframe_block_num > 0:
            mask[:, :, :firstframe_block_num, :] = True
            mask[:, :, :, :firstframe_block_num] = True

    if return_binary:
        return mask.to(torch.int8)

    select_idx = get_mask_index(mask)
    select_idx = select_idx[0].transpose(0, 1)
    select_num_idx = mask[0].transpose(0, 1).sum(dim=-1)
    return select_idx, select_num_idx


# ---------------------------------------------------------------------------
# rearrange_with_remaining — spatial token reordering (native torch, no einops)
# ---------------------------------------------------------------------------
def rearrange_with_remaining(tensor, latent_shape_q, latent_shape_k=None, input_layout=None):  # pylint: disable=unused-argument
    """Rearrange tokens from (frame, h, w) order to block-interleaved (hn, wn, hb, wb).

    Transforms:
        BSND: b (f h w) n d  ->  b (f hn wn hb wb) n d
        BNSD: b n (f h w) d  ->  b n (f hn wn hb wb) d
    where hb=wb=8, hn=h//8, wn=w//8.
    """
    tq, hq, wq = latent_shape_q
    first_frame_len = hq * wq
    frame_num = tq
    hn, wn = hq // 8, wq // 8
    aligned = (hq % 8 == 0) and (wq % 8 == 0)

    if input_layout == "BSND":
        b, _, n, d = tensor.shape

        if aligned:
            # Fast path: single reshape+permute+reshape
            return (tensor
                    .reshape(b, frame_num, hn, 8, wn, 8, n, d)
                    .permute(0, 1, 2, 4, 3, 5, 6, 7)
                    .contiguous()
                    .reshape(b, frame_num * hn * wn * 64, n, d))

        # Remainder path: split first frame from rest
        tensor_first = tensor[:, :first_frame_len, :, :]
        tensor_rest = tensor[:, first_frame_len:, :, :]
        f_rest = frame_num - 1

        # Reshape to (b, f, h, w, n, d)
        tensor_hwt = tensor_rest.reshape(b, f_rest, hq, wq, n, d)

        # Split h dimension at 8-aligned boundary
        hq_block = (hq // 8) * 8
        hq_rem = hq % 8
        if hq_rem != 0:
            t_block = tensor_hwt[:, :, :hq_block, :, :, :]
            t_h_r = tensor_hwt[:, :, hq_block:, :, :, :].reshape(b, f_rest, -1, n, d)
        else:
            t_block = tensor_hwt
            t_h_r = None

        # Split w dimension at 8-aligned boundary
        wq_block = (wq // 8) * 8
        wq_rem = wq % 8
        if wq_rem != 0:
            t_main = t_block[:, :, :, :wq_block, :, :]
            t_w_r = t_block[:, :, :, wq_block:, :, :].reshape(b, f_rest, -1, n, d)
        else:
            t_main = t_block
            t_w_r = None

        # Block-rearrange the aligned portion
        t_main = (t_main
                  .reshape(b, f_rest, hn, 8, wn, 8, n, d)
                  .permute(0, 1, 2, 4, 3, 5, 6, 7)
                  .contiguous()
                  .reshape(b, f_rest, -1, n, d))

        # Concatenate remainder blocks
        if t_h_r is not None:
            t_main = torch.cat([t_main, t_h_r], dim=2)
        if t_w_r is not None:
            t_main = torch.cat([t_main, t_w_r], dim=2)

        t_main = t_main.reshape(b, -1, n, d)
        return torch.cat([tensor_first, t_main], dim=1)

    b, n, _, d = tensor.shape

    if aligned:
        # Fast path: single reshape+permute+reshape
        return (tensor
                .reshape(b, n, frame_num, hn, 8, wn, 8, d)
                .permute(0, 1, 2, 3, 5, 4, 6, 7)
                .contiguous()
                .reshape(b, n, frame_num * hn * wn * 64, d))

    # Remainder path
    tensor_first = tensor[:, :, :first_frame_len, :]
    tensor_rest = tensor[:, :, first_frame_len:, :]
    f_rest = frame_num - 1

    tensor_hwt = tensor_rest.reshape(b, n, f_rest, hq, wq, d)

    hq_block = (hq // 8) * 8
    hq_rem = hq % 8
    if hq_rem != 0:
        t_block = tensor_hwt[:, :, :, :hq_block, :, :]
        t_h_r = tensor_hwt[:, :, :, hq_block:, :, :].reshape(b, n, f_rest, -1, d)
    else:
        t_block = tensor_hwt
        t_h_r = None

    wq_block = (wq // 8) * 8
    wq_rem = wq % 8
    if wq_rem != 0:
        t_main = t_block[:, :, :, :, :wq_block, :]
        t_w_r = t_block[:, :, :, :, wq_block:, :].reshape(b, n, f_rest, -1, d)
    else:
        t_main = t_block
        t_w_r = None

    t_main = (t_main
              .reshape(b, n, f_rest, hn, 8, wn, 8, d)
              .permute(0, 1, 2, 3, 5, 4, 6, 7)
              .contiguous()
              .reshape(b, n, f_rest, -1, d))

    if t_h_r is not None:
        t_main = torch.cat([t_main, t_h_r], dim=3)
    if t_w_r is not None:
        t_main = torch.cat([t_main, t_w_r], dim=3)

    t_main = t_main.reshape(b, n, -1, d)
    return torch.cat([tensor_first, t_main], dim=2)


# ---------------------------------------------------------------------------
# inv_rearrange_with_remaining — inverse spatial token reordering (v3-style)
# ---------------------------------------------------------------------------
def inv_rearrange_with_remaining(tensor, latent_shape_q, latent_shape_k=None, input_layout=None):  # pylint: disable=unused-argument
    """Inverse of rearrange_with_remaining: block-interleaved -> (frame, h, w) order.

    Transforms:
        BSND: b (f hn wn hb wb) n d  ->  b (f h w) n d
        BNSD: b n (f hn wn hb wb) d  ->  b n (f h w) d
    where hb=wb=8, hn=h//8, wn=w//8.
    """
    tq, hq, wq = latent_shape_q
    first_frame_len = hq * wq
    frame_num = tq
    hn, wn = hq // 8, wq // 8
    aligned = (hq % 8 == 0) and (wq % 8 == 0)

    if input_layout == "BSND":
        b, _, n, d = tensor.shape

        if aligned:
            # Fast path: reshape -> permute -> reshape
            return (tensor
                    .reshape(b, frame_num, hn, wn, 8, 8, n, d)
                    .permute(0, 1, 2, 4, 3, 5, 6, 7)
                    .contiguous()
                    .reshape(b, frame_num * hq * wq, n, d))

        # Remainder path
        tensor_first = tensor[:, :first_frame_len, :, :]
        tensor_rest = tensor[:, first_frame_len:, :, :]
        f_rest = frame_num - 1

        tensor_rest = tensor_rest.reshape(b, f_rest, hq * wq, n, d)

        # Explicit slicing by known block sizes (avoids torch.split)
        hq_block = (hq // 8) * 8
        wq_block = (wq // 8) * 8
        hq_rem = hq % 8
        wq_rem = wq % 8
        block_size = hn * wn * 64
        h_rem_size = hq_rem * wq
        w_rem_size = hq_block * wq_rem

        t_block = tensor_rest[:, :, :block_size, :, :]
        t_h_r = tensor_rest[:, :, block_size:block_size + h_rem_size, :, :] if hq_rem > 0 else None
        t_w_r = (tensor_rest[:, :, block_size + h_rem_size:block_size + h_rem_size + w_rem_size, :, :]
                 if wq_rem > 0 else None)

        # Un-block-rearrange the aligned portion
        t_block = (t_block
                   .reshape(b, f_rest, hn, wn, 8, 8, n, d)
                   .permute(0, 1, 2, 4, 3, 5, 6, 7)
                   .contiguous()
                   .reshape(b, f_rest, hq_block, wq_block, n, d))

        if wq_rem > 0:
            t_block = torch.cat([t_block, t_w_r.reshape(b, f_rest, hq_block, wq_rem, n, d)], dim=3)
        if hq_rem > 0:
            t_block = torch.cat([t_block, t_h_r.reshape(b, f_rest, hq_rem, wq, n, d)], dim=2)

        tensor_rest = t_block.reshape(b, -1, n, d)
        return torch.cat([tensor_first, tensor_rest], dim=1)

    b, n, _, d = tensor.shape

    if aligned:
        # Fast path
        return (tensor
                .reshape(b, n, frame_num, hn, wn, 8, 8, d)
                .permute(0, 1, 2, 3, 5, 4, 6, 7)
                .contiguous()
                .reshape(b, n, frame_num * hq * wq, d))

    # Remainder path
    tensor_first = tensor[:, :, :first_frame_len, :]
    tensor_rest = tensor[:, :, first_frame_len:, :]
    f_rest = frame_num - 1

    tensor_rest = tensor_rest.reshape(b, n, f_rest, hq * wq, d)

    hq_block = (hq // 8) * 8
    wq_block = (wq // 8) * 8
    hq_rem = hq % 8
    wq_rem = wq % 8
    block_size = hn * wn * 64
    h_rem_size = hq_rem * wq
    w_rem_size = hq_block * wq_rem

    t_block = tensor_rest[:, :, :, :block_size, :]
    t_h_r = tensor_rest[:, :, :, block_size:block_size + h_rem_size, :] if hq_rem > 0 else None
    t_w_r = (tensor_rest[:, :, :, block_size + h_rem_size:block_size + h_rem_size + w_rem_size, :]
             if wq_rem > 0 else None)

    t_block = (t_block
               .reshape(b, n, f_rest, hn, wn, 8, 8, d)
               .permute(0, 1, 2, 3, 5, 4, 6, 7)
               .contiguous()
               .reshape(b, n, f_rest, hq_block, wq_block, d))

    if wq_rem > 0:
        t_block = torch.cat([t_block, t_w_r.reshape(b, n, f_rest, hq_block, wq_rem, d)], dim=4)
    if hq_rem > 0:
        t_block = torch.cat([t_block, t_h_r.reshape(b, n, f_rest, hq_rem, wq, d)], dim=3)

    tensor_rest = t_block.reshape(b, n, -1, d)
    return torch.cat([tensor_first, tensor_rest], dim=2)


# ---------------------------------------------------------------------------
# do_tensor_rearrange_pooling — rearrange + pool QKV, returning pooled for mask gen
# ---------------------------------------------------------------------------
def do_tensor_rearrange_pooling(query, key, value, text_len, pool_size,
                                latent_shape_q, latent_shape_k, input_layout):
    """Rearrange Q/K/V tokens and compute block-wise pooled representations.

    When text_len > 0, text tokens are separated before rearrange and re-attached
    afterwards.  When text_len == 0, Q/K/V are processed independently to avoid
    the cat+chunk roundtrip.

    Returns:
        query_, key_, value_: rearranged Q/K/V
        tensor_pool:          pooled QKV for mask generation
    """
    if text_len != 0:
        tensor = torch.cat((query, key, value), dim=0)
        if input_layout == "BSND":
            tensor_t = tensor[:, :text_len, :, :]
            tensor_i = tensor[:, text_len:, :, :]
        else:
            tensor_t = tensor[:, :, :text_len, :]
            tensor_i = tensor[:, :, text_len:, :]

        tensor_i_2 = rearrange_with_remaining(tensor_i, latent_shape_q, latent_shape_k, input_layout)
        tensor_i_pool = avgpool(tensor_i_2, pool_size, input_layout)
        tensor_t_pool = avgpool(tensor_t, pool_size, input_layout)

        if input_layout == "BSND":
            tensor = torch.cat((tensor_i_2, tensor_t), dim=1)
            tensor_pool = torch.cat((tensor_i_pool, tensor_t_pool), dim=1)
        else:
            tensor = torch.cat((tensor_i_2, tensor_t), dim=2)
            tensor_pool = torch.cat((tensor_i_pool, tensor_t_pool), dim=2)

        query_, key_, value_ = torch.chunk(tensor, 3, dim=0)
        return query_, key_, value_, tensor_pool

    # No text: avoid cat+chunk roundtrip — process Q/K/V independently
    q2 = rearrange_with_remaining(query, latent_shape_q, latent_shape_k, input_layout)
    k2 = rearrange_with_remaining(key, latent_shape_q, latent_shape_k, input_layout)
    v2 = rearrange_with_remaining(value, latent_shape_q, latent_shape_k, input_layout)

    q_pool = avgpool(q2, pool_size, input_layout)
    k_pool = avgpool(k2, pool_size, input_layout)
    v_pool = avgpool(v2, pool_size, input_layout)
    tensor_pool = torch.cat([q_pool, k_pool, v_pool], dim=0)

    return q2, k2, v2, tensor_pool


# ---------------------------------------------------------------------------
# do_tensor_inv_rearrange — inverse rearrange for attention output
# ---------------------------------------------------------------------------
def do_tensor_inv_rearrange(tensor, text_len, latent_shape_q, latent_shape_k, input_layout):
    """Inverse rearrange: restore original spatial token order from block-interleaved.

    When text_len > 0, text tokens (at the end) are separated and re-attached
    without spatial rearrangement.
    """
    if text_len != 0:
        if input_layout == "BSND":
            tensor_t = tensor[:, -text_len:, :, :]
            tensor_i = tensor[:, :-text_len, :, :]
            tensor_i = inv_rearrange_with_remaining(tensor_i, latent_shape_q, latent_shape_k, input_layout)
            return torch.cat((tensor_t, tensor_i), dim=1)
        # BNSD
        tensor_t = tensor[:, :, -text_len:, :]
        tensor_i = tensor[:, :, :-text_len, :]
        tensor_i = inv_rearrange_with_remaining(tensor_i, latent_shape_q, latent_shape_k, input_layout)
        return torch.cat((tensor_t, tensor_i), dim=2)

    return inv_rearrange_with_remaining(tensor, latent_shape_q, latent_shape_k, input_layout)


# ---------------------------------------------------------------------------
# do_tensor_pooling — pool-only (no rearrange), for legacy use
# ---------------------------------------------------------------------------
def do_tensor_pooling(tensor, text_len):
    """Pool image and text token regions separately and concatenate."""
    tensor_t = tensor[:, :text_len, :, :]
    tensor_i = tensor[:, text_len:, :, :]
    tensor_i_pool = avgpool(tensor_i, pool_size=128)
    tensor_t_pool = avgpool(tensor_t, pool_size=128)
    return torch.cat((tensor_t_pool, tensor_i_pool), dim=1)


# ---------------------------------------------------------------------------
# check_params
# ---------------------------------------------------------------------------
MAX_TOKEN = 2147483647


def check_params(input_layout, sparse_type):
    """Validate input_layout and sparse_type parameters."""
    if input_layout not in ['BSND', 'BNSD']:
        raise ValueError(f"The input_layout must be in ['BSND', 'BNSD'], but got {input_layout}.")
    if sparse_type not in [None, 'rf_v2']:
        raise ValueError(f"sparse_type must be None or 'rf_v2', but got {sparse_type}.")


# ---------------------------------------------------------------------------
# sparse_attention — main entry point
# ---------------------------------------------------------------------------
def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    is_causal: Optional[bool] = False,
    head_num: int = 1,
    input_layout: str = "BNSD",
    inner_precise: int = 0,
    sparse_type: Optional[str] = None,
    txt_len: int = 0,
    block_size: int = 128,
    latent_shape_q: Optional[list[int]] = None,
    latent_shape_k: Optional[list[int]] = None,
    keep_sink: bool = True,
    keep_recent: bool = True,
    cdf_threshold: float = 1.0,
    sparsity: float = 0.0,
    **_kwargs
):  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements,too-many-branches,unused-argument
    r"""
    High-level sparse attention entry point.

    Supports two modes:

    - **dense** (``sparse_type=None``): Calls ``torch_npu.npu_fusion_attention`` directly.
    - **block-sparse** (``sparse_type="rf_v2"``): Token rearrangement → block-wise pooling →
      top-k sparse mask generation → ``rain_fusion_attention`` → inverse rearrangement.

    Input tensors support ``float16`` and ``bfloat16`` dtypes.

    Args:
        q (Tensor): Query tensor.
        k (Tensor): Key tensor.
        v (Tensor): Value tensor.
        attn_mask (Tensor, optional): Reserved (unused in current rf_v2 path). Default: ``None``.
        scale (float, optional): Attention scale factor. ``None`` auto-sets to ``head_dim ** -0.5``.
            Default: ``None``.
        is_causal (bool, optional): Whether to apply causal mask. Effective in dense mode
            (``sparse_type=None``); unsupported in rf_v2 sparse path. Default: ``False``.
        head_num (int, optional): Number of attention heads. Default: ``1``.
        input_layout (str, optional): Input layout, ``"BSND"`` or ``"BNSD"``. Default: ``"BNSD"``.
        inner_precise (int, optional): Precision mode. ``0`` for high-precision, ``1`` for
            high-performance. Default: ``0``.
        sparse_type (str, optional): Sparse type. ``None`` for dense, ``"rf_v2"`` for block-sparse.
            ``"rf_v3"`` and ``"ada_bsa"`` are reserved for future support.
            Default: ``None``.
        txt_len (int, optional): Number of text prefix tokens. When ``>0``, text tokens are
            separated and not rearranged spatially (rf_v2 only). Default: ``0``.
        block_size (int, optional): Block size for pooling and attention (rf_v2 only).
            Default: ``128``.
        latent_shape_q (list[int], optional): Latent spatial grid ``(t, h, w)`` for query.
            Required for rf_v2 path. For example, ``(1, 64, 64)`` for a single 64×64 frame
            with 4096 tokens. Default: ``None``.
        latent_shape_k (list[int], optional): Latent spatial grid ``(t, h, w)`` for key/value.
            Reuses ``latent_shape_q`` when not provided (rf_v2 only). Default: ``None``.
        keep_sink (bool, optional): Whether to keep sink tokens. Reserved for ``ada_bsa`` mode
            (unused in rf_v2 path). Default: ``True``.
        keep_recent (bool, optional): Whether to keep recent tokens. Reserved for ``ada_bsa`` mode
            (unused in rf_v2 path). Default: ``True``.
        cdf_threshold (float, optional): CDF threshold. Reserved for ``ada_bsa`` mode
            (unused in rf_v2 path). Default: ``1.0``.
        sparsity (float, optional): Sparsity ratio in ``[0, 1]``. ``0.0`` for full attention,
            ``0.5`` prunes 50% of KV blocks (rf_v2 only). Default: ``0.0``.
        **_kwargs: Additional keyword arguments (reserved for future use).

    Returns:
        Tensor, Attention output with same shape as input ``q``.

    Raises:
        ValueError: If ``input_layout`` is not ``"BSND"`` or ``"BNSD"``.
        ValueError: If ``sparse_type`` is not ``None`` or ``"rf_v2"``.

    Examples:
        >>> # rf_v2 sparse attention (sparsity=0.0 for full dense)
        >>> import math
        >>> import torch
        >>> from lite_boost.ops.sparse_attention import sparse_attention
        >>> device = torch.device("npu:0")
        >>> batch_size, num_heads, seq_len, head_dim = 1, 3, 4096, 128
        >>> scale = head_dim ** -0.5
        >>> latent_shape = (1, 64, 64)
        >>> q = torch.randn(batch_size, seq_len, num_heads, head_dim,
        ...                 dtype=torch.float16, device=device)
        >>> k = torch.randn(batch_size, seq_len, num_heads, head_dim,
        ...                 dtype=torch.float16, device=device)
        >>> v = torch.randn(batch_size, seq_len, num_heads, head_dim,
        ...                 dtype=torch.float16, device=device)
        >>> out = sparse_attention(
        ...     q=q, k=k, v=v,
        ...     scale=scale, head_num=num_heads,
        ...     input_layout="BSND", inner_precise=0,
        ...     sparse_type="rf_v2",
        ...     block_size=128, latent_shape_q=latent_shape,
        ...     sparsity=0.0)
        >>> print(out.shape)
        (1, 4096, 3, 128)
    """
    check_params(input_layout, sparse_type)
    batch, head_dim = q.shape[0], q.shape[-1]
    scale = head_dim ** -0.5 if scale is None else scale

    if sparse_type == "rf_v2":
        q_rf, k_rf, v_rf, qkv_pool = do_tensor_rearrange_pooling(
            q, k, v, txt_len, block_size, latent_shape_q, latent_shape_k, input_layout
        )
        select_idx, select_num_idx = get_blockwise_mask(
            qkv_pool, txt_len, sparsity, scale, block_size, latent_shape_q, latent_shape_k, input_layout)

        if input_layout == "BSND":
            q_seq, kv_seq = q_rf.shape[1], k_rf.shape[1]
            layout = "TND"
            q_rf = q_rf.reshape(-1, head_num, head_dim)
            k_rf = k_rf.reshape(-1, head_num, head_dim)
            v_rf = v_rf.reshape(-1, head_num, head_dim)
        else:
            q_seq, kv_seq = q_rf.shape[2], k_rf.shape[2]
            layout = input_layout
        actual_seq_lengths = [q_seq for _ in range(batch)]
        actual_seq_lengths_kv = [kv_seq for _ in range(batch)]

        from . import rain_fusion_attention

        out, _ = rain_fusion_attention(
            q_rf, k_rf, v_rf,
            select_idx, select_num_idx,
            block_shape=[block_size, block_size],
            attn_mask=None,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            block_table=None,
            q_input_layout=layout,
            kv_input_layout=layout,
            num_key_value_heads=head_num,
            mask_type=0,
            scale_value=scale,
            inner_precise=inner_precise,
            block_size=0,
        )
        if layout == "TND":
            out = out.reshape(batch, q_seq, head_num, head_dim)
        out = do_tensor_inv_rearrange(out, txt_len, latent_shape_q, latent_shape_k, input_layout)
    elif sparse_type is None:
        import torch_npu
        out = torch_npu.npu_fusion_attention(
            q, k, v,
            input_layout=input_layout,
            scale=scale,
            pre_tockens=MAX_TOKEN,
            next_tockens=MAX_TOKEN,
            head_num=head_num)[0]
    else:
        raise ValueError(f"sparse_type must be None or 'rf_v2', but got {sparse_type}.")
    return out
