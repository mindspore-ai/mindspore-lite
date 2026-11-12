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
"""Graph-mode greedy decode for Qwen3-30B-A3B (Qwen3MoeForCausalLM, 1p / TP).

Load: load_qwen3_moe_bf16 -> boost_qwen3_30B_A3B(world_size, mode="tp",
    fuse_rms_norm/fuse_ffn/fuse_moe/fuse_moe_router=True, use_eager=False) BEFORE
    model.to(device).
Env: LB_PREFILL_GRAPH=1 LB_IFA_MASK=2 (not 0), LB_TP_SEPARATE_MM_AR=1 for the TP graph path;
    never set LB_GRAPH_PFA_V2=4 with a session.
Serve: sess = CommonPrefixSession(model, prefix_ids, device=...);
    sess.generate(input_ids, attention_mask); sess.close().
"""
from __future__ import annotations

import logging
import os
import time
import types

import torch
import torch_npu
from transformers import StaticCache

from .model import _get_causal_mask_2048

logger = logging.getLogger(__name__)

_LB_DECODE_SEQ_LEN_KV = 1
_LB_DECODE_POS_T = None
_LB_DECODE_QKVROPE_ROWS = {}
_LB_DECODE_QKVROPE_STATIC = False


def _tp_barrier():
    """Rendezvous all TP ranks; no-op for 1p / uninitialized process group."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()

PREFILL_BUCKETS = [128, 256, 512, 1024, 2048]
_PREFILL_GRAPHS: dict = {}

_LB_PREFILL_WRITE_BASE = 0
_LB_PREFILL_POS_OFFSET = 0
_LB_PREFIX_KV_ACTIVE = None

_LB_DEBUG_DUMP = {}


def _select_prefill_bucket(prompt_len):
    """Pick the smallest bucket ≥ prompt_len."""
    for b in PREFILL_BUCKETS:
        if prompt_len <= b:
            return b
    return None


def _make_static_cache(model, max_cache_len):
    """Build a StaticCache pre-allocated to `max_cache_len`."""
    if hasattr(model, "model"):
        text_model = model.model
    else:
        text_model = model
    config = text_model.config
    use_bsnd = os.environ.get("LB_CACHE_BSND", "1") == "1"
    if use_bsnd:
        _ensure_bsnd_static_layer_patch()
    cache = StaticCache(config=config, max_batch_size=1, max_cache_len=max_cache_len)
    if use_bsnd:
        cache._lb_bsnd = True
    return cache


def _ensure_bsnd_static_layer_patch():
    """Class-level StaticLayer patch: BSND allocation + layout-aware update."""
    from transformers.cache_utils import StaticLayer

    if getattr(StaticLayer, "_lb_bsnd_patched", False):
        return

    def _lazy_bsnd(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.max_batch_size = key_states.shape[0]
        self.num_heads = key_states.shape[2]
        self.v_head_dim = value_states.shape[-1]
        self.k_head_dim = key_states.shape[-1]
        self.keys = torch.zeros(
            (self.max_batch_size, self.max_cache_len, self.num_heads, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.max_batch_size, self.max_cache_len, self.num_heads, self.v_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.cumulative_length = self.cumulative_length.to(self.device)
        self._lb_pos0 = torch.zeros(1, dtype=torch.long, device=self.device)
        self._keys_flat = None
        self._values_flat = None
        self._lb_bsnd = True
        try:
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)
            torch._dynamo.mark_static_address(self.cumulative_length)
        except Exception:
            logger.debug("mark_static_address unavailable on this cache; "
                         "graph capture may re-analyze addresses", exc_info=True)
        self.is_initialized = True

    def _update_bsnd(self, key_states, value_states, *args, **kwargs):  # pylint: disable=unused-argument
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        kv_length = key_states.shape[1]
        if _LB_CAPTURING_PREFILL and kv_length > 1:
            _base = _LB_PREFILL_WRITE_BASE
            self.cumulative_length.add_(kv_length)
            self.keys[:, _base:_base + kv_length, :, :] = key_states
            self.values[:, _base:_base + kv_length, :, :] = value_states
            return self.keys, self.values
        if kv_length == 1 and self.keys.shape[0] == 1:
            cache_position = self._lb_pos0 + self.cumulative_length
            self.cumulative_length.add_(kv_length)
            if self._keys_flat is None:
                self._keys_flat = self.keys.view(self.max_cache_len, -1)
                self._values_flat = self.values.view(self.max_cache_len, -1)
            self._keys_flat.index_copy_(0, cache_position, key_states.reshape(1, -1))
            self._values_flat.index_copy_(0, cache_position, value_states.reshape(1, -1))
            return self.keys, self.values
        cache_position = torch.arange(
            kv_length, dtype=torch.long, device=self.device
        ) + self.cumulative_length
        self.cumulative_length.add_(kv_length)
        self.keys.index_copy_(1, cache_position, key_states)
        self.values.index_copy_(1, cache_position, value_states)
        return self.keys, self.values

    StaticLayer.lazy_initialization = _lazy_bsnd
    StaticLayer.update = _update_bsnd
    StaticLayer._lb_bsnd_patched = True


def _decode_position_ids(past_len, batch_size, device):
    """Build position_ids for a single decode step."""
    pos_1d = torch.arange(past_len, past_len + 1, dtype=torch.long, device=device)
    position_ids = pos_1d.view(1, 1).expand(batch_size, 1)
    return position_ids

_STATIC_BUFFERS: dict = {}

_IFA_MASK_ARANGE: dict = {}
_IFA_MASK_SHARED: dict = {}


def _get_static_mask(layer_idx, device, max_cache_len, batch=1):
    """Get or create the static decode mask for this layer + device (3D float16)."""
    key = (layer_idx, str(device))
    buf = _STATIC_BUFFERS.get(key)
    if buf is None or buf["max_cache_len"] < max_cache_len:
        mask = torch.full(
            (batch, 1, max_cache_len), -1e4,
            dtype=torch.float16, device=device,
        )
        buf = {"mask": mask, "max_cache_len": max_cache_len}
        _STATIC_BUFFERS[key] = buf
    return buf["mask"]


def _get_static_mask_4d(layer_idx, device, max_cache_len, batch=1):
    """Get or create the static decode mask for this layer + device (4D bool)."""
    key = (layer_idx, str(device), "4d")
    buf = _STATIC_BUFFERS.get(key)
    if buf is None or buf["max_cache_len"] < max_cache_len:
        mask = torch.ones(
            (batch, 1, 1, max_cache_len), dtype=torch.bool, device=device,
        )
        buf = {"mask": mask, "max_cache_len": max_cache_len}
        _STATIC_BUFFERS[key] = buf
    return buf["mask"]


def reset_static_masks(real_len, max_cache_len, device, num_layers=None):
    """Reset all static decode masks."""
    if num_layers is not None:
        for layer_idx in range(num_layers):
            _ = _get_static_mask(layer_idx, device, max_cache_len)
            _ = _get_static_mask_4d(layer_idx, device, max_cache_len)
    for _, buf in _STATIC_BUFFERS.items():
        if buf["max_cache_len"] < max_cache_len:
            continue
        mask = buf["mask"]
        if mask.ndim == 3 and mask.dtype == torch.float16:
            mask.fill_(-1e4)
            if real_len > 0:
                mask[:, :, :real_len] = 0
        elif mask.ndim == 4 and mask.dtype == torch.bool:
            mask.fill_(True)
            if real_len > 0:
                mask[:, :, :, :real_len] = False
        else:
            mask.fill_(True)
            if real_len > 0:
                if mask.ndim == 3:
                    mask[:, :, :real_len] = False
                elif mask.ndim == 4:
                    mask[:, :, :, :real_len] = False

_PFA_PREFILL_MASKS: dict = {}


def _pfa_prefill_mask_cache(seq_len, device, prefix_len=0):
    """Causal bool mask (True = masked)."""
    key = (seq_len, prefix_len, str(device))
    m = _PFA_PREFILL_MASKS.get(key)
    if m is None:
        if prefix_len > 0:
            m = torch.cat(
                (
                    torch.zeros(seq_len, prefix_len, dtype=torch.bool, device=device),
                    torch.triu(
                        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1,
                    ),
                ),
                dim=1,
            ).unsqueeze(0).unsqueeze(0)
        else:
            m = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
        _PFA_PREFILL_MASKS[key] = m
    return m

_MROPE_POSITIONS: dict = {}
_MROPE_COS_SIN: dict = {}
_MROPE_MAX_SEQ = 8192


def _mrope_positions_cache(T, device, offset=0):
    key = (T, offset, str(device))
    p = _MROPE_POSITIONS.get(key)
    if p is None:
        p = torch.arange(offset, offset + T, dtype=torch.int64, device=device)
        _MROPE_POSITIONS[key] = p
    return p


def _mrope_cos_sin_cache(device, inv_freq):
    key = str(device)
    cs = _MROPE_COS_SIN.get(key)
    if cs is None:
        inv = inv_freq.to(device=device, dtype=torch.float32).reshape(-1)
        pos = torch.arange(_MROPE_MAX_SEQ, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv)
        cs = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)
        _MROPE_COS_SIN[key] = cs
    return cs

_QKVROPE_COS: dict = {}
_QKVROPE_SIN: dict = {}
_QKVROPE_INDEX: dict = {}


def _qkvrope_cos_sin_caches(device, inv_freq):
    """Return the cached cos/sin tables for the fused QKV-RoPE path."""
    key = str(device)
    cos_t = _QKVROPE_COS.get(key)
    if cos_t is None:
        inv = inv_freq.to(device=device, dtype=torch.float32).reshape(-1)
        pos = torch.arange(_MROPE_MAX_SEQ, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv)
        cos_t = torch.cat((freqs.cos(), freqs.cos()), dim=-1).to(torch.bfloat16)
        sin_t = torch.cat((freqs.sin(), freqs.sin()), dim=-1).to(torch.bfloat16)
        _QKVROPE_COS[key] = cos_t
        _QKVROPE_SIN[key] = sin_t
    return cos_t, _QKVROPE_SIN[key]


def _qkvrope_skip_index(num_token, device):
    key = (num_token, str(device))
    idx = _QKVROPE_INDEX.get(key)
    if idx is None:
        idx = torch.full((num_token,), -1, dtype=torch.int64, device=device)
        _QKVROPE_INDEX[key] = idx
    return idx


def _qkvrope_decode_rows(device, inv_freq):
    """Static [1, head_dim] cos/sin rows + the full tables, for decode T=1.

    The row buffers are created once (before graph capture) so the captured
    aclnnQkvRmsNormRopeCache call reads this exact tensor object; the replay
    loop refreshes their contents per step.
    """
    key = str(device)
    rows = _LB_DECODE_QKVROPE_ROWS.get(key)
    if rows is None:
        cos_t, sin_t = _qkvrope_cos_sin_caches(device, inv_freq)
        rows = (cos_t[:1].clone(), sin_t[:1].clone(), cos_t, sin_t)
        _LB_DECODE_QKVROPE_ROWS[key] = rows
    return rows


def _refresh_qkvrope_decode_rows(inv_freq, pos, device):
    """Point the static cos/sin rows at rope position `pos` (host int)."""
    key = str(device)
    rows = _LB_DECODE_QKVROPE_ROWS.get(key)
    if rows is None:
        if inv_freq is None:
            return
        rows = _qkvrope_decode_rows(device, inv_freq)
    cos_row, sin_row, cos_t, sin_t = rows
    idx = min(max(int(pos), 0), cos_t.shape[0] - 1)
    cos_row.copy_(cos_t[idx:idx + 1])
    sin_row.copy_(sin_t[idx:idx + 1])


def qwen3_30B_full_attn_forward_graph(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    **kwargs,
):  # pylint: disable=unused-argument
    """Graph-safe `Qwen3MoeAttention.forward` for StaticCache."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        apply_rotary_pos_emb,
    )

    input_shape = hidden_states.shape[:-1]
    seq_len = hidden_states.shape[1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    use_pfa_prefill = os.environ.get("LB_PREFILL_GRAPH", "0") == "1"
    use_qkvrope = (
        use_pfa_prefill and seq_len > 1
        and hidden_states.shape[0] == 1
        and os.environ.get("LB_PREFILL_QKVROPE", "1") == "1"
        and getattr(self, "_lb_inv_freq", None) is not None
        and self.head_dim == 128
    )
    use_pfa_bsnd = (
        use_pfa_prefill and seq_len > 1
        and os.environ.get("LB_PREFILL_PFA_LAYOUT", "BSND") == "BSND"
    )
    k_pf = v_pf = None
    use_qkvrope_dec = (
        seq_len == 1
        and os.environ.get("LB_DECODE_QKVROPE", "0") == "1"
        and getattr(self, "_lb_inv_freq", None) is not None
        and _LB_DECODE_POS_T is not None
        and _LB_DECODE_QKVROPE_STATIC
    )

    if use_qkvrope:
        import lite_boost.ops  # pylint: disable=unused-import
        if os.environ.get("LB_QKVROPE_DEBUG", "0") == "1":
            logger.info("%s", f"[qkvrope] active layer={self.layer_idx} T={seq_len}")
        t_prefill = seq_len
        n_q = self.q_proj.out_features // self.head_dim
        n_kv = self.k_proj.out_features // self.head_dim
        cos_t, sin_t = _qkvrope_cos_sin_caches(hidden_states.device, self._lb_inv_freq)
        _pos_off = _LB_PREFILL_POS_OFFSET
        w_qkv = getattr(self, "_lb_qkv_weight", None)
        if w_qkv is not None:
            qkv = torch.nn.functional.linear(hidden_states, w_qkv).view(t_prefill, -1)
        else:
            q_t = torch.nn.functional.linear(hidden_states, self.q_proj.weight).view(t_prefill, -1)
            k_t = torch.nn.functional.linear(hidden_states, self.k_proj.weight).view(t_prefill, -1)
            v_t = torch.nn.functional.linear(hidden_states, self.v_proj.weight).view(t_prefill, -1)
            qkv = torch.cat((q_t, k_t, v_t), dim=1)
        q_out, k_out, v_out = torch.ops.lite_boost.qkv_rms_norm_rope_cache(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_t[_pos_off:_pos_off + t_prefill],
            sin_t[_pos_off:_pos_off + t_prefill],
            _qkvrope_skip_index(t_prefill, hidden_states.device),
            [1, t_prefill, n_q + 2 * n_kv, self.head_dim],
            [n_q, n_kv, n_kv],
            float(getattr(self.q_norm, "variance_epsilon", 1e-6)),
            "PA_NZ",
        )
        q_bs = q_out.view(1, t_prefill, n_q, self.head_dim)
        k_bs = k_out.view(1, t_prefill, n_kv, self.head_dim)
        v_bs = v_out.view(1, t_prefill, n_kv, self.head_dim)
        key_states = k_bs.transpose(1, 2)
        value_states = v_bs.transpose(1, 2)
        if use_pfa_bsnd:
            query_states, k_pf, v_pf = q_bs, k_bs, v_bs
        else:
            query_states = q_bs.transpose(1, 2)
    elif use_qkvrope_dec:
        import lite_boost.ops  # pylint: disable=unused-import
        n_q = self.q_proj.out_features // self.head_dim
        n_kv = self.k_proj.out_features // self.head_dim
        cos_row, sin_row, _, _ = _qkvrope_decode_rows(
            hidden_states.device, self._lb_inv_freq
        )
        w_qkv_d = getattr(self, "_lb_qkv_weight", None)
        if w_qkv_d is not None:
            qkv = torch.nn.functional.linear(
                hidden_states, w_qkv_d
            ).view(1, -1)
        else:
            qkv = torch.cat((
                torch.nn.functional.linear(hidden_states, self.q_proj.weight),
                torch.nn.functional.linear(hidden_states, self.k_proj.weight),
                torch.nn.functional.linear(hidden_states, self.v_proj.weight),
            ), dim=-1).view(1, -1)
        q_out, k_out, v_out = torch.ops.lite_boost.qkv_rms_norm_rope_cache(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_row, sin_row,
            _qkvrope_skip_index(1, hidden_states.device),
            [1, 1, n_q + 2 * n_kv, self.head_dim],
            [n_q, n_kv, n_kv],
            float(getattr(self.q_norm, "variance_epsilon", 1e-6)),
            "PA_NZ",
        )
        query_states = q_out.view(1, 1, n_q, self.head_dim).transpose(1, 2)
        key_states = k_out.view(1, 1, n_kv, self.head_dim).transpose(1, 2)
        value_states = v_out.view(1, 1, n_kv, self.head_dim).transpose(1, 2)
    else:
        q_bsnd = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        k_bsnd = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        v_bsnd = self.v_proj(hidden_states).view(hidden_shape)

        use_mrope = (
            use_pfa_prefill and seq_len > 1
            and q_bsnd.shape[0] == 1
            and os.environ.get("LB_PREFILL_MROPE", "1") == "1"
            and getattr(self, "_lb_inv_freq", None) is not None
        )

        if use_mrope:
            t_prefill = q_bsnd.shape[1]
            positions = _mrope_positions_cache(
                t_prefill, q_bsnd.device, offset=_LB_PREFILL_POS_OFFSET
            )
            cos_sin = _mrope_cos_sin_cache(q_bsnd.device, self._lb_inv_freq)
            q_rot, k_rot = torch_npu.npu_mrope(
                positions,
                q_bsnd.reshape(t_prefill, -1),
                k_bsnd.reshape(t_prefill, -1),
                cos_sin,
                self.head_dim,
                mrope_section=[0, 0, 0],
                rotary_mode="half",
            )
            q_bs = q_rot.view(1, t_prefill, -1, self.head_dim)
            k_bs = k_rot.view(1, t_prefill, -1, self.head_dim)
            key_states = k_bs.transpose(1, 2)
            value_states = v_bsnd.transpose(1, 2)
            if use_pfa_bsnd:
                query_states, k_pf, v_pf = q_bs, k_bs, v_bsnd
            else:
                query_states = q_bs.transpose(1, 2)
        else:
            use_mrope_dec = (
                os.environ.get("LB_DECODE_MROPE", "1") == "1"
                and _LB_DECODE_POS_T is not None
                and getattr(self, "_lb_inv_freq", None) is not None
            )
            if use_mrope_dec:
                cos_sin = _mrope_cos_sin_cache(hidden_states.device, self._lb_inv_freq)
                q_rot, k_rot = torch_npu.npu_mrope(
                    _LB_DECODE_POS_T.view(-1),
                    q_bsnd.reshape(1, -1), k_bsnd.reshape(1, -1),
                    cos_sin, self.head_dim,
                    mrope_section=[0, 0, 0], rotary_mode="half",
                )
                query_states = q_rot.view(1, 1, -1, self.head_dim).transpose(1, 2)
                key_states = k_rot.view(1, 1, -1, self.head_dim).transpose(1, 2)
                value_states = v_bsnd.transpose(1, 2)
            else:
                query_states = q_bsnd.transpose(1, 2)
                key_states = k_bsnd.transpose(1, 2)
                value_states = v_bsnd.transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                _cache_dtype = self.k_proj.weight.dtype
                if query_states.dtype != _cache_dtype:
                    query_states = query_states.to(_cache_dtype)
                if key_states.dtype != _cache_dtype:
                    key_states = key_states.to(_cache_dtype)
                if value_states.dtype != _cache_dtype:
                    value_states = value_states.to(_cache_dtype)

    if seq_len > 1:
        if past_key_values is not None:
            if getattr(past_key_values, "_lb_bsnd", False):
                if k_pf is not None:
                    past_key_values.update(k_pf, v_pf, self.layer_idx)
                else:
                    past_key_values.update(
                        key_states.transpose(1, 2), value_states.transpose(1, 2),
                        self.layer_idx,
                    )
            else:
                past_key_values.update(key_states, value_states, self.layer_idx)
        k_for_attn = k_pf if k_pf is not None else key_states
        v_for_attn = v_pf if v_pf is not None else value_states

        prefix_kv_layer = None
        if _LB_PREFIX_KV_ACTIVE is not None:
            prefix_kv_layer = _LB_PREFIX_KV_ACTIVE[self.layer_idx]
        _fascore_sp = (
            prefix_kv_layer is not None
            and k_pf is not None
            and os.environ.get("LB_PREFILL_FASCORE", "1") == "1"
            and os.environ.get("LB_PREFILL_FASCORE_PREFIX", "1") == "1"
        )
        if prefix_kv_layer is not None and not _fascore_sp:
            if k_pf is not None:
                k_for_attn = torch.cat((prefix_kv_layer["k"], k_pf), dim=1)
                v_for_attn = torch.cat((prefix_kv_layer["v"], v_pf), dim=1)
            else:
                k_for_attn = torch.cat(
                    (prefix_kv_layer["k"].transpose(1, 2), key_states), dim=2
                )
                v_for_attn = torch.cat(
                    (prefix_kv_layer["v"].transpose(1, 2), value_states), dim=2
                )

        if use_pfa_prefill:
            pfa_layout_pf = "BSND" if k_pf is not None else "BNSD"
            _pf_plen = prefix_kv_layer["k"].shape[1] if prefix_kv_layer is not None else 0
            causal_mask_pf = _pfa_prefill_mask_cache(
                seq_len, hidden_states.device, prefix_len=_pf_plen
            )
            if pfa_layout_pf == "BSND":
                num_heads_pf = query_states.shape[2]
                num_kv_heads_pf = k_for_attn.shape[2]
            else:
                num_heads_pf = query_states.shape[1]
                num_kv_heads_pf = k_for_attn.shape[1]
            _pfa_window_kwargs = {}
            if prefix_kv_layer is not None:
                _pfa_window_kwargs = {
                    "pre_tokens": 2147483647, "next_tokens": 2147483647,
                }
            if os.environ.get("LB_PREFILL_FASCORE", "1") == "1":
                _sp_kwargs = {}
                if _fascore_sp:
                    _sp_kwargs = {
                        "key_shared_prefix": prefix_kv_layer["k"],
                        "value_shared_prefix": prefix_kv_layer["v"],
                        "actual_shared_prefix_len": [prefix_kv_layer["k"].shape[1]],
                    }
                attn_out = torch_npu.npu_fused_infer_attention_score(
                    query_states, k_for_attn, v_for_attn,
                    num_heads=num_heads_pf,
                    num_key_value_heads=num_kv_heads_pf,
                    input_layout=pfa_layout_pf,
                    scale=self.scaling,
                    atten_mask=causal_mask_pf,
                    sparse_mode=0,
                    **_pfa_window_kwargs,
                    **_sp_kwargs,
                )[0]
            else:
                attn_out = torch_npu.npu_prompt_flash_attention(
                    query_states, k_for_attn, v_for_attn,
                    num_heads=num_heads_pf,
                    num_key_value_heads=num_kv_heads_pf,
                    input_layout=pfa_layout_pf,
                    scale_value=self.scaling,
                    atten_mask=causal_mask_pf,
                    sparse_mode=0,
                    **_pfa_window_kwargs,
                )
            if pfa_layout_pf == "BSND":
                attn_output = attn_out.reshape(*input_shape, -1)
            else:
                attn_output = attn_out.transpose(1, 2).reshape(*input_shape, -1)
            attn_output = self.o_proj(attn_output)
            if os.environ.get("LB_DEBUG_ATTN", "0") == "1" and self.layer_idx == 0:
                _LB_DEBUG_DUMP["q"] = query_states.detach().clone()
                _LB_DEBUG_DUMP["k"] = k_for_attn.detach().clone()
                _LB_DEBUG_DUMP["v"] = v_for_attn.detach().clone()
                _LB_DEBUG_DUMP["mask"] = causal_mask_pf.detach().clone()
                _LB_DEBUG_DUMP["attn_out"] = attn_output.detach().clone()
            return attn_output, None

        if prefix_kv_layer is not None:
            raise RuntimeError(
                "common-prefix suffix prefill requires the PFA prefill path "
                "(LB_PREFILL_GRAPH=1); the npu_fusion_attention fallback uses a "
                "right-down-causal template that cannot express prefix+suffix masks"
            )
        causal_mask = _get_causal_mask_2048(hidden_states.device)
        q_bsnd = query_states.transpose(1, 2)
        k_bsnd = k_for_attn.transpose(1, 2)
        v_bsnd = v_for_attn.transpose(1, 2)
        num_heads_local = q_bsnd.shape[2]
        attn_out = torch_npu.npu_fusion_attention(
            q_bsnd, k_bsnd, v_bsnd,
            num_heads_local,
            input_layout="BSND",
            keep_prob=1.0,
            scale=self.scaling,
            atten_mask=causal_mask,
            sparse_mode=3,
        )[0]
        attn_output = attn_out.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    cache_bsnd = getattr(past_key_values, "_lb_bsnd", False)
    if cache_bsnd:
        key_cache_full, value_cache_full = past_key_values.update(
            key_states.transpose(1, 2), value_states.transpose(1, 2),
            self.layer_idx,
        )
        key_cache_bnsd = key_cache_full.transpose(1, 2)
        value_cache_bnsd = value_cache_full.transpose(1, 2)
        max_cache_len = key_cache_full.shape[1]
    else:
        key_cache_full, value_cache_full = past_key_values.update(
            key_states, value_states, self.layer_idx
        )
        key_cache_bnsd = key_cache_full
        value_cache_bnsd = value_cache_full
        max_cache_len = key_cache_full.shape[2]
    batch = hidden_states.shape[0]

    cur_len_tensor = past_key_values.get_seq_length()
    if not torch.is_tensor(cur_len_tensor):
        cur_len_tensor = torch.tensor(
            [cur_len_tensor], dtype=torch.int64, device=hidden_states.device
        )

    use_pfa = os.environ.get("LB_GRAPH_PFA", "0")
    use_pfa_v2 = os.environ.get("LB_GRAPH_PFA_V2", "0")
    ifa_mask_mode = os.environ.get("LB_IFA_MASK", "0")
    _ifa_enabled = (
        os.environ.get("LB_DECODE_IFA", "1") == "1"
        and hasattr(torch_npu, "npu_incre_flash_attention")
    )
    # Set below unless the IFA fast path above returns first.
    new_pos = None
    if not (_ifa_enabled and ifa_mask_mode in ("0", "2")):
        new_pos = (cur_len_tensor - 1).clamp(min=0).to(torch.int64).view(1)

    if _ifa_enabled:
        ifa_mask = None
        if ifa_mask_mode == "2":
            ar = _IFA_MASK_ARANGE.get(str(hidden_states.device))
            if ar is None or ar.shape[-1] < max_cache_len:
                ar = torch.arange(
                    max_cache_len, dtype=torch.int32,
                    device=hidden_states.device,
                ).view(1, 1, 1, -1)
                _IFA_MASK_ARANGE[str(hidden_states.device)] = ar
            shared = _IFA_MASK_SHARED.get(str(hidden_states.device))
            if (
                self.layer_idx == 0
                or shared is None
                or shared.shape[-1] != max_cache_len
                or shared.device != ar.device
            ):
                shared = ar >= cur_len_tensor.view(1, 1, 1, 1).to(torch.int32)
                _IFA_MASK_SHARED[str(hidden_states.device)] = shared
            ifa_mask = shared
        elif ifa_mask_mode == "1":
            ifa_mask = _get_static_mask_4d(
                self.layer_idx, hidden_states.device, max_cache_len, batch=batch,
            )
            ifa_mask.index_copy_(
                3,
                new_pos,
                torch.zeros(
                    batch, 1, 1, 1, dtype=ifa_mask.dtype, device=ifa_mask.device,
                ),
            )
        q_bsnd_ifa = query_states.transpose(1, 2)
        if cache_bsnd:
            k_bsnd_ifa = key_cache_full
            v_bsnd_ifa = value_cache_full
        else:
            k_bsnd_ifa = key_cache_full.transpose(1, 2)
            v_bsnd_ifa = value_cache_full.transpose(1, 2)
        attn_out = torch_npu.npu_incre_flash_attention(
            q_bsnd_ifa, k_bsnd_ifa, v_bsnd_ifa,
            num_heads=q_bsnd_ifa.shape[2],
            scale_value=self.scaling,
            input_layout="BSND",
            num_key_value_heads=k_bsnd_ifa.shape[2],
            inner_precise=0,
            atten_mask=ifa_mask,
        )
        attn_output = attn_out.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    if use_pfa_v2 == "1":
        static_mask_3d = _get_static_mask(self.layer_idx, hidden_states.device, max_cache_len, batch=batch)
        static_mask_3d.index_copy_(
            2,
            new_pos,
            torch.zeros(batch, 1, 1, dtype=static_mask_3d.dtype, device=static_mask_3d.device),
        )
        static_mask_4d_fp16 = static_mask_3d.unsqueeze(2)
        q_bnsd = query_states
        k_bnsd = key_cache_bnsd
        v_bnsd = value_cache_bnsd
        num_heads_local = q_bnsd.shape[1]
        num_kv_heads_local = k_bnsd.shape[1]

        attn_out = torch_npu.npu_prompt_flash_attention(
            q_bnsd, k_bnsd, v_bnsd,
            num_heads=num_heads_local,
            num_key_value_heads=num_kv_heads_local,
            input_layout="BNSD",
            scale_value=self.scaling,
            atten_mask=static_mask_4d_fp16,
            pre_tokens=2147483647,
            next_tokens=0,
            sparse_mode=0,
        )
        attn_output = attn_out.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    if use_pfa_v2 == "2":
        static_mask_3d = _get_static_mask(self.layer_idx, hidden_states.device, max_cache_len, batch=batch)
        static_mask_3d.index_copy_(
            2,
            new_pos,
            torch.zeros(batch, 1, 1, dtype=static_mask_3d.dtype, device=static_mask_3d.device),
        )
        static_mask_4d_fp16 = static_mask_3d.unsqueeze(2)
        q_bnsd = query_states
        k_bnsd = key_cache_bnsd
        v_bnsd = value_cache_bnsd
        num_heads_local = q_bnsd.shape[1]
        num_kv_heads_local = k_bnsd.shape[1]
        kv_group = num_heads_local // num_kv_heads_local
        if kv_group > 1:
            k_bnsd = (k_bnsd.unsqueeze(2)
                                 .expand(-1, -1, kv_group, -1, -1)
                                 .reshape(batch, num_heads_local, max_cache_len, -1)
                                 .contiguous())
            v_bnsd = (v_bnsd.unsqueeze(2)
                                 .expand(-1, -1, kv_group, -1, -1)
                                 .reshape(batch, num_heads_local, max_cache_len, -1)
                                 .contiguous())
            num_kv_heads_local = num_heads_local

        attn_out = torch_npu.npu_prompt_flash_attention(
            q_bnsd, k_bnsd, v_bnsd,
            num_heads=num_heads_local,
            num_key_value_heads=num_kv_heads_local,
            input_layout="BNSD",
            scale_value=self.scaling,
            atten_mask=static_mask_4d_fp16,
            pre_tokens=2147483647,
            next_tokens=0,
            sparse_mode=0,
        )
        attn_output = attn_out.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    if use_pfa_v2 == "3":
        q_bnsd = query_states
        k_bnsd = key_cache_bnsd
        v_bnsd = value_cache_bnsd
        num_heads_local = q_bnsd.shape[1]
        num_kv_heads_local = k_bnsd.shape[1]
        kv_group = num_heads_local // num_kv_heads_local
        if kv_group > 1:
            k_bnsd = (k_bnsd.unsqueeze(2)
                                 .expand(-1, -1, kv_group, -1, -1)
                                 .reshape(batch, num_heads_local, max_cache_len, -1)
                                 .contiguous())
            v_bnsd = (v_bnsd.unsqueeze(2)
                                 .expand(-1, -1, kv_group, -1, -1)
                                 .reshape(batch, num_heads_local, max_cache_len, -1)
                                 .contiguous())
            num_kv_heads_local = num_heads_local

        attn_out = torch_npu.npu_prompt_flash_attention(
            q_bnsd, k_bnsd, v_bnsd,
            num_heads=num_heads_local,
            num_key_value_heads=num_kv_heads_local,
            input_layout="BNSD",
            scale_value=self.scaling,
            pre_tokens=2147483647,
            next_tokens=0,
            sparse_mode=0,
        )
        attn_output = attn_out.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    if use_pfa_v2 == "4":
        q_bnsd = query_states
        k_bnsd = key_cache_bnsd
        v_bnsd = value_cache_bnsd
        num_heads_local = q_bnsd.shape[1]
        num_kv_heads_local = k_bnsd.shape[1]

        seq_len_kv = _LB_DECODE_SEQ_LEN_KV
        attn_out = torch_npu.npu_prompt_flash_attention(
            q_bnsd, k_bnsd, v_bnsd,
            num_heads=num_heads_local,
            num_key_value_heads=num_kv_heads_local,
            input_layout="BNSD",
            scale_value=self.scaling,
            actual_seq_lengths_kv=[seq_len_kv],
            pre_tokens=2147483647,
            next_tokens=0,
            sparse_mode=0,
        )
        attn_output = attn_out.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    if use_pfa == "1":
        static_mask_3d = _get_static_mask(self.layer_idx, hidden_states.device, max_cache_len, batch=batch)
        static_mask_3d.index_copy_(
            2,
            new_pos,
            torch.zeros(batch, 1, 1, dtype=static_mask_3d.dtype, device=static_mask_3d.device),
        )
        q_bnsd = query_states
        k_bnsd = key_cache_bnsd
        v_bnsd = value_cache_bnsd
        num_heads_local = q_bnsd.shape[1]
        num_kv_heads_local = k_bnsd.shape[1]

        pfa_mode = os.environ.get("LB_GRAPH_PFA", "1")
        if pfa_mode == "2":
            kv_group = num_heads_local // num_kv_heads_local
            if kv_group > 1:
                k_bnsd = (k_bnsd.unsqueeze(2)
                                 .expand(-1, -1, kv_group, -1, -1)
                                 .reshape(batch, num_heads_local, max_cache_len, -1)
                                 .contiguous())
                v_bnsd = (v_bnsd.unsqueeze(2)
                                 .expand(-1, -1, kv_group, -1, -1)
                                 .reshape(batch, num_heads_local, max_cache_len, -1)
                                 .contiguous())
                num_kv_heads_local = num_heads_local
            static_mask_4d_fp16 = static_mask_3d.unsqueeze(2)
            attn_out = torch_npu.npu_prompt_flash_attention(
                q_bnsd, k_bnsd, v_bnsd,
                num_heads=num_heads_local,
                num_key_value_heads=num_kv_heads_local,
                input_layout="BNSD_BSND",
                scale_value=self.scaling,
                atten_mask=static_mask_4d_fp16,
                pre_tokens=2147483647,
                next_tokens=0,
                sparse_mode=0,
            )
        else:
            static_mask_4d_fp16 = static_mask_3d.unsqueeze(2)
            attn_out = torch_npu.npu_prompt_flash_attention(
                q_bnsd, k_bnsd, v_bnsd,
                num_heads=num_heads_local,
                num_key_value_heads=num_kv_heads_local,
                input_layout="BNSD_BSND",
                scale_value=self.scaling,
                atten_mask=static_mask_4d_fp16,
                pre_tokens=2147483647,
                next_tokens=0,
                sparse_mode=0,
            )
        attn_output = attn_out.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    use_manual = os.environ.get("LB_GRAPH_MANUAL", "0")
    if use_manual in ("1", "2"):
        q_bnsd = query_states
        num_heads_local = q_bnsd.shape[1]
        num_kv_heads_local = key_cache_bnsd.shape[1]
        kv_group = num_heads_local // num_kv_heads_local

        if use_manual == "1":
            cur_len_int = past_key_values.get_seq_length(self.layer_idx)
            if torch.is_tensor(cur_len_int):
                cur_len_int = int(cur_len_int.item())
            k_bnsd = key_cache_bnsd[:, :, :cur_len_int, :]
            v_bnsd = value_cache_bnsd[:, :, :cur_len_int, :]
            if kv_group > 1:
                k_bnsd = k_bnsd.repeat_interleave(kv_group, dim=1)
                v_bnsd = v_bnsd.repeat_interleave(kv_group, dim=1)
            scores = torch.matmul(q_bnsd, k_bnsd.transpose(-2, -1)) * self.scaling
            attn = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn, v_bnsd)
        else:
            k_bnsd = key_cache_bnsd
            v_bnsd = value_cache_bnsd
            if kv_group > 1:
                k_bnsd = k_bnsd.repeat_interleave(kv_group, dim=1)
                v_bnsd = v_bnsd.repeat_interleave(kv_group, dim=1)
            static_mask_3d = _get_static_mask(self.layer_idx, hidden_states.device, max_cache_len, batch=batch)
            static_mask_3d.index_copy_(
                2,
                new_pos,
                torch.zeros(batch, 1, 1, dtype=static_mask_3d.dtype, device=static_mask_3d.device),
            )
            scores = torch.matmul(q_bnsd, k_bnsd.transpose(-2, -1)) * self.scaling
            scores = scores + static_mask_3d.unsqueeze(1)
            attn = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn, v_bnsd)

        attn_output = attn_out.transpose(1, 2).reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    use_slice = os.environ.get("LB_NO_MASK_SLICE", "1") == "1"

    q_bsnd = query_states.transpose(1, 2)
    num_heads_local = q_bsnd.shape[2]

    if use_slice:
        cur_len_int = past_key_values.get_seq_length(self.layer_idx)
        if torch.is_tensor(cur_len_int):
            cur_len_int = int(cur_len_int.item())
        k_bsnd = key_cache_bnsd.transpose(1, 2)[:, :cur_len_int, :, :].contiguous()
        v_bsnd = value_cache_bnsd.transpose(1, 2)[:, :cur_len_int, :, :].contiguous()
        attn_out = torch_npu.npu_fusion_attention(
            q_bsnd, k_bsnd, v_bsnd,
            num_heads_local,
            input_layout="BSND",
            keep_prob=1.0,
            scale=self.scaling,
            sparse_mode=0,
        )[0]
        attn_output = attn_out.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    static_mask_4d = _get_static_mask_4d(self.layer_idx, hidden_states.device, max_cache_len, batch=batch)
    static_mask_4d.index_copy_(
        3,
        new_pos,
        torch.zeros(batch, 1, 1, 1, dtype=static_mask_4d.dtype, device=static_mask_4d.device),
    )

    k_bsnd = key_cache_bnsd.transpose(1, 2)
    v_bsnd = value_cache_bnsd.transpose(1, 2)

    attn_out = torch_npu.npu_fusion_attention(
        q_bsnd, k_bsnd, v_bsnd,
        num_heads_local,
        input_layout="BSND",
        keep_prob=1.0,
        scale=self.scaling,
        atten_mask=static_mask_4d,
        sparse_mode=0,
    )[0]
    attn_output = attn_out.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def _patch_attn_for_graph(model):
    """Patch every `Qwen3MoeAttention.forward` to the graph-safe variant."""
    text_model = model.model if hasattr(model, "model") else model
    rotary_emb = getattr(text_model, "rotary_emb", None)
    inv_freq = getattr(rotary_emb, "inv_freq", None) if rotary_emb is not None else None
    fuse_qkv = os.environ.get("LB_FUSED_QKV", "0") == "1"
    for layer in text_model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            if not hasattr(attn, "_original_forward"):
                attn._original_forward = attn.forward
            attn.forward = types.MethodType(qwen3_30B_full_attn_forward_graph, attn)
            if inv_freq is not None:
                attn._lb_inv_freq = inv_freq
            if fuse_qkv and not hasattr(attn, "_lb_qkv_weight"):
                with torch.no_grad():
                    _nq = attn.q_proj.out_features
                    _nk = attn.k_proj.out_features
                    _dev = attn.q_proj.weight.device
                    _dt = attn.q_proj.weight.dtype
                    _w_cpu = torch.cat(
                        (attn.q_proj.weight.data.to("cpu"),
                         attn.k_proj.weight.data.to("cpu"),
                         attn.v_proj.weight.data.to("cpu")),
                        dim=0,
                    )
                    _tiny = torch.empty(0, dtype=_dt, device=_dev)
                    attn.q_proj.weight.data = _tiny
                    attn.k_proj.weight.data = _tiny
                    attn.v_proj.weight.data = _tiny
                    try:
                        _w_fused = _w_cpu.to(_dev)
                        attn._lb_qkv_weight = _w_fused
                        attn.q_proj.weight.data = _w_fused[:_nq]
                        attn.k_proj.weight.data = _w_fused[_nq:_nq + _nk]
                        attn.v_proj.weight.data = _w_fused[_nq + _nk:]
                    except RuntimeError:
                        _w_q = _w_cpu[:_nq].to(_dev)
                        _w_k = _w_cpu[_nq:_nq + _nk].to(_dev)
                        _w_v = _w_cpu[_nq + _nk:].to(_dev)
                        attn.q_proj.weight.data = _w_q
                        attn.k_proj.weight.data = _w_k
                        attn.v_proj.weight.data = _w_v


def _unpatch_attn_after_graph(model):
    """Restore the original attention forward (revert `_patch_attn_for_graph`)."""
    text_model = model.model if hasattr(model, "model") else model
    for layer in text_model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "_original_forward"):
            attn.forward = attn._original_forward
            del attn._original_forward


def _graph_safe_text_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    **kwargs,
):
    """Graph-safe `Qwen3MoeModel.forward` for decode (T=1)."""
    seq_len = (inputs_embeds.shape[1] if inputs_embeds is not None
               else input_ids.shape[1] if input_ids is not None else 0)
    if seq_len > 1:
        if os.environ.get("LB_PREFILL_GRAPH", "0") != "1":
            return self._forward_eager(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            t_prefill = inputs_embeds.shape[1]
            position_ids = torch.arange(
                t_prefill, dtype=torch.long, device=inputs_embeds.device,
            ).unsqueeze(0)
        causal_mask = None
        hidden_states = inputs_embeds
        _skip_rotary = (
            os.environ.get("LB_PREFILL_MROPE", "1") == "1"
            and getattr(self.layers[0].self_attn, "_lb_inv_freq", None) is not None
        )
        if _skip_rotary:
            position_embeddings = (None, None)
        else:
            position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        for _, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            MoeModelOutputWithPast,
        )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    text_position_ids = position_ids

    causal_mask = None

    hidden_states = inputs_embeds
    _skip_rotary_dec = (
        (os.environ.get("LB_DECODE_MROPE", "1") == "1"
         or os.environ.get("LB_DECODE_QKVROPE", "1") == "1")
        and text_position_ids is not None
        and getattr(self.layers[0].self_attn, "_lb_inv_freq", None) is not None
    )
    if _skip_rotary_dec:
        global _LB_DECODE_POS_T
        _LB_DECODE_POS_T = text_position_ids
        position_embeddings = (None, None)
    else:
        position_embeddings = self.rotary_emb(hidden_states, position_ids=text_position_ids)

    for _, decoder_layer in enumerate(self.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        MoeModelOutputWithPast,
    )
    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


def _patch_text_model_for_graph(model):
    """Patch `Qwen3MoeModel.forward` to the graph-safe variant."""
    text_model = model.model if hasattr(model, "model") else model
    if not hasattr(text_model, "_forward_eager"):
        text_model._forward_eager = text_model.forward
    text_model.forward = types.MethodType(_graph_safe_text_model_forward, text_model)


def _unpatch_text_model_after_graph(model):
    """Restore the original text model forward (revert `_patch_text_model_for_graph`)."""
    text_model = model.model if hasattr(model, "model") else model
    if hasattr(text_model, "_forward_eager"):
        text_model.forward = text_model._forward_eager
        del text_model._forward_eager

_LB_CAPTURING_PREFILL = False


def _kv_cache_zero_range(cache, start, end=None):
    """Zero cache positions [start, end) across all layers, layout-aware."""
    with torch.inference_mode():
        for layer in cache.layers:
            if not hasattr(layer, "keys") or not torch.is_tensor(layer.keys):
                continue
            if getattr(layer, "_lb_bsnd", False):
                layer.keys[:, start:end, :, :] = 0
                layer.values[:, start:end, :, :] = 0
            else:
                layer.keys[:, :, start:end, :] = 0
                layer.values[:, :, start:end, :] = 0


def _make_fast_static_layer_update():
    """Patch `StaticLayer.update` (class-level) with a prefill-capture fast path."""
    from transformers.cache_utils import StaticLayer

    if getattr(StaticLayer, "_lb_bsnd_patched", False):
        return
    if getattr(StaticLayer, "_lb_fast_update_patched", False):
        return
    _orig_update = StaticLayer.update

    def _fast_update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        kv_length = key_states.shape[-2]
        if _LB_CAPTURING_PREFILL and kv_length > 1:
            self.cumulative_length.add_(kv_length)
            self.keys[:, :, :kv_length, :] = key_states
            self.values[:, :, :kv_length, :] = value_states
            return self.keys, self.values
        return _orig_update(self, key_states, value_states, *args, **kwargs)

    StaticLayer.update = _fast_update
    StaticLayer._lb_fast_update_patched = True
    StaticLayer._lb_orig_update = _orig_update


def _cann_moe_grouped_forward(self, hidden_states, top_k_index, top_k_weights):
    """Fully-fused CANN MoE flow (aclnnMoeInitRoutingV2 + GroupedMatmulV5 + MoeTokenUnpermute)."""
    num_tokens = hidden_states.shape[0]

    use_grouped_decode = os.environ.get("LB_MOE_DECODE_GROUPED", "0") == "1"

    if num_tokens > 1 or use_grouped_decode:
        from lite_boost.layers.moe import cann_moe_forward
        return cann_moe_forward(
            hidden_states, top_k_index, top_k_weights,
            self.gate_up_proj, self.down_proj, act_fn=self.act_fn,
        )

    return _fused_moe_grouped_forward(self, hidden_states, top_k_index, top_k_weights)


def _fused_moe_grouped_forward(self, hidden_states, top_k_index, top_k_weights):
    """Graph-safe fused MoE: sort routing (T>1) or slot-major bmm (T=1)."""
    from lite_boost.layers.moe import bmm_moe_forward, sort_moe_forward

    if hidden_states.shape[0] > 1:
        return sort_moe_forward(
            hidden_states, top_k_index, top_k_weights,
            self.gate_up_proj, self.down_proj, act_fn=self.act_fn,
        )
    return bmm_moe_forward(
        hidden_states, top_k_index, top_k_weights,
        self.gate_up_proj, self.down_proj, act_fn=self.act_fn,
    )


def _batched_moe_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Legacy graph-safe MoE forward (dense per-expert loop for T>1)."""
    num_tokens = hidden_states.shape[0]
    top_k = top_k_index.shape[-1]
    _ = hidden_states.shape[-1]

    if num_tokens > 1:
        num_experts = self.num_experts
        weight_matrix = torch.zeros(
            num_tokens, num_experts,
            dtype=top_k_weights.dtype, device=hidden_states.device,
        )
        weight_matrix.scatter_add_(1, top_k_index, top_k_weights)
        final = torch.zeros_like(hidden_states)
        for e in range(num_experts):
            gu_e = self.gate_up_proj[e]
            dp_e = self.down_proj[e]
            gate_up = torch.nn.functional.linear(hidden_states, gu_e)
            gate, up = gate_up.chunk(2, dim=-1)
            inter = self.act_fn(gate) * up
            expert_out = torch.nn.functional.linear(inter, dp_e)
            w_e = weight_matrix[:, e].unsqueeze(-1)
            final = final.addcmul(expert_out, w_e)
        return final

    flat_idx = top_k_index.reshape(-1)
    expert_gu = torch.index_select(self.gate_up_proj, 0, flat_idx)
    expert_dp = torch.index_select(self.down_proj, 0, flat_idx)

    hidden_expanded = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_states.shape[-1])

    gate_up = torch.bmm(
        hidden_expanded.unsqueeze(1),
        expert_gu.transpose(1, 2),
    )
    gate, up = gate_up.chunk(2, dim=-1)
    current_hidden = self.act_fn(gate) * up

    expert_out = torch.bmm(current_hidden, expert_dp.transpose(1, 2))
    expert_out = expert_out.squeeze(1)

    weights_flat = top_k_weights.reshape(-1, 1)
    weighted = expert_out * weights_flat
    final = weighted.reshape(num_tokens, top_k, -1).sum(dim=1)
    return final


def _fused_router_forward(self, hidden_states):
    """Fused softmax+topk router via `npu_moe_gating_top_k`."""
    from lite_boost.layers.moe import fused_router_forward
    return fused_router_forward(
        hidden_states, self.weight, self.top_k,
        self.norm_topk_prob, self.hidden_dim,
    )


def patch_moe_for_graph_decode(model):
    """Patch `Qwen3MoeExperts.forward` to a graph-safe variant."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeExperts,
        Qwen3MoeTopKRouter,
    )
    moe_mode = os.environ.get("LB_MOE_FUSED", "2")
    if moe_mode == "2":
        fwd = _cann_moe_grouped_forward
    elif moe_mode == "1":
        fwd = _fused_moe_grouped_forward
    else:
        fwd = _batched_moe_experts_forward
    fused_router = (
        moe_mode == "2"
        and os.environ.get("LB_FUSED_ROUTER", "1") == "1"
        and hasattr(torch_npu, "npu_moe_gating_top_k")
    )
    text_model = model.model if hasattr(model, "model") else model
    patched = 0
    for layer in text_model.layers:
        mlp = getattr(layer, "mlp", None)
        inner_mlp = getattr(mlp, "_moe", mlp)
        experts = getattr(inner_mlp, "experts", None)
        if experts is not None and isinstance(experts, Qwen3MoeExperts):
            if getattr(experts, "_lb_moe_fuse_owner", None) == "boost":
                continue
            if not hasattr(experts, "_original_forward"):
                experts._original_forward = experts.forward
            experts.forward = types.MethodType(fwd, experts)
            patched += 1
        if fused_router:
            gate = getattr(inner_mlp, "gate", None)
            if gate is not None and isinstance(gate, Qwen3MoeTopKRouter):
                if getattr(gate, "_lb_moe_router_fuse_owner", None) == "boost":
                    continue
                if not hasattr(gate, "_original_forward"):
                    gate._original_forward = gate.forward
                gate.forward = types.MethodType(_fused_router_forward, gate)
    return patched


def unpatch_moe_after_graph_decode(model):
    """Restore the original MoE experts forward (revert `patch_moe_for_graph_decode`)."""
    text_model = model.model if hasattr(model, "model") else model
    for layer in text_model.layers:
        mlp = getattr(layer, "mlp", None)
        inner_mlp = getattr(mlp, "_moe", mlp)
        experts = getattr(inner_mlp, "experts", None)
        if experts is not None and hasattr(experts, "_original_forward"):
            if getattr(experts, "_lb_moe_fuse_owner", None) == "boost":
                continue
            experts.forward = experts._original_forward
            del experts._original_forward
        gate = getattr(inner_mlp, "gate", None)
        if gate is not None and hasattr(gate, "_original_forward"):
            if getattr(gate, "_lb_moe_router_fuse_owner", None) == "boost":
                continue
            gate.forward = gate._original_forward
            del gate._original_forward

_LB_PREFILL_T_VALID = 0


def _capture_prefill_graph(model, bucket_size, device, cache, pad_token_id,
                           write_base=0, prefix_kv=None):
    """Capture a prefill graph for the given bucket size."""
    global _LB_PREFILL_WRITE_BASE, _LB_PREFILL_POS_OFFSET, _LB_PREFIX_KV_ACTIVE
    _saved_globals = (_LB_PREFILL_WRITE_BASE, _LB_PREFILL_POS_OFFSET, _LB_PREFIX_KV_ACTIVE)
    _LB_PREFILL_WRITE_BASE = write_base
    _LB_PREFILL_POS_OFFSET = write_base
    _LB_PREFIX_KV_ACTIVE = prefix_kv
    try:
        return _capture_prefill_graph_inner(
            model, bucket_size, device, cache, pad_token_id, write_base,
        )
    finally:
        (_LB_PREFILL_WRITE_BASE, _LB_PREFILL_POS_OFFSET, _LB_PREFIX_KV_ACTIVE) = _saved_globals


def _capture_prefill_graph_inner(model, bucket_size, device, cache, pad_token_id,
                                 write_base):
    """Capture body — runs with the suffix-prefill globals already set."""
    text_model = model.model if hasattr(model, "model") else model
    capture_stream = torch_npu.npu.Stream(device=device)
    capture_stream.wait_stream(torch_npu.npu.current_stream())

    static_ids = torch.full((1, bucket_size), pad_token_id, dtype=torch.long, device=device)

    with torch_npu.npu.stream(capture_stream):
        for _ in range(2):
            with torch.inference_mode():
                _pf_out = text_model(
                    input_ids=static_ids,
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=cache,
                    use_cache=True,
                )
                _pf_lm = getattr(model, "lm_head", None)
                if _pf_lm is not None:
                    _ = _pf_lm(_pf_out.last_hidden_state[:, -1:, :])
    capture_stream.synchronize()
    torch_npu.npu.synchronize()
    _tp_barrier()

    with torch.inference_mode():
        for layer in cache.layers:
            if hasattr(layer, "cumulative_length") and torch.is_tensor(layer.cumulative_length):
                layer.cumulative_length.copy_(
                    torch.tensor([write_base], dtype=layer.cumulative_length.dtype,
                                 device=layer.cumulative_length.device).reshape(layer.cumulative_length.shape))

    lm_head_weight = model.lm_head.weight
    _ = getattr(model.lm_head, "bias", None)

    cum_tensors = [
        layer.cumulative_length for layer in cache.layers
        if hasattr(layer, "cumulative_length") and torch.is_tensor(layer.cumulative_length)
    ]
    g = torch_npu.npu.NPUGraph()
    _make_fast_static_layer_update()
    global _LB_CAPTURING_PREFILL
    with torch_npu.npu.stream(capture_stream):
        with torch_npu.npu.graph(g, stream=capture_stream):
            with torch.inference_mode():
                for t in cum_tensors:
                    t.fill_(write_base)
                _LB_CAPTURING_PREFILL = True
                try:
                    text_out = text_model(
                        input_ids=static_ids,
                        attention_mask=None,
                        position_ids=None,
                        past_key_values=cache,
                        use_cache=True,
                    )
                finally:
                    _LB_CAPTURING_PREFILL = False
                static_hidden = text_out.last_hidden_state
                if lm_head_weight is not None:
                    static_logits_last = model.lm_head(static_hidden[:, -1:, :])
                else:
                    static_logits_last = None
    capture_stream.synchronize()
    torch_npu.npu.synchronize()
    _tp_barrier()

    max_cache_len = cache.max_cache_len if hasattr(cache, "max_cache_len") else cache.get_max_cache_len()
    if max_cache_len > write_base + bucket_size:
        _kv_cache_zero_range(cache, write_base + bucket_size, max_cache_len)

    return {
        "graph": g,
        "static_ids": static_ids,
        "static_hidden": static_hidden,
        "static_logits_last": static_logits_last,
        "lm_head": model.lm_head if lm_head_weight is not None else None,
        "stream": capture_stream,
        "bucket_size": bucket_size,
        "write_base": write_base,
        "cum_tensors": cum_tensors,
    }


def _replay_prefill_graph(prefill_graph_entry, input_ids, prompt_len, cache):
    """Replay a captured prefill graph for the given input."""
    bucket_size = prefill_graph_entry["bucket_size"]
    write_base = prefill_graph_entry.get("write_base", 0)
    static_ids = prefill_graph_entry["static_ids"]
    g = prefill_graph_entry["graph"]
    stream = prefill_graph_entry["stream"]
    _timing_on = os.environ.get("LB_PREFILL_TIMING", "0") == "1"

    _t = [time.time()]
    static_ids[:, :prompt_len] = input_ids[0, :prompt_len]
    stream.wait_stream(torch_npu.npu.current_stream())
    _t.append(time.time())

    g.replay()
    _t.append(time.time())

    if prompt_len != bucket_size:
        with torch_npu.npu.stream(stream):
            with torch.inference_mode():
                for t in prefill_graph_entry["cum_tensors"]:
                    t.fill_(write_base + prompt_len)
                _kv_cache_zero_range(cache, write_base + prompt_len, write_base + bucket_size)
    _t.append(time.time())

    stream.synchronize()
    _t.append(time.time())

    if prompt_len == bucket_size and prefill_graph_entry.get("static_logits_last") is not None:
        logits = prefill_graph_entry["static_logits_last"]
    else:
        with torch.inference_mode():
            hidden_last = prefill_graph_entry["static_hidden"][:, prompt_len - 1 : prompt_len, :]
            logits = prefill_graph_entry["lm_head"](hidden_last)
    _t.append(time.time())

    if _timing_on:
        import sys as _sys_rp
        _lbl = ["copy_ids+wait", "replay_launch", "post_fix", "sync", "lm_head"]
        _s = " | ".join(f"{_lbl[i]}={(_t[i+1]-_t[i])*1000:.2f}ms" for i in range(len(_lbl)))
        print(f"[prefill-replay] total={(_t[-1]-_t[0])*1000:.2f}ms  {_s}", file=_sys_rp.stderr)
    return logits


def prefill_warmup(model, input_ids, attention_mask):
    """Run one throwaway prefill to absorb first-call lazy-init / aclnn compile."""
    text_model = model.model if hasattr(model, "model") else model
    with torch.inference_mode():
        _t_out = text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        _lm = getattr(model, "lm_head", None)
        if _lm is not None:
            _ = _lm(_t_out.last_hidden_state[:, -1:, :])
    torch_npu.npu.synchronize()

_SUFFIX_PREFILL_GRAPHS: dict = {}


def prefill_prefix(model, prefix_ids, device="npu:0", timing=None):
    """Prefill the shared prefix ONCE and return its K/V store."""
    if os.environ.get("LB_PREFILL_GRAPH", "0") != "1":
        raise RuntimeError("prefill_prefix requires LB_PREFILL_GRAPH=1")

    prefix_len = prefix_ids.shape[1]
    bucket = _select_prefill_bucket(prefix_len)
    if bucket is None:
        raise ValueError(
            f"prefix_len {prefix_len} exceeds the largest prefill bucket "
            f"{max(PREFILL_BUCKETS)}"
        )

    pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 151645

    max_cache_len = max(prefix_len + 64, *PREFILL_BUCKETS)
    cache = _make_static_cache(model, max_cache_len)
    _patch_text_model_for_graph(model)
    _patch_attn_for_graph(model)
    _ = patch_moe_for_graph_decode(model)
    try:
        _sync_pp = torch_npu.npu.synchronize
        _sync_pp()
        t_cap0 = time.time()
        _prefix_graphs = {}
        _prefix_graphs[bucket] = _capture_prefill_graph(
            model, bucket, device, cache, pad_token_id,
        )
        _sync_pp()
        t_repl0 = time.time()
        _logits = _replay_prefill_graph(
            _prefix_graphs[bucket], prefix_ids, prefix_len, cache,
        )
        _sync_pp()
        if timing is not None:
            timing["prefix_prefill_capture_s"] = time.time() - t_cap0
            timing["prefix_prefill_replay_s"] = time.time() - t_repl0
        layers_store = []
        with torch.inference_mode():
            for layer in cache.layers:
                if not getattr(layer, "_lb_bsnd", False):
                    raise RuntimeError(
                        "common prefix requires the BSND static cache "
                        "(LB_CACHE_BSND=1, the default)"
                    )
                layers_store.append({
                    "k": layer.keys[:, :prefix_len, :, :].clone(),
                    "v": layer.values[:, :prefix_len, :, :].clone(),
                })
        _sync_pp()
        del _logits, cache
    finally:
        unpatch_moe_after_graph_decode(model)
        _unpatch_attn_after_graph(model)
        _unpatch_text_model_after_graph(model)
    return {"prefix_len": prefix_len, "bucket": bucket, "layers": layers_store}


def _seed_prefix_kv(cache, prefix_store):
    """Force-init a fresh StaticCache and copy the shared-prefix K/V in."""
    with torch.inference_mode():
        prefix_len = prefix_store["prefix_len"]
        for layer, store in zip(cache.layers, prefix_store["layers"]):
            if not layer.is_initialized:
                layer.lazy_initialization(
                    store["k"][:, :1, :, :], store["v"][:, :1, :, :],
                )
            layer.keys[:, :prefix_len, :, :].copy_(store["k"])
            layer.values[:, :prefix_len, :, :].copy_(store["v"])
            layer.cumulative_length.fill_(prefix_len)
    torch_npu.npu.synchronize()


def prefix_session_create(model, prefix_kv, max_new_tokens=256,
                          device="npu:0", timing=None):
    """Create a reusable common-prefix session for multi-request serving."""
    prefix_len = prefix_kv["prefix_len"]
    max_cache_len = prefix_len + max_new_tokens + 16
    max_bucket = max(PREFILL_BUCKETS)
    max_cache_len = max(max_cache_len, prefix_len + max_bucket)
    if os.environ.get("LB_PREFILL_GRAPH", "0") != "1":
        raise RuntimeError("prefix_session_create requires LB_PREFILL_GRAPH=1")
    if os.environ.get("LB_IFA_MASK", "0") == "0":
        raise RuntimeError(
            "common-prefix multi-request sessions require LB_IFA_MASK=2 "
            "(or 1): the default no-mask IFA would attend stale K/V left "
            "by earlier requests beyond cumulative_length"
        )
    _sync_sess = torch_npu.npu.synchronize
    _sync_sess()
    t_sess_0 = time.time()
    dummy_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
    cache = _make_static_cache(model, max_cache_len)
    _seed_prefix_kv(cache, prefix_kv)
    text_model = model.model if hasattr(model, "model") else model
    inv_freq = getattr(getattr(text_model, "rotary_emb", None), "inv_freq", None)
    _sync_sess()
    if timing is not None:
        timing["session_create_s"] = time.time() - t_sess_0
    return {
        "prefix_len": prefix_len,
        "max_cache_len": max_cache_len,
        "cache": cache,
        "suffix_graphs": {},
        "decode": None,
        "inv_freq": inv_freq,
        "device": device,
    }


def prefix_session_destroy(session):
    """Drop the session's graphs + cache (frees their device memory)."""
    if session is None:
        return
    session["suffix_graphs"].clear()
    session["decode"] = None
    session["cache"] = None


def _common_token_prefix_len(a, b) -> int:
    """Largest k such that a[:k] == b[:k] (token-id lists)."""
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    return k


class CommonPrefixSession:
    """High-level common-prefix serving handle — one-call setup for the fixed-prefix, repeated-suffix workload."""
    def __init__(self, model, prefix_ids, max_new_tokens=1,
                 device="npu:0", warmup=True, timing=None):
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._ref_len = int(prefix_ids.shape[1])
        self._ref_ids = prefix_ids[0].tolist()
        self._prefix_len = self._ref_len
        if warmup:
            prefill_warmup(model, prefix_ids,
                           prefix_ids.new_ones(()).expand_as(prefix_ids))
        self.prefix_kv = prefill_prefix(model, prefix_ids,
                                        device=device, timing=timing)
        self.session = prefix_session_create(model, self.prefix_kv,
                                             max_new_tokens=max_new_tokens,
                                             device=device, timing=timing)

    def generate(self, input_ids, attention_mask, timing=None):
        """Serve one request: input_ids [1, L] = prefix + suffix tokens, on device."""
        if input_ids.shape[1] <= self._prefix_len:
            raise ValueError(
                f"request has {input_ids.shape[1]} tokens; must be longer "
                f"than the shared prefix ({self._prefix_len}) for a non-empty "
                "suffix")
        full_ids = input_ids[0].tolist()
        k = _common_token_prefix_len(full_ids, self._ref_ids)
        suffix_len = len(full_ids) - k
        if k < 32 or suffix_len <= 0:
            raise ValueError(
                f"invalid prefix/suffix token split: common={k}, "
                f"suffix={suffix_len} of {len(full_ids)}")
        if suffix_len > max(PREFILL_BUCKETS):
            raise ValueError(
                f"suffix length {suffix_len} exceeds the largest prefill "
                f"bucket {max(PREFILL_BUCKETS)}; common-prefix decode has no "
                "eager fallback")
        if k != self._prefix_len:
            prefix_session_destroy(self.session)
            self.prefix_kv = prefill_prefix(self.model, input_ids[:, :k],
                                            device=self.device, timing=timing)
            self.session = prefix_session_create(
                self.model, self.prefix_kv,
                max_new_tokens=self.max_new_tokens, device=self.device,
                timing=timing)
            self._prefix_len = k
        return graph_generate(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            device=self.device,
            capture_graph=True,
            timing=timing,
            prefix_kv=self.prefix_kv,
            session=self.session,
        )

    def close(self):
        """Destroy the NPU prefix session and release its resources."""
        if self.session is not None:
            prefix_session_destroy(self.session)
            self.session = None


def graph_generate(
    model,
    input_ids,
    attention_mask,
    max_new_tokens: int = 256,
    device: str = "npu:0",
    num_warmup: int = 3,
    capture_graph: bool = True,
    stream=None,
    timing: dict | None = None,
    streamer=None,
    prefix_kv=None,
    session=None,
):
    """Greedy decode with optional `torch.npu.graph` capture (1p)."""
    def _sync():
        torch_npu.npu.synchronize()

    prompt_len = input_ids.shape[1]
    prefix_len = prefix_kv["prefix_len"] if prefix_kv is not None else 0
    suffix_len = prompt_len - prefix_len
    if suffix_len <= 0:
        raise ValueError(
            f"input_ids ({prompt_len} tokens) must be longer than the "
            f"common prefix ({prefix_len} tokens)"
        )
    max_cache_len = prompt_len + max_new_tokens + 16
    if os.environ.get("LB_PREFILL_GRAPH", "0") == "1":
        max_bucket = max(PREFILL_BUCKETS)
        max_cache_len = max(max_cache_len, prefix_len + max_bucket)

    generated = []

    if session is not None:
        if prefix_kv is None:
            raise ValueError("session requires prefix_kv (same prefix)")
        if prefix_kv["prefix_len"] != session["prefix_len"]:
            raise ValueError(
                f"prefix_kv length {prefix_kv['prefix_len']} does not match "
                f"session prefix length {session['prefix_len']}"
            )
        if os.environ.get("LB_GRAPH_PFA_V2", "0") == "4":
            raise RuntimeError(
                "session reuse is incompatible with LB_GRAPH_PFA_V2=4 "
                "(multi-graph PFA freezes seq-len per capture)"
            )
        cache = session["cache"]
        max_cache_len = session["max_cache_len"]
    else:
        cache = _make_static_cache(model, max_cache_len)
    _patch_text_model_for_graph(model)
    _patch_attn_for_graph(model)
    _ = patch_moe_for_graph_decode(model)

    if prefix_kv is not None and session is None:
        _sync()
        t_seed_0 = time.time()
        _seed_prefix_kv(cache, prefix_kv)
        _sync()
        if timing is not None:
            timing["prefix_copy_s"] = time.time() - t_seed_0

    use_prefill_graph = (
        os.environ.get("LB_PREFILL_GRAPH", "0") == "1"
        and capture_graph
    )
    if prefix_kv is not None and not use_prefill_graph:
        raise RuntimeError(
            "common-prefix mode requires LB_PREFILL_GRAPH=1 with graph capture "
            "(suffix prefill needs the bucketed PFA prefill graph)"
        )
    prefill_bucket = (
        _select_prefill_bucket(suffix_len if prefix_kv is not None else prompt_len)
        if use_prefill_graph else None
    )
    if prefix_kv is not None and prefill_bucket is None:
        raise RuntimeError(
            f"suffix length {suffix_len} exceeds the largest prefill bucket "
            f"{max(PREFILL_BUCKETS)}; common-prefix mode has no eager fallback"
        )
    use_prefill_graph = use_prefill_graph and prefill_bucket is not None

    if use_prefill_graph:
        pad_token_id = getattr(model.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(model.config, "eos_token_id", None)
        if pad_token_id is None:
            tok = getattr(getattr(model, "tokenizer", None), "pad_token_id", None)
            if tok is not None:
                pad_token_id = tok
            else:
                pad_token_id = 151645

        if prefix_kv is None:
            _graphs_dict, _graph_key = _PREFILL_GRAPHS, prefill_bucket
            _capture_kwargs = {}
        elif session is not None:
            _graphs_dict, _graph_key = session["suffix_graphs"], prefill_bucket
            _capture_kwargs = {
                "write_base": prefix_len,
                "prefix_kv": prefix_kv["layers"],
            }
        else:
            _graphs_dict = _SUFFIX_PREFILL_GRAPHS
            _graph_key = (prefill_bucket, prefix_len)
            _capture_kwargs = {
                "write_base": prefix_len,
                "prefix_kv": prefix_kv["layers"],
            }

        _sync()
        t_prefill_cap_0 = time.time()
        if _graph_key not in _graphs_dict:
            if timing is not None:
                timing["prefill_capture_first"] = True
            _graphs_dict[_graph_key] = _capture_prefill_graph(
                model, prefill_bucket, device, cache, pad_token_id,
                **_capture_kwargs,
            )
        _sync()
        t_prefill_cap = time.time() - t_prefill_cap_0
        if timing is not None:
            timing["prefill_capture"] = t_prefill_cap

        _sync()
        t_prefill_0 = time.time()
        logits_pf = _replay_prefill_graph(
            _graphs_dict[_graph_key],
            input_ids if prefix_kv is None else input_ids[:, prefix_len:],
            prompt_len if prefix_kv is None else suffix_len,
            cache,
        )
        _sync()
        t_prefill = time.time() - t_prefill_0
        if timing is not None:
            timing["prefill"] = t_prefill
            timing["prefill_bucket"] = prefill_bucket
        next_tok = logits_pf[:, -1:, :].argmax(dim=-1)
        generated.append(next_tok.item())
        if streamer is not None:
            streamer.put(next_tok.item())
    else:
        _sync()
        t_prefill_0 = time.time()
        _pf_text_model = model.model if hasattr(model, "model") else model
        with torch.inference_mode():
            _pf_out = _pf_text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            logits_pf = model.lm_head(
                _pf_out.last_hidden_state[:, -1:, :]
            )
        _sync()
        t_prefill = time.time() - t_prefill_0
        if timing is not None:
            timing["prefill"] = t_prefill
        next_tok = logits_pf[:, -1:, :].argmax(dim=-1)
        generated.append(next_tok.item())
        if streamer is not None:
            streamer.put(next_tok.item())

    remaining = max_new_tokens - 1
    if remaining <= 0:
        unpatch_moe_after_graph_decode(model)
        _unpatch_attn_after_graph(model)
        _unpatch_text_model_after_graph(model)
        if streamer is not None:
            streamer.end()
        return generated

    text_model = model.model if hasattr(model, "model") else model
    num_layers = len(text_model.layers)
    reset_static_masks(prompt_len, max_cache_len, device, num_layers=num_layers)

    if not capture_graph:
        _sync()
        t_loop_0 = time.time()

        prof_enabled = os.environ.get("LB_PROFILE_DECODE", "0") == "1"
        prof_steps = int(os.environ.get("LB_PROFILE_DECODE_STEPS", "5"))
        prof_path = os.environ.get("LB_PROFILE_OUT", None)
        prof_ctx = None
        if prof_enabled:
            if prof_path is None:
                prof_path = f"./prof_decode_{int(time.time())}"
            logger.info("%s", f"[prof] decode-only profiling enabled, out={prof_path}, steps={prof_steps}")
            from torch_npu.profiler import profile, ProfilerActivity
            prof_ctx = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=prof_steps, repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_path),
            )
            prof_ctx.__enter__()

        for _ in range(remaining):
            past_len = int(cache.get_seq_length())
            position_ids = _decode_position_ids(past_len, 1, device)
            with torch.inference_mode():
                out = model(
                    input_ids=next_tok,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            generated.append(next_tok.item())
            if streamer is not None:
                streamer.put(next_tok.item())
            if prof_ctx is not None:
                prof_ctx.step()
        if prof_ctx is not None:
            prof_ctx.__exit__(None, None, None)
            logger.info("%s", f"[prof] decode profiling finished, data at {prof_path}")
        _sync()
        t_loop = time.time() - t_loop_0
        if timing is not None:
            timing["decode_loop"] = t_loop
            timing["decode_per_tok_ms"] = t_loop / remaining * 1000
        unpatch_moe_after_graph_decode(model)
        _unpatch_attn_after_graph(model)
        _unpatch_text_model_after_graph(model)
        if streamer is not None:
            streamer.end()
        return generated

    _session_reuse_decode = (
        session is not None and session["decode"] is not None
    )
    if _session_reuse_decode:
        _de = session["decode"]
        static_input_ids = _de["static_input_ids"]
        static_position_ids = _de["static_position_ids"]
        capture_stream = _de["stream"]
        g = _de["graph"]
        static_logits = _de["static_logits"]
        static_next_tok = _de.get("static_next_tok")
        _qkv_inv_freq = session["inv_freq"]
        global _LB_DECODE_QKVROPE_STATIC
        _LB_DECODE_QKVROPE_STATIC = _de["qkvrope_static"]
        use_multi_graph_pfa = False
        multi_graphs = None
        multi_graph_static_logits = None
        if timing is not None:
            timing["warmup"] = 0.0
            timing["capture"] = 0.0
            timing["decode_graph_reused"] = True
        static_input_ids.copy_(next_tok)
        past_len = int(cache.get_seq_length())
        static_position_ids.copy_(_decode_position_ids(past_len, 1, device))
        if _LB_DECODE_QKVROPE_STATIC:
            _refresh_qkvrope_decode_rows(_qkv_inv_freq, past_len, device)

    if not _session_reuse_decode:
        static_input_ids = torch.zeros(1, 1, dtype=input_ids.dtype, device=device)
        static_input_ids.copy_(next_tok)

        static_position_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        past_len = int(cache.get_seq_length())
        static_position_ids.copy_(_decode_position_ids(past_len, 1, device))

        _LB_DECODE_QKVROPE_STATIC = (
            os.environ.get("LB_DECODE_QKVROPE", "1") == "1"
        )
        _qkv_inv_freq = None
        if _LB_DECODE_QKVROPE_STATIC:
            _tm_rows = model.model if hasattr(model, "model") else model
            _qkv_inv_freq = getattr(getattr(_tm_rows, "rotary_emb", None), "inv_freq", None)
            _refresh_qkvrope_decode_rows(_qkv_inv_freq, past_len, device)

        capture_stream = stream or torch_npu.npu.Stream(device=device)

        text_model_warmup = model.model if hasattr(model, "model") else model

        text_model = text_model_warmup
        lm_head_chunks = 8
        lm_head_weight = model.lm_head.weight
        lm_head_bias = getattr(model.lm_head, "bias", None)
        _vp_world = getattr(model.lm_head, "_vp_world", 1)
        if _vp_world > 1:
            import torch.distributed as dist

        def _chunked_lm_head(hidden, weight, bias=None, chunks=lm_head_chunks):
            vocab_size = weight.shape[0]
            chunk_v = (vocab_size + chunks - 1) // chunks
            outs = []
            for i in range(chunks):
                v_lo = i * chunk_v
                v_hi = min(v_lo + chunk_v, vocab_size)
                w_slice = weight[v_lo:v_hi, :]
                b_slice = bias[v_lo:v_hi] if bias is not None else None
                outs.append(torch.nn.functional.linear(hidden, w_slice, b_slice))
            out = torch.cat(outs, dim=-1)
            if _vp_world > 1:
                parts = [torch.empty_like(out) for _ in range(_vp_world)]
                dist.all_gather(parts, out)
                out = torch.cat(parts, dim=-1)
            return out

        _sync()
        t_warmup_0 = time.time()
        with torch_npu.npu.stream(capture_stream):
            for _ in range(num_warmup):
                with torch.inference_mode():
                    _t_out = text_model_warmup(
                        input_ids=static_input_ids,
                        attention_mask=None,
                        position_ids=static_position_ids,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    _ = _chunked_lm_head(
                        _t_out.last_hidden_state[:, -1:, :],
                        lm_head_weight, lm_head_bias, lm_head_chunks,
                    )
        capture_stream.synchronize()
        _sync()
        t_warmup = time.time() - t_warmup_0
        if timing is not None:
            timing["warmup"] = t_warmup
        _tp_barrier()

        use_multi_graph_pfa = os.environ.get("LB_GRAPH_PFA_V2", "0") == "4"
        multi_graphs = None
        multi_graph_static_logits = None
        static_next_tok = None
        if use_multi_graph_pfa:
            global _LB_DECODE_SEQ_LEN_KV
            multi_graphs = []
            multi_graph_static_logits = []

            with torch.inference_mode():
                for layer in cache.layers:
                    if hasattr(layer, "cumulative_length") and torch.is_tensor(layer.cumulative_length):
                        layer.cumulative_length.copy_(
                            torch.tensor([prompt_len], dtype=layer.cumulative_length.dtype,
                                         device=layer.cumulative_length.device).reshape(layer.cumulative_length.shape))

            _sync()
            t_capture_0 = time.time()
            for i in range(remaining):
                target_len = prompt_len + i
                with torch.inference_mode():
                    for layer in cache.layers:
                        if hasattr(layer, "cumulative_length") and torch.is_tensor(layer.cumulative_length):
                            layer.cumulative_length.copy_(
                                torch.tensor([target_len], dtype=layer.cumulative_length.dtype,
                                             device=layer.cumulative_length.device)
                    .reshape(layer.cumulative_length.shape))
                _LB_DECODE_SEQ_LEN_KV = prompt_len + i + 1
                g_i = torch_npu.npu.NPUGraph()
                with torch_npu.npu.graph(g_i, stream=capture_stream):
                    with torch.inference_mode():
                        text_out = text_model(
                            input_ids=static_input_ids,
                            attention_mask=None,
                            position_ids=static_position_ids,
                            past_key_values=cache,
                            use_cache=True,
                        )
                        static_hidden_i = text_out.last_hidden_state
                        static_logits_i = _chunked_lm_head(
                            static_hidden_i, lm_head_weight, lm_head_bias, lm_head_chunks)
                multi_graphs.append(g_i)
                multi_graph_static_logits.append(static_logits_i)
            _sync()
            t_capture = time.time() - t_capture_0
            if timing is not None:
                timing["capture"] = t_capture
        else:
            _sync()
            t_capture_0 = time.time()
            g = torch_npu.npu.NPUGraph()
            _argmax_in_graph = (
                os.environ.get("LB_DECODE_ARGMAX_IN_GRAPH", "1") == "1"
            )
            static_next_tok = None
            with torch_npu.npu.graph(g, stream=capture_stream):
                with torch.inference_mode():
                    text_out = text_model(
                        input_ids=static_input_ids,
                        attention_mask=None,
                        position_ids=static_position_ids,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    static_hidden = text_out.last_hidden_state
                    static_logits = _chunked_lm_head(static_hidden, lm_head_weight, lm_head_bias, lm_head_chunks)
                    if _argmax_in_graph:
                        static_next_tok = static_logits[:, -1:, :].argmax(dim=-1)
            _sync()
            t_capture = time.time() - t_capture_0
            if timing is not None:
                timing["capture"] = t_capture

            if session is not None:
                session["decode"] = {
                    "graph": g,
                    "static_input_ids": static_input_ids,
                    "static_position_ids": static_position_ids,
                    "static_logits": static_logits,
                    "static_next_tok": static_next_tok,
                    "stream": capture_stream,
                    "qkvrope_static": _LB_DECODE_QKVROPE_STATIC,
                }

    if not use_multi_graph_pfa:
        with torch.inference_mode():
            for layer in cache.layers:
                if hasattr(layer, "cumulative_length") and torch.is_tensor(layer.cumulative_length):
                    layer.cumulative_length.fill_(prompt_len)
            _kv_cache_zero_range(cache, prompt_len)
            reset_static_masks(prompt_len, max_cache_len, device, num_layers=num_layers)
        _sync()
        _tp_barrier()

    _sync()
    t_replay_0 = time.time()
    t_copy_total = 0.0
    t_replay_only = 0.0
    t_argmax_total = 0.0
    t_lmhead_total = 0.0
    t_item_total = 0.0

    prof_enabled = os.environ.get("LB_PROFILE_GRAPH_DECODE", "0") == "1"
    prof_steps = int(os.environ.get("LB_PROFILE_DECODE_STEPS", "5"))
    prof_path = os.environ.get("LB_PROFILE_OUT", None)
    prof_ctx = None
    if prof_enabled:
        if prof_path is None:
            prof_path = f"./prof_graph_decode_{int(time.time())}"
        logger.info("%s", f"[prof] graph decode profiling enabled, out={prof_path}, steps={prof_steps}")
        from torch_npu.profiler import profile, ProfilerActivity
        prof_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=prof_steps, repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_path),
        )
        prof_ctx.__enter__()

    _use_argmax_graph = (
        not use_multi_graph_pfa and static_next_tok is not None
    )
    _host_len = (
        not use_multi_graph_pfa
        and os.environ.get("LB_DECODE_HOST_LEN", "1") == "1"
    )
    for _ in range(remaining):
        past_len = prompt_len + _ if _host_len else int(cache.get_seq_length())
        new_pos = _decode_position_ids(past_len, 1, device)
        t_c0 = time.time()
        static_input_ids.copy_(next_tok)
        static_position_ids.copy_(new_pos)
        if _LB_DECODE_QKVROPE_STATIC:
            _refresh_qkvrope_decode_rows(
                _qkv_inv_freq, past_len, device
            )
        if use_multi_graph_pfa:
            with torch.inference_mode():
                for layer in cache.layers:
                    if hasattr(layer, "cumulative_length") and torch.is_tensor(layer.cumulative_length):
                        layer.cumulative_length.copy_(
                            torch.tensor([past_len], dtype=layer.cumulative_length.dtype,
                                         device=layer.cumulative_length.device).reshape(layer.cumulative_length.shape))
        t_copy_total += time.time() - t_c0
        t_r0 = time.time()
        if use_multi_graph_pfa:
            multi_graphs[_].replay()
            current_logits = multi_graph_static_logits[_]
        else:
            g.replay()
            current_logits = static_logits
        capture_stream.synchronize()
        torch_npu.npu.synchronize()
        t_replay_only += time.time() - t_r0
        t_a0 = time.time()
        if _use_argmax_graph:
            next_tok = static_next_tok
        else:
            with torch.inference_mode():
                next_tok = current_logits[:, -1:, :].argmax(dim=-1)
            torch_npu.npu.synchronize()
        t_argmax_total += time.time() - t_a0
        t_i0 = time.time()
        generated.append(next_tok.item())
        if streamer is not None:
            streamer.put(next_tok.item())
        t_item_total += time.time() - t_i0
        if prof_ctx is not None and _ == prof_steps - 1:
            prof_ctx.__exit__(None, None, None)
            prof_ctx = None
    if prof_ctx is not None:
        prof_ctx.__exit__(None, None, None)
    _sync()
    t_replay = time.time() - t_replay_0
    if timing is not None:
        timing["decode_replay"] = t_replay
        timing["decode_per_tok_ms"] = t_replay / remaining * 1000
        timing["decode_copy_ms"] = t_copy_total / remaining * 1000
        timing["decode_replay_only_ms"] = t_replay_only / remaining * 1000
        timing["decode_argmax_ms"] = t_argmax_total / remaining * 1000
        timing["decode_lmhead_ms"] = t_lmhead_total / remaining * 1000
        timing["decode_item_ms"] = t_item_total / remaining * 1000

    _LB_DECODE_QKVROPE_STATIC = False
    unpatch_moe_after_graph_decode(model)
    _unpatch_attn_after_graph(model)
    _unpatch_text_model_after_graph(model)

    if streamer is not None:
        streamer.end()
    return generated
