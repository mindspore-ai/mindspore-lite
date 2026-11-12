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
"""Qwen3-30B-A3B LiteBoost adapter: full-attention forward, MoE+TP sharding."""
from __future__ import annotations

import logging
import os
import types
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from lite_boost.parallel import (
    get_hcom_name,
    is_distributed_active,
    shard_colwise,
    shard_rowwise,
)

logger = logging.getLogger(__name__)


class _RowParallelLinear(nn.Module):
    """Wrap a row-sharded `nn.Linear` so its forward fuses matmul + HCCL all-reduce."""

    def __init__(self, linear: nn.Linear, rank: int, world: int):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.rank = rank
        self.world = world
        self._hcom = None
        self._x2 = None

    def forward(self, x):
        """Row-parallel linear: local matmul then all-reduce across TP ranks."""
        if self.world <= 1:
            return F.linear(x, self.weight, self.bias)
        bias = self.bias if self.rank == 0 else None
        if os.environ.get("LB_TP_SEPARATE_MM_AR") == "1":
            partial = F.linear(x, self.weight, bias)
            dist.all_reduce(partial, op=dist.ReduceOp.SUM)
            return partial
        if self._x2 is None:
            self._x2 = self.weight.transpose(0, 1).contiguous()
        if self._hcom is None and is_distributed_active():
            self._hcom = get_hcom_name()
        import torch_npu
        if bias is not None:
            return torch_npu.npu_mm_all_reduce_base(x, self._x2, self._hcom, bias=bias)
        return torch_npu.npu_mm_all_reduce_base(x, self._x2, self._hcom)


def _fused_rms_norm_forward(self, x):
    """Fused RMSNorm forward using `torch_npu.npu_rms_norm` (single aclnn op)."""
    import torch_npu
    out, _ = torch_npu.npu_rms_norm(x, self._fused_gamma, epsilon=self.variance_epsilon)
    return out


def _patch_rms_norm(rms_norm_module):
    """Patch a single `Qwen3MoeRMSNorm` instance to use the fused forward."""
    with torch.no_grad():
        gamma = rms_norm_module.weight.float().contiguous()
    rms_norm_module.register_buffer("_fused_gamma", gamma, persistent=False)
    rms_norm_module.forward = types.MethodType(_fused_rms_norm_forward, rms_norm_module)


def _fused_ffn_forward(self, x):
    """Fused dense FFN forward: gate_proj + up_proj → single MatMul, silu+mul → npu_swiglu."""
    import torch_npu
    fused = torch.nn.functional.linear(x, self._fused_gate_up_weight)
    act = torch_npu.npu_swiglu(fused, dim=-1)
    return self.down_proj(act)


def _patch_ffn(mlp_module):
    """Patch a single dense `Qwen3MoeMLP` to use the fused FFN forward."""
    with torch.no_grad():
        fused_w = torch.cat(
            [mlp_module.gate_proj.weight, mlp_module.up_proj.weight], dim=0
        ).contiguous()
    mlp_module.register_buffer("_fused_gate_up_weight", fused_w, persistent=False)
    mlp_module.forward = types.MethodType(_fused_ffn_forward, mlp_module)


def _fused_residual_layer_forward(self, hidden_states, attention_mask=None,
                                  position_ids=None, past_key_values=None,
                                  use_cache=False, position_embeddings=None,
                                  **kwargs):
    """Decoder-layer forward fusing the mid-layer residual add + post-norm."""
    import torch_npu
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states, _, residual = torch_npu.npu_add_rms_norm(
        residual, hidden_states, self._lb_paln_gamma, self._lb_paln_eps,
    )
    hidden_states = self.mlp(hidden_states)
    return residual + hidden_states


def _patch_residual_layer_forward(layer):
    """Patch one `Qwen3MoeDecoderLayer` to `_fused_residual_layer_forward`."""
    if getattr(layer, "_lb_paln_gamma", None) is not None:
        return False
    try:
        import torch_npu
        if not hasattr(torch_npu, "npu_add_rms_norm"):
            return False
    except ImportError:
        return False
    post_ln = layer.post_attention_layernorm
    with torch.no_grad():
        gamma = post_ln.weight.to(torch.bfloat16).contiguous()
    layer.register_buffer("_lb_paln_gamma", gamma, persistent=False)
    layer._lb_paln_eps = float(post_ln.variance_epsilon)
    layer.forward = types.MethodType(_fused_residual_layer_forward, layer)
    return True

_CAUSAL_MASK_CACHE: dict = {}


def _get_causal_mask_2048(device):
    """Return the cached 2048×2048 bool upper-triangular causal mask."""
    key = str(device)
    cached = _CAUSAL_MASK_CACHE.get(key)
    if cached is None:
        mask2048 = torch.triu(
            torch.ones(2048, 2048, dtype=torch.bool, device=device), diagonal=1
        )
        cached = mask2048
        _CAUSAL_MASK_CACHE[key] = cached
    return cached


def qwen3_30B_full_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    **kwargs,
):  # pylint: disable=unused-argument
    """Drop-in `Qwen3MoeAttention.forward` using `torch_npu.npu_fusion_attention` (causal, sparse_mode=3)."""
    import torch_npu
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        apply_rotary_pos_emb,
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    q_bsnd = query_states.transpose(1, 2)
    k_bsnd = key_states.transpose(1, 2)
    v_bsnd = value_states.transpose(1, 2)

    causal_mask = _get_causal_mask_2048(q_bsnd.device)

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


def qwen3_30B_eager_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    **kwargs,
):  # pylint: disable=unused-argument
    """Debug forward: use HF's eager_attention_forward instead of flash_attention."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        apply_rotary_pos_emb, eager_attention_forward,
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attn_output, _ = eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=self.scaling,
        dropout=0.0,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def _shard_experts_intra_tp(experts: nn.Module, rank: int, world: int):
    """Intra-expert tensor-parallel shard of `Qwen3MoeExperts`."""
    inter = experts.intermediate_dim
    assert inter % world == 0, (
        f"MoE intermediate_size ({inter}) must be divisible by world_size ({world}) "
        f"for intra-expert TP. Use a different shard strategy or a different world size."
    )
    inter_r = inter // world
    gup = experts.gate_up_proj.data
    gate_full = gup[:, :inter, :]
    up_full = gup[:, inter:2 * inter, :]
    gate_r = gate_full[:, rank * inter_r:(rank + 1) * inter_r, :]
    up_r = up_full[:, rank * inter_r:(rank + 1) * inter_r, :]
    new_gup = torch.cat([gate_r, up_r], dim=1).contiguous()
    experts.gate_up_proj = nn.Parameter(new_gup, requires_grad=False)
    dp = experts.down_proj.data
    new_dp = dp[:, :, rank * inter_r:(rank + 1) * inter_r].contiguous()
    experts.down_proj = nn.Parameter(new_dp, requires_grad=False)
    experts.intermediate_dim = inter_r
    return experts


class _MoEAllReduce(nn.Module):
    """Wrap a sharded `Qwen3MoeSparseMoeBlock` so the MoE block's output is all-reduced across TP ranks."""

    def __init__(self, moe_block: nn.Module, rank: int, world: int):
        super().__init__()
        self._moe = moe_block
        self.rank = rank
        self.world = world
        self._hcom = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """TP-wrapped MoE forward: local experts then all-reduce the output."""
        local_out = self._moe(hidden_states)
        if self.world <= 1:
            return local_out
        if is_distributed_active():
            dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
        return local_out


def shard_qwen3_30B_tp(model, rank: int, world: int):
    """In-place tensor-parallel shard of a `Qwen3MoeForCausalLM`."""
    text_model = _get_text_model(model)
    layers = text_model.layers

    for i, layer in enumerate(layers):
        attn = layer.self_attn
        n_heads = attn.config.num_attention_heads
        n_kv = attn.config.num_key_value_heads
        assert n_heads % world == 0 and n_kv % world == 0, (
            f"num_attention_heads ({n_heads}) and num_key_value_heads ({n_kv}) "
            f"must be divisible by world_size ({world})."
        )
        shard_colwise(attn.q_proj, rank, world)
        shard_colwise(attn.k_proj, rank, world)
        shard_colwise(attn.v_proj, rank, world)
        shard_rowwise(attn.o_proj, rank, world)
        attn.o_proj = _RowParallelLinear(attn.o_proj, rank, world)

        mlp = layer.mlp
        if isinstance(mlp, _MoEAllReduce):
            continue
        inner = getattr(mlp, "_moe", mlp)
        if hasattr(inner, "experts") and hasattr(inner, "gate"):
            _ = _shard_experts_intra_tp(inner.experts, rank, world)
            layer.mlp = _MoEAllReduce(inner, rank, world)
        elif hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
            shard_colwise(mlp.gate_proj, rank, world)
            shard_colwise(mlp.up_proj, rank, world)
            shard_rowwise(mlp.down_proj, rank, world)
            mlp.down_proj = _RowParallelLinear(mlp.down_proj, rank, world)
        else:
            raise TypeError(
                f"Unexpected MLP type at layer {i}: {type(mlp).__name__}; "
                f"expected a MoE block (experts+gate) or a dense MLP."
            )

    if os.environ.get("LB_TP_VOCAB_PARALLEL", "0") == "1":
        _shard_vocab_parallel(model, rank, world)
    return model


def _shard_vocab_parallel(model, rank: int, world: int):
    """Vocab-parallel embed_tokens + lm_head (LB_TP_VOCAB_PARALLEL=1)."""
    text_model = _get_text_model(model)
    emb = text_model.embed_tokens
    lm = getattr(model, "lm_head", None)
    vocab_size = emb.weight.shape[0]
    if vocab_size % world != 0:
        raise ValueError(
            f"vocab_size {vocab_size} not divisible by TP world {world}; "
            "LB_TP_VOCAB_PARALLEL cannot shard it."
        )
    v_per = vocab_size // world
    v_lo, v_hi = v_per * rank, v_per * (rank + 1)
    if rank == 0:
        _saved = 2 * v_per * emb.weight.shape[1] * 2 / 1e9
        logger.info("%s", f"  vocab_parallel=True (embed+lm_head vocab rows [{v_lo}:{v_hi}] "
              f"of {vocab_size}; ~{_saved:.2f}GB saved/rank)")

    emb.weight = nn.Parameter(
        emb.weight.data[v_lo:v_hi, :].contiguous(), requires_grad=False
    )
    emb._vp_offset = v_lo
    emb._vp_size = v_per
    pidx = getattr(emb, "padding_idx", None)
    emb._vp_padding_idx = pidx - v_lo if (pidx is not None and v_lo <= pidx < v_hi) else None

    def _vp_embed_forward(self, input_ids):
        local = (input_ids - self._vp_offset).clamp_(0, self._vp_size - 1)
        own = (input_ids >= self._vp_offset) & (
            input_ids < self._vp_offset + self._vp_size
        )
        out = F.embedding(local, self.weight, padding_idx=self._vp_padding_idx)
        out = out * own.to(out.dtype).unsqueeze(-1)
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    emb.forward = types.MethodType(_vp_embed_forward, emb)

    if lm is not None:
        lm.weight = nn.Parameter(
            lm.weight.data[v_lo:v_hi, :].contiguous(), requires_grad=False
        )
        if lm.bias is not None:
            lm.bias = nn.Parameter(
                lm.bias.data[v_lo:v_hi].contiguous(), requires_grad=False
            )
        lm._vp_world = world

        def _vp_lm_head_forward(self, hidden):
            part = F.linear(hidden, self.weight, self.bias)
            if self._vp_world <= 1:
                return part
            parts = [torch.empty_like(part) for _ in range(self._vp_world)]
            dist.all_gather(parts, part)
            return torch.cat(parts, dim=-1)

        lm.forward = types.MethodType(_vp_lm_head_forward, lm)
    return model


def _get_text_model(model):
    """Locate the text decoder inside a Qwen3Moe model."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner
    if hasattr(model, "layers"):
        return model
    raise AttributeError(
        f"Could not locate the text decoder on {type(model).__name__}; "
        f"expected .model.layers (Qwen3MoeForCausalLM) or .layers (Qwen3MoeModel)."
    )


def _transpose_param_1_2(module, attr):
    """Physically transpose a [E, a, b] parameter to [E, b, a] in place."""
    w = getattr(module, attr)
    try:
        new_w = w.data.transpose(1, 2).contiguous()
    except RuntimeError as e:
        if "out of memory" not in str(e):
            raise
        dev = w.device
        w_cpu = w.data.to("cpu")
        setattr(module, attr, None)
        del w
        new_w = w_cpu.transpose(1, 2).contiguous()
        del w_cpu
        new_w = new_w.to(dev)
    setattr(module, attr, nn.Parameter(new_w, requires_grad=False))


def patch_fused_moe(text_model, *, mode="cann", decode_grouped=True,
                    fuse_router=False, owner="boost"):
    """Patch every MoE experts module to a fused CANN forward."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeExperts,
        Qwen3MoeTopKRouter,
    )
    from lite_boost.layers.moe import (
        install_fused_experts_forward, install_fused_router_forward,
    )
    import torch_npu

    repack = os.environ.get("LB_MOE_PACKED_LAYOUT", "0") == "1"
    repacked = 0

    patched = routers = 0
    for layer in text_model.layers:
        mlp = getattr(layer, "mlp", None)
        inner_mlp = getattr(mlp, "_moe", mlp)
        experts = getattr(inner_mlp, "experts", None)
        if experts is not None and isinstance(experts, Qwen3MoeExperts):
            if (repack and not getattr(experts, "_lb_moe_weights_x_in_out", False)):
                _transpose_param_1_2(experts, "gate_up_proj")
                _transpose_param_1_2(experts, "down_proj")
                experts._lb_moe_weights_x_in_out = True
                repacked += 1
            if install_fused_experts_forward(
                experts, mode=mode, decode_grouped=decode_grouped, owner=owner,
            ):
                patched += 1
        if fuse_router:
            gate = getattr(inner_mlp, "gate", None)
            if gate is not None and isinstance(gate, Qwen3MoeTopKRouter):
                if install_fused_router_forward(gate, owner=owner):
                    routers += 1
    if repacked:
        torch_npu.npu.empty_cache()
        logger.info("%s", f"  patch_fused_moe: repacked {repacked} layers to [E, in, out]")
    return patched, routers
