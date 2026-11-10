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
"""Reusable fused-MoE forward paths for Ascend 910B NPU (CANN 9.x aclnn MoE ops)."""
from __future__ import annotations

import types

import torch
import torch_npu

__all__ = [
    "cann_moe_forward",
    "sort_moe_forward",
    "bmm_moe_forward",
    "fused_router_forward",
    "make_fused_experts_forward",
    "install_fused_experts_forward",
    "restore_experts_forward",
    "install_fused_router_forward",
    "restore_fused_router_forward",
]

_TRIU_CACHE: dict = {}


def _glist_cumsum() -> bool:
    """Legacy cumsum-mode group_list (LB_MOE_GLIST_CUMSUM=1); default count mode."""
    import os
    return os.environ.get("LB_MOE_GLIST_CUMSUM", "0") == "1"


def _fast_cumsum_int64(cnt: torch.Tensor) -> torch.Tensor:
    """Bit-exact int64 prefix sum of int tensor `cnt` via fp32 cube matmul."""
    num_experts = cnt.shape[0]
    key = (str(cnt.device), num_experts)
    tri = _TRIU_CACHE.get(key)
    if tri is None:
        tri = torch.triu(torch.ones(num_experts, num_experts, dtype=torch.float32, device=cnt.device))
        _TRIU_CACHE[key] = tri
    return (cnt.to(torch.float32).unsqueeze(0) @ tri).to(torch.int64).squeeze(0)


def cann_moe_forward(hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj, *,
                     act_fn=None, weights_x_in_out=False, tuning_config=None):
    """Fully-fused CANN MoE flow (aclnnMoeInitRoutingV2 + GroupedMatmul + MoeTokenUnpermute)."""
    num_tokens = hidden_states.shape[0]
    num_experts = gate_up_proj.shape[0]
    top_k = top_k_index.shape[-1]
    permuted, slot_to_pos, expert_cnt, _ = torch_npu.npu_moe_init_routing_v2(
        hidden_states, top_k_index.to(torch.int32),
        active_num=num_tokens * top_k, expert_capacity=0, expert_num=num_experts,
        drop_pad_mode=0, expert_tokens_num_type=1, expert_tokens_num_flag=True,
        quant_mode=-1, active_expert_range=[0, num_experts], row_idx_type=0,
    )
    if _glist_cumsum():
        gl = _fast_cumsum_int64(expert_cnt)
        glt = 0
    else:
        gl = expert_cnt.to(torch.int64) if expert_cnt.dtype != torch.int64 else expert_cnt
        glt = 1
    gu_w = gate_up_proj if weights_x_in_out else gate_up_proj.transpose(1, 2)
    dn_w = down_proj if weights_x_in_out else down_proj.transpose(1, 2)
    gu_out = torch.ops.npu.npu_grouped_matmul(
        [permuted], [gu_w],
        group_list=gl, group_type=0, split_item=3, group_list_type=glt,
        tuning_config=tuning_config,
    )[0]
    if hasattr(torch_npu, "npu_swiglu"):
        inter = torch_npu.npu_swiglu(gu_out, dim=-1)
    else:
        gate, up = gu_out.chunk(2, dim=-1)
        inter = act_fn(gate) * up
    dn_out = torch.ops.npu.npu_grouped_matmul(
        [inter], [dn_w],
        group_list=gl, group_type=0, split_item=3, group_list_type=glt,
        tuning_config=tuning_config,
    )[0]
    return torch_npu.npu_moe_token_unpermute(
        dn_out, slot_to_pos, probs=top_k_weights,
    )


def sort_moe_forward(hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj, *,
                     act_fn=None, weights_x_in_out=False):
    """torch.sort-routed fused MoE (graph-safe, static shapes)."""
    _ = hidden_states.shape[0]
    top_k = top_k_index.shape[-1]
    num_experts = gate_up_proj.shape[0]
    flat_experts = top_k_index.reshape(-1).long()
    sorted_experts, perm = torch.sort(flat_experts, stable=True)
    token_of_sorted_slot = perm // top_k
    gathered = hidden_states.index_select(0, token_of_sorted_slot)
    counts = (sorted_experts.unsqueeze(-1) == torch.arange(num_experts, device=sorted_experts.device)).sum(0)
    if _glist_cumsum():
        gl = _fast_cumsum_int64(counts)
        glt = 0
    else:
        gl = counts.to(torch.int64)
        glt = 1
    gu_w = gate_up_proj if weights_x_in_out else gate_up_proj.transpose(1, 2)
    dn_w = down_proj if weights_x_in_out else down_proj.transpose(1, 2)
    gu_out = torch.ops.npu.npu_grouped_matmul(
        [gathered], [gu_w],
        group_list=gl, group_type=0, split_item=3, group_list_type=glt,
    )[0]
    gate, up = gu_out.chunk(2, dim=-1)
    inter = act_fn(gate) * up
    dn_out = torch.ops.npu.npu_grouped_matmul(
        [inter], [dn_w],
        group_list=gl, group_type=0, split_item=3, group_list_type=glt,
    )[0]
    w_sorted = top_k_weights.reshape(-1).index_select(0, perm).unsqueeze(-1)
    weighted = dn_out * w_sorted
    return torch.zeros_like(hidden_states).index_add_(0, token_of_sorted_slot, weighted)


def bmm_moe_forward(hidden_states, top_k_index, top_k_weights, gate_up_proj, down_proj, *,
                    act_fn=None, weights_x_in_out=False):
    """Slot-major batched-matmul MoE (dtype-agnostic, incl."""
    num_tokens = hidden_states.shape[0]
    top_k = top_k_index.shape[-1]
    flat_idx = top_k_index.reshape(-1).long()
    expert_gu = torch.index_select(gate_up_proj, 0, flat_idx)
    expert_dp = torch.index_select(down_proj, 0, flat_idx)

    hidden_expanded = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(
        -1, hidden_states.shape[-1]
    )
    if weights_x_in_out:
        gate_up = torch.bmm(hidden_expanded.unsqueeze(1), expert_gu)
    else:
        gate_up = torch.bmm(hidden_expanded.unsqueeze(1), expert_gu.transpose(1, 2))
    gate, up = gate_up.chunk(2, dim=-1)
    current_hidden = act_fn(gate) * up
    if weights_x_in_out:
        expert_out = torch.bmm(current_hidden, expert_dp)
    else:
        expert_out = torch.bmm(current_hidden, expert_dp.transpose(1, 2))
    expert_out = expert_out.squeeze(1)

    weights_flat = top_k_weights.reshape(-1, 1)
    weighted = expert_out * weights_flat
    return weighted.reshape(num_tokens, top_k, -1).sum(dim=1)


def fused_router_forward(hidden_states, weight, top_k, norm_topk_prob, hidden_dim):
    """Fused softmax+topk(+renorm) router via one ``npu_moe_gating_top_k`` call."""
    import os
    router_logits = torch.nn.functional.linear(
        hidden_states.reshape(-1, hidden_dim), weight,
    )
    renorm = 1 if norm_topk_prob else 0
    if renorm and os.environ.get("LB_ROUTER_RENORM", "1") == "0":
        y_out, idx_out, _ = torch_npu.npu_moe_gating_top_k(router_logits, top_k)
        y32 = y_out.float()
        router_scores = (y32 / y32.sum(dim=-1, keepdim=True)).to(router_logits.dtype)
    else:
        y_out, idx_out, _ = torch_npu.npu_moe_gating_top_k(
            router_logits, top_k, renorm=renorm)
        router_scores = y_out.to(router_logits.dtype)
    return router_logits, router_scores, idx_out


def make_fused_experts_forward(*, mode="cann", decode_grouped=False):
    """Build a bound-able ``forward(self, hidden_states, top_k_index, top_k_weights)``."""
    has_cann_ops = (
        hasattr(torch_npu, "npu_moe_init_routing_v2")
        and hasattr(torch_npu, "npu_moe_token_unpermute")
    )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        import os
        grouped_decode = decode_grouped or os.environ.get("LB_MOE_DECODE_GROUPED", "0") == "1"
        num_tokens = hidden_states.shape[0]
        weights_x_in_out = getattr(self, "_lb_moe_weights_x_in_out", False)
        use_cann = (
            mode == "cann"
            and has_cann_ops
            and hidden_states.dtype in (torch.bfloat16, torch.float16)
            and (num_tokens > 1 or grouped_decode)
        )
        if use_cann:
            tuning = None
            if os.environ.get("LB_MOE_TUNING", "0") == "1":
                tuning = [max(1, num_tokens * top_k_index.shape[-1]
                              // self.gate_up_proj.shape[0])]
            try:
                return cann_moe_forward(
                    hidden_states, top_k_index, top_k_weights,
                    self.gate_up_proj, self.down_proj, act_fn=self.act_fn,
                    weights_x_in_out=weights_x_in_out,
                    tuning_config=tuning,
                )
            except (AttributeError, RuntimeError):
                pass
        if num_tokens > 1 and mode in ("cann", "sort"):
            return sort_moe_forward(
                hidden_states, top_k_index, top_k_weights,
                self.gate_up_proj, self.down_proj, act_fn=self.act_fn,
                weights_x_in_out=weights_x_in_out,
            )
        return bmm_moe_forward(
            hidden_states, top_k_index, top_k_weights,
            self.gate_up_proj, self.down_proj, act_fn=self.act_fn,
            weights_x_in_out=weights_x_in_out,
        )

    return forward


def install_fused_experts_forward(experts, *, mode="cann", decode_grouped=False, owner="boost"):
    """Patch an HF experts module in place to the fused forward."""
    if getattr(experts, "_lb_moe_fuse_owner", None) is not None:
        return False
    if not hasattr(experts, "_original_forward"):
        experts._original_forward = experts.forward
    experts._lb_moe_fuse_owner = owner
    experts.forward = types.MethodType(
        make_fused_experts_forward(mode=mode, decode_grouped=decode_grouped),
        experts,
    )
    return True


def restore_experts_forward(experts, *, owner=None):
    """Restore the original experts forward (revert ``install_fused_experts_forward``)."""
    if getattr(experts, "_lb_moe_fuse_owner", None) is None:
        return False
    if owner is not None and experts._lb_moe_fuse_owner != owner:
        return False
    if hasattr(experts, "_original_forward"):
        experts.forward = experts._original_forward
        del experts._original_forward
    del experts._lb_moe_fuse_owner
    return True


def _fused_router_method(self, hidden_states):
    return fused_router_forward(
        hidden_states, self.weight, self.top_k,
        self.norm_topk_prob, self.hidden_dim,
    )


def install_fused_router_forward(router, *, owner="boost"):
    """Patch an HF MoE router module to the fused ``npu_moe_gating_top_k`` forward."""
    if not hasattr(torch_npu, "npu_moe_gating_top_k"):
        return False
    if getattr(router, "_lb_moe_router_fuse_owner", None) is not None:
        return False
    if not hasattr(router, "_original_forward"):
        router._original_forward = router.forward
    router._lb_moe_router_fuse_owner = owner
    router.forward = types.MethodType(_fused_router_method, router)
    return True


def restore_fused_router_forward(router, *, owner=None):
    """Restore the original router forward (revert ``install_fused_router_forward``)."""
    if getattr(router, "_lb_moe_router_fuse_owner", None) is None:
        return False
    if owner is not None and router._lb_moe_router_fuse_owner != owner:
        return False
    if hasattr(router, "_original_forward"):
        router.forward = router._original_forward
        del router._original_forward
    del router._lb_moe_router_fuse_owner
    return True
