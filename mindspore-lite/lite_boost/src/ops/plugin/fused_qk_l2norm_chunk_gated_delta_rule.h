/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file fused_qk_l2norm_chunk_gated_delta_rule.h
 * @brief FusedQkL2NormChunkGatedDeltaRule operator — L2 normalization of q/k fused into the
 *        chunked (prefill) Gated Delta Rule.
 *
 * Wraps the AscendC `aclnnFusedQkL2NormChunkGatedDeltaRule` shipped under ascend_a2
 * (ascend910b). This is a standalone operator, separate from ChunkGatedDeltaRule: it fuses
 * the two host-side steps that otherwise bracket that op —
 *
 *   q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)   (row L2 norm)
 *   out  = out.masked_fill(pos >= actual_seq_len, 0)                  (pad-row zeroing)
 *
 * — into the kernel. The caller therefore passes RAW (unnormalized) q/k in TND layout:
 *
 *   query             [T, Nk, Dk]   bf16   (raw; normalized in-kernel)
 *   key               [T, Nk, Dk]   bf16   (raw; normalized in-kernel)
 *   value             [T, Nv, Dv]   bf16
 *   beta              [T, Nv]       bf16
 *   initial_state     [B, Nv, Dv, Dk]  bf16   (Dv-first, matches op storage)
 *   actual_seq_lengths [B]          int32   (T = sum(actual_seq_lengths))
 *   g (optional)      [T, Nv]       float32 (< 0); absent -> no decay gate (hasGamma=0)
 *   out               [T, Nv, Dv]   bf16
 *   final_state       [B, Nv, Dv, Dk] bf16
 *   attr scale_value       float (default 1.0)
 *   attr out_zero_pad_rows bool  (default false)
 *
 * The Python binding handles BNSD<->TND conversion and the (Dk,Dv)<->(Dv,Dk) state transpose.
 * Targets ascend910b only. In-kernel normalization requires the dk-aligned head dim to be a
 * power of two in [64, 256] (tiling rejects anything else, so a caller can never silently
 * receive un-normalized q/k).
 */

#ifndef LITE_BOOST_OPS_PLUGIN_FUSED_QK_L2NORM_CHUNK_GATED_DELTA_RULE_H_
#define LITE_BOOST_OPS_PLUGIN_FUSED_QK_L2NORM_CHUNK_GATED_DELTA_RULE_H_

#include <tuple>
#include "ATen/Tensor.h"
#include "c10/util/Optional.h"

namespace lite_boost {

/**
 * @brief NPU implementation of FusedQkL2NormChunkGatedDeltaRule (ascend910b / ascend_a2 op spec).
 *
 * Tensors are passed in TND layout (the Python binding converts from BNSD); query/key are
 * expected in their raw, un-normalized form. `g` is optional: pass c10::nullopt (or an empty
 * at::Tensor) to take the no-decay-gate path. `out_zero_pad_rows` makes the kernel zero the
 * output rows outside [0, actual_seq_len), replacing the caller's masked_fill.
 *
 * @returns (out [T,Nv,Dv] bf16, final_state [B,Nv,Dv,Dk] bf16)
 */
std::tuple<at::Tensor, at::Tensor> FusedQkL2NormChunkGatedDeltaRuleLiteBoostImplNPU(
  const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &beta,
  const at::Tensor &initial_state, const at::Tensor &actual_seq_lengths, const c10::optional<at::Tensor> &g,
  double scale_value, bool out_zero_pad_rows);

}  // namespace lite_boost

#endif  // LITE_BOOST_OPS_PLUGIN_FUSED_QK_L2NORM_CHUNK_GATED_DELTA_RULE_H_
