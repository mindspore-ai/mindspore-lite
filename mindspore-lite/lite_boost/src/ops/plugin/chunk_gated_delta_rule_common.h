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
 * @file chunk_gated_delta_rule_common.h
 * @brief Output/optional-input preparation shared by the two chunked Gated Delta Rule plugins.
 *
 * ChunkGatedDeltaRule and FusedQkL2NormChunkGatedDeltaRule call different aclnn ops (the
 * fused one L2-normalizes query/key in-kernel and zeroes pad rows), but their tensors are
 * laid out identically: `out` mirrors value's TND shape, `final_state` mirrors
 * initial_state, and an absent `g` is passed to the op-api as an *empty* tensor. Keeping
 * that in one place stops the two plugins from drifting apart on dtype/layout rules.
 */

#ifndef LITE_BOOST_OPS_PLUGIN_CHUNK_GATED_DELTA_RULE_COMMON_H_
#define LITE_BOOST_OPS_PLUGIN_CHUNK_GATED_DELTA_RULE_COMMON_H_

#include "torch/library.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace lite_boost {

/**
 * @brief Tensors handed to an aclnn chunked-GDR op.
 */
struct ChunkGatedDeltaRuleTensors {
  at::Tensor out;          //!< [T, Nv, Dv] bf16, value's shape and NPU format.
  at::Tensor final_state;  //!< [B, Nv, Dv, Dk] bf16, initial_state's shape and NPU format.
  at::Tensor g;            //!< [T, Nv] float32 decay gate; empty when the caller passed none.
};

/**
 * @brief Allocate out/final_state and materialize the optional decay gate.
 *
 * Both outputs are created with `empty_with_format` so they inherit value's /
 * initial_state's NPU format (the kernel rejects a format mismatch). The dtypes follow
 * the operands -- value/initial_state are already bf16 when they reach the plugin, and
 * the ascend_a2 tiling rejects fp16. An absent `g` becomes an empty at::Tensor, which is
 * how the CANN op-api says "optional input not provided" (hasGamma=0 in the kernel).
 *
 * @param value           TND value tensor the output mirrors.
 * @param initial_state   recurrent state the final state mirrors.
 * @param g               optional decay gate.
 * @return the three tensors in aclnn call order.
 */
inline ChunkGatedDeltaRuleTensors PrepareChunkGatedDeltaRuleTensors(const at::Tensor &value,
                                                                    const at::Tensor &initial_state,
                                                                    const c10::optional<at::Tensor> &g) {
  ChunkGatedDeltaRuleTensors tensors;
  tensors.out =
    at_npu::native::empty_with_format(value.sizes().vec(), value.options(), at_npu::native::get_npu_format(value));
  tensors.final_state = at_npu::native::empty_with_format(initial_state.sizes().vec(), initial_state.options(),
                                                          at_npu::native::get_npu_format(initial_state));
  tensors.g = c10::value_or_else(g, [] { return at::Tensor(); });
  return tensors;
}

}  // namespace lite_boost

#endif  // LITE_BOOST_OPS_PLUGIN_CHUNK_GATED_DELTA_RULE_COMMON_H_
