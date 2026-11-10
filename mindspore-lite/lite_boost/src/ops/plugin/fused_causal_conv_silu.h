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

/* fused_causal_conv_silu.h FusedCausalConvSilu operator — fused depthwise causal-conv1d(K=4) + SiLU (prefill). */

#ifndef LITE_BOOST_OPS_PLUGIN_FUSED_CAUSAL_CONV_SILU_H_
#define LITE_BOOST_OPS_PLUGIN_FUSED_CAUSAL_CONV_SILU_H_

#include <tuple>
#include "ATen/Tensor.h"

namespace lite_boost {

/* NPU implementation of FusedCausalConvSilu (ascend910b / ascend_a2 op spec).
 *
 * @param x       activations [B, T, S] (BTS layout, batch-major token rows)
 * @param weight  depthwise kernel [K=4, S]
 * @param bias    per-channel bias [S] (required by this op)
 * @param seq_len int32 valid-length vector (the fixed-shape harness pads with it)
 * @returns (y [B, T, S], conv_state [B, K, S])
 */
std::tuple<at::Tensor, at::Tensor> FusedCausalConvSiluLiteBoostImplNPU(const at::Tensor &x, const at::Tensor &weight,
                                                                       const at::Tensor &bias,
                                                                       const at::Tensor &seq_len);

}  // namespace lite_boost

#endif
