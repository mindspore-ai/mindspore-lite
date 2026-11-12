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

#include "plugin/fused_causal_conv_silu.h"
#include <string_view>
#include "torch/library.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "plugin/pytorch_npu_helper.h"

namespace {
constexpr std::string_view kOpNameFusedCausalConvSilu = "aclnnFusedCausalConvSilu";
}  // namespace

namespace lite_boost {

std::tuple<at::Tensor, at::Tensor> FusedCausalConvSiluLiteBoostImplNPU(const at::Tensor &x, const at::Tensor &weight,
                                                                       const at::Tensor &bias,
                                                                       const at::Tensor &seq_len) {
  // x [B, T, S] in, y [B, T, S] out (same options/format as x). conv_state keeps the
  // trailing K taps of the pre-SiLU convolution per channel so the decode path can seed
  // the next step: [B, K, S].
  const int64_t T = x.size(1);
  const int64_t C = weight.size(1);
  const int64_t K = weight.size(0);
  at::Tensor y = at_npu::native::empty_with_format({1, T, C}, x.options(), at_npu::native::get_npu_format(x));
  at::Tensor conv_state = at_npu::native::empty_with_format({1, K, C}, x.options(), at_npu::native::get_npu_format(x));

  EXEC_NPU_CMD<kOpNameFusedCausalConvSilu>(x, weight, bias, seq_len, y, conv_state);

  return std::tuple<at::Tensor, at::Tensor>(y, conv_state);
}

}  // namespace lite_boost
