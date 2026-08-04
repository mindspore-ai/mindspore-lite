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

#include "plugin/quant_matmul_w4a8.h"
#include <string_view>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "plugin/pytorch_npu_helper.h"

namespace {
constexpr std::string_view kOpName = "aclnnQuantMatmulW4a8";
}  // namespace

at::Tensor QuantMatmulW4a8LiteBoostImplNPU(const at::Tensor &act, const at::Tensor &weight, const at::Tensor &scale,
                                           const at::Tensor &bias, const at::Tensor &x_scale,
                                           const at::Tensor &output_bias) {
  int64_t M = act.size(0);
  int64_t N = weight.size(0);
  auto out =
    at_npu::native::empty_with_format({M, N}, act.options().dtype(at::kBFloat16), at_npu::native::get_npu_format(act));
  EXEC_NPU_CMD<kOpName>(act, weight, scale, bias, x_scale, output_bias, out);
  return out;
}
