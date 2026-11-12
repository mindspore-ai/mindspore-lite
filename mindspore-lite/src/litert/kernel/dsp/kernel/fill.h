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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_FILL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_FILL_H_

#include <vector>
#include "src/litert/kernel/dsp/dsp_kernel.h"

namespace mindspore::kernel {

// Fill and FillV2 share one implementation but use opposite input layouts
// (see fill_infer.c / fillv2_infer.c). The kernel picks indices by type():
//   Fill:   in[0]=value tensor, in[1]=dst_shape tensor
//   FillV2: in[0]=dst_shape tensor, in[1]=value tensor
// The value tensor must be a scalar on device memory.
class FillDSPKernel : public DSPKernel {
 public:
  FillDSPKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : DSPKernel(parameter, inputs, outputs, ctx) {}
  ~FillDSPKernel() override = default;
  int Prepare() override;
  int CheckSpecs() override;
  int Run() override;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_FILL_H_
