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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_GATHER_D_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_GATHER_D_H_

#include <vector>
#include "src/litert/kernel/dsp/dsp_kernel.h"

namespace mindspore::kernel {

// GatherD: in[0]=x, in[1]=axis (int32 scalar host tensor), in[2]=indices (int32).
// Output shape == indices shape, output dtype == x dtype.
class GatherDDSPKernel : public DSPKernel {
 public:
  GatherDDSPKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                   const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : DSPKernel(parameter, inputs, outputs, ctx) {}
  ~GatherDDSPKernel() override = default;
  int Prepare() override;
  int CheckSpecs() override;
  int Run() override;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_GATHER_D_H_
