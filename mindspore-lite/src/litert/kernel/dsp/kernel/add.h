/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_ADD_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_ADD_H_

#include <vector>
#include <string>
#include <map>
#include "src/litert/kernel/dsp/dsp_kernel.h"
#include "src/litert/kernel/cpu/nnacl_c/arithmetic_parameter.h"

namespace mindspore::kernel {

class AddDSPKernel : public DSPKernel {
 public:
  AddDSPKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
               const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : DSPKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<ArithmeticParameter *>(parameter);
  }
  ~AddDSPKernel() override = default;
  int Prepare() override;
  int CheckSpecs() override;
  int Run() override;

 protected:
  ArithmeticParameter *param_{nullptr};
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_ADD_H_
