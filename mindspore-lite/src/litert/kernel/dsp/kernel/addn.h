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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_ADDN_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_ADDN_H_

#include <vector>
#include "src/litert/kernel/dsp/dsp_kernel.h"

namespace mindspore::kernel {

class AddNDSPKernel : public DSPKernel {
 public:
  AddNDSPKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : DSPKernel(parameter, inputs, outputs, ctx) {}
  ~AddNDSPKernel() override = default;

  int Prepare() override;
  int CheckSpecs() override;
  int Run() override;

 protected:
  // The two arrays armc walks over. They live here so that Run() only writes into memory it
  // already owns instead of allocating on every inference.
  std::vector<uint64_t> input_ptrs_;
  std::vector<int> element_nums_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_ADDN_H_
