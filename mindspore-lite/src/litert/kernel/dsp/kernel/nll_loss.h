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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_NLL_LOSS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_NLL_LOSS_H_

#include "src/litert/kernel/dsp/dsp_kernel.h"
#include "nnacl_c/nllloss_parameter.h"

namespace mindspore::kernel {

class NLLLossDSPKernel : public DSPKernel {
 public:
  using DSPKernel::DSPKernel;

  int Prepare() override;
  int CheckSpecs() override;
  int Run() override;

 private:
  // Map Lite ReductionType (Sum=0/Mean=1/None=2) to SDK (None=0/Sum=1/Mean=2).
  static int GetSdkReduction(ReductionType reduction);
  // Validate label values are within [0, class_num) to avoid SDK OOB reads.
  int ValidateLabels();
  NLLLossParameter *param_{nullptr};
  int batch_{0};
  int class_num_{0};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_NLL_LOSS_H_
