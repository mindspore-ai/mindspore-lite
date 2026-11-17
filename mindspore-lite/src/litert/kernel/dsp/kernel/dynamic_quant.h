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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_DYNAMIC_QUANT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_DYNAMIC_QUANT_H_

#include <vector>
#include "src/litert/kernel/dsp/dsp_kernel.h"
#include "nnacl_c/dynamic_quant_parameter.h"

namespace mindspore::kernel {

class DynamicQuantDSPKernel : public DSPKernel {
 public:
  using DSPKernel::DSPKernel;

  int Prepare() override;
  int CheckSpecs() override;
  int Run() override;

 private:
  // Compute per-tensor min/max from the input (converting fp16 to fp32 on the fly).
  int FindMinMax(float *min_value, float *max_value);
  DynamicQuantParameter *param_{nullptr};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_DYNAMIC_QUANT_H_
