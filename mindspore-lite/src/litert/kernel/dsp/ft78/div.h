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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_DIV_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_DIV_H_

#include <vector>
#include <string>
#include "src/litert/kernel/dsp/dsp_kernel.h"

namespace mindspore::kernel {
class DivDSPKernel : public DSPKernel {
 public:
  using DSPKernel::DSPKernel;

  ~DivDSPKernel() override = default;
  int Prepare() override;
  int CheckSpecs() override;
  int Run() override;

  int DivRunFp32();
  int DivRunFp64();
  int DivRunInt8();
  int DivRunInt16();
  int DivRunInt32();
  int DivRunComplex64();
  int DivRunComplex128();

 private:
  std::string kernel_name_;
  uint64_t core_mask_;
  bool optimize_{false};
  bool first_scalar_{false};
};
}  // namespace mindspore::kernel
#endif
