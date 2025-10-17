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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_EXP_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_EXP_H_

#include <vector>
#include <string>
#include "src/litert/kernel/dsp/dsp_kernel.h"

namespace mindspore::kernel {
class ExpDSPKernel : public DSPKernel {
 public:
  using DSPKernel::DSPKernel;

  ~ExpDSPKernel() override = default;
  int Prepare() override;
  int CheckSpecs() override;
  int Run() override;

  int ExpRunFp32();
  int ExpRunFp16();
  int ExpRunInt16();
  int ExpRunInt32();
  int ExpRunComplex64();

 private:
  std::string kernel_name_;
  uint64_t core_mask_;
  float in_scale_;
  float out_scale_;
  int scale_;
};
}  // namespace mindspore::kernel
#endif
