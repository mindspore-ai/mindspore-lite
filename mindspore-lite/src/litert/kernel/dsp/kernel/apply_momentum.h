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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_APPLY_MOMENTUM_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_APPLY_MOMENTUM_H_

#include <string>
#include <memory>
#include "src/litert/kernel/dsp/dsp_kernel.h"
#include "src/litert/kernel/dsp/dsp_allocator.h"

namespace mindspore::kernel {

constexpr size_t kApplyMomentumInputTensorSize = 5;
constexpr size_t kApplyMomentumOutputTensorSize = 1;
constexpr size_t kApplyMomentumFloatParamSize = 2;
constexpr size_t kApplyMomentumIntParamSize = 2;

constexpr size_t kApplyMomentumWeightIdx = 0;
constexpr size_t kApplyMomentumAccumulateIdx = 1;
constexpr size_t kApplyMomentumLrIdx = 2;
constexpr size_t kApplyMomentumGradientIdx = 3;
constexpr size_t kApplyMomentumMomentumIdx = 4;

class ApplyMomentumDSPKernel : public DSPKernel {
 public:
  using DSPKernel::DSPKernel;

  ~ApplyMomentumDSPKernel() override;

  int Prepare() override;
  int CheckSpecs() override;
  int Run() override;

 private:
  void *float_params_buffer_ = nullptr;
  void *int_params_buffer_ = nullptr;
  std::shared_ptr<lite::dsp::DSPAllocator> allocator_{};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_APPLY_MOMENTUM_H_
