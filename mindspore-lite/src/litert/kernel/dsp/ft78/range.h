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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_FT78_RANGE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_FT78_RANGE_H_

#include <cstdint>
#include <string>
#include "src/litert/kernel/dsp/dsp_kernel.h"

namespace mindspore::kernel {
class RangeDSPKernel : public DSPKernel {
 public:
  RangeDSPKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                 const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : DSPKernel(parameter, inputs, outputs, ctx) {}
  ~RangeDSPKernel() override = default;

  int CheckSpecs() override;
  int Prepare() override;
  int Run() override;

 private:
  int RangeRunFp32();
  int RangeRunFp64();
  int RangeRunInt8();
  int RangeRunInt16();
  int RangeRunInt32();

  std::string kernel_name_;
  uint64_t core_mask_ = 0;
  float start_ = 0.f;
  float delta_ = 0.f;
  double start_double_ = 0.0;
  double delta_double_ = 0.0;
  int32_t start_i32_ = 0;
  int32_t delta_i32_ = 0;
  int16_t start_i16_ = 0;
  int16_t delta_i16_ = 0;
  int8_t start_i8_ = 0;
  int8_t delta_i8_ = 0;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_FT78_RANGE_H_
