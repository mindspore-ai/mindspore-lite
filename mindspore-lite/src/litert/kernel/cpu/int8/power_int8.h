/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_POWER_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_POWER_INT8_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/int8/quantize.h"
#include "nnacl/pow_parameter.h"

namespace mindspore::kernel {
class PowerInt8CPUKernel : public LiteKernel {
 public:
  PowerInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                     const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<PowParameter *>(op_parameter_);
  }
  ~PowerInt8CPUKernel() {}

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoPower(int task_id);

 private:
  PowParameter *param_;
  int8_t *input_data_ = nullptr;
  int8_t *output_data_ = nullptr;
  int8_t *exp_ptr_ = nullptr;
  PowQuantArg quant_arg_;
  bool broadcast_ = false;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_POWER_INT8_H_
