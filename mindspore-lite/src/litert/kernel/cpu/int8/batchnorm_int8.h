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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_BATCHNORM_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_BATCHNORM_INT8_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/int8/batchnorm_int8.h"
#include "nnacl/batchnorm_parameter.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class BatchnormInt8CPUKernel : public LiteKernel {
 public:
  BatchnormInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}
  ~BatchnormInt8CPUKernel() override;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int InitConstTensor();
  int InitFusedConstTensor();
  int DoExecute(int tid);

 private:
  int8_t *in_addr_ = nullptr;
  int8_t *out_addr_ = nullptr;
  float *alpha_addr_ = nullptr;
  float *beta_addr_ = nullptr;
  int channel_;
  int unit_ = 0;
  int units_ = 0;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_BATCHNORM_INT8_H_
