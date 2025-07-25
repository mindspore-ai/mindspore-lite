/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_GRAD_DROPOUT_FP16_GRAD_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_GRAD_DROPOUT_FP16_GRAD_H_

#include <vector>
#include "src/litert/lite_kernel.h"

namespace mindspore::kernel {
class DropoutGradCPUKernelFp16 : public LiteKernel {
 public:
  DropoutGradCPUKernelFp16(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                           const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx), thread_count_(ctx->thread_num_) {}

  ~DropoutGradCPUKernelFp16() override = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoExecute(int task_id);

 private:
  float scale_ = 1.0f;
  int thread_count_ = 1;
};

}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_GRAD_DROPOUT_FP16_GRAD_H_
