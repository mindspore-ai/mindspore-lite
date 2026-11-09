/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_GRAD_CONVOLUTION_GRAD_FILTER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_GRAD_CONVOLUTION_GRAD_FILTER_H_

#include <vector>
#include "src/litert/lite_kernel.h"

namespace mindspore::kernel {
class ConvolutionGradFilterCPUKernel : public LiteKernel {
 public:
  explicit ConvolutionGradFilterCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                          const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)

      : LiteKernel(parameter, inputs, outputs, ctx) {}
  ~ConvolutionGradFilterCPUKernel() override {}

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoExecute(int task_id);
  int DoReduce(int task_id);
  // Accumulate one GEMM partial into the dW target (private slice lock-free, or
  // shared dw under lock_); kept out of DoExecute to bound its nesting depth.
  void AccumulateDw(float *mat_tmp, float *mat_c, int k, int n);

 private:
  size_t ws_size_ = 0;
  bool do_img2col_ = true;
  bool do_dw_ = false;
  std::mutex lock_;
  size_t mat_alloc_ = 0;
  // Per-thread private dW buffer plus a single end-of-Run reduction, avoiding
  // global-mutex contention. When use_private_dw_ is false (single thread, batch 1,
  // depthwise, or buffer too large), the original locked AddMatrix-into-shared-dw
  // path is used and stays bit-identical.
  size_t dw_size_ = 0;            // floats per private dW buffer
  size_t dw_private_offset_ = 0;  // float offset of the private-dW region in workspace
  bool use_private_dw_ = false;
  size_t ws_thread_num_ = 1;  // thread count the workspace was sized for at ReSize
#ifdef ENABLE_ARM32
  const int chunk_ = C4NUM * 2;
#else
  const int chunk_ = C12NUM * 2;
#endif
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_GRAD_CONVOLUTION_GRAD_FILTER_H_
