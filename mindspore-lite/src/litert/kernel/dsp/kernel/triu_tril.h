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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_TRIU_TRIL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_TRIU_TRIL_H_

#include <vector>
#include "src/litert/kernel/dsp/dsp_kernel.h"

namespace mindspore::kernel {

// Shared implementation for Tril and Triu. Subclasses only provide the op name
// and the launch call (tril_func / triu_func); CheckSpecs and Run are shared.
class TriuTrilDSPKernelBase : public DSPKernel {
 public:
  TriuTrilDSPKernelBase(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                        const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : DSPKernel(parameter, inputs, outputs, ctx) {}
  ~TriuTrilDSPKernelBase() override = default;
  int Prepare() override;
  int CheckSpecs() override;
  int Run() override;

 protected:
  virtual const char *OpName() const = 0;
  virtual int Launch(uint64_t in_ptr, uint64_t out_ptr, int k, int height, int width, int64_t out_elems, int dtype,
                     int mtype) = 0;
};

class TrilDSPKernel : public TriuTrilDSPKernelBase {
 public:
  TrilDSPKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : TriuTrilDSPKernelBase(parameter, inputs, outputs, ctx) {}
  ~TrilDSPKernel() override = default;

 protected:
  const char *OpName() const override { return "tril"; }
  int Launch(uint64_t in_ptr, uint64_t out_ptr, int k, int height, int width, int64_t out_elems, int dtype,
             int mtype) override;
};

class TriuDSPKernel : public TriuTrilDSPKernelBase {
 public:
  TriuDSPKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : TriuTrilDSPKernelBase(parameter, inputs, outputs, ctx) {}
  ~TriuDSPKernel() override = default;

 protected:
  const char *OpName() const override { return "triu"; }
  int Launch(uint64_t in_ptr, uint64_t out_ptr, int k, int height, int width, int64_t out_elems, int dtype,
             int mtype) override;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_TRIU_TRIL_H_
