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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_CONCAT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_CONCAT_H_

#include <cstdint>
#include <vector>
#include "src/litert/kernel/dsp/dsp_kernel.h"

namespace mindspore::kernel {

class ConcatDSPKernel : public DSPKernel {
 public:
  using DSPKernel::DSPKernel;

  ~ConcatDSPKernel() override = default;

  int Prepare() override;
  int CheckSpecs() override;
  int Run() override;

 private:
  // Concat takes a variable number of inputs, so the shape table the DSP is handed cannot be written
  // once at construction. ResolveGeometry rebuilds it and, in doing so, re-checks every rule; it runs
  // from CheckSpecs at creation, from Prepare because DSPKernel::ReSize only re-runs Prepare, and from
  // Run because DSPKernelCreator skips CheckSpecs while a shape is still unknown and because a resize
  // may leave the cached table behind.
  int ResolveGeometry();
  int CheckOutput(const lite::Tensor *output, int axis_extent) const;
  // The rules the DSP path has to hold before every launch, checked from all three entries.
  int CheckContract();
  int CheckQuantParams() const;

  std::vector<int> input_shapes_;
  std::vector<uint64_t> input_ptrs_;
  int axis_ = 0;
  int rank_ = 0;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_CONCAT_H_
