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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_CROP_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_CROP_H_

#include <array>
#include <cstdint>
#include "src/litert/kernel/dsp/dsp_kernel.h"

namespace mindspore::kernel {

class CropDSPKernel : public DSPKernel {
 public:
  using DSPKernel::DSPKernel;

  ~CropDSPKernel() override = default;

  int Prepare() override;
  int CheckSpecs() override;
  int Run() override;

 private:
  // crop_func walks four dimensional NHWC shapes, both for the input and for the output.
  static constexpr size_t kRank = 4;

  // The axis of the Crop attribute, already folded into the rank of the input.
  int64_t ResolveAxis() const;
  // Expands the attribute offsets of op_parameter_ into one value per dimension.
  void BuildFullOffset(int64_t axis);
  // Checks the resolved offsets against the shapes currently in hand.
  int CheckGeometry() const;
  // Resolves the axis and the offsets and rejects a geometry that would make the DSP read outside
  // the input. CheckSpecs calls it once at creation, Prepare calls it again because
  // DSPKernel::ReSize only re-runs Prepare, and a shape that is still unknown at creation time
  // makes DSPKernelCreator skip CheckSpecs altogether.
  int ResolveGeometry();

  std::array<int, kRank> full_offset_{};  // offset of every dimension, set by ResolveGeometry
  int axis_ = 0;                          // the axis ResolveGeometry used
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_CROP_H_
