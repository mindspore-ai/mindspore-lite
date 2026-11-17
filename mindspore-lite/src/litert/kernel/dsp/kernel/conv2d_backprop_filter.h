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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_CONV2D_BACKPROP_FILTER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_CONV2D_BACKPROP_FILTER_H_

#include "src/litert/kernel/dsp/dsp_kernel.h"

namespace mindspore::kernel {

class Conv2DBackpropFilterDSPKernel : public DSPKernel {
 public:
  using DSPKernel::DSPKernel;

  ~Conv2DBackpropFilterDSPKernel() override = default;

  int Prepare() override;
  int CheckSpecs() override;
  int Run() override;

 private:
  // The three inputs and the weight gradient they describe. ResolveGeometry runs from CheckSpecs at
  // creation, from Prepare because DSPKernel::ReSize only re-runs Prepare, and from Run because
  // DSPKernelCreator skips CheckSpecs while a shape is still unknown and because a resize may leave
  // everything cached below behind.
  int ResolveGeometry();
  int CheckTensors() const;
  int CheckWeightShape() const;

  const lite::Tensor *Dy() const;
  const lite::Tensor *X() const;
  const lite::Tensor *WeightShape() const;
  const lite::Tensor *WeightGrad() const;

  int batch_ = 0;
  int input_h_ = 0;
  int input_w_ = 0;
  int input_channel_ = 0;
  int output_h_ = 0;
  int output_w_ = 0;
  int output_channel_ = 0;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_DSP_KERNEL_CONV2D_BACKPROP_FILTER_H_
