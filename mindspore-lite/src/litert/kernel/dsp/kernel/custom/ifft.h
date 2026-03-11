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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CUSTOM_KERNEL_IFFT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CUSTOM_KERNEL_IFFT_H_

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include "include/registry/register_kernel_interface.h"
#include "include/registry/register_kernel.h"
#include "include/errorcode.h"
#include "src/litert/kernel/dsp/custom_kernel.h"

namespace mindspore::lite {

class IFFTKernel : public CustomKernel {
 public:
  IFFTKernel(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs,
             const schema::Primitive *primitive, const mindspore::Context *ctx)
      : CustomKernel(inputs, outputs, primitive, ctx) {}
  ~IFFTKernel();
  int Prepare() override;
  int CheckSpecs(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs) override;
  int Run() override;

 private:
  uint64_t w_device_ptr_ = 0;
  void *w_ptr_ = nullptr;
  int64_t length_ = 0;
  void *temp_fft_ptr_ = nullptr;
  uint64_t temp_fft_device_ptr_ = 0;
  std::shared_ptr<lite::dsp::DSPAllocator> allocator_{};
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CUSTOM_KERNEL_IFFT_H_
