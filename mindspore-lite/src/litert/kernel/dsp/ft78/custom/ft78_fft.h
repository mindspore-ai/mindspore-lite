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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CUSTOM_KERNEL_FFT_FT78_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CUSTOM_KERNEL_FFT_FT78_H_

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include "include/registry/register_kernel_interface.h"
#include "include/registry/register_kernel.h"
#include "include/errorcode.h"
#include "src/litert/kernel/dsp/base/fft_base.h"

namespace mindspore::lite {

class FFTFT78Kernel : public FFTBaseKernel {
 public:
  using FFTBaseKernel::FFTBaseKernel;
  ~FFTFT78Kernel() = default;
  int Prepare() override;
};

}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CUSTOM_KERNEL_FFT_FT78_H_
