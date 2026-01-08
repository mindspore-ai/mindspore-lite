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

#include "src/litert/kernel/dsp/ft04/custom/ft04_fft.h"
#include <vector>
#include "src/common/utils.h"

namespace mindspore::lite {

namespace {
const auto kComplex64 = DataType::kNumberTypeComplex64;
}  // namespace

int FFTFT04Kernel::Prepare() {
  int core_mask = 0xf;  // set core mask to all cores
  SetCoreMask(core_mask);
  auto ret = FFTBaseKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FFTFT04Kernel prepare failed.";
    return kLiteError;
  }
  return kSuccess;
}

REGISTER_CUSTOM_KERNEL(DSP, FTMatrix, kComplex64, Custom_FT_FFT, CustomKernelCreator<FFTFT04Kernel>)
REGISTER_CUSTOM_KERNEL(DSP, FTMatrix, kComplex64, Custom_FT_IFFT, CustomKernelCreator<FFTFT04Kernel>)
}  // namespace mindspore::lite
