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

#include "tools/converter/adapter/dsp/infer/custom_fft_infer.h"
#include <vector>
#include <memory>
#include "include/api/status.h"
#include "include/registry/register_kernel_interface.h"
#include "tools/converter/adapter/dsp/infer/custom_common.h"

namespace mindspore::lite {

Status CustomFFTInfer::Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                             const schema::Primitive *primitive) {
  if (inputs->size() != 1 || outputs->size() != 1) {
    return kLiteInferInvalid;
  }
  outputs->front().SetFormat(inputs->front().format());
  outputs->front().SetDataType(inputs->front().DataType());
  auto ret = common::CheckIsDynamicShape(*inputs);
  if (ret == kLiteInferInvalid) {
    outputs->front().SetShape({-1});  // shape{-1} shows that shape need to be inferred when running.
    return kLiteInferInvalid;
  } else if (ret != kSuccess) {
    return kLiteError;
  }
  outputs->front().SetShape(inputs->front().Shape());
  return kSuccess;
}

std::shared_ptr<kernel::KernelInterface> CustomFFTInferCreator() { return std::make_shared<CustomFFTInfer>(); }
REGISTER_CUSTOM_KERNEL_INTERFACE(FTMatrix, Custom_FT_FFT, CustomFFTInferCreator)
}  // namespace mindspore::lite
