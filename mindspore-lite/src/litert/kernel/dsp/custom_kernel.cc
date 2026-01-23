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

#include "src/litert/kernel/dsp/custom_kernel.h"
#include <algorithm>
#include <string>
#include <vector>

namespace mindspore::lite {

inline void ConvertMSTensorsToLiteTensors(const std::vector<MSTensor> &src_tensors,
                                          std::vector<mindspore::lite::Tensor *> *dst_tensors) {
  dst_tensors->clear();
  std::transform(src_tensors.begin(), src_tensors.end(), std::back_inserter(*dst_tensors),
                 [](const MSTensor &src_tensor) -> mindspore::lite::Tensor * {
                   if (src_tensor.impl() == nullptr) {
                     MS_LOG(ERROR) << "Tensor " << src_tensor.Name() << " is nullptr.";
                     return nullptr;
                   }
                   auto lite_impl = std::static_pointer_cast<LiteTensorImpl>(src_tensor.impl());
                   if (!lite_impl) {
                     MS_LOG(ERROR) << "Failed to cast MSTensor to LiteTensorImpl.";
                     return nullptr;
                   }
                   return static_cast<mindspore::lite::Tensor *>(lite_impl->lite_tensor());
                 });
}

int CustomKernel::CheckOutputs(const std::vector<mindspore::MSTensor> &outputs) {
  for (auto &output : outputs) {
    auto output_shape = output.Shape();
    if (std::find(output_shape.begin(), output_shape.end(), -1) != output_shape.end()) {
      return kLiteInferInvalid;
    }
  }
  return kSuccess;
}

void CustomKernel::ConvertTensors() {
  ConvertMSTensorsToLiteTensors(inputs_, &in_tensors_);
  ConvertMSTensorsToLiteTensors(outputs_, &out_tensors_);
}

int CustomKernel::PreProcess() {
  if (CheckOutputs(outputs_) != kSuccess) {
    if (primitive_ == nullptr) {
      MS_LOG(ERROR) << "primitive_ is nullptr";
      return kLiteError;
    }
    MS_LOG(DEBUG) << "infer shape along running";
    auto status = registry::RegisterKernelInterface::GetKernelInterface(std::string{}, primitive_, this)
                    ->Infer(&inputs_, &outputs_, primitive_);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "infer failed.";
      return kLiteError;
    }
    auto ret = ReSize();
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "resize failed.";
      return ret;
    }
  }
  for (auto &output : outputs_) {
    // malloc data for output tensor
    auto data = output.MutableData();
    if (data == nullptr) {
      MS_LOG(ERROR) << "Get data failed.";
      return kLiteError;
    }
  }
  return kSuccess;
}

void CustomKernel::ParseAttrData() {
  if (primitive_ == nullptr) {
    MS_LOG(WARNING) << "primitive_ is nullptr";
    return;
  }
  auto prim = primitive_->value_as_Custom();
  if (prim->attr() == nullptr) {
    MS_LOG(DEBUG) << "kernel attribute is null";
    return;
  }
  if (prim->attr()->size() < 1) {
    return;
  }
  for (size_t i = 0; i < prim->attr()->size(); ++i) {
    auto attr = prim->attr()->Get(i);
    auto attr_key = attr->name()->str();
    auto data_bytes = attr->data();
    if (data_bytes == nullptr) {
      continue;
    }
    auto data_size = data_bytes->size();
    std::vector<uint8_t> attr_data(data_size);
    for (size_t j = 0; j < data_size; ++j) {
      auto data = static_cast<uint8_t>(data_bytes->Get(j));
      attr_data[j] = data;
    }
    attrs_[attr_key] = attr_data;
  }
}

int CustomKernel::Execute() {
  auto ret = PreProcess();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "run kernel PreProcess failed, name: " << this->name();
    return ret;
  }
  ConvertTensors();
  ret = Run();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "run kernel failed, name: " << this->name();
    return ret;
  }
  return kSuccess;
}
}  // namespace mindspore::lite
