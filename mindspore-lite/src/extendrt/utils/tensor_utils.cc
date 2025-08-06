/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include <memory>
#include <algorithm>
#include <utility>

#include "extendrt/utils/tensor_utils.h"
#include "mindspore/ccsrc/kernel/framework_utils.h"

namespace mindspore {
kernel::AddressPtr CloudTensorUtils::LiteTensorToAddressPtr(const lite::Tensor *lite_tensor) {
  kernel::AddressPtr address_ptr = std::make_shared<kernel::Address>(lite_tensor->data(), lite_tensor->Size());
  return address_ptr;
}

std::vector<mindspore::kernel::AddressPtr> CloudTensorUtils::LiteTensorToAddressPtrVec(
  const std::vector<lite::Tensor *> &lite_tensors) {
  kernel::AddressPtrList address_list;

  for (auto lite_tensor : lite_tensors) {
    kernel::AddressPtr address = LiteTensorToAddressPtr(lite_tensor);
    address_list.push_back(address);
  }

  return address_list;
}

kernel::KernelTensor *CloudTensorUtils::LiteTensorToKernelTensorPtr(const lite::Tensor *lite_tensor) {
  kernel::AddressPtr address = LiteTensorToAddressPtr(lite_tensor);
  kernel::KernelTensor *kernel_tensor_ptr = new (std::nothrow) kernel::KernelTensor();
  if (kernel_tensor_ptr == nullptr) {
    return kernel_tensor_ptr;
  }
  kernel_tensor_ptr->SetData(address);
  kernel_tensor_ptr->set_format(lite_tensor->format());
  kernel_tensor_ptr->SetType(std::make_shared<TensorType>(TypeIdToType(lite_tensor->data_type())));

  auto lite_shape = lite_tensor->shape();
  std::vector<int64_t> shape;
  for (size_t i = 0; i < lite_shape.size(); i++) {
    shape.push_back(lite_shape[i]);
  }
  kernel_tensor_ptr->SetShape(std::make_shared<abstract::TensorShape>(std::move(shape)));
  return kernel_tensor_ptr;
}

std::vector<kernel::KernelTensor *> CloudTensorUtils::LiteTensorToKernelTensorPtrVec(
  const std::vector<lite::Tensor *> &lite_tensors) {
  std::vector<kernel::KernelTensor *> kernel_tensor_list;

  for (auto lite_tensor : lite_tensors) {
    if (lite_tensor == nullptr) {
      continue;
    }
    auto kernel_tensor_ptr = LiteTensorToKernelTensorPtr(lite_tensor);
    kernel_tensor_list.push_back(kernel_tensor_ptr);
  }

  return kernel_tensor_list;
}
}  // namespace mindspore
