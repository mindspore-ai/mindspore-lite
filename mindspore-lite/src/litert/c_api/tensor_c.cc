/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "include/c_api/tensor_c.h"
#include "include/api/status.h"
#include "src/tensor.h"
#include "src/litert/cxx_api/tensor/tensor_impl.h"
#include "src/litert/inner_allocator.h"

MSTensorHandle MSTensorCreate(const char *name, MSDataType type, const int64_t *shape, size_t shape_num,
                              const void *data, size_t data_len) {
  if (name == nullptr || shape == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  std::vector<int32_t> vec_shape(shape_num);
  for (size_t i = 0; i < shape_num; i++) {
    vec_shape[i] = shape[i];
  }
  auto lite_tensor =
    mindspore::lite::Tensor::CreateTensor(name, static_cast<mindspore::TypeId>(type), vec_shape, data, data_len);
  auto lite_tensor_impl = std::make_shared<mindspore::LiteTensorImpl>(lite_tensor);
  if (lite_tensor_impl == nullptr || lite_tensor_impl->lite_tensor() == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate tensor impl.";
    return nullptr;
  }
  lite_tensor_impl->set_from_session(false);
  auto impl = new (std::nothrow) mindspore::MSTensor(lite_tensor_impl);
  if (impl == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate MSTensor.";
    return nullptr;
  }
  return impl;
}

void MSTensorDestroy(MSTensorHandle *tensor) {
  if (tensor == nullptr || *tensor == nullptr) {
    MS_LOG(ERROR) << "tensor is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::MSTensor *>(*tensor);
  delete impl;
  *tensor = nullptr;
}

MSTensorHandle MSTensorClone(MSTensorHandle tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  auto clone_impl = impl->Clone();
  if (clone_impl == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate tensor impl.";
    return nullptr;
  }
  std::static_pointer_cast<mindspore::LiteTensorImpl>(clone_impl->impl())->set_own_data(false);
  clone_impl->SetTensorName(impl->Name() + "_duplicate");
  return clone_impl;
}

void MSTensorSetName(MSTensorHandle tensor, const char *name) {
  if (tensor == nullptr || name == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  impl->SetTensorName(name);
}

const char *MSTensorGetName(const MSTensorHandle tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  auto ms_tensor = static_cast<mindspore::MSTensor *>(tensor);
  return std::static_pointer_cast<mindspore::LiteTensorImpl>(ms_tensor->impl())->Name().c_str();
}

void MSTensorSetDataType(MSTensorHandle tensor, MSDataType type) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  impl->SetDataType(static_cast<mindspore::DataType>(type));
}

MSDataType MSTensorGetDataType(const MSTensorHandle tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return kMSDataTypeUnknown;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  auto dtype = impl->DataType();
  return static_cast<MSDataType>(dtype);
}

void MSTensorSetShape(MSTensorHandle tensor, const int64_t *shape, size_t shape_num) {
  if (tensor == nullptr || shape == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  std::vector<int64_t> vec_shape(shape_num);
  for (size_t i = 0; i < shape_num; i++) {
    vec_shape[i] = shape[i];
  }
  impl->SetShape(vec_shape);
}

const int64_t *MSTensorGetShape(const MSTensorHandle tensor, size_t *shape_num) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  *shape_num = impl->Shape().size();
  return impl->Shape().data();
}

void MSTensorSetFormat(MSTensorHandle tensor, MSFormat format) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  return impl->SetFormat(static_cast<mindspore::Format>(format));
}

MSFormat MSTensorGetFormat(const MSTensorHandle tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return kMSFormatNHWC;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  return static_cast<MSFormat>(impl->format());
}

void MSTensorSetData(MSTensorHandle tensor, void *data) {
  if (tensor == nullptr || data == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  return impl->SetData(data, true);
}

const void *MSTensorGetData(const MSTensorHandle tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  return impl->Data().get();
}

void *MSTensorGetMutableData(const MSTensorHandle tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  return impl->MutableData();
}

int64_t MSTensorGetElementNum(const MSTensorHandle tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return 0;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  return impl->ElementNum();
}

size_t MSTensorGetDataSize(const MSTensorHandle tensor) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return 0;
  }
  auto impl = static_cast<mindspore::MSTensor *>(tensor);
  return impl->DataSize();
}
