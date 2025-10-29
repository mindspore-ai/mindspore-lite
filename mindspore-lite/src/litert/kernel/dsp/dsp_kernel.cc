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

#include "src/litert/kernel/dsp/dsp_kernel.h"
#include "src/litert/infer_manager.h"
#include "src/litert/weight_decoder.h"
#include "src/common/file_utils.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
bool DSPKernel::MallocDataDone() {
  for (auto &out_tensor : out_tensors_) {
    if (out_tensor->data() == nullptr) {
      return false;
    }
    auto allocator = out_tensor->allocator();
    if (allocator == nullptr) {
      return false;
    }
    auto buffer =
      reinterpret_cast<mindspore::lite::dsp::DSPAllocator *>(allocator.get())->GetDeviceMemPtr(out_tensor->data());
    if (buffer == 0) {
      return false;
    }
  }
  return true;
}

int DSPKernel::PreProcess() {
  if (MallocDataDone()) {
    return RET_OK;
  }
  int ret = ReSize();
  if (ret != RET_OK) {
    return ret;
  }
  for (size_t i = 0; i < out_tensors_.size(); ++i) {
    auto *output = out_tensors_.at(i);
    CHECK_NULL_RETURN(output);
    CHECK_NULL_RETURN(output->allocator());
    ret = output->MallocData();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "MallocData failed";
      return ret;
    }
    output->ResetRefCount();
  }
  return RET_OK;
}

int DSPKernel::InferShape() {
  if (InferShapeDone()) {
    return RET_OK;
  }
  auto ret = lite::KernelInferShape(in_tensors_, out_tensors_, op_parameter_);
  if (ret != RET_OK) {
    MS_LOG(WARNING) << "InferShape failed, type: "
                    << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(type()));
    return ret;
  }
  return RET_OK;
}

int DSPKernel::ReSize() {
  if (InferShapeDone()) {
    return RET_OK;
  }
  auto ret = InferShape();
  if (ret != RET_OK) {
    return ret;
  }

  ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ReSize failed for kernel prepare!";
    return ret;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
