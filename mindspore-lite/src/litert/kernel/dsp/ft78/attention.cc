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

#include "src/litert/kernel/dsp/ft78/attention.h"
#include <cstdint>
#include "src/litert/kernel_registry.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel/cpu/nnacl_c/attention_parameter.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Attention;

namespace mindspore::kernel {
namespace {
constexpr int kAttentionArgsCount = 8;
}

int AttentionDSPKernel::Prepare() { return RET_OK; }

int AttentionDSPKernel::CheckSpecs() {
  if (in_tensors_.size() < INPUT_TENSOR_SIZE_4) {
    MS_LOG(WARNING) << "Attention expects 4 inputs (Q, K, V, args), got " << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "Attention expects exactly 1 output, got " << out_tensors_.size();
    return RET_ERROR;
  }
  auto dtype = in_tensors_[0]->data_type();
  if (dtype != kNumberTypeFloat32) {
    MS_LOG(WARNING) << "Attention FT78 only supports float32, got dtype " << static_cast<int>(dtype);
    return RET_ERROR;
  }
  if (in_tensors_[1]->data_type() != dtype || in_tensors_[2]->data_type() != dtype) {
    MS_LOG(WARNING) << "Attention input tensors must share the same dtype.";
    return RET_ERROR;
  }
  if (out_tensors_[0]->data_type() != dtype) {
    MS_LOG(WARNING) << "Attention output dtype mismatch.";
    return RET_ERROR;
  }
  auto args_tensor = in_tensors_[INPUT_TENSOR_SIZE_3];
  if (args_tensor == nullptr || args_tensor->data_type() != kNumberTypeInt32) {
    MS_LOG(WARNING) << "Attention args tensor must be int32.";
    return RET_ERROR;
  }
  if (args_tensor->ElementsNum() < kAttentionArgsCount) {
    MS_LOG(WARNING) << "Attention args tensor expects at least " << kAttentionArgsCount << " elements.";
    return RET_ERROR;
  }
  if (DeriveAttentionShape(nullptr, nullptr, nullptr, nullptr) != RET_OK) {
    MS_LOG(WARNING) << "Attention FT78 failed to derive shape parameters.";
    return RET_ERROR;
  }
  return RET_OK;
}

int AttentionDSPKernel::DeriveAttentionShape(int *batch, int *seq_len, int *head_num, int *head_dim) const {
  if (in_tensors_.empty()) {
    MS_LOG(ERROR) << "Attention tensors are empty.";
    return RET_ERROR;
  }
  const auto &q_shape = in_tensors_[0]->shape();
  if (q_shape.size() < DIMENSION_3D) {
    MS_LOG(ERROR) << "Attention expects Q tensor rank >= 3, got " << q_shape.size();
    return RET_ERROR;
  }
  int batch_size = q_shape[0];
  int query_seq_len = q_shape[1];
  int derived_head_num = 0;
  int derived_head_dim = 0;
  auto *param = reinterpret_cast<AttentionParameter *>(op_parameter_);
  if (param != nullptr) {
    derived_head_num = param->head_num_;
    derived_head_dim = param->head_size_;
  }
  if (q_shape.size() >= DIMENSION_4D) {
    if (derived_head_num <= 0) {
      derived_head_num = q_shape[2];
    }
    if (derived_head_dim <= 0) {
      derived_head_dim = q_shape[3];
    }
  }
  if (derived_head_num <= 0 || derived_head_dim <= 0) {
    MS_LOG(ERROR) << "Invalid head parameters: head_num=" << derived_head_num << ", head_dim=" << derived_head_dim;
    return RET_ERROR;
  }
  if (batch_size <= 0 || query_seq_len <= 0) {
    MS_LOG(ERROR) << "Invalid batch or sequence length: batch=" << batch_size << ", seq=" << query_seq_len;
    return RET_ERROR;
  }
  if (batch != nullptr) {
    *batch = batch_size;
  }
  if (seq_len != nullptr) {
    *seq_len = query_seq_len;
  }
  if (head_num != nullptr) {
    *head_num = derived_head_num;
  }
  if (head_dim != nullptr) {
    *head_dim = derived_head_dim;
  }
  return RET_OK;
}

int AttentionDSPKernel::RunFp32() {
  kernel_name_ = "fp_attention_s";
  core_mask_ = 0xff;
  return dsp_runtime_->RunKernel(kernel_name_, kernel_args_, core_mask_);
}

int AttentionDSPKernel::Run() {
  if (in_tensors_.size() < INPUT_TENSOR_SIZE_4) {
    MS_LOG(ERROR) << "Attention requires inputs (Q, K, V, args).";
    return RET_ERROR;
  }
  if (in_tensors_[0]->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Attention currently supports float32 only.";
    return RET_ERROR;
  }
  auto args_tensor = in_tensors_[INPUT_TENSOR_SIZE_3];
  if (args_tensor == nullptr) {
    MS_LOG(ERROR) << "Attention args tensor is null.";
    return RET_ERROR;
  }
  if (args_tensor->data_type() != kNumberTypeInt32) {
    MS_LOG(ERROR) << "Attention args tensor must be int32.";
    return RET_ERROR;
  }
  if (args_tensor->data() == nullptr) {
    MS_LOG(ERROR) << "Attention args tensor has no data.";
    return RET_ERROR;
  }
  if (args_tensor->ElementsNum() < kAttentionArgsCount) {
    MS_LOG(ERROR) << "Attention args tensor expects at least " << kAttentionArgsCount << " elements.";
    return RET_ERROR;
  }
  auto *args_data = reinterpret_cast<const int32_t *>(args_tensor->data());
  int batch = 0;
  int seq_len = 0;
  int head_num = 0;
  int head_dim = 0;
  if (DeriveAttentionShape(&batch, &seq_len, &head_num, &head_dim) != RET_OK) {
    MS_LOG(ERROR) << "Failed to derive attention shapes.";
    return RET_ERROR;
  }
  if (args_data[0] != batch || args_data[1] != seq_len || args_data[2] != head_num || args_data[3] != head_dim) {
    MS_LOG(WARNING) << "Attention args tensor shape metadata mismatch: provided=" << args_data[0] << "," << args_data[1]
                    << "," << args_data[2] << "," << args_data[3] << " derived=" << batch << "," << seq_len << ","
                    << head_num << "," << head_dim;
  }
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "Attention failed to get DSP allocator.";
    return RET_ERROR;
  }
  uint64_t q_device_ptr = allocator->GetDeviceMemPtr(in_tensors_[0]->data());
  uint64_t k_device_ptr = allocator->GetDeviceMemPtr(in_tensors_[1]->data());
  uint64_t v_device_ptr = allocator->GetDeviceMemPtr(in_tensors_[2]->data());
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(out_tensors_[0]->data());
  uint64_t args_device_ptr = allocator->GetDeviceMemPtr(args_tensor->data());
  if (q_device_ptr == 0 || k_device_ptr == 0 || v_device_ptr == 0 || out_device_ptr == 0 || args_device_ptr == 0) {
    MS_LOG(ERROR) << "Attention encountered invalid device pointer.";
    return RET_ERROR;
  }

  SetKernelArg({q_device_ptr, k_device_ptr, v_device_ptr, out_device_ptr, args_device_ptr});

  int ret = RunFp32();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Attention DSP run failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_Attention, DSPKernelCreator<AttentionDSPKernel>)

}  // namespace mindspore::kernel
