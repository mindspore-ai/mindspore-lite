/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel/cpu/fp32/sparse_segment_sum_fp32.h"
#include <vector>
#include <functional>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl_c/common_func.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SparseSegmentSum;

namespace mindspore::kernel {
namespace {
const uint32_t kInput_data = 0;
const uint32_t kInput_indices = 1;
const uint32_t kInput_segment_ids = 2;
const uint32_t kOutput_data = 0;

int ValidateShape(int segment_ids_num, int indices_num, const std::string &kernel_name) {
  if (segment_ids_num <= 0 || indices_num <= 0 || segment_ids_num != indices_num) {
    MS_LOG(ERROR) << "For '" << kernel_name << "', segment_ids and indices must be 1D, positive, and equal length.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ValidateSegmentIds(int segment_ids_num, const int32_t *in_segment_ids_ptr, const std::string &kernel_name) {
  if (in_segment_ids_ptr[0] != 0) {
    MS_LOG(ERROR) << "For '" << kernel_name << "', segment_ids must start from 0.";
    return RET_ERROR;
  }
  int32_t prev_id = in_segment_ids_ptr[0];
  for (int i = 1; i < segment_ids_num; i++) {
    int32_t cur_id = in_segment_ids_ptr[i];
    if (cur_id < prev_id) {
      MS_LOG(ERROR) << "For '" << kernel_name << "', segment_ids must be non-decreasing at index " << i;
      return RET_ERROR;
    }
    prev_id = cur_id;
  }
  return RET_OK;
}

int ValidateIndices(int indices_num, int input_dim0, const int32_t *in_indcie_ptr, const std::string &kernel_name) {
  for (int i = 0; i < indices_num; i++) {
    int32_t idx = in_indcie_ptr[i];
    if (idx < 0 || idx >= input_dim0) {
      MS_LOG(ERROR) << "For '" << kernel_name << "', indices[" << i << "]=" << idx << " out of range [0, " << input_dim0
                    << ").";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace
int SparseSegmentSumCPUKernel::PreProcess() { return RET_OK; }

int SparseSegmentSumCPUKernel::Prepare() { return RET_OK; }

int SparseSegmentSumCPUKernel::Run() {
  std::vector<int> in_data_shape = in_tensors_[kInput_data]->shape();
  std::vector<int> in_indcie_shape = in_tensors_[kInput_indices]->shape();
  std::vector<int> in_segment_ids_shape = in_tensors_[kInput_segment_ids]->shape();

  std::vector<int> out_data_shape;

  auto in_segment_ids_ptr = reinterpret_cast<int32_t *>(in_tensors_[kInput_segment_ids]->data());
  auto in_indcie_ptr = reinterpret_cast<int32_t *>(in_tensors_[kInput_indices]->data());
  const auto segment_ids_num = in_segment_ids_shape[0];
  const auto indices_num = in_indcie_shape[0];
  const auto input_dim0 = in_data_shape[0];
  if (ValidateShape(segment_ids_num, indices_num, this->name_) != RET_OK) {
    return RET_ERROR;
  }
  if (ValidateSegmentIds(segment_ids_num, in_segment_ids_ptr, this->name_) != RET_OK) {
    return RET_ERROR;
  }
  if (ValidateIndices(indices_num, input_dim0, in_indcie_ptr, this->name_) != RET_OK) {
    return RET_ERROR;
  }

  out_data_shape.emplace_back(in_segment_ids_ptr[segment_ids_num - 1] + 1);
  for (size_t i = 1; i < in_data_shape.size(); i++) {
    out_data_shape.emplace_back(in_data_shape[i]);
  }

  out_tensors_.at(kOutput_data)->set_shape(out_data_shape);
  out_tensors_.at(kOutput_data)->FreeData();

  constexpr size_t kMultiply = 1;
  size_t n =
    std::accumulate(in_data_shape.begin(), in_data_shape.end(), kMultiply, std::multiplies<int>()) / in_data_shape[0];
  size_t m =
    std::accumulate(in_segment_ids_shape.begin(), in_segment_ids_shape.end(), kMultiply, std::multiplies<int>());
  int oldindex = -1;

  int32_t *in_data_ptr_int32 = nullptr;
  int32_t *out_data_ptr_int32 = nullptr;
  float *in_data_ptr_fp32 = nullptr;
  float *out_data_ptr_fp32 = nullptr;

  auto input_data_type = in_tensors_[kInput_data]->data_type();

  switch (input_data_type) {
    case kNumberTypeInt32:
      in_data_ptr_int32 = reinterpret_cast<int32_t *>(in_tensors_[kInput_data]->data());
      out_data_ptr_int32 = reinterpret_cast<int32_t *>(out_tensors_[kOutput_data]->MutableData());
      for (size_t i = 0; i < m; i++) {
        if (oldindex != in_segment_ids_ptr[i]) {
          oldindex = in_segment_ids_ptr[i];
          for (size_t j = 0; j < n; j++) {
            out_data_ptr_int32[j + oldindex * n] = 0;
          }
        }
        for (size_t j = 0; j < n; j++) {
          out_data_ptr_int32[j + oldindex * n] += in_data_ptr_int32[j + in_indcie_ptr[i] * n];
        }
      }
      break;
    case kNumberTypeFloat32:
      in_data_ptr_fp32 = reinterpret_cast<float *>(in_tensors_[kInput_data]->data());
      out_data_ptr_fp32 = reinterpret_cast<float *>(out_tensors_[kOutput_data]->MutableData());
      for (size_t i = 0; i < m; i++) {
        if (oldindex != in_segment_ids_ptr[i]) {
          oldindex = in_segment_ids_ptr[i];
          for (size_t j = 0; j < n; j++) {
            out_data_ptr_fp32[j + oldindex * n] = 0;
          }
        }
        for (size_t j = 0; j < n; j++) {
          out_data_ptr_fp32[j + oldindex * n] += in_data_ptr_fp32[j + in_indcie_ptr[i] * n];
        }
      }
      break;
    default:
      MS_LOG(ERROR) << "Unsupported data type: " << input_data_type << " of SparseFillEmptyRows cpu kernel.";
      return RET_ERROR;
  }

  for (auto *output : this->out_tensors()) {
    output->ResetRefCount();
  }

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_SparseSegmentSum, LiteKernelCreator<SparseSegmentSumCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SparseSegmentSum, LiteKernelCreator<SparseSegmentSumCPUKernel>)
}  // namespace mindspore::kernel
