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

#include "src/litert/kernel/dsp/kernel/cos.h"
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Cos;

namespace mindspore::kernel {

namespace {
constexpr size_t kInIdx = 0;
constexpr size_t kOutIdx = 0;
}  // namespace

int CosDSPKernel::Prepare() { return CheckContract(); }

int CosDSPKernel::CheckSpecs() { return CheckContract(); }

int CosDSPKernel::CheckContract() const {
  if (op_parameter_ == nullptr) {
    MS_LOG(ERROR) << "Cos kernel parameter is null.";
    return RET_ERROR;
  }
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "Cos expects 1 input, got " << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "Cos expects 1 output, got " << out_tensors_.size();
    return RET_ERROR;
  }
  auto *input = in_tensors_[kInIdx];
  auto *output = out_tensors_[kOutIdx];
  if (input == nullptr) {
    MS_LOG(ERROR) << "Cos input tensor is null.";
    return RET_ERROR;
  }
  if (output == nullptr) {
    MS_LOG(ERROR) << "Cos output tensor is null.";
    return RET_ERROR;
  }
  // cos_func reads the input with its own dtype but stores the cosine of an integer input as fp32
  // (armc_lib/tests/cases/test_cos.c reserves a float output for the int16 and int32 cases), so an
  // integer input is only accepted together with an fp32 output. Otherwise the DSP would overwrite an
  // integer buffer with floats, and the scheduler has to keep the CPU kernel instead. Checked before
  // the element counts, which are still unknown while a dynamic output shape carries a -1.
  const auto in_dtype = input->data_type();
  const auto out_dtype = output->data_type();
  const bool is_integer_input =
    (in_dtype == kNumberTypeInt8 || in_dtype == kNumberTypeInt16 || in_dtype == kNumberTypeInt32);
  if (is_integer_input && out_dtype != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Cos of an integer input writes fp32 results, but the output dtype is " << out_dtype << ".";
    return RET_ERROR;
  }
  if (!is_integer_input && out_dtype != in_dtype) {
    MS_LOG(ERROR) << "Cos output dtype " << out_dtype << " differs from the input dtype " << in_dtype << ".";
    return RET_ERROR;
  }
  const auto in_elements = input->ElementsNum();
  if (in_elements <= 0) {
    MS_LOG(ERROR) << "Cos input holds " << in_elements << " elements, nothing to compute.";
    return RET_ERROR;
  }
  if (output->ElementsNum() != in_elements) {
    MS_LOG(ERROR) << "Cos output holds " << output->ElementsNum() << " elements while the input holds " << in_elements
                  << ".";
    return RET_ERROR;
  }
  return RET_OK;
}

int CosDSPKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "Cos DSP allocator is null.";
    return RET_ERROR;
  }
  // A graph whose output shape was unknown at creation never reached CheckSpecs, and a resized
  // graph only re-runs Prepare, so the contract is settled again before any pointer goes over.
  if (CheckContract() != RET_OK) {
    return RET_ERROR;
  }
  auto *input = in_tensors_[kInIdx];
  uint64_t x_device_ptr = allocator->GetDeviceMemPtr(input->data());
  if (x_device_ptr == 0) {
    MS_LOG(ERROR) << "Cos input device ptr is null.";
    return RET_ERROR;
  }
  auto *output = out_tensors_[kOutIdx];
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(output->data());
  if (out_device_ptr == 0) {
    MS_LOG(ERROR) << "Cos output device ptr is null.";
    return RET_ERROR;
  }
  const auto length = static_cast<uint64_t>(input->ElementsNum());
  int ret = cos_func(x_device_ptr, out_device_ptr, length, static_cast<unsigned int>(core_mask_),
                     static_cast<int>(input->data_type()), static_cast<int>(GetMemType()));
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Cos run failed, ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_Cos, DSPKernelCreator<CosDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_Cos, DSPKernelCreator<CosDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_Cos, DSPKernelCreator<CosDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_Cos, DSPKernelCreator<CosDSPKernel>)
#endif

#ifdef SUPPORT_FT78
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_Cos, DSPKernelCreator<CosDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_Cos, DSPKernelCreator<CosDSPKernel>)
#endif
}  // namespace mindspore::kernel
