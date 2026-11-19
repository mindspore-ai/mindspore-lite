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

#include "src/litert/kernel/dsp/kernel/clip.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/cpu/nnacl_c/clip_parameter.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Clip;

namespace mindspore::kernel {

namespace {
constexpr size_t kInIdx = 0;
constexpr size_t kOutIdx = 0;
}  // namespace

int ClipDSPKernel::Prepare() { return CheckContract(); }

int ClipDSPKernel::CheckSpecs() { return CheckContract(); }

int ClipDSPKernel::CheckContract() const {
  if (op_parameter_ == nullptr) {
    MS_LOG(ERROR) << "Clip kernel parameter is null.";
    return RET_ERROR;
  }
  // A Clip whose bounds arrive as the second and third inputs cannot be answered here: clip_func
  // takes the bounds as two host floats, and reading a device tensor would mean copying it first.
  // The CPU kernel keeps that form, so this one declines and the scheduler stays on CPU. A designed
  // decline is a warning, the way exp.cc and sub.cc report the same thing, so that a graph landing on
  // the CPU kernel does not read as a failure in the log.
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "Clip with tensor bounds is not supported by the DSP kernel, it expects 1 input, got "
                    << in_tensors_.size() << ".";
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "Clip expects 1 output, got " << out_tensors_.size() << ".";
    return RET_ERROR;
  }
  auto *input = in_tensors_[kInIdx];
  auto *output = out_tensors_[kOutIdx];
  if (input == nullptr) {
    MS_LOG(ERROR) << "Clip input tensor is null.";
    return RET_ERROR;
  }
  if (output == nullptr) {
    MS_LOG(ERROR) << "Clip output tensor is null.";
    return RET_ERROR;
  }
  // clip_func stores every element as the dtype it read, so the output must be of that dtype.
  if (output->data_type() != input->data_type()) {
    MS_LOG(ERROR) << "Clip output dtype " << output->data_type() << " differs from the input dtype "
                  << input->data_type() << ".";
    return RET_ERROR;
  }
  const auto in_elements = input->ElementsNum();
  if (in_elements <= 0) {
    MS_LOG(ERROR) << "Clip input holds " << in_elements << " elements, nothing to compute.";
    return RET_ERROR;
  }
  if (output->ElementsNum() != in_elements) {
    MS_LOG(ERROR) << "Clip output holds " << output->ElementsNum() << " elements while the input holds " << in_elements
                  << ".";
    return RET_ERROR;
  }
  // Rejected the same way the CPU kernel rejects it, and the negated form also catches a bound that
  // is not a number, which no comparison of min to max would catch on its own.
  auto *param = reinterpret_cast<ClipParameter *>(op_parameter_);
  if (!(param->min_val_ < param->max_val_)) {
    MS_LOG(ERROR) << "Clip min " << param->min_val_ << " must be below max " << param->max_val_ << ".";
    return RET_ERROR;
  }
  return RET_OK;
}

int ClipDSPKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "Clip DSP allocator is null.";
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
    MS_LOG(ERROR) << "Clip input device ptr is null.";
    return RET_ERROR;
  }
  auto *output = out_tensors_[kOutIdx];
  uint64_t out_device_ptr = allocator->GetDeviceMemPtr(output->data());
  if (out_device_ptr == 0) {
    MS_LOG(ERROR) << "Clip output device ptr is null.";
    return RET_ERROR;
  }
  const auto length = static_cast<uint64_t>(input->ElementsNum());
  auto *param = reinterpret_cast<ClipParameter *>(op_parameter_);
  int ret = clip_func(x_device_ptr, out_device_ptr, length, param->min_val_, param->max_val_,
                      static_cast<unsigned int>(core_mask_), static_cast<int>(input->data_type()),
                      static_cast<int>(GetMemType()));
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Clip run failed, ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_Clip, DSPKernelCreator<ClipDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_Clip, DSPKernelCreator<ClipDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_Clip, DSPKernelCreator<ClipDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_Clip, DSPKernelCreator<ClipDSPKernel>)
#endif

#ifdef SUPPORT_FT78
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_Clip, DSPKernelCreator<ClipDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_Clip, DSPKernelCreator<ClipDSPKernel>)
#endif
}  // namespace mindspore::kernel
