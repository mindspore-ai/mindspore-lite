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

#include "src/litert/kernel/dsp/kernel/activation.h"
#include "src/litert/kernel_registry.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_CELU;
using mindspore::schema::ActivationType_ELU;
using mindspore::schema::ActivationType_GELU;
using mindspore::schema::ActivationType_HARD_TANH;
using mindspore::schema::ActivationType_HSIGMOID;
using mindspore::schema::ActivationType_HSWISH;
using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::ActivationType_SIGMOID;
using mindspore::schema::ActivationType_SOFTPLUS;
using mindspore::schema::ActivationType_SWISH;
using mindspore::schema::ActivationType_TANH;
using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::kernel {

namespace {
// The kernel is registered for the whole PrimitiveType_Activation family, so the
// activation subtype has to be validated at creation time. Rejecting here lets the
// scheduler fall back to the CPU kernel instead of failing on the first Run.
bool IsSupportedActivationType(int type) {
  switch (type) {
    case ActivationType_RELU:
    case ActivationType_RELU6:
    case ActivationType_LEAKY_RELU:
    case ActivationType_SIGMOID:
    case ActivationType_TANH:
    case ActivationType_SWISH:
    case ActivationType_HSWISH:
    case ActivationType_HSIGMOID:
    case ActivationType_HARD_TANH:
    case ActivationType_GELU:
    case ActivationType_SOFTPLUS:
    case ActivationType_ELU:
    case ActivationType_CELU:
      return true;
    default:
      return false;
  }
}
}  // namespace

constexpr int kInIdx = 0;
constexpr int kOutIdx = 0;

int ActivationDSPKernel::CheckSpecs() {
  if (param_ == nullptr) {
    MS_LOG(ERROR) << "ActivationDSPKernel param is null.";
    return RET_ERROR;
  }
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_1 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "ActivationDSPKernel invalid tensor count, in size: " << in_tensors_.size()
                  << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (!IsSupportedActivationType(param_->type_)) {
    MS_LOG(ERROR) << "ActivationDSPKernel unsupported activation type: " << param_->type_;
    return RET_ERROR;
  }
  // Run passes the input element count as the length of both device buffers, so a
  // smaller output buffer would be written past its end. Reject the mismatch here.
  const int input_num = in_tensors_[kInIdx]->ElementsNum();
  const int out_num = out_tensors_[kOutIdx]->ElementsNum();
  if (input_num != out_num) {
    MS_LOG(ERROR) << "ActivationDSPKernel element count mismatch, input: " << input_num << ", out: " << out_num;
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationDSPKernel::Prepare() { return RET_OK; }

int ActivationDSPKernel::DispatchActivation(uint64_t input_device_ptr, uint64_t output_device_ptr, uint64_t length,
                                            int dtype, int mtype) {
  int ret = RET_ERROR;
  switch (param_->type_) {
    case ActivationType_RELU:
      ret = relu_func(input_device_ptr, output_device_ptr, length, core_mask_, dtype, mtype);
      break;
    case ActivationType_RELU6:
      ret = relu6_func(input_device_ptr, output_device_ptr, length, core_mask_, dtype, mtype);
      break;
    case ActivationType_LEAKY_RELU:
      ret = leaky_relu_func(input_device_ptr, output_device_ptr, length, param_->alpha_, core_mask_, dtype, mtype);
      break;
    case ActivationType_SIGMOID:
      ret = sigmoid_func(input_device_ptr, output_device_ptr, length, core_mask_, dtype, mtype);
      break;
    case ActivationType_TANH:
      ret = tanh_func(input_device_ptr, output_device_ptr, length, core_mask_, dtype, mtype);
      break;
    case ActivationType_SWISH:
      ret = swish_func(input_device_ptr, output_device_ptr, length, core_mask_, dtype, mtype);
      break;
    case ActivationType_HSWISH:
      ret = hswish_func(input_device_ptr, output_device_ptr, length, core_mask_, dtype, mtype);
      break;
    case ActivationType_HSIGMOID:
      ret = hsigmoid_func(input_device_ptr, output_device_ptr, length, core_mask_, dtype, mtype);
      break;
    case ActivationType_HARD_TANH:
      ret = hardtanh_func(input_device_ptr, output_device_ptr, length, param_->min_val_, param_->max_val_, core_mask_,
                          dtype, mtype);
      break;
    case ActivationType_GELU:
      ret = gelu_func(input_device_ptr, output_device_ptr, length, param_->approximate_, core_mask_, dtype, mtype);
      break;
    case ActivationType_SOFTPLUS:
      ret = softplus_func(input_device_ptr, output_device_ptr, length, core_mask_, dtype, mtype);
      break;
    case ActivationType_ELU:
      ret = elu_func(input_device_ptr, output_device_ptr, length, param_->alpha_, core_mask_, dtype, mtype);
      break;
    case ActivationType_CELU:
      ret = celu_func(input_device_ptr, output_device_ptr, length, param_->alpha_, core_mask_, dtype, mtype);
      break;
    default:
      MS_LOG(ERROR) << "ActivationDSPKernel unsupported activation type: " << param_->type_;
      return RET_ERROR;
  }
  return ret;
}

int ActivationDSPKernel::Run() {
  if (param_ == nullptr) {
    MS_LOG(ERROR) << "ActivationDSPKernel param is null.";
    return RET_ERROR;
  }
  auto *input = in_tensors_[kInIdx];
  auto *output = out_tensors_[kOutIdx];
  auto data_type = input->data_type();
  auto mem_type = GetMemType();
  if (dsp_runtime_ == nullptr) {
    MS_LOG(ERROR) << "ActivationDSPKernel dsp runtime is null.";
    return RET_ERROR;
  }
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "ActivationDSPKernel allocator is null.";
    return RET_ERROR;
  }
  uint64_t input_device_ptr = allocator->GetDeviceMemPtr(input->data());
  if (input_device_ptr == 0) {
    MS_LOG(ERROR) << "ActivationDSPKernel input device ptr is null.";
    return RET_ERROR;
  }
  uint64_t output_device_ptr = allocator->GetDeviceMemPtr(output->data());
  if (output_device_ptr == 0) {
    MS_LOG(ERROR) << "ActivationDSPKernel output device ptr is null.";
    return RET_ERROR;
  }
  uint64_t length = input->ElementsNum();
  int dtype = static_cast<int>(data_type);
  int mtype = static_cast<int>(mem_type);
  int ret = DispatchActivation(input_device_ptr, output_device_ptr, length, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "ActivationDSPKernel run failed, activation type: " << param_->type_;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_Activation, DSPKernelCreator<ActivationDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_Activation, DSPKernelCreator<ActivationDSPKernel>)
#endif
}  // namespace mindspore::kernel
