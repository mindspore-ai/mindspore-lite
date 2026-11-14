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

#include "src/litert/kernel/dsp/kernel/activation_grad.h"
#include "src/litert/kernel/cpu/nnacl_c/fp32_grad/activation_grad_fp32.h"
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
using mindspore::schema::ActivationType_TANH;
using mindspore::schema::PrimitiveType_ActivationGrad;

namespace mindspore::kernel {

namespace {
// The kernel is registered for the whole PrimitiveType_ActivationGrad family, so the
// activation subtype has to be validated at creation time. Rejecting here lets the
// scheduler fall back to the CPU kernel instead of failing on the first Run.
bool IsSupportedActivationType(int type) {
  switch (type) {
    case ActivationType_RELU:
    case ActivationType_RELU6:
    case ActivationType_LEAKY_RELU:
    case ActivationType_SIGMOID:
    case ActivationType_TANH:
    case ActivationType_HSWISH:
    case ActivationType_HSIGMOID:
    case ActivationType_GELU:
    case ActivationType_ELU:
    case ActivationType_SOFTPLUS:
    case ActivationType_CELU:
    case ActivationType_HARD_TANH:
      return true;
    default:
      return false;
  }
}
}  // namespace

constexpr int kGradIdx = 0;
constexpr int kInputIdx = 1;
constexpr int kOutIdx = 0;
constexpr float kHardTanhDefaultMin = -1.0f;
constexpr float kHardTanhDefaultMax = 1.0f;

int ActivationGradDSPKernel::CheckSpecs() {
  if (param_ == nullptr) {
    MS_LOG(ERROR) << "ActivationGradDSPKernel param is null.";
    return RET_ERROR;
  }
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_2 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "ActivationGradDSPKernel invalid tensor count, in size: " << in_tensors_.size()
                  << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (!IsSupportedActivationType(param_->type_)) {
    MS_LOG(ERROR) << "ActivationGradDSPKernel unsupported activation type: " << param_->type_;
    return RET_ERROR;
  }
  // DispatchActivationGrad drives the grad/input/output device buffers with a single
  // length, so mismatched element counts would make the device read or write out of
  // bounds. Reject them here so the scheduler can fall back to CPU.
  const int grad_num = in_tensors_[kGradIdx]->ElementsNum();
  const int input_num = in_tensors_[kInputIdx]->ElementsNum();
  const int out_num = out_tensors_[kOutIdx]->ElementsNum();
  if (grad_num != input_num || input_num != out_num) {
    MS_LOG(ERROR) << "ActivationGradDSPKernel element count mismatch, grad: " << grad_num << ", input: " << input_num
                  << ", out: " << out_num;
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationGradDSPKernel::Prepare() { return RET_OK; }

int ActivationGradDSPKernel::DispatchActivationGrad(uint64_t grad_device_ptr, uint64_t input_device_ptr,
                                                    uint64_t output_device_ptr, int dtype, int mtype) {
  int ret = RET_ERROR;
  uint64_t length = in_tensors_[kInputIdx]->ElementsNum();
  switch (param_->type_) {
    case ActivationType_RELU:
      ret = relu_grad_func(grad_device_ptr, input_device_ptr, output_device_ptr, length, core_mask_, dtype, mtype);
      break;
    case ActivationType_RELU6:
      ret = relu6_grad_func(grad_device_ptr, input_device_ptr, output_device_ptr, length, core_mask_, dtype, mtype);
      break;
    case ActivationType_LEAKY_RELU:
      ret = lrelu_grad_func(grad_device_ptr, input_device_ptr, output_device_ptr, length, param_->alpha_, core_mask_,
                            dtype, mtype);
      break;
    case ActivationType_SIGMOID:
      ret = sigmoid_grad_func(grad_device_ptr, input_device_ptr, output_device_ptr, length, core_mask_, dtype, mtype);
      break;
    case ActivationType_TANH:
      ret = tanh_grad_func(grad_device_ptr, input_device_ptr, output_device_ptr, length, core_mask_, dtype, mtype);
      break;
    case ActivationType_HSWISH:
      ret = hswish_grad_func(grad_device_ptr, input_device_ptr, output_device_ptr, length, core_mask_, dtype, mtype);
      break;
    case ActivationType_HSIGMOID:
      ret = hsigmoid_grad_func(grad_device_ptr, input_device_ptr, output_device_ptr, length, core_mask_, dtype, mtype);
      break;
    case ActivationType_GELU:
      ret = gelu_grad_func(grad_device_ptr, input_device_ptr, output_device_ptr, length, core_mask_, dtype, mtype);
      break;
    case ActivationType_ELU:
      ret = elu_grad_func(grad_device_ptr, input_device_ptr, output_device_ptr, length, param_->alpha_, core_mask_,
                          dtype, mtype);
      break;
    case ActivationType_SOFTPLUS:
      ret = softplus_grad_func(grad_device_ptr, input_device_ptr, output_device_ptr, length, core_mask_, dtype, mtype);
      break;
    case ActivationType_CELU:
      ret = celu_grad_func(grad_device_ptr, input_device_ptr, output_device_ptr, length, param_->alpha_, core_mask_,
                           dtype, mtype);
      break;
    case ActivationType_HARD_TANH:
      ret = hardtanh_grad_func(grad_device_ptr, input_device_ptr, output_device_ptr, length, kHardTanhDefaultMin,
                               kHardTanhDefaultMax, core_mask_, dtype, mtype);
      break;
    default:
      MS_LOG(ERROR) << "ActivationGradDSPKernel unsupported activation type: " << param_->type_;
      return RET_ERROR;
  }
  return ret;
}

int ActivationGradDSPKernel::Run() {
  if (param_ == nullptr) {
    MS_LOG(ERROR) << "ActivationGradDSPKernel param is null.";
    return RET_ERROR;
  }
  auto *grad_tensor = in_tensors_[kGradIdx];
  auto *input_tensor = in_tensors_[kInputIdx];
  auto *output = out_tensors_[kOutIdx];
  auto data_type = input_tensor->data_type();
  auto mem_type = GetMemType();
  auto allocator = dsp_runtime_->GetAllocator();
  uint64_t grad_device_ptr = allocator->GetDeviceMemPtr(grad_tensor->data());
  if (grad_device_ptr == 0) {
    MS_LOG(ERROR) << "ActivationGradDSPKernel grad device ptr is null.";
    return RET_ERROR;
  }
  uint64_t input_device_ptr = allocator->GetDeviceMemPtr(input_tensor->data());
  if (input_device_ptr == 0) {
    MS_LOG(ERROR) << "ActivationGradDSPKernel input device ptr is null.";
    return RET_ERROR;
  }
  uint64_t output_device_ptr = allocator->GetDeviceMemPtr(output->data());
  if (output_device_ptr == 0) {
    MS_LOG(ERROR) << "ActivationGradDSPKernel output device ptr is null.";
    return RET_ERROR;
  }
  int dtype = static_cast<int>(data_type);
  int mtype = static_cast<int>(mem_type);
  int ret = DispatchActivationGrad(grad_device_ptr, input_device_ptr, output_device_ptr, dtype, mtype);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "ActivationGradDSPKernel run failed, activation type: " << param_->type_;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_ActivationGrad, DSPKernelCreator<ActivationGradDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_ActivationGrad, DSPKernelCreator<ActivationGradDSPKernel>)
#endif
}  // namespace mindspore::kernel
