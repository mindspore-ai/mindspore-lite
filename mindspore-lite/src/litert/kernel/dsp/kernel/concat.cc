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

#include "src/litert/kernel/dsp/kernel/concat.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/cpu/nnacl_c/concat_parameter.h"
#include "armc/include/operator.h"

using mindspore::kernel::KERNEL_ARCH::kDSP;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Concat;

namespace mindspore::kernel {

namespace {
constexpr size_t kFirstInput = 0;
constexpr size_t kOutIdx = 0;
// armc_lib/tests/cases/test_concat.c exercises up to eight inputs, and that is the envelope this
// kernel commits to. Beyond it the kernel refuses, which is what keeps such a graph on the CPU kernel.
constexpr size_t kMaxConcatInputs = 8;
}  // namespace

int ConcatDSPKernel::Prepare() { return CheckContract(); }

int ConcatDSPKernel::CheckSpecs() { return CheckContract(); }

int ConcatDSPKernel::CheckContract() {
  if (ResolveGeometry() != RET_OK) {
    return RET_ERROR;
  }
  return CheckQuantParams();
}

// concat_func moves bytes around; it cannot requantize. An int8 join is only correct when every input
// carries the scale and zero point the output is read with, which is the same pair the CPU int8 kernel
// pulls from the tensors in concat_int8.cc. An exact match is what is required, not a close one: two
// scales that merely round alike would still put different numbers in the same buffer.
int ConcatDSPKernel::CheckQuantParams() const {
  if (in_tensors_[kFirstInput]->data_type() != kNumberTypeInt8) {
    return RET_OK;
  }
  auto *param = reinterpret_cast<ConcatParameter *>(op_parameter_);
  const auto *in_args = param->quant_arg_.in_args_;
  double output_scale = 0.0;
  int32_t output_zero_point = 0;
  if (in_args != nullptr) {
    output_scale = param->quant_arg_.out_args_.scale_;
    output_zero_point = param->quant_arg_.out_args_.zp_;
  } else {
    const auto out_quant = out_tensors_[kOutIdx]->quant_params();
    if (out_quant.empty()) {
      MS_LOG(ERROR) << "Concat int8 output carries no quantization parameters to join against.";
      return RET_ERROR;
    }
    output_scale = out_quant.front().scale;
    output_zero_point = out_quant.front().zeroPoint;
  }
  for (size_t i = 0; i < in_tensors_.size(); ++i) {
    double scale = 0.0;
    int32_t zero_point = 0;
    if (in_args != nullptr) {
      scale = in_args[i].scale_;
      zero_point = in_args[i].zp_;
    } else {
      const auto in_quant = in_tensors_[i]->quant_params();
      // The CPU int8 kernel refuses an int8 tensor without quant parameters too, so agreeing is not
      // even decidable here and the join belongs on the CPU.
      if (in_quant.empty()) {
        MS_LOG(ERROR) << "Concat int8 input " << i << " carries no quantization parameters.";
        return RET_ERROR;
      }
      scale = in_quant.front().scale;
      zero_point = in_quant.front().zeroPoint;
    }
    if (scale != output_scale || zero_point != output_zero_point) {
      MS_LOG(ERROR) << "Concat input " << i << " is quantized with scale " << scale << " and zero point " << zero_point
                    << " while the output uses scale " << output_scale << " and zero point " << output_zero_point
                    << ". The DSP cannot requantize, so the join stays on the CPU.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ConcatDSPKernel::ResolveGeometry() {
  if (op_parameter_ == nullptr) {
    MS_LOG(ERROR) << "Concat kernel parameter is null.";
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(ERROR) << "Concat expects 1 output, got " << out_tensors_.size() << ".";
    return RET_ERROR;
  }
  auto *output = out_tensors_[kOutIdx];
  if (output == nullptr) {
    MS_LOG(ERROR) << "Concat output tensor is null.";
    return RET_ERROR;
  }
  const auto input_count = in_tensors_.size();
  if (input_count < INPUT_TENSOR_SIZE_2) {
    MS_LOG(ERROR) << "Concat joins at least 2 inputs, got " << input_count << ".";
    return RET_ERROR;
  }
  if (input_count > kMaxConcatInputs) {
    MS_LOG(ERROR) << "Concat supports up to " << kMaxConcatInputs << " inputs, got " << input_count << ".";
    return RET_ERROR;
  }
  auto *first = in_tensors_[kFirstInput];
  if (first == nullptr) {
    MS_LOG(ERROR) << "Concat first input tensor is null.";
    return RET_ERROR;
  }
  const auto &first_shape = first->shape();
  rank_ = static_cast<int>(first_shape.size());
  if (rank_ <= 0) {
    MS_LOG(ERROR) << "Concat input has no rank.";
    return RET_ERROR;
  }
  // armc normalizes a negative axis the same way, against the rank of the first input.
  auto *param = reinterpret_cast<ConcatParameter *>(op_parameter_);
  axis_ = (param->axis_ < 0) ? param->axis_ + rank_ : param->axis_;
  if (axis_ < 0 || axis_ >= rank_) {
    MS_LOG(ERROR) << "Concat axis " << param->axis_ << " is outside the input rank " << rank_ << ".";
    return RET_ERROR;
  }
  // concat_func reads one shape per input, laid out one after another, so the table has to be built
  // in input order and stay alive until the launch returns.
  input_shapes_.clear();
  input_shapes_.reserve(input_count * rank_);
  int axis_extent = 0;
  for (size_t i = 0; i < input_count; ++i) {
    auto *tensor = in_tensors_[i];
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Concat input " << i << " is null.";
      return RET_ERROR;
    }
    if (tensor->data_type() != first->data_type()) {
      MS_LOG(ERROR) << "Concat input " << i << " dtype " << tensor->data_type() << " differs from the first input "
                    << first->data_type() << ".";
      return RET_ERROR;
    }
    const auto &shape = tensor->shape();
    if (static_cast<int>(shape.size()) != rank_) {
      MS_LOG(ERROR) << "Concat input " << i << " has rank " << shape.size() << " but input 0 has rank " << rank_ << ".";
      return RET_ERROR;
    }
    for (int dim = 0; dim < rank_; ++dim) {
      if (shape[dim] <= 0) {
        MS_LOG(ERROR) << "Concat input " << i << " dimension " << dim << " is " << shape[dim] << ", which is not a "
                      << "shape that can be concatenated.";
        return RET_ERROR;
      }
      if (dim == axis_) {
        axis_extent += shape[dim];
      } else if (shape[dim] != first_shape[dim]) {
        MS_LOG(ERROR) << "Concat input " << i << " dimension " << dim << " is " << shape[dim] << " but input 0 has "
                      << first_shape[dim] << ".";
        return RET_ERROR;
      }
      input_shapes_.push_back(shape[dim]);
    }
  }
  return CheckOutput(output, axis_extent);
}

int ConcatDSPKernel::CheckOutput(const lite::Tensor *output, int axis_extent) const {
  auto *first = in_tensors_[kFirstInput];
  if (output->data_type() != first->data_type()) {
    MS_LOG(ERROR) << "Concat output dtype " << output->data_type() << " differs from the input dtype "
                  << first->data_type() << ".";
    return RET_ERROR;
  }
  // The output has to be exactly the concatenation of the inputs. Anything else means the DSP would
  // write past the buffer, or leave part of it untouched.
  const auto &first_shape = first->shape();
  const auto &out_shape = output->shape();
  if (static_cast<int>(out_shape.size()) != rank_) {
    MS_LOG(ERROR) << "Concat output has rank " << out_shape.size() << " but the inputs have rank " << rank_ << ".";
    return RET_ERROR;
  }
  for (int dim = 0; dim < rank_; ++dim) {
    const int expected = (dim == axis_) ? axis_extent : first_shape[dim];
    if (out_shape[dim] != expected) {
      MS_LOG(ERROR) << "Concat output dimension " << dim << " is " << out_shape[dim] << " but the inputs join to "
                    << expected << ".";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ConcatDSPKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto allocator = dsp_runtime_->GetAllocator();
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "Concat DSP allocator is null.";
    return RET_ERROR;
  }
  // A graph whose output shape was unknown at creation never reached CheckSpecs, and a resized graph
  // may skip Prepare altogether, so the shape table and the quantization parameters are settled again
  // before anything is handed over.
  if (CheckContract() != RET_OK) {
    return RET_ERROR;
  }
  input_ptrs_.clear();
  input_ptrs_.reserve(in_tensors_.size());
  for (size_t i = 0; i < in_tensors_.size(); ++i) {
    const uint64_t device_ptr = allocator->GetDeviceMemPtr(in_tensors_[i]->data());
    if (device_ptr == 0) {
      MS_LOG(ERROR) << "Concat input " << i << " device ptr is null.";
      return RET_ERROR;
    }
    input_ptrs_.push_back(device_ptr);
  }
  const uint64_t out_device_ptr = allocator->GetDeviceMemPtr(out_tensors_[kOutIdx]->data());
  if (out_device_ptr == 0) {
    MS_LOG(ERROR) << "Concat output device ptr is null.";
    return RET_ERROR;
  }
  auto *first = in_tensors_[kFirstInput];
  int ret = concat_func(input_ptrs_.data(), input_shapes_.data(), static_cast<int>(in_tensors_.size()), axis_, rank_,
                        out_device_ptr, static_cast<unsigned int>(core_mask_), static_cast<int>(first->data_type()),
                        static_cast<int>(GetMemType()));
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Concat run failed, ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kDSP, kNumberTypeInt16, PrimitiveType_Concat, DSPKernelCreator<ConcatDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeInt32, PrimitiveType_Concat, DSPKernelCreator<ConcatDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat32, PrimitiveType_Concat, DSPKernelCreator<ConcatDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex64, PrimitiveType_Concat, DSPKernelCreator<ConcatDSPKernel>)

#ifdef SUPPORT_FT04
REG_KERNEL(kDSP, kNumberTypeFloat16, PrimitiveType_Concat, DSPKernelCreator<ConcatDSPKernel>)
#endif

#ifdef SUPPORT_FT78
REG_KERNEL(kDSP, kNumberTypeInt8, PrimitiveType_Concat, DSPKernelCreator<ConcatDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeFloat64, PrimitiveType_Concat, DSPKernelCreator<ConcatDSPKernel>)
REG_KERNEL(kDSP, kNumberTypeComplex128, PrimitiveType_Concat, DSPKernelCreator<ConcatDSPKernel>)
#endif
}  // namespace mindspore::kernel
