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
#include "src/litert/kernel/cpu/int8/space_to_depth_int8.h"
#include <cfloat>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "nnacl_c/base/space_to_depth_base.h"
#include "nnacl_c/int8/space_to_depth_int8.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_SpaceToDepth;

namespace mindspore::kernel {
SpaceToDepthInt8CPUKernel::~SpaceToDepthInt8CPUKernel() {
  if (in_quant_arg_ != nullptr) {
    free(in_quant_arg_);
    in_quant_arg_ = nullptr;
  }
  if (out_quant_arg_ != nullptr) {
    free(out_quant_arg_);
    out_quant_arg_ = nullptr;
  }
}

int SpaceToDepthInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[kInputIndex]);
  CHECK_NULL_RETURN(out_tensors_[kOutputIndex]);
  if (in_tensors_[kInputIndex]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[kOutputIndex]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[kInputIndex]->data_type()
                  << ", output data_type is " << out_tensors_[kOutputIndex]->data_type();
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(op_parameter_);
  auto *space_to_depth_param = reinterpret_cast<SpaceToDepthParameter *>(op_parameter_);
  MS_CHECK_TRUE_MSG(space_to_depth_param->block_size_ > 0, RET_PARAM_INVALID, "Input block_size should > 0.");
  args_.block_size_ = space_to_depth_param->block_size_;
  args_.date_type_len = sizeof(int8_t);
  args_.op_parameter_.thread_num_ = op_parameter_->thread_num_;

  in_quant_arg_ = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
  MS_CHECK_TRUE_MSG(in_quant_arg_ != nullptr, RET_ERROR, "Malloc QuantArg for SpaceToDepth int8 op failed.");
  auto in_quant_args = in_tensors_.at(kInputIndex)->quant_params();
  CHECK_LESS_RETURN(in_quant_args.size(), 1);
  in_quant_arg_->scale_ = in_quant_args.front().scale;
  in_quant_arg_->zp_ = in_quant_args.front().zeroPoint;
  out_quant_arg_ = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
  MS_CHECK_TRUE_MSG(out_quant_arg_ != nullptr, RET_ERROR, "Malloc QuantArg for SpaceToDepth int8 op failed.");
  auto out_quant_args = out_tensors_.at(kOutputIndex)->quant_params();
  CHECK_LESS_RETURN(out_quant_args.size(), 1);
  out_quant_arg_->scale_ = out_quant_args.front().scale;
  out_quant_arg_->zp_ = out_quant_args.front().zeroPoint;

  same_quant_ =
    std::abs(in_quant_arg_->scale_ - out_quant_arg_->scale_) < FLT_EPSILON && in_quant_arg_->zp_ == out_quant_arg_->zp_;

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SpaceToDepthInt8CPUKernel::ReSize() {
  if (in_tensors_[kInputIndex]->format() != mindspore::NHWC) {
    MS_LOG(ERROR) << "space_to_depth only support NHWC now!";
    return RET_FORMAT_ERR;
  }
  return RET_OK;
}

int SpaceToDepthInt8CPUKernel::SpaceToDepth(int task_id) {
  auto input = in_tensors_[kInputIndex];
  auto output = out_tensors_[kOutputIndex];
  const int8_t *input_data = reinterpret_cast<const int8_t *>(input->data());
  CHECK_NULL_RETURN(input_data);
  int8_t *output_data = reinterpret_cast<int8_t *>(output->data());
  CHECK_NULL_RETURN(output_data);
  auto in_shape = input->shape();
  auto out_shape = output->shape();

  if (same_quant_) {
    auto ret =
      SpaceToDepthForNHWC(input_data, output_data, in_shape.data(), out_shape.data(), in_shape.size(), &args_, task_id);
    if (ret != NNACL_OK) {
      MS_LOG(ERROR) << "SpaceToDepthForNHWC failed.";
      return RET_ERROR;
    }
  } else {
    auto ret = SpaceToDepthForNHWCInt8(input_data, output_data, in_shape.data(), out_shape.data(), in_shape.size(),
                                       &args_, in_quant_arg_, out_quant_arg_, task_id);
    if (ret != NNACL_OK) {
      MS_LOG(ERROR) << "SpaceToDepthForNHWCInt8 failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int SpaceToDepthInt8Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = static_cast<SpaceToDepthInt8CPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  auto ret = kernel->SpaceToDepth(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SpaceToDepthInt8Run failed, ret: " << ret;
  }
  return ret;
}

int SpaceToDepthInt8CPUKernel::Run() {
  auto ret = ParallelLaunch(ms_context_, SpaceToDepthInt8Run, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParallelLaunch failed, ret: " << ret;
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_SpaceToDepth, LiteKernelCreator<SpaceToDepthInt8CPUKernel>)
}  // namespace mindspore::kernel
