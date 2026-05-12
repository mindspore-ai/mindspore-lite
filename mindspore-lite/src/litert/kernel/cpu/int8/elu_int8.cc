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

#include "src/litert/kernel/cpu/int8/elu_int8.h"
#include <limits>
#include <algorithm>
#include <cmath>
#include "nnacl_c/int8/elu_int8.h"
#include "nnacl_c/int8/quantize.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::ActivationType_ELU;

namespace mindspore::kernel {
int EluInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C1NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), C1NUM);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", output data_type is "
                  << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto in_quant_args = input_tensor->quant_params();
  MS_CHECK_TRUE_MSG(!in_quant_args.empty(), RET_PARAM_INVALID, "Input quant param cannot be empty.");
  quant_elu_parm_.in_args_.scale_ = static_cast<float>(in_quant_args.front().scale);
  quant_elu_parm_.in_args_.zp_ = in_quant_args.front().zeroPoint;
  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto out_quant_args = out_tensor->quant_params();
  MS_CHECK_TRUE_MSG(!out_quant_args.empty(), RET_PARAM_INVALID, "Output quant param cannot be empty.");
  quant_elu_parm_.out_args_.scale_ = static_cast<float>(out_quant_args.front().scale);
  quant_elu_parm_.out_args_.zp_ = out_quant_args.front().zeroPoint;
  CHECK_NULL_RETURN(op_parameter_);
  auto *activation_param = reinterpret_cast<ActivationParameter *>(op_parameter_);
  quant_elu_parm_.alpha_ = activation_param->alpha_;

  // Initialize the lookup table
  auto ret = EluInt8InitLUT(&quant_elu_parm_, table_list_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "EluInt8InitLUT Initialized failed.";
    return ret;
  }
  return ReSize();
}

int EluInt8CPUKernel::ReSize() {
  auto *input_tensor = in_tensors_.at(kInputIndex);
  MS_CHECK_GT(input_tensor->ElementsNum(), 0, RET_ERROR);
  quant_elu_parm_.element_num = input_tensor->ElementsNum();
  quant_elu_parm_.thread_num_ = op_parameter_->thread_num_;
  return RET_OK;
}

int EluInt8CPUKernel::DoActivation(int task_id) {
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto out_tensor = out_tensors_.at(kOutputIndex);

  int8_t *input_data = reinterpret_cast<int8_t *>(input_tensor->data());
  CHECK_NULL_RETURN(input_data);
  int8_t *output_data = reinterpret_cast<int8_t *>(out_tensor->data());
  CHECK_NULL_RETURN(output_data);
  auto length = input_tensor->ElementsNum();
  int stride = UP_DIV(length, op_parameter_->thread_num_);
  int count = MSMIN(stride, length - stride * task_id);
  int offset = stride * task_id;

  auto ret = EluInt8(input_data + offset, count, output_data + offset, table_list_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoActivation task_id " << task_id << " failed.";
    return ret;
  }
  return RET_OK;
}

int EluInt8Run(void *cdata, int task_id, float, float) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "input cdata is nullptr!";
    return RET_ERROR;
  }
  auto elu_kernel = reinterpret_cast<EluInt8CPUKernel *>(cdata);
  auto error_code = elu_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "EluInt8Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int EluInt8CPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, EluInt8Run, this, op_parameter_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "EluInt8Run function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
