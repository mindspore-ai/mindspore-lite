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

#include "src/litert/kernel/cpu/int8/celu_int8.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;

namespace mindspore::kernel {
int CeluInt8CPUKernel::Prepare() {
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
  quant_celu_parm_.in_args_.scale_ = static_cast<float>(in_quant_args.front().scale);
  quant_celu_parm_.in_args_.zp_ = in_quant_args.front().zeroPoint;
  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto out_quant_args = out_tensor->quant_params();
  MS_CHECK_TRUE_MSG(!out_quant_args.empty(), RET_PARAM_INVALID, "Output quant param cannot be empty.");
  quant_celu_parm_.out_args_.scale_ = static_cast<float>(out_quant_args.front().scale);
  quant_celu_parm_.out_args_.zp_ = out_quant_args.front().zeroPoint;
  CHECK_NULL_RETURN(op_parameter_);
  auto *activation_param = reinterpret_cast<ActivationParameter *>(op_parameter_);
  quant_celu_parm_.alpha_ = activation_param->alpha_;

  auto ret = CeluInt8InitLUT(&quant_celu_parm_, table_list_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CeluInt8InitLUT initialized failed.";
    return ret;
  }
  return ReSize();
}

int CeluInt8CPUKernel::ReSize() {
  auto *input_tensor = in_tensors_.at(kInputIndex);
  MS_CHECK_GT(input_tensor->ElementsNum(), 0, RET_ERROR);
  quant_celu_parm_.element_num = input_tensor->ElementsNum();
  quant_celu_parm_.thread_num_ = op_parameter_->thread_num_;
  return RET_OK;
}

int CeluInt8CPUKernel::DoActivation(int task_id) {
  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto *out_tensor = out_tensors_.at(kOutputIndex);
  int8_t *input_data = reinterpret_cast<int8_t *>(input_tensor->data());
  CHECK_NULL_RETURN(input_data);
  int8_t *output_data = reinterpret_cast<int8_t *>(out_tensor->data());
  CHECK_NULL_RETURN(output_data);
  const int length = input_tensor->ElementsNum();
  const int stride = UP_DIV(length, op_parameter_->thread_num_);
  const int count = MSMIN(stride, length - stride * task_id);
  const int offset = stride * task_id;
  return CeluInt8(input_data + offset, count, output_data + offset, table_list_);
}

int CeluInt8Run(void *cdata, int task_id, float, float) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "input cdata is nullptr!";
    return RET_ERROR;
  }
  auto *celu_kernel = reinterpret_cast<CeluInt8CPUKernel *>(cdata);
  auto error_code = celu_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "CeluInt8Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int CeluInt8CPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, CeluInt8Run, this, op_parameter_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "CeluInt8Run function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
