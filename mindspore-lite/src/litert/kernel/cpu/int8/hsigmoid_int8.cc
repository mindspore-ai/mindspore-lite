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

#include "src/litert/kernel/cpu/int8/hsigmoid_int8.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;

namespace mindspore::kernel {
int HardSigmoidInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C1NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), C1NUM);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "HardSigmoid int8 datatype error, input0 data_type is " << in_tensors_[0]->data_type()
                  << ", output data_type is " << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto in_quant_args = input_tensor->quant_params();
  MS_CHECK_TRUE_MSG(!in_quant_args.empty(), RET_PARAM_INVALID, "HardSigmoid int8 input quant param cannot be empty.");
  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto out_quant_args = out_tensor->quant_params();
  MS_CHECK_TRUE_MSG(!out_quant_args.empty(), RET_PARAM_INVALID, "HardSigmoid int8 output quant param cannot be empty.");
  CHECK_NULL_RETURN(op_parameter_);
  auto *activation_param = reinterpret_cast<ActivationParameter *>(op_parameter_);
  auto ret = HardSigmoidInt8InitLUT(static_cast<float>(in_quant_args.front().scale), in_quant_args.front().zeroPoint,
                                    static_cast<float>(out_quant_args.front().scale), out_quant_args.front().zeroPoint,
                                    activation_param->alpha_, activation_param->beta_, table_list_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "HardSigmoidInt8InitLUT initialized failed.";
    return ret;
  }
  return ReSize();
}

int HardSigmoidInt8CPUKernel::ReSize() { return RET_OK; }

int HardSigmoidInt8CPUKernel::DoActivation(int task_id) {
  auto input_data = reinterpret_cast<int8_t *>(in_tensors_.at(kInputIndex)->MutableData());
  CHECK_NULL_RETURN(input_data);
  auto output_data = reinterpret_cast<int8_t *>(out_tensors_.at(kOutputIndex)->MutableData());
  CHECK_NULL_RETURN(output_data);
  auto length = in_tensors_.at(kInputIndex)->ElementsNum();
  MS_CHECK_GT(length, 0, RET_ERROR);
  int stride = UP_DIV(length, op_parameter_->thread_num_);
  int count = MSMIN(stride, length - stride * task_id);
  int offset = stride * task_id;
  auto ret = HardSigmoidInt8(input_data + offset, count, output_data + offset, table_list_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "HardSigmoid int8 task id " << task_id << " failed.";
    return ret;
  }
  return RET_OK;
}

int HardSigmoidInt8Run(void *cdata, int task_id, float, float) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "input cdata is nullptr!";
    return RET_ERROR;
  }
  auto activation_kernel = reinterpret_cast<HardSigmoidInt8CPUKernel *>(cdata);
  auto error_code = activation_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "HardSigmoidInt8Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int HardSigmoidInt8CPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, HardSigmoidInt8Run, this, op_parameter_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "HardSigmoidInt8Run function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

}  // namespace mindspore::kernel
