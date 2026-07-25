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

#include "src/litert/kernel/cpu/int8/gelu_int8.h"
#include "nnacl_c/op_base.h"
#include "nnacl_c/errorcode.h"
#include "nnacl_c/activation_parameter.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int GeluInt8CPUKernel::Prepare() {
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
  const auto &in_quant_args = in_tensors_[0]->quant_params();
  const auto &out_quant_args = out_tensors_[0]->quant_params();
  MS_CHECK_TRUE_MSG(!in_quant_args.empty(), RET_ERROR, "Input quant param cannot be empty.");
  MS_CHECK_TRUE_MSG(!out_quant_args.empty(), RET_ERROR, "Output quant param cannot be empty.");

  quant_arg_.in_args_.scale_ = in_quant_args.front().scale;
  quant_arg_.in_args_.zp_ = in_quant_args.front().zeroPoint;
  quant_arg_.out_args_.scale_ = out_quant_args.front().scale;
  quant_arg_.out_args_.zp_ = out_quant_args.front().zeroPoint;

  // Honour the approximate flag so the LUT is built with the erf (approximate=false)
  // or tanh (approximate=true) curve declared by the source model.
  bool approximate = false;
  if (op_parameter_ != nullptr) {
    approximate = reinterpret_cast<ActivationParameter *>(op_parameter_)->approximate_;
  }
  if (GeluInt8InitLUT(&quant_arg_, table_, approximate) != NNACL_OK) {
    MS_LOG(ERROR) << "GeluInt8InitLUT failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GeluInt8CPUKernel::ReSize() { return RET_OK; }

int GeluInt8CPUKernel::DoActivation(int task_id) {
  auto input_addr = reinterpret_cast<int8_t *>(in_tensors_.at(0)->MutableData());
  auto output_addr = reinterpret_cast<int8_t *>(out_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(input_addr);
  CHECK_NULL_RETURN(output_addr);
  auto length = in_tensors_.at(0)->ElementsNum();
  MS_CHECK_GT(length, 0, RET_ERROR);
  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);
  if (count <= 0) {
    return RET_OK;
  }
  auto ret = GeluInt8(input_addr + stride * task_id, count, output_addr + stride * task_id, table_);
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "DoActivation gelu int8 task id " << task_id << " failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GeluInt8Run(void *cdata, int task_id, float, float) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "input cdata is nullptr!";
    return RET_ERROR;
  }
  auto activation_kernel = reinterpret_cast<GeluInt8CPUKernel *>(cdata);
  auto error_code = activation_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "GeluInt8Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int GeluInt8CPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, GeluInt8Run, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "GeluInt8Run function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
