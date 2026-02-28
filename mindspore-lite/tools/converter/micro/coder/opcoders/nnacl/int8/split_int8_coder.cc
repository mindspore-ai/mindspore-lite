/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/nnacl/int8/split_int8_coder.h"
#include "nnacl_c/int8/split_int8.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "src/litert/kernel/cpu/base/split_base.h"

using mindspore::schema::PrimitiveType_Split;

namespace mindspore::lite::micro::nnacl {
int SplitInt8Coder::Prepare(CoderContext *const context) {
  auto status = mindspore::kernel::SplitBaseCPUKernel::CheckAndInitSplitParam(
    *input_tensor_, reinterpret_cast<SplitParameter *>(parameter_));
  if (status != RET_OK) {
    MS_LOG(ERROR) << "CheckAndInitSplitParam failed";
    return status;
  }
  return RET_OK;
}

int SplitInt8Coder::DoCode(CoderContext *const context) {
  Collect(context, {"nnacl_c/int8/split_int8.h"}, {"split_int8.c"});
  auto param = reinterpret_cast<SplitParameter *>(parameter_);
  int num_unit = param->split_count_ * param->num_split_;

  NNaclInt8Serializer code;
  MS_CHECK_EQ(output_tensors_.size(), 0, RET_ERROR);
  MS_CHECK_LT(static_cast<size_t>(param->num_split_), output_tensors_.size(), RET_ERROR);
  code << "    void *output_ptrs[" << output_tensors_.size() << "] = {";
  for (int i = 0; i < param->num_split_; i++) {
    code << allocator_->GetRuntimeAddr(output_tensors_.at(i)) << ",";
  }
  code << "};\n";
  code << "    int input_dim[" << input_tensor_->shape().size() << "] = {";
  for (auto &dim : input_tensor_->shape()) {
    code << dim << ",";
  }
  code << "};\n";
  code << "    int split_sizes[" << param->num_split_ << "] = {";
  for (int i = 0; i < param->num_split_; i++) {
    code << param->split_sizes_[i] << ",";
  }
  code << "};\n";
  code.CodeStruct("split_param", *param);
  code.CodeFunction("Int8DoSplit", input_tensor_, "(void *)output_ptrs", "input_dim", "0", num_unit, "&split_param");
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Split, CPUOpCoderCreator<SplitInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
