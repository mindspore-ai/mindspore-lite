/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "coder/opcoders/nnacl/fp32/batchnorm_fp32_coder.h"
#include <string>
#include <vector>
#include "nnacl_c/fp32/batchnorm_fp32.h"
#include "nnacl_c/op_base.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"

using mindspore::schema::PrimitiveType_BatchNorm;
using mindspore::schema::PrimitiveType_FusedBatchNorm;

namespace mindspore::lite::micro::nnacl {
int BatchnormFP32Coder::Init() {
  auto bn_parameter = reinterpret_cast<BatchNormParameter *>(OperatorCoder::parameter_);
  std::vector<int> input_shapes = input_tensor_->shape();
  if (input_shapes.empty()) {
    return RET_ERROR;
  }
  int n_dim = static_cast<int>(input_shapes.size());
  batchnorm_struct_.epsilon_ = bn_parameter->epsilon_;
  batchnorm_struct_.momentum_ = bn_parameter->momentum_;
  batchnorm_struct_.channel_ = input_shapes.at(n_dim - 1);
  batchnorm_struct_.unit_ = 1;
  for (int i = 0; i < n_dim - 1; i++) {
    batchnorm_struct_.unit_ *= input_shapes.at(i);
  }
  if (default_momentum_ < 0.0f) {
    default_momentum_ = bn_parameter->momentum_;
  }
  return RET_OK;
}

int BatchnormFP32Coder::DoCode(CoderContext *const context) {
  // attribute
  if (Init() != RET_OK) {
    MS_LOG(ERROR) << "BatchnormFP32Coder Init error";
    return RET_ERROR;
  }
  Collect(context,
          {
            "nnacl_c/fp32/batchnorm_fp32.h",
            "nnacl_c/kernel/batch_norm.h",
          },
          {
            "batchnorm_fp32.c",
            "tensor_c_utils.c",
          });
  NNaclFp32Serializer code;
  if (parameter_->type_ == PrimType_FusedBatchNorm) {
    MS_CHECK_TRUE(input_tensors_.size() == DIMENSION_5D, "inputs size is not equal to five");
    Tensor *scale_tensor = input_tensors_.at(SECOND_INPUT);
    Tensor *offset_tensor = input_tensors_.at(THIRD_INPUT);
    Tensor *mean_tensor = input_tensors_.at(FOURTH_INPUT);
    Tensor *var_tensor = input_tensors_.at(FIFTH_INPUT);
    MS_CHECK_PTR(scale_tensor);
    MS_CHECK_PTR(offset_tensor);
    MS_CHECK_PTR(mean_tensor);
    MS_CHECK_PTR(var_tensor);
    code.CodeStruct("bn_struct", batchnorm_struct_);
    code.CodeFunction("FusedBatchNormFp32", input_tensor_, scale_tensor, offset_tensor, mean_tensor, var_tensor,
                      "&bn_struct", kDefaultTaskId, kDefaultThreadNum, output_tensor_);
  } else {
    MS_CHECK_TRUE(input_tensors_.size() == DIMENSION_3D, "inputs size is not equal to three");
    Tensor *mean_tensor = input_tensors_.at(1);
    Tensor *var_tensor = input_tensors_.at(kInputSize1);
    MS_CHECK_PTR(mean_tensor);
    MS_CHECK_PTR(var_tensor);
    code.CodeStruct("bn_struct", batchnorm_struct_);
    code.CodeFunction("BatchNormFp32", input_tensor_, mean_tensor, var_tensor, "&bn_struct", kDefaultTaskId,
                      kDefaultThreadNum, output_tensor_);
  }
  MS_LOG(INFO) << "BatchnormFP32Code has been called";
  context->AppendCode(code.str());
  return lite::RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_FusedBatchNorm, CPUOpCoderCreator<BatchnormFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_BatchNorm, CPUOpCoderCreator<BatchnormFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
