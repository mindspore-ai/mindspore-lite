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

#include "coder/opcoders/nnacl/fp32/depth_to_space_fp32_coder.h"
#include <vector>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/utils/common.h"
#include "nnacl_c/nnacl_common.h"

using mindspore::schema::PrimitiveType_DepthToSpace;

namespace mindspore::lite::micro::nnacl {
int DepthToSpaceFp32Coder::Prepare(CoderContext *const context) {
  CHECK_NULL_RETURN(parameter_);
  auto *depth_to_space_param = reinterpret_cast<DepthToSpaceParameter *>(parameter_);
  MS_CHECK_TRUE_MSG(input_tensor_ != nullptr, RET_INPUT_PARAM_INVALID, "input tensor is nullptr.");
  MS_CHECK_TRUE_MSG(output_tensor_ != nullptr, RET_INPUT_PARAM_INVALID, "output_tensor_ tensor is nullptr.");
  auto in_shape = input_tensor_->shape();
  auto out_shape = output_tensor_->shape();
  MS_CHECK_TRUE_MSG(in_shape.size() == DIMENSION_4D, RET_INPUT_PARAM_INVALID, "input tensor must be 4D.");
  MS_CHECK_TRUE_MSG(out_shape.size() == DIMENSION_4D, RET_INPUT_PARAM_INVALID, "output tensor must be 4D.");

  int32_t in_strides[DIMENSION_4D] = {0};
  ComputeStrides(in_shape.data(), in_strides, DIMENSION_4D);
  args_.in_stride_dim0_ = in_strides[Index0];
  args_.in_stride_dim1_ = in_strides[Index1];
  args_.in_stride_dim2_ = in_strides[Index2];

  int32_t out_strides[DIMENSION_4D] = {0};
  ComputeStrides(out_shape.data(), out_strides, DIMENSION_4D);
  args_.out_stride_dim0_ = out_strides[Index0];
  args_.out_stride_dim1_ = out_strides[Index1];
  args_.out_stride_dim2_ = out_strides[Index2];

  args_.data_type_size_ = sizeof(float);
  args_.block_size_ = depth_to_space_param->block_size_;
  return RET_OK;
}

int DepthToSpaceFp32Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {"nnacl_c/base/depth_to_space_base.h", "nnacl_c/kernel/depth_to_space.h",
           "nnacl_c/depth_to_space_parameter.h", "nnacl_c/fp32/pack_fp32.h"},
          {"depth_to_space_base.c", "pack_fp32.c"});

  NNaclFp32Serializer code;

  code.CodeStruct("depth_to_space_args", args_);

  auto in_shape = input_tensor_->shape();
  code.CodeArray("in_shape", in_shape.data(), DIMENSION_4D, true);

  auto *depth_to_space_param = reinterpret_cast<DepthToSpaceParameter *>(parameter_);
  if (depth_to_space_param->mode_ == 1) {
    code.CodeFunction("DepthToSpaceCRDForNHWC", input_tensor_, output_tensor_, "in_shape", "&depth_to_space_args");
  } else {
    code.CodeFunction("DepthToSpaceForNHWC", input_tensor_, output_tensor_, "in_shape", "&depth_to_space_args");
  }

  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_DepthToSpace,
                   CPUOpCoderCreator<DepthToSpaceFp32Coder>)
}  // namespace mindspore::lite::micro::nnacl
