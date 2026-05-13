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

#include "coder/opcoders/nnacl/fp32/space_to_depth_fp32_coder.h"
#include <vector>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/utils/common.h"

using mindspore::schema::PrimitiveType_SpaceToDepth;

namespace mindspore::lite::micro::nnacl {
int SpaceToDepthFp32Coder::Prepare(CoderContext *const context) {
  CHECK_NULL_RETURN(parameter_);
  auto *space_to_depth_param = reinterpret_cast<SpaceToDepthParameter *>(parameter_);
  args_.block_size_ = space_to_depth_param->block_size_;
  MS_CHECK_TRUE_MSG(input_tensor_ != nullptr, RET_PARAM_INVALID, "input tensor is nullptr.");
  MS_CHECK_TRUE_MSG(output_tensor_ != nullptr, RET_PARAM_INVALID, "output tensor is nullptr.");
  auto in_shape = input_tensor_->shape();
  auto out_shape = output_tensor_->shape();
  MS_CHECK_TRUE_MSG(in_shape.size() == DIMENSION_4D, RET_PARAM_INVALID, "input must be 4D");
  MS_CHECK_TRUE_MSG(out_shape.size() == DIMENSION_4D, RET_PARAM_INVALID, "output must be 4D");
  return RET_OK;
}

int SpaceToDepthFp32Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {"nnacl_c/base/space_to_depth_base.h", "nnacl_c/space_to_depth_parameter.h", "nnacl_c/fp32/pack_fp32.h"},
          {"space_to_depth_base.c", "pack_fp32.c"});

  NNaclFp32Serializer code;
  code.precision(kPrecision);
  code << "    SpaceToDepthParameter space_to_depth_param;\n";
  code << "    memset(&space_to_depth_param, 0, sizeof(SpaceToDepthParameter));\n";
  code << "    space_to_depth_param.op_parameter_.thread_num_ = 1;\n";
  code << "    space_to_depth_param.block_size_ = " << args_.block_size_ << ";\n";
  code << "    space_to_depth_param.date_type_len = " << sizeof(float) << ";\n";

  auto in_shape = input_tensor_->shape();
  auto out_shape = output_tensor_->shape();
  code.CodeArray("in_shape", in_shape.data(), DIMENSION_4D, true);
  code.CodeArray("out_shape", out_shape.data(), DIMENSION_4D, true);
  code.CodeFunction("SpaceToDepthForNHWC", input_tensor_, output_tensor_, "in_shape", "out_shape", in_shape.size(),
                    "&space_to_depth_param", 0);

  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_SpaceToDepth,
                   CPUOpCoderCreator<SpaceToDepthFp32Coder>)
}  // namespace mindspore::lite::micro::nnacl
