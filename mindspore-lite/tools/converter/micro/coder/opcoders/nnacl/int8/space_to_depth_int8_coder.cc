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

#include "coder/opcoders/nnacl/int8/space_to_depth_int8_coder.h"
#include <cfloat>
#include <vector>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/utils/common.h"

using mindspore::schema::PrimitiveType_SpaceToDepth;

namespace mindspore::lite::micro::nnacl {
int SpaceToDepthInt8Coder::Prepare(CoderContext *const context) {
  CHECK_NULL_RETURN(parameter_);
  auto *space_to_depth_param = reinterpret_cast<SpaceToDepthParameter *>(parameter_);
  args_.block_size_ = space_to_depth_param->block_size_;
  MS_CHECK_TRUE_MSG(input_tensor_ != nullptr, RET_PARAM_INVALID, "input tensor is nullptr.");
  MS_CHECK_TRUE_MSG(output_tensor_ != nullptr, RET_PARAM_INVALID, "output tensor is nullptr.");
  auto in_shape = input_tensor_->shape();
  auto out_shape = output_tensor_->shape();
  MS_CHECK_TRUE_MSG(in_shape.size() == DIMENSION_4D, RET_PARAM_INVALID, "input must be 4D.");
  MS_CHECK_TRUE_MSG(out_shape.size() == DIMENSION_4D, RET_PARAM_INVALID, "output must be 4D");
  auto in_quant_params = input_tensor_->quant_params();
  auto out_quant_params = output_tensor_->quant_params();
  MS_CHECK_TRUE_MSG(!in_quant_params.empty(), RET_ERROR, "in_quant_params is empty.");
  MS_CHECK_TRUE_MSG(!out_quant_params.empty(), RET_ERROR, "out_quant_params is empty.");
  in_quant_arg_.scale_ = static_cast<float>(in_quant_params.front().scale);
  in_quant_arg_.zp_ = in_quant_params.front().zeroPoint;
  out_quant_arg_.scale_ = static_cast<float>(out_quant_params.front().scale);
  out_quant_arg_.zp_ = out_quant_params.front().zeroPoint;
  same_quant_ =
    (std::abs(in_quant_arg_.scale_ - out_quant_arg_.scale_) < FLT_EPSILON && in_quant_arg_.zp_ == out_quant_arg_.zp_);
  return RET_OK;
}

int SpaceToDepthInt8Coder::DoCode(CoderContext *const context) {
  if (same_quant_) {
    Collect(context,
            {"nnacl_c/base/space_to_depth_base.h", "nnacl_c/space_to_depth_parameter.h", "nnacl_c/fp32/pack_fp32.h"},
            {"space_to_depth_base.c", "pack_fp32.c"});
  } else {
    Collect(context,
            {"nnacl_c/int8/space_to_depth_int8.h", "nnacl_c/space_to_depth_parameter.h", "nnacl_c/int8/quantize.h",
             "nnacl_c/common_func.h", "nnacl_c/fp32/pack_fp32.h"},
            {"space_to_depth_int8.c", "pack_fp32.c"});
  }

  NNaclInt8Serializer code;
  code.precision(kPrecision);

  code << "    SpaceToDepthParameter space_to_depth_param;\n";
  code << "    memset(&space_to_depth_param, 0, sizeof(SpaceToDepthParameter));\n";
  code << "    space_to_depth_param.op_parameter_.thread_num_ = 1;\n";
  code << "    space_to_depth_param.block_size_ = " << args_.block_size_ << ";\n";
  code << "    space_to_depth_param.date_type_len = " << sizeof(int8_t) << ";\n";

  auto in_shape = input_tensor_->shape();
  auto out_shape = output_tensor_->shape();
  code.CodeArray("in_shape", in_shape.data(), DIMENSION_4D, true);
  code.CodeArray("out_shape", out_shape.data(), DIMENSION_4D, true);
  if (same_quant_) {
    code.CodeFunction("SpaceToDepthForNHWC", input_tensor_, output_tensor_, "in_shape", "out_shape", in_shape.size(),
                      "&space_to_depth_param", 0);
  } else {
    code.CodeBaseStruct("QuantArg", "in_quant_arg", in_quant_arg_.scale_, in_quant_arg_.zp_);
    code.CodeBaseStruct("QuantArg", "out_quant_arg", out_quant_arg_.scale_, out_quant_arg_.zp_);
    code.CodeFunction("SpaceToDepthForNHWCInt8", input_tensor_, output_tensor_, "in_shape", "out_shape",
                      in_shape.size(), "&space_to_depth_param", "&in_quant_arg", "&out_quant_arg", 0);
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_SpaceToDepth, CPUOpCoderCreator<SpaceToDepthInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
