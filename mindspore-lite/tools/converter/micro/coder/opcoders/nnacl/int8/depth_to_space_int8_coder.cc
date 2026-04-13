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

#include "coder/opcoders/nnacl/int8/depth_to_space_int8_coder.h"
#include <cfloat>
#include <vector>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/utils/common.h"
#include "nnacl_c/nnacl_common.h"

using mindspore::schema::PrimitiveType_DepthToSpace;

namespace mindspore::lite::micro::nnacl {
int DepthToSpaceInt8Coder::Prepare(CoderContext *const context) {
  CHECK_NULL_RETURN(parameter_);
  auto *depth_to_space_param = reinterpret_cast<DepthToSpaceParameter *>(parameter_);
  MS_CHECK_TRUE_MSG(input_tensor_ != nullptr, RET_PARAM_INVALID, "input tensor is nullptr.");
  MS_CHECK_TRUE_MSG(output_tensor_ != nullptr, RET_PARAM_INVALID, "output tensor is nullptr.");
  auto in_shape = input_tensor_->shape();
  auto out_shape = output_tensor_->shape();
  MS_CHECK_TRUE_MSG(in_shape.size() == DIMENSION_4D, RET_PARAM_INVALID, "input must be 4D.");
  MS_CHECK_TRUE_MSG(out_shape.size() == DIMENSION_4D, RET_PARAM_INVALID, "output must be 4D.");

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

  args_.data_type_size_ = sizeof(int8_t);
  args_.block_size_ = depth_to_space_param->block_size_;

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

int DepthToSpaceInt8Coder::DoCode(CoderContext *const context) {
  if (same_quant_) {
    Collect(context,
            {"nnacl_c/base/depth_to_space_base.h", "nnacl_c/kernel/depth_to_space.h",
             "nnacl_c/depth_to_space_parameter.h", "nnacl_c/fp32/pack_fp32.h"},
            {"depth_to_space_base.c", "pack_fp32.c"});
  } else {
    Collect(context,
            {"nnacl_c/int8/depth_to_space_int8.h", "nnacl_c/kernel/depth_to_space.h",
             "nnacl_c/depth_to_space_parameter.h", "nnacl_c/int8/quantize.h", "nnacl_c/fp32/pack_fp32.h"},
            {"depth_to_space_int8.c", "pack_fp32.c"});
  }

  NNaclInt8Serializer code;
  code.precision(kPrecision);

  code.CodeStruct("depth_to_space_args", args_);

  auto in_shape = input_tensor_->shape();
  code.CodeArray("in_shape", in_shape.data(), DIMENSION_4D, true);
  auto *depth_to_space_param = reinterpret_cast<DepthToSpaceParameter *>(parameter_);
  if (same_quant_) {
    if (depth_to_space_param->mode_ == 1) {
      code.CodeFunction("DepthToSpaceCRDForNHWC", input_tensor_, output_tensor_, "in_shape", "&depth_to_space_args");
    } else {
      code.CodeFunction("DepthToSpaceForNHWC", input_tensor_, output_tensor_, "in_shape", "&depth_to_space_args");
    }
  } else {
    code.CodeBaseStruct<false>("QuantArg", "in_quant_arg", in_quant_arg_.scale_, in_quant_arg_.zp_);
    code.CodeBaseStruct<false>("QuantArg", "out_quant_arg", out_quant_arg_.scale_, out_quant_arg_.zp_);
    if (depth_to_space_param->mode_ == 1) {
      code.CodeFunction("DepthToSpaceCRDForNHWCInt8", input_tensor_, output_tensor_, "in_shape", "&depth_to_space_args",
                        "&in_quant_arg", "&out_quant_arg");
    } else {
      code.CodeFunction("DepthToSpaceForNHWCInt8", input_tensor_, output_tensor_, "in_shape", "&depth_to_space_args",
                        "&in_quant_arg", "&out_quant_arg");
    }
  }

  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_DepthToSpace, CPUOpCoderCreator<DepthToSpaceInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
