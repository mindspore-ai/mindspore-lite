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

#include "coder/opcoders/nnacl/int8/maximum_int8_coder.h"
#include "coder/log.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "nnacl_c/base/arithmetic_base.h"

using mindspore::schema::PrimitiveType_Maximum;
using mindspore::schema::PrimitiveType_Minimum;

namespace mindspore::lite::micro::nnacl {

int MaximumInt8Coder::Prepare(CoderContext *const context) {
  input0_ = input_tensors().at(0);
  input1_ = input_tensors().at(1);
  element_num_ = static_cast<int32_t>(output_tensor_->ElementsNum());
  arith_para_ = reinterpret_cast<ArithmeticParameter *>(parameter_);
  // Scalar broadcast must also go through TileDimensionsInt8 (with CalcMultiplesAndStrides),
  // otherwise ElementMaximumInt8 indexes input1[index] out of bounds on a 1-element scalar.
  arith_para_->broadcasting_ = (input0_->shape() != input1_->shape());
  if (arith_para_->broadcasting_) {
    CalcMultiplesAndStrides(arith_para_);
  }
  if (input0_->quant_params().size() > 0) {
    in0_scale_ = input0_->quant_params().front().scale;
    in0_zp_ = input0_->quant_params().front().zeroPoint;
  }
  if (input1_->quant_params().size() > 0) {
    in1_scale_ = input1_->quant_params().front().scale;
    in1_zp_ = input1_->quant_params().front().zeroPoint;
  }
  if (output_tensor_->quant_params().size() > 0) {
    out_scale_ = output_tensor_->quant_params().front().scale;
    out_zp_ = output_tensor_->quant_params().front().zeroPoint;
  }
  return RET_OK;
}

int MaximumInt8Coder::DoCode(CoderContext *const context) {
  Collect(context, {"nnacl_c/int8/arithmetic_int8.h"}, {"arithmetic_int8.c", "arithmetic_base.c"});

  NNaclInt8Serializer code;
  code << "  ArithmeticQuantArg quant_arg_0;\n";
  code << "  quant_arg_0.in0_args_.scale_ = " << in0_scale_ << ";\n";
  code << "  quant_arg_0.in0_args_.zp_ = " << in0_zp_ << ";\n";
  code << "  quant_arg_0.in1_args_.scale_ = " << in1_scale_ << ";\n";
  code << "  quant_arg_0.in1_args_.zp_ = " << in1_zp_ << ";\n";
  code << "  quant_arg_0.out_args_.scale_ = " << out_scale_ << ";\n";
  code << "  quant_arg_0.out_args_.zp_ = " << out_zp_ << ";\n";

  if (arith_para_->broadcasting_) {
    tile0_data_ = static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, output_tensor_->Size(), kWorkspace));
    MS_CHECK_PTR(tile0_data_);
    tile1_data_ = static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, output_tensor_->Size(), kWorkspace));
    MS_CHECK_PTR(tile1_data_);
    code.CodeStruct("arith_para", *arith_para_);
    code.CodeFunction("TileDimensionsInt8", input0_, input1_, tile0_data_, tile1_data_, "&arith_para");
    code.CodeFunction("ElementMaximumInt8", tile0_data_, tile1_data_, output_tensor_, element_num_, "&quant_arg_0");
  } else {
    code.CodeFunction("ElementMaximumInt8", input0_, input1_, output_tensor_, element_num_, "&quant_arg_0");
  }
  context->AppendCode(code.str());
  return RET_OK;
}

int MinimumInt8Coder::Prepare(CoderContext *const context) {
  input0_ = input_tensors().at(0);
  input1_ = input_tensors().at(1);
  element_num_ = static_cast<int32_t>(output_tensor_->ElementsNum());
  arith_para_ = reinterpret_cast<ArithmeticParameter *>(parameter_);
  // Scalar broadcast must also go through TileDimensionsInt8 (with CalcMultiplesAndStrides),
  // otherwise ElementMinimumInt8 indexes input1[index] out of bounds on a 1-element scalar.
  arith_para_->broadcasting_ = (input0_->shape() != input1_->shape());
  if (arith_para_->broadcasting_) {
    CalcMultiplesAndStrides(arith_para_);
  }
  if (input0_->quant_params().size() > 0) {
    in0_scale_ = input0_->quant_params().front().scale;
    in0_zp_ = input0_->quant_params().front().zeroPoint;
  }
  if (input1_->quant_params().size() > 0) {
    in1_scale_ = input1_->quant_params().front().scale;
    in1_zp_ = input1_->quant_params().front().zeroPoint;
  }
  if (output_tensor_->quant_params().size() > 0) {
    out_scale_ = output_tensor_->quant_params().front().scale;
    out_zp_ = output_tensor_->quant_params().front().zeroPoint;
  }
  return RET_OK;
}

int MinimumInt8Coder::DoCode(CoderContext *const context) {
  Collect(context, {"nnacl_c/int8/arithmetic_int8.h"}, {"arithmetic_int8.c", "arithmetic_base.c"});

  NNaclInt8Serializer code;
  code << "  ArithmeticQuantArg quant_arg_0;\n";
  code << "  quant_arg_0.in0_args_.scale_ = " << in0_scale_ << ";\n";
  code << "  quant_arg_0.in0_args_.zp_ = " << in0_zp_ << ";\n";
  code << "  quant_arg_0.in1_args_.scale_ = " << in1_scale_ << ";\n";
  code << "  quant_arg_0.in1_args_.zp_ = " << in1_zp_ << ";\n";
  code << "  quant_arg_0.out_args_.scale_ = " << out_scale_ << ";\n";
  code << "  quant_arg_0.out_args_.zp_ = " << out_zp_ << ";\n";

  if (arith_para_->broadcasting_) {
    tile0_data_ = static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, output_tensor_->Size(), kWorkspace));
    MS_CHECK_PTR(tile0_data_);
    tile1_data_ = static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, output_tensor_->Size(), kWorkspace));
    MS_CHECK_PTR(tile1_data_);
    code.CodeStruct("arith_para", *arith_para_);
    code.CodeFunction("TileDimensionsInt8", input0_, input1_, tile0_data_, tile1_data_, "&arith_para");
    code.CodeFunction("ElementMinimumInt8", tile0_data_, tile1_data_, output_tensor_, element_num_, "&quant_arg_0");
  } else {
    code.CodeFunction("ElementMinimumInt8", input0_, input1_, output_tensor_, element_num_, "&quant_arg_0");
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Maximum, CPUOpCoderCreator<MaximumInt8Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Minimum, CPUOpCoderCreator<MinimumInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
