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

#include "coder/opcoders/nnacl/int8/mul_int8_coder.h"
#include <string>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "nnacl_c/int8/quantize.h"
#include "coder/opcoders/file_collector.h"
#include "coder/utils/common.h"
#include "nnacl_c/mul_parameter.h"
#include "nnacl_c/arithmetic_parameter.h"

using mindspore::schema::PrimitiveType_MulFusion;

namespace mindspore::lite::micro::nnacl {
int MulInt8Coder::Prepare(CoderContext *const context) {
  MS_CHECK_TRUE(OperatorCoder::input_tensors().size() == Num2, "MulInt8 only support 2 inputs");
  input1_ = OperatorCoder::input_tensors().at(0);
  input2_ = OperatorCoder::input_tensors().at(1);

  MS_CHECK_PTR(input1_);
  MS_CHECK_PTR(input2_);
  MS_CHECK_PTR(output_tensor_);
  // check if need to broadcast
  arith_para_ = reinterpret_cast<ArithmeticParameter *>(parameter_);
  MS_CHECK_PTR(arith_para_);

  // Set broadcasting flag based on shape comparison
  auto input1_shape = input1_->shape();
  auto input2_shape = input2_->shape();
  arith_para_->broadcasting_ = (input1_shape != input2_shape);

  need_broadcast_ = arith_para_->broadcasting_;
  if (!need_broadcast_) {
    block_size_ = input1_->ElementsNum();
  }
  MS_CHECK_TRUE(!input1_->quant_params().empty(), "input1_ quant_params is empty");
  MS_CHECK_TRUE(!input2_->quant_params().empty(), "input2_ quant_params is empty");
  MS_CHECK_TRUE(!output_tensor_->quant_params().empty(), "output quant_params is empty");

  input_1_offset_ = -input1_->quant_params().at(0).zeroPoint;
  input_2_offset_ = -input2_->quant_params().at(0).zeroPoint;
  out_offset_ = output_tensor_->quant_params().at(0).zeroPoint;
  const double input1_scale = input1_->quant_params().at(0).scale;
  const double input2_scale = input2_->quant_params().at(0).scale;
  const double output_scale = output_tensor_->quant_params().at(0).scale;
  const double real_multiplier = input1_scale * input2_scale / output_scale;

  QuantizeMultiplierSmallerThanOne(real_multiplier, &out_mult_, &right_shift);

  CalculateActivationRangeQuantized(false, false, out_offset_, output_scale, &out_activation_min_,
                                    &out_activation_max_);

  return RET_OK;
}

int MulInt8Coder::DoCode(CoderContext *const context) {
  NNaclInt8Serializer code;
  code.precision(kPrecision);
  auto element_num = output_tensor_->ElementsNum();
  std::vector<LiteQuantParam> input1_quant_args = input1_->quant_params();
  std::vector<LiteQuantParam> input2_quant_args = input2_->quant_params();
  std::vector<LiteQuantParam> out_quant_args = output_tensor_->quant_params();
  Collect(context,
          {
            "nnacl_c/int8/arithmetic_int8.h",
            "nnacl_c/int8/fixed_point.h",
            "nnacl_c/int8/mul_int8.h",
          },
          {
            "arithmetic_base.c",
            "arithmetic_int8.c",
            "fixed_point.c",
            "mul_int8.c",
          });
  ::MulQuantArg quant_arg;
  quant_arg.in0_quant_args_ = {static_cast<float>(input1_quant_args.at(0).scale), -input1_quant_args.at(0).zeroPoint};
  quant_arg.in1_quant_args_ = {static_cast<float>(input2_quant_args.at(0).scale), -input2_quant_args.at(0).zeroPoint};
  quant_arg.out_quant_arg_ = {static_cast<float>(out_quant_args.at(0).scale), out_quant_args.at(0).zeroPoint};
  quant_arg.output_multiplier_ = out_mult_;
  quant_arg.output_activation_max_ = out_activation_max_;
  quant_arg.output_activation_min_ = out_activation_min_;
  quant_arg.shift_left_ = right_shift < 0 ? -right_shift : 0;
  quant_arg.shift_right_ = right_shift > 0 ? right_shift : 0;

  code.CodeStruct("quant", quant_arg);
  if (need_broadcast_) {
    ArithmeticParameter tile_para;
    auto out_shape = output_tensor_->shape();
    tile_para.ndim_ = out_shape.size();
    auto in_shape0 = input1_->shape();
    MS_CHECK_TRUE_MSG(out_shape.size() >= in_shape0.size(), RET_ERROR,
                      "Mul first-input shape size is larger than out.");
    for (size_t i = 0; i < out_shape.size() - in_shape0.size(); ++i) {
      tile_para.in_shape0_[i] = 1;
    }
    for (size_t i = out_shape.size() - in_shape0.size(); i < out_shape.size(); ++i) {
      tile_para.in_shape0_[i] = input1_->DimensionSize(i - (out_shape.size() - in_shape0.size()));
    }
    auto in_shape1 = input2_->shape();
    MS_CHECK_TRUE_MSG(out_shape.size() >= in_shape1.size(), RET_ERROR,
                      "Mul second-input shape size is larger than out.");
    for (size_t i = 0; i < out_shape.size() - in_shape1.size(); ++i) {
      tile_para.in_shape1_[i] = 1;
    }
    for (size_t i = out_shape.size() - in_shape1.size(); i < out_shape.size(); ++i) {
      tile_para.in_shape1_[i] = input2_->DimensionSize(i - (out_shape.size() - in_shape1.size()));
    }
    for (size_t i = 0; i < out_shape.size(); ++i) {
      tile_para.out_shape_[i] = output_tensor_->DimensionSize(i);
    }
    tile0_data_ = static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, output_tensor_->Size(), kWorkspace));
    MS_CHECK_PTR(tile0_data_);
    tile1_data_ = static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, output_tensor_->Size(), kWorkspace));
    MS_CHECK_PTR(tile1_data_);

    code.CodeStruct("tile_para", tile_para);

    code.CodeFunction("TileDimensionsInt8", input1_, input2_, tile0_data_, tile1_data_, "&tile_para");
    code.CodeFunction("Mul", tile0_data_, tile1_data_, output_tensor_, element_num, "&quant");
  } else {
    code.CodeFunction("Mul", input1_, input2_, output_tensor_, block_size_, "&quant");
  }

  MS_LOG(INFO) << "MulInt8Coder has been called";
  context->AppendCode(code.str());
  return RET_OK;
}
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_MulFusion, CPUOpCoderCreator<MulInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
