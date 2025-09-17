/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl_c/infer/lstm_grad_weight_infer.h"
#include "nnacl_c/infer/infer_register.h"
#include "nnacl_c/infer/common_infer.h"
#include "nnacl_c/fp32_grad/lstm_grad_fp32.h"

int LstmGradWeightInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                             OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 5, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[FIRST_INPUT];
  const TensorC *H = inputs[SECOND_INPUT];
  const TensorC *Y = inputs[THIRD_INPUT];

  TensorC *output = outputs[FIRST_INPUT];
  for (int i = 0; i < outputs_size; i++) {
    SetDataTypeFormat(outputs[i], input);
  }

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  if (input->shape_size_ != C3NUM || H->shape_size_ != C3NUM || Y->shape_size_ != C3NUM) {
    return NNACL_ERR;
  }
  LstmGradParameter *param = (LstmGradParameter *)parameter;
  int has_bias = param->has_bias_;
  int output_shape[3] = {0, 1, 1};
  int dir_mul = (param->bidirectional_) ? C2NUM : C1NUM;
  int bias_multiplier = has_bias ? C2NUM : 0;
  // gate_size = 4 * hidden_size
  // output_shape[0] = (4 * hidden_size * input_size + 4 * hidden_size * hidden_size + bias_multiplier * 4 *
  // hidden_size) * dir_mul
  NNACL_CHECK_TRUE_RET((((int64_t)dir_mul) * ((int64_t)(param->hidden_size_))) <= (INT_MAX / C4NUM),
                       NNACL_ERRCODE_MUL_OVERFLOW);
  NNACL_CHECK_TRUE_RET(
    (((int64_t)(param->input_size_)) + ((int64_t)(param->hidden_size_))) <= (INT_MAX - bias_multiplier),
    NNACL_ERRCODE_ADD_OVERFLOW);
  output_shape[0] =
    (C4NUM * param->hidden_size_ * dir_mul) * (param->input_size_ + param->hidden_size_ + bias_multiplier);
  SetShapeArray(output, output_shape, C3NUM);

  return NNACL_OK;
}

REG_INFER(LSTMGradWeight, PrimType_LSTMGradWeight, LstmGradWeightInferShape)
