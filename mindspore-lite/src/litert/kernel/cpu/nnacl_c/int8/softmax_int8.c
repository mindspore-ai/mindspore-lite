/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "nnacl_c/int8/softmax_int8.h"
#include "nnacl_c/errorcode.h"

static inline int32_t SoftmaxInt8ScaleDiff(int32_t input_diff, int32_t input_multiplier,
                                           const SoftmaxQuantArg *quant_param) {
  return SaturatingRoundingDoublingHighMul(input_diff * input_multiplier, quant_param->output_multiplier_);
}

static inline int8_t SoftmaxInt8ToOutput(int shifted_scale, int exp_value, int output_shift,
                                         const SoftmaxQuantArg *quant_param) {
  int unsat_output = RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(shifted_scale, exp_value), output_shift);
  int raw_output = unsat_output + quant_param->output_activation_min_;
  return (int8_t)MSMAX(quant_param->output_activation_min_, MSMIN(raw_output, quant_param->output_activation_max_));
}

static void SoftmaxInt8BuildExpLut(int32_t *exp_lut, int32_t *rescaled_lut, int32_t input_multiplier,
                                   const SoftmaxQuantArg *quant_param) {
  for (int diff = 0; diff <= UINT8_MAX; ++diff) {
    int32_t input_scaled = SoftmaxInt8ScaleDiff(-diff, input_multiplier, quant_param);
    int32_t exp_value = exp_on_negative_values(input_scaled, 5);
    exp_lut[diff] = exp_value;
    rescaled_lut[diff] = Rescale(exp_value, 0, 12);
  }
}

int SoftmaxInt8(const int8_t *input_ptr, int8_t *output_ptr, int count, int32_t *exp_data, int32_t *sum_data,
                const int32_t *input_shape, int n_dim, int32_t axis, const SoftmaxQuantArg *quant_param) {
  int32_t exp_lut[UINT8_MAX + 1];
  int32_t rescaled_lut[UINT8_MAX + 1];
  int axis_shape_size = input_shape[axis];
  int inner_size = 1;
  if (n_dim > DIMENSION_5D) {
    return NNACL_ERR;
  }
  for (int i = axis + 1; i < n_dim; i++) {
    inner_size *= input_shape[i];
  }
  int32_t input_multiplier = 1 << (unsigned int)quant_param->shift_left_;
  SoftmaxInt8BuildExpLut(exp_lut, rescaled_lut, input_multiplier, quant_param);

  for (int o = 0; o < count; o++) {
    int outter_offset = o * axis_shape_size * inner_size;

    for (int c = 0; c < inner_size; c++) {
      int8_t max_row = quant_param->output_activation_min_;
      for (int i = 0; i < axis_shape_size; ++i) {
        int axis_offset = outter_offset + c + i * inner_size;
        max_row = MSMAX(max_row, input_ptr[axis_offset]);
      }

      int32_t exp_sum = 0;
      for (int i = 0; i < axis_shape_size; ++i) {
        int axis_offset = outter_offset + c + i * inner_size;
        const uint8_t input_diff = (uint8_t)((int32_t)max_row - (int32_t)input_ptr[axis_offset]);
        const int32_t exp_val = exp_lut[input_diff];
        exp_data[axis_offset] = exp_val;
        exp_sum += rescaled_lut[input_diff];
      }
      sum_data[c] = exp_sum;
    }
    for (int i = 0; i < axis_shape_size; ++i) {
      int axis_offset = outter_offset + i * inner_size;
      for (int c = 0; c < inner_size; ++c) {
        int num_bits_over_unit;
        int shifted_scale = ComputerReciprocal(sum_data[c], 12, &num_bits_over_unit);
        output_ptr[axis_offset + c] =
          SoftmaxInt8ToOutput(shifted_scale, exp_data[axis_offset + c], num_bits_over_unit + 31 - 8, quant_param);
      }
    }
  }
  return NNACL_OK;
}
