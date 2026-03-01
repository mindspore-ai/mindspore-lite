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
#include <iostream>
#include <cmath>
#include "gtest/gtest.h"
#include "nnacl_c/int8/pooling_int8.h"
#include "nnacl_c/kernel/pooling.h"
#include "nnacl_c/int8/quant_dtype_cast_int8.h"

namespace mindspore {
class AvgPoolInt8Test : public ::testing::Test {
 public:
  AvgPoolInt8Test() {}
};

static float get_cosine_similarity(const float *arr1, const float *arr2, size_t size) {
  if (arr1 == nullptr || arr2 == nullptr || size == 0) {
    return 0.0;
  }
  float dot_product = 0.0f;
  float norm1 = 0.0f;
  float norm2 = 0.0f;
  for (size_t i = 0; i < size; i++) {
    dot_product += arr1[i] * arr2[i];
    norm1 += arr1[i] * arr1[i];
    norm2 += arr2[i] * arr2[i];
  }
  norm1 = std::sqrt(norm1);
  norm2 = std::sqrt(norm2);
  if (norm1 == 0 && norm2 == 0) {
    return 1.0;
  }
  float cosine_similarity = dot_product / (norm1 * norm2);
  return cosine_similarity;
}

TEST_F(AvgPoolInt8Test, Testcase01) {
  std::vector<float> inputs = {0.5826,  1.3132, -0.3709, -0.7568, 1.6641, 0.9497, 0.3350, 0.3038, -0.0539,
                               -0.0863, 0.7875, -0.1636, -0.2774, 1.5798, 0.3837, 2.0600, 0.9837, 0.7603,
                               -0.9778, 0.3408, -0.1488, -0.8994, 1.1542, 0.0830, 3.6240};
  std::vector<float> outputs(1 * 1 * 5 * 5, 0.0f);
  std::vector<float> input_shape = {1, 5, 5, 1};
  std::vector<float> output_shape = {1, 5, 5, 1};
  std::vector<int8_t> quant_inputs(1 * 1 * 5 * 5, 0);
  std::vector<int8_t> quant_outputs(1 * 1 * 5 * 5, 0);
  std::vector<float> golden = {0.792872, 0.523098, 0.125017, 0.108567, 0.180946, 0.631665, 0.378341, 0.217135, 0.269774,
                               0.450720, 0.829061, 0.648115, 0.289513, 0.217135, 0.197395, 0.434270, 0.470459, 0.253324,
                               0.740233, 0.006580, 0.506648, 0.648115, 0.180946, 0.829061, 0.776422};
  DoQuantizeFp32ToInt8(inputs.data(), quant_inputs.data(), 0.0180464033, -74, 25, -128, 127);
  const PoolingParameter pooling_parameter = {
    {"", 17, 1, 0}, (PoolMode)(2), (RoundType)(2), (PadType)(0), (ActType)(0), 0, false, 3, 3, 1, 1, 1, 1, 1, 1};
  PoolingComputeParam compute = {5, 5, 1, 1, 5, 5, 1, 1, 3, 3, -128, 127};
  static QuantArg quant_in = {0.018046033037, -74};
  static QuantArg quant_out = {0.00328992424, -128};
  static QuantArg *quant[2] = {&quant_in, &quant_out};
  static QuantArg **quant_addr = (QuantArg **)(&quant);
  AvgPoolingInt8(quant_inputs.data(), quant_outputs.data(), &pooling_parameter, &compute, quant_addr);
  DoDequantizeInt8ToFp32(quant_outputs.data(), outputs.data(), 0.00328992424, -128, 25);
  for (auto iter = outputs.begin(); iter != outputs.end(); iter++) {
    printf("[%f]", *iter);
  }
  float sim = get_cosine_similarity(outputs.data(), golden.data(), 1 * 5 * 5);
  ASSERT_GT(sim, 0.9);
}
}  // namespace mindspore
