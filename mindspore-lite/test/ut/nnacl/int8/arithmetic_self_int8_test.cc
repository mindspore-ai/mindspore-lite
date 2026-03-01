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
#include "nnacl_c/int8/quant_dtype_cast_int8.h"
#include "nnacl_c/int8/quantize.h"
#include "nnacl_c/int8/arithmetic_self_int8.h"

namespace mindspore {
class ArithmeticSelfInt8Test : public ::testing::Test {
 public:
  ArithmeticSelfInt8Test() {}
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

TEST_F(ArithmeticSelfInt8Test, Testcase01) {
  std::vector<float> inputs = {0.5826,  1.3132, -0.3709, -0.7568, 1.6641, 0.9497, 0.3350, 0.3038, -0.0539,
                               -0.0863, 0.7875, -0.1636, -0.2774, 1.5798, 0.3837, 2.0600, 0.9837, 0.7603,
                               -0.9778, 0.3408, -0.1488, -0.8994, 1.1542, 0.0830, 3.6240};
  std::vector<float> outputs(1 * 1 * 5 * 5, 0.0f);
  std::vector<int8_t> quant_inputs(1 * 1 * 5 * 5, 0);
  std::vector<int8_t> quant_outputs(1 * 1 * 5 * 5, 0);
  std::vector<float> golden = {0.000000, 0.003290, 0.000000, 0.000000, 0.838931, 0.000000, 0.000000, 0.000000, 0.000000,
                               0.000000, 0.000000, 0.000000, 0.000000, 0.838931, 0.000000, 0.000000, 0.000000, 0.000000,
                               0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000};
  DoQuantizeFp32ToInt8(inputs.data(), quant_inputs.data(), 0.0180464033, -74, 25, -128, 127);
  const ArithSelfQuantArg param = {{1, 0}, {0.501961, -128}, -128, 127, 0, 0, 0};
  Int8ElementExp(quant_inputs.data(), quant_outputs.data(), 25, param);
  DoDequantizeInt8ToFp32(quant_outputs.data(), outputs.data(), 0.00328992424, -128, 25);
  for (auto iter = outputs.begin(); iter != outputs.end(); iter++) {
    printf("[%f]", *iter);
  }
  float sim = get_cosine_similarity(outputs.data(), golden.data(), 1 * 5 * 5);
  ASSERT_GT(sim, 0.9);
}
}  // namespace mindspore
