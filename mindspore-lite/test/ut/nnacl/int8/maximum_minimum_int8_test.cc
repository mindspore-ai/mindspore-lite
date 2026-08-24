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
#include <iostream>
#include <cmath>
#include <vector>
#include "gtest/gtest.h"
#include "nnacl_c/int8/quantize.h"
#include "nnacl_c/int8/quant_dtype_cast_int8.h"
#include "nnacl_c/int8/arithmetic_int8.h"

namespace mindspore {
class MaximumMinimumInt8Test : public ::testing::Test {
 public:
  MaximumMinimumInt8Test() {}
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
  return dot_product / (norm1 * norm2);
}

static const float accuracy_threshold = 0.99;

// Testcase1: ElementMaximumInt8 on mixed-sign 1D input
TEST_F(MaximumMinimumInt8Test, Maximum_Mixed) {
  const float scale = 0.05f;
  const int32_t zp = 0;
  std::vector<float> input0 = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
  std::vector<float> input1 = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 3.0f};
  std::vector<float> benchmark = {1.0f, 1.0f, 3.0f, 2.0f, 5.0f, 3.0f};
  const int length = 6;
  std::vector<int8_t> quant_input0(length, 0);
  std::vector<int8_t> quant_input1(length, 0);
  std::vector<int8_t> quant_output(length, 0);
  std::vector<float> output(length, 0.0f);
  DoQuantizeFp32ToInt8(input0.data(), quant_input0.data(), scale, zp, length, -128, 127);
  DoQuantizeFp32ToInt8(input1.data(), quant_input1.data(), scale, zp, length, -128, 127);
  ArithmeticQuantArg quant_arg = {{scale, zp}, {scale, zp}, {scale, zp}};
  ElementMaximumInt8(quant_input0.data(), quant_input1.data(), quant_output.data(), length, &quant_arg);
  DoDequantizeInt8ToFp32(quant_output.data(), output.data(), scale, zp, length);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), length);
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: ElementMinimumInt8 on mixed-sign 1D input
TEST_F(MaximumMinimumInt8Test, Minimum_Mixed) {
  const float scale = 0.05f;
  const int32_t zp = 0;
  std::vector<float> input0 = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
  std::vector<float> input1 = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 3.0f};
  std::vector<float> benchmark = {0.0f, -2.0f, -1.0f, -4.0f, -2.0f, -6.0f};
  const int length = 6;
  std::vector<int8_t> quant_input0(length, 0);
  std::vector<int8_t> quant_input1(length, 0);
  std::vector<int8_t> quant_output(length, 0);
  std::vector<float> output(length, 0.0f);
  DoQuantizeFp32ToInt8(input0.data(), quant_input0.data(), scale, zp, length, -128, 127);
  DoQuantizeFp32ToInt8(input1.data(), quant_input1.data(), scale, zp, length, -128, 127);
  ArithmeticQuantArg quant_arg = {{scale, zp}, {scale, zp}, {scale, zp}};
  ElementMinimumInt8(quant_input0.data(), quant_input1.data(), quant_output.data(), length, &quant_arg);
  DoDequantizeInt8ToFp32(quant_output.data(), output.data(), scale, zp, length);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), length);
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: ElementMaximumInt8 with broadcasting [1,2,4,1] x [1,2,1,4] -> [1,2,4,4]
// Exercises TileDimensionsInt8 so the broadcast path (per特性文档) is covered.
TEST_F(MaximumMinimumInt8Test, Maximum_Broadcast) {
  const int out_size = 1 * 2 * 4 * 4;
  std::vector<int8_t> input0 = {-60, 71, 126, 127, 56, -91, 45, 22};
  std::vector<int8_t> input1 = {-61, -111, 117, 13, -25, -21, -128, -94};
  std::vector<int8_t> tile_input0(out_size, 0);
  std::vector<int8_t> tile_input1(out_size, 0);
  std::vector<int8_t> output(out_size, 0);
  std::vector<int8_t> benchmark = {-60, -60, 117, 13, 71,  71,  117, 71,  126, 126, 126, 126, 127, 127, 127, 127,
                                   56,  56,  56,  56, -25, -21, -91, -91, 45,  45,  45,  45,  22,  22,  22,  22};
  ArithmeticParameter tile_para = {
    {"", 5, 1, 0}, true,           4,  0, {1, 2, 4, 1}, 8, {1, 2, 1, 4}, 8, {1, 2, 4, 4}, 32, {8, 4, 1, 1},
    {8, 4, 4, 1},  {32, 16, 4, 1}, {}, {}};
  TileDimensionsInt8(input0.data(), input1.data(), tile_input0.data(), tile_input1.data(), &tile_para);
  ArithmeticQuantArg quant_arg = {{1.0f, 0}, {1.0f, 0}, {1.0f, 0}};
  ElementMaximumInt8(tile_input0.data(), tile_input1.data(), output.data(), out_size, &quant_arg);

  std::vector<float> output_fp32(out_size, 0.0f);
  std::vector<float> benchmark_fp32(benchmark.begin(), benchmark.end());
  for (int i = 0; i < out_size; ++i) {
    output_fp32[i] = static_cast<float>(output[i]);
  }
  float similarity = get_cosine_similarity(output_fp32.data(), benchmark_fp32.data(), out_size);
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
