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
#include "gtest/gtest.h"
#include "nnacl_c/int8/quantize.h"
#include "nnacl_c/arithmetic_parameter.h"
#include "nnacl_c/int8/div_int8.h"
#include "nnacl_c/int8/arithmetic_int8.h"

namespace mindspore {
class DivInt8Test : public ::testing::Test {
 public:
  DivInt8Test() {}
};

float get_cosine_similarity_int8(const int8_t *arr1, const int8_t *arr2, size_t cmp_size);
static const float accuracy_threshold = 0.99;

// input1:1*2*4*1 ,input2:1*2*1*4
TEST_F(DivInt8Test, Broadcasting) {
  std::vector<int8_t> input0 = {-88, 35, -116, -10, -62, -40, 29, -128};
  std::vector<int8_t> input1 = {-68, -26, -2, 93, 52, 38, 127, 5};
  const int out_shape = 1 * 2 * 4 * 4;
  std::vector<int8_t> output;
  output.resize(out_shape, 0);

  std::vector<int8_t> tile_input1;
  tile_input1.resize(out_shape, 0);
  std::vector<int8_t> tile_input2;
  tile_input2.resize(out_shape, 0);
  std::vector<int8_t> input0_shape = {1, 2, 4, 1};
  std::vector<int8_t> input1_shape = {1, 2, 1, 4};
  std::vector<int8_t> benchmark = {82, 20, 25, 29, -128, 72, 55, 41, 127, 9,  18, 27, -57, 53, 44, 37,
                                   32, 32, 32, 32, 35,   35, 34, 36, 43,  45, 39, 51, 23,  22, 27, 17};
  const DivQuantArg quant = {{0.0208667, 60}, {0.0208667, 60}, {0.0705703, 32}, -128, 127, 1901900160, 4};
  ArithmeticParameter tile_para = {
    {"", 0, 1, 0}, true, 4, 0, {1, 2, 4, 1}, 0, {1, 2, 1, 4}, 0, {1, 2, 4, 4}, 0, {}, {}, {}, {}, {}};
  if (input0_shape != input1_shape) {
    TileDimensionsInt8(input0.data(), input1.data(), tile_input1.data(), tile_input2.data(), &tile_para);
  }
  DivInt8(tile_input1.data(), tile_input2.data(), output.data(), 32, &quant);
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
