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
#include "nnacl_c/int8/add_int8.h"
#include "nnacl_c/int8/arithmetic_int8.h"

namespace mindspore {
class AddInt8Test : public ::testing::Test {
 public:
  AddInt8Test() {}
};

float get_cosine_similarity_int8(const int8_t *arr1, const int8_t *arr2, size_t cmp_size);
static const float accuracy_threshold = 0.99;

// Test case: input0 [1,2,4,1] + input1 [1,2,1,4] -> output [1,2,4,4]
// This tests dual-direction broadcasting where both inputs need to be expanded

TEST_F(AddInt8Test, Broadcasting) {
  std::vector<int8_t> input0 = {-60, 71, 126, 127, 56, -91, 45, 22};
  std::vector<int8_t> input1 = {-61, -111, 117, 13, -25, -21, -128, -94};
  const int out_shape = 1 * 2 * 4 * 4;
  std::vector<int8_t> output;
  output.resize(out_shape, 0);
  std::vector<int8_t> tile_input1;
  tile_input1.resize(out_shape, 0);
  std::vector<int8_t> tile_input2;
  tile_input2.resize(out_shape, 0);
  std::vector<int8_t> input0_shape = {1, 2, 4, 1};
  std::vector<int8_t> input1_shape = {1, 2, 1, 4};
  std::vector<int8_t> benchmark = {-74, -102, 24,  -33, -2,  -29, 96,   39,   28, 1, 126, 69,  29, 1,  127, 70,
                                   10,  12,   -47, -28, -71, -69, -128, -109, 4,  6, -53, -34, -9, -7, -66, -47};
  const AddQuantParameter quant = {20, -128, 127,       {-14, 0, 0, 1073741824}, {-14, 0, 0, 1073741824}, 8,
                                   0,  19,   1181851660};
  ArithmeticParameter tile_para = {
    {"", 5, 1, 0}, true,           4,  0, {1, 2, 4, 1}, 8, {1, 2, 1, 4}, 8, {1, 2, 4, 4}, 32, {8, 4, 1, 1},
    {8, 4, 4, 1},  {32, 16, 4, 1}, {}, {}};
  if (input0_shape != input1_shape) {
    TileDimensionsInt8(input0.data(), input1.data(), tile_input1.data(), tile_input2.data(), &tile_para);
  }
  AddInt8(tile_input1.data(), tile_input2.data(), output.data(), 32, &quant);
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
