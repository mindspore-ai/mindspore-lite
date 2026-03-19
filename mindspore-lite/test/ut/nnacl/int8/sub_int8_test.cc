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
#include "nnacl_c/int8/sub_int8.h"
#include "nnacl_c/int8/quantize.h"
#include "nnacl_c/int8/arithmetic_int8.h"

namespace mindspore {
class SubInt8Test : public ::testing::Test {
 public:
  SubInt8Test() {}
};

float get_cosine_similarity_int8(const int8_t *arr1, const int8_t *arr2, size_t cmp_size);
static const float accuracy_threshold = 0.99;

// 4D_01::Broadcasting
TEST_F(SubInt8Test, ElementSubInt8_4D_01) {
  std::vector<int8_t> input0 = {-128, 60, -42, -41, -82, -55, -77, -1};
  std::vector<int8_t> input1 = {74, 41, 127, -7, 25, 80, 24, 102};
  const int out_shape = 1 * 2 * 4 * 4;
  std::vector<int8_t> output;
  output.resize(out_shape, 0);
  std::vector<int8_t> tile_input1;
  tile_input1.resize(out_shape, 0);
  std::vector<int8_t> tile_input2;
  tile_input2.resize(out_shape, 0);

  std::vector<int8_t> input0_shape = {1, 2, 4, 1};
  std::vector<int8_t> input1_shape = {1, 2, 1, 4};
  std::vector<int8_t> benchmark = {
    -86, -60, -128, -22, 63, 89,  21, 127, -18, 8,   -60, 46,  -17, 9,  -59, 47,
    -11, -54, -10,  -72, 11, -33, 11, -50, -7,  -50, -6,  -68, 53,  10, 54,  -8,
  };
  const SubQuantArg quant = {{0.000186498, 64},
                             {0.000186498, 64},
                             {0.000235443, 74},
                             -128,
                             127,
                             1073741824,
                             1073741824,
                             1701059113,
                             0,
                             0,
                             19,
                             1048576,
                             1048576,
                             0,
                             0,
                             0,
                             19};
  ArithmeticParameter tile_para = {
    {"", 0, 1, 0}, false, 4, 0, {1, 2, 4, 1}, 0, {1, 2, 1, 4}, 0, {1, 2, 4, 4}, 0, {}, {}, {}, {}, {}};
  if (input0_shape != input1_shape) {
    TileDimensionsInt8(input0.data(), input1.data(), tile_input1.data(), tile_input2.data(), &tile_para);
  }
  SubInt8(tile_input1.data(), tile_input2.data(), output.data(), 32, &quant);
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
