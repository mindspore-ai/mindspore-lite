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
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include "gtest/gtest.h"
#include "nnacl_c/int8/space_to_batch_int8.h"
#include "nnacl_c/nnacl_common.h"

namespace mindspore {
class SpaceToBatchInt8Test : public ::testing::Test {
 public:
  SpaceToBatchInt8Test() {}
};

float get_cosine_similarity_int8(const int8_t *arr1, const int8_t *arr2, size_t cmp_size);
static float accuracy_threshold = 0.99;

// Testcase1: DoSpaceToBatchNHWCInt8, input [1,4,4,1] block_sizes=[2,2], output [4,2,2,1]
TEST_F(SpaceToBatchInt8Test, DoSpaceToBatch_1x4x4x1_block2x2) {
  std::vector<int8_t> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<int8_t> benchmark = {1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16};
  const int length = 4 * 2 * 2 * 1;
  std::vector<int8_t> output(length, 0);

  int32_t in_shape[4] = {1, 4, 4, 1};
  int32_t out_shape[4] = {4, 2, 2, 1};
  int32_t block_sizes[2] = {2, 2};

  DoSpaceToBatchNHWCInt8(input.data(), output.data(), block_sizes, in_shape, out_shape);

  std::cout << "SpaceToBatchInt8Test DoSpaceToBatch_1x4x4x1_block2x2 output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nSpaceToBatchInt8Test DoSpaceToBatch_1x4x4x1_block2x2 benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: DoSpaceToBatchPaddingNHWCInt8, input [1,2,3,2] block=[2,2] padding=[0,1,1,0], zp=-1
TEST_F(SpaceToBatchInt8Test, DoSpaceToBatchPadding_1x2x3x2_block2x2_pad0110_zpm1) {
  std::vector<int8_t> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int8_t> benchmark = {-1, -1, 3, 4, 1, 2, 5, 6, -1, -1, 9, 10, 7, 8, 11, 12};
  const int length = 4 * 1 * 2 * 2;
  std::vector<int8_t> output(length, 0);

  SpaceToBatchParameter param = {};
  param.block_sizes_[0] = 2;
  param.block_sizes_[1] = 2;
  param.paddings_[0] = 0;
  param.paddings_[1] = 1;
  param.paddings_[2] = 1;
  param.paddings_[3] = 0;
  param.input_shape_[0] = 1;
  param.input_shape_[1] = 2;
  param.input_shape_[2] = 3;
  param.input_shape_[3] = 2;
  param.output_shape_[0] = 4;
  param.output_shape_[1] = 1;
  param.output_shape_[2] = 2;
  param.output_shape_[3] = 2;
  param.m_ = 2;

  DoSpaceToBatchPaddingNHWCInt8(input.data(), output.data(), &param, -1);

  std::cout << "SpaceToBatchInt8Test DoSpaceToBatchPadding_1x2x3x2_block2x2_pad0110_zpm1 output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nSpaceToBatchInt8Test DoSpaceToBatchPadding_1x2x3x2_block2x2_pad0110_zpm1 benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
