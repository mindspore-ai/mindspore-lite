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
#include "nnacl_c/errorcode.h"
#include "nnacl_c/base/space_to_depth_base.h"

namespace mindspore {
class SpaceToDepthFp32Test : public ::testing::Test {
 public:
  SpaceToDepthFp32Test() {}
};

float get_cosine_similarity(const float *arr1, const float *arr2, size_t cmp_size);
static float accuracy_threshold = 0.99;

// Testcase1: input 1x4x4x1 block_size=2, output 1x2x2x4
TEST_F(SpaceToDepthFp32Test, SpaceToDepth_1x4x4x1) {
  // input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
  std::vector<float> input = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  std::vector<float> benchmark = {1.0f, 2.0f,  5.0f,  6.0f,  3.0f,  4.0f,  7.0f,  8.0f,
                                  9.0f, 10.0f, 13.0f, 14.0f, 11.0f, 12.0f, 15.0f, 16.0f};
  const int length = 1 * 2 * 2 * 4;
  std::vector<float> output(length, 0.0f);

  int in_shape[] = {1, 4, 4, 1};
  int out_shape[] = {1, 2, 2, 4};
  SpaceToDepthParameter param = {};
  param.op_parameter_.thread_num_ = 1;
  param.block_size_ = 2;
  param.date_type_len = sizeof(float);

  auto ret = SpaceToDepthForNHWC(input.data(), output.data(), in_shape, out_shape, 4, &param, 0);
  ASSERT_EQ(ret, NNACL_OK);

  std::cout << "SpaceToDepthFp32Test-SpaceToDepth_1x4x4x1 output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nSpaceToDepthFp32Test-SpaceToDepth_1x4x4x1 benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: input 1x2x2x2 block_size=2, output 1x1x1x8
TEST_F(SpaceToDepthFp32Test, SpaceToDepth_1x2x2x2) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> benchmark = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  const int length = 1 * 1 * 1 * 8;
  std::vector<float> output(length, 0.0f);

  int in_shape[] = {1, 2, 2, 2};
  int out_shape[] = {1, 1, 1, 8};
  SpaceToDepthParameter param = {};
  param.op_parameter_.thread_num_ = 1;
  param.block_size_ = 2;
  param.date_type_len = sizeof(float);

  auto ret = SpaceToDepthForNHWC(input.data(), output.data(), in_shape, out_shape, 4, &param, 0);
  ASSERT_EQ(ret, NNACL_OK);

  std::cout << "SpaceToDepthFp32Test-SpaceToDepth_1x2x2x2 output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nSpaceToDepthFp32Test-SpaceToDepth_1x2x2x2 benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
