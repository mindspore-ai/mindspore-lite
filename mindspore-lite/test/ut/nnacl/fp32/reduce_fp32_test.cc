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
#include "nnacl_c/fp32/reduce_fp32.h"

namespace mindspore {
class ReduceFp32Test : public ::testing::Test {
 public:
  ReduceFp32Test() {}
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

static const float accuracy_threshold = 0.999;

// Testcase1: ReduceProd along axis 1 of a 2x3 matrix -> {1*2*3, 4*5*6}
TEST_F(ReduceFp32Test, ReduceProd_Axis1) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> benchmark = {6.0f, 120.0f};
  std::vector<float> output(2, 0.0f);
  ReduceProd(2, 1, 3, input.data(), output.data(), 0, 1);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), 2);
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: ReduceProd along axis 0 of a 2x3 matrix -> {1*4, 2*5, 3*6}
TEST_F(ReduceFp32Test, ReduceProd_Axis0) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> benchmark = {4.0f, 10.0f, 18.0f};
  std::vector<float> output(3, 0.0f);
  ReduceProd(1, 3, 2, input.data(), output.data(), 0, 1);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), 3);
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: ReduceProd with a zero element -> product contains zero
TEST_F(ReduceFp32Test, ReduceProd_Zero) {
  std::vector<float> input = {2.0f, 0.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> benchmark = {0.0f, 120.0f};
  std::vector<float> output(2, 0.0f);
  ReduceProd(2, 1, 3, input.data(), output.data(), 0, 1);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), 2);
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase4: ReduceProd over all axes of a 2x3 matrix via two ReduceProd calls -> 720
// Exercises the multi-axis reduce path (per特性文档 axes=[0,1]).
TEST_F(ReduceFp32Test, ReduceProd_TwoAxes) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> mid(3, 0.0f);
  std::vector<float> output(1, 0.0f);
  std::vector<float> benchmark = {720.0f};
  ReduceProd(1, 3, 2, input.data(), mid.data(), 0, 1);
  ReduceProd(1, 1, 3, mid.data(), output.data(), 0, 1);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), 1);
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
