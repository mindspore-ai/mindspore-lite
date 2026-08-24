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
#include "nnacl_c/fp32/arithmetic_fp32.h"
#include "nnacl_c/fp32/add_fp32.h"

namespace mindspore {
class ArithmeticFp32Test : public ::testing::Test {
 public:
  ArithmeticFp32Test() {}
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

// Testcase1: ElementMaximum on mixed-sign 1D input
TEST_F(ArithmeticFp32Test, Maximum_Mixed) {
  std::vector<float> input0 = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
  std::vector<float> input1 = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 3.0f};
  std::vector<float> benchmark = {1.0f, 1.0f, 3.0f, 2.0f, 5.0f, 3.0f};
  const int length = 6;
  std::vector<float> output(length, 0.0f);
  ElementMaximum(input0.data(), input1.data(), output.data(), length);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), length);
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: ElementMinimum on mixed-sign 1D input
TEST_F(ArithmeticFp32Test, Minimum_Mixed) {
  std::vector<float> input0 = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
  std::vector<float> input1 = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 3.0f};
  std::vector<float> benchmark = {0.0f, -2.0f, -1.0f, -4.0f, -2.0f, -6.0f};
  const int length = 6;
  std::vector<float> output(length, 0.0f);
  ElementMinimum(input0.data(), input1.data(), output.data(), length);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), length);
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: ElementAdd on mixed-sign 2D input (2x3)
TEST_F(ArithmeticFp32Test, Add_Mixed_2D) {
  std::vector<float> input0 = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
  std::vector<float> input1 = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 3.0f};
  std::vector<float> benchmark = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f};
  const int length = 6;
  std::vector<float> output(length, 0.0f);
  ElementAdd(input0.data(), input1.data(), output.data(), length);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), length);
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase4: ElementMaximum on all-positive 1D input
TEST_F(ArithmeticFp32Test, Maximum_Positive) {
  std::vector<float> input0 = {0.5f, 1.5f, 2.5f, 3.5f};
  std::vector<float> input1 = {1.0f, 1.0f, 3.0f, 3.0f};
  std::vector<float> benchmark = {1.0f, 1.5f, 3.0f, 3.5f};
  const int length = 4;
  std::vector<float> output(length, 0.0f);
  ElementMaximum(input0.data(), input1.data(), output.data(), length);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), length);
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase5: ElementAdd on all-negative 1D input
TEST_F(ArithmeticFp32Test, Add_Negative) {
  std::vector<float> input0 = {-1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> input1 = {-0.5f, -0.5f, -0.5f, -0.5f};
  std::vector<float> benchmark = {-1.5f, -2.5f, -3.5f, -4.5f};
  const int length = 4;
  std::vector<float> output(length, 0.0f);
  ElementAdd(input0.data(), input1.data(), output.data(), length);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), length);
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
