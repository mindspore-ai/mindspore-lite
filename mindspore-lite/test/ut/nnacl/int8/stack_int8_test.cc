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
#include "nnacl_c/base/stack_base.h"

namespace mindspore {
class StackInt8Test : public ::testing::Test {
 public:
  StackInt8Test() {}
};

float get_cosine_similarity_int8(const int8_t *arr1, const int8_t *arr2, size_t cmp_size);
static float accuracy_threshold = 0.99;

// Stack/Unstack are data-movement ops: the int8 path reuses the same byte-copy
// `Stack()` as fp32 (no quantisation math), so these cases mirror stack_fp32_test
// with int8 elements (incl. negative values to exercise the signed int8 range).

// Testcase1: 2 inputs, each 1x3 (int8) -> stack to [2, 3]
TEST_F(StackInt8Test, Stack_2Inputs_1x3) {
  std::vector<int8_t> input1 = {1, -2, 3};
  std::vector<int8_t> input2 = {4, -5, 6};
  std::vector<int8_t> benchmark = {1, -2, 3, 4, -5, 6};
  const int length = 2 * 3;
  std::vector<int8_t> output(length, 0);
  void *inputs[2] = {input1.data(), input2.data()};
  Stack(inputs, output.data(), 2, 3 * sizeof(int8_t), 0, 1);

  std::cout << "StackInt8Test-Stack_2Inputs_1x3 output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nStackInt8Test-Stack_2Inputs_1x3 benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: 3 inputs, each 2x2 (int8) -> stack to [3, 2, 2]
TEST_F(StackInt8Test, Stack_3Inputs_2x2) {
  std::vector<int8_t> input1 = {1, 2, 3, 4};
  std::vector<int8_t> input2 = {5, 6, 7, 8};
  std::vector<int8_t> input3 = {9, 10, 11, 12};
  std::vector<int8_t> benchmark = {1, 2, 5, 6, 9, 10, 3, 4, 7, 8, 11, 12};
  const int length = 3 * 2 * 2;
  std::vector<int8_t> output(length, 0);
  void *inputs[3] = {input1.data(), input2.data(), input3.data()};
  Stack(inputs, output.data(), 3, 2 * sizeof(int8_t), 0, 2);

  std::cout << "StackInt8Test-Stack_3Inputs_2x2 output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nStackInt8Test-Stack_3Inputs_2x2 benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: 2 inputs, each 2x3 (int8) -> stack to [2, 2, 3]
TEST_F(StackInt8Test, Stack_2Inputs_2x3) {
  std::vector<int8_t> input1 = {1, -2, 3, 4, -5, 6};
  std::vector<int8_t> input2 = {7, -8, 9, 10, -11, 12};
  std::vector<int8_t> benchmark = {1, -2, 3, 7, -8, 9, 4, -5, 6, 10, -11, 12};
  const int length = 2 * 2 * 3;
  std::vector<int8_t> output(length, 0);
  void *inputs[2] = {input1.data(), input2.data()};
  Stack(inputs, output.data(), 2, 3 * sizeof(int8_t), 0, 2);

  std::cout << "StackInt8Test-Stack_2Inputs_2x3 output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nStackInt8Test-Stack_2Inputs_2x3 benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
