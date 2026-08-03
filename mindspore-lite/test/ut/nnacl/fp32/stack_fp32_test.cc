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
class StackFp32Test : public ::testing::Test {
 public:
  StackFp32Test() {}
};

float get_cosine_similarity(const float *arr1, const float *arr2, size_t cmp_size);
static float accuracy_threshold = 0.99;

// Testcase1: 2 inputs, each 1x3 -> stack to [2, 3]
TEST_F(StackFp32Test, Stack_2Inputs_1x3) {
  std::vector<float> input1 = {1.0f, 2.0f, 3.0f};
  std::vector<float> input2 = {4.0f, 5.0f, 6.0f};
  // outer_start=0, outer_end=1, copy_size=3*sizeof(float), input_num=2
  // i=0: copy input1[0:3], input2[0:3] -> [1,2,3, 4,5,6]
  std::vector<float> benchmark = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const int length = 2 * 3;
  std::vector<float> output(length, 0.0f);
  void *inputs[2] = {input1.data(), input2.data()};
  Stack(inputs, output.data(), 2, 3 * sizeof(float), 0, 1);

  std::cout << "StackFp32Test-Stack_2Inputs_1x3 output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nStackFp32Test-Stack_2Inputs_1x3 benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: 3 inputs, each 2x2 -> stack to [3, 2, 2]
TEST_F(StackFp32Test, Stack_3Inputs_2x2) {
  std::vector<float> input1 = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> input2 = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> input3 = {9.0f, 10.0f, 11.0f, 12.0f};
  // outer_start=0, outer_end=2, copy_size=2*sizeof(float), input_num=3
  // i=0: copy input1[0:2], input2[0:2], input3[0:2] -> [1,2, 5,6, 9,10]
  // i=1: copy input1[2:4], input2[2:4], input3[2:4] -> [3,4, 7,8, 11,12]
  std::vector<float> benchmark = {1.0f, 2.0f, 5.0f, 6.0f, 9.0f, 10.0f, 3.0f, 4.0f, 7.0f, 8.0f, 11.0f, 12.0f};
  const int length = 3 * 2 * 2;
  std::vector<float> output(length, 0.0f);
  void *inputs[3] = {input1.data(), input2.data(), input3.data()};
  Stack(inputs, output.data(), 3, 2 * sizeof(float), 0, 2);

  std::cout << "StackFp32Test-Stack_3Inputs_2x2 output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nStackFp32Test-Stack_3Inputs_2x2 benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: 2 inputs, each 2x3 -> stack to [2, 2, 3]
TEST_F(StackFp32Test, Stack_2Inputs_2x3) {
  std::vector<float> input1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> input2 = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  // outer_start=0, outer_end=2, copy_size=3*sizeof(float), input_num=2
  // i=0: copy input1[0:3], input2[0:3] -> [1,2,3, 7,8,9]
  // i=1: copy input1[3:6], input2[3:6] -> [4,5,6, 10,11,12]
  std::vector<float> benchmark = {1.0f, 2.0f, 3.0f, 7.0f, 8.0f, 9.0f, 4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f};
  const int length = 2 * 2 * 3;
  std::vector<float> output(length, 0.0f);
  void *inputs[2] = {input1.data(), input2.data()};
  Stack(inputs, output.data(), 2, 3 * sizeof(float), 0, 2);

  std::cout << "StackFp32Test-Stack_2Inputs_2x3 output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nStackFp32Test-Stack_2Inputs_2x3 benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

}  // namespace mindspore
