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
#include "nnacl_c/fp32/arithmetic_self_fp32.h"

namespace mindspore {
class ErfFp32Test : public ::testing::Test {
 public:
  ErfFp32Test() {}
};

float get_cosine_similarity(const float *arr1, const float *arr2, size_t cmp_size);
static float accuracy_threshold = 0.99;

// Testcase1: Erf with 2D input tensor (2x3)
TEST_F(ErfFp32Test, Erf_2D) {
  std::vector<float> input = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f};
  std::vector<float> benchmark = {0.0f, 0.842701f, -0.842701f, 0.995322f, -0.995322f, 0.520500f};
  const int length = 2 * 3;
  std::vector<float> output(length, 0.0f);
  ElementErf(input.data(), output.data(), length);

  std::cout << "ErfFp32Test-Erf_2D output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nErfFp32Test-Erf_2D benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: Erf with 2D input tensor (1x2x2)
TEST_F(ErfFp32Test, Erf_3D) {
  std::vector<float> input = {-1.0f, 0.0f, 1.0f, -0.5f};
  std::vector<float> benchmark = {-0.842701f, 0.0f, 0.842701f, -0.520500f};
  const int length = 1 * 2 * 2;
  std::vector<float> output(length, 0.0f);
  ElementErf(input.data(), output.data(), length);

  std::cout << "ErfFp32Test-Erf_3D output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\n ErfFp32Test-Erf_3D benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: Erf with 4D input tensor (1x2x2x3)
TEST_F(ErfFp32Test, Erf_4D) {
  std::vector<float> input = {-1.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, -4.0f, 4.0f, -5.0f, 5.0f, -6.0f, 6.0f};
  std::vector<float> benchmark = {-0.842701f, 0.842701f, -0.995322f, 0.995322f, -0.999978f, 0.999978f,
                                  -1.0f,      1.0f,      -1.0f,      1.0f,      -1.0f,      1.0f};
  const int length = 1 * 2 * 2 * 3;
  std::vector<float> output(length, 0.0f);
  ElementErf(input.data(), output.data(), length);

  std::cout << "ErfFp32Test-Erf_4D output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\n ErfFp32Test-Erf_4D benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
