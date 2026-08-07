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
#include "nnacl_c/fp32/activation_fp32.h"

namespace mindspore {
class CeluFp32Test : public ::testing::Test {
 public:
  CeluFp32Test() {}
};

float get_cosine_similarity(const float *arr1, const float *arr2, size_t cmp_size);
extern float accuracy_threshold;

// Testcase1: Celu with alpha=1.0 and 2D input tensor
TEST_F(CeluFp32Test, Celu_alpha1) {
  std::vector<float> input = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f};
  std::vector<float> benchmark = {0.0f, 1.0f, -0.632121f, 2.0f, -0.864665f, 0.5f};
  const int out_shape = 2 * 3;
  std::vector<float> output(out_shape, 0.0f);
  float alpha = 1.0f;
  Celu(input.data(), out_shape, output.data(), alpha);

  std::cout << "CeluFp32Test-Celu_alpha1 output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\n CeluFp32Test-Celu_alpha1 benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: Celu with alpha=2.0 and 2D input tensor
TEST_F(CeluFp32Test, Celu_alpha2) {
  std::vector<float> input = {-1.5f, -0.5f, 0.0f, 0.5f};
  std::vector<float> benchmark = {-1.055267f, -0.442398f, 0.0f, 0.5f};
  const int out_shape = 2 * 2;
  std::vector<float> output(out_shape, 0.0f);
  float alpha = 2.0f;
  Celu(input.data(), out_shape, output.data(), alpha);

  std::cout << "CeluFp32Test-Celu_alpha2 output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\n CeluFp32Test-Celu_alpha2 benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: Celu with alpha=0.5 and 3D input tensor
TEST_F(CeluFp32Test, Celu_alpha05) {
  std::vector<float> input = {-3.0f, -2.0f, -1.0f, -0.25f, 0.25f, 1.0f, 2.0f, 3.0f, -4.0f, 4.0f, -5.0f, 5.0f};
  std::vector<float> benchmark = {-0.498761f, -0.490842f, -0.432332f, -0.196735f, 0.25f,      1.0f,
                                  2.0f,       3.0f,       -0.499832f, 4.0f,       -0.499977f, 5.0f};
  const int out_shape = 2 * 2 * 3;
  std::vector<float> output(out_shape, 0.0f);
  float alpha = 0.5f;
  Celu(input.data(), out_shape, output.data(), alpha);

  std::cout << "CeluFp32Test-Celu_alpha05 output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\n CeluFp32Test-Celu_alpha05 benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
