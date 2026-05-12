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
class EluFp32Test : public ::testing::Test {
 public:
  EluFp32Test() {}
};

float get_cosine_similarity(const float *arr1, const float *arr2, size_t cmp_size);
extern float accuracy_threshold;

// Testcase1: Elu with alpha=1.0 and 2D input tensor
TEST_F(EluFp32Test, Elu_alpha1) {
  std::vector<float> input = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f};
  std::vector<float> benchmark = {0.0f, 1.0f, -0.632121f, 2.0f, -0.864665f, 0.5f};
  const int out_shape = 2 * 3;
  std::vector<float> output(out_shape, 0.0f);
  float alpha = 1.0f;
  Elu(input.data(), out_shape, output.data(), alpha);

  std::cout << "EluFp32Test-Elu_alpha1 output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\n EluFp32Test-Elu_alpha1 benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: Elu with alpha=2.0 and 2D input tensor
TEST_F(EluFp32Test, Elu_alpha2) {
  std::vector<float> input = {-1.0f, 0.0f, 1.0f, -0.5f};
  std::vector<float> benchmark = {-1.264241f, 0.0f, 1.0f, -0.786939f};
  const int out_shape = 2 * 2;
  std::vector<float> output(out_shape, 0.0f);
  float alpha = 2.0f;
  Elu(input.data(), out_shape, output.data(), alpha);

  std::cout << "EluFp32Test-Elu_alpha2 output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\n EluFp32Test-Elu_alpha2 benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: Elu with alpha=-1.0 and 3D input tensor
TEST_F(EluFp32Test, Elu_alphaf1) {
  std::vector<float> input = {-1.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, -4.0f, 4.0f, -5.0f, 5.0f, -6.0f, 6.0f};
  std::vector<float> benchmark = {0.632121f, 1.0f, 0.864665f, 2.0f, 0.950213f, 3.0f,
                                  0.981684f, 4.0f, 0.993262f, 5.0f, 0.997521f, 6.0f};
  const int out_shape = 2 * 2 * 3;
  std::vector<float> output(out_shape, 0.0f);
  float alpha = -1.0f;
  Elu(input.data(), out_shape, output.data(), alpha);

  std::cout << "EluFp32Test-Elu_alphaf1 output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\n EluFp32Test-Elu_alphaf1 benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
