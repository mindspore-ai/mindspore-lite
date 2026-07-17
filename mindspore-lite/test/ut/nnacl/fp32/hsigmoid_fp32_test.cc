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
#include <iostream>
#include <cmath>
#include <vector>
#include "gtest/gtest.h"
#include "nnacl_c/fp32/activation_fp32.h"

namespace mindspore {
class HSigmoidFp32Test : public ::testing::Test {
 public:
  HSigmoidFp32Test() {}
};

float get_cosine_similarity(const float *arr1, const float *arr2, size_t cmp_size);
static const float accuracy_threshold = 0.99f;

// Testcase1: HSigmoid with alpha=0.2, beta=0.5 and 2D input tensor
TEST_F(HSigmoidFp32Test, HSigmoid_alpha0p2_beta0p5_2D) {
  std::vector<float> input = {-6.0f, -3.0f, 0.0f, 1.5f, 3.0f, 6.0f};
  std::vector<float> benchmark = {0.000000f, 0.000000f, 0.500000f, 0.800000f, 1.000000f, 1.000000f};
  const int out_shape = 2 * 3;
  std::vector<float> output(out_shape, 0.0f);
  const float alpha = 0.2f;
  const float beta = 0.5f;
  HSigmoid(input.data(), out_shape, output.data(), alpha, beta);

  std::cout << "HSigmoidFp32Test-HSigmoid_alpha0p2_beta0p5_2D output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nHSigmoidFp32Test-HSigmoid_alpha0p2_beta0p5_2D benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: HSigmoid with alpha=1/6, beta=0.5 and 2D input tensor
TEST_F(HSigmoidFp32Test, HSigmoid_alpha1div6_beta0p5_2D) {
  std::vector<float> input = {-4.0f, -1.0f, 0.0f, 1.0f};
  std::vector<float> benchmark = {0.000000f, 0.333333f, 0.500000f, 0.666667f};
  const int out_shape = 2 * 2;
  std::vector<float> output(out_shape, 0.0f);
  const float alpha = 1.0f / 6.0f;
  const float beta = 0.5f;
  HSigmoid(input.data(), out_shape, output.data(), alpha, beta);

  std::cout << "HSigmoidFp32Test-HSigmoid_alpha1div6_beta0p5_2D output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nHSigmoidFp32Test-HSigmoid_alpha1div6_beta0p5_2D benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: HSigmoid with alpha=0.125, beta=0.4 and 3D input tensor
TEST_F(HSigmoidFp32Test, HSigmoid_alpha0p125_beta0p4_3D) {
  std::vector<float> input = {-10.0f, -2.5f, -0.5f, 0.5f, 2.5f, 10.0f, -1.2f, 1.2f, -3.0f, 3.0f, 4.2f, -4.2f};
  std::vector<float> benchmark = {0.000000f, 0.087500f, 0.337500f, 0.462500f, 0.712500f, 1.000000f,
                                  0.250000f, 0.550000f, 0.025000f, 0.775000f, 0.925000f, 0.000000f};
  const int out_shape = 2 * 2 * 3;
  std::vector<float> output(out_shape, 0.0f);
  const float alpha = 0.125f;
  const float beta = 0.4f;
  HSigmoid(input.data(), out_shape, output.data(), alpha, beta);

  std::cout << "HSigmoidFp32Test-HSigmoid_alpha0p125_beta0p4_3D output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nHSigmoidFp32Test-HSigmoid_alpha0p125_beta0p4_3D benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
