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
class GeluFp32Test : public ::testing::Test {
 public:
  GeluFp32Test() {}
};

float get_cosine_similarity(const float *arr1, const float *arr2, size_t cmp_size);
static float accuracy_threshold = 0.99;

// Testcase1: Gelu (erf form, approximate=false) with 2D input tensor (2x3)
TEST_F(GeluFp32Test, Gelu_erf_2D) {
  std::vector<float> input = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f};
  std::vector<float> benchmark = {0.000000f, 0.841345f, -0.158655f, 1.954500f, -0.045500f, 0.345731f};
  const int length = 2 * 3;
  std::vector<float> output(length, 0.0f);
  Gelu(input.data(), length, output.data(), false);

  std::cout << "GeluFp32Test-Gelu_erf_2D output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\n GeluFp32Test-Gelu_erf_2D benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: Gelu (tanh form, approximate=true) with 2D input tensor (2x3)
// Covers the tanh-approximation dispatch so the erf/tanh branch is actually exercised.
TEST_F(GeluFp32Test, Gelu_tanh_2D) {
  std::vector<float> input = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f};
  std::vector<float> benchmark = {0.000000f, 0.841192f, -0.158808f, 1.954598f, -0.045402f, 0.345714f};
  const int length = 2 * 3;
  std::vector<float> output(length, 0.0f);
  Gelu(input.data(), length, output.data(), true);

  std::cout << "GeluFp32Test-Gelu_tanh_2D output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\n GeluFp32Test-Gelu_tanh_2D benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: Gelu (erf form) with 3D input tensor (2x2x2), mixed sign domain
TEST_F(GeluFp32Test, Gelu_erf_3D) {
  std::vector<float> input = {-1.0f, 0.0f, 1.0f, -0.5f, 3.0f, -3.0f, 0.25f, -0.25f};
  std::vector<float> benchmark = {-0.158655f, 0.000000f,  0.841345f, -0.154269f,
                                  2.995950f,  -0.004050f, 0.149677f, -0.100323f};
  const int length = 2 * 2 * 2;
  std::vector<float> output(length, 0.0f);
  Gelu(input.data(), length, output.data(), false);

  std::cout << "GeluFp32Test-Gelu_erf_3D output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\n GeluFp32Test-Gelu_erf_3D benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase4: Gelu (erf form) with 4D input tensor (1x2x2x3), wider value range
TEST_F(GeluFp32Test, Gelu_erf_4D) {
  std::vector<float> input = {-1.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, -4.0f, 4.0f, 0.5f, -0.5f, 6.0f, -6.0f};
  std::vector<float> benchmark = {-0.158655f, 0.841345f, -0.045500f, 1.954500f,  -0.004050f, 2.995950f,
                                  -0.000127f, 3.999873f, 0.345731f,  -0.154269f, 6.000000f,  -0.000000f};
  const int length = 1 * 2 * 2 * 3;
  std::vector<float> output(length, 0.0f);
  Gelu(input.data(), length, output.data(), false);

  std::cout << "GeluFp32Test-Gelu_erf_4D output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\n GeluFp32Test-Gelu_erf_4D benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase5: Gelu (tanh form) with the same 4D tensor as Testcase4, so erf vs tanh
// dispatch on a larger tensor is directly comparable.
TEST_F(GeluFp32Test, Gelu_tanh_4D) {
  std::vector<float> input = {-1.0f, 1.0f, -2.0f, 2.0f, -3.0f, 3.0f, -4.0f, 4.0f, 0.5f, -0.5f, 6.0f, -6.0f};
  std::vector<float> benchmark = {-0.158808f, 0.841192f, -0.045402f, 1.954598f,  -0.003637f, 2.996363f,
                                  -0.000070f, 3.999930f, 0.345714f,  -0.154286f, 6.000000f,  -0.000000f};
  const int length = 1 * 2 * 2 * 3;
  std::vector<float> output(length, 0.0f);
  Gelu(input.data(), length, output.data(), true);

  std::cout << "GeluFp32Test-Gelu_tanh_4D output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\n GeluFp32Test-Gelu_tanh_4D benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Note: an all-zero input is intentionally not a case here — gelu(0) = 0 is correct,
// but two all-zero vectors have an undefined cosine (0/0), so the shared cosine
// assertion cannot express it. The erf/tanh dispatch and shape coverage above are
// sufficient for this kernel.
}  // namespace mindspore
