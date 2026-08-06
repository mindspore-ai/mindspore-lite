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
#include "nnacl_c/int8/hsigmoid_int8.h"

namespace mindspore {
class HardSigmoidInt8Test : public ::testing::Test {
 public:
  HardSigmoidInt8Test() {}
};

float get_cosine_similarity_int8(const int8_t *arr1, const int8_t *arr2, size_t cmp_size);
static const float accuracy_threshold = 0.99f;

// Testcase1: HardSigmoidInt8 with scale=1.0, zp=0 and 2D input tensor
TEST_F(HardSigmoidInt8Test, HardSigmoidInt8_2D) {
  std::vector<int8_t> input = {-128, -64, -16, 0, 16, 64, 100, 127};
  std::vector<int8_t> benchmark = {0, 0, 0, 1, 1, 1, 1, 1};
  const int out_shape = 2 * 4;
  std::vector<int8_t> output(out_shape, 0);
  const float input_scale = 1.0f;
  const int32_t input_zp = 0;
  const float output_scale = 1.0f;
  const int32_t output_zp = 0;
  const float alpha = 0.2f;
  const float beta = 0.5f;
  int8_t table[256] = {0};
  HardSigmoidInt8InitLUT(input_scale, input_zp, output_scale, output_zp, alpha, beta, table);
  HardSigmoidInt8(input.data(), out_shape, output.data(), table);

  std::cout << "HardSigmoidInt8Test-HardSigmoidInt8_2D output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nHardSigmoidInt8Test-HardSigmoidInt8_2D benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: HardSigmoidInt8 with scale=0.1, zp=0 and 3D input tensor
TEST_F(HardSigmoidInt8Test, HardSigmoidInt8_3D) {
  std::vector<int8_t> input = {-128, -64, -20, -5, 0, 20, 64, 100, 127};
  std::vector<int8_t> benchmark = {0, 0, 1, 4, 5, 9, 10, 10, 10};
  const int out_shape = 3 * 3;
  std::vector<int8_t> output(out_shape, 0);
  const float input_scale = 0.1f;
  const int32_t input_zp = 0;
  const float output_scale = 0.1f;
  const int32_t output_zp = 0;
  const float alpha = 0.2f;
  const float beta = 0.5f;
  int8_t table[256] = {0};
  HardSigmoidInt8InitLUT(input_scale, input_zp, output_scale, output_zp, alpha, beta, table);
  HardSigmoidInt8(input.data(), out_shape, output.data(), table);

  std::cout << "HardSigmoidInt8Test-HardSigmoidInt8_3D output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nHardSigmoidInt8Test-HardSigmoidInt8_3D benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: HardSigmoidInt8 with non-zero zero-point and 3D input tensor
TEST_F(HardSigmoidInt8Test, HardSigmoidInt8_3D_NonZeroZp) {
  std::vector<int8_t> input = {-128, -100, -60, -20, 0, 20, 40, 80, 100, 127, -1, 1};
  std::vector<int8_t> benchmark = {-5, -5, -3, 10, 16, 22, 28, 41, 45, 45, 16, 16};
  const int out_shape = 2 * 2 * 3;
  std::vector<int8_t> output(out_shape, 0);
  const float input_scale = 0.05f;
  const int32_t input_zp = -3;
  const float output_scale = 0.02f;
  const int32_t output_zp = -5;
  const float alpha = 0.125f;
  const float beta = 0.4f;
  int8_t table[256] = {0};
  HardSigmoidInt8InitLUT(input_scale, input_zp, output_scale, output_zp, alpha, beta, table);
  HardSigmoidInt8(input.data(), out_shape, output.data(), table);

  std::cout << "HardSigmoidInt8Test-HardSigmoidInt8_3D_NonZeroZp output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nHardSigmoidInt8Test-HardSigmoidInt8_3D_NonZeroZp benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
