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
#include "nnacl_c/int8/celu_int8.h"

namespace mindspore {
class CeluInt8Test : public ::testing::Test {
 public:
  CeluInt8Test() {}
};

float get_cosine_similarity_int8(const int8_t *arr1, const int8_t *arr2, size_t cmp_size);
static const float accuracy_threshold = 0.99;

// Testcase1: CeluInt8 with scale=1.0 and 2D input tensor
TEST_F(CeluInt8Test, CeluInt8_2D) {
  std::vector<int8_t> input = {0, 1, -1, 2, -2, 50, -50, 127};
  std::vector<int8_t> benchmark = {0, 1, -1, 2, -1, 50, -1, 127};
  const int out_shape = static_cast<int>(input.size());
  std::vector<int8_t> output(out_shape, 0);
  CeluQuantArg quant_arg = {};
  quant_arg.in_args_.scale_ = 1.0f;
  quant_arg.in_args_.zp_ = 0;
  quant_arg.out_args_.scale_ = 1.0f;
  quant_arg.out_args_.zp_ = 0;
  quant_arg.alpha_ = 1.0f;
  int8_t table[256] = {0};
  CeluInt8InitLUT(&quant_arg, table);
  CeluInt8(input.data(), out_shape, output.data(), table);

  std::cout << "CeluInt8Test-CeluInt8_2D output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\n CeluInt8Test-CeluInt8_2D benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: CeluInt8 with non-trivial scale and 3D input tensor
TEST_F(CeluInt8Test, CeluInt8_3D) {
  std::vector<int8_t> input = {-128, -96, -64, -32, -1, 0, 1, 32, 64, 96, 127, 15};
  std::vector<int8_t> benchmark = {-10, -10, -10, -10, -1, 0, 1, 32, 64, 96, 127, 15};
  const int out_shape = 2 * 2 * 3;
  std::vector<int8_t> output(out_shape, 0);
  CeluQuantArg quant_arg = {};
  quant_arg.in_args_.scale_ = 0.1f;
  quant_arg.in_args_.zp_ = 0;
  quant_arg.out_args_.scale_ = 0.1f;
  quant_arg.out_args_.zp_ = 0;
  quant_arg.alpha_ = 1.0f;
  int8_t table[256] = {0};
  CeluInt8InitLUT(&quant_arg, table);
  CeluInt8(input.data(), out_shape, output.data(), table);

  std::cout << "CeluInt8Test-CeluInt8_3D output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\n CeluInt8Test-CeluInt8_3D benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: CeluInt8 with asymmetric quantization and alpha=2.0
TEST_F(CeluInt8Test, CeluInt8_3D_Asymmetric) {
  std::vector<int8_t> input = {-128, -105, -82, -59, -35, -12, 11, 34, 57, 81, 104, 127};
  std::vector<int8_t> benchmark = {-128, -128, -128, -128, -128, -128, -128, -77, -27, 26, 77, 127};
  const int out_shape = 2 * 2 * 3;
  std::vector<int8_t> output(out_shape, 0);
  CeluQuantArg quant_arg = {};
  quant_arg.in_args_.scale_ = 0.04313725605607032776f;
  quant_arg.in_args_.zp_ = 11;
  quant_arg.out_args_.scale_ = 0.01960784383118152618f;
  quant_arg.out_args_.zp_ = -128;
  quant_arg.alpha_ = 2.0f;
  int8_t table[256] = {0};
  CeluInt8InitLUT(&quant_arg, table);
  CeluInt8(input.data(), out_shape, output.data(), table);

  std::cout << "CeluInt8Test-CeluInt8_3D_Asymmetric output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\n CeluInt8Test-CeluInt8_3D_Asymmetric benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
