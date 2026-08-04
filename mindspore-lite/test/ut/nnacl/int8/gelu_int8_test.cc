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
#include "nnacl_c/int8/gelu_int8.h"

namespace mindspore {
class GeluInt8Test : public ::testing::Test {
 public:
  GeluInt8Test() {}
};

float get_cosine_similarity_int8(const int8_t *arr1, const int8_t *arr2, size_t cmp_size);
static float accuracy_threshold = 0.99;

// Testcase1: GeluInt8 (erf LUT, approximate=false), trivial quant scale=1.0 zp=0
TEST_F(GeluInt8Test, GeluInt8_erf_simple) {
  std::vector<int8_t> input = {0, 1, -1, 2, -2, 50, -50, 127};
  std::vector<int8_t> benchmark = {0, 1, 0, 2, 0, 50, 0, 127};
  const int length = input.size();
  std::vector<int8_t> output(length, 0);
  GeluQuantArg quant_arg = {};
  quant_arg.in_args_.scale_ = 1.0f;
  quant_arg.in_args_.zp_ = 0;
  quant_arg.out_args_.scale_ = 1.0f;
  quant_arg.out_args_.zp_ = 0;
  int8_t table[256] = {0};
  GeluInt8InitLUT(&quant_arg, table, false);
  GeluInt8(input.data(), length, output.data(), table);

  std::cout << "GeluInt8Test-GeluInt8_erf_simple output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\n GeluInt8Test-GeluInt8_erf_simple benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: GeluInt8 (tanh LUT, approximate=true), same trivial quant — exercises the
// tanh branch of GeluInt8InitLUT (after int8 quantisation erf/tanh outputs coincide for
// these inputs, but the init path is genuinely dispatched).
TEST_F(GeluInt8Test, GeluInt8_tanh_simple) {
  std::vector<int8_t> input = {0, 1, -1, 2, -2, 50, -50, 127};
  std::vector<int8_t> benchmark = {0, 1, 0, 2, 0, 50, 0, 127};
  const int length = input.size();
  std::vector<int8_t> output(length, 0);
  GeluQuantArg quant_arg = {};
  quant_arg.in_args_.scale_ = 1.0f;
  quant_arg.in_args_.zp_ = 0;
  quant_arg.out_args_.scale_ = 1.0f;
  quant_arg.out_args_.zp_ = 0;
  int8_t table[256] = {0};
  GeluInt8InitLUT(&quant_arg, table, true);
  GeluInt8(input.data(), length, output.data(), table);

  std::cout << "GeluInt8Test-GeluInt8_tanh_simple output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\n GeluInt8Test-GeluInt8_tanh_simple benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: GeluInt8 (erf LUT) with a realistic asymmetric quantiser spanning the GELU
// transition region — the discriminating case that actually traces the curve.
TEST_F(GeluInt8Test, GeluInt8_erf_realistic) {
  std::vector<int8_t> input = {-128, -105, -82, -59, -35, -12, 11, 34, 57, 81, 104, 127};
  std::vector<int8_t> benchmark = {-128, -128, -128, -128, -128, -128, -128, -86, -29, 26, 77, 127};
  const int length = 2 * 2 * 3;
  std::vector<int8_t> output(length, 0);
  GeluQuantArg quant_arg = {};
  quant_arg.in_args_.scale_ = 0.04313725605607032776f;
  quant_arg.in_args_.zp_ = 11;
  quant_arg.out_args_.scale_ = 0.01960784383118152618f;
  quant_arg.out_args_.zp_ = -128;
  int8_t table[256] = {0};
  GeluInt8InitLUT(&quant_arg, table, false);
  GeluInt8(input.data(), length, output.data(), table);

  std::cout << "GeluInt8Test-GeluInt8_erf_realistic output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\n GeluInt8Test-GeluInt8_erf_realistic benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase4: GeluInt8 (tanh LUT) with the same realistic quantiser as Testcase3.
TEST_F(GeluInt8Test, GeluInt8_tanh_realistic) {
  std::vector<int8_t> input = {-128, -105, -82, -59, -35, -12, 11, 34, 57, 81, 104, 127};
  std::vector<int8_t> benchmark = {-128, -128, -128, -128, -128, -128, -128, -86, -29, 26, 77, 127};
  const int length = 2 * 2 * 3;
  std::vector<int8_t> output(length, 0);
  GeluQuantArg quant_arg = {};
  quant_arg.in_args_.scale_ = 0.04313725605607032776f;
  quant_arg.in_args_.zp_ = 11;
  quant_arg.out_args_.scale_ = 0.01960784383118152618f;
  quant_arg.out_args_.zp_ = -128;
  int8_t table[256] = {0};
  GeluInt8InitLUT(&quant_arg, table, true);
  GeluInt8(input.data(), length, output.data(), table);

  std::cout << "GeluInt8Test-GeluInt8_tanh_realistic output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\n GeluInt8Test-GeluInt8_tanh_realistic benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase5: GeluInt8 (erf LUT) with a mid-range symmetric quantiser (scale=0.1, zp=0)
// to sample the GELU curve at a different resolution.
TEST_F(GeluInt8Test, GeluInt8_erf_mid) {
  std::vector<int8_t> input = {0, 10, -10, 50, -50, -100, 100, 127};
  std::vector<int8_t> benchmark = {0, 8, -2, 50, 0, 0, 100, 127};
  const int length = 2 * 2 * 2;
  std::vector<int8_t> output(length, 0);
  GeluQuantArg quant_arg = {};
  quant_arg.in_args_.scale_ = 0.1f;
  quant_arg.in_args_.zp_ = 0;
  quant_arg.out_args_.scale_ = 0.1f;
  quant_arg.out_args_.zp_ = 0;
  int8_t table[256] = {0};
  GeluInt8InitLUT(&quant_arg, table, false);
  GeluInt8(input.data(), length, output.data(), table);

  std::cout << "GeluInt8Test-GeluInt8_erf_mid output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\n GeluInt8Test-GeluInt8_erf_mid benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
