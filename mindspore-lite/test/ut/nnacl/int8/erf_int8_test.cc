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
#include "nnacl_c/int8/arithmetic_self_int8.h"

namespace mindspore {
class ErfInt8Test : public ::testing::Test {
 public:
  ErfInt8Test() {}
};

float get_cosine_similarity_int8(const int8_t *arr1, const int8_t *arr2, size_t cmp_size);
static float accuracy_threshold = 0.99;

// Testcase1: ErfInt8 with scale=1.0, zp=0
TEST_F(ErfInt8Test, Erf_2D) {
  std::vector<int8_t> input = {0, 1, -1, 2, -2, 50, -50, 127};
  std::vector<int8_t> benchmark = {0, 1, -1, 1, -1, 1, -1, 1};
  const int length = 2 * 4;
  std::vector<int8_t> output(length, 0);

  ArithSelfQuantArg para = {};
  para.in_args_.scale_ = 1.0f;
  para.in_args_.zp_ = 0;
  para.out_args_.scale_ = 1.0f;
  para.out_args_.zp_ = 0;
  para.output_activation_min_ = -128;
  para.output_activation_max_ = 127;

  Int8ElementErf(input.data(), output.data(), length, para);
  std::cout << "ErfInt8Test Erf_2D output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nErfInt8Test Erf_2D benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: ErfInt8 with scale=0.1, zp=0
TEST_F(ErfInt8Test, Erf_3D) {
  std::vector<int8_t> input = {0, 10, -10, 50, -50, -100, 100, 127};
  std::vector<int8_t> benchmark = {0, 8, -8, 10, -10, -10, 10, 10};
  const int length = 2 * 2 * 2;
  std::vector<int8_t> output(length, 0);

  ArithSelfQuantArg para = {};
  para.in_args_.scale_ = 0.1f;
  para.in_args_.zp_ = 0;
  para.out_args_.scale_ = 0.1f;
  para.out_args_.zp_ = 0;
  para.output_activation_min_ = -128;
  para.output_activation_max_ = 127;

  Int8ElementErf(input.data(), output.data(), length, para);
  std::cout << "ErfInt8Test Erf_3D output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nErfInt8Test Erf_3D benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: ErfInt8 with more values
TEST_F(ErfInt8Test, Erf_4D) {
  std::vector<int8_t> input = {106, 95, -77, -94, 102, -58, -46, 80,  26, 21,  -60, -4,
                               56,  50, 54,  59,  76,  -47, -11, 127, 32, -50, 60,  -4};
  std::vector<int8_t> benchmark = {114, 106, -64, -85, 111, -41, -27, 96,  49, 45,  -44, 19,
                                   76,  71,  75,  79,  92,  -29, 12,  127, 55, -32, 80,  19};
  const int length = 2 * 2 * 3 * 2;
  std::vector<int8_t> output(length, 0);

  ArithSelfQuantArg para = {};
  para.in_args_.scale_ = 0.00343784f;
  para.in_args_.zp_ = 128;
  para.out_args_.scale_ = 0.0030782f;
  para.out_args_.zp_ = -128;
  para.output_activation_min_ = -128;
  para.output_activation_max_ = 127;

  Int8ElementErf(input.data(), output.data(), length, para);
  std::cout << "ErfInt8Test Testcase03 output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nErfInt8Test Testcase03 benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
