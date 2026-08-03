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
#include "nnacl_c/base/unstack_base.h"

namespace mindspore {
class UnstackInt8Test : public ::testing::Test {
 public:
  UnstackInt8Test() {}
};

float get_cosine_similarity_int8(const int8_t *arr1, const int8_t *arr2, size_t cmp_size);
static float accuracy_threshold = 0.99;

// Stack/Unstack are data-movement ops: the int8 path reuses the same byte-copy
// `Unstack()` as fp32 (no quantisation math), so these cases mirror unstack_fp32_test
// with int8 elements (incl. negative values to exercise the signed int8 range).

// Testcase1: input [2, 3] (int8) unstack along axis=0 -> 2 outputs of [3]
TEST_F(UnstackInt8Test, Unstack_2x3_Axis0) {
  std::vector<int8_t> input = {1, -2, 3, 4, -5, 6};
  std::vector<int8_t> benchmark0 = {1, -2, 3};
  std::vector<int8_t> benchmark1 = {4, -5, 6};
  const int out_size = 3;
  std::vector<int8_t> output0(out_size, 0);
  std::vector<int8_t> output1(out_size, 0);
  void *outputs[2] = {output0.data(), output1.data()};

  UnstackParameter para = {};
  para.num_ = 2;
  para.pre_dims_ = 1;
  para.axis_dim_ = 2;
  para.after_dims_ = 3;
  Unstack(input.data(), outputs, &para, sizeof(int8_t));

  std::cout << "UnstackInt8Test-Unstack_2x3_Axis0 output0:\n";
  std::for_each(output0.begin(), output0.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nUnstackInt8Test-Unstack_2x3_Axis0 output1:\n";
  std::for_each(output1.begin(), output1.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity0 = get_cosine_similarity_int8(output0.data(), benchmark0.data(), output0.size());
  float similarity1 = get_cosine_similarity_int8(output1.data(), benchmark1.data(), output1.size());
  ASSERT_GT(similarity0, accuracy_threshold);
  ASSERT_GT(similarity1, accuracy_threshold);
}

// Testcase2: input [3, 2] (int8) unstack along axis=0 -> 3 outputs of [2]
TEST_F(UnstackInt8Test, Unstack_3x2_Axis0) {
  std::vector<int8_t> input = {1, 2, 3, 4, 5, 6};
  std::vector<int8_t> benchmark0 = {1, 2};
  std::vector<int8_t> benchmark1 = {3, 4};
  std::vector<int8_t> benchmark2 = {5, 6};
  const int out_size = 2;
  std::vector<int8_t> output0(out_size, 0);
  std::vector<int8_t> output1(out_size, 0);
  std::vector<int8_t> output2(out_size, 0);
  void *outputs[3] = {output0.data(), output1.data(), output2.data()};

  UnstackParameter para = {};
  para.num_ = 3;
  para.pre_dims_ = 1;
  para.axis_dim_ = 3;
  para.after_dims_ = 2;
  Unstack(input.data(), outputs, &para, sizeof(int8_t));

  std::cout << "UnstackInt8Test-Unstack_3x2_Axis0 output0:\n";
  std::for_each(output0.begin(), output0.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nUnstackInt8Test-Unstack_3x2_Axis0 output1:\n";
  std::for_each(output1.begin(), output1.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nUnstackInt8Test-Unstack_3x2_Axis0 output2:\n";
  std::for_each(output2.begin(), output2.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity0 = get_cosine_similarity_int8(output0.data(), benchmark0.data(), output0.size());
  float similarity1 = get_cosine_similarity_int8(output1.data(), benchmark1.data(), output1.size());
  float similarity2 = get_cosine_similarity_int8(output2.data(), benchmark2.data(), output2.size());
  ASSERT_GT(similarity0, accuracy_threshold);
  ASSERT_GT(similarity1, accuracy_threshold);
  ASSERT_GT(similarity2, accuracy_threshold);
}

// Testcase3: input [2, 2, 3] (int8) unstack along axis=1 -> 2 outputs of [2, 3]
TEST_F(UnstackInt8Test, Unstack_2x2x3_Axis1) {
  std::vector<int8_t> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int8_t> benchmark0 = {1, 2, 3, 7, 8, 9};
  std::vector<int8_t> benchmark1 = {4, 5, 6, 10, 11, 12};
  const int out_size = 2 * 3;
  std::vector<int8_t> output0(out_size, 0);
  std::vector<int8_t> output1(out_size, 0);
  void *outputs[2] = {output0.data(), output1.data()};

  UnstackParameter para = {};
  para.num_ = 2;
  para.pre_dims_ = 2;
  para.axis_dim_ = 2;
  para.after_dims_ = 3;
  Unstack(input.data(), outputs, &para, sizeof(int8_t));

  std::cout << "UnstackInt8Test-Unstack_2x2x3_Axis1 output0:\n";
  std::for_each(output0.begin(), output0.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nUnstackInt8Test-Unstack_2x2x3_Axis1 output1:\n";
  std::for_each(output1.begin(), output1.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity0 = get_cosine_similarity_int8(output0.data(), benchmark0.data(), output0.size());
  float similarity1 = get_cosine_similarity_int8(output1.data(), benchmark1.data(), output1.size());
  ASSERT_GT(similarity0, accuracy_threshold);
  ASSERT_GT(similarity1, accuracy_threshold);
}
}  // namespace mindspore
