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
class UnstackFp32Test : public ::testing::Test {
 public:
  UnstackFp32Test() {}
};

float get_cosine_similarity(const float *arr1, const float *arr2, size_t cmp_size);
static float accuracy_threshold = 0.99;

// Testcase1: input [2, 3] unstack along axis=0 -> 2 outputs of [3]
TEST_F(UnstackFp32Test, Unstack_2x3_Axis0) {
  // input shape [2, 3], axis=0, pre_dims=1, axis_dim=2, after_dims=3, num=2
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> benchmark0 = {1.0f, 2.0f, 3.0f};
  std::vector<float> benchmark1 = {4.0f, 5.0f, 6.0f};
  const int out_size = 3;
  std::vector<float> output0(out_size, 0.0f);
  std::vector<float> output1(out_size, 0.0f);
  void *outputs[2] = {output0.data(), output1.data()};

  UnstackParameter para = {};
  para.num_ = 2;
  para.pre_dims_ = 1;
  para.axis_dim_ = 2;
  para.after_dims_ = 3;
  Unstack(input.data(), outputs, &para, sizeof(float));

  std::cout << "UnstackFp32Test-Unstack_2x3_Axis0 output0:\n";
  std::for_each(output0.begin(), output0.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nUnstackFp32Test-Unstack_2x3_Axis0 output1:\n";
  std::for_each(output1.begin(), output1.end(), [](float value) { std::cout << value << ", "; });
  float similarity0 = get_cosine_similarity(output0.data(), benchmark0.data(), output0.size());
  float similarity1 = get_cosine_similarity(output1.data(), benchmark1.data(), output1.size());
  ASSERT_GT(similarity0, accuracy_threshold);
  ASSERT_GT(similarity1, accuracy_threshold);
}

// Testcase2: input [3, 2] unstack along axis=0 -> 3 outputs of [2]
TEST_F(UnstackFp32Test, Unstack_3x2_Axis0) {
  // input shape [3, 2], axis=0, pre_dims=1, axis_dim=3, after_dims=2, num=3
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> benchmark0 = {1.0f, 2.0f};
  std::vector<float> benchmark1 = {3.0f, 4.0f};
  std::vector<float> benchmark2 = {5.0f, 6.0f};
  const int out_size = 2;
  std::vector<float> output0(out_size, 0.0f);
  std::vector<float> output1(out_size, 0.0f);
  std::vector<float> output2(out_size, 0.0f);
  void *outputs[3] = {output0.data(), output1.data(), output2.data()};

  UnstackParameter para = {};
  para.num_ = 3;
  para.pre_dims_ = 1;
  para.axis_dim_ = 3;
  para.after_dims_ = 2;
  Unstack(input.data(), outputs, &para, sizeof(float));

  std::cout << "UnstackFp32Test-Unstack_3x2_Axis0 output0:\n";
  std::for_each(output0.begin(), output0.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nUnstackFp32Test-Unstack_3x2_Axis0 output1:\n";
  std::for_each(output1.begin(), output1.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nUnstackFp32Test-Unstack_3x2_Axis0 output2:\n";
  std::for_each(output2.begin(), output2.end(), [](float value) { std::cout << value << ", "; });
  float similarity0 = get_cosine_similarity(output0.data(), benchmark0.data(), output0.size());
  float similarity1 = get_cosine_similarity(output1.data(), benchmark1.data(), output1.size());
  float similarity2 = get_cosine_similarity(output2.data(), benchmark2.data(), output2.size());
  ASSERT_GT(similarity0, accuracy_threshold);
  ASSERT_GT(similarity1, accuracy_threshold);
  ASSERT_GT(similarity2, accuracy_threshold);
}

// Testcase3: input [2, 2, 3] unstack along axis=1 -> 2 outputs of [2, 3]
TEST_F(UnstackFp32Test, Unstack_2x2x3_Axis1) {
  // input shape [2, 2, 3], axis=1, pre_dims=2, axis_dim=2, after_dims=3, num=2
  // layout: [[1,2,3, 4,5,6], [7,8,9, 10,11,12]]
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  // output[0]: j=0, i=0 -> offset=0*2*3+0*3=0 -> [1,2,3]
  //            j=0, i=1 -> offset=1*2*3+0*3=6 -> [7,8,9]
  // output[1]: j=1, i=0 -> offset=0*2*3+1*3=3 -> [4,5,6]
  //            j=1, i=1 -> offset=1*2*3+1*3=9 -> [10,11,12]
  std::vector<float> benchmark0 = {1.0f, 2.0f, 3.0f, 7.0f, 8.0f, 9.0f};
  std::vector<float> benchmark1 = {4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f};
  const int out_size = 2 * 3;
  std::vector<float> output0(out_size, 0.0f);
  std::vector<float> output1(out_size, 0.0f);
  void *outputs[2] = {output0.data(), output1.data()};

  UnstackParameter para = {};
  para.num_ = 2;
  para.pre_dims_ = 2;
  para.axis_dim_ = 2;
  para.after_dims_ = 3;
  Unstack(input.data(), outputs, &para, sizeof(float));

  std::cout << "UnstackFp32Test-Unstack_2x2x3_Axis1 output0:\n";
  std::for_each(output0.begin(), output0.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nUnstackFp32Test-Unstack_2x2x3_Axis1 output1:\n";
  std::for_each(output1.begin(), output1.end(), [](float value) { std::cout << value << ", "; });
  float similarity0 = get_cosine_similarity(output0.data(), benchmark0.data(), output0.size());
  float similarity1 = get_cosine_similarity(output1.data(), benchmark1.data(), output1.size());
  ASSERT_GT(similarity0, accuracy_threshold);
  ASSERT_GT(similarity1, accuracy_threshold);
}

TEST_F(UnstackFp32Test, Unstack_1x2x3x2_Axis1) {
  // input shape [1, 2, 3, 2], axis=1, pre_dims=1, axis_dim=2, after_dims=6, num=2
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  std::vector<float> benchmark0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> benchmark1 = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  const int out_size = 6;
  std::vector<float> output0(out_size, 0.0f);
  std::vector<float> output1(out_size, 0.0f);
  void *outputs[2] = {output0.data(), output1.data()};

  UnstackParameter para = {};
  para.num_ = 2;
  para.pre_dims_ = 1;
  para.axis_dim_ = 2;
  para.after_dims_ = 6;
  Unstack(input.data(), outputs, &para, sizeof(float));

  float similarity0 = get_cosine_similarity(output0.data(), benchmark0.data(), output0.size());
  float similarity1 = get_cosine_similarity(output1.data(), benchmark1.data(), output1.size());
  ASSERT_GT(similarity0, accuracy_threshold);
  ASSERT_GT(similarity1, accuracy_threshold);
}

TEST_F(UnstackFp32Test, Unstack_2x4x3_Axis2) {
  // input shape [2, 4, 3], axis=2, pre_dims=8, axis_dim=3, after_dims=1, num=3
  std::vector<float> input = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                              13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f};
  std::vector<float> benchmark0 = {1.0f, 4.0f, 7.0f, 10.0f, 13.0f, 16.0f, 19.0f, 22.0f};
  std::vector<float> benchmark1 = {2.0f, 5.0f, 8.0f, 11.0f, 14.0f, 17.0f, 20.0f, 23.0f};
  std::vector<float> benchmark2 = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f, 21.0f, 24.0f};
  const int out_size = 8;
  std::vector<float> output0(out_size, 0.0f);
  std::vector<float> output1(out_size, 0.0f);
  std::vector<float> output2(out_size, 0.0f);
  void *outputs[3] = {output0.data(), output1.data(), output2.data()};

  UnstackParameter para = {};
  para.num_ = 3;
  para.pre_dims_ = 8;
  para.axis_dim_ = 3;
  para.after_dims_ = 1;
  Unstack(input.data(), outputs, &para, sizeof(float));

  float similarity0 = get_cosine_similarity(output0.data(), benchmark0.data(), output0.size());
  float similarity1 = get_cosine_similarity(output1.data(), benchmark1.data(), output1.size());
  float similarity2 = get_cosine_similarity(output2.data(), benchmark2.data(), output2.size());
  ASSERT_GT(similarity0, accuracy_threshold);
  ASSERT_GT(similarity1, accuracy_threshold);
  ASSERT_GT(similarity2, accuracy_threshold);
}

}  // namespace mindspore
