/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "gtest/gtest.h"
#include "nnacl_c/int8/softmax_int8.h"

namespace mindspore {
class SoftmaxInt8Test : public ::testing::Test {
 public:
  SoftmaxInt8Test() {}
};

static float get_cosine_similarity_int8(const int8_t *arr1, const int8_t *arr2, size_t cmp_size) {
  if (arr1 == nullptr || arr2 == nullptr || cmp_size == 0) {
    return 0.0f;
  }
  float dot_product = 0.0f;
  float norm1 = 0.0f;
  float norm2 = 0.0f;
  for (size_t i = 0; i < cmp_size; ++i) {
    dot_product += (float)arr1[i] * (float)arr2[i];
    norm1 += (float)arr1[i] * (float)arr1[i];
    norm2 += (float)arr2[i] * (float)arr2[i];
  }
  norm1 = std::sqrt(norm1);
  norm2 = std::sqrt(norm2);
  float norms_product = norm1 * norm2;
  const float FLOAT_EPS = 1e-6f;
  if (std::fabs(norms_product) < FLOAT_EPS) {
    return 0.0f;
  }
  float cosine_similarity = dot_product / norms_product;
  return cosine_similarity;
}

TEST_F(SoftmaxInt8Test, Testcase01) {
  const int input_size = 4;
  const int count = 1;
  const int n_dim = 2;
  std::vector<int32_t> input_shape = {1, 4, 0, 0, 0};
  std::vector<int32_t> exp_data(input_size, 0);
  std::vector<int32_t> sum_data(input_size, 0);
  std::vector<int8_t> input_data = {-1, 1, 1, -1};
  std::vector<int8_t> benchmark_data = {58, 70, 70, 58};
  std::vector<int8_t> outputs(input_size, 0);
  int m0_thread_num = 1;
  SoftmaxQuantArg quant_params = {{0.0, -8}, {0.0, -128}, 0, 127, 1700000000, 23, 17};
  const SoftmaxParameter softmax_parameter = {{"", 138, m0_thread_num, 0}, 1};
  SoftmaxInt8(input_data.data(), outputs.data(), count, exp_data.data(), sum_data.data(), input_shape.data(), n_dim,
              softmax_parameter.axis_, &quant_params);
  for (auto iter = outputs.begin(); iter != outputs.end(); iter++) {
    printf("[%d]", (int)(*iter));
  }
  float sim = get_cosine_similarity_int8(outputs.data(), benchmark_data.data(), 4);
  ASSERT_GT(sim, 0.9);
}
}  // namespace mindspore
