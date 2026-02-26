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
#include "gtest/gtest.h"
#include "nnacl_c/fp32/lstm_fp32.h"

namespace mindspore {
class LstmFp32Test : public ::testing::Test {
 public:
  LstmFp32Test() {}
};

static float get_cosine_similarity(const float *arr1, const float *arr2, size_t size) {
  if (arr1 == nullptr || arr2 == nullptr || size == 0) {
    return 0.0;
  }
  float dot_product = 0.0f;
  float norm1 = 0.0f;
  float norm2 = 0.0f;
  for (size_t i = 0; i < size; i++) {
    dot_product += arr1[i] * arr2[i];
    norm1 += arr1[i] * arr1[i];
    norm2 += arr2[i] * arr2[i];
  }
  norm1 = std::sqrt(norm1);
  norm2 = std::sqrt(norm2);
  if (norm1 == 0 && norm2 == 0) {
    return 1.0;
  }
  float cosine_similarity = dot_product / (norm1 * norm2);
  return cosine_similarity;
}

TEST_F(LstmFp32Test, Testcase01) {
  const int batch = 2;
  const int col = 4;
  const int col_align = 4;
  bool is_bidirectional = true;
  std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  const int output_shape = static_cast<int>(input.size());
  std::vector<float> benchmark = {2, 2, 2, 2, 1, 2, 3, 4, 9, 10, 11, 12, 2, 2, 2, 2};
  std::vector<float> outputs;
  outputs.resize(output_shape, 2);
  std::vector<int32_t> order = {1, 2, 3};
  PackLstmBias(outputs.data(), input.data(), batch, col, col_align, is_bidirectional, order.data());
  for (auto iter = outputs.begin(); iter != outputs.end(); iter++) {
    printf("[%f]", *iter);
  }
  float similarity = get_cosine_similarity(outputs.data(), benchmark.data(), 16);
  ASSERT_GT(similarity, 0.99);
}

}  // namespace mindspore
