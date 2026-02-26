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
#include "nnacl_c/base/gather_base.h"
#include "nnacl_c/int8/gather_int8.h"

namespace mindspore {
class GatherInt8Test : public ::testing::Test {
 public:
  GatherInt8Test() {}
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

TEST_F(GatherInt8Test, Testcase01) {
  std::vector<int> index_data = {0, 1, 2, 3, 4, 5, 10, 11, 12, 13};
  std::vector<int8_t> input = {19,  47, 18,  11,  41, 10,  23,  19, 20, 20, 12, 24, 24,   14, 5, 10, 35, 16, 50,  0,
                               119, 47, 118, 111, 41, -10, -25, 0,  1,  3,  7,  88, -122, 0,  5, 6,  90, 10, 122, -33};
  std::vector<int> input_shape = {2, 20};
  std::vector<int8_t> outputs(2 * 1 * 10, 0);
  int64_t output_size = 2;
  int64_t byte_inner_size = sizeof(int8_t);
  int64_t limit = input_shape[1];
  int64_t index_num = index_data.size();
  const GatherQuantArg param = {1, 0, 0};
  std::vector<int8_t> benchmark = {19, 47, 18, 11, 41, 10, 12, 24, 24, 14, 119, 47, 118, 111, 41, -10, 7, 88, -122, 0};
  GatherInt8Int32Index(input.data(), outputs.data(), output_size, byte_inner_size, limit, index_data.data(), index_num,
                       param);
  for (auto iter = outputs.begin(); iter != outputs.end(); iter++) {
    printf("[%d]", (int)(*iter));
  }
  float sim = get_cosine_similarity_int8(outputs.data(), benchmark.data(), 2 * 10);
  ASSERT_GT(sim, 0.9);
}
}  // namespace mindspore
