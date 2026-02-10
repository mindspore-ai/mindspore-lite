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
#include "nnacl_c/mul_parameter.h"
#include "nnacl_c/int8/mul_int8.h"

namespace mindspore {
class MulInt8Test : public ::testing::Test {
 public:
  MulInt8Test() {}
};

float get_cosine_similarity_int8(const int8_t *arr1, const int8_t *arr2, size_t cmp_size) {
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

float accuracy_threshold = 0.99;

TEST_F(MulInt8Test, Mul) {
  std::vector<int8_t> input0 = {-19, -55, 127, -128};
  std::vector<int8_t> input1 = {-128, 127, -19, -55};
  std::vector<int8_t> benchmark = {-128, 4, 70, -128};
  int64_t real_dst_count = 4;
  const MulQuantArg quant_arg = {
    {0.0274509806, 128}, {0.0274509806, 128}, {0.105882354, -128}, 1956284316, -128, 127, 0, 7};
  const int length = static_cast<int>(input0.size());
  std::vector<int8_t> output(length);
  Mul(input0.data(), input1.data(), output.data(), real_dst_count, &quant_arg);
  std::cout << "MulInt8Test output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\n MulInt8Test benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

}  // namespace mindspore
