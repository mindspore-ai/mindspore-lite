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
#include "nnacl_c/arithmetic_parameter.h"
#include "nnacl_c/int8/arithmetic_int8.h"

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

// input1:1*2*4*1 ,input2:1*2*1*4
TEST_F(MulInt8Test, Broadcasting) {
  std::vector<int8_t> input0 = {-13, -111, -5, -21, -111, -128, -81, 33};
  std::vector<int8_t> input1 = {104, 83, 97, 35, 127, 90, 108, 70};
  const int out_shape = 1 * 2 * 4 * 4;
  std::vector<int8_t> output;
  output.resize(out_shape, 0);

  std::vector<float> output_fp32;
  output_fp32.resize(out_shape, 0);

  std::vector<int8_t> tile_input1;
  tile_input1.resize(out_shape, 0);
  std::vector<int8_t> tile_input2;
  tile_input2.resize(out_shape, 0);
  std::vector<int8_t> input0_shape = {1, 2, 4, 1};
  std::vector<int8_t> input1_shape = {1, 2, 1, 4};
  std::vector<int8_t> benchmark = {42,   31,  38,  5,   -96,  -92,  -95,  -83,  53,  41,  49,  12,  31,  21, 27,  -3,
                                   -101, -94, -97, -90, -127, -116, -121, -110, -53, -54, -54, -55, 127, 94, 110, 77};

  const MulQuantArg quant = {{0.01488248724490404129, 86},
                             {0.01488248724490404129, 86},
                             {0.02984090149402618408, -61},
                             2040229084,
                             -128,
                             127,
                             0,
                             7};
  ArithmeticParameter tile_para = {
    {"", 0, 1, 119},
    true,
    4,
    -1299769064,
    {1, 2, 4, 1, 32675, -242294839, 22010, -242294839, 22010, -242294839},
    94536282857416,
    {1, 2, 1, 4, -1367742912, 32675, 16, 0, -242294840, 22010},
    140340983568336,
    {1, 2, 4, 4, 4098, 0, 0, 22010},
    0,
    {},
    {},
    {},
    {0, 0, 0, 0, 0, 8, 0, -1053270544, 32765, -1367742912},
    {32675, 0, 0, -242351840, 22010, -1053270688, 32765, -1367744416, 32675, -1367744528}};
  if (input0_shape != input1_shape) {
    TileDimensionsInt8(input0.data(), input1.data(), tile_input1.data(), tile_input2.data(), &tile_para);
  }
  Mul(tile_input1.data(), tile_input2.data(), output.data(), 32, &quant);
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
