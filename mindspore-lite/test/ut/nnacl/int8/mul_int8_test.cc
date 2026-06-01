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
TEST_F(MulInt8Test, Broadcasting1) {
  std::vector<int8_t> input0 = {-13, -111, -5, -21, -111, -128, -81, 33};
  std::vector<int8_t> input1 = {104, 83, 97, 35, 127, 90, 108, 70};
  const int out_shape = 1 * 2 * 4 * 4;
  std::vector<int8_t> output;
  output.resize(out_shape, 0);

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

// input1:2*1*1*5 ,input2:2*3*4*5
TEST_F(MulInt8Test, Broadcasting2) {
  std::vector<int8_t> input0 = {-47, -19, 25, 11, -48, 42, -20, -1, 12, -113};
  std::vector<int8_t> input1 = {
    -39, 94,  -67, -16, 50,  -33, -84,  -11,  -30, -7,  -41, 21,  82,  -120, -49,  27,   -97, -7,  1,   -10,
    -6,  127, -3,  -29, -41, 93,  13,   -31,  99,  58,  27,  45,  -33, 51,   -23,  -10,  -4,  31,  22,  -34,
    25,  16,  71,  -42, -25, 53,  -112, -12,  -26, 24,  -3,  -24, -53, -16,  -57,  79,   42,  -12, 40,  10,
    -34, -6,  16,  19,  20,  53,  16,   -116, -7,  -54, -62, 34,  -87, -33,  2,    -124, 1,   58,  22,  -36,
    -89, 36,  62,  20,  -61, -6,  -41,  73,   -63, -51, -26, 36,  -3,  -1,   -107, 60,   9,   -9,  -68, -103,
    -50, -17, 121, -15, -3,  13,  -9,   2,    29,  -98, 6,   52,  42,  -128, 19,   -87,  -2,  13,  -34, -25};
  const int out_shape = 2 * 3 * 4 * 5;
  std::vector<int8_t> output;
  output.resize(out_shape, 0);
  std::vector<int8_t> tile_input1;
  tile_input1.resize(out_shape, 0);
  std::vector<int8_t> tile_input2;
  tile_input2.resize(out_shape, 0);
  std::vector<int8_t> input0_shape = {2, 1, 1, 5};
  std::vector<int8_t> input1_shape = {2, 3, 4, 5};
  std::vector<int8_t> benchmark = {
    -18,  -55, -68, -39, -73, -22, -24, -39, -44, -38, -17, -42, 10,  -70, -11, -58,  -22, -36, -34, -36,
    -38,  -60, -34, -43, -16, -99, -41, -49, -5,  -78, -58, -46, -50, -19, -28, -36,  -38, -17, -28, -21,
    -57,  -41, 4,   -47, -26, -74, -19, -39, -42, -57, -40, -34, -60, -39, -6,  -90,  -46, -39, -23, -48,
    -57,  -37, -34, -29, -83, 11,  -42, -49, -37, 39,  -79, -45, -46, -45, -53, -128, -39, -30, -28, 9,
    -100, -45, -29, -28, 50,  -35, -31, -28, -54, 34,  -51, -45, -36, -35, 126, 16,   -40, -37, -56, 119,
    -70,  -35, -23, -39, -45, -21, -37, -36, -25, 111, -26, -48, -32, -75, -81, -99,  -38, -35, -45, -9};
  const MulQuantArg quant = {{0.01948853582143783569, 8},
                             {0.01948853582143783569, 8},
                             {0.02424382604658603668, -37},
                             1076557196,
                             -128,
                             127,
                             0,
                             5};
  ArithmeticParameter tile_para = {{"", 0, 1, 22037},
                                   true,
                                   4,
                                   -120261208,
                                   {2, 1, 1, 5, 32686, -1622385335, 22037, -1622385335, 22037, -1622385335},
                                   94650866883912,
                                   {2, 3, 4, 5, -199825856, 32686, 16, 1066661069, -1622385336, 22037},
                                   140389396125648,
                                   {2, 3, 4, 5, 4098, 0, 0, 22037},
                                   0,
                                   {},
                                   {},
                                   {},
                                   {0, 0, 0, 0, 0, 8, 32765, 873809520, 32765, -199825856},
                                   {32686, 0, 0, 873791776, 32765, 873809376, 32765, -199827360, 32686, -199827472}};
  if (input0_shape != input1_shape) {
    TileDimensionsInt8(input0.data(), input1.data(), tile_input1.data(), tile_input2.data(), &tile_para);
  }
  Mul(tile_input1.data(), tile_input2.data(), output.data(), out_shape, &quant);
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// input1:2*3*4*5 ,input2:2*1*1*5
TEST_F(MulInt8Test, Broadcasting3) {
  std::vector<int8_t> input0 = {
    -7,  56,  28,  -59, 4,   38,  -34, -2,  35,   -5,  127, 12,  -78, 83,  -128, 5,   59,  21,  76,  5,
    -6,  -26, 92,  20,  -58, 21,  46,  -59, -109, -66, -99, 50,  -80, 29,  -60,  -1,  50,  -30, -25, -50,
    20,  53,  -45, -12, -29, -39, -32, -68, -74,  -74, -5,  -39, -46, 24,  -42,  7,   34,  -47, -62, -7,
    -4,  94,  -61, 2,   -63, 13,  -30, -15, 11,   -30, 64,  28,  36,  -58, -35,  -57, -45, 18,  28,  -63,
    -38, -81, -58, -7,  -65, -95, 55,  1,   5,    -28, 43,  -49, -18, 25,  -9,   -65, -22, 69,  18,  -12,
    -4,  14,  -51, -21, -22, -10, 20,  12,  8,    -50, 34,  36,  -32, -74, -17,  -19, -14, -26, 8,   11};
  std::vector<int8_t> input1 = {40, -48, -8, -14, -95, 51, -72, 27, -72, 13};
  const int out_shape = 2 * 3 * 4 * 5;
  std::vector<int8_t> output;
  output.resize(out_shape, 0);
  std::vector<int8_t> tile_input1;
  tile_input1.resize(out_shape, 0);
  std::vector<int8_t> tile_input2;
  tile_input2.resize(out_shape, 0);
  std::vector<int8_t> input0_shape = {2, 3, 4, 5};
  std::vector<int8_t> input1_shape = {2, 1, 1, 5};
  std::vector<int8_t> benchmark = {
    -21, -65,  -23, -25, -49, 18,   -14, -25, -27, -37, 94,   -40, -31, -28, 127, -11, -67, -23, -27, -50,
    -20, -19,  -17, -27, 34,  3,    -59, -30, -24, 44,  -100, -62, -31, -27, 36,  -16, -62, -27, -26, 23,
    2,   -63,  -29, -26, -5,  -48,  -15, -30, -25, 55,  -19,  -11, -29, -27, 12,  -9,  -53, -29, -25, -34,
    -17, -128, -57, -40, -47, 1,    -10, -27, -49, -33, 54,   -65, 6,   17,  -35, -72, 5,   -6,  -65, -47,
    -52, 39,   -55, -32, -48, -111, -91, -17, -43, -32, 32,   8,   -29, -62, -24, -80, -17, 27,  -56, -26,
    -17, -52,  -51, -18, -30, -23,  -58, -10, -46, -42, 23,   -73, -38, 32,  -28, -32, -25, -34, -46, -16};
  const MulQuantArg quant = {{0.02049582637846469879, 13},
                             {0.02049582637846469879, 13},
                             {0.02596961893141269684, -26},
                             1111591287,
                             -128,
                             127,
                             0,
                             5};
  ArithmeticParameter tile_para = {{"", 0, 1, 22020},
                                   true,
                                   4,
                                   434488744,
                                   {2, 3, 4, 5, 32545, 209972553, 22020, 209972553, 22020, 209972553},
                                   94575389830472,
                                   {2, 1, 1, 5, 354924096, 32545, 16, 1066728667, 209972552, 22020},
                                   139780065519568,
                                   {2, 3, 4, 5, 4098, 0, 0, 22020},
                                   0,
                                   {},
                                   {},
                                   {},
                                   {0, 0, 0, 0, 0, 8, 32765, -1435457152, 32765, 354924096},
                                   {32545, 0, 0, -1435500256, 32765, -1435457296, 32765, 354922592, 32545, 354922480}};
  if (input0_shape != input1_shape) {
    TileDimensionsInt8(input0.data(), input1.data(), tile_input1.data(), tile_input2.data(), &tile_para);
  }
  Mul(tile_input1.data(), tile_input2.data(), output.data(), out_shape, &quant);
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

}  // namespace mindspore
