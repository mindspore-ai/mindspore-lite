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
#include <cmath>
#include <vector>
#include "gtest/gtest.h"
#include "nnacl_c/fp32/arithmetic_fp32.h"
#include "nnacl_c/fp32/add_fp32.h"
#include "nnacl_c/fp32/sub_fp32.h"
#include "nnacl_c/fp32/mul_fp32.h"
#include "nnacl_c/fp32/div_fp32.h"
#include "nnacl_c/arithmetic_parameter.h"

namespace mindspore {
class BroadcastFp32Test : public ::testing::Test {
 public:
  BroadcastFp32Test() {}
};

float get_cosine_similarity(const float *arr1, const float *arr2, size_t cmp_size);
static float accuracy_threshold = 0.99;

// Test case: Add FP32 broadcasting [1,2,4,1] + [1,2,1,4] -> [1,2,4,4]
// This tests dual-direction broadcasting where both inputs need to be expanded
TEST_F(BroadcastFp32Test, Add_Broadcasting_1241_plus_1214) {
  // Input data: [1,2,4,1] has 8 elements, [1,2,1,4] has 8 elements
  std::vector<float> input0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> input1 = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};

  const int out_shape = 1 * 2 * 4 * 4;  // 32 elements
  std::vector<float> output(out_shape, 0.0f);

  // Broadcasted buffers
  std::vector<float> tile_input0(out_shape, 0.0f);
  std::vector<float> tile_input1(out_shape, 0.0f);
  std::vector<float> benchmark = {
    11.0f, 21.0f, 31.0f, 41.0f,  // Row 0
    12.0f, 22.0f, 32.0f, 42.0f,  // Row 1
    13.0f, 23.0f, 33.0f, 43.0f,  // Row 2
    14.0f, 24.0f, 34.0f, 44.0f,  // Row 3
    55.0f, 65.0f, 75.0f, 85.0f,  // Row 4
    56.0f, 66.0f, 76.0f, 86.0f,  // Row 5
    57.0f, 67.0f, 77.0f, 87.0f,  // Row 6
    58.0f, 68.0f, 78.0f, 88.0f   // Row 7
  };

  // ArithmeticParameter matching the execution command provided
  ArithmeticParameter arithmetic_parameter = {
    {"", 5, 1, 0},   // op_parameter_
    true,            // broadcasting_
    4,               // ndim_
    0,               // activation_type_
    {1, 2, 4, 1},    // in_shape0_
    8,               // in_elements_num0_
    {1, 2, 1, 4},    // in_shape1_
    8,               // in_elements_num1_
    {1, 2, 4, 4},    // out_shape_
    32,              // out_elements_num_
    {8, 4, 1, 1},    // in_strides0_
    {8, 4, 4, 1},    // in_strides1_
    {32, 16, 4, 1},  // out_strides
    {1, 1, 1, 4},    // multiples0_
    {1, 1, 4, 1}     // multiples1_
  };
  TileDimensionsFp32(input0.data(), input1.data(), tile_input0.data(), tile_input1.data(), &arithmetic_parameter);
  ElementAdd(tile_input0.data(), tile_input1.data(), output.data(), out_shape);
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Test case: Sub FP32 broadcasting [1,2,4,1] - [1,2,1,4] -> [1,2,4,4]
TEST_F(BroadcastFp32Test, Sub_Broadcasting_1241_minus_1214) {
  std::vector<float> input0 = {50.0f, 60.0f, 70.0f, 80.0f, 90.0f, 100.0f, 110.0f, 120.0f};
  std::vector<float> input1 = {10.0f, 20.0f, 30.0f, 40.0f, 5.0f, 15.0f, 25.0f, 35.0f};

  const int out_shape = 1 * 2 * 4 * 4;  // 32 elements
  std::vector<float> output(out_shape, 0.0f);
  std::vector<float> tile_input0(out_shape, 0.0f);
  std::vector<float> tile_input1(out_shape, 0.0f);

  std::vector<float> benchmark = {40.0f, 30.0f, 20.0f,  10.0f, 50.0f, 40.0f, 30.0f,  20.0f,  60.0f, 50.0f, 40.0f,
                                  30.0f, 70.0f, 60.0f,  50.0f, 40.0f, 85.0f, 75.0f,  65.0f,  55.0f, 95.0f, 85.0f,
                                  75.0f, 65.0f, 105.0f, 95.0f, 85.0f, 75.0f, 115.0f, 105.0f, 95.0f, 85.0f};

  ArithmeticParameter arithmetic_parameter = {{"", 152, 1, 0},
                                              true,
                                              4,
                                              0,
                                              {1, 2, 4, 1},
                                              8,
                                              {1, 2, 1, 4},
                                              8,
                                              {1, 2, 4, 4},
                                              32,
                                              {8, 4, 1, 1},
                                              {8, 4, 4, 1},
                                              {32, 16, 4, 1},
                                              {1, 1, 1, 4},
                                              {1, 1, 4, 1}};

  TileDimensionsFp32(input0.data(), input1.data(), tile_input0.data(), tile_input1.data(), &arithmetic_parameter);
  ElementSub(tile_input0.data(), tile_input1.data(), output.data(), out_shape);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Test case: Mul FP32 broadcasting [1,2,4,1] * [1,2,1,4] -> [1,2,4,4]
TEST_F(BroadcastFp32Test, Mul_Broadcasting_1241_multiply_1214) {
  std::vector<float> input0 = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<float> input1 = {10.0f, 20.0f, 30.0f, 40.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  const int out_shape = 1 * 2 * 4 * 4;  // 32 elements
  std::vector<float> output(out_shape, 0.0f);
  std::vector<float> tile_input0(out_shape, 0.0f);
  std::vector<float> tile_input1(out_shape, 0.0f);

  std::vector<float> benchmark = {20.0f,  40.0f, 60.0f,  80.0f,  30.0f,  60.0f, 90.0f, 120.0f, 40.0f, 80.0f, 120.0f,
                                  160.0f, 50.0f, 100.0f, 150.0f, 200.0f, 12.0f, 24.0f, 36.0f,  48.0f, 14.0f, 28.0f,
                                  42.0f,  56.0f, 16.0f,  32.0f,  48.0f,  64.0f, 18.0f, 36.0f,  54.0f, 72.0f};

  ArithmeticParameter arithmetic_parameter = {{"", 99, 1, 0},
                                              true,
                                              4,
                                              0,
                                              {1, 2, 4, 1},
                                              8,
                                              {1, 2, 1, 4},
                                              8,
                                              {1, 2, 4, 4},
                                              32,
                                              {8, 4, 1, 1},
                                              {8, 4, 4, 1},
                                              {32, 16, 4, 1},
                                              {1, 1, 1, 4},
                                              {1, 1, 4, 1}};

  TileDimensionsFp32(input0.data(), input1.data(), tile_input0.data(), tile_input1.data(), &arithmetic_parameter);
  ElementMul(tile_input0.data(), tile_input1.data(), output.data(), out_shape);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Test case: Div FP32 broadcasting [1,2,4,1] / [1,2,1,4] -> [1,2,4,4]
TEST_F(BroadcastFp32Test, Div_Broadcasting_1241_divide_1214) {
  std::vector<float> input0 = {100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
  std::vector<float> input1 = {10.0f, 20.0f, 2.0f, 4.0f, 50.0f, 100.0f, 25.0f, 200.0f};

  const int out_shape = 1 * 2 * 4 * 4;  // 32 elements
  std::vector<float> output(out_shape, 0.0f);
  std::vector<float> tile_input0(out_shape, 0.0f);
  std::vector<float> tile_input1(out_shape, 0.0f);

  std::vector<float> benchmark = {10.0f, 5.0f,  50.0f, 25.0f,  20.0f,  10.0f, 100.0f, 50.0f, 30.0f, 15.0f, 150.0f,
                                  75.0f, 40.0f, 20.0f, 200.0f, 100.0f, 10.0f, 5.0f,   20.0f, 2.5f,  12.0f, 6.0f,
                                  24.0f, 3.0f,  14.0f, 7.0f,   28.0f,  3.5f,  16.0f,  8.0f,  32.0f, 4.0f};

  ArithmeticParameter arithmetic_parameter = {{"", 47, 1, 0},
                                              true,
                                              4,
                                              0,
                                              {1, 2, 4, 1},
                                              8,
                                              {1, 2, 1, 4},
                                              8,
                                              {1, 2, 4, 4},
                                              32,
                                              {8, 4, 1, 1},
                                              {8, 4, 4, 1},
                                              {32, 16, 4, 1},
                                              {1, 1, 1, 4},
                                              {1, 1, 4, 1}};

  TileDimensionsFp32(input0.data(), input1.data(), tile_input0.data(), tile_input1.data(), &arithmetic_parameter);
  ElementDiv(tile_input0.data(), tile_input1.data(), output.data(), out_shape);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Test case: FloorDiv FP32 broadcasting [1,2,4,1] // [1,2,1,4] -> [1,2,4,4]
TEST_F(BroadcastFp32Test, FloorDiv_Broadcasting_1241_floor_div_1214) {
  std::vector<float> input0 = {100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
  std::vector<float> input1 = {15.0f, 25.0f, 3.0f, 6.0f, 50.0f, 100.0f, 20.0f, 40.0f};

  const int out_shape = 1 * 2 * 4 * 4;  // 32 elements
  std::vector<float> output(out_shape, 0.0f);
  std::vector<float> tile_input0(out_shape, 0.0f);
  std::vector<float> tile_input1(out_shape, 0.0f);

  std::vector<float> benchmark = {6.0f,  4.0f,  33.0f, 16.0f,  13.0f, 8.0f,  66.0f, 33.0f, 20.0f, 12.0f, 100.0f,
                                  50.0f, 26.0f, 16.0f, 133.0f, 66.0f, 10.0f, 5.0f,  25.0f, 12.0f, 12.0f, 6.0f,
                                  30.0f, 15.0f, 14.0f, 7.0f,   35.0f, 17.0f, 16.0f, 8.0f,  40.0f, 20.0f};

  ArithmeticParameter arithmetic_parameter = {{"", 64, 1, 0},
                                              true,
                                              4,
                                              0,
                                              {1, 2, 4, 1},
                                              8,
                                              {1, 2, 1, 4},
                                              8,
                                              {1, 2, 4, 4},
                                              32,
                                              {8, 4, 1, 1},
                                              {8, 4, 4, 1},
                                              {32, 16, 4, 1},
                                              {1, 1, 1, 4},
                                              {1, 1, 4, 1}};

  TileDimensionsFp32(input0.data(), input1.data(), tile_input0.data(), tile_input1.data(), &arithmetic_parameter);
  ElementFloorDiv(tile_input0.data(), tile_input1.data(), output.data(), out_shape);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Test case: FloorMod FP32 broadcasting [1,2,4,1] % [1,2,1,4] -> [1,2,4,4]
TEST_F(BroadcastFp32Test, FloorMod_Broadcasting_1241_floor_mod_1214) {
  std::vector<float> input0 = {100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
  std::vector<float> input1 = {15.0f, 25.0f, 3.0f, 6.0f, 50.0f, 100.0f, 20.0f, 40.0f};

  const int out_shape = 1 * 2 * 4 * 4;  // 32 elements
  std::vector<float> output(out_shape, 0.0f);
  std::vector<float> tile_input0(out_shape, 0.0f);
  std::vector<float> tile_input1(out_shape, 0.0f);

  std::vector<float> benchmark = {10.0f, 0.0f,  1.0f, 4.0f, 5.0f, 0.0f,  2.0f, 2.0f, 0.0f,  0.0f, 0.0f,
                                  0.0f,  10.0f, 0.0f, 1.0f, 4.0f, 0.0f,  0.0f, 0.0f, 20.0f, 0.0f, 0.0f,
                                  0.0f,  0.0f,  0.0f, 0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 0.0f,  0.0f};

  ArithmeticParameter arithmetic_parameter = {{"", 65, 1, 0},
                                              true,
                                              4,
                                              0,
                                              {1, 2, 4, 1},
                                              8,
                                              {1, 2, 1, 4},
                                              8,
                                              {1, 2, 4, 4},
                                              32,
                                              {8, 4, 1, 1},
                                              {8, 4, 4, 1},
                                              {32, 16, 4, 1},
                                              {1, 1, 1, 4},
                                              {1, 1, 4, 1}};

  TileDimensionsFp32(input0.data(), input1.data(), tile_input0.data(), tile_input1.data(), &arithmetic_parameter);
  ElementFloorMod(tile_input0.data(), tile_input1.data(), output.data(), out_shape);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Test case: Maximum FP32 broadcasting [1,2,4,1] max [1,2,1,4] -> [1,2,4,4]
TEST_F(BroadcastFp32Test, Maximum_Broadcasting_1241_max_1214) {
  std::vector<float> input0 = {10.0f, 20.0f, 30.0f, 40.0f, 5.0f, 15.0f, 25.0f, 35.0f};
  std::vector<float> input1 = {50.0f, 60.0f, 70.0f, 80.0f, 100.0f, 200.0f, 300.0f, 400.0f};

  const int out_shape = 1 * 2 * 4 * 4;  // 32 elements
  std::vector<float> output(out_shape, 0.0f);
  std::vector<float> tile_input0(out_shape, 0.0f);
  std::vector<float> tile_input1(out_shape, 0.0f);

  std::vector<float> benchmark = {50.0f,  60.0f,  70.0f,  80.0f,  50.0f,  60.0f,  70.0f,  80.0f,
                                  50.0f,  60.0f,  70.0f,  80.0f,  50.0f,  60.0f,  70.0f,  80.0f,
                                  100.0f, 200.0f, 300.0f, 400.0f, 100.0f, 200.0f, 300.0f, 400.0f,
                                  100.0f, 200.0f, 300.0f, 400.0f, 100.0f, 200.0f, 300.0f, 400.0f};

  ArithmeticParameter arithmetic_parameter = {{"", 90, 1, 0},
                                              true,
                                              4,
                                              0,
                                              {1, 2, 4, 1},
                                              8,
                                              {1, 2, 1, 4},
                                              8,
                                              {1, 2, 4, 4},
                                              32,
                                              {8, 4, 1, 1},
                                              {8, 4, 4, 1},
                                              {32, 16, 4, 1},
                                              {1, 1, 1, 4},
                                              {1, 1, 4, 1}};

  TileDimensionsFp32(input0.data(), input1.data(), tile_input0.data(), tile_input1.data(), &arithmetic_parameter);
  ElementMaximum(tile_input0.data(), tile_input1.data(), output.data(), out_shape);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Test case: Minimum FP32 broadcasting [1,2,4,1] min [1,2,1,4] -> [1,2,4,4]
TEST_F(BroadcastFp32Test, Minimum_Broadcasting_1241_min_1214) {
  std::vector<float> input0 = {50.0f, 60.0f, 70.0f, 80.0f, 100.0f, 200.0f, 300.0f, 400.0f};
  std::vector<float> input1 = {10.0f, 20.0f, 30.0f, 40.0f, 5.0f, 15.0f, 25.0f, 35.0f};

  const int out_shape = 1 * 2 * 4 * 4;  // 32 elements
  std::vector<float> output(out_shape, 0.0f);
  std::vector<float> tile_input0(out_shape, 0.0f);
  std::vector<float> tile_input1(out_shape, 0.0f);

  std::vector<float> benchmark = {10.0f, 20.0f, 30.0f, 40.0f, 10.0f, 20.0f, 30.0f, 40.0f, 10.0f, 20.0f, 30.0f,
                                  40.0f, 10.0f, 20.0f, 30.0f, 40.0f, 5.0f,  15.0f, 25.0f, 35.0f, 5.0f,  15.0f,
                                  25.0f, 35.0f, 5.0f,  15.0f, 25.0f, 35.0f, 5.0f,  15.0f, 25.0f, 35.0f};

  ArithmeticParameter arithmetic_parameter = {{"", 96, 1, 0},
                                              true,
                                              4,
                                              0,
                                              {1, 2, 4, 1},
                                              8,
                                              {1, 2, 1, 4},
                                              8,
                                              {1, 2, 4, 4},
                                              32,
                                              {8, 4, 1, 1},
                                              {8, 4, 4, 1},
                                              {32, 16, 4, 1},
                                              {1, 1, 1, 4},
                                              {1, 1, 4, 1}};

  TileDimensionsFp32(input0.data(), input1.data(), tile_input0.data(), tile_input1.data(), &arithmetic_parameter);
  ElementMinimum(tile_input0.data(), tile_input1.data(), output.data(), out_shape);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

}  // namespace mindspore
