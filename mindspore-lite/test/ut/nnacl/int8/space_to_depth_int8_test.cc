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
#include "nnacl_c/errorcode.h"
#include "nnacl_c/int8/space_to_depth_int8.h"

namespace mindspore {
class SpaceToDepthInt8Test : public ::testing::Test {
 public:
  SpaceToDepthInt8Test() {}
};

float get_cosine_similarity_int8(const int8_t *arr1, const int8_t *arr2, size_t cmp_size);
static float accuracy_threshold = 0.99;

// Testcase1: input 1x4x4x1 block_size=2, scale=1.0 zp=0, output 1x2x2x4
TEST_F(SpaceToDepthInt8Test, Testcase01) {
  std::vector<int8_t> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<int8_t> benchmark = {1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16};
  const int length = 1 * 2 * 2 * 4;
  std::vector<int8_t> output(length, 0);

  int32_t in_shape[] = {1, 4, 4, 1};
  int32_t out_shape[] = {1, 2, 2, 4};
  SpaceToDepthParameter param = {};
  param.op_parameter_.thread_num_ = 1;
  param.block_size_ = 2;

  QuantArg in_quant_arg = {};
  in_quant_arg.scale_ = 1.0f;
  in_quant_arg.zp_ = 0;
  QuantArg out_quant_arg = {};
  out_quant_arg.scale_ = 1.0f;
  out_quant_arg.zp_ = 0;

  auto ret = SpaceToDepthForNHWCInt8(input.data(), output.data(), in_shape, out_shape, 4, &param, &in_quant_arg,
                                     &out_quant_arg, 0);
  ASSERT_EQ(ret, NNACL_OK);

  std::cout << "SpaceToDepthInt8Test Testcase01 output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nSpaceToDepthInt8Test Testcase01 benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: input 1x4x4x2 block_size=2, in_scale=0.5 in_zp=10, out_scale=1.0 out_zp=5
TEST_F(SpaceToDepthInt8Test, Testcase02) {
  std::vector<int8_t> input = {10, 20, 30, 40, 50, 60, 70, 80, 11, 21, 31, 41, 51, 61, 71, 81,
                               12, 22, 32, 42, 52, 62, 72, 82, 13, 23, 33, 43, 53, 63, 73, 83};
  // scale = 0.5, bias = -10*0.5 = -5.0, output_zp = 5
  // output_tmp = round(input_val * 0.5 - 5.0) + 5
  std::vector<int8_t> benchmark = {5, 10, 15, 20, 6, 11, 16, 21, 25, 30, 35, 40, 26, 31, 36, 41,
                                   6, 11, 16, 21, 7, 12, 17, 22, 26, 31, 36, 41, 27, 32, 37, 42};
  const int length = 1 * 2 * 2 * 8;
  std::vector<int8_t> output(length, 0);

  int32_t in_shape[] = {1, 4, 4, 2};
  int32_t out_shape[] = {1, 2, 2, 8};
  SpaceToDepthParameter param = {};
  param.op_parameter_.thread_num_ = 1;
  param.block_size_ = 2;

  QuantArg in_quant_arg = {};
  in_quant_arg.scale_ = 0.5f;
  in_quant_arg.zp_ = 10;
  QuantArg out_quant_arg = {};
  out_quant_arg.scale_ = 1.0f;
  out_quant_arg.zp_ = 5;

  auto ret = SpaceToDepthForNHWCInt8(input.data(), output.data(), in_shape, out_shape, 4, &param, &in_quant_arg,
                                     &out_quant_arg, 0);
  ASSERT_EQ(ret, NNACL_OK);

  std::cout << "SpaceToDepthInt8Test Testcase02 output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nSpaceToDepthInt8Test Testcase02 benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
