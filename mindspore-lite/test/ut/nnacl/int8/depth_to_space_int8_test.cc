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
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include "gtest/gtest.h"
#include "nnacl_c/int8/depth_to_space_int8.h"
#include "nnacl_c/kernel/depth_to_space.h"
#include "nnacl_c/nnacl_common.h"

namespace mindspore {
class DepthToSpaceInt8Test : public ::testing::Test {
 public:
  DepthToSpaceInt8Test() {}
};

float get_cosine_similarity_int8(const int8_t *arr1, const int8_t *arr2, size_t cmp_size);
static float accuracy_threshold = 0.99f;

// Testcase1: DepthToSpaceForNHWCInt8 (DCR) with input [1,2,2,4], block=2
// in_scale=1.0, in_zp=0, out_scale=2.0, out_zp=1
TEST_F(DepthToSpaceInt8Test, DCR_1x2x2x4_scale1zp0_outscale2zp1) {
  std::vector<int8_t> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  // scale = 1.0/2.0 = 0.5, bias = 0, output = round(input*0.5) + 1
  std::vector<int8_t> benchmark = {1, 2, 3, 4, 3, 3, 5, 5, 5, 6, 7, 8, 7, 7, 9, 9};
  const int length = 16;
  std::vector<int8_t> output(length, 0);

  int32_t in_shape[4] = {1, 2, 2, 4};
  int32_t out_shape[4] = {1, 4, 4, 1};
  int32_t in_strides[4] = {0};
  int32_t out_strides[4] = {0};
  ComputeStrides(in_shape, in_strides, 4);
  ComputeStrides(out_shape, out_strides, 4);

  DepthToSpaceArgs args = {};
  args.in_stride_dim0_ = in_strides[0];
  args.in_stride_dim1_ = in_strides[1];
  args.in_stride_dim2_ = in_strides[2];
  args.out_stride_dim0_ = out_strides[0];
  args.out_stride_dim1_ = out_strides[1];
  args.out_stride_dim2_ = out_strides[2];
  args.data_type_size_ = sizeof(int8_t);
  args.block_size_ = 2;

  QuantArg in_quant_arg = {};
  in_quant_arg.scale_ = 1.0f;
  in_quant_arg.zp_ = 0;
  QuantArg out_quant_arg = {};
  out_quant_arg.scale_ = 2.0f;
  out_quant_arg.zp_ = 1;

  DepthToSpaceForNHWCInt8(input.data(), output.data(), in_shape, &args, &in_quant_arg, &out_quant_arg);

  std::cout << "DepthToSpaceInt8Test DCR_1x2x2x4 output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nDepthToSpaceInt8Test DCR_1x2x2x4 benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: DepthToSpaceForNHWCInt8 (DCR) with input [1,1,2,8], block=2
// in_scale=0.5, in_zp=10, out_scale=0.25, out_zp=-5
TEST_F(DepthToSpaceInt8Test, DCR_1x1x2x8_scale0p5zp10_outscale0p25zpm5) {
  std::vector<int8_t> input = {10, -20, 30, -40, 50, -60, 70, -80, -10, 20, -30, 40, -50, 60, -70, 80};
  // scale = 0.5/0.25 = 2.0, bias = -10*2.0 = -20.0, output = round(val*2.0 - 20.0) + (-5)
  std::vector<int8_t> benchmark = {-5, -65, 35, -105, -45, 15, -85, 55, 75, -128, 115, -128, -125, 95, -128, 127};
  const int length = 16;
  std::vector<int8_t> output(length, 0);

  int32_t in_shape[4] = {1, 1, 2, 8};
  int32_t out_shape[4] = {1, 2, 4, 2};
  int32_t in_strides[4] = {0};
  int32_t out_strides[4] = {0};
  ComputeStrides(in_shape, in_strides, 4);
  ComputeStrides(out_shape, out_strides, 4);

  DepthToSpaceArgs args = {};
  args.in_stride_dim0_ = in_strides[0];
  args.in_stride_dim1_ = in_strides[1];
  args.in_stride_dim2_ = in_strides[2];
  args.out_stride_dim0_ = out_strides[0];
  args.out_stride_dim1_ = out_strides[1];
  args.out_stride_dim2_ = out_strides[2];
  args.data_type_size_ = sizeof(int8_t);
  args.block_size_ = 2;

  QuantArg in_quant_arg = {};
  in_quant_arg.scale_ = 0.5f;
  in_quant_arg.zp_ = 10;
  QuantArg out_quant_arg = {};
  out_quant_arg.scale_ = 0.25f;
  out_quant_arg.zp_ = -5;

  DepthToSpaceForNHWCInt8(input.data(), output.data(), in_shape, &args, &in_quant_arg, &out_quant_arg);

  std::cout << "DepthToSpaceInt8Test DCR_1x1x2x8 output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nDepthToSpaceInt8Test DCR_1x1x2x8 benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: DepthToSpaceCRDForNHWCInt8 (CRD) with input [1,2,2,4], block=2
// in_scale=1.0, in_zp=0, out_scale=2.0, out_zp=1
TEST_F(DepthToSpaceInt8Test, CRD_1x2x2x4_scale1zp0_outscale2zp1) {
  std::vector<int8_t> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  // CRD layout + scale=0.5, bias=0, output = round(input*0.5) + 1
  std::vector<int8_t> benchmark = {1, 2, 3, 4, 3, 3, 5, 5, 5, 6, 7, 8, 7, 7, 9, 9};
  const int length = 16;
  std::vector<int8_t> output(length, 0);

  int32_t in_shape[4] = {1, 2, 2, 4};
  int32_t out_shape[4] = {1, 4, 4, 1};
  int32_t in_strides[4] = {0};
  int32_t out_strides[4] = {0};
  ComputeStrides(in_shape, in_strides, 4);
  ComputeStrides(out_shape, out_strides, 4);

  DepthToSpaceArgs args = {};
  args.in_stride_dim0_ = in_strides[0];
  args.in_stride_dim1_ = in_strides[1];
  args.in_stride_dim2_ = in_strides[2];
  args.out_stride_dim0_ = out_strides[0];
  args.out_stride_dim1_ = out_strides[1];
  args.out_stride_dim2_ = out_strides[2];
  args.data_type_size_ = sizeof(int8_t);
  args.block_size_ = 2;

  QuantArg in_quant_arg = {};
  in_quant_arg.scale_ = 1.0f;
  in_quant_arg.zp_ = 0;
  QuantArg out_quant_arg = {};
  out_quant_arg.scale_ = 2.0f;
  out_quant_arg.zp_ = 1;

  DepthToSpaceCRDForNHWCInt8(input.data(), output.data(), in_shape, &args, &in_quant_arg, &out_quant_arg);

  std::cout << "DepthToSpaceInt8Test CRD_1x2x2x4 output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nDepthToSpaceInt8Test CRD_1x2x2x4 benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase4: DepthToSpaceCRDForNHWCInt8 (CRD) with input [1,1,2,8], block=2
// in_scale=0.5, in_zp=10, out_scale=0.25, out_zp=-5
TEST_F(DepthToSpaceInt8Test, CRD_1x1x2x8_scale0p5zp10_outscale0p25zpm5) {
  std::vector<int8_t> input = {10, -20, 30, -40, 50, -60, 70, -80, -10, 20, -30, 40, -50, 60, -70, 80};
  // CRD layout + scale=2.0, bias=-20.0, output = round(val*2.0-20.0) + (-5)
  std::vector<int8_t> benchmark = {-5, 75, -65, -128, -45, -125, 15, 95, 35, 115, -105, -128, -85, -128, 55, 127};
  const int length = 16;
  std::vector<int8_t> output(length, 0);

  int32_t in_shape[4] = {1, 1, 2, 8};
  int32_t out_shape[4] = {1, 2, 4, 2};
  int32_t in_strides[4] = {0};
  int32_t out_strides[4] = {0};
  ComputeStrides(in_shape, in_strides, 4);
  ComputeStrides(out_shape, out_strides, 4);

  DepthToSpaceArgs args = {};
  args.in_stride_dim0_ = in_strides[0];
  args.in_stride_dim1_ = in_strides[1];
  args.in_stride_dim2_ = in_strides[2];
  args.out_stride_dim0_ = out_strides[0];
  args.out_stride_dim1_ = out_strides[1];
  args.out_stride_dim2_ = out_strides[2];
  args.data_type_size_ = sizeof(int8_t);
  args.block_size_ = 2;

  QuantArg in_quant_arg = {};
  in_quant_arg.scale_ = 0.5f;
  in_quant_arg.zp_ = 10;
  QuantArg out_quant_arg = {};
  out_quant_arg.scale_ = 0.25f;
  out_quant_arg.zp_ = -5;

  DepthToSpaceCRDForNHWCInt8(input.data(), output.data(), in_shape, &args, &in_quant_arg, &out_quant_arg);

  std::cout << "DepthToSpaceInt8Test CRD_1x1x2x8 output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\nDepthToSpaceInt8Test CRD_1x1x2x8 benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
