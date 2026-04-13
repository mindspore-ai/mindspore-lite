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
#include "nnacl_c/base/depth_to_space_base.h"
#include "nnacl_c/kernel/depth_to_space.h"
#include "nnacl_c/nnacl_common.h"

namespace mindspore {
class DepthToSpaceFp32Test : public ::testing::Test {
 public:
  DepthToSpaceFp32Test() {}
};

float get_cosine_similarity(const float *arr1, const float *arr2, size_t cmp_size);
static float accuracy_threshold = 0.99f;

// Testcase1: DepthToSpaceForNHWC (DCR) with input [1,2,2,4], block=2 -> output [1,4,4,1]
TEST_F(DepthToSpaceFp32Test, DCR_1x2x2x4_block2) {
  // input: N=1, H=2, W=2, C=4
  std::vector<float> input = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  std::vector<float> benchmark = {1.0f, 2.0f,  5.0f,  6.0f,  3.0f,  4.0f,  7.0f,  8.0f,
                                  9.0f, 10.0f, 13.0f, 14.0f, 11.0f, 12.0f, 15.0f, 16.0f};
  const int length = 16;
  std::vector<float> output(length, 0.0f);

  int in_shape[4] = {1, 2, 2, 4};
  int out_shape[4] = {1, 4, 4, 1};
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
  args.data_type_size_ = sizeof(float);
  args.block_size_ = 2;

  DepthToSpaceForNHWC(input.data(), output.data(), in_shape, &args);

  std::cout << "DepthToSpaceFp32Test DCR_1x2x2x4_block2 output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nDepthToSpaceFp32Test DCR_1x2x2x4_block2 benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: DepthToSpaceForNHWC (DCR) with input [1,1,2,8], block=2 -> output [1,2,4,2]
TEST_F(DepthToSpaceFp32Test, DCR_1x1x2x8_block2) {
  // input: N=1, H=1, W=2, C=8
  std::vector<float> input = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  std::vector<float> benchmark = {1.0f, 2.0f, 3.0f, 4.0f, 9.0f,  10.0f, 11.0f, 12.0f,
                                  5.0f, 6.0f, 7.0f, 8.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  const int length = 16;
  std::vector<float> output(length, 0.0f);

  int in_shape[4] = {1, 1, 2, 8};
  int out_shape[4] = {1, 2, 4, 2};
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
  args.data_type_size_ = sizeof(float);
  args.block_size_ = 2;

  DepthToSpaceForNHWC(input.data(), output.data(), in_shape, &args);

  std::cout << "DepthToSpaceFp32Test DCR_1x1x2x8_block2 output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nDepthToSpaceFp32Test DCR_1x1x2x8_block2 benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: DepthToSpaceCRDForNHWC (CRD) with input [1,2,2,4], block=2 -> output [1,4,4,1]
TEST_F(DepthToSpaceFp32Test, CRD_1x2x2x4_block2) {
  // input: N=1, H=2, W=2, C=4
  std::vector<float> input = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  std::vector<float> benchmark = {1.0f, 2.0f,  5.0f,  6.0f,  3.0f,  4.0f,  7.0f,  8.0f,
                                  9.0f, 10.0f, 13.0f, 14.0f, 11.0f, 12.0f, 15.0f, 16.0f};
  const int length = 16;
  std::vector<float> output(length, 0.0f);

  int in_shape[4] = {1, 2, 2, 4};
  int out_shape[4] = {1, 4, 4, 1};
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
  args.data_type_size_ = sizeof(float);
  args.block_size_ = 2;

  DepthToSpaceCRDForNHWC(input.data(), output.data(), in_shape, &args);

  std::cout << "DepthToSpaceFp32Test CRD_1x2x2x4_block2 output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nDepthToSpaceFp32Test CRD_1x2x2x4_block2 benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase4: DepthToSpaceCRDForNHWC (CRD) with input [1,1,2,8], block=2 -> output [1,2,4,2]
TEST_F(DepthToSpaceFp32Test, CRD_1x1x2x8_block2) {
  // input: N=1, H=1, W=2, C=8
  std::vector<float> input = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  std::vector<float> benchmark = {1.0f, 5.0f, 2.0f, 6.0f, 9.0f,  13.0f, 10.0f, 14.0f,
                                  3.0f, 7.0f, 4.0f, 8.0f, 11.0f, 15.0f, 12.0f, 16.0f};
  const int length = 16;
  std::vector<float> output(length, 0.0f);

  int in_shape[4] = {1, 1, 2, 8};
  int out_shape[4] = {1, 2, 4, 2};
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
  args.data_type_size_ = sizeof(float);
  args.block_size_ = 2;

  DepthToSpaceCRDForNHWC(input.data(), output.data(), in_shape, &args);

  std::cout << "DepthToSpaceFp32Test CRD_1x1x2x8_block2 output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nDepthToSpaceFp32Test CRD_1x1x2x8_block2 benchmark:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
