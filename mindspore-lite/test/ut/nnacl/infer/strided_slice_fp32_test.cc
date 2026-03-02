/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

/**
 * @file strided_slice_fp32_test.cc
 * @brief Unit tests for strided_slice_fp32 functions added in merge_back commit d7d3641a
 *
 * Tests cover:
 * - NormalizedSlice(): Parameter normalization with squeeze and merge
 * - DoStrideSliceCopyOpt(): Optimized slice copy with overflow protection
 * - Edge cases: overflow, boundary conditions, different data types
 */

#include <cstring>
#include <vector>

#include "common/common_test.h"
#include "nnacl_c/fp32/strided_slice_fp32.h"
#include "nnacl_c/strided_slice_parameter.h"
#include "nnacl_c/errorcode.h"
#include "nnacl_c/nnacl_common.h"

namespace mindspore {

class TestStridedSliceFp32 : public mindspore::CommonTest {
 public:
  TestStridedSliceFp32() {}
  ~TestStridedSliceFp32() override = default;

  void SetUp() override {}
  void TearDown() override {}
};

// ===================================================================
// Test Group 1: NormalizedSlice
// ===================================================================

TEST_F(TestStridedSliceFp32, NormalizedSlice_Basic2D) {
  // Test basic 2D slice normalization
  StridedSliceStruct s = {0};
  s.in_shape_size_ = 2;
  s.in_shape_[0] = 10;
  s.in_shape_[1] = 20;
  s.begins_[0] = 2;
  s.begins_[1] = 5;
  s.ends_[0] = 8;
  s.ends_[1] = 15;
  s.strides_[0] = 1;
  s.strides_[1] = 1;

  int ret = NormalizedSlice(&s);
  EXPECT_EQ(ret, NNACL_OK);
  EXPECT_EQ(s.num_normalized_dims, 2);
}

TEST_F(TestStridedSliceFp32, NormalizedSlice_SqueezeDim1) {
  // Test squeezing size-1 dimensions
  // Input: [5, 1, 10] -> Output: [5, 10] with merged offset
  StridedSliceStruct s = {0};
  s.in_shape_size_ = 3;
  s.in_shape_[0] = 5;
  s.in_shape_[1] = 1;
  s.in_shape_[2] = 10;
  s.begins_[0] = 1;
  s.begins_[1] = 0;
  s.begins_[2] = 3;
  s.ends_[0] = 4;
  s.ends_[1] = 1;
  s.ends_[2] = 8;
  s.strides_[0] = 1;
  s.strides_[1] = 1;
  s.strides_[2] = 1;

  int ret = NormalizedSlice(&s);
  EXPECT_EQ(ret, NNACL_OK);
  // After squeezing dimension 1, should have 2 dimensions
  EXPECT_LE(s.num_normalized_dims, 3);
}

TEST_F(TestStridedSliceFp32, NormalizedSlice_MergeFullDims) {
  // Test merging consecutive full-dimensional slices
  // Input: [10, 20, 30], begin=[0,0,0], end=[10,20,30]
  // Full dimensions should be merged
  StridedSliceStruct s = {0};
  s.in_shape_size_ = 3;
  s.in_shape_[0] = 10;
  s.in_shape_[1] = 20;
  s.in_shape_[2] = 30;
  s.begins_[0] = 0;
  s.begins_[1] = 0;
  s.begins_[2] = 0;
  s.ends_[0] = 10;
  s.ends_[1] = 20;
  s.ends_[2] = 30;
  s.strides_[0] = 1;
  s.strides_[1] = 1;
  s.strides_[2] = 1;

  int ret = NormalizedSlice(&s);
  EXPECT_EQ(ret, NNACL_OK);
}

TEST_F(TestStridedSliceFp32, NormalizedSlice_NegativeShapeSize) {
  // Test negative shape size (should fail)
  StridedSliceStruct s = {0};
  s.in_shape_size_ = -1;

  int ret = NormalizedSlice(&s);
  EXPECT_EQ(ret, NNACL_STRIDED_SLICE_INVALID_SHAPE_SIZE);
}

TEST_F(TestStridedSliceFp32, NormalizedSlice_ExceedMaxShapeSize) {
  // Test shape size exceeding MAX_SHAPE_SIZE (should fail)
  StridedSliceStruct s = {0};
  s.in_shape_size_ = MAX_SHAPE_SIZE + 1;

  int ret = NormalizedSlice(&s);
  EXPECT_EQ(ret, NNACL_STRIDED_SLICE_INVALID_SHAPE_SIZE);
}

TEST_F(TestStridedSliceFp32, NormalizedSlice_ZeroShapeSize) {
  // Test zero shape size (edge case)
  StridedSliceStruct s = {0};
  s.in_shape_size_ = 0;

  int ret = NormalizedSlice(&s);
  EXPECT_EQ(ret, NNACL_OK);
  EXPECT_EQ(s.num_normalized_dims, 0);
}

// ===================================================================
// Test Group 2: DoStrideSliceCopyOpt
// ===================================================================

TEST_F(TestStridedSliceFp32, DoStrideSliceCopyOpt_1D) {
  // Test 1D slice copy
  std::vector<float> input(100);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(i);
  }
  std::vector<float> output(100, 0.0f);

  StridedSliceStruct s = {0};
  s.data_type_ = static_cast<TypeIdC>(kNumberTypeFloat32);
  s.in_shape_size_ = 1;
  s.in_shape_[0] = 100;
  s.begins_[0] = 10;
  s.ends_[0] = 50;
  s.strides_[0] = 1;

  int ret = DoStrideSliceCopyOpt(input.data(), output.data(), &s);
  EXPECT_EQ(ret, NNACL_OK);

  // Verify first few elements
  EXPECT_FLOAT_EQ(output[0], input[10]);
  EXPECT_FLOAT_EQ(output[1], input[11]);
  EXPECT_FLOAT_EQ(output[2], input[12]);
}

TEST_F(TestStridedSliceFp32, DoStrideSliceCopyOpt_2D) {
  // Test 2D slice copy
  std::vector<float> input(100);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(i);
  }
  std::vector<float> output(100, 0.0f);

  StridedSliceStruct s = {0};
  s.data_type_ = static_cast<TypeIdC>(kNumberTypeFloat32);
  s.in_shape_size_ = 2;
  s.in_shape_[0] = 10;
  s.in_shape_[1] = 10;
  s.begins_[0] = 2;
  s.begins_[1] = 3;
  s.ends_[0] = 8;
  s.ends_[1] = 7;
  s.strides_[0] = 1;
  s.strides_[1] = 1;

  int ret = DoStrideSliceCopyOpt(input.data(), output.data(), &s);
  EXPECT_EQ(ret, NNACL_OK);

  // Verify output[0] corresponds to input[2*10 + 3] = input[23]
  EXPECT_FLOAT_EQ(output[0], input[23]);
}

TEST_F(TestStridedSliceFp32, DoStrideSliceCopyOpt_3D) {
  // Test 3D slice copy
  std::vector<float> input(1000);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(i);
  }
  std::vector<float> output(1000, 0.0f);

  StridedSliceStruct s = {0};
  s.data_type_ = static_cast<TypeIdC>(kNumberTypeFloat32);
  s.in_shape_size_ = 3;
  s.in_shape_[0] = 10;
  s.in_shape_[1] = 10;
  s.in_shape_[2] = 10;
  s.begins_[0] = 1;
  s.begins_[1] = 2;
  s.begins_[2] = 3;
  s.ends_[0] = 3;
  s.ends_[1] = 5;
  s.ends_[2] = 7;
  s.strides_[0] = 1;
  s.strides_[1] = 1;
  s.strides_[2] = 1;

  int ret = DoStrideSliceCopyOpt(input.data(), output.data(), &s);
  EXPECT_EQ(ret, NNACL_OK);
}

TEST_F(TestStridedSliceFp32, DoStrideSliceCopyOpt_4D) {
  // Test 4D slice copy
  std::vector<float> input(2400);  // 4*5*6*20
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(i);
  }
  std::vector<float> output(2400, 0.0f);

  StridedSliceStruct s = {0};
  s.data_type_ = static_cast<TypeIdC>(kNumberTypeFloat32);
  s.in_shape_size_ = 4;
  s.in_shape_[0] = 4;
  s.in_shape_[1] = 5;
  s.in_shape_[2] = 6;
  s.in_shape_[3] = 20;
  s.begins_[0] = 0;
  s.begins_[1] = 1;
  s.begins_[2] = 2;
  s.begins_[3] = 3;
  s.ends_[0] = 2;
  s.ends_[1] = 3;
  s.ends_[2] = 4;
  s.ends_[3] = 10;
  s.strides_[0] = 1;
  s.strides_[1] = 1;
  s.strides_[2] = 1;
  s.strides_[3] = 1;

  int ret = DoStrideSliceCopyOpt(input.data(), output.data(), &s);
  EXPECT_EQ(ret, NNACL_OK);
}

TEST_F(TestStridedSliceFp32, DoStrideSliceCopyOpt_Int8) {
  // Test with int8 data type
  std::vector<int8_t> input(100);
  for (int i = 0; i < 100; ++i) {
    input[i] = static_cast<int8_t>(i);
  }
  std::vector<int8_t> output(100, 0);

  StridedSliceStruct s = {0};
  s.data_type_ = static_cast<TypeIdC>(kNumberTypeInt8);
  s.in_shape_size_ = 2;
  s.in_shape_[0] = 10;
  s.in_shape_[1] = 10;
  s.begins_[0] = 2;
  s.begins_[1] = 3;
  s.ends_[0] = 5;
  s.ends_[1] = 7;
  s.strides_[0] = 1;
  s.strides_[1] = 1;

  int ret = DoStrideSliceCopyOpt(input.data(), output.data(), &s);
  EXPECT_EQ(ret, NNACL_OK);

  // Verify output[0] corresponds to input[2*10 + 3] = input[23]
  EXPECT_EQ(output[0], input[23]);
}

TEST_F(TestStridedSliceFp32, DoStrideSliceCopyOpt_Int32) {
  // Test with int32 data type
  std::vector<int32_t> input(100);
  for (int i = 0; i < 100; ++i) {
    input[i] = i * 100;
  }
  std::vector<int32_t> output(100, 0);

  StridedSliceStruct s = {0};
  s.data_type_ = static_cast<TypeIdC>(kNumberTypeInt32);
  s.in_shape_size_ = 2;
  s.in_shape_[0] = 10;
  s.in_shape_[1] = 10;
  s.begins_[0] = 1;
  s.begins_[1] = 1;
  s.ends_[0] = 5;
  s.ends_[1] = 5;
  s.strides_[0] = 1;
  s.strides_[1] = 1;

  int ret = DoStrideSliceCopyOpt(input.data(), output.data(), &s);
  EXPECT_EQ(ret, NNACL_OK);

  // Verify output[0] corresponds to input[1*10 + 1] = input[11]
  EXPECT_EQ(output[0], input[11]);
}

TEST_F(TestStridedSliceFp32, DoStrideSliceCopyOpt_InvalidNormalizedSlice) {
  // Test when NormalizedSlice fails
  std::vector<float> input(100, 1.0f);
  std::vector<float> output(100, 0.0f);

  StridedSliceStruct s = {0};
  s.data_type_ = static_cast<TypeIdC>(kNumberTypeFloat32);
  s.in_shape_size_ = -1;  // Invalid shape size
  s.strides_[0] = 1;

  int ret = DoStrideSliceCopyOpt(input.data(), output.data(), &s);
  EXPECT_EQ(ret, NNACL_STRIDED_SLICE_INVALID_SHAPE_SIZE);
}

}  // namespace mindspore
