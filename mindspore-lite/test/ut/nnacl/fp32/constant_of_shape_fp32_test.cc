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
#include <vector>
#include "gtest/gtest.h"
#include "nnacl_c/fp32/constant_of_shape_fp32.h"

namespace mindspore {
class ConstantOfShapeFp32Test : public ::testing::Test {
 public:
  ConstantOfShapeFp32Test() {}
};

// Testcase1: fill scalar 1.0 into a 1D tensor (16 elements)
TEST_F(ConstantOfShapeFp32Test, FillOne_1D) {
  const int length = 16;
  std::vector<float> output(length, 0.0f);
  ConstantOfShapeFp32(output.data(), 0, length, 1.0f);

  for (int i = 0; i < length; ++i) {
    ASSERT_FLOAT_EQ(output[i], 1.0f);
  }
}

// Testcase2: fill scalar 0.5 into a 2D tensor (4x8)
TEST_F(ConstantOfShapeFp32Test, FillHalf_2D) {
  const int length = 4 * 8;
  std::vector<float> output(length, 0.0f);
  ConstantOfShapeFp32(output.data(), 0, length, 0.5f);

  for (int i = 0; i < length; ++i) {
    ASSERT_FLOAT_EQ(output[i], 0.5f);
  }
}

// Testcase3: fill int32 scalar 7 into a 1D tensor
TEST_F(ConstantOfShapeFp32Test, FillInt32) {
  const int length = 8;
  std::vector<int32_t> output(length, 0);
  ConstantOfShapeInt32(output.data(), 0, length, 7);

  for (int i = 0; i < length; ++i) {
    ASSERT_EQ(output[i], 7);
  }
}

// Testcase4: fill bool true into a 1D tensor
TEST_F(ConstantOfShapeFp32Test, FillBool) {
  const int length = 8;
  bool output[8] = {false};
  ConstantOfShapeBool(output, 0, length, true);

  for (int i = 0; i < length; ++i) {
    ASSERT_EQ(output[i], true);
  }
}
}  // namespace mindspore
