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
class ConstantOfShapeInt8Test : public ::testing::Test {
 public:
  ConstantOfShapeInt8Test() {}
};

// Testcase1: fill quantized zero (q=0) into a 1D tensor
TEST_F(ConstantOfShapeInt8Test, FillQuantizedZero) {
  const int length = 16;
  std::vector<int8_t> output(length, 0);
  ConstantOfShapeInt8(output.data(), 0, length, 0);

  for (int i = 0; i < length; ++i) {
    ASSERT_EQ(output[i], 0);
  }
}

// Testcase2: fill a positive quantized value (q=64) into a 2D tensor
TEST_F(ConstantOfShapeInt8Test, FillQuantizedPositive) {
  const int length = 4 * 8;
  std::vector<int8_t> output(length, 0);
  ConstantOfShapeInt8(output.data(), 0, length, 64);

  for (int i = 0; i < length; ++i) {
    ASSERT_EQ(output[i], 64);
  }
}
}  // namespace mindspore
