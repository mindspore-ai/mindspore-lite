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
#include "nnacl_c/errorcode.h"
#include "nnacl_c/fp32/reduce_fp32.h"

namespace mindspore {
class ReduceMaxFp32Test : public ::testing::Test {};

TEST_F(ReduceMaxFp32Test, AllNegativeValues) {
  constexpr int inner_size = 17;
  const std::vector<float> input = {
    -20.0f, -19.0f, -18.0f, -17.0f, -16.0f, -15.0f, -14.0f, -13.0f, -12.0f, -11.0f, -10.0f, -9.0f,
    -8.0f,  -7.0f,  -6.0f,  -5.0f,  -4.0f,  -30.0f, -18.0f, -28.0f, -16.0f, -26.0f, -14.0f, -24.0f,
    -12.0f, -22.0f, -10.0f, -20.0f, -8.0f,  -18.0f, -6.0f,  -16.0f, -4.0f,  -14.0f,
  };
  const std::vector<float> expected = {
    -20.0f, -18.0f, -18.0f, -16.0f, -16.0f, -14.0f, -14.0f, -12.0f, -12.0f,
    -10.0f, -10.0f, -8.0f,  -8.0f,  -6.0f,  -6.0f,  -4.0f,  -4.0f,
  };
  std::vector<float> output(inner_size);

  ASSERT_EQ(ReduceMax(1, inner_size, 2, input.data(), output.data(), 0, 1), NNACL_OK);
  EXPECT_EQ(output, expected);
}

TEST_F(ReduceMaxFp32Test, NegativeInfinity) {
  const std::vector<float> input = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  float output = 0.0f;

  ASSERT_EQ(ReduceMax(1, 1, static_cast<int>(input.size()), input.data(), &output, 0, 1), NNACL_OK);
  EXPECT_TRUE(std::isinf(output));
  EXPECT_LT(output, 0.0f);
}
}  // namespace mindspore
