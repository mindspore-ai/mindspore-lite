/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "common/common_test.h"
#include "nnacl_c/fp32/unique_fp32.h"

namespace mindspore {
class TestUniqueFp32 : public mindspore::CommonTest {
 public:
  TestUniqueFp32() {}
};

// the Unique kernel is registered in the nnacl KernelBase registry, so the test calls
// the compute function directly instead of going through lite::KernelRegistry
TEST_F(TestUniqueFp32, Unique) {
  float input_data[] = {1, 1, 2, 4, 4, 4, 7, 8, 8};
  float output_data0[9] = {0};
  int output_data1[9] = {0};
  int32_t output_data0_len = 0;

  Unique(input_data, 9, output_data0, &output_data0_len, output_data1);

  float expect0[] = {1, 2, 4, 7, 8};
  int expect1[] = {0, 0, 1, 2, 2, 2, 3, 4, 4};
  EXPECT_EQ(output_data0_len, 5);
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(output_data0[i], expect0[i]);
  }
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(output_data1[i], expect1[i]);
  }
}

TEST_F(TestUniqueFp32, UniqueInt32) {
  int32_t input_data[] = {5, 5, 3, 9, 9, 9, 1};
  int32_t output_data0[7] = {0};
  int output_data1[7] = {0};
  int32_t output_data0_len = 0;

  UniqueInt(input_data, 7, output_data0, &output_data0_len, output_data1);

  int32_t expect0[] = {5, 3, 9, 1};
  int expect1[] = {0, 0, 1, 2, 2, 2, 3};
  EXPECT_EQ(output_data0_len, 4);
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(output_data0[i], expect0[i]);
  }
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(output_data1[i], expect1[i]);
  }
}
}  // namespace mindspore
