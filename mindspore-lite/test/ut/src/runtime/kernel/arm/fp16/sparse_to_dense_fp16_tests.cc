/**
 * Copyright 2020-2026 Huawei Technologies Co., Ltd
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

#include "common/common_test.h"
#include "nnacl_c/fp16/sparse_to_dense_fp16.h"

namespace mindspore {

class TestSparseToDenseFp16 : public mindspore::CommonTest {
 public:
  TestSparseToDenseFp16() {}
};

// Test: thread_num = 0 should return error (core bug fix verification)
TEST_F(TestSparseToDenseFp16, ThreadNumZeroReturnsError) {
  std::vector<int> indices_vec = {0, 0, 1, 1};
  std::vector<float16_t> sparse_values = {1.0f, 2.0f};
  float16_t default_value = 0.0f;
  std::vector<float16_t> output(4, 0.0f);

  auto *param = static_cast<SparseToDenseParameter *>(malloc(sizeof(SparseToDenseParameter)));
  memset(param, 0, sizeof(SparseToDenseParameter));
  param->op_parameter_.thread_num_ = 0;  // Core fix: verify this returns error
  param->index_num = 2;
  param->output_shape[0] = 2;
  param->output_shape[1] = 2;
  param->output_stride[0] = 2;
  param->output_stride[1] = 1;

  int ret = SparseToDenseFp16(indices_vec.data(), sparse_values.data(), default_value, output.data(), param, 0);

  // Core fix: thread_num = 0 should return NNACL_ERR (not NNACL_OK)
  ASSERT_EQ(ret, NNACL_ERR);

  free(param);
}

}  // namespace mindspore
