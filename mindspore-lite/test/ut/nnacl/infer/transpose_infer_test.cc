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
#include "common/common_test.h"
#include "nnacl_c/infer/transpose_infer.h"

namespace mindspore {

class TransposeInferTest : public mindspore::CommonTest {
 public:
  TransposeInferTest() {}
};

TEST_F(TransposeInferTest, TransposeInferTest0) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC();
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 5;
  inputs[0]->shape_[2] = 6;
  inputs[0]->shape_[3] = 7;
  // the infer reads the permutation from the second input tensor
  inputs[1] = new TensorC();
  int32_t perm_data[] = {2, 1, 3, 0};
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 4;
  inputs[1]->data_type_ = kNumberTypeInt32;
  inputs[1]->data_ = perm_data;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC();
  TransposeParameter *parameter = new TransposeParameter();
  int ret = TransposeInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 6);
  ASSERT_EQ(outputs[0]->shape_[1], 5);
  ASSERT_EQ(outputs[0]->shape_[2], 7);
  ASSERT_EQ(outputs[0]->shape_[3], 4);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
