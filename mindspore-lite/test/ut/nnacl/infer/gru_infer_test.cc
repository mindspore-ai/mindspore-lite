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
#include "nnacl/infer/gru_infer.h"

namespace mindspore {

class GruInferTest : public mindspore::CommonTest {
 public:
  GruInferTest() {}
};

TEST_F(GruInferTest, GruInferTest0) {
  size_t inputs_size = 5;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 5;
  inputs[0]->shape_[2] = 6;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 3;
  inputs[1]->shape_[1] = 9;
  inputs[2] = new TensorC;
  inputs[2]->shape_size_ = 3;
  inputs[2]->shape_[1] = 9;
  inputs[3] = new TensorC;
  inputs[3]->shape_[1] = 18;
  inputs[4] = new TensorC;

  std::vector<TensorC *> outputs(2, NULL);
  outputs[0] = new TensorC;
  outputs[1] = new TensorC;
  GruParameter *parameter = new GruParameter;
  parameter->bidirectional_ = true;
  OpParameter *param = reinterpret_cast<OpParameter *>(parameter);
  int ret = GruInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(), param);
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  ASSERT_EQ(outputs[0]->shape_[2], 5);
  ASSERT_EQ(outputs[0]->shape_[3], 3);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs[0]->format_, Format_NHWC);

  ASSERT_EQ(outputs[1]->shape_size_, 3);
  ASSERT_EQ(outputs[1]->shape_[0], 2);
  ASSERT_EQ(outputs[1]->shape_[1], 5);
  ASSERT_EQ(outputs[1]->shape_[2], 3);
  ASSERT_EQ(outputs[1]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs[1]->format_, Format_NHWC);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

// bidirectional_ is false and inputs_size is 6
TEST_F(GruInferTest, GruInferTest1) {
  size_t inputs_size = 6;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 5;
  inputs[0]->shape_[2] = 6;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new TensorC;
  inputs[1]->shape_size_ = 3;
  inputs[1]->shape_[1] = 9;
  inputs[2] = new TensorC;
  inputs[2]->shape_size_ = 3;
  inputs[2]->shape_[1] = 9;
  inputs[3] = new TensorC;
  inputs[3]->shape_[1] = 18;
  inputs[4] = new TensorC;
  inputs[5] = new TensorC;
  inputs[5]->shape_size_ = 1;
  inputs[5]->shape_[0] = -1;

  std::vector<TensorC *> outputs(2, NULL);
  outputs[0] = new TensorC;
  outputs[1] = new TensorC;
  GruParameter *parameter = new GruParameter;
  parameter->bidirectional_ = false;
  int ret = GruInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                          reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 1);
  ASSERT_EQ(outputs[0]->shape_[2], 5);
  ASSERT_EQ(outputs[0]->shape_[3], 3);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs[0]->format_, Format_NHWC);

  ASSERT_EQ(outputs[1]->shape_size_, 3);
  ASSERT_EQ(outputs[1]->shape_[0], 1);
  ASSERT_EQ(outputs[1]->shape_[1], 5);
  ASSERT_EQ(outputs[1]->shape_[2], 3);
  ASSERT_EQ(outputs[1]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs[1]->format_, Format_NHWC);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
