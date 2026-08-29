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
#include "nnacl_c/infer/fill_infer.h"
#include "nnacl_c/fill_parameter.h"

namespace mindspore {

class FillInferTest : public mindspore::CommonTest {
 public:
  FillInferTest() {}
};

TEST_F(FillInferTest, FillInferTest0) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC();
  // the infer reads the dst dims from a second input tensor (1-D int32)
  inputs[1] = new TensorC();
  inputs[1]->data_type_ = kNumberTypeInt32;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 4;
  int32_t dst_dims[] = {1, 2, 3, 4};
  inputs[1]->data_ = dst_dims;
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 1;
  inputs[0]->shape_[1] = 2;
  inputs[0]->shape_[2] = 3;
  inputs[0]->shape_[3] = 4;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC();
  FillParameter *parameter = new FillParameter();
  int ret = FillInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  ASSERT_EQ(outputs[0]->shape_[2], 3);
  ASSERT_EQ(outputs[0]->shape_[3], 4);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(FillInferTest, FillInferTest1) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC();
  // the infer reads the dst dims from a second input tensor (1-D int32)
  inputs[1] = new TensorC();
  inputs[1]->data_type_ = kNumberTypeInt32;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 3;
  int32_t dst_dims[] = {4, 2, 3};
  inputs[1]->data_ = dst_dims;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC();
  FillParameter *parameter = new FillParameter();
  int ret = FillInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  ASSERT_EQ(outputs[0]->shape_[2], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(FillInferTest, FillInferTest2) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC();
  // the infer reads the dst dims from a second input tensor (1-D int32)
  inputs[1] = new TensorC();
  inputs[1]->data_type_ = kNumberTypeInt32;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 2;
  int32_t dst_dims[] = {4, 2};
  inputs[1]->data_ = dst_dims;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC();
  FillParameter *parameter = new FillParameter();
  int ret = FillInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(FillInferTest, FillInferTest3) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC();
  // the infer reads the dst dims from a second input tensor (1-D int32)
  inputs[1] = new TensorC();
  inputs[1]->data_type_ = kNumberTypeInt32;
  inputs[1]->shape_size_ = 1;
  inputs[1]->shape_[0] = 1;
  int32_t dst_dims[] = {4};
  inputs[1]->data_ = dst_dims;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC();
  FillParameter *parameter = new FillParameter();
  int ret = FillInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 1);
  ASSERT_EQ(outputs[0]->shape_[0], 4);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
