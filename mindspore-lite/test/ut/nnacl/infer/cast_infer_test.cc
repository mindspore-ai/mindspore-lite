/**
 * Copyright 2020~2025 Huawei Technologies Co., Ltd
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
#include "nnacl_c/infer/cast_infer.h"

namespace mindspore {

class CastInferTest : public mindspore::CommonTest {
 public:
  CastInferTest() {}
};

TEST_F(CastInferTest, CastInferTest0) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new (std::nothrow) TensorC;
  if (inputs[0] == nullptr) {
    return;
  }
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 5;
  inputs[0]->data_type_ = kNumberTypeFloat32;
  inputs[1] = new (std::nothrow) TensorC;
  if (inputs[1] == nullptr) {
    return;
  }
  inputs[1]->data_ = new (std::nothrow) int;
  if (inputs[1]->data_ == nullptr) {
    return;
  }
  *static_cast<int *>(inputs[1]->data_) = kNumberTypeInt32;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new (std::nothrow) TensorC;
  if (outputs[0] == nullptr) {
    return;
  }
  OpParameter *parameter = new (std::nothrow) OpParameter;
  if (parameter == nullptr) {
    return;
  }
  int ret = CastInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 2);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs[0]->shape_[0], 2);
  ASSERT_EQ(outputs[0]->shape_[1], 5);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(CastInferTest, CastInferTest1) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new (std::nothrow) TensorC;
  if (inputs[0] == nullptr) {
    return;
  }
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  inputs[0]->data_type_ = kNumberTypeFloat32;
  size_t outputs_size = 1;
  std::vector<TensorC *> outputs(outputs_size, NULL);
  outputs[0] = new (std::nothrow) TensorC;
  if (outputs[0] == nullptr) {
    return;
  }
  OpParameter *parameter = new (std::nothrow) OpParameter;
  if (parameter == nullptr) {
    return;
  }
  int ret = CastInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_INPUT_TENSOR_ERROR);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(CastInferTest, CastInferTest2) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new (std::nothrow) TensorC;
  if (inputs[0] == nullptr) {
    return;
  }
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  inputs[0]->data_type_ = kNumberTypeFloat32;
  inputs[1] = nullptr;

  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new (std::nothrow) TensorC;
  if (outputs[0] == nullptr) {
    return;
  }
  OpParameter *parameter = new OpParameter;
  if (parameter == nullptr) {
    return;
  }
  int ret = CastInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_NULL_PTR);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    if (inputs[i] != nullptr) {
      delete inputs[i];
    }
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(CastInferTest, CastInferTest3) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new (std::nothrow) TensorC;
  if (inputs[0] == nullptr) {
    return;
  }
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 5;
  inputs[0]->data_type_ = kNumberTypeFloat32;
  inputs[1] = new (std::nothrow) TensorC;
  if (inputs[1] == nullptr) {
    return;
  }
  inputs[1]->data_ = new (std::nothrow) int;
  if (inputs[1]->data_ == nullptr) {
    return;
  }
  *static_cast<int *>(inputs[1]->data_) = kNumberTypeInt32;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = nullptr;
  OpParameter *parameter = new OpParameter;
  if (parameter == nullptr) {
    return;
  }
  int ret = CastInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_NULL_PTR);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i] != nullptr) {
      delete outputs[i];
    }
  }
}

TEST_F(CastInferTest, CastInferTest4) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new (std::nothrow) TensorC;
  if (inputs[0] == nullptr) {
    return;
  }
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 5;
  inputs[0]->data_type_ = kNumberTypeUInt16;
  inputs[1] = new (std::nothrow) TensorC;
  if (inputs[1] == nullptr) {
    return;
  }
  inputs[1]->data_ = new (std::nothrow) int;
  if (inputs[1]->data_ == nullptr) {
    return;
  }
  *static_cast<int *>(inputs[1]->data_) = kNumberTypeInt32;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new (std::nothrow) TensorC;
  if (outputs[0] == nullptr) {
    return;
  }
  OpParameter *parameter = new (std::nothrow) OpParameter;
  if (parameter == nullptr) {
    return;
  }
  int ret = CastInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_INFER_INVALID);
  ASSERT_EQ(inputs[0]->shape_size_, 2);
  ASSERT_EQ(inputs[0]->shape_[0], 2);
  ASSERT_EQ(inputs[0]->shape_[1], 5);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(CastInferTest, CastInferTest5) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new (std::nothrow) TensorC;
  if (inputs[0] == nullptr) {
    return;
  }
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 1;
  inputs[0]->shape_[1] = 3;
  inputs[0]->shape_[2] = 128;
  inputs[0]->shape_[3] = 128;
  inputs[0]->data_type_ = kNumberTypeFloat16;
  inputs[1] = new (std::nothrow) TensorC;
  if (inputs[1] == nullptr) {
    return;
  }
  inputs[1]->data_ = new (std::nothrow) int;
  if (inputs[1]->data_ == nullptr) {
    return;
  }
  *static_cast<int *>(inputs[1]->data_) = kNumberTypeInt32;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new (std::nothrow) TensorC;
  if (outputs[0] == nullptr) {
    return;
  }
  OpParameter *parameter = new (std::nothrow) OpParameter;
  if (parameter == nullptr) {
    return;
  }
  int ret = CastInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 3);
  ASSERT_EQ(outputs[0]->shape_[2], 128);
  ASSERT_EQ(outputs[0]->shape_[3], 128);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}
}  // namespace mindspore
