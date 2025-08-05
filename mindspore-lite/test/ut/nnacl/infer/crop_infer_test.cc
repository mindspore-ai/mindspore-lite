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
#include "nnacl_c/infer/crop_infer.h"

namespace mindspore {

class CropInferTest : public mindspore::CommonTest {
 public:
  CropInferTest() {}
};

TEST_F(CropInferTest, CropInferTest0) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new (std::nothrow) TensorC;
  if (inputs[0] == nullptr) {
    return;
  }
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new (std::nothrow) TensorC;
  if (inputs[1] == nullptr) {
    return;
  }
  inputs[1]->shape_size_ = 3;
  inputs[1]->shape_[0] = 5;
  inputs[1]->shape_[1] = 6;
  inputs[1]->shape_[2] = 7;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new (std::nothrow) TensorC;
  if (outputs[0] == nullptr) {
    return;
  }
  CropParameter *parameter = new (std::nothrow) CropParameter;
  if (parameter == nullptr) {
    return;
  }
  int ret = CropInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 5);
  ASSERT_EQ(outputs[0]->shape_[1], 6);
  ASSERT_EQ(outputs[0]->shape_[2], 7);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(outputs[0]->format_, Format_NHWC);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    if (inputs[i] != nullptr) {
      delete inputs[i];
    }
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i] != nullptr) {
      delete outputs[i];
    }
  }
}

TEST_F(CropInferTest, CropInferTest1) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new (std::nothrow) TensorC;
  if (inputs[0] == nullptr) {
    return;
  }
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new (std::nothrow) TensorC;
  if (inputs[1] == nullptr) {
    return;
  }
  inputs[1]->shape_size_ = 3;
  inputs[1]->shape_[0] = 5;
  inputs[1]->shape_[1] = 6;
  inputs[1]->shape_[2] = 7;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = nullptr;
  CropParameter *parameter = new (std::nothrow) CropParameter;
  if (parameter == nullptr) {
    return;
  }
  int ret = CropInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_NULL_PTR);
  ASSERT_EQ(inputs[0]->shape_size_, 2);
  ASSERT_EQ(inputs[0]->shape_[0], 4);
  ASSERT_EQ(inputs[0]->shape_[1], 3);
  ASSERT_EQ(inputs[0]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(inputs[0]->format_, Format_NHWC);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    if (inputs[i] != nullptr) {
      delete inputs[i];
    }
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i] != nullptr) {
      delete outputs[i];
    }
  }
}

TEST_F(CropInferTest, CropInferTest2) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new (std::nothrow) TensorC;
  if (inputs[0] == nullptr) {
    return;
  }
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = nullptr;

  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new (std::nothrow) TensorC;
  if (outputs[0] == nullptr) {
    return;
  }
  CropParameter *parameter = new (std::nothrow) CropParameter;
  if (parameter == nullptr) {
    return;
  }
  int ret = CropInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_NULL_PTR);
  ASSERT_EQ(inputs[0]->shape_size_, 2);
  ASSERT_EQ(inputs[0]->shape_[0], 4);
  ASSERT_EQ(inputs[0]->shape_[1], 3);
  ASSERT_EQ(inputs[0]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(inputs[0]->format_, Format_NHWC);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    if (inputs[i] != nullptr) {
      delete inputs[i];
    }
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i] != nullptr) {
      delete outputs[i];
    }
  }
}

TEST_F(CropInferTest, CropInferTest6) {
  size_t inputs_size = 3;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new (std::nothrow) TensorC;
  if (inputs[0] == nullptr) {
    return;
  }
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new (std::nothrow) TensorC;
  if (inputs[1] == nullptr) {
    return;
  }
  inputs[1]->shape_size_ = 3;
  inputs[1]->shape_[0] = 5;
  inputs[1]->shape_[1] = 6;
  inputs[1]->shape_[2] = 7;

  inputs[2] = new (std::nothrow) TensorC;
  if (inputs[2] == nullptr) {
    return;
  }
  inputs[2]->shape_size_ = 1;
  inputs[2]->shape_[0] = 10;
  inputs[2]->data_type_ = kNumberTypeInt32;
  inputs[2]->format_ = Format_NHWC;

  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new (std::nothrow) TensorC;
  if (outputs[0] == nullptr) {
    return;
  }
  CropParameter *parameter = new (std::nothrow) CropParameter;
  if (parameter == nullptr) {
    return;
  }
  int ret = CropInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_INPUT_TENSOR_ERROR);
  ASSERT_EQ(inputs[0]->shape_size_, 2);
  ASSERT_EQ(inputs[0]->shape_[0], 4);
  ASSERT_EQ(inputs[0]->shape_[1], 3);
  ASSERT_EQ(inputs[0]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(inputs[0]->format_, Format_NHWC);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    if (inputs[i] != nullptr) {
      delete inputs[i];
    }
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i] != nullptr) {
      delete outputs[i];
    }
  }
}

TEST_F(CropInferTest, CropInferTest4) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new (std::nothrow) TensorC;
  if (inputs[0] == nullptr) {
    return;
  }
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new (std::nothrow) TensorC;
  if (inputs[1] == nullptr) {
    return;
  }
  inputs[1]->shape_size_ = 3;
  inputs[1]->shape_[0] = 5;
  inputs[1]->shape_[1] = 6;
  inputs[1]->shape_[2] = 7;

  std::vector<TensorC *> outputs(2, NULL);
  outputs[0] = new (std::nothrow) TensorC;
  if (outputs[0] == nullptr) {
    return;
  }
  outputs[1] = new (std::nothrow) TensorC;
  if (outputs[1] == nullptr) {
    return;
  }
  CropParameter *parameter = new (std::nothrow) CropParameter;
  if (parameter == nullptr) {
    return;
  }
  int ret = CropInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_INPUT_TENSOR_ERROR);
  ASSERT_EQ(inputs[0]->shape_size_, 2);
  ASSERT_EQ(inputs[0]->shape_[0], 4);
  ASSERT_EQ(inputs[0]->shape_[1], 3);
  ASSERT_EQ(inputs[0]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(inputs[0]->format_, Format_NHWC);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    if (inputs[i] != nullptr) {
      delete inputs[i];
    }
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i] != nullptr) {
      delete outputs[i];
    }
  }
}

TEST_F(CropInferTest, CropInferTest5) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new (std::nothrow) TensorC;
  if (inputs[0] == nullptr) {
    return;
  }
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 4;
  inputs[0]->shape_[1] = 3;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new (std::nothrow) TensorC;
  if (inputs[1] == nullptr) {
    return;
  }
  inputs[1]->shape_size_ = 3;
  inputs[1]->shape_[0] = 5;
  inputs[1]->shape_[1] = 6;
  inputs[1]->shape_[2] = 7;

  std::vector<TensorC *> outputs(2, NULL);
  outputs[0] = new (std::nothrow) TensorC;
  if (outputs[0] == nullptr) {
    return;
  }
  outputs[1] = new (std::nothrow) TensorC;
  if (outputs[1] == nullptr) {
    return;
  }
  CropParameter *parameter = new (std::nothrow) CropParameter;
  if (parameter == nullptr) {
    return;
  }
  parameter->axis_ = -5;
  int ret = CropInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_ERR);
  ASSERT_EQ(inputs[0]->shape_size_, 2);
  ASSERT_EQ(inputs[0]->shape_[0], 4);
  ASSERT_EQ(inputs[0]->shape_[1], 3);
  ASSERT_EQ(inputs[0]->data_type_, kNumberTypeInt32);
  ASSERT_EQ(inputs[0]->format_, Format_NHWC);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    if (inputs[i] != nullptr) {
      delete inputs[i];
    }
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i] != nullptr) {
      delete outputs[i];
    }
  }
}
}  // namespace mindspore
