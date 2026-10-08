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
#include "nnacl_c/base/crop_base.h"

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
  // axis_ must be initialized: CropInferShape validates it against inputs[0]
  // and rejects out-of-range axes, so leaving it uninitialized makes this test
  // depend on heap garbage (fails intermittently across builds).
  CropParameter *parameter = new (std::nothrow) CropParameter();
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

  // the infer accepts exactly one output; the out-of-range axis is what must be rejected
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new (std::nothrow) TensorC;
  if (outputs[0] == nullptr) {
    return;
  }
  CropParameter *parameter = new (std::nothrow) CropParameter();
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
TEST_F(CropInferTest, CropInferShape_rejects_negative_offsets) {
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
  inputs[1]->shape_size_ = 2;
  inputs[1]->shape_[0] = 2;
  inputs[1]->shape_[1] = 2;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new (std::nothrow) TensorC;
  if (outputs[0] == nullptr) {
    return;
  }
  CropParameter *parameter = new (std::nothrow) CropParameter();
  if (parameter == nullptr) {
    return;
  }
  parameter->axis_ = 0;
  parameter->offset_size_ = 1;
  parameter->offset_[0] = -1;
  int ret = CropInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_ERR);
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

TEST_F(CropInferTest, CropInferShape_rejects_offsets_dim_mismatch) {
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
  inputs[1]->shape_size_ = 2;
  inputs[1]->shape_[0] = 2;
  inputs[1]->shape_[1] = 2;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new (std::nothrow) TensorC;
  if (outputs[0] == nullptr) {
    return;
  }
  CropParameter *parameter = new (std::nothrow) CropParameter();
  if (parameter == nullptr) {
    return;
  }
  parameter->axis_ = 1;
  parameter->offset_size_ = 2;
  parameter->offset_[0] = 0;
  parameter->offset_[1] = 0;
  int ret = CropInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_ERR);
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

TEST_F(CropInferTest, CropPadOffset_rejects_negative_offsets) {
  CropParameter parameter;
  parameter.axis_ = 0;
  parameter.offset_size_ = 2;
  parameter.offset_[0] = -1;
  parameter.offset_[1] = 0;
  int64_t in_offset[2] = {0, 0};
  int ret = CropPadOffset(2, &parameter, in_offset);
  ASSERT_EQ(ret, NNACL_ERR);
}

TEST_F(CropInferTest, CropPadOffset_rejects_negative_axis) {
  CropParameter parameter;
  parameter.axis_ = -1;
  parameter.offset_size_ = 1;
  parameter.offset_[0] = 7;
  int64_t in_offset[4] = {0, 0, 0, 0};
  int ret = CropPadOffset(4, &parameter, in_offset);
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(in_offset[0], 0);
  ASSERT_EQ(in_offset[1], 0);
  ASSERT_EQ(in_offset[2], 0);
  ASSERT_EQ(in_offset[3], 7);
}

TEST_F(CropInferTest, CropPadOffset_keeps_valid_offsets) {
  CropParameter parameter;
  parameter.axis_ = 0;
  parameter.offset_size_ = 2;
  parameter.offset_[0] = 1;
  parameter.offset_[1] = 0;
  int64_t in_offset[2] = {0, 0};
  int ret = CropPadOffset(2, &parameter, in_offset);
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(in_offset[0], 1);
  ASSERT_EQ(in_offset[1], 0);
}

// A model may declare an output extent larger than the input provides; reading
// crop rows would run past the end of the input buffer.
TEST_F(CropInferTest, CropCheckBounds_rejects_out_larger_than_in) {
  int64_t in_offset[2] = {0, 0};
  int in_shape[2] = {2, 3};
  int out_shape[2] = {4, 3};
  int ret = CropCheckBounds(in_offset, in_shape, out_shape, 2);
  ASSERT_EQ(ret, NNACL_ERR);
}

TEST_F(CropInferTest, CropCheckBounds_rejects_offset_plus_out_beyond_in) {
  int64_t in_offset[2] = {1, 0};
  int in_shape[2] = {2, 3};
  int out_shape[2] = {2, 3};
  int ret = CropCheckBounds(in_offset, in_shape, out_shape, 2);
  ASSERT_EQ(ret, NNACL_ERR);
}

TEST_F(CropInferTest, CropCheckBounds_accepts_inbound_crop) {
  int64_t in_offset[2] = {0, 1};
  int in_shape[2] = {2, 3};
  int out_shape[2] = {2, 2};
  int ret = CropCheckBounds(in_offset, in_shape, out_shape, 2);
  ASSERT_EQ(ret, NNACL_OK);
}

// CropStruct::in_offset_ holds COMM_SHAPE_SIZE(4) entries; a 5D+ model would
// make the caller write (and CropCheckBounds read) past the end of that array.
TEST_F(CropInferTest, CropPadOffset_rejects_oversized_dim) {
  CropParameter parameter;
  parameter.axis_ = 0;
  parameter.offset_size_ = 1;
  parameter.offset_[0] = 0;
  int64_t in_offset[8] = {0};
  int ret = CropPadOffset(5, &parameter, in_offset);
  ASSERT_EQ(ret, NNACL_ERR);
}

TEST_F(CropInferTest, CropInferShape_rejects_oversized_offsets) {
  size_t inputs_size = 2;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new (std::nothrow) TensorC;
  if (inputs[0] == nullptr) {
    return;
  }
  inputs[0]->shape_size_ = 5;
  inputs[0]->shape_[0] = 2;
  inputs[0]->shape_[1] = 2;
  inputs[0]->shape_[2] = 2;
  inputs[0]->shape_[3] = 2;
  inputs[0]->shape_[4] = 2;
  inputs[0]->data_type_ = kNumberTypeInt32;
  inputs[0]->format_ = Format_NHWC;
  inputs[1] = new (std::nothrow) TensorC;
  if (inputs[1] == nullptr) {
    return;
  }
  inputs[1]->shape_size_ = 5;
  inputs[1]->shape_[0] = 2;
  inputs[1]->shape_[1] = 2;
  inputs[1]->shape_[2] = 2;
  inputs[1]->shape_[3] = 2;
  inputs[1]->shape_[4] = 2;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new (std::nothrow) TensorC;
  if (outputs[0] == nullptr) {
    return;
  }
  // Populate bounds offset_size_ to COMM_SHAPE_SIZE, so a param claiming 5
  // offsets can only come from a hostile source writing raw bytes. offset_
  // physically holds 4 entries; the extra zeroed slot after the struct keeps
  // the pre-fix read of offset_[4] deterministic instead of reading heap noise.
  unsigned char storage[sizeof(CropParameter) + sizeof(int64_t)] = {0};
  CropParameter *parameter = new (storage) CropParameter();
  parameter->axis_ = 0;
  parameter->offset_size_ = COMM_SHAPE_SIZE + 1;
  int ret = CropInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                           reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_ERR);
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
