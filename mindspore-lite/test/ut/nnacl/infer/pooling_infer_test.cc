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
#include "nnacl_c/infer/pooling_infer.h"

namespace mindspore {

class PoolingInferTest : public mindspore::CommonTest {
 public:
  PoolingInferTest() {}
};

TEST_F(PoolingInferTest, PoolingInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC();
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 21;
  inputs[0]->shape_[1] = 58;
  inputs[0]->shape_[2] = 58;
  inputs[0]->shape_[3] = 3;
  // pooling infer only accepts NHWC (or NWC) input format
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC();
  PoolingParameter *parameter = new PoolingParameter();
  parameter->window_w_ = 2;
  parameter->window_h_ = 2;
  parameter->stride_w_ = 2;
  parameter->stride_h_ = 2;
  parameter->pad_mode_ = Pad_pad;
  parameter->pad_u_ = 0;
  parameter->pad_d_ = 0;
  parameter->pad_r_ = 0;
  parameter->pad_l_ = 0;
  parameter->global_ = false;
  parameter->round_type_ = RoundType_Ceil;
  int ret = PoolingInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 21);
  ASSERT_EQ(outputs[0]->shape_[1], 29);
  ASSERT_EQ(outputs[0]->shape_[2], 29);
  ASSERT_EQ(outputs[0]->shape_[3], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(PoolingInferTest, PoolingInferTest1) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC();
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 21;
  inputs[0]->shape_[1] = 14;
  inputs[0]->shape_[2] = 14;
  inputs[0]->shape_[3] = 3;
  // pooling infer only accepts NHWC (or NWC) input format
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC();
  PoolingParameter *parameter = new PoolingParameter();
  parameter->window_w_ = 3;
  parameter->window_h_ = 3;
  parameter->stride_w_ = 1;
  parameter->stride_h_ = 1;
  parameter->pad_mode_ = Pad_pad;
  parameter->pad_u_ = 0;
  parameter->pad_d_ = 0;
  parameter->pad_r_ = 0;
  parameter->pad_l_ = 0;
  parameter->global_ = false;
  parameter->pad_mode_ = Pad_same;
  parameter->round_type_ = RoundType_Ceil;
  int ret = PoolingInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 21);
  ASSERT_EQ(outputs[0]->shape_[1], 14);
  ASSERT_EQ(outputs[0]->shape_[2], 14);
  ASSERT_EQ(outputs[0]->shape_[3], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(PoolingInferTest, PoolingInferTest2) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC();
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 21;
  inputs[0]->shape_[1] = 60;
  inputs[0]->shape_[2] = 60;
  inputs[0]->shape_[3] = 3;
  // pooling infer only accepts NHWC (or NWC) input format
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC();
  PoolingParameter *parameter = new PoolingParameter();
  parameter->window_w_ = 3;
  parameter->window_h_ = 3;
  parameter->stride_w_ = 2;
  parameter->stride_h_ = 2;
  parameter->pad_mode_ = Pad_pad;
  parameter->pad_u_ = 0;
  parameter->pad_d_ = 0;
  parameter->pad_r_ = 0;
  parameter->pad_l_ = 0;
  parameter->global_ = false;
  parameter->pad_mode_ = Pad_valid;
  parameter->round_type_ = RoundType_Floor;
  int ret = PoolingInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 21);
  ASSERT_EQ(outputs[0]->shape_[1], 29);
  ASSERT_EQ(outputs[0]->shape_[2], 29);
  ASSERT_EQ(outputs[0]->shape_[3], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(PoolingInferTest, PoolingInferTest3) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC();
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 21;
  inputs[0]->shape_[1] = 7;
  inputs[0]->shape_[2] = 7;
  inputs[0]->shape_[3] = 3;
  // pooling infer only accepts NHWC (or NWC) input format
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC();
  PoolingParameter *parameter = new PoolingParameter();
  parameter->window_w_ = 7;
  parameter->window_h_ = 7;
  parameter->stride_w_ = 1;
  parameter->stride_h_ = 1;
  parameter->pad_mode_ = Pad_pad;
  parameter->pad_u_ = 0;
  parameter->pad_d_ = 0;
  parameter->pad_r_ = 0;
  parameter->pad_l_ = 0;
  parameter->global_ = false;
  parameter->pad_mode_ = Pad_valid;
  parameter->round_type_ = RoundType_Floor;
  int ret = PoolingInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 21);
  ASSERT_EQ(outputs[0]->shape_[1], 1);
  ASSERT_EQ(outputs[0]->shape_[2], 1);
  ASSERT_EQ(outputs[0]->shape_[3], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(PoolingInferTest, PoolingInferTest4) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC();
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 21;
  inputs[0]->shape_[1] = 31;
  inputs[0]->shape_[2] = 31;
  inputs[0]->shape_[3] = 3;
  // pooling infer only accepts NHWC (or NWC) input format
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC();
  PoolingParameter *parameter = new PoolingParameter();
  parameter->window_w_ = 2;
  parameter->window_h_ = 2;
  parameter->stride_w_ = 2;
  parameter->stride_h_ = 2;
  parameter->pad_mode_ = Pad_pad;
  parameter->pad_u_ = 0;
  parameter->pad_d_ = 0;
  parameter->pad_r_ = 0;
  parameter->pad_l_ = 0;
  parameter->global_ = false;
  parameter->pad_mode_ = Pad_pad;
  parameter->round_type_ = RoundType_Ceil;
  int ret = PoolingInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 21);
  ASSERT_EQ(outputs[0]->shape_[1], 16);
  ASSERT_EQ(outputs[0]->shape_[2], 16);
  ASSERT_EQ(outputs[0]->shape_[3], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(PoolingInferTest, PoolingInferTest5) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC();
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 21;
  inputs[0]->shape_[1] = 16;
  inputs[0]->shape_[2] = 16;
  inputs[0]->shape_[3] = 3;
  // pooling infer only accepts NHWC (or NWC) input format
  inputs[0]->format_ = Format_NHWC;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC();
  PoolingParameter *parameter = new PoolingParameter();
  parameter->window_w_ = 2;
  parameter->window_h_ = 2;
  parameter->stride_w_ = 2;
  parameter->stride_h_ = 2;
  parameter->pad_mode_ = Pad_pad;
  parameter->pad_u_ = 0;
  parameter->pad_d_ = 0;
  parameter->pad_r_ = 0;
  parameter->pad_l_ = 0;
  parameter->global_ = false;
  parameter->pad_mode_ = Pad_pad;
  parameter->round_type_ = RoundType_Ceil;
  int ret = PoolingInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 4);
  ASSERT_EQ(outputs[0]->shape_[0], 21);
  ASSERT_EQ(outputs[0]->shape_[1], 8);
  ASSERT_EQ(outputs[0]->shape_[2], 8);
  ASSERT_EQ(outputs[0]->shape_[3], 3);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

// 3D (1D pooling) NCW input [1,2,68] with the schema-default NCHW label: must be
// accepted, and the channel dim takes the H slot with window/stride = 1 on H.
// Guards the fix for ONNX MaxPool/AvgPool over [1,2,68] k5 s5 (converter abort
// "Unexpected input format 0" -> InferSubgraph ret -500).
TEST_F(PoolingInferTest, PoolingInfer3D_accepts_default_NCHW_label) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC();
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 1;
  inputs[0]->shape_[1] = 2;
  inputs[0]->shape_[2] = 68;
  inputs[0]->format_ = Format_NCHW;  // exporters never label intermediate tensors
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC();
  PoolingParameter *parameter = new PoolingParameter();
  parameter->window_w_ = 5;
  parameter->window_h_ = 1;
  parameter->stride_w_ = 5;
  parameter->stride_h_ = 1;
  parameter->pad_mode_ = Pad_valid;
  parameter->pad_u_ = 0;
  parameter->pad_d_ = 0;
  parameter->pad_r_ = 0;
  parameter->pad_l_ = 0;
  parameter->global_ = false;
  parameter->round_type_ = RoundType_Floor;
  int ret = PoolingInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  ASSERT_EQ(outputs[0]->shape_[2], 13);
  ASSERT_EQ(outputs[0]->format_, Format_NCHW);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

// 3D global pooling reduces only W per channel: H slot (=C) is kept with window 1,
// output [N, C, 1].
TEST_F(PoolingInferTest, PoolingInfer3D_global_reduces_W_only) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC();
  inputs[0]->shape_size_ = 3;
  inputs[0]->shape_[0] = 1;
  inputs[0]->shape_[1] = 2;
  inputs[0]->shape_[2] = 68;
  inputs[0]->format_ = Format_NCHW;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC();
  PoolingParameter *parameter = new PoolingParameter();
  parameter->window_w_ = 1;
  parameter->window_h_ = 1;
  parameter->stride_w_ = 1;
  parameter->stride_h_ = 1;
  parameter->pad_mode_ = Pad_valid;
  parameter->pad_u_ = 0;
  parameter->pad_d_ = 0;
  parameter->pad_r_ = 0;
  parameter->pad_l_ = 0;
  parameter->global_ = true;
  parameter->round_type_ = RoundType_Floor;
  int ret = PoolingInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_OK);
  ASSERT_EQ(outputs[0]->shape_size_, 3);
  ASSERT_EQ(outputs[0]->shape_[0], 1);
  ASSERT_EQ(outputs[0]->shape_[1], 2);
  ASSERT_EQ(outputs[0]->shape_[2], 1);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

// The 3D relaxation must not leak into 4D: a 4D tensor with the default NCHW
// label is still rejected (Format_NCHW == 0, the schema default).
TEST_F(PoolingInferTest, PoolingInfer4D_still_rejects_NCHW_label) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC();
  inputs[0]->shape_size_ = 4;
  inputs[0]->shape_[0] = 1;
  inputs[0]->shape_[1] = 16;
  inputs[0]->shape_[2] = 25;
  inputs[0]->shape_[3] = 24;
  inputs[0]->format_ = Format_NCHW;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC();
  PoolingParameter *parameter = new PoolingParameter();
  parameter->window_w_ = 2;
  parameter->window_h_ = 2;
  parameter->stride_w_ = 2;
  parameter->stride_h_ = 2;
  parameter->pad_mode_ = Pad_valid;
  parameter->pad_u_ = 0;
  parameter->pad_d_ = 0;
  parameter->pad_r_ = 0;
  parameter->pad_l_ = 0;
  parameter->global_ = false;
  parameter->round_type_ = RoundType_Floor;
  int ret = PoolingInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                              reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_FORMAT_ERROR);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}
}  // namespace mindspore
