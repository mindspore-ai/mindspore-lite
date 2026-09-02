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
#include "common/common_test.h"
#include "nnacl_c/infer/strided_slice_grad_infer.h"

namespace mindspore {

class StridedSliceGradInferTest : public mindspore::CommonTest {
 public:
  StridedSliceGradInferTest() {}
};

// Test: out-of-range begins are zeroed instead of failing the infer
TEST_F(StridedSliceGradInferTest, InShapeRecalculate) {
  // input order: dy, shapex, begins, ends, strides
  TensorC input_tensors[5] = {};
  input_tensors[0].shape_size_ = 2;
  input_tensors[0].shape_[0] = 5;
  input_tensors[0].shape_[1] = 10;
  input_tensors[0].data_type_ = kNumberTypeFloat32;

  std::vector<int> shape_x = {5, 10};
  input_tensors[1].shape_size_ = 1;
  input_tensors[1].shape_[0] = 2;
  input_tensors[1].data_ = shape_x.data();
  input_tensors[1].data_type_ = kNumberTypeInt32;

  std::vector<int> begins = {8, 3};  // begins[0] > shapex[0] (5), must be clamped
  input_tensors[2].shape_size_ = 1;
  input_tensors[2].shape_[0] = 2;
  input_tensors[2].data_ = begins.data();
  input_tensors[2].data_type_ = kNumberTypeInt32;

  std::vector<int> ends = {10, 8};
  input_tensors[3].shape_size_ = 1;
  input_tensors[3].shape_[0] = 2;
  input_tensors[3].data_ = ends.data();
  input_tensors[3].data_type_ = kNumberTypeInt32;

  std::vector<int> strides = {1, 1};
  input_tensors[4].shape_size_ = 1;
  input_tensors[4].shape_[0] = 2;
  input_tensors[4].data_ = strides.data();
  input_tensors[4].data_type_ = kNumberTypeInt32;

  const TensorC *inputs[] = {&input_tensors[0], &input_tensors[1], &input_tensors[2], &input_tensors[3],
                             &input_tensors[4]};
  TensorC output_tensor = {};
  TensorC *outputs[] = {&output_tensor};
  StridedSliceParameter parameter = {};

  int ret = StridedSliceGradInferShape(inputs, 5, outputs, 1, reinterpret_cast<OpParameter *>(&parameter));

  ASSERT_EQ(ret, NNACL_OK);
  // the output shape comes from the shapex tensor
  ASSERT_EQ(output_tensor.shape_size_, 2);
  ASSERT_EQ(output_tensor.shape_[0], 5);
  ASSERT_EQ(output_tensor.shape_[1], 10);
  // begins[0]=8 exceeds shapex[0]=5, so begins/ends on that axis are zeroed
  ASSERT_EQ(parameter.begins_[0], 0);
  ASSERT_EQ(parameter.ends_[0], 0);
  // the in-range axis keeps its original slice bounds
  ASSERT_EQ(parameter.begins_[1], 3);
  ASSERT_EQ(parameter.ends_[1], 8);
  ASSERT_EQ(parameter.in_shape_[0], 5);
  ASSERT_EQ(parameter.in_shape_[1], 10);
}

}  // namespace mindspore
