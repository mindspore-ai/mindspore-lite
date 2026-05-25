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

// Test: in_shape_ recalculation after begins_ adjustment (core fix verification)
TEST_F(StridedSliceGradInferTest, InShapeRecalculate) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->shape_size_ = 2;
  inputs[0]->shape_[0] = 5;
  inputs[0]->shape_[1] = 10;

  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;

  StridedSliceParameter *parameter = new StridedSliceParameter;
  // Set begins_ larger than output_shape_ to trigger adjustment
  parameter->begins_[0] = 8;  // > output_shape_[0] (5), will be clamped to 5
  parameter->begins_[1] = 3;
  parameter->ends_[0] = 10;
  parameter->ends_[1] = 8;
  parameter->strides_[0] = 1;
  parameter->strides_[1] = 1;
  parameter->num_axes_ = 2;
  parameter->in_shape_length_ = 2;

  int ret = StridedSliceGradInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                                       reinterpret_cast<OpParameter *>(parameter));

  ASSERT_EQ(ret, NNACL_OK);
  // Verify in_shape_ is correctly recalculated after begins_ adjustment
  // begins_[0] should be clamped to output_shape_[0] = 5
  // in_shape_[0] = max(0, ends_[0] - adjusted_begins_[0]) = max(0, 10 - 5) = 5
  ASSERT_EQ(parameter->in_shape_[0], 5);
  ASSERT_EQ(parameter->in_shape_[1], 5);  // ends_[1] - begins_[1] = 8 - 3 = 5

  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
