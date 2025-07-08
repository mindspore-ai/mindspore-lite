/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "nnacl/infer/grid_sampler_infer.h"
#include "nnacl/infer/infer_register.h"
#include "nnacl/grid_sampler_parameter.h"

int GridSamplerInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                          OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];

  SetDataTypeFormat(output, input);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  if (input->shape_size_ != inputs[1]->shape_size_) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  if (input->shape_size_ < DIMENSION_4D) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  SetShapeTensor(output, input);
  for (size_t i = DIMENSION_2D; i < input->shape_size_; ++i) {
    output->shape_[i] = inputs[1]->shape_[i - 1];
  }
  return NNACL_OK;
}

REG_INFER(GridSampler, PrimType_Inner_GridSampler, GridSamplerInferShape)
