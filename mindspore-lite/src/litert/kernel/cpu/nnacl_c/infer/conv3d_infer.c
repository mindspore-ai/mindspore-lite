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
#include "nnacl_c/infer/conv3d_infer.h"
#include "nnacl_c/infer/infer_register.h"
#include "nnacl_c/tensor_c_utils.h"

int Conv3dInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
  // The InferShape of Conv3D is not implemented here, it just prevents the InferShape process from being interrupted
  // and makes the nodes shape are {}.
  return NNACL_OK;
}

REG_INFER(Conv3D, PrimType_Inner_Conv3D, Conv3dInferShape)
