/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "nnacl_c/base/crop_base.h"
#include "nnacl_c/errorcode.h"

int CropPadOffset(int input_dim, CropParameter *crop_para, int64_t *in_offset) {
  // in_offset is sized COMM_SHAPE_SIZE by every caller; larger dims would
  // write past the end of the caller's array.
  NNACL_CHECK_TRUE_RET(input_dim > 0 && input_dim <= COMM_SHAPE_SIZE, NNACL_ERR);
  int64_t axis = crop_para->axis_;
  int offsets_size = crop_para->offset_size_;
  if (axis < 0) {
    axis += input_dim;
  }
  if (axis < 0 || axis > input_dim) {
    return NNACL_ERR;
  }
  if (offsets_size > 1) {
    NNACL_CHECK_TRUE_RET(axis + offsets_size == input_dim, NNACL_ERR);
  }
  for (int i = 0; i < input_dim; i++) {
    int64_t crop_offset = 0;
    if (i >= axis) {
      if (offsets_size == 1) {
        crop_offset = crop_para->offset_[0];
      } else if (offsets_size > 1) {
        if (i - axis < CROP_OFFSET_MAX_SIZE) {
          crop_offset = crop_para->offset_[i - axis];
        }
      }
    }
    if (crop_offset < 0) {
      return NNACL_ERR;
    }
    in_offset[i] = crop_offset;
  }
  return NNACL_OK;
}

int CropCheckBounds(const int64_t *in_offset, const int *in_shape, const int *out_shape, int dim) {
  NNACL_CHECK_NULL_RETURN_ERR(in_offset);
  NNACL_CHECK_NULL_RETURN_ERR(in_shape);
  NNACL_CHECK_NULL_RETURN_ERR(out_shape);
  for (int i = 0; i < dim; i++) {
    if (in_offset[i] < 0 || out_shape[i] < 0 || in_offset[i] > (int64_t)in_shape[i] - (int64_t)out_shape[i]) {
      return NNACL_ERR;
    }
  }
  return NNACL_OK;
}
