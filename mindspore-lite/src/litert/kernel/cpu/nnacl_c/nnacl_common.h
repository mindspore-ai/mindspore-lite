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

#ifndef NNACL_NNACL_COMMON_H_
#define NNACL_NNACL_COMMON_H_

#include <stdint.h>
#include <limits.h>
#include "nnacl_c/errorcode.h"
#include "nnacl_c/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline size_t DataTypeCSize(TypeIdC type) {
  switch (type) {
    case kNumberTypeFloat64:
      return sizeof(double);
    case kNumberTypeFloat:
    case kNumberTypeFloat32:
      return sizeof(float);
    case kNumberTypeInt8:
      return sizeof(int8_t);
    case kNumberTypeUInt8:
      return sizeof(uint8_t);
    case kNumberTypeFloat16:
    case kNumberTypeInt16:
      return sizeof(int16_t);
    case kNumberTypeInt32:
      return sizeof(int32_t);
    case kNumberTypeInt64:
      return sizeof(int64_t);
    case kNumberTypeUInt16:
      return sizeof(uint16_t);
    case kNumberTypeUInt32:
      return sizeof(uint32_t);
    case kNumberTypeUInt64:
      return sizeof(uint64_t);
    case kNumberTypeComplex64:
      return sizeof(float) + sizeof(float);
    case kNumberTypeComplex128:
      return sizeof(double) + sizeof(double);
    case kNumberTypeBool:
      return sizeof(bool);
    case kObjectTypeString:
      return sizeof(char);
    case kObjectTypeTensorType:
      return 0;
    case kMetaTypeTypeType:
      return sizeof(int);
    default:
      return 0;
  }
}

static inline int ComputeStrideOverflowCheck(int dim, int64_t stride) {
  int stride_overflow = stride > INT_MAX || stride < INT_MIN;
  int dynamic_shape = dim == -1 || stride < 0;
  int mul_overflow = !dynamic_shape && dim != 0 && stride > INT_MAX / dim;
  if (stride_overflow || mul_overflow) {
    return NNACL_ERRCODE_MUL_OVERFLOW;
  }
  return NNACL_OK;
}

static inline int ComputeStrideCheck(const int *shape, int index, int64_t stride) {
  int dim = shape[index];
  if (dim < -1) {
    return NNACL_ERR;
  }
  return ComputeStrideOverflowCheck(dim, stride);
}

static inline int ComputeStrideByIndex(const int *shape, int *strides, int index, int64_t *stride) {
  int ret = ComputeStrideCheck(shape, index, *stride);
  if (ret != NNACL_OK) {
    return ret;
  }
  strides[index] = (int)(*stride);
  *stride *= shape[index];
  return NNACL_OK;
}

static inline int ComputeStrides(const int *shape, int *strides, const int ndim) {
  if (shape == NULL || strides == NULL || ndim < 0) {
    return NNACL_ERR;
  }
  int64_t stride = 1;
  int ret = NNACL_OK;
  for (int i = ndim - 1; i >= 0 && ret == NNACL_OK; i--) {
    ret = ComputeStrideByIndex(shape, strides, i, &stride);
  }
  return ret;
}

static inline void ComputeAxisDims(const int *shape, int shape_size, int axis, int *out_count, int *axis_count,
                                   int *in_count) {
  *out_count = 1;
  *in_count = 1;
  for (int i = 0; i < shape_size; i++) {
    if (i < axis) {
      *out_count = (*out_count) * shape[i];
    }
    if (i == axis) {
      *axis_count = shape[axis];
    }
    if (i > axis) {
      *in_count = (*in_count) * shape[i];
    }
  }
}

static const unsigned int FP32_BIT_SIZE = 32;
static const unsigned int FP32_EXPONENT_BIAS = 127;
static const unsigned int FP32_SIGNIFICAND = 23;
static const unsigned int FP32_EXPONENT_MAX = 255;
static const unsigned int FP16_BIT_SIZE = 16;
static const unsigned int FP16_EXPONENT_BIAS = 15;
static const unsigned int FP16_SIGNIFICAND = 10;
static const int FP16_EXPONENT_MAX = 30;
static const int FP16_EXPONENT_MIN = -10;
static const int FP16_SHIFT = 13;
static const int EXPONENT_BIAS_VALUE_14 = 14;
static const int EXPONENT_BIAS_VALUE_15 = 15;
static const int EXPONENT_BIAS_VALUE_30 = 30;
static const int EXPONENT_BIAS_VALUE_128 = 128;
static const int BITS_SHIFT_SIZE_7 = 7;
static const int BITS_SHIFT_SIZE_10 = 10;
static const int BITS_SHIFT_SIZE_12 = 12;
static const int BITS_SHIFT_SIZE_13 = 13;
static const int BITS_SHIFT_SIZE_15 = 15;
static const int BITS_SHIFT_SIZE_23 = 23;
static const int BITS_SHIFT_SIZE_31 = 31;
float ShortToFloat32(uint16_t src_value);
uint16_t Float32ToShort(float src_value);
float Bf16ToFloat32(uint16_t src_value);

#ifdef __cplusplus
}
#endif
#endif  // NNACL_NNACL_COMMON_H_
