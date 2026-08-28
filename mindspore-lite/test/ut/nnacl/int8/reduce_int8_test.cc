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
#include <iostream>
#include <cmath>
#include <vector>
#include "gtest/gtest.h"
#include "nnacl_c/errorcode.h"
#include "nnacl_c/int8/quantize.h"
#include "nnacl_c/int8/quant_dtype_cast_int8.h"
#include "nnacl_c/int8/reduce_int8.h"

namespace mindspore {
class ReduceInt8Test : public ::testing::Test {
 public:
  ReduceInt8Test() {}
};

static float get_cosine_similarity(const float *arr1, const float *arr2, size_t size) {
  if (arr1 == nullptr || arr2 == nullptr || size == 0) {
    return 0.0;
  }
  float dot_product = 0.0f;
  float norm1 = 0.0f;
  float norm2 = 0.0f;
  for (size_t i = 0; i < size; i++) {
    dot_product += arr1[i] * arr2[i];
    norm1 += arr1[i] * arr1[i];
    norm2 += arr2[i] * arr2[i];
  }
  norm1 = std::sqrt(norm1);
  norm2 = std::sqrt(norm2);
  if (norm1 == 0 && norm2 == 0) {
    return 1.0;
  }
  return dot_product / (norm1 * norm2);
}

static const float accuracy_threshold = 0.99;

static void FillReduceProdQuantArg(ReduceQuantArg *quant, float in_scale, int32_t in_zp, float out_scale,
                                   int32_t out_zp, int axis_size) {
  quant->in_scale_ = in_scale;
  quant->in_zp_ = in_zp;
  quant->out_scale_ = out_scale;
  quant->out_zp_ = out_zp;
  double prod_multiplier = std::pow(in_scale, axis_size - 1);
  int shift = 0;
  QuantizeMultiplierSmallerThanOne(prod_multiplier, &quant->prod_multiplier_, &shift);
  quant->prod_left_shift_ = shift < 0 ? -shift : 0;
  quant->prod_right_shift_ = shift > 0 ? shift : 0;
  QuantizeMultiplierSmallerThanOne(in_scale / out_scale, &quant->in_out_multiplier_, &shift);
  quant->in_out_left_shift_ = shift < 0 ? -shift : 0;
  quant->in_out_right_shift_ = shift > 0 ? shift : 0;
}

// Testcase1: ReduceProdLastAxis along a size-3 axis of around-one values -> product ~= 1.0
TEST_F(ReduceInt8Test, ReduceProdLastAxis_AroundOne) {
  const float in_scale = 0.01f;
  const int32_t in_zp = 0;
  const float out_scale = 0.01f;
  const int32_t out_zp = 0;
  const int axis_size = 3;
  const int inner_size = 1;
  const int outer_size = 1;

  ReduceQuantArg quant = {};
  FillReduceProdQuantArg(&quant, in_scale, in_zp, out_scale, out_zp, axis_size);

  // input int32 quantized values: real 1.0 -> q = 100 (scale 0.01, zp 0)
  std::vector<int32_t> src = {100, 100, 100};
  std::vector<int8_t> dst(outer_size * inner_size, 0);
  std::vector<float> output(outer_size * inner_size, 0.0f);
  std::vector<float> benchmark = {1.0f};
  ReduceProdLastAxis(outer_size, inner_size, axis_size, src.data(), dst.data(), &quant, 0, 1);
  DoDequantizeInt8ToFp32(dst.data(), output.data(), out_scale, out_zp, outer_size * inner_size);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), outer_size * inner_size);
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: the product of quantized deltas along a size-5 axis exceeds int32 range
// (100^5 = 1e10 > INT32_MAX); the kernel must accumulate in 64-bit instead of aborting
// and leaving the output buffer unwritten (micro int8 accuracy bug, shape 2x3x4x5 axis=-1).
TEST_F(ReduceInt8Test, ReduceProdLastAxis_Int32Overflow) {
  const float in_scale = 0.01f;
  const float out_scale = 0.01f;
  const int axis_size = 5;
  const int inner_size = 1;
  const int outer_size = 2;

  ReduceQuantArg quant = {};
  FillReduceProdQuantArg(&quant, in_scale, 0, out_scale, 0, axis_size);

  // real values: (1, -1, 1, -1, 1) and (-1, -1, -1, -1, -1) -> products 1 and -1
  std::vector<int32_t> src = {100, -100, 100, -100, 100, -100, -100, -100, -100, -100};
  std::vector<int8_t> dst(outer_size, 0);
  std::vector<float> output(outer_size, 0.0f);
  std::vector<float> benchmark = {1.0f, -1.0f};
  int ret = ReduceProdLastAxis(outer_size, inner_size, axis_size, src.data(), dst.data(), &quant, 0, 1);
  ASSERT_EQ(ret, NNACL_OK);
  DoDequantizeInt8ToFp32(dst.data(), output.data(), out_scale, 0, outer_size);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), outer_size);
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: multi-axis reduce chain (TFLite NHWC shape 1x2x2x2, axes=[1,2]): the
// intermediate ReduceProdInt8 stage feeds ReduceProdLastAxis; large quantized deltas
// must survive both stages without int32 overflow.
TEST_F(ReduceInt8Test, ReduceProd_MultiAxis_Chain) {
  const float in_scale = 0.047f;
  const float out_scale = 0.2223f;
  const int inner = 4;
  const int outer = 1;

  // stage 1: reduce axis 1 (axis_size=2), output kept in input quant domain (int32)
  ReduceQuantArg quant_stage1 = {};
  FillReduceProdQuantArg(&quant_stage1, in_scale, 0, out_scale, -128, 2);
  // stage 2: reduce axis 2 (axis_size=2), writes the int8 output
  ReduceQuantArg quant_stage2 = {};
  FillReduceProdQuantArg(&quant_stage2, in_scale, 0, out_scale, -128, 2);

  // NHWC input q values, real = q * in_scale: c0 = (1, 2, 3, 4), c1 = (0.5, 2, 3, 1)
  // flat layout (h, w, c): [c0, c1, c0, c1, c0, c1, c0, c1]
  std::vector<int32_t> src = {21, 11, 43, 43, 64, 64, 85, 21};
  std::vector<int32_t> mid(outer * inner, 0);
  std::vector<int8_t> dst(2, 0);
  std::vector<float> output(2, 0.0f);
  std::vector<float> benchmark = {24.0f, 3.1f};
  int ret = ReduceProdInt8(outer, inner, 2, src.data(), mid.data(), &quant_stage1, 0, 1);
  ASSERT_EQ(ret, NNACL_OK);
  ret = ReduceProdLastAxis(outer, 2, 2, mid.data(), dst.data(), &quant_stage2, 0, 1);
  ASSERT_EQ(ret, NNACL_OK);
  DoDequantizeInt8ToFp32(dst.data(), output.data(), out_scale, -128, 2);

  float similarity = get_cosine_similarity(output.data(), benchmark.data(), 2);
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
