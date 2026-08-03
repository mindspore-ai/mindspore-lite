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
#include "nnacl_c/int8/trilu_int8.h"

namespace mindspore {
class TriuTrilInt8Test : public ::testing::Test {
 public:
  TriuTrilInt8Test() {}
};

// Why the int8 variant exists: upstream TriuByte1 writes literal 0 for masked elements, which dequantizes to
// -zp*scale != 0 when out_zp != 0 and collapses accuracy. The custom TriuInt8/TrilInt8 requantize kept elements
// (input -> output) and write out_zp for masked elements. The asymmetric cases below specifically verify this fix.
static float accuracy_threshold = 0.99f;

static float get_cosine_similarity_int8(const int8_t *arr1, const int8_t *arr2, size_t cmp_size) {
  if (arr1 == nullptr || arr2 == nullptr || cmp_size == 0) {
    return 0.0f;
  }
  float dot_product = 0.0f;
  float norm1 = 0.0f;
  float norm2 = 0.0f;
  for (size_t i = 0; i < cmp_size; ++i) {
    dot_product += static_cast<float>(arr1[i]) * static_cast<float>(arr2[i]);
    norm1 += static_cast<float>(arr1[i]) * static_cast<float>(arr1[i]);
    norm2 += static_cast<float>(arr2[i]) * static_cast<float>(arr2[i]);
  }
  norm1 = std::sqrt(norm1);
  norm2 = std::sqrt(norm2);
  float norms_product = norm1 * norm2;
  const float FLOAT_EPS = 1e-6f;
  if (std::fabs(norms_product) < FLOAT_EPS) {
    return 0.0f;
  }
  return dot_product / norms_product;
}

// Build an element-wise increasing int8 input over the [-6, 6] mixed-sign range via symmetric quant (scale=0.1, zp=0).
static std::vector<int8_t> make_int8_input(const std::vector<int> &shape, float scale, int zp) {
  size_t total = 1;
  for (int d : shape) {
    total *= static_cast<size_t>(d);
  }
  std::vector<int8_t> data(total);
  for (size_t i = 0; i < total; ++i) {
    float fv = static_cast<float>(-6.0 + 12.0 * i / (total > 1 ? total - 1 : 1));
    int q = static_cast<int>(std::round(fv / scale)) + zp;
    q = q > 127 ? 127 : q;
    q = q < -128 ? -128 : q;
    data[i] = static_cast<int8_t>(q);
  }
  return data;
}

// Independent reference path: input int8 -> dequantize to float -> keep/mask by the triangle relation
// (masked = 0.0) -> requantize back to int8. This is a different rounding path from the function under test
// (which does int8->int8 ratio requantize), so the cosine drops below 1 when they diverge, giving the test
// real discriminating power (it catches masked-written-as-literal-0, wrong scale/zp, or flipped direction).
static std::vector<int8_t> reference_int8(const std::vector<int8_t> &input, const std::vector<int> &shape, int k,
                                          bool upper, float in_scale, int in_zp, float out_scale, int out_zp) {
  size_t total = input.size();
  const int height = shape[shape.size() - 2];
  const int width = shape[shape.size() - 1];
  int num = 1;
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    num *= shape[i];
  }
  std::vector<int8_t> out(total, static_cast<int8_t>(out_zp));
  for (int m = 0; m < num; ++m) {
    const int plane = m * height * width;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        bool keep = upper ? (h + k <= w) : (h + k >= w);
        int idx = plane + h * width + w;
        if (keep) {
          // float dequantize -> requantize, independent of the integer path under test
          float fv = in_scale * (static_cast<int>(input[idx]) - in_zp);
          int q = static_cast<int>(std::round(fv / out_scale)) + out_zp;
          q = q > 127 ? 127 : q;
          q = q < -128 ? -128 : q;
          out[idx] = static_cast<int8_t>(q);
        } else {
          out[idx] = static_cast<int8_t>(out_zp);
        }
      }
    }
  }
  return out;
}

// TC-008: [4,8] upper k=0 symmetric quant (scale=0.1, zp=0) — int8 baseline, non-square M!=N.
TEST_F(TriuTrilInt8Test, TriuInt8_4x8_k0) {
  std::vector<int> shape = {4, 8};
  const float in_scale = 0.1f;
  const int in_zp = 0;
  const float out_scale = 0.1f;
  const int out_zp = 0;
  auto input = make_int8_input(shape, in_scale, in_zp);
  const int height = shape[0];
  const int width = shape[1];
  const int num = 1;
  std::vector<int8_t> output(input.size(), 0);
  TriuInt8(input.data(), height, width, 0, output.data(), num, in_scale, in_zp, out_scale, out_zp);

  // Reference output: independent float dequantize -> triangle -> requantize path.
  auto benchmark = reference_int8(input, shape, 0, true, in_scale, in_zp, out_scale, out_zp);
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// TC-009: [8,8] lower k=-1 asymmetric quant — k<0 boundary + out_zp != 0, verifies the masked->out_zp fix.
// Uses an 8x8 square rather than a [4,8] non-square: lower k=-1 on 8x8 keeps 28/64 (~44%), avoiding the [4,8] k=-2
// case where kept is only 9% and gets drowned by the masked identical terms, inflating the cosine.
TEST_F(TriuTrilInt8Test, TrilInt8_8x8_kn1_asymmetric) {
  std::vector<int> shape = {8, 8};
  const float in_scale = 0.1f;
  const int in_zp = 0;
  const float out_scale = 0.05f;  // differs from in_scale -> kept elements must requantize (0.05 avoids large-area
                                  // saturation, keeping discriminating power)
  const int out_zp = -128;        // asymmetric zp; writing literal 0 for masked would collapse
  auto input = make_int8_input(shape, in_scale, in_zp);
  const int height = shape[0];
  const int width = shape[1];
  const int num = 1;
  std::vector<int8_t> output(input.size(), 0);
  TrilInt8(input.data(), height, width, -1, output.data(), num, in_scale, in_zp, out_scale, out_zp);

  // Reference output: independent float dequantize -> triangle -> requantize path.
  auto benchmark = reference_int8(input, shape, -1, false, in_scale, in_zp, out_scale, out_zp);
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// TC-010: [2,8,8] upper k=1 asymmetric quant — 3D batch num=2 + out_zp != 0, combining the batch and asymmetric
// verification points.
TEST_F(TriuTrilInt8Test, TriuInt8_3D_k1_asymmetric) {
  std::vector<int> shape = {2, 8, 8};
  const float in_scale = 0.1f;
  const int in_zp = 0;
  const float out_scale = 0.05f;  // same as above, avoids saturation drowning out the signal
  const int out_zp = -128;
  auto input = make_int8_input(shape, in_scale, in_zp);
  const int height = shape[shape.size() - 2];
  const int width = shape[shape.size() - 1];
  int num = 1;
  for (size_t i = 0; i < shape.size() - 2; ++i) {
    num *= shape[i];
  }
  std::vector<int8_t> output(input.size(), 0);
  TriuInt8(input.data(), height, width, 1, output.data(), num, in_scale, in_zp, out_scale, out_zp);

  // Reference output: independent float dequantize -> triangle -> requantize path.
  auto benchmark = reference_int8(input, shape, 1, true, in_scale, in_zp, out_scale, out_zp);
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
