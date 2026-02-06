/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "common/common_test.h"
#include "nnacl_c/int8/matmul_int8.h"

namespace mindspore {

class MatmulInt8Test : public mindspore::CommonTest {
 public:
  MatmulInt8Test() {}
};

static float get_cosine_similarity_int8(const int8_t *arr1, const int8_t *arr2, size_t cmp_size) {
  if (arr1 == nullptr || arr2 == nullptr || cmp_size == 0) {
    return 0.0f;
  }
  float dot_product = 0.0f;
  float norm1 = 0.0f;
  float norm2 = 0.0f;
  for (size_t i = 0; i < cmp_size; ++i) {
    dot_product += (float)arr1[i] * (float)arr2[i];
    norm1 += (float)arr1[i] * (float)arr1[i];
    norm2 += (float)arr2[i] * (float)arr2[i];
  }
  norm1 = std::sqrt(norm1);
  norm2 = std::sqrt(norm2);
  float norms_product = norm1 * norm2;
  const float FLOAT_EPS = 1e-6f;
  if (std::fabs(norms_product) < FLOAT_EPS) {
    return 0.0f;
  }
  float cosine_similarity = dot_product / norms_product;
  return cosine_similarity;
}

TEST_F(MatmulInt8Test, Int8) {
  std::vector<int8_t> originInputs(1 * 64, 0);
  std::vector<int32_t> weightBiasPtr_(8 * 1, 0);
  std::vector<int8_t> originOutputs(1 * 6, 0);
  std::vector<int8_t> weight = {
    -103, -25,  -77, -122, -51, -3,   -97, 49,   18,  -68,  -50,  -85, -73, -37, -70, -26,  -65,  -43, 28,   -95,
    -28,  -66,  -49, 42,   -35, 10,   5,   48,   -30, 3,    -104, -50, -73, 50,  40,  44,   -124, -34, -82,  -16,
    -85,  -63,  -5,  -83,  -84, -127, -12, -15,  -19, -84,  52,   -42, 30,  -98, 49,  -70,  -90,  -67, -122, -25,
    13,   -40,  -3,  -71,  -99, 66,   -70, -74,  -85, -99,  -58,  16,  -13, 21,  -4,  59,   -1,   51,  51,   -24,
    -102, -16,  -62, 12,   50,  -78,  51,  -26,  -44, 98,   -63,  77,  -80, 106, 50,  -124, 75,   -11, 78,   43,
    -50,  72,   107, -66,  -69, -19,  -24, 46,   -49, -69,  44,   -8,  95,  -77, 45,  -127, 63,   107, -4,   -36,
    14,   88,   34,  -81,  -27, -55,  -82, 64,   -30, 31,   -40,  87,  67,  8,   59,  -91,  83,   -39, 82,   110,
    57,   27,   19,  11,   26,  -6,   -40, -41,  -69, 60,   17,   -14, -48, 63,  93,  43,   -37,  3,   -16,  -53,
    -96,  -29,  87,  -74,  127, 106,  15,  -97,  2,   90,   52,   -65, -5,  26,  -75, 17,   -56,  -45, -106, -12,
    -24,  21,   110, -86,  -19, 10,   -46, 30,   115, 67,   -73,  105, 68,  37,  14,  96,   3,    36,  -76,  -64,
    -24,  -66,  -24, -75,  -12, -101, 35,  9,    -69, -41,  -26,  37,  25,  17,  -31, -35,  -60,  8,   70,   2,
    77,   -85,  -19, -92,  -3,  -127, -78, -61,  40,  -3,   -18,  13,  -57, -15, -55, -89,  -15,  98,  -68,  31,
    -7,   66,   64,  -53,  32,  -101, 8,   -61,  -9,  19,   76,   60,  76,  41,  48,  -39,  16,   -87, 49,   -16,
    -52,  -19,  6,   -27,  87,  85,   40,  -56,  -15, 12,   39,   29,  -23, -62, 1,   43,   40,   29,  -88,  40,
    32,   5,    -29, 6,    -91, -103, -29, 55,   81,  27,   -22,  13,  -56, -96, -30, -38,  -104, -94, -127, -70,
    44,   -32,  -17, 11,   -10, -9,   -24, 2,    35,  -127, -35,  52,  71,  -12, -16, 70,   -108, -76, 31,   -79,
    -24,  -63,  -82, 80,   5,   -108, 15,  90,   31,  -85,  45,   -68, -12, -68, 78,  -48,  -27,  87,  7,    -64,
    -74,  -74,  8,   -32,  -67, 73,   -32, -127, 88,  72,   -16,  -25, 83,  -95, -81, 85,   -105, -43, 64,   -115,
    -114, -113, -90, 80,   -33, -22,  -38, 15,   -26, -20,  58,   -88, -20, -20, -19, 82,   70,   62,  -67,  -49,
    -47,  -54,  85,  -54,
  };
  std::vector<int32_t> bias = {-320481, -37863, 63316, -70816, -105629, -133896};
  int32_t tmp_weight_zp = 1;
  CalcInputSums(originInputs.data(), 1, 64, tmp_weight_zp, weightBiasPtr_.data(), RowMajor);
  float filter_scale[6] = {0.00164039805532, 0.0015292350436, 0.0012470819056, 0.001622696756, 0.0016628912417};
  int32_t filter_zp[6] = {0, 0, 0, 0, 0, 0};
  int32_t left_shift[6] = {0, 0, 0, 0, 0, 0};
  int32_t right_shift[6] = {-9, -9, -10, -9, -9, -9};
  int32_t multiplier[6] = {1284268351, 1197238817, 1952682041, 1270409989, 1299997195, 1301878292};
  const MatmulQuantParameter matmul_quant_parameter = {{0.14809137587789, -128},
                                                       {0, 0},
                                                       {0.20798070729, 50},
                                                       -128,
                                                       127,
                                                       filter_scale,
                                                       filter_zp,
                                                       left_shift,
                                                       right_shift,
                                                       multiplier};
  int32_t *cur_left = matmul_quant_parameter.left_shift_ + 0;
  int32_t *cur_right = matmul_quant_parameter.right_shift_ + 0;
  int32_t *cur_mul = matmul_quant_parameter.quant_multiplier_ + 0;
  int32_t *cur_zp = matmul_quant_parameter.filter_zp_ + 0;
  MatmulInt8LowMemory(originInputs.data(), weight.data(), originOutputs.data(), 1, 6, 64, weightBiasPtr_.data(),
                      bias.data(), -128, 127, 50, cur_mul, cur_left, cur_right, 6, true, cur_zp, false, false);
  std::vector<int8_t> benchmark = {-128, 9, 106, -32, -75, -109};
  for (auto iter = originOutputs.begin(); iter != originOutputs.end(); iter++) {
    printf("[%d]", (int)(*iter));
  }
  float sim = get_cosine_similarity_int8(originOutputs.data(), benchmark.data(), 6);
  ASSERT_GT(sim, 0.9);
}

}  // namespace mindspore
