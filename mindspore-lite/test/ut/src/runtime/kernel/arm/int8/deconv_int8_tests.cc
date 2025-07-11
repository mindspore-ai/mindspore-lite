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

#include <iostream>
#include <memory>
#include "schema/inner/model_generated.h"
#include "common/common_test.h"
#include "src/common/file_utils.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/pack.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/int8/deconv_int8.h"
#include "src/litert/kernel/cpu/int8/deconvolution_int8.h"

using mindspore::lite::DeviceType;

namespace mindspore {
using mindspore::lite::LiteQuantParam;
using mindspore::lite::Tensor;
class TestDeconvInt8 : public mindspore::CommonTest {
 public:
  TestDeconvInt8() {}
};

TEST_F(TestDeconvInt8, PackInputTest1) {
  /* 6 x 20 */
  int8_t in[] = {40,  24,  94,  122, 67,  34,  -89, 31,  -43, 121, 48,  -54, 44,   -91,  35,  89,  -37, 114,  -8,  103,
                 -22, 32,  26,  112, -92, -23, 43,  9,   81,  118, -73, -54, 65,   -99,  51,  -90, 121, -62,  119, -93,
                 21,  -92, -1,  -82, -71, -54, 63,  -93, 92,  -93, 99,  122, -104, -16,  -8,  -32, 90,  -126, 51,  91,
                 4,   70,  -7,  116, 99,  81,  -79, 124, -14, 28,  97,  9,   -97,  99,   88,  -15, 54,  26,   77,  -25,
                 113, 119, 119, -75, -17, 7,   7,   1,   69,  66,  40,  -13, 80,   -115, -98, -8,  -17, 31,   88,  65,
                 -1,  -15, -98, 77,  56,  119, -20, -32, -54, -58, -16, 52,  121,  126,  -33, 43,  92,  -34,  -17, -52};
  int8_t co[] = {40,  24,   94,  122, 67,   34,  -89, 31,  -43, 121, 48,  -54, 44,  -91, 35,  89,  -22, 32,   26,  112,
                 -92, -23,  43,  9,   81,   118, -73, -54, 65,  -99, 51,  -90, 21,  -92, -1,  -82, -71, -54,  63,  -93,
                 92,  -93,  99,  122, -104, -16, -8,  -32, 4,   70,  -7,  116, 99,  81,  -79, 124, -14, 28,   97,  9,
                 -97, 99,   88,  -15, -37,  114, -8,  103, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,    0,   0,
                 121, -62,  119, -93, 0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   90,  -126, 51,  91,
                 0,   0,    0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   54,  26,  77,  -25, 0,   0,    0,   0,
                 0,   0,    0,   0,   0,    0,   0,   0,   113, 119, 119, -75, -17, 7,   7,   1,   69,  66,   40,  -13,
                 80,  -115, -98, -8,  -1,   -15, -98, 77,  56,  119, -20, -32, -54, -58, -16, 52,  121, 126,  -33, 43,
                 0,   0,    0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,    0,   0,
                 0,   0,    0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   -17, 31,  88,  65,  0,   0,    0,   0,
                 0,   0,    0,   0,   0,    0,   0,   0,   92,  -34, -17, -52, 0,   0,   0,   0,   0,   0,    0,   0,
                 0,   0,    0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,    0,   0,
                 0,   0,    0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0};
  int8_t dst[8 * 32] = {0};
  RowMajor2Row16x4MajorInt8(in, dst, 6, 20);
  ASSERT_EQ(0, CompareOutputData(dst, co, 8 * 32, 1));
}

TEST_F(TestDeconvInt8, InputSumTest1) {
  int8_t packed_a[] = {
    -6,  76,  32,  80,  -73, 8,   -85, -3,  114, 80,  30,  42,  15,  15,  15,  15,  -41, 117,  62,  -76, -77, -111,
    88,  105, 68,  105, -74, 13,  15,  15,  15,  15,  51,  94,  31,  -52, -92, -4,  -35, -71,  101, -93, 46,  -65,
    15,  15,  15,  15,  57,  -41, -51, 77,  1,   9,   73,  -19, -36, 57,  81,  -24, 15,  15,   15,  15,  40,  103,
    112, 109, -41, -68, 57,  61,  55,  -20, 3,   2,   15,  15,  15,  15,  17,  -16, -31, 58,   -4,  67,  -4,  -95,
    -5,  -72, 81,  15,  15,  15,  15,  15,  -7,  -16, -47, 112, 114, -26, -98, 53,  15,  -49,  26,  19,  15,  15,
    15,  15,  19,  8,   -57, -35, -79, 118, 29,  21,  37,  -48, 83,  7,   15,  15,  15,  15,   124, 113, -5,  15,
    -8,  107, -65, -88, 50,  -47, -80, -84, 15,  15,  15,  15,  3,   -45, 92,  42,  -20, -101, 106, -10, 89,  67,
    55,  10,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,   15,  15,  15,  15,
    15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15};
  int32_t filter_zp = -20;

  int32_t input_sum[12] = {0};
  int32_t correct_input_sum[] = {-7100, -4780, 580, -4880, -9460, -1420, -3120, -3260, -1840, -6960, -4800, -4800};
  DeConvPackInputSum(packed_a, input_sum, filter_zp, 12, 16, true);
  ASSERT_EQ(0, CompareOutputData(input_sum, correct_input_sum, 12, 0));

  int32_t input_sum_4[4] = {0};
  int32_t correct_input_sum_4[] = {-18400, -13160, -7340, -12940};
  DeConvPackInputSum(packed_a, input_sum_4, filter_zp, 4, 16 * 3, true);
  ASSERT_EQ(0, CompareOutputData(input_sum_4, correct_input_sum_4, 4, 0));
}

TEST_F(TestDeconvInt8, MatMulOptTest1) {
  /* 10 * 12 */
  int8_t a_src_ptr[] = {-6,  76,  32,  80,  -73, 8,    -85, -3,  114, 80,  30,  42,  /* row 1 */
                        -41, 117, 62,  -76, -77, -111, 88,  105, 68,  105, -74, 13,  /* row 2 */
                        51,  94,  31,  -52, -92, -4,   -35, -71, 101, -93, 46,  -65, /* row 3 */
                        57,  -41, -51, 77,  1,   9,    73,  -19, -36, 57,  81,  -24, /* row 4 */
                        40,  103, 112, 109, -41, -68,  57,  61,  55,  -20, 3,   2,   /* row 5 */
                        17,  -16, -31, 58,  -4,  67,   -4,  -95, -5,  -72, 81,  15,  /* row 6 */
                        -7,  -16, -47, 112, 114, -26,  -98, 53,  15,  -49, 26,  19,  /* row 7 */
                        19,  8,   -57, -35, -79, 118,  29,  21,  37,  -48, 83,  7,   /* row 8 */
                        124, 113, -5,  15,  -8,  107,  -65, -88, 50,  -47, -80, -84, /* row 9 */
                        3,   -45, 92,  42,  -20, -101, 106, -10, 89,  67,  55,  10};
  int32_t input_zp = 15;
  /* ic:12  oc:6  hw:3 */
  int8_t b_src_ptr[] = {
    92,  27,  22,   52,   -112, -20,  -57, -2,   89,  32,  93,   -66,  -25, -54,  94,   -97, -119, -98, /* ic1 */
    101, -99, 77,   -83,  76,   95,   59,  97,   8,   40,  -109, -20,  67,  -107, 37,   -6,  -54,  -20, /* ic2 */
    -30, 36,  -106, -103, -3,   -86,  -82, 59,   4,   -75, -50,  -106, 55,  104,  -117, -71, -20,  -85, /* ic3 */
    -77, 16,  -25,  -58,  4,    80,   -75, 94,   32,  -68, 2,    40,   56,  -103, 11,   -98, -70,  -69, /* ic4 */
    0,   57,  -6,   82,   66,   -112, -61, 33,   -77, -53, 95,   -38,  87,  -46,  -3,   81,  -47,  43,  /* ic5 */
    21,  26,  -45,  -57,  50,   -24,  -82, -114, 61,  46,  -53,  78,   -24, 31,   -7,   37,  29,   38,  /* ic6 */
    45,  106, 52,   -42,  31,   -6,   -61, -87,  2,   79,  -5,   -42,  43,  -106, -104, 7,   91,   -63, /* ic7 */
    58,  97,  -15,  74,   -96,  15,   -23, -3,   -47, -97, 100,  -54,  26,  -46,  35,   26,  100,  -80, /* ic8 */
    34,  -25, 96,   -67,  -80,  -27,  66,  41,   41,  -43, -43,  -38,  -4,  -64,  31,   7,   -8,   6,   /* ic9 */
    -2,  39,  -119, 53,   75,   -91,  -44, 77,   -62, 22,  -44,  78,   -67, -48,  -115, -4,  43,   81,  /* ic10 */
    40,  -20, -5,   -89,  60,   -62,  -4,  -48,  66,  -64, -69,  62,   17,  -89,  1,    87,  81,   32,  /* ic11 */
    -29, 51,  40,   27,   66,   67,   11,  -69,  85,  -79, -106, 55,   22,  -23,  62,   69,  -74,  49}; /* ic12 */
  int32_t filter_zp = -20;

  /*
   * ----------------------   pack input  ------------------------- */
  int8_t packed_a[12 * 16] = {0};
  memset(packed_a, static_cast<int8_t>(input_zp), 12 * 16);
  int8_t correct_packed_a[] = {-6,  76,  32,  80,  -73, 8,    -85, -3,  114, 80,  30,  42,  0, 0, 0, 0,  /* row 1 */
                               -41, 117, 62,  -76, -77, -111, 88,  105, 68,  105, -74, 13,  0, 0, 0, 0,  /* row 2 */
                               51,  94,  31,  -52, -92, -4,   -35, -71, 101, -93, 46,  -65, 0, 0, 0, 0,  /* row 3 */
                               57,  -41, -51, 77,  1,   9,    73,  -19, -36, 57,  81,  -24, 0, 0, 0, 0,  /* row 4 */
                               40,  103, 112, 109, -41, -68,  57,  61,  55,  -20, 3,   2,   0, 0, 0, 0,  /* row 5 */
                               17,  -16, -31, 58,  -4,  67,   -4,  -95, -5,  -72, 81,  15,  0, 0, 0, 0,  /* row 6 */
                               -7,  -16, -47, 112, 114, -26,  -98, 53,  15,  -49, 26,  19,  0, 0, 0, 0,  /* row 7 */
                               19,  8,   -57, -35, -79, 118,  29,  21,  37,  -48, 83,  7,   0, 0, 0, 0,  /* row 8 */
                               124, 113, -5,  15,  -8,  107,  -65, -88, 50,  -47, -80, -84, 0, 0, 0, 0,  /* row 8 */
                               3,   -45, 92,  42,  -20, -101, 106, -10, 89,  67,  55,  10,  0, 0, 0, 0,  /* row 8 */
                               0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0, 0, 0, 0,  /* row 8 */
                               0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0, 0, 0, 0}; /* row 8 */
  RowMajor2Row16x4MajorInt8(a_src_ptr, packed_a, 10, 12);
  ASSERT_EQ(0, CompareOutputData(packed_a, correct_packed_a, 16 * 12, 0));

  /*
   * ----------------------   pack weight  ------------------------- */
  int8_t packed_b[16 * 3 * 8] = {0};
  memset(packed_b, static_cast<int8_t>(filter_zp), 16 * 3 * 8);
  int8_t correct_packed_b[] = {
    /* col major   16 * 24 */
    92,   101,  -30,  -77,  0,    21,   45,   58,  34,  -2,   40,  -29,  -20, -20, -20, -20, /* 1 */
    27,   -99,  36,   16,   57,   26,   106,  97,  -25, 39,   -20, 51,   -20, -20, -20, -20, /* 2 */
    22,   77,   -106, -25,  -6,   -45,  52,   -15, 96,  -119, -5,  40,   -20, -20, -20, -20, /* 3 */
    52,   -83,  -103, -58,  82,   -57,  -42,  74,  -67, 53,   -89, 27,   -20, -20, -20, -20, /* 4 */
    -112, 76,   -3,   4,    66,   50,   31,   -96, -80, 75,   60,  66,   -20, -20, -20, -20, /* 5 */
    -20,  95,   -86,  80,   -112, -24,  -6,   15,  -27, -91,  -62, 67,   -20, -20, -20, -20, /* 6 */
    -20,  -20,  -20,  -20,  -20,  -20,  -20,  -20, -20, -20,  -20, -20,  -20, -20, -20, -20, /* 7 */
    -20,  -20,  -20,  -20,  -20,  -20,  -20,  -20, -20, -20,  -20, -20,  -20, -20, -20, -20, /* 8 */
    -57,  59,   -82,  -75,  -61,  -82,  -61,  -23, 66,  -44,  -4,  11,   -20, -20, -20, -20, /* 9 */
    -2,   97,   59,   94,   33,   -114, -87,  -3,  41,  77,   -48, -69,  -20, -20, -20, -20, /* 10 */
    89,   8,    4,    32,   -77,  61,   2,    -47, 41,  -62,  66,  85,   -20, -20, -20, -20, /* 11 */
    32,   40,   -75,  -68,  -53,  46,   79,   -97, -43, 22,   -64, -79,  -20, -20, -20, -20, /* 12 */
    93,   -109, -50,  2,    95,   -53,  -5,   100, -43, -44,  -69, -106, -20, -20, -20, -20, /* 13 */
    -66,  -20,  -106, 40,   -38,  78,   -42,  -54, -38, 78,   62,  55,   -20, -20, -20, -20, /* 14 */
    -20,  -20,  -20,  -20,  -20,  -20,  -20,  -20, -20, -20,  -20, -20,  -20, -20, -20, -20, /* 15 */
    -20,  -20,  -20,  -20,  -20,  -20,  -20,  -20, -20, -20,  -20, -20,  -20, -20, -20, -20, /* 16 */
    -25,  67,   55,   56,   87,   -24,  43,   26,  -4,  -67,  17,  22,   -20, -20, -20, -20, /* 17 */
    -54,  -107, 104,  -103, -46,  31,   -106, -46, -64, -48,  -89, -23,  -20, -20, -20, -20, /* 18 */
    94,   37,   -117, 11,   -3,   -7,   -104, 35,  31,  -115, 1,   62,   -20, -20, -20, -20, /* 19 */
    -97,  -6,   -71,  -98,  81,   37,   7,    26,  7,   -4,   87,  69,   -20, -20, -20, -20, /* 20 */
    -119, -54,  -20,  -70,  -47,  29,   91,   100, -8,  43,   81,  -74,  -20, -20, -20, -20, /* 21 */
    -98,  -20,  -85,  -69,  43,   38,   -63,  -80, 6,   81,   32,  49,   -20, -20, -20, -20,
    -20,  -20,  -20,  -20,  -20,  -20,  -20,  -20, -20, -20,  -20, -20,  -20, -20, -20, -20,
    -20,  -20,  -20,  -20,  -20,  -20,  -20,  -20, -20, -20,  -20, -20,  -20, -20, -20, -20};
  DeConvWeightTransInt8(b_src_ptr, packed_b, 12, 6, 3, true);
  /* kernel : 12x1x3x6   nhwc   */
  ASSERT_EQ(0, CompareOutputData(packed_b, correct_packed_b, 16 * 3 * 8, 0));

  /*
   * ----------------------   calculate input_sum   ------------------------- */
  int32_t input_sum[12] = {0};
  int32_t correct_input_sum[] = {-5900, -3580, 1780, -3680, -8260, -220, -1920, -2060, -640, -5760, 0, 0};
  DeConvPackInputSum(packed_a, input_sum, filter_zp, 12, 16, true);
  ASSERT_EQ(0, CompareOutputData(input_sum, correct_input_sum, 12, 0));

  /*
   * ----------------------   calculate weight_sum   ------------------------- */
  int32_t weight_sum[3 * 8] = {0};
  int32_t correct_weight_sum[] = {-7395, -8265, -3090, -435, -5655, -1035, 0,     0,     1695,  -4770, -6630, 300,
                                  -765,  -2835, 0,     0,    -7395, 4665,  -2475, -4170, -2880, -1110, 0,     0};
  DeConvPackWeightSum(packed_b, weight_sum, input_zp, filter_zp, 16, 24, true);
  ASSERT_EQ(0, CompareOutputData(weight_sum, correct_weight_sum, 3 * 8, 0));

  /*
   * ----------------------   do matmul   ------------------------- */
  int32_t tmp_output[12 * 24] = {0};
  int32_t correct_tmp_output[] = {
    -1624,  -19061, 1795,   -17119, /*1*/
    14706,  417,    7306,   1357,   /*2*/
    9653,   -44022, 19414,  -36187, /*3*/
    -2041,  6874,   -5766,  3072,   /*4*/
    9842,   2395,   12464,  -18826, /*5*/
    -12267, -17853, 4617,   -19468, /*6*/
    -15734, -6112,  2122,   14259,  /*7*/
    11098,  -9520,  12407,  -15239, /*8*/
    10309,  -34271, 9740,   -14607, /*9*/
    -5027,  12313,  -508,   -10808, /*10*/
    -7395,  -8265,  -3090,  -435,   /*11*/
    -7395,  -8265,  -3090,  -435,   /*12*/
    1604,   14898,  0,      0,      /*1*/
    -8212,  9471,   0,      0,      /*2*/
    -23430, 6343,   0,      0,      /*3*/
    4020,   -3740,  0,      0,      /*4*/
    -9730,  22378,  0,      0,      /*5*/
    4702,   4740,   0,      0,      /*6*/
    -7541,  5461,   0,      0,      /*7*/
    -6633,  8356,   0,      0,      /*8*/
    -16854, 9147,   0,      0,      /*9*/
    -4018,  -11524, 0,      0,      /*10*/
    -5655,  -1035,  0,      0,      /*11*/
    -5655,  -1035,  0,      0,      /*12*/
    17194,  28501,  13376,  -9359,  /*1*/
    21454,  22425,  -21049, 6603,   /*2*/
    23479,  -658,   12866,  9739,   /*3*/
    -12173, -7558,  3862,   10238,  /*4*/
    4110,   31945,  10069,  -7376,  /*5*/
    -1948,  -20322, 16439,  3260,   /*6*/
    1712,   12743,  -8132,  -27744, /*7*/
    7633,   -33916, 18755,  11300,  /*8*/
    3686,   9222,   10103,  26102,  /*9*/
    17,     13135,  785,    -6305,  /*10*/
    1695,   -4770,  -6630,  300,    /*11*/
    1695,   -4770,  -6630,  300,    /*12*/
    -27325, 14957,  0,      0,      /*1*/
    -12191, -21866, 0,      0,      /*2*/
    -21690, -18554, 0,      0,      /*3*/
    8737,   14529,  0,      0,      /*4*/
    -1774,  -19575, 0,      0,      /*5*/
    -12761, 13286,  0,      0,      /*6*/
    20523,  2488,   0,      0,      /*7*/
    -12782, 12688,  0,      0,      /*8*/
    -1194,  -10523, 0,      0,      /*9*/
    -4044,  -9671,  0,      0,      /*10*/
    -765,   -2835,  0,      0,      /*11*/
    -765,   -2835,  0,      0,      /*12*/
    -4671,  -4173,  8675,   -8560,  /*1*/
    -1597,  -4946,  -20214, -6752,  /*2*/
    -11439, 5138,   11119,  -17661, /*3*/
    -6690,  -17301, -5541,  -4356,  /*4*/
    22347,  -11778, 2389,   -22030, /*5*/
    -5176,  -242,   8786,   -994,   /*6*/
    9104,   -7208,  24117,  3724,   /*7*/
    -13648, -1840,  12265,  10347,  /*8*/
    -10325, 7184,   19374,  -29001, /*9*/
    3979,   -6704,  -23278, -8124,  /*10*/
    -7395,  4665,   -2475,  -4170,  /*11*/
    -7395,  4665,   -2475,  -4170,  /*12*/
    -9132,  8560,   0,      0,      /*1*/
    19264,  -10169, 0,      0,      /*2*/
    -15133, -13678, 0,      0,      /*3*/
    7894,   -51,    0,      0,      /*4*/
    -4775,  -29785, 0,      0,      /*5*/
    -12597, 4088,   0,      0,      /*6*/
    -17420, 1815,   0,      0,      /*7*/
    15796,  3101,   0,      0,      /*8*/
    -37969, -10818, 0,      0,      /*9*/
    12714,  -7827,  0,      0,      /*10*/
    -2880,  -1110,  0,      0,      /*11*/
    -2880,  -1110,  0,      0       /*12*/
  };

  MatMulInt8_16x4(packed_a, packed_b, tmp_output, 12, 24, 16, input_sum, weight_sum);
  ASSERT_EQ(0, CompareOutputData(tmp_output, correct_tmp_output, 12 * 3 * 8, 0));
}

int DeConvInt8TestInit1(std::vector<lite::Tensor *> *inputs_, std::vector<lite::Tensor *> *outputs_,
                        ConvParameter *conv_param, int8_t **correct) {
  /* float data from deconv fp32 testcase : DeConvTestInit2 */
  /*   vq = (vi - zp) * s     vi = vq / s + zp */
  auto *in_t = new Tensor(kNumberTypeInt8, {1, 4, 2, 3}, mindspore::NHWC, lite::Category::VAR);
  in_t->MallocData();
  int8_t in[] = {6, 43, 38, 24, -8, 12, 41, -24, -20, 41, -19, -6, -26, -6, 23, -31, 34, 45, 8, 45, -39, -27, -48, 12};
  memcpy(in_t->MutableData(), in, sizeof(int8_t) * in_t->ElementsNum());
  auto *in_quant_arg = new LiteQuantParam();
  in_quant_arg->zeroPoint = -19, in_quant_arg->scale = 0.31228156;
  in_t->AddQuantParam(*in_quant_arg);
  delete in_quant_arg;
  inputs_->push_back(in_t);

  auto *weight_t = new Tensor(kNumberTypeInt8, {3, 3, 3, 2}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  weight_t->MallocData();
  int8_t weight[] = {66, 89, 98, 74,  95, 86, 125, 95, 105, 83, 116, 94, 90, 80, 86, 59, 72, 92,
                     64, 76, 92, 80,  90, 87, 106, 55, 105, 60, 75,  53, 81, 81, 98, 81, 86, 59,
                     74, 82, 97, 105, 71, 67, 79,  87, 72,  79, 80,  76, 96, 80, 83, 71, 61, 79};
  memcpy(weight_t->MutableData(), weight, sizeof(int8_t) * weight_t->ElementsNum());
  auto *w_quant_arg = new LiteQuantParam();
  w_quant_arg->zeroPoint = 83, w_quant_arg->scale = 0.023649725490196;
  weight_t->AddQuantParam(*w_quant_arg);
  delete w_quant_arg;
  inputs_->push_back(weight_t);

  auto *out_t = new Tensor(kNumberTypeInt8, {1, 7, 3, 2}, mindspore::NHWC, lite::Category::VAR);
  out_t->MallocData();
  auto *out_quant_arg = new LiteQuantParam();
  out_quant_arg->zeroPoint = 31, out_quant_arg->scale = 0.3439215686275;
  out_t->AddQuantParam(*out_quant_arg);
  delete out_quant_arg;
  outputs_->push_back(out_t);

  *correct = reinterpret_cast<int8_t *>(malloc(out_t->ElementsNum() * sizeof(int8_t)));
  int8_t co_nchw[] = {57, 76, 49, 71,  8, 61, 57, 127, 56, 46, -11, 61, 23, 31,  34, 50, 59, 49, 78, 17, 6,
                      -3, -5, 23, -11, 6, -5, 33, 64,  30, 21, 18,  25, 21, -15, 0,  4,  31, 36, 2,  17, 43};
  PackNCHWToNHWCInt8(co_nchw, *correct, out_t->Batch(), out_t->Width() * out_t->Height(), out_t->Channel());

  conv_param->kernel_h_ = conv_param->kernel_w_ = 3;
  conv_param->pad_u_ = conv_param->pad_l_ = 1;
  conv_param->stride_h_ = conv_param->stride_w_ = 2;
  conv_param->dilation_h_ = conv_param->dilation_w_ = 1;
  return out_t->ElementsNum();
}

TEST_F(TestDeconvInt8, DeConvInt8Test1) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  auto deconv_param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(deconv_param, nullptr);
  memset(deconv_param, 0, sizeof(ConvParameter));
  auto *ctx = new lite::InnerContext;
  deconv_param->op_parameter_.thread_num_ = 1;
  deconv_param->op_parameter_.is_zero_shape_ = false;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  int8_t *correct;
  int total_size = DeConvInt8TestInit1(&inputs_, &outputs_, deconv_param, &correct);
  auto *deconv = new kernel::DeConvInt8CPUKernel(reinterpret_cast<OpParameter *>(deconv_param), inputs_, outputs_, ctx);

  int ret = deconv->Prepare();
  ASSERT_EQ(0, ret);

  ret = deconv->Run();
  ASSERT_EQ(0, ret);

  int8_t *out_data = reinterpret_cast<int8_t *>(outputs_[0]->data());
  ASSERT_EQ(0, CompareOutputData(out_data, correct, total_size, 3));

  delete deconv;
  delete ctx;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  free(correct);
}
}  // namespace mindspore
