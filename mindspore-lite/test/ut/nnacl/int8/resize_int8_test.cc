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
#include "gtest/gtest.h"
#include "nnacl_c/int8/resize_int8.h"
#include "nnacl_c/base/resize_base.h"

namespace mindspore {
class ResizeInt8Test : public ::testing::Test {
 public:
  ResizeInt8Test() {}
};

float get_cosine_similarity_int8(const int8_t *arr1, const int8_t *arr2, size_t cmp_size);
extern float accuracy_threshold;

// Testcase1: NHWC TFLite format, ResizeBilinear with input1x5x4x1_output1x6x8x1_align-corners
TEST_F(ResizeInt8Test, ResizeBilinear_align_corners) {
  std::vector<int8_t> input = {-115, -103, -90, -77, -64, -52, -39, -26, -13, -1,
                               12,   25,   38,  51,  63,  76,  89,  102, 114, 127};
  std::vector<int8_t> benchmark = {-115, -109, -104, -99, -93, -88, -82, -77, -74, -69, -63, -58, -52, -47, -41, -36,
                                   -33,  -28,  -23,  -17, -12, -6,  0,   4,   7,   12,  18,  23,  28,  34,  39,  45,
                                   48,   53,   59,   64,  69,  75,  80,  86,  89,  94,  100, 105, 110, 115, 121, 127};
  int out_shape = 1 * 6 * 8 * 1;
  std::vector<int8_t> output(out_shape, 0);
  float x_axis_index[8] = {0.0f, 0.428571f, 1.28571f, 1.71429f, 2.14286f, 2.57143f, 3.0f};
  int32_t x_axis_lower[8] = {0, 0, 0, 1, 1, 2, 2, 3};
  int32_t x_axis_upper[8] = {1, 1, 1, 2, 2, 3, 3, 3};
  float y_axis_index[6] = {0.0f, 0.8f, 1.6f, 2.4f, 3.2f, 4.0f};
  int32_t y_axis_lower[6] = {0, 0, 1, 2, 3, 4};
  int32_t y_axis_upper[6] = {1, 1, 2, 3, 4, 4};
  float ratio_x = 0.428571f;
  float ratio_y = 0.8f;
  const ResizeFloatScaleQuantArg resize_float_quant = {ratio_x,      ratio_y,      x_axis_index, x_axis_lower,
                                                       x_axis_upper, y_axis_index, y_axis_lower, y_axis_upper};
  int batch = 1;
  int in_h = 5;
  int in_w = 4;
  int out_h = 6;
  int out_w = 8;
  int channel = 1;
  int index = 0;
  int count = out_h * out_w;
  ResizeBilinearWithFloatScaleInt8(input.data(), output.data(), batch, in_h, in_w, out_h, out_w, channel, index, count,
                                   resize_float_quant);

  std::cout << "ResizeInt8Test-ResizeBilinear_align_corners output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\n ResizeInt8Test-ResizeBilinear_align_corners benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: NHWC TFLite format, ResizeBilinear with input2x1x2x3_output2x3x3x3_half_pixel
TEST_F(ResizeInt8Test, ResizeBilinear_half_pixel) {
  std::vector<int8_t> input = {
    -107, -86, -64, -43, -22, -1, 21, 42, 63, 85, 106, 127,
  };
  std::vector<int8_t> benchmark = {
    -106, -86, -63, -75, -54, -32, -42, -22, 0,   -107, -86, -64, -75, -54, -32, -43, -22, -1,
    -107, -86, -64, -75, -54, -32, -43, -21, -1,  21,   42,  62,  53,  74,  95,  85,  105, 127,
    21,   42,  63,  53,  74,  95,  85,  106, 127, 20,   41,  62,  43,  74,  95,  85,  105, 126,
  };
  int out_shape = 2 * 3 * 3 * 3;
  std::vector<int8_t> output(out_shape, 0);
  float x_axis_index[3] = {-0.166667f, 0.5f, 1.166667f};
  int32_t x_axis_lower[3] = {0, 0, 1};
  int32_t x_axis_upper[3] = {0, 1, 1};
  float y_axis_index[3] = {-0.333333f, 0.0f, 0.333333f};
  int32_t y_axis_lower[3] = {0, 0, 0};
  int32_t y_axis_upper[3] = {0, 0, 0};
  float ratio_x = 0.666667f;
  float ratio_y = 0.333333f;
  const ResizeFloatScaleQuantArg resize_float_quant = {ratio_x,      ratio_y,      x_axis_index, x_axis_lower,
                                                       x_axis_upper, y_axis_index, y_axis_lower, y_axis_upper};
  int batch = 2;
  int in_h = 1;
  int in_w = 2;
  int out_h = 3;
  int out_w = 3;
  int channel = 3;
  int index = 0;
  int count = out_h * out_w;
  ResizeBilinearWithFloatScaleInt8(input.data(), output.data(), batch, in_h, in_w, out_h, out_w, channel, index, count,
                                   resize_float_quant);

  std::cout << "ResizeInt8Test-ResizeBilinear_half_pixel output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\n ResizeInt8Test-ResizeBilinear_half_pixel benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: NHWC TFLite format, ResizeNearestNeighbor with input1x2x2x1_output1x3x4x1_align-corners
TEST_F(ResizeInt8Test, ResizeNearestNeighbor_align_corners) {
  std::vector<int8_t> input = {-64, -1, 63, 127};
  std::vector<int8_t> benchmark = {
    -64, -64, -1, -1, 63, 63, 127, 127, 63, 63, 127, 127,
  };
  int out_shape = 1 * 3 * 4 * 1;
  std::vector<int8_t> output(out_shape, 0);
  const int32_t input_shape[4] = {1, 2, 2, 1};
  const int32_t output_shape[4] = {1, 3, 4, 1};
  bool align_corners = true;
  int coordinate_transform_mode = 1;
  int nearest_mode = 0;
  int task_id = 0;
  int thread_num = 1;
  ResizeNearestNeighborInt8Simple(input.data(), output.data(), input_shape, output_shape, align_corners,
                                  coordinate_transform_mode, nearest_mode, task_id, thread_num);

  std::cout << "ResizeInt8Test-ResizeNearestNeighbor_align_corners output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\n ResizeInt8Test-ResizeNearestNeighbor_align_corners benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase4: NHWC TFLite format, ResizeNearestNeighbor with input1x2x2x1_output1x3x3x1_half-pixel
TEST_F(ResizeInt8Test, ResizeNearestNeighbor_half_pixel) {
  std::vector<int8_t> input = {-64, -1, 63, 127};
  std::vector<int8_t> benchmark = {-64, -1, -1, 63, 127, 127, 63, 127, 127};
  int out_shape = 1 * 3 * 3 * 1;
  std::vector<int8_t> output(out_shape, 0);
  const int32_t input_shape[4] = {1, 2, 2, 1};
  const int32_t output_shape[4] = {1, 3, 3, 1};
  bool align_corners = false;
  int coordinate_transform_mode = 2;
  int nearest_mode = 0;
  int task_id = 0;
  int thread_num = 1;
  ResizeNearestNeighborInt8Simple(input.data(), output.data(), input_shape, output_shape, align_corners,
                                  coordinate_transform_mode, nearest_mode, task_id, thread_num);

  std::cout << "ResizeInt8Test-ResizeNearestNeighbor_half_pixel output:\n";
  std::for_each(output.begin(), output.end(), [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << "\n ResizeInt8Test-ResizeNearestNeighbor_half_pixel benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(),
                [](int8_t value) { std::cout << static_cast<int32_t>(value) << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity_int8(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
