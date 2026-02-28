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
#include "nnacl_c/fp32/resize_fp32.h"
#include "nnacl_c/base/resize_base.h"

namespace mindspore {
class ResizeFp32Test : public ::testing::Test {
 public:
  ResizeFp32Test() {}
};

float get_cosine_similarity(const float *arr1, const float *arr2, size_t cmp_size);
extern float accuracy_threshold;

// Testcase1: NCHW ONNX format, ResizeBilinear with input1x2x1x2_output1x2x1x4_align-corners
TEST_F(ResizeFp32Test, ResizeBilinear_align_corners) {
  std::vector<float> input = {1.0f, 3.0f, 2.0f, 4.0f};
  std::vector<float> benchmark = {1.0f, 3.0f, 1.333333f, 3.333333f, 1.666666f, 3.666666f, 2.0f, 4.0f};
  const int out_shape = 1 * 2 * 1 * 4;
  std::vector<float> output(out_shape, 0.0f);
  std::vector<float> line0(out_shape, 0.0f);
  std::vector<float> line1(out_shape, 0.0f);
  const int32_t input_shape[4] = {1, 1, 2, 2};
  const int32_t output_shape[4] = {1, 1, 4, 2};
  const int32_t y_bottoms[1] = {0};
  const int32_t y_tops[1] = {0};
  const int32_t x_lefts[4] = {0, 0, 0, 1};
  const int32_t x_rights[4] = {1, 1, 1, 1};
  const float y_weights[1] = {1};
  const float x_weights[4] = {1.0f, 0.666666f, 0.333333f, 1.0f};
  int h_begin = 0;
  int h_end = 1;
  ResizeBilinear(input.data(), output.data(), input_shape, output_shape, y_bottoms, y_tops, x_lefts, x_rights,
                 y_weights, x_weights, line0.data(), line1.data(), h_begin, h_end);

  std::cout << "ResizeFp32Test-ResizeBilinear_align_corners output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\n ResizeFp32Test-ResizeBilinear_align_corners benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: NCHW ONNX format, ResizeBilinear with input2x2x1x2_output2x2x1x3_half-pixel
TEST_F(ResizeFp32Test, ResizeBilinear_half_pixel) {
  std::vector<float> input = {1.0f, 3.0f, 2.0f, 4.0f, 5.0f, 7.0f, 6.0f, 8.0f};
  std::vector<float> benchmark = {1.0f, 3.0f, 1.5f, 3.5f, 2.0f, 4.0f, 5.0f, 7.0f, 5.5f, 7.5f, 6.0f, 8.0f};
  const int out_shape = 2 * 2 * 1 * 3;
  std::vector<float> output(out_shape, 0.0f);
  std::vector<float> line0(out_shape, 0.0f);
  std::vector<float> line1(out_shape, 0.0f);
  const int32_t input_shape[4] = {2, 1, 2, 2};
  const int32_t output_shape[4] = {2, 1, 3, 2};
  const int32_t y_bottoms[1] = {0};
  const int32_t y_tops[1] = {0};
  const int32_t x_lefts[3] = {0, 0, 1};
  const int32_t x_rights[3] = {1, 1, 1};
  const float y_weights[1] = {1};
  const float x_weights[3] = {1.0f, 0.5f, 0.833333f};
  int h_begin = 0;
  int h_end = 1;
  ResizeBilinear(input.data(), output.data(), input_shape, output_shape, y_bottoms, y_tops, x_lefts, x_rights,
                 y_weights, x_weights, line0.data(), line1.data(), h_begin, h_end);

  std::cout << "ResizeFp32Test-ResizeBilinear_half_pixel output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\n ResizeFp32Test-ResizeBilinear_half_pixel benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: NCHW ONNX format, ResizeNearestNeighbor with input1x2x2x3_output1x2x3x4_align-corners_round-prefer-floor
TEST_F(ResizeFp32Test, ResizeNearestNeighbor_align_corners) {
  std::vector<float> input = {1.0f, 7.0f, 2.0f, 8.0f, 3.0f, 9.0f, 4.0f, 10.0f, 5.0f, 11.0f, 6.0f, 12.0f};
  std::vector<float> benchmark = {1.0f, 7.0f, 2.0f, 8.0f, 2.0f, 8.0f,  3.0f, 9.0f,  1.0f, 7.0f,  2.0f, 8.0f,
                                  2.0f, 8.0f, 3.0f, 9.0f, 4.0f, 10.0f, 5.0f, 11.0f, 5.0f, 11.0f, 6.0f, 12.0f};
  const int out_shape = 1 * 2 * 3 * 4;
  std::vector<float> output(out_shape, 0.0f);
  const int32_t input_shape[4] = {1, 2, 3, 2};
  const int32_t output_shape[4] = {1, 3, 4, 2};
  int coordinate_transform_mode = 1;
  int nearest_mode = 1;
  int task_id = 0;
  int thread_num = 1;
  ResizeNearestNeighbor(input.data(), output.data(), input_shape, output_shape, CalculateAlignCorners,
                        coordinate_transform_mode, nearest_mode, task_id, thread_num);

  std::cout << "ResizeFp32Test-ResizeNearestNeighbor_align_corners output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\n ResizeNearestNeighbor_align_corners benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase4: NHWC TFLite format, ResizeNearestNeighbor with input1x2x2x1_output1x4x4x1_half-pixel
TEST_F(ResizeFp32Test, ResizeNearestNeighbor_half_pixel) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> benchmark = {
    1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 3.0f, 3.0f, 4.0f, 4.0f,
  };
  const int out_shape = 1 * 4 * 4 * 1;
  std::vector<float> output(out_shape, 0.0f);
  const int32_t input_shape[4] = {1, 2, 2, 1};
  const int32_t output_shape[4] = {1, 4, 4, 1};
  int coordinate_transform_mode = 2;
  int nearest_mode = 0;
  int task_id = 0;
  int thread_num = 1;
  ResizeNearestNeighbor(input.data(), output.data(), input_shape, output_shape, CalculateHalfPixel,
                        coordinate_transform_mode, nearest_mode, task_id, thread_num);

  std::cout << "ResizeFp32Test-ResizeNearestNeighbor_half_pixel output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\n ResizeNearestNeighbor_half_pixel benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  std::cout << std::endl;
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
