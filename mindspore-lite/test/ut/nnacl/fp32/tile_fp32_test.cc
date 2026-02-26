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
#include "nnacl_c/base/tile_base.h"

namespace mindspore {
class TileFp32Test : public ::testing::Test {
 public:
  TileFp32Test() {}
};

float get_cosine_similarity(const float *arr1, const float *arr2, size_t cmp_size) {
  if (arr1 == nullptr || arr2 == nullptr || cmp_size == 0) {
    return 0.0f;
  }
  float dot_product = 0.0f;
  float norm1 = 0.0f;
  float norm2 = 0.0f;
  for (size_t i = 0; i < cmp_size; ++i) {
    dot_product += arr1[i] * arr2[i];
    norm1 += arr1[i] * arr1[i];
    norm2 += arr2[i] * arr2[i];
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

extern float accuracy_threshold;

// Testcase1: input2x2_repeats2x3
TEST_F(TileFp32Test, Tile_2D) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> benchmark = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 4.0f, 3.0f, 4.0f,
                                  1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 4.0f, 3.0f, 4.0f};
  const int out_shape = 4 * 6;
  std::vector<float> output(out_shape, 0.0f);
  TileStruct tile_struct = {0};
  tile_struct.dims_[0] = 0;
  tile_struct.dims_[1] = 1;
  tile_struct.dims_size_ = 2;
  tile_struct.multiples_[0] = 2;
  tile_struct.multiples_[1] = 3;
  tile_struct.in_shape_[0] = 2;
  tile_struct.in_shape_[1] = 2;
  tile_struct.out_shape_[0] = 4;
  tile_struct.out_shape_[1] = 6;
  tile_struct.in_strides_[0] = 2;
  tile_struct.in_strides_[1] = 1;
  tile_struct.out_strides_[0] = 6;
  tile_struct.out_strides_[1] = 1;
  tile_struct.in_dim_ = 2;
  tile_struct.data_size_ = 4;
  Tile(input.data(), output.data(), &tile_struct);

  std::cout << "TileFp32Test-Tile_2D output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\n TileFp32Test-Tile_2D benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase2: input2x3x2_repeats1x2x1
TEST_F(TileFp32Test, Tile_3D) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  std::vector<float> benchmark = {
    1.0f, 2.0f, 1.0f, 2.0f,  5.0f,  6.0f,  1.0f, 2.0f, 1.0f, 2.0f,  5.0f,  6.0f,
    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
  };
  const int out_shape = 2 * 6 * 2;
  std::vector<float> output(out_shape, 0.0f);
  TileStruct tile_struct = {0};
  tile_struct.dims_[0] = 0;
  tile_struct.dims_[1] = 1;
  tile_struct.dims_[2] = 2;
  tile_struct.dims_size_ = 3;
  tile_struct.multiples_[0] = 1;
  tile_struct.multiples_[1] = 2;
  tile_struct.multiples_[2] = 1;
  tile_struct.in_shape_[0] = 2;
  tile_struct.in_shape_[1] = 3;
  tile_struct.in_shape_[2] = 2;
  tile_struct.out_shape_[0] = 2;
  tile_struct.out_shape_[1] = 6;
  tile_struct.out_shape_[2] = 2;
  tile_struct.in_strides_[0] = 6;
  tile_struct.in_strides_[1] = 2;
  tile_struct.in_strides_[2] = 1;
  tile_struct.out_strides_[0] = 12;
  tile_struct.out_strides_[1] = 2;
  tile_struct.out_strides_[2] = 1;
  tile_struct.in_dim_ = 3;
  tile_struct.data_size_ = 4;
  Tile(input.data(), output.data(), &tile_struct);

  std::cout << "TileFp32Test-Tile_3D output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nTileFp32Test-Tile_3D benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// Testcase3: input2x2x2x2_repeats2x2x1x1
TEST_F(TileFp32Test, Tile_4D) {
  std::vector<float> input = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  std::vector<float> benchmark = {
    1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
    9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
    1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
    9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
  };
  const int out_shape = 4 * 4 * 2 * 2;
  std::vector<float> output(out_shape, 0.0f);
  TileStruct tile_struct = {0};
  tile_struct.dims_[0] = 0;
  tile_struct.dims_[1] = 1;
  tile_struct.dims_[2] = 2;
  tile_struct.dims_[3] = 3;
  tile_struct.dims_size_ = 4;
  tile_struct.multiples_[0] = 2;
  tile_struct.multiples_[1] = 2;
  tile_struct.multiples_[2] = 1;
  tile_struct.multiples_[3] = 1;
  tile_struct.in_shape_[0] = 2;
  tile_struct.in_shape_[1] = 2;
  tile_struct.in_shape_[2] = 2;
  tile_struct.in_shape_[3] = 2;
  tile_struct.out_shape_[0] = 4;
  tile_struct.out_shape_[1] = 4;
  tile_struct.out_shape_[2] = 2;
  tile_struct.out_shape_[3] = 2;
  tile_struct.in_strides_[0] = 8;
  tile_struct.in_strides_[1] = 4;
  tile_struct.in_strides_[2] = 2;
  tile_struct.in_strides_[3] = 1;
  tile_struct.out_strides_[0] = 16;
  tile_struct.out_strides_[1] = 4;
  tile_struct.out_strides_[2] = 2;
  tile_struct.out_strides_[3] = 1;
  tile_struct.in_dim_ = 4;
  tile_struct.data_size_ = 4;
  Tile(input.data(), output.data(), &tile_struct);

  std::cout << "TileFp32Test-Tile_4D output:\n";
  std::for_each(output.begin(), output.end(), [](float value) { std::cout << value << ", "; });
  std::cout << "\nTileFp32Test-Tile_4D benchmark_data:\n";
  std::for_each(benchmark.begin(), benchmark.end(), [](float value) { std::cout << value << ", "; });
  float similarity = get_cosine_similarity(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

}  // namespace mindspore
