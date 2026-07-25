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
#include "nnacl_c/fp32/triu_tril_fp32.h"
#include "nnacl_c/tensor_c.h"
#include "nnacl_c/kernel.h"
#include "nnacl_c/errorcode.h"

namespace mindspore {
class TriuTrilFp32Test : public ::testing::Test {
 public:
  TriuTrilFp32Test() {}
};

// FP32 cases exercise TriuByte4/TrilByte4 (4-byte width = float32), which are the functions the coder actually calls.
// TriuByte1/2/8 (1/2/8-byte widths) are not on this PR's coder path, so they are not unit-tested here.
static float accuracy_threshold = 0.99f;

float get_cosine_similarity_fp32(const float *arr1, const float *arr2, size_t cmp_size) {
  if (arr1 == nullptr || arr2 == nullptr || cmp_size == 0) {
    return 0.0f;
  }
  double dot_product = 0.0;
  double norm1 = 0.0;
  double norm2 = 0.0;
  for (size_t i = 0; i < cmp_size; ++i) {
    double v1 = static_cast<double>(arr1[i]);
    double v2 = static_cast<double>(arr2[i]);
    dot_product += v1 * v2;
    norm1 += v1 * v1;
    norm2 += v2 * v2;
  }
  norm1 = std::sqrt(norm1);
  norm2 = std::sqrt(norm2);
  double norms_product = norm1 * norm2;
  const double EPS = 1e-6;
  if (std::fabs(norms_product) < EPS) {
    return 0.0f;
  }
  return static_cast<float>(dot_product / norms_product);
}

// Build an element-wise increasing float32 input over the [-6, 6] mixed-sign range.
static std::vector<float> make_fp32_input(const std::vector<int> &shape) {
  size_t total = 1;
  for (int d : shape) {
    total *= static_cast<size_t>(d);
  }
  std::vector<float> data(total);
  for (size_t i = 0; i < total; ++i) {
    data[i] = static_cast<float>(-6.0 + 12.0 * i / (total > 1 ? total - 1 : 1));
  }
  return data;
}

// TC-001: non-square [4,8] upper k=0 — baseline, covers M!=N (the old gtest only used square matrices).
TEST_F(TriuTrilFp32Test, TriuByte4_4x8_k0) {
  std::vector<int> shape = {4, 8};
  auto input = make_fp32_input(shape);
  const int height = shape[0];
  const int width = shape[1];
  const int num = 1;
  std::vector<float> output(input.size(), 0.0f);

  TriuByte4(input.data(), output.data(), 0, height, width, num);

  std::vector<float> benchmark(input.size(), 0.0f);
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      benchmark[h * width + w] = (h <= w) ? input[h * width + w] : 0.0f;
    }
  }
  float similarity = get_cosine_similarity_fp32(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// TC-002: non-square [4,8] lower k=0 — lower-direction baseline.
TEST_F(TriuTrilFp32Test, TrilByte4_4x8_k0) {
  std::vector<int> shape = {4, 8};
  auto input = make_fp32_input(shape);
  const int height = shape[0];
  const int width = shape[1];
  const int num = 1;
  std::vector<float> output(input.size(), 0.0f);

  TrilByte4(input.data(), output.data(), 0, height, width, num);

  std::vector<float> benchmark(input.size(), 0.0f);
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      benchmark[h * width + w] = (h >= w) ? input[h * width + w] : 0.0f;
    }
  }
  float similarity = get_cosine_similarity_fp32(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// TC-003: non-square [16,32] upper k=1 — k>0 shrinks the upper triangle below the main diagonal.
TEST_F(TriuTrilFp32Test, TriuByte4_16x32_k1) {
  std::vector<int> shape = {16, 32};
  auto input = make_fp32_input(shape);
  const int height = shape[0];
  const int width = shape[1];
  const int num = 1;
  std::vector<float> output(input.size(), 0.0f);

  TriuByte4(input.data(), output.data(), 1, height, width, num);

  std::vector<float> benchmark(input.size(), 0.0f);
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      benchmark[h * width + w] = (h + 1 <= w) ? input[h * width + w] : 0.0f;
    }
  }
  float similarity = get_cosine_similarity_fp32(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// TC-004: non-square [16,32] lower k=-1 — k<0 shrinks the lower triangle above the main diagonal.
TEST_F(TriuTrilFp32Test, TrilByte4_16x32_kn1) {
  std::vector<int> shape = {16, 32};
  auto input = make_fp32_input(shape);
  const int height = shape[0];
  const int width = shape[1];
  const int num = 1;
  std::vector<float> output(input.size(), 0.0f);

  TrilByte4(input.data(), output.data(), -1, height, width, num);

  std::vector<float> benchmark(input.size(), 0.0f);
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      benchmark[h * width + w] = (h + (-1) >= w) ? input[h * width + w] : 0.0f;
    }
  }
  float similarity = get_cosine_similarity_fp32(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// TC-005: non-square [8,16] upper k=5 — k boundary: when k>=height the upper-triangle off-diagonal region (incl. the
// main diagonal) goes all-zero.
TEST_F(TriuTrilFp32Test, TriuByte4_8x16_k5) {
  std::vector<int> shape = {8, 16};
  auto input = make_fp32_input(shape);
  const int height = shape[0];
  const int width = shape[1];
  const int num = 1;
  std::vector<float> output(input.size(), 0.0f);

  TriuByte4(input.data(), output.data(), 5, height, width, num);

  std::vector<float> benchmark(input.size(), 0.0f);
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      benchmark[h * width + w] = (h + 5 <= w) ? input[h * width + w] : 0.0f;
    }
  }
  float similarity = get_cosine_similarity_fp32(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// TC-006: 3D batched [2,8,8] upper k=0 — num=2 batch loop (triangular extract on each of the last two dims).
TEST_F(TriuTrilFp32Test, TriuByte4_3D_k0) {
  std::vector<int> shape = {2, 8, 8};
  auto input = make_fp32_input(shape);
  const int height = shape[shape.size() - 2];
  const int width = shape[shape.size() - 1];
  int num = 1;
  for (size_t i = 0; i < shape.size() - 2; ++i) {
    num *= shape[i];
  }
  std::vector<float> output(input.size(), 0.0f);

  TriuByte4(input.data(), output.data(), 0, height, width, num);

  std::vector<float> benchmark(input.size(), 0.0f);
  for (int m = 0; m < num; ++m) {
    const int plane = m * height * width;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        benchmark[plane + h * width + w] = (h <= w) ? input[plane + h * width + w] : 0.0f;
      }
    }
  }
  float similarity = get_cosine_similarity_fp32(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}

// TC-007: 4D batched [2,2,8,8] upper k=0 — num=4 batch, highest-rank coverage (the old gtest topped out at 3D).
TEST_F(TriuTrilFp32Test, TriuByte4_4D_k0) {
  std::vector<int> shape = {2, 2, 8, 8};
  auto input = make_fp32_input(shape);
  const int height = shape[shape.size() - 2];
  const int width = shape[shape.size() - 1];
  int num = 1;
  for (size_t i = 0; i < shape.size() - 2; ++i) {
    num *= shape[i];
  }
  std::vector<float> output(input.size(), 0.0f);

  TriuByte4(input.data(), output.data(), 0, height, width, num);

  std::vector<float> benchmark(input.size(), 0.0f);
  for (int m = 0; m < num; ++m) {
    const int plane = m * height * width;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        benchmark[plane + h * width + w] = (h <= w) ? input[plane + h * width + w] : 0.0f;
      }
    }
  }
  float similarity = get_cosine_similarity_fp32(output.data(), benchmark.data(), output.size());
  ASSERT_GT(similarity, accuracy_threshold);
}
}  // namespace mindspore
