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
#include <cstdint>
#include <vector>
#include "gtest/gtest.h"
#include "nnacl_c/fp16/arithmetic_fp16.h"

namespace mindspore {
namespace {
using CompareFunc = int (*)(const float16_t *, const float16_t *, uint8_t *, int);
using OptCompareFunc = int (*)(const float16_t *, const float16_t *, uint8_t *, int, bool);

enum CompareType { kNotEqual, kEqual, kLess, kLessEqual, kGreater, kGreaterEqual };

struct CompareCase {
  const char *name;
  CompareType type;
  CompareFunc func;
  OptCompareFunc opt_func;
};

const std::vector<CompareCase> kCompareCases = {
  {"NotEqual", kNotEqual, ElementNotEqualFp16, ElementOptNotEqualFp16},
  {"Equal", kEqual, ElementEqualFp16, ElementOptEqualFp16},
  {"Less", kLess, ElementLessFp16, ElementOptLessFp16},
  {"LessEqual", kLessEqual, ElementLessEqualFp16, ElementOptLessEqualFp16},
  {"Greater", kGreater, ElementGreaterFp16, ElementOptGreaterFp16},
  {"GreaterEqual", kGreaterEqual, ElementGreaterEqualFp16, ElementOptGreaterEqualFp16},
};

uint8_t Compare(float16_t lhs, float16_t rhs, CompareType type) {
  switch (type) {
    case kNotEqual:
      return static_cast<uint8_t>(lhs != rhs);
    case kEqual:
      return static_cast<uint8_t>(lhs == rhs);
    case kLess:
      return static_cast<uint8_t>(lhs < rhs);
    case kLessEqual:
      return static_cast<uint8_t>(lhs <= rhs);
    case kGreater:
      return static_cast<uint8_t>(lhs > rhs);
    case kGreaterEqual:
      return static_cast<uint8_t>(lhs >= rhs);
    default:
      return 0;
  }
}

void CheckOutput(const CompareCase &compare_case, const std::vector<float16_t> &input0,
                 const std::vector<float16_t> &input1, const std::vector<uint8_t> &output, bool first_scalar,
                 bool second_scalar) {
  for (size_t i = 0; i < output.size(); ++i) {
    auto lhs = input0[first_scalar ? 0 : i];
    auto rhs = input1[second_scalar ? 0 : i];
    EXPECT_EQ(output[i], Compare(lhs, rhs, compare_case.type)) << compare_case.name << " failed at index " << i;
  }
}

std::vector<float16_t> CreateInput(size_t size, int multiplier) {
  std::vector<float16_t> input(size);
  for (size_t i = 0; i < size; ++i) {
    input[i] = static_cast<float16_t>((static_cast<int>(i) * multiplier) % 5);
  }
  return input;
}
}  // namespace

class ArithmeticCompareFp16Test : public ::testing::Test {};

TEST_F(ArithmeticCompareFp16Test, TensorToTensorOutputIsZeroOrOne) {
  for (auto size : {7, 8, 9, 17, 120}) {
    auto input0 = CreateInput(size, 1);
    auto input1 = CreateInput(size, 2);
    std::vector<uint8_t> output(size);
    for (const auto &compare_case : kCompareCases) {
      ASSERT_EQ(compare_case.func(input0.data(), input1.data(), output.data(), size), NNACL_OK);
      CheckOutput(compare_case, input0, input1, output, false, false);
    }
  }
}

TEST_F(ArithmeticCompareFp16Test, FirstScalarOutputIsZeroOrOne) {
  const std::vector<float16_t> input0 = {static_cast<float16_t>(2.0f)};
  for (auto size : {7, 8, 9, 17, 120}) {
    auto input1 = CreateInput(size, 2);
    std::vector<uint8_t> output(size);
    for (const auto &compare_case : kCompareCases) {
      ASSERT_EQ(compare_case.opt_func(input0.data(), input1.data(), output.data(), size, true), NNACL_OK);
      CheckOutput(compare_case, input0, input1, output, true, false);
    }
  }
}

TEST_F(ArithmeticCompareFp16Test, SecondScalarOutputIsZeroOrOne) {
  const std::vector<float16_t> input1 = {static_cast<float16_t>(3.0f)};
  for (auto size : {7, 8, 9, 17, 120}) {
    auto input0 = CreateInput(size, 1);
    std::vector<uint8_t> output(size);
    for (const auto &compare_case : kCompareCases) {
      ASSERT_EQ(compare_case.opt_func(input0.data(), input1.data(), output.data(), size, false), NNACL_OK);
      CheckOutput(compare_case, input0, input1, output, false, true);
    }
  }
}
}  // namespace mindspore
