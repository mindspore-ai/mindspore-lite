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
#include <vector>
#include "gtest/gtest.h"
#include "nnacl_c/fp32/arg_min_max_fp32.h"

namespace mindspore {
namespace {

void InitArgMinParam(ArgMinMaxComputeParam *param, const std::vector<int32_t> &in_shape,
                     const std::vector<int32_t> &out_shape, int32_t axis, int32_t topk, bool keep_dims, bool out_value,
                     ArgElement *arg_elements) {
  *param = {};
  param->axis_ = axis;
  param->dims_size_ = static_cast<int32_t>(in_shape.size());
  param->topk_ = topk;
  param->get_max_ = false;
  param->keep_dims_ = keep_dims;
  param->out_value_ = out_value;
  ComputeStrides(in_shape.data(), param->in_strides_, param->dims_size_);
  ComputeStrides(out_shape.data(), param->out_strides_, static_cast<int>(out_shape.size()));
  param->arg_elements_ = arg_elements;
}

}  // namespace

class ArgMinFp32Test : public ::testing::Test {
 public:
  ArgMinFp32Test() {}
};

// Testcase1: ArgMin with keep_dims=false, topk=1 and 3D input tensor (2x3x2)
TEST_F(ArgMinFp32Test, ArgMin_Index_3D) {
  std::vector<float> input = {4.0f, 1.0f, 2.0f, 3.0f, 5.0f, 0.0f, 9.0f, 8.0f, 7.0f, 6.0f, 10.0f, 4.0f};
  std::vector<int32_t> benchmark = {1, 2, 1, 2};
  const std::vector<int32_t> in_shape = {2, 3, 2};
  const std::vector<int32_t> out_shape = {2, 2};
  const int length = 2 * 2;
  std::vector<int32_t> output(length, 0);
  ArgMinMaxComputeParam param = {};
  InitArgMinParam(&param, in_shape, out_shape, 1, 1, false, false, nullptr);

  ArgMinMaxFp32(input.data(), output.data(), nullptr, in_shape.data(), &param);

  for (int i = 0; i < length; ++i) {
    EXPECT_EQ(output[i], benchmark[i]);
  }
}

// Testcase2: ArgMin with keep_dims=true, topk=1, out_value=true and 3D input tensor (2x3x4)
TEST_F(ArgMinFp32Test, ArgMin_Value_3D) {
  std::vector<float> input = {3.0f,  1.0f, 5.0f, 2.0f, 4.0f, 9.0f, 8.0f, 7.0f, 6.0f, 0.0f,  11.0f, 10.0f,
                              -1.0f, 2.0f, 3.0f, 4.0f, 8.0f, 7.0f, 6.0f, 5.0f, 9.0f, 12.0f, 1.0f,  0.0f};
  std::vector<float> benchmark = {1.0f, 4.0f, 0.0f, -1.0f, 5.0f, 0.0f};
  const std::vector<int32_t> in_shape = {2, 3, 4};
  const std::vector<int32_t> out_shape = {2, 3, 1};
  const int length = 2 * 3 * 1;
  std::vector<float> output(length, 0.0f);
  ArgMinMaxComputeParam param = {};
  InitArgMinParam(&param, in_shape, out_shape, 2, 1, true, true, nullptr);

  ArgMinMaxFp32(input.data(), output.data(), nullptr, in_shape.data(), &param);

  for (int i = 0; i < length; ++i) {
    EXPECT_FLOAT_EQ(output[i], benchmark[i]);
  }
}

// Testcase3: ArgMin with keep_dims=true, topk=2 and 3D input tensor (2x3x2)
TEST_F(ArgMinFp32Test, ArgMin_TopK2_3D) {
  std::vector<float> input = {5.0f, 1.0f, 2.0f, 7.0f, 3.0f, 0.0f, 4.0f, 9.0f, 6.0f, 3.0f, 1.0f, 8.0f};
  std::vector<int32_t> benchmark_index = {1, 2, 2, 0, 2, 1, 0, 2};
  std::vector<float> benchmark_value = {2.0f, 0.0f, 3.0f, 1.0f, 1.0f, 3.0f, 4.0f, 8.0f};
  const std::vector<int32_t> in_shape = {2, 3, 2};
  const std::vector<int32_t> out_shape = {2, 2, 2};
  const int length = 2 * 2 * 2;
  std::vector<int32_t> output_index(length, 0);
  std::vector<float> output_value(length, 0.0f);
  std::vector<ArgElement> arg_elements(3);
  ArgMinMaxComputeParam param = {};
  InitArgMinParam(&param, in_shape, out_shape, 1, 2, true, false, arg_elements.data());

  ArgMinMaxFp32(input.data(), output_index.data(), output_value.data(), in_shape.data(), &param);

  for (int i = 0; i < length; ++i) {
    EXPECT_EQ(output_index[i], benchmark_index[i]);
    EXPECT_FLOAT_EQ(output_value[i], benchmark_value[i]);
  }
}

}  // namespace mindspore
