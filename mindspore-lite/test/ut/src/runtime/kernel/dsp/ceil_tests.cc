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

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "ut/src/runtime/kernel/dsp/dsp_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel_registry.h"

#ifdef SUPPORT_FT04

namespace mindspore::lite::dsp::test {

namespace {
constexpr int kCeilLength = 100;
constexpr int kValueCenter = kCeilLength / 2;
constexpr int kSampleParity = 2;
constexpr float kHalfFraction = 0.5f;
constexpr float kWholeFraction = 0.0f;
}  // namespace

class TestDSP_Ceil : public DSPCommonTest {};

TEST_F(TestDSP_Ceil, Ceil_Fp32) {
  InitDSPRuntime();
  const std::vector<int> shape = {kCeilLength};
  auto *input = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  auto *input_data = reinterpret_cast<float *>(input->MutableData());
  for (int j = 0; j < kCeilLength; ++j) {
    // Half of the samples are exact integers and the other half sit halfway between two integers,
    // so a floor or a round-to-nearest is caught by one of the two groups.
    const float fraction = (j % kSampleParity == 0) ? kHalfFraction : kWholeFraction;
    input_data[j] = static_cast<float>(j - kValueCenter) + fraction;
  }
  auto *output = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(kCeilLength) * sizeof(float));

  // LiteKernel::~LiteKernel() releases op_parameter_ with free(), so the parameter has to
  // come from malloc rather than new.
  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_Ceil);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Ceil};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({input}, {output}, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<float *>(output->MutableData());
  for (int j = 0; j < kCeilLength; ++j) {
    ASSERT_FLOAT_EQ(out_data[j], std::ceil(input_data[j]));
  }

  delete kernel;
  delete ctx;
  delete input;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Ceil, Ceil_Fp16) {
  InitDSPRuntime();
  const std::vector<int> shape = {kCeilLength};
  auto *input = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  auto *input_data = reinterpret_cast<uint16_t *>(input->MutableData());
  for (int j = 0; j < kCeilLength; ++j) {
    const float fraction = (j % kSampleParity == 0) ? kHalfFraction : kWholeFraction;
    input_data[j] = Fp32ToFp16(static_cast<float>(j - kValueCenter) + fraction);
  }
  auto *output = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(kCeilLength) * sizeof(uint16_t));

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_Ceil);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_Ceil};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({input}, {output}, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<uint16_t *>(output->MutableData());
  for (int j = 0; j < kCeilLength; ++j) {
    ASSERT_FLOAT_EQ(Fp16ToFp32(out_data[j]), std::ceil(Fp16ToFp32(input_data[j])));
  }

  delete kernel;
  delete ctx;
  delete input;
  delete output;
  UninitDSPRuntime();
}
}  // namespace mindspore::lite::dsp::test
#endif  // SUPPORT_FT04
