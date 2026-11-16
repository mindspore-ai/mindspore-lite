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
constexpr int kCosLength = 100;
constexpr int kValueCenter = kCosLength / 2;
constexpr float kFpStep = 0.1f;
constexpr float kFp32ErrBound = 1e-3f;
constexpr float kFp16ErrBound = 1e-3f;
}  // namespace

class TestDSP_Cos : public DSPCommonTest {};

TEST_F(TestDSP_Cos, Cos_Fp32) {
  InitDSPRuntime();
  const std::vector<int> shape = {kCosLength};
  auto *input = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  auto *input_data = reinterpret_cast<float *>(input->MutableData());
  for (int j = 0; j < kCosLength; ++j) {
    // Values spread over both halves of the cosine period so a wrong phase or a swapped element
    // order cannot pass.
    input_data[j] = static_cast<float>(j - kValueCenter) * kFpStep;
  }
  auto *output = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(kCosLength) * sizeof(float));

  // LiteKernel::~LiteKernel() releases op_parameter_ with free(), so the parameter has to
  // come from malloc rather than new.
  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_Cos);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Cos};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({input}, {output}, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<float *>(output->MutableData());
  for (int j = 0; j < kCosLength; ++j) {
    ASSERT_NEAR(out_data[j], std::cos(input_data[j]), kFp32ErrBound);
  }

  delete kernel;
  delete ctx;
  delete input;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Cos, Cos_Fp16) {
  InitDSPRuntime();
  const std::vector<int> shape = {kCosLength};
  auto *input = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  auto *input_data = reinterpret_cast<uint16_t *>(input->MutableData());
  for (int j = 0; j < kCosLength; ++j) {
    input_data[j] = Fp32ToFp16(static_cast<float>(j - kValueCenter) * kFpStep);
  }
  auto *output = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(kCosLength) * sizeof(uint16_t));

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_Cos);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_Cos};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({input}, {output}, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<uint16_t *>(output->MutableData());
  for (int j = 0; j < kCosLength; ++j) {
    // The reference takes the stored half value, not the fp32 one, so that the rounding of the
    // input itself is not charged to the kernel.
    ASSERT_NEAR(Fp16ToFp32(out_data[j]), std::cos(Fp16ToFp32(input_data[j])), kFp16ErrBound);
  }

  delete kernel;
  delete ctx;
  delete input;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Cos, Cos_Int16) {
  InitDSPRuntime();
  const std::vector<int> shape = {kCosLength};
  auto *input = new lite::Tensor(kNumberTypeInt16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  auto *input_data = reinterpret_cast<int16_t *>(input->MutableData());
  for (int j = 0; j < kCosLength; ++j) {
    input_data[j] = static_cast<int16_t>(j - kValueCenter);
  }
  // armc writes the cosine of an integer input as fp32, so the output tensor is fp32.
  auto *output = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(kCosLength) * sizeof(float));

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_Cos);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt16, NHWC, schema::PrimitiveType_Cos};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({input}, {output}, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<float *>(output->MutableData());
  for (int j = 0; j < kCosLength; ++j) {
    ASSERT_NEAR(out_data[j], std::cos(static_cast<float>(input_data[j])), kFp32ErrBound);
  }

  delete kernel;
  delete ctx;
  delete input;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Cos, Cos_Int32) {
  InitDSPRuntime();
  const std::vector<int> shape = {kCosLength};
  auto *input = new lite::Tensor(kNumberTypeInt32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  auto *input_data = reinterpret_cast<int32_t *>(input->MutableData());
  for (int j = 0; j < kCosLength; ++j) {
    input_data[j] = j - kValueCenter;
  }
  auto *output = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(kCosLength) * sizeof(float));

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_Cos);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_Cos};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({input}, {output}, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<float *>(output->MutableData());
  for (int j = 0; j < kCosLength; ++j) {
    ASSERT_NEAR(out_data[j], std::cos(static_cast<float>(input_data[j])), kFp32ErrBound);
  }

  delete kernel;
  delete ctx;
  delete input;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Cos, Cos_Int16_IntegerOutputIsRejected) {
  InitDSPRuntime();
  const std::vector<int> shape = {kCosLength};
  auto *input = new lite::Tensor(kNumberTypeInt16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  auto *input_data = reinterpret_cast<int16_t *>(input->MutableData());
  for (int j = 0; j < kCosLength; ++j) {
    input_data[j] = static_cast<int16_t>(j - kValueCenter);
  }
  // cos_func writes one fp32 per element for an integer input, so an integer output of half the size
  // would be overrun; the kernel has to refuse the combination instead.
  auto *output = new lite::Tensor(kNumberTypeInt16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_Cos);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt16, NHWC, schema::PrimitiveType_Cos};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({input}, {output}, param, ctx, key);
  ASSERT_EQ(kernel, nullptr);

  delete ctx;
  delete input;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Cos, Cos_Int16_UnknownOutputShapeIsRefused) {
  InitDSPRuntime();
  const std::vector<int> shape = {kCosLength};
  auto *input = new lite::Tensor(kNumberTypeInt16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  auto *input_data = reinterpret_cast<int16_t *>(input->MutableData());
  for (int j = 0; j < kCosLength; ++j) {
    input_data[j] = static_cast<int16_t>(j - kValueCenter);
  }
  // A dynamic model leaves the output shape unknown, which makes DSPKernelCreator hand the kernel
  // back without running CheckSpecs at all. Neither Prepare nor Run may then hand an integer output
  // buffer to cos_func, whose result for an integer input is fp32.
  const std::vector<int> unknown_shape = {-1};
  auto *output = new lite::Tensor(kNumberTypeInt16, unknown_shape, NHWC, lite::Category::CONST_TENSOR);

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_Cos);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt16, NHWC, schema::PrimitiveType_Cos};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({input}, {output}, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_ERROR, kernel->Prepare());
  ASSERT_EQ(RET_ERROR, kernel->Run());

  delete kernel;
  delete ctx;
  delete input;
  delete output;
  UninitDSPRuntime();
}
}  // namespace mindspore::lite::dsp::test
#endif  // SUPPORT_FT04
