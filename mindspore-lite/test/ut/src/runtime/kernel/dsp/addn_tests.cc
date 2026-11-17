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
constexpr int kAddNLength = 100;
constexpr int kScalarElementNum = 1;
constexpr size_t kTwoInputs = 2;
constexpr size_t kThreeInputs = 3;
constexpr int kComplexLength = 64;
constexpr int kComplexComponents = 2;
constexpr float kFp32ErrBound = 1e-4f;
constexpr float kFp16ErrBound = 5e-2f;
constexpr float kBroadcastValue = 2.5f;
}  // namespace

class TestDSP_AddN : public DSPCommonTest {};

TEST_F(TestDSP_AddN, AddN_Fp32_TwoInputs) {
  InitDSPRuntime();
  const std::vector<int> shape = {kAddNLength};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  for (size_t i = 0; i < kTwoInputs; ++i) {
    auto *input = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
    auto *input_data = reinterpret_cast<float *>(input->MutableData());
    for (int j = 0; j < kAddNLength; ++j) {
      // Distinct per-input values so that a wrong input order is detectable.
      input_data[j] = static_cast<float>(j) * static_cast<float>(i + 1);
    }
    inputs.push_back(input);
  }
  auto *output = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(kAddNLength) * sizeof(float));
  outputs.push_back(output);

  // LiteKernel::~LiteKernel() releases op_parameter_ with free(), so the parameter has to
  // come from malloc rather than new.
  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_AddN);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_AddN};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<float *>(output->MutableData());
  // Expected value is the sum of the per-input scales; computing it keeps the test in sync
  // with the input count instead of hard-coding the sum.
  float expected_scale = 0.0f;
  for (size_t i = 0; i < kTwoInputs; ++i) {
    expected_scale += static_cast<float>(i + 1);
  }
  for (int j = 0; j < kAddNLength; ++j) {
    ASSERT_NEAR(out_data[j], static_cast<float>(j) * expected_scale, kFp32ErrBound);
  }

  delete kernel;
  delete ctx;
  for (auto *t : inputs) {
    delete t;
  }
  for (auto *t : outputs) {
    delete t;
  }
  UninitDSPRuntime();
}

TEST_F(TestDSP_AddN, AddN_Fp32_ThreeInputs) {
  InitDSPRuntime();
  const std::vector<int> shape = {kAddNLength};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  for (size_t i = 0; i < kThreeInputs; ++i) {
    auto *input = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
    auto *input_data = reinterpret_cast<float *>(input->MutableData());
    for (int j = 0; j < kAddNLength; ++j) {
      input_data[j] = static_cast<float>(j) * static_cast<float>(i + 1);
    }
    inputs.push_back(input);
  }
  auto *output = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(kAddNLength) * sizeof(float));
  outputs.push_back(output);

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_AddN);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_AddN};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<float *>(output->MutableData());
  float expected_scale = 0.0f;
  for (size_t i = 0; i < kThreeInputs; ++i) {
    expected_scale += static_cast<float>(i + 1);
  }
  for (int j = 0; j < kAddNLength; ++j) {
    ASSERT_NEAR(out_data[j], static_cast<float>(j) * expected_scale, kFp32ErrBound);
  }

  delete kernel;
  delete ctx;
  for (auto *t : inputs) {
    delete t;
  }
  for (auto *t : outputs) {
    delete t;
  }
  UninitDSPRuntime();
}

TEST_F(TestDSP_AddN, AddN_Fp16_TwoInputs) {
  InitDSPRuntime();
  const std::vector<int> shape = {kAddNLength};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  for (size_t i = 0; i < kTwoInputs; ++i) {
    auto *input = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
    auto *input_data = reinterpret_cast<uint16_t *>(input->MutableData());
    for (int j = 0; j < kAddNLength; ++j) {
      input_data[j] = Fp32ToFp16(static_cast<float>(j) * static_cast<float>(i + 1));
    }
    inputs.push_back(input);
  }
  auto *output = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(kAddNLength) * sizeof(uint16_t));
  outputs.push_back(output);

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_AddN);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_AddN};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<uint16_t *>(output->MutableData());
  float expected_scale = 0.0f;
  for (size_t i = 0; i < kTwoInputs; ++i) {
    expected_scale += static_cast<float>(i + 1);
  }
  for (int j = 0; j < kAddNLength; ++j) {
    const float expect = Fp16ToFp32(Fp32ToFp16(static_cast<float>(j) * expected_scale));
    ASSERT_NEAR(Fp16ToFp32(out_data[j]), expect, kFp16ErrBound);
  }

  delete kernel;
  delete ctx;
  for (auto *t : inputs) {
    delete t;
  }
  for (auto *t : outputs) {
    delete t;
  }
  UninitDSPRuntime();
}

TEST_F(TestDSP_AddN, AddN_Int32_TwoInputs) {
  InitDSPRuntime();
  const std::vector<int> shape = {kAddNLength};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  for (size_t i = 0; i < kTwoInputs; ++i) {
    auto *input = new lite::Tensor(kNumberTypeInt32, shape, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
    auto *input_data = reinterpret_cast<int32_t *>(input->MutableData());
    for (int j = 0; j < kAddNLength; ++j) {
      // Distinct per-input values so that a wrong input order is detectable.
      input_data[j] = j * static_cast<int32_t>(i + 1);
    }
    inputs.push_back(input);
  }
  auto *output = new lite::Tensor(kNumberTypeInt32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(kAddNLength) * sizeof(int32_t));
  outputs.push_back(output);

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_AddN);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_AddN};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<int32_t *>(output->MutableData());
  int32_t expected_scale = 0;
  for (size_t i = 0; i < kTwoInputs; ++i) {
    expected_scale += static_cast<int32_t>(i + 1);
  }
  for (int j = 0; j < kAddNLength; ++j) {
    ASSERT_EQ(out_data[j], j * expected_scale);
  }

  delete kernel;
  delete ctx;
  for (auto *t : inputs) {
    delete t;
  }
  for (auto *t : outputs) {
    delete t;
  }
  UninitDSPRuntime();
}

TEST_F(TestDSP_AddN, AddN_Int16_TwoInputs) {
  InitDSPRuntime();
  const std::vector<int> shape = {kAddNLength};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  for (size_t i = 0; i < kTwoInputs; ++i) {
    auto *input = new lite::Tensor(kNumberTypeInt16, shape, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
    auto *input_data = reinterpret_cast<int16_t *>(input->MutableData());
    for (int j = 0; j < kAddNLength; ++j) {
      // Distinct per-input values so that a wrong input order is detectable.
      input_data[j] = static_cast<int16_t>(j * static_cast<int>(i + 1));
    }
    inputs.push_back(input);
  }
  auto *output = new lite::Tensor(kNumberTypeInt16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(kAddNLength) * sizeof(int16_t));
  outputs.push_back(output);

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_AddN);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt16, NHWC, schema::PrimitiveType_AddN};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<int16_t *>(output->MutableData());
  int16_t expected_scale = 0;
  for (size_t i = 0; i < kTwoInputs; ++i) {
    expected_scale += static_cast<int16_t>(i + 1);
  }
  for (int j = 0; j < kAddNLength; ++j) {
    ASSERT_EQ(out_data[j], static_cast<int16_t>(j * expected_scale));
  }

  delete kernel;
  delete ctx;
  for (auto *t : inputs) {
    delete t;
  }
  for (auto *t : outputs) {
    delete t;
  }
  UninitDSPRuntime();
}

TEST_F(TestDSP_AddN, AddN_Complex64_TwoInputs) {
  InitDSPRuntime();
  // A complex element is one item for the DSP kernel and two floats in the host buffer.
  const std::vector<int> shape = {kComplexLength};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  for (size_t i = 0; i < kTwoInputs; ++i) {
    auto *input = new lite::Tensor(kNumberTypeComplex64, shape, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
    auto *input_data = reinterpret_cast<float *>(input->MutableData());
    for (int j = 0; j < kComplexLength; ++j) {
      const float scale = static_cast<float>(i + 1);
      input_data[j * kComplexComponents] = static_cast<float>(j) * scale;
      input_data[j * kComplexComponents + 1] = -static_cast<float>(j) * scale;
    }
    inputs.push_back(input);
  }
  auto *output = new lite::Tensor(kNumberTypeComplex64, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(kComplexLength) * kComplexComponents * sizeof(float));
  outputs.push_back(output);

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_AddN);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeComplex64, NHWC, schema::PrimitiveType_AddN};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<float *>(output->MutableData());
  float expected_scale = 0.0f;
  for (size_t i = 0; i < kTwoInputs; ++i) {
    expected_scale += static_cast<float>(i + 1);
  }
  for (int j = 0; j < kComplexLength; ++j) {
    ASSERT_NEAR(out_data[j * kComplexComponents], static_cast<float>(j) * expected_scale, kFp32ErrBound);
    ASSERT_NEAR(out_data[j * kComplexComponents + 1], -static_cast<float>(j) * expected_scale, kFp32ErrBound);
  }

  delete kernel;
  delete ctx;
  for (auto *t : inputs) {
    delete t;
  }
  for (auto *t : outputs) {
    delete t;
  }
  UninitDSPRuntime();
}

TEST_F(TestDSP_AddN, AddN_Fp32_ScalarBroadcast) {
  InitDSPRuntime();
  const std::vector<int> shape = {kAddNLength};
  const std::vector<int> scalar_shape = {kScalarElementNum};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  // armc treats every input with one element as a broadcast operand, so the scalar is handed
  // over as its own one element tensor, like the scalar broadcast cases of the armc suite.
  auto *scalar = new lite::Tensor(kNumberTypeFloat32, scalar_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(scalar->MallocData(allocator_), lite::RET_OK);
  auto *scalar_data = reinterpret_cast<float *>(scalar->MutableData());
  scalar_data[0] = kBroadcastValue;
  inputs.push_back(scalar);
  for (size_t i = 0; i < kTwoInputs; ++i) {
    auto *input = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
    auto *input_data = reinterpret_cast<float *>(input->MutableData());
    for (int j = 0; j < kAddNLength; ++j) {
      input_data[j] = static_cast<float>(j) * static_cast<float>(i + 1);
    }
    inputs.push_back(input);
  }
  auto *output = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(kAddNLength) * sizeof(float));
  outputs.push_back(output);

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_AddN);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_AddN};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<float *>(output->MutableData());
  float expected_scale = 0.0f;
  for (size_t i = 0; i < kTwoInputs; ++i) {
    expected_scale += static_cast<float>(i + 1);
  }
  for (int j = 0; j < kAddNLength; ++j) {
    ASSERT_NEAR(out_data[j], kBroadcastValue + static_cast<float>(j) * expected_scale, kFp32ErrBound);
  }

  delete kernel;
  delete ctx;
  for (auto *t : inputs) {
    delete t;
  }
  for (auto *t : outputs) {
    delete t;
  }
  UninitDSPRuntime();
}

TEST_F(TestDSP_AddN, AddN_Fp32_ScalarBroadcastOnSecondInput) {
  InitDSPRuntime();
  const std::vector<int> shape = {kAddNLength};
  const std::vector<int> scalar_shape = {kScalarElementNum};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  // The one element operand sits on the second input here, unlike the case above where it comes
  // first: the CPU reference branches on which operand is the scalar (the first_scalar flag of
  // AddNCompute), so both positions need their own coverage.
  auto *tensor = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(tensor->MallocData(allocator_), lite::RET_OK);
  auto *tensor_data = reinterpret_cast<float *>(tensor->MutableData());
  for (int j = 0; j < kAddNLength; ++j) {
    tensor_data[j] = static_cast<float>(j);
  }
  inputs.push_back(tensor);
  auto *scalar = new lite::Tensor(kNumberTypeFloat32, scalar_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(scalar->MallocData(allocator_), lite::RET_OK);
  auto *scalar_data = reinterpret_cast<float *>(scalar->MutableData());
  scalar_data[0] = kBroadcastValue;
  inputs.push_back(scalar);
  auto *output = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(kAddNLength) * sizeof(float));
  outputs.push_back(output);

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_AddN);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_AddN};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<float *>(output->MutableData());
  for (int j = 0; j < kAddNLength; ++j) {
    ASSERT_NEAR(out_data[j], static_cast<float>(j) + kBroadcastValue, kFp32ErrBound);
  }

  delete kernel;
  delete ctx;
  for (auto *t : inputs) {
    delete t;
  }
  for (auto *t : outputs) {
    delete t;
  }
  UninitDSPRuntime();
}

TEST_F(TestDSP_AddN, AddN_SameElementCountDifferentShapeRejected) {
  InitDSPRuntime();
  // addn_func receives element counts only, so accepting this input would amount to declaring
  // {50, 2} and {100} to be the same shape. CheckSpecs compares the shapes instead and leaves a
  // graph that only matches in element count to the CPU kernel.
  const std::vector<int> shape = {kAddNLength};
  const std::vector<int> same_elements_shape = {kAddNLength / 2, 2};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  auto *input0 = new lite::Tensor(kNumberTypeFloat32, same_elements_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input0->MallocData(allocator_), lite::RET_OK);
  inputs.push_back(input0);
  auto *input1 = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input1->MallocData(allocator_), lite::RET_OK);
  inputs.push_back(input1);
  auto *output = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  outputs.push_back(output);

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_AddN);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_AddN};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  // CheckSpecs fails inside DSPKernelCreator, which deletes the kernel and frees param with it,
  // so the parameter must not be released again here.
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  EXPECT_EQ(kernel, nullptr);

  delete ctx;
  for (auto *t : inputs) {
    delete t;
  }
  for (auto *t : outputs) {
    delete t;
  }
  UninitDSPRuntime();
}

TEST_F(TestDSP_AddN, AddN_SingleInputRejected) {
  InitDSPRuntime();
  const std::vector<int> shape = {kAddNLength};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  auto *input = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  inputs.push_back(input);
  auto *output = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  outputs.push_back(output);

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_AddN);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_AddN};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  // CheckSpecs fails inside DSPKernelCreator, which deletes the kernel and frees param with it,
  // so the parameter must not be released again here.
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  EXPECT_EQ(kernel, nullptr);

  delete ctx;
  for (auto *t : inputs) {
    delete t;
  }
  for (auto *t : outputs) {
    delete t;
  }
  UninitDSPRuntime();
}
}  // namespace mindspore::lite::dsp::test
#endif  // SUPPORT_FT04
