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
constexpr int kAssignAddLength = 100;
constexpr int kComplexLength = 64;
constexpr int kComplexComponents = 2;
constexpr int kImagOffset = 1;
constexpr int kValueModulus = 7;
constexpr int kIntAddend = 5;
constexpr float kFp32ErrBound = 1e-6f;
constexpr float kFp16ErrBound = 5e-2f;
// The variable and the addend use different generators so a kernel that copies instead of
// adding, or doubles one operand, cannot pass by accident.
constexpr float kVariableScale = 0.25f;
constexpr float kAddendScale = 0.5f;
constexpr float kAddendOffset = 1.5f;
constexpr float kComplexScale = 0.5f;
constexpr float kComplexAddend = 1.0f;
}  // namespace

class TestDSP_AssignAdd : public DSPCommonTest {};

TEST_F(TestDSP_AssignAdd, AssignAdd_Fp32) {
  InitDSPRuntime();
  const std::vector<int> shape = {kAssignAddLength};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  auto *variable = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::VAR);
  ASSERT_EQ(variable->MallocData(allocator_), lite::RET_OK);
  auto *value = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(value->MallocData(allocator_), lite::RET_OK);
  // A converted Lite graph keeps the AssignAdd output as a tensor of its own, still without data
  // when the kernel is created; the update is observed through the variable input.
  auto *output = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::VAR);
  inputs.push_back(variable);
  inputs.push_back(value);
  outputs.push_back(output);

  auto *variable_data = reinterpret_cast<float *>(variable->MutableData());
  auto *value_data = reinterpret_cast<float *>(value->MutableData());
  std::vector<float> expected(kAssignAddLength);
  for (int i = 0; i < kAssignAddLength; ++i) {
    const float variable_value = static_cast<float>(i % kValueModulus) * kVariableScale;
    const float addend = static_cast<float>(i % kValueModulus) * kAddendScale + kAddendOffset;
    variable_data[i] = variable_value;
    value_data[i] = addend;
    expected[i] = variable_value + addend;
  }

  // LiteKernel::~LiteKernel() releases op_parameter_ with free(), so the parameter has to
  // come from malloc rather than new.
  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_AssignAdd);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_AssignAdd};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  for (int i = 0; i < kAssignAddLength; ++i) {
    ASSERT_NEAR(variable_data[i], expected[i], kFp32ErrBound);
  }

  delete kernel;
  delete ctx;
  delete variable;
  delete value;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_AssignAdd, AssignAdd_Fp16) {
  InitDSPRuntime();
  const std::vector<int> shape = {kAssignAddLength};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  auto *variable = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::VAR);
  ASSERT_EQ(variable->MallocData(allocator_), lite::RET_OK);
  auto *value = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(value->MallocData(allocator_), lite::RET_OK);
  // A converted Lite graph keeps the AssignAdd output as a tensor of its own, still without data
  // when the kernel is created; the update is observed through the variable input.
  auto *output = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::VAR);
  inputs.push_back(variable);
  inputs.push_back(value);
  outputs.push_back(output);

  auto *variable_data = reinterpret_cast<uint16_t *>(variable->MutableData());
  auto *value_data = reinterpret_cast<uint16_t *>(value->MutableData());
  std::vector<float> expected(kAssignAddLength);
  for (int i = 0; i < kAssignAddLength; ++i) {
    const float variable_value = static_cast<float>(i % kValueModulus) * kVariableScale;
    const float addend = static_cast<float>(i % kValueModulus) * kAddendScale + kAddendOffset;
    variable_data[i] = Fp32ToFp16(variable_value);
    value_data[i] = Fp32ToFp16(addend);
    expected[i] = variable_value + addend;
  }

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_AssignAdd);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_AssignAdd};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  for (int i = 0; i < kAssignAddLength; ++i) {
    ASSERT_NEAR(Fp16ToFp32(variable_data[i]), expected[i], kFp16ErrBound);
  }

  delete kernel;
  delete ctx;
  delete variable;
  delete value;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_AssignAdd, AssignAdd_Int32) {
  InitDSPRuntime();
  const std::vector<int> shape = {kAssignAddLength};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  auto *variable = new lite::Tensor(kNumberTypeInt32, shape, NHWC, lite::Category::VAR);
  ASSERT_EQ(variable->MallocData(allocator_), lite::RET_OK);
  auto *value = new lite::Tensor(kNumberTypeInt32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(value->MallocData(allocator_), lite::RET_OK);
  // A converted Lite graph keeps the AssignAdd output as a tensor of its own, still without data
  // when the kernel is created; the update is observed through the variable input.
  auto *output = new lite::Tensor(kNumberTypeInt32, shape, NHWC, lite::Category::VAR);
  inputs.push_back(variable);
  inputs.push_back(value);
  outputs.push_back(output);

  auto *variable_data = reinterpret_cast<int32_t *>(variable->MutableData());
  auto *value_data = reinterpret_cast<int32_t *>(value->MutableData());
  std::vector<int32_t> expected(kAssignAddLength);
  for (int i = 0; i < kAssignAddLength; ++i) {
    const int32_t variable_value = i % kValueModulus;
    const int32_t addend = variable_value + kIntAddend;
    variable_data[i] = variable_value;
    value_data[i] = addend;
    expected[i] = variable_value + addend;
  }

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_AssignAdd);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_AssignAdd};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  for (int i = 0; i < kAssignAddLength; ++i) {
    ASSERT_EQ(variable_data[i], expected[i]);
  }

  delete kernel;
  delete ctx;
  delete variable;
  delete value;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_AssignAdd, AssignAdd_Int16) {
  InitDSPRuntime();
  const std::vector<int> shape = {kAssignAddLength};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  auto *variable = new lite::Tensor(kNumberTypeInt16, shape, NHWC, lite::Category::VAR);
  ASSERT_EQ(variable->MallocData(allocator_), lite::RET_OK);
  auto *value = new lite::Tensor(kNumberTypeInt16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(value->MallocData(allocator_), lite::RET_OK);
  // A converted Lite graph keeps the AssignAdd output as a tensor of its own, still without data
  // when the kernel is created; the update is observed through the variable input.
  auto *output = new lite::Tensor(kNumberTypeInt16, shape, NHWC, lite::Category::VAR);
  inputs.push_back(variable);
  inputs.push_back(value);
  outputs.push_back(output);

  auto *variable_data = reinterpret_cast<int16_t *>(variable->MutableData());
  auto *value_data = reinterpret_cast<int16_t *>(value->MutableData());
  std::vector<int16_t> expected(kAssignAddLength);
  for (int i = 0; i < kAssignAddLength; ++i) {
    const int16_t variable_value = static_cast<int16_t>(i % kValueModulus);
    const int16_t addend = static_cast<int16_t>(variable_value + kIntAddend);
    variable_data[i] = variable_value;
    value_data[i] = addend;
    expected[i] = static_cast<int16_t>(variable_value + addend);
  }

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_AssignAdd);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt16, NHWC, schema::PrimitiveType_AssignAdd};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  for (int i = 0; i < kAssignAddLength; ++i) {
    ASSERT_EQ(variable_data[i], expected[i]);
  }

  delete kernel;
  delete ctx;
  delete variable;
  delete value;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_AssignAdd, AssignAdd_Complex64) {
  InitDSPRuntime();
  // A complex element is one item for the DSP kernel, two floats in the host buffer.
  const std::vector<int> shape = {kComplexLength};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  auto *variable = new lite::Tensor(kNumberTypeComplex64, shape, NHWC, lite::Category::VAR);
  ASSERT_EQ(variable->MallocData(allocator_), lite::RET_OK);
  auto *value = new lite::Tensor(kNumberTypeComplex64, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(value->MallocData(allocator_), lite::RET_OK);
  // A converted Lite graph keeps the AssignAdd output as a tensor of its own, still without data
  // when the kernel is created; the update is observed through the variable input.
  auto *output = new lite::Tensor(kNumberTypeComplex64, shape, NHWC, lite::Category::VAR);
  inputs.push_back(variable);
  inputs.push_back(value);
  outputs.push_back(output);

  const int component_num = kComplexLength * kComplexComponents;
  auto *variable_data = reinterpret_cast<float *>(variable->MutableData());
  auto *value_data = reinterpret_cast<float *>(value->MutableData());
  std::vector<float> expected(component_num);
  for (int i = 0; i < kComplexLength; ++i) {
    const float base = static_cast<float>(i % kValueModulus) * kComplexScale;
    variable_data[i * kComplexComponents] = base;
    variable_data[i * kComplexComponents + kImagOffset] = base + kAddendOffset;
    value_data[i * kComplexComponents] = base + kComplexAddend;
    value_data[i * kComplexComponents + kImagOffset] = base - kComplexAddend;
    expected[i * kComplexComponents] = variable_data[i * kComplexComponents] + value_data[i * kComplexComponents];
    expected[i * kComplexComponents + kImagOffset] =
      variable_data[i * kComplexComponents + kImagOffset] + value_data[i * kComplexComponents + kImagOffset];
  }

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_AssignAdd);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeComplex64, NHWC, schema::PrimitiveType_AssignAdd};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  for (int i = 0; i < component_num; ++i) {
    ASSERT_NEAR(variable_data[i], expected[i], kFp32ErrBound);
  }

  delete kernel;
  delete ctx;
  delete variable;
  delete value;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_AssignAdd, AssignAdd_ValueElementsMismatchRejected) {
  InitDSPRuntime();
  const std::vector<int> shape = {kAssignAddLength};
  const std::vector<int> short_shape = {kAssignAddLength - 1};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  auto *variable = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::VAR);
  ASSERT_EQ(variable->MallocData(allocator_), lite::RET_OK);
  auto *value = new lite::Tensor(kNumberTypeFloat32, short_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(value->MallocData(allocator_), lite::RET_OK);
  auto *output = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::VAR);
  inputs.push_back(variable);
  inputs.push_back(value);
  outputs.push_back(output);

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_AssignAdd);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_AssignAdd};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  // CheckSpecs fails inside DSPKernelCreator, which deletes the kernel and frees param with it,
  // so the parameter must not be released again here.
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  EXPECT_EQ(kernel, nullptr);

  delete ctx;
  delete variable;
  delete value;
  delete output;
  UninitDSPRuntime();
}
}  // namespace mindspore::lite::dsp::test
#endif  // SUPPORT_FT04
