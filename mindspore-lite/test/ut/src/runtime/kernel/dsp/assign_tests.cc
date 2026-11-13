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
constexpr int kAssignLength = 100;
constexpr int kComplexLength = 64;
constexpr int kComplexComponents = 2;
constexpr float kFp32ErrBound = 1e-6f;
constexpr float kFp16ErrBound = 5e-2f;
}  // namespace

class TestDSP_Assign : public DSPCommonTest {};

TEST_F(TestDSP_Assign, Assign_Fp32) {
  InitDSPRuntime();
  const std::vector<int> shape = {kAssignLength};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  // Assign is in-place: the variable tensor is both input[0] and output[0].
  auto *variable = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::VAR);
  ASSERT_EQ(variable->MallocData(allocator_), lite::RET_OK);
  auto *value = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(value->MallocData(allocator_), lite::RET_OK);
  inputs.push_back(variable);
  inputs.push_back(value);
  outputs.push_back(variable);
  std::memset(variable->MutableData(), 0, static_cast<size_t>(kAssignLength) * sizeof(float));
  auto *value_data = reinterpret_cast<float *>(value->MutableData());
  for (int i = 0; i < kAssignLength; ++i) {
    value_data[i] = static_cast<float>(i) + 1.0f;
  }

  // LiteKernel::~LiteKernel() releases op_parameter_ with free(), so the parameter has to
  // come from malloc rather than new.
  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_Assign);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Assign};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *variable_data = reinterpret_cast<float *>(variable->MutableData());
  for (int i = 0; i < kAssignLength; ++i) {
    ASSERT_NEAR(variable_data[i], value_data[i], kFp32ErrBound);
  }

  delete kernel;
  delete ctx;
  delete variable;
  delete value;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Assign, Assign_Fp16) {
  InitDSPRuntime();
  const std::vector<int> shape = {kAssignLength};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  auto *variable = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::VAR);
  ASSERT_EQ(variable->MallocData(allocator_), lite::RET_OK);
  auto *value = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(value->MallocData(allocator_), lite::RET_OK);
  inputs.push_back(variable);
  inputs.push_back(value);
  outputs.push_back(variable);
  std::memset(variable->MutableData(), 0, static_cast<size_t>(kAssignLength) * sizeof(uint16_t));
  auto *value_data = reinterpret_cast<uint16_t *>(value->MutableData());
  for (int i = 0; i < kAssignLength; ++i) {
    value_data[i] = Fp32ToFp16(static_cast<float>(i) + 1.0f);
  }

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_Assign);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_Assign};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *variable_data = reinterpret_cast<uint16_t *>(variable->MutableData());
  for (int i = 0; i < kAssignLength; ++i) {
    ASSERT_NEAR(Fp16ToFp32(variable_data[i]), Fp16ToFp32(value_data[i]), kFp16ErrBound);
  }

  delete kernel;
  delete ctx;
  delete variable;
  delete value;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Assign, Assign_Int32) {
  InitDSPRuntime();
  const std::vector<int> shape = {kAssignLength};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  auto *variable = new lite::Tensor(kNumberTypeInt32, shape, NHWC, lite::Category::VAR);
  ASSERT_EQ(variable->MallocData(allocator_), lite::RET_OK);
  auto *value = new lite::Tensor(kNumberTypeInt32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(value->MallocData(allocator_), lite::RET_OK);
  inputs.push_back(variable);
  inputs.push_back(value);
  outputs.push_back(variable);
  std::memset(variable->MutableData(), 0, static_cast<size_t>(kAssignLength) * sizeof(int32_t));
  auto *value_data = reinterpret_cast<int32_t *>(value->MutableData());
  for (int i = 0; i < kAssignLength; ++i) {
    value_data[i] = i - kAssignLength;
  }

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_Assign);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_Assign};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *variable_data = reinterpret_cast<int32_t *>(variable->MutableData());
  for (int i = 0; i < kAssignLength; ++i) {
    ASSERT_EQ(variable_data[i], value_data[i]);
  }

  delete kernel;
  delete ctx;
  delete variable;
  delete value;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Assign, Assign_Int16) {
  InitDSPRuntime();
  const std::vector<int> shape = {kAssignLength};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  auto *variable = new lite::Tensor(kNumberTypeInt16, shape, NHWC, lite::Category::VAR);
  ASSERT_EQ(variable->MallocData(allocator_), lite::RET_OK);
  auto *value = new lite::Tensor(kNumberTypeInt16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(value->MallocData(allocator_), lite::RET_OK);
  inputs.push_back(variable);
  inputs.push_back(value);
  outputs.push_back(variable);
  std::memset(variable->MutableData(), 0, static_cast<size_t>(kAssignLength) * sizeof(int16_t));
  auto *value_data = reinterpret_cast<int16_t *>(value->MutableData());
  for (int i = 0; i < kAssignLength; ++i) {
    value_data[i] = static_cast<int16_t>(i - kAssignLength);
  }

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_Assign);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt16, NHWC, schema::PrimitiveType_Assign};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *variable_data = reinterpret_cast<int16_t *>(variable->MutableData());
  for (int i = 0; i < kAssignLength; ++i) {
    ASSERT_EQ(variable_data[i], value_data[i]);
  }

  delete kernel;
  delete ctx;
  delete variable;
  delete value;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Assign, Assign_Complex64) {
  InitDSPRuntime();
  // A complex element is one item for the DSP kernel and two floats in the host buffer.
  const std::vector<int> shape = {kComplexLength};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  auto *variable = new lite::Tensor(kNumberTypeComplex64, shape, NHWC, lite::Category::VAR);
  ASSERT_EQ(variable->MallocData(allocator_), lite::RET_OK);
  auto *value = new lite::Tensor(kNumberTypeComplex64, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(value->MallocData(allocator_), lite::RET_OK);
  inputs.push_back(variable);
  inputs.push_back(value);
  outputs.push_back(variable);
  const int component_num = kComplexLength * kComplexComponents;
  auto *variable_data = reinterpret_cast<float *>(variable->MutableData());
  auto *value_data = reinterpret_cast<float *>(value->MutableData());
  for (int i = 0; i < kComplexLength; ++i) {
    const float base = static_cast<float>(i) * 0.5f;
    variable_data[i * kComplexComponents] = 0.0f;
    variable_data[i * kComplexComponents + 1] = 0.0f;
    value_data[i * kComplexComponents] = base + 1.0f;
    value_data[i * kComplexComponents + 1] = -base;
  }

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_Assign);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeComplex64, NHWC, schema::PrimitiveType_Assign};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, param, ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  for (int i = 0; i < component_num; ++i) {
    ASSERT_NEAR(variable_data[i], value_data[i], kFp32ErrBound);
  }

  delete kernel;
  delete ctx;
  delete variable;
  delete value;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Assign, Assign_SeparateOutputRejected) {
  InitDSPRuntime();
  const std::vector<int> shape = {kAssignLength};
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  auto *variable = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::VAR);
  ASSERT_EQ(variable->MallocData(allocator_), lite::RET_OK);
  auto *value = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(value->MallocData(allocator_), lite::RET_OK);
  // A buffer of its own makes this tensor a copy destination instead of the updated variable,
  // which the DSP path cannot serve: the kernel has to be rejected so that the scheduler keeps
  // this node on the CPU kernel rather than returning an output nobody writes.
  auto *output = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::VAR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  inputs.push_back(variable);
  inputs.push_back(value);
  outputs.push_back(output);

  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_Assign);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Assign};
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
