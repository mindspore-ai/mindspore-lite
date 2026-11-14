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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include "ut/src/runtime/kernel/dsp/dsp_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "include/api/model.h"
#include "nnacl_c/activation_parameter.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel/dsp/dsp_subgraph.h"
#include "src/litert/kernel_registry.h"

#ifdef SUPPORT_FT04

namespace mindspore::lite::dsp::test {

namespace {
constexpr float kInputRangeStart = -5.0f;
constexpr float kInputRangeEnd = 5.0f;

void FillLinearInput(float *data, int num) {
  for (int i = 0; i < num; ++i) {
    data[i] =
      kInputRangeStart + (kInputRangeEnd - kInputRangeStart) * static_cast<float>(i) / (static_cast<float>(num) - 1.0f);
  }
}

float RefRelu(float x) { return (x > 0.0f) ? x : 0.0f; }

float RefRelu6(float x) {
  float v = (x > 0.0f) ? x : 0.0f;
  return (v < 6.0f) ? v : 6.0f;
}

float RefLeakyRelu(float x, float alpha) { return (x >= 0.0f) ? x : alpha * x; }

float RefSigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

float RefTanh(float x) { return tanhf(x); }

float RefSwish(float x) { return x / (1.0f + expf(-x)); }

float RefHSwish(float x) {
  float v = (x + 3.0f) / 6.0f;
  if (v < 0.0f) {
    v = 0.0f;
  }
  if (v > 1.0f) {
    v = 1.0f;
  }
  return x * v;
}

float RefHSigmoid(float x) {
  float v = (x + 3.0f) / 6.0f;
  if (v < 0.0f) {
    v = 0.0f;
  }
  if (v > 1.0f) {
    v = 1.0f;
  }
  return v;
}

float RefHardTanh(float x, float min_val, float max_val) {
  if (x < min_val) {
    return min_val;
  }
  if (x > max_val) {
    return max_val;
  }
  return x;
}

float RefGelu(float x) {
  float c = 0.044715f * x * x * x;
  return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + c)));
}

float RefSoftplus(float x) { return (x > 88.0f) ? x : logf(1.0f + expf(x)); }

float RefElu(float x, float alpha) { return (x >= 0.0f) ? x : alpha * (expf(x) - 1.0f); }

float RefCelu(float x, float alpha) { return (x >= 0.0f) ? x : alpha * (expf(x / alpha) - 1.0f); }

ActivationParameter *CreateActivationParam(schema::ActivationType type, float alpha = 0.0f, float min_val = 0.0f,
                                           float max_val = 0.0f, bool approximate = false) {
  // LiteKernel::~LiteKernel() releases op_parameter_ with free(), so the parameter has
  // to be allocated by malloc: a new'ed parameter would be freed with the wrong
  // deallocator (and leaked, since the kernel owns it).
  auto *param = static_cast<ActivationParameter *>(malloc(sizeof(ActivationParameter)));
  if (param == nullptr) {
    return nullptr;
  }
  (void)memset(param, 0, sizeof(ActivationParameter));
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_Activation);
  param->type_ = static_cast<int>(type);
  param->alpha_ = alpha;
  param->min_val_ = min_val;
  param->max_val_ = max_val;
  param->approximate_ = approximate;
  param->beta_ = 0.0f;
  return param;
}
}  // namespace

class TestDSP_Activation : public DSPCommonTest {};
TEST_F(TestDSP_Activation, Activation_Relu_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationParam(schema::ActivationType_RELU);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Activation};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto in_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefRelu(in_data[i]);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
  // The runtime must outlive the tensors and the kernel: both release device memory
  // through the DSP allocator owned by the runtime.
  UninitDSPRuntime();
}

TEST_F(TestDSP_Activation, Activation_Relu6_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationParam(schema::ActivationType_RELU6);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Activation};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto in_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefRelu6(in_data[i]);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
  // The runtime must outlive the tensors and the kernel: both release device memory
  // through the DSP allocator owned by the runtime.
  UninitDSPRuntime();
}

TEST_F(TestDSP_Activation, Activation_LeakyRelu_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationParam(schema::ActivationType_LEAKY_RELU, 0.1f);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Activation};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto in_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefLeakyRelu(in_data[i], 0.1f);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
  // The runtime must outlive the tensors and the kernel: both release device memory
  // through the DSP allocator owned by the runtime.
  UninitDSPRuntime();
}

TEST_F(TestDSP_Activation, Activation_Sigmoid_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationParam(schema::ActivationType_SIGMOID);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Activation};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto in_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefSigmoid(in_data[i]);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
  // The runtime must outlive the tensors and the kernel: both release device memory
  // through the DSP allocator owned by the runtime.
  UninitDSPRuntime();
}

TEST_F(TestDSP_Activation, Activation_Tanh_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationParam(schema::ActivationType_TANH);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Activation};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto in_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefTanh(in_data[i]);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-3));
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
  // The runtime must outlive the tensors and the kernel: both release device memory
  // through the DSP allocator owned by the runtime.
  UninitDSPRuntime();
}

TEST_F(TestDSP_Activation, Activation_Swish_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationParam(schema::ActivationType_SWISH);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Activation};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto in_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefSwish(in_data[i]);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
  // The runtime must outlive the tensors and the kernel: both release device memory
  // through the DSP allocator owned by the runtime.
  UninitDSPRuntime();
}

TEST_F(TestDSP_Activation, Activation_HSwish_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationParam(schema::ActivationType_HSWISH);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Activation};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto in_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefHSwish(in_data[i]);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
  // The runtime must outlive the tensors and the kernel: both release device memory
  // through the DSP allocator owned by the runtime.
  UninitDSPRuntime();
}

TEST_F(TestDSP_Activation, Activation_HSigmoid_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationParam(schema::ActivationType_HSIGMOID);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Activation};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto in_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefHSigmoid(in_data[i]);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
  // The runtime must outlive the tensors and the kernel: both release device memory
  // through the DSP allocator owned by the runtime.
  UninitDSPRuntime();
}

TEST_F(TestDSP_Activation, Activation_HardTanh_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationParam(schema::ActivationType_HARD_TANH, 0.0f, -1.0f, 1.0f);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Activation};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto in_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefHardTanh(in_data[i], -1.0f, 1.0f);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
  // The runtime must outlive the tensors and the kernel: both release device memory
  // through the DSP allocator owned by the runtime.
  UninitDSPRuntime();
}

TEST_F(TestDSP_Activation, Activation_Gelu_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationParam(schema::ActivationType_GELU, 0.0f, 0.0f, 0.0f, true);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Activation};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto in_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefGelu(in_data[i]);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-3));
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
  // The runtime must outlive the tensors and the kernel: both release device memory
  // through the DSP allocator owned by the runtime.
  UninitDSPRuntime();
}

TEST_F(TestDSP_Activation, Activation_Softplus_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationParam(schema::ActivationType_SOFTPLUS);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Activation};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto in_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefSoftplus(in_data[i]);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
  // The runtime must outlive the tensors and the kernel: both release device memory
  // through the DSP allocator owned by the runtime.
  UninitDSPRuntime();
}

TEST_F(TestDSP_Activation, Activation_Elu_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationParam(schema::ActivationType_ELU, 1.0f);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Activation};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto in_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefElu(in_data[i], 1.0f);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
  // The runtime must outlive the tensors and the kernel: both release device memory
  // through the DSP allocator owned by the runtime.
  UninitDSPRuntime();
}

TEST_F(TestDSP_Activation, Activation_Celu_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationParam(schema::ActivationType_CELU, 1.0f);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Activation};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto in_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefCelu(in_data[i], 1.0f);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
  // The runtime must outlive the tensors and the kernel: both release device memory
  // through the DSP allocator owned by the runtime.
  UninitDSPRuntime();
}

TEST_F(TestDSP_Activation, Activation_Relu_Fp16) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationParam(schema::ActivationType_RELU);
  auto input = new lite::Tensor(kNumberTypeFloat16, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat16, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::vector<float> input_data(num);
  FillLinearInput(input_data.data(), num);
  auto input_fp16 = reinterpret_cast<uint16_t *>(input->MutableData());
  for (int i = 0; i < num; ++i) {
    input_fp16[i] = Fp32ToFp16(input_data[i]);
  }
  std::memset(output->MutableData(), 0, static_cast<size_t>(num) * sizeof(uint16_t));

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_Activation};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefRelu(Fp16ToFp32(input_fp16[i]));
  }
  std::vector<float> actual(num);
  auto output_fp16 = reinterpret_cast<uint16_t *>(outputs_[0]->MutableData());
  for (int i = 0; i < num; ++i) {
    actual[i] = Fp16ToFp32(output_fp16[i]);
  }
  ASSERT_EQ(0, CompareOutputData(actual.data(), correct.data(), num, 1e-1));
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
  // The runtime must outlive the tensors and the kernel: both release device memory
  // through the DSP allocator owned by the runtime.
  UninitDSPRuntime();
}

TEST_F(TestDSP_Activation, Activation_Sigmoid_Fp16) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationParam(schema::ActivationType_SIGMOID);
  auto input = new lite::Tensor(kNumberTypeFloat16, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat16, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::vector<float> input_data(num);
  FillLinearInput(input_data.data(), num);
  auto input_fp16 = reinterpret_cast<uint16_t *>(input->MutableData());
  for (int i = 0; i < num; ++i) {
    input_fp16[i] = Fp32ToFp16(input_data[i]);
  }
  std::memset(output->MutableData(), 0, static_cast<size_t>(num) * sizeof(uint16_t));

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_Activation};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefSigmoid(Fp16ToFp32(input_fp16[i]));
  }
  std::vector<float> actual(num);
  auto output_fp16 = reinterpret_cast<uint16_t *>(outputs_[0]->MutableData());
  for (int i = 0; i < num; ++i) {
    actual[i] = Fp16ToFp32(output_fp16[i]);
  }
  ASSERT_EQ(0, CompareOutputData(actual.data(), correct.data(), num, 1e-1));
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
  // The runtime must outlive the tensors and the kernel: both release device memory
  // through the DSP allocator owned by the runtime.
  UninitDSPRuntime();
}

// Negative cases: the DSP kernel is registered for the whole PrimitiveType_Activation
// family, so the checks in CheckSpecs must reject unsupported graphs at creation time
// (letting the scheduler fall back to the CPU kernel) instead of failing on Run.
TEST_F(TestDSP_Activation, Activation_UnsupportedType_Rejected) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  std::vector<int> shape = {10, 10};
  auto param = CreateActivationParam(schema::ActivationType_LINEAR);
  ASSERT_NE(param, nullptr);
  auto input = new lite::Tensor(kNumberTypeFloat32, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs.push_back(output);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Activation};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
  EXPECT_EQ(kernel, nullptr);
  delete kernel;
  delete ctx;
  delete input;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Activation, Activation_InvalidTensorCount_Rejected) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  std::vector<int> shape = {10, 10};
  auto param = CreateActivationParam(schema::ActivationType_RELU);
  ASSERT_NE(param, nullptr);
  auto input = new lite::Tensor(kNumberTypeFloat32, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs.push_back(input);
  // A second input is invalid for Activation: CheckSpecs must reject it.
  auto extra = new lite::Tensor(kNumberTypeFloat32, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  extra->MallocData(allocator_);
  inputs.push_back(extra);
  auto output = new lite::Tensor(kNumberTypeFloat32, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs.push_back(output);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Activation};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
  EXPECT_EQ(kernel, nullptr);
  delete kernel;
  delete ctx;
  delete input;
  delete extra;
  delete output;
  UninitDSPRuntime();
}

}  // namespace mindspore::lite::dsp::test
#endif
