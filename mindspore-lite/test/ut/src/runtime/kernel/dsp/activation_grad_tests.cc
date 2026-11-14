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
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include "ut/src/runtime/kernel/dsp/dsp_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "include/api/model.h"
#include "nnacl_c/fp32_grad/activation_grad_fp32.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel/dsp/dsp_subgraph.h"
#include "src/litert/kernel_registry.h"

#ifdef SUPPORT_FT04

namespace mindspore::lite::dsp::test {

namespace {
constexpr float kGradRangeStart = -3.0f;
constexpr float kGradRangeEnd = 3.0f;

void FillLinearInput(float *data, int num) {
  for (int i = 0; i < num; ++i) {
    data[i] =
      kGradRangeStart + (kGradRangeEnd - kGradRangeStart) * static_cast<float>(i) / (static_cast<float>(num) - 1.0f);
  }
}

void FillRangeInput(float *data, int num, float start, float end) {
  for (int i = 0; i < num; ++i) {
    data[i] = start + (end - start) * static_cast<float>(i) / (static_cast<float>(num) - 1.0f);
  }
}

float RefReluGrad(float dy, float y) { return (y > 0.0f) ? dy : 0.0f; }

float RefRelu6Grad(float dy, float y) { return (y > 0.0f && y <= 6.0f) ? dy : 0.0f; }

float RefLeakyReluGrad(float dy, float y, float alpha) { return (y > 0.0f) ? dy : alpha * dy; }

float RefSigmoidGrad(float dy, float y) { return dy * y * (1.0f - y); }

float RefTanhGrad(float dy, float y) { return dy * (1.0f - y * y); }

float RefHSwishGrad(float dy, float y) {
  float h = (2.0f * y + 3.0f) / 6.0f;
  if (y < -3.0f) {
    h = 0.0f;
  } else if (y > 3.0f) {
    h = 1.0f;
  }
  return dy * h;
}

float RefHSigmoidGrad(float dy, float y) {
  if (y < -3.0f || y > 3.0f) {
    return 0.0f;
  }
  return dy / 6.0f;
}

float RefGeluGrad(float dy, float y) {
  float sqrt2 = 1.4142135623730951f;
  float sqrt2pi = 2.5066282746310002f;
  float term1 = 0.5f * (1.0f + erff(y / sqrt2));
  float term2 = y * expf(-y * y / 2.0f) / sqrt2pi;
  return dy * (term1 + term2);
}

float RefEluGrad(float dy, float y, float alpha) { return (y > 0.0f) ? dy : alpha * expf(y) * dy; }

float RefSoftplusGrad(float dy, float y) { return dy / (1.0f + expf(-y)); }

float RefCeluGrad(float dy, float y, float alpha) { return (y > 0.0f) ? dy : dy * expf(y / alpha); }

float RefHardTanhGrad(float dy, float y, float min_val, float max_val) {
  return (y > min_val && y <= max_val) ? dy : 0.0f;
}

ActivationGradParameter *CreateActivationGradParam(schema::ActivationType type, float alpha = 0.0f) {
  auto *param = new ActivationGradParameter();
  param->op_parameter.type_ = static_cast<int>(schema::PrimitiveType_ActivationGrad);
  param->type_ = static_cast<int>(type);
  param->alpha_ = alpha;
  return param;
}
}  // namespace

class TestDSP_ActivationGrad : public DSPCommonTest {};
TEST_F(TestDSP_ActivationGrad, ActivationGrad_Relu_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationGradParam(schema::ActivationType_RELU);
  auto grad = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  grad->MallocData(allocator_);
  inputs_.push_back(grad);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(grad->MutableData()), num);
  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_ActivationGrad};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto grad_data = reinterpret_cast<float *>(grad->MutableData());
  auto input_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefReluGrad(grad_data[i], input_data[i]);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_ActivationGrad, ActivationGrad_Relu6_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationGradParam(schema::ActivationType_RELU6);
  auto grad = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  grad->MallocData(allocator_);
  inputs_.push_back(grad);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(grad->MutableData()), num);
  FillRangeInput(reinterpret_cast<float *>(input->MutableData()), num, -1.0f, 6.5f);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_ActivationGrad};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto grad_data = reinterpret_cast<float *>(grad->MutableData());
  auto input_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefRelu6Grad(grad_data[i], input_data[i]);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_ActivationGrad, ActivationGrad_LeakyRelu_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationGradParam(schema::ActivationType_LEAKY_RELU, 0.1f);
  auto grad = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  grad->MallocData(allocator_);
  inputs_.push_back(grad);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(grad->MutableData()), num);
  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_ActivationGrad};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto grad_data = reinterpret_cast<float *>(grad->MutableData());
  auto input_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefLeakyReluGrad(grad_data[i], input_data[i], 0.1f);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_ActivationGrad, ActivationGrad_Sigmoid_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationGradParam(schema::ActivationType_SIGMOID);
  auto grad = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  grad->MallocData(allocator_);
  inputs_.push_back(grad);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(grad->MutableData()), num);
  auto input_data = reinterpret_cast<float *>(input->MutableData());
  for (int i = 0; i < num; ++i) {
    input_data[i] = 0.3f + 0.4f * static_cast<float>(i) / (static_cast<float>(num) - 1.0f);
  }
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_ActivationGrad};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  std::vector<float> correct(num);
  auto grad_data = reinterpret_cast<float *>(grad->MutableData());
  for (int i = 0; i < num; ++i) {
    correct[i] = RefSigmoidGrad(grad_data[i], input_data[i]);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_ActivationGrad, ActivationGrad_Tanh_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationGradParam(schema::ActivationType_TANH);
  auto grad = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  grad->MallocData(allocator_);
  inputs_.push_back(grad);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(grad->MutableData()), num);
  auto input_data = reinterpret_cast<float *>(input->MutableData());
  for (int i = 0; i < num; ++i) {
    input_data[i] = tanhf(0.8f + 0.2f * static_cast<float>(i) / (static_cast<float>(num) - 1.0f));
  }
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_ActivationGrad};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  std::vector<float> correct(num);
  auto grad_data = reinterpret_cast<float *>(grad->MutableData());
  for (int i = 0; i < num; ++i) {
    correct[i] = RefTanhGrad(grad_data[i], input_data[i]);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_ActivationGrad, ActivationGrad_HSwish_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationGradParam(schema::ActivationType_HSWISH);
  auto grad = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  grad->MallocData(allocator_);
  inputs_.push_back(grad);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(grad->MutableData()), num);
  FillRangeInput(reinterpret_cast<float *>(input->MutableData()), num, -4.0f, 4.0f);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_ActivationGrad};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto grad_data = reinterpret_cast<float *>(grad->MutableData());
  auto input_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefHSwishGrad(grad_data[i], input_data[i]);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_ActivationGrad, ActivationGrad_HSigmoid_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationGradParam(schema::ActivationType_HSIGMOID);
  auto grad = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  grad->MallocData(allocator_);
  inputs_.push_back(grad);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(grad->MutableData()), num);
  FillRangeInput(reinterpret_cast<float *>(input->MutableData()), num, -4.0f, 4.0f);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_ActivationGrad};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto grad_data = reinterpret_cast<float *>(grad->MutableData());
  auto input_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefHSigmoidGrad(grad_data[i], input_data[i]);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_ActivationGrad, ActivationGrad_Gelu_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationGradParam(schema::ActivationType_GELU);
  auto grad = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  grad->MallocData(allocator_);
  inputs_.push_back(grad);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(grad->MutableData()), num);
  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_ActivationGrad};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto grad_data = reinterpret_cast<float *>(grad->MutableData());
  auto input_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefGeluGrad(grad_data[i], input_data[i]);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 5e-3));
  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_ActivationGrad, ActivationGrad_Elu_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationGradParam(schema::ActivationType_ELU, 1.0f);
  auto grad = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  grad->MallocData(allocator_);
  inputs_.push_back(grad);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(grad->MutableData()), num);
  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_ActivationGrad};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto grad_data = reinterpret_cast<float *>(grad->MutableData());
  auto input_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefEluGrad(grad_data[i], input_data[i], 1.0f);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_ActivationGrad, ActivationGrad_Softplus_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationGradParam(schema::ActivationType_SOFTPLUS);
  auto grad = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  grad->MallocData(allocator_);
  inputs_.push_back(grad);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(grad->MutableData()), num);
  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_ActivationGrad};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto grad_data = reinterpret_cast<float *>(grad->MutableData());
  auto input_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefSoftplusGrad(grad_data[i], input_data[i]);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_ActivationGrad, ActivationGrad_Celu_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  // alpha != 1 keeps CELU distinct from ELU (which is passed alpha=1 above), so this
  // case really exercises the celu_grad_func dispatch and the alpha forwarding.
  auto param = CreateActivationGradParam(schema::ActivationType_CELU, 2.0f);
  auto grad = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  grad->MallocData(allocator_);
  inputs_.push_back(grad);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(grad->MutableData()), num);
  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_ActivationGrad};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto grad_data = reinterpret_cast<float *>(grad->MutableData());
  auto input_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefCeluGrad(grad_data[i], input_data[i], 2.0f);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_ActivationGrad, ActivationGrad_HardTanh_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationGradParam(schema::ActivationType_HARD_TANH);
  auto grad = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  grad->MallocData(allocator_);
  inputs_.push_back(grad);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  FillLinearInput(reinterpret_cast<float *>(grad->MutableData()), num);
  FillLinearInput(reinterpret_cast<float *>(input->MutableData()), num);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_ActivationGrad};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  auto grad_data = reinterpret_cast<float *>(grad->MutableData());
  auto input_data = reinterpret_cast<float *>(input->MutableData());
  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefHardTanhGrad(grad_data[i], input_data[i], -1.0f, 1.0f);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(), num, 1e-4));
  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_ActivationGrad, ActivationGrad_Sigmoid_Fp16) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationGradParam(schema::ActivationType_SIGMOID);
  auto grad = new lite::Tensor(kNumberTypeFloat16, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  grad->MallocData(allocator_);
  inputs_.push_back(grad);
  auto input = new lite::Tensor(kNumberTypeFloat16, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat16, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::vector<float> grad_data(num);
  FillLinearInput(grad_data.data(), num);
  std::vector<float> input_data(num);
  for (int i = 0; i < num; ++i) {
    input_data[i] = 0.3f + 0.4f * static_cast<float>(i) / (static_cast<float>(num) - 1.0f);
  }
  auto grad_fp16 = reinterpret_cast<uint16_t *>(grad->MutableData());
  auto input_fp16 = reinterpret_cast<uint16_t *>(input->MutableData());
  for (int i = 0; i < num; ++i) {
    grad_fp16[i] = Fp32ToFp16(grad_data[i]);
    input_fp16[i] = Fp32ToFp16(input_data[i]);
  }
  std::memset(output->MutableData(), 0, static_cast<size_t>(num) * sizeof(uint16_t));

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_ActivationGrad};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefSigmoidGrad(grad_data[i], input_data[i]);
  }
  std::vector<float> actual(num);
  auto output_fp16 = reinterpret_cast<uint16_t *>(outputs_[0]->MutableData());
  for (int i = 0; i < num; ++i) {
    actual[i] = Fp16ToFp32(output_fp16[i]);
  }
  ASSERT_EQ(0, CompareOutputData(actual.data(), correct.data(), num, 1e-1));
  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_ActivationGrad, ActivationGrad_Relu_Fp16) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = CreateActivationGradParam(schema::ActivationType_RELU);
  auto grad = new lite::Tensor(kNumberTypeFloat16, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  grad->MallocData(allocator_);
  inputs_.push_back(grad);
  auto input = new lite::Tensor(kNumberTypeFloat16, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);
  auto output = new lite::Tensor(kNumberTypeFloat16, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::vector<float> grad_data(num);
  FillLinearInput(grad_data.data(), num);
  std::vector<float> input_data(num);
  FillLinearInput(input_data.data(), num);
  auto grad_fp16 = reinterpret_cast<uint16_t *>(grad->MutableData());
  auto input_fp16 = reinterpret_cast<uint16_t *>(input->MutableData());
  for (int i = 0; i < num; ++i) {
    grad_fp16[i] = Fp32ToFp16(grad_data[i]);
    input_fp16[i] = Fp32ToFp16(input_data[i]);
  }
  std::memset(output->MutableData(), 0, static_cast<size_t>(num) * sizeof(uint16_t));

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_ActivationGrad};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(0, kernel->Prepare());
  EXPECT_EQ(0, kernel->Run());

  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = RefReluGrad(grad_data[i], input_data[i]);
  }
  std::vector<float> actual(num);
  auto output_fp16 = reinterpret_cast<uint16_t *>(outputs_[0]->MutableData());
  for (int i = 0; i < num; ++i) {
    actual[i] = Fp16ToFp32(output_fp16[i]);
  }
  ASSERT_EQ(0, CompareOutputData(actual.data(), correct.data(), num, 1e-1));
  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) {
    delete t;
  }
  for (auto t : outputs_) {
    delete t;
  }
  delete kernel;
}
}  // namespace mindspore::lite::dsp::test
#endif
