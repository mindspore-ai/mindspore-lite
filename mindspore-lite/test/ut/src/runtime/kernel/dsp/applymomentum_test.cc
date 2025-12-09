/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "ut/src/runtime/kernel/dsp/dsp_test.h"
#include "ut/src/runtime/kernel/opencl/common.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/dsp/dsp_subgraph.h"
#ifdef SUPPORT_FT78
#include "src/litert/kernel/dsp/ft78/applymomentum.h"
#endif
#ifdef SUPPORT_FT04
#include "src/litert/kernel/dsp/ft04/applymomentum.h"
#endif
#include "src/litert/kernel/cpu/nnacl_c/fp32_grad/optimizer.h"

namespace mindspore::lite::dsp::test {
namespace {
constexpr int kTensorLength = 10000;
constexpr float kLearningRate = 0.0001f;
constexpr float kMomentum = 1.0f;

ApplyMomentumParameter *CreateApplyMomentumParameter(bool use_nesterov) {
  auto *param = opencl::test::CreateParameter<ApplyMomentumParameter>(schema::PrimitiveType_ApplyMomentum);
  if (param == nullptr) {
    return nullptr;
  }
  param->use_nesterov_ = use_nesterov;
  param->grad_scale_ = 1.0f;
  return param;
}

void ApplyMomentumReference(std::vector<float> *weight, std::vector<float> *accumulate,
                            const std::vector<float> &gradient, float lr, float momentum, bool use_nesterov) {
  for (size_t i = 0; i < weight->size(); ++i) {
    float grad = gradient[i];
    float accu = (*accumulate)[i];
    accu = accu * momentum + grad;
    (*accumulate)[i] = accu;
    float update = use_nesterov ? (accu * momentum + grad) : accu;
    (*weight)[i] -= update * lr;
  }
}

std::vector<float> BuildWeightData() {
  std::vector<float> data(kTensorLength);
  for (int i = 0; i < kTensorLength; ++i) {
    data[i] = 0.25f + 0.00035f * static_cast<float>(i);
  }
  return data;
}

std::vector<float> BuildAccumulateData() {
  std::vector<float> data(kTensorLength);
  for (int i = 0; i < kTensorLength; ++i) {
    data[i] = 0.01f + 0.0002f * static_cast<float>(i % 97);
  }
  return data;
}

std::vector<float> BuildGradientData() {
  std::vector<float> data(kTensorLength);
  for (int i = 0; i < kTensorLength; ++i) {
    float angle = static_cast<float>(i) * 0.01f;
    data[i] = 0.02f * std::sin(angle);
  }
  return data;
}

}  // namespace

class TestDSP_ApplyMomentum : public DSPCommonTest {};

TEST_F(TestDSP_ApplyMomentum, ApplyMomentum_Fp32_NesterovFalse) {
  InitDSPRuntime();

  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  std::vector<lite::Tensor *> tensors_to_delete;

  std::vector<int> param_shape = {kTensorLength};
  std::vector<int> scalar_shape = {1};

  auto weight_tensor = new lite::Tensor(kNumberTypeFloat32, param_shape, mindspore::NHWC, lite::Category::VAR);
  weight_tensor->MallocData(allocator_);
  inputs.push_back(weight_tensor);
  outputs.push_back(weight_tensor);
  tensors_to_delete.push_back(weight_tensor);

  auto accumulate_tensor = new lite::Tensor(kNumberTypeFloat32, param_shape, mindspore::NHWC, lite::Category::VAR);
  accumulate_tensor->MallocData(allocator_);
  inputs.push_back(accumulate_tensor);
  tensors_to_delete.push_back(accumulate_tensor);

  auto lr_tensor = new lite::Tensor(kNumberTypeFloat32, scalar_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  lr_tensor->MallocData(allocator_);
  inputs.push_back(lr_tensor);
  tensors_to_delete.push_back(lr_tensor);

  auto gradient_tensor =
    new lite::Tensor(kNumberTypeFloat32, param_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  gradient_tensor->MallocData(allocator_);
  inputs.push_back(gradient_tensor);
  tensors_to_delete.push_back(gradient_tensor);

  auto momentum_tensor =
    new lite::Tensor(kNumberTypeFloat32, scalar_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  momentum_tensor->MallocData(allocator_);
  inputs.push_back(momentum_tensor);
  tensors_to_delete.push_back(momentum_tensor);

  auto initial_weight = BuildWeightData();
  auto initial_accumulate = BuildAccumulateData();
  auto gradients = BuildGradientData();

  std::copy(initial_weight.begin(), initial_weight.end(), reinterpret_cast<float *>(weight_tensor->MutableData()));
  std::copy(initial_accumulate.begin(), initial_accumulate.end(),
            reinterpret_cast<float *>(accumulate_tensor->MutableData()));
  std::copy(gradients.begin(), gradients.end(), reinterpret_cast<float *>(gradient_tensor->MutableData()));

  reinterpret_cast<float *>(lr_tensor->MutableData())[0] = kLearningRate;
  reinterpret_cast<float *>(momentum_tensor->MutableData())[0] = kMomentum;

  auto expected_weight = initial_weight;
  auto expected_accumulate = initial_accumulate;
  ApplyMomentumReference(&expected_weight, &expected_accumulate, gradients, kLearningRate, kMomentum, false);

  auto ctx = new lite::InnerContext;
  ASSERT_NE(ctx, nullptr);
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  auto *param = CreateApplyMomentumParameter(false);
  ASSERT_NE(param, nullptr);

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_ApplyMomentum};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);

  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);

  ASSERT_EQ(lite::RET_OK, kernel->Prepare());
  ASSERT_EQ(lite::RET_OK, kernel->Run());

  auto weight_after = reinterpret_cast<float *>(inputs[kernel::kApplyMomentumWeightIdx]->MutableData());
  float sum_abs_err = 0.f;
  for (int i = 0; i < kTensorLength; ++i) {
    float abs_err = std::fabs(weight_after[i] - expected_weight[i]);
    sum_abs_err += abs_err;
  }

  ASSERT_EQ(0, CompareOutputData(weight_after, expected_weight.data(), kTensorLength, 1e-5f));

  UninitDSPRuntime();

  delete ctx;
  for (auto *tensor : tensors_to_delete) {
    delete tensor;
  }
  delete kernel;
}

#ifdef SUPPORT_FT04
TEST_F(TestDSP_ApplyMomentum, ApplyMomentum_Fp16_NesterovTrue) {
  InitDSPRuntime();

  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  std::vector<lite::Tensor *> tensors_to_delete;

  std::vector<int> param_shape = {kTensorLength};
  std::vector<int> scalar_shape = {1};

  auto weight_tensor = new lite::Tensor(kNumberTypeFloat16, param_shape, mindspore::NHWC, lite::Category::VAR);
  weight_tensor->MallocData(allocator_);
  inputs.push_back(weight_tensor);
  outputs.push_back(weight_tensor);
  tensors_to_delete.push_back(weight_tensor);

  auto accumulate_tensor = new lite::Tensor(kNumberTypeFloat16, param_shape, mindspore::NHWC, lite::Category::VAR);
  accumulate_tensor->MallocData(allocator_);
  inputs.push_back(accumulate_tensor);
  tensors_to_delete.push_back(accumulate_tensor);

  auto lr_tensor = new lite::Tensor(kNumberTypeFloat16, scalar_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  lr_tensor->MallocData(allocator_);
  inputs.push_back(lr_tensor);
  tensors_to_delete.push_back(lr_tensor);

  auto gradient_tensor =
    new lite::Tensor(kNumberTypeFloat16, param_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  gradient_tensor->MallocData(allocator_);
  inputs.push_back(gradient_tensor);
  tensors_to_delete.push_back(gradient_tensor);

  auto momentum_tensor =
    new lite::Tensor(kNumberTypeFloat16, scalar_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  momentum_tensor->MallocData(allocator_);
  inputs.push_back(momentum_tensor);
  tensors_to_delete.push_back(momentum_tensor);

  auto initial_weight_fp32 = BuildWeightData();
  auto initial_accumulate_fp32 = BuildAccumulateData();
  auto gradients_fp32 = BuildGradientData();

  auto *weight_half = reinterpret_cast<uint16_t *>(weight_tensor->MutableData());
  auto *accumulate_half = reinterpret_cast<uint16_t *>(accumulate_tensor->MutableData());
  auto *gradient_half = reinterpret_cast<uint16_t *>(gradient_tensor->MutableData());

  for (int i = 0; i < kTensorLength; ++i) {
    weight_half[i] = fp32_to_fp16(initial_weight_fp32[i]);
    accumulate_half[i] = fp32_to_fp16(initial_accumulate_fp32[i]);
    gradient_half[i] = fp32_to_fp16(gradients_fp32[i]);
  }

  reinterpret_cast<uint16_t *>(lr_tensor->MutableData())[0] = fp32_to_fp16(kLearningRate);
  reinterpret_cast<uint16_t *>(momentum_tensor->MutableData())[0] = fp32_to_fp16(kMomentum);

  auto expected_weight = initial_weight_fp32;
  auto expected_accumulate = initial_accumulate_fp32;
  ApplyMomentumReference(&expected_weight, &expected_accumulate, gradients_fp32, kLearningRate, kMomentum, true);

  std::vector<float> expected_weight_quantized(kTensorLength);
  for (int i = 0; i < kTensorLength; ++i) {
    expected_weight_quantized[i] = fp16_to_fp32(fp32_to_fp16(expected_weight[i]));
  }

  auto ctx = new lite::InnerContext;
  ASSERT_NE(ctx, nullptr);
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  auto *param = CreateApplyMomentumParameter(true);
  ASSERT_NE(param, nullptr);

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_ApplyMomentum};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);

  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);

  ASSERT_EQ(lite::RET_OK, kernel->Prepare());
  ASSERT_EQ(lite::RET_OK, kernel->Run());

  auto weight_after_half = reinterpret_cast<uint16_t *>(inputs[kernel::kApplyMomentumWeightIdx]->MutableData());
  std::vector<float> weight_after_fp32(kTensorLength);
  for (int i = 0; i < kTensorLength; ++i) {
    weight_after_fp32[i] = fp16_to_fp32(weight_after_half[i]);
  }

  float sum_abs_err = 0.f;
  for (int i = 0; i < kTensorLength; ++i) {
    float abs_err = std::fabs(weight_after_fp32[i] - expected_weight_quantized[i]);
    sum_abs_err += abs_err;
  }

  ASSERT_EQ(0, CompareOutputData(weight_after_fp32.data(), expected_weight_quantized.data(), kTensorLength, 1e-3f));

  UninitDSPRuntime();

  delete ctx;
  for (auto *tensor : tensors_to_delete) {
    delete tensor;
  }
  delete kernel;
}
#endif  // SUPPORT_FT04

}  // namespace mindspore::lite::dsp::test
