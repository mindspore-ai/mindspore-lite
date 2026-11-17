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
#include <cstdint>
#include <cstring>
#include <vector>
#include "ut/src/runtime/kernel/dsp/dsp_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel/dsp/dsp_subgraph.h"
#include "src/litert/kernel_registry.h"
#include "nnacl_c/nllloss_parameter.h"

#ifdef SUPPORT_FT04

namespace mindspore::lite::dsp::test {

namespace {
constexpr int kBatch = 6;
constexpr int kClass = 5;
constexpr int kLabelOutOfRange = kClass;
constexpr float kErrBound = 5e-3f;

using Param = ::NLLLossParameter;

struct CaseConfig {
  TypeId dtype;
  ::ReductionType reduction;
  std::vector<int> loss_grad_shape;
};

void FillTyped(TypeId dtype, lite::Tensor *tensor, const std::vector<float> &data) {
  if (dtype == kNumberTypeFloat32) {
    std::memcpy(tensor->MutableData(), data.data(), data.size() * sizeof(float));
    return;
  }
  auto *h = reinterpret_cast<uint16_t *>(tensor->MutableData());
  for (size_t i = 0; i < data.size(); ++i) {
    h[i] = Fp32ToFp16(data[i]);
  }
}

float ReadTyped(TypeId dtype, const void *data, size_t index) {
  if (dtype == kNumberTypeFloat32) {
    return reinterpret_cast<const float *>(data)[index];
  }
  return Fp16ToFp32(reinterpret_cast<const uint16_t *>(data)[index]);
}
}  // namespace

class TestDSP_NLLLossGrad : public DSPCommonTest {
 public:
  void RunCase(const CaseConfig &cfg) {
    InitDSPRuntime();
    const size_t n_loss_grad = cfg.reduction == Reduction_None ? kBatch : 1;
    std::vector<float> logits_f(kBatch * kClass, -0.3f);
    std::vector<float> loss_grad_f(n_loss_grad);
    for (size_t i = 0; i < n_loss_grad; ++i) {
      loss_grad_f[i] = 0.5f + i * 0.1f;
    }
    std::vector<int32_t> labels(kBatch);
    std::vector<float> weight_f(kClass);
    for (int i = 0; i < kBatch; ++i) {
      labels[i] = i % kClass;
    }
    for (int c = 0; c < kClass; ++c) {
      weight_f[c] = 0.5f + c * 0.1f;
    }
    const float total_weight = 2.0f;

    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    auto *logits = new lite::Tensor(cfg.dtype, {kBatch, kClass}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, logits->MallocData(allocator_));
    FillTyped(cfg.dtype, logits, logits_f);
    auto *loss_grad = new lite::Tensor(cfg.dtype, cfg.loss_grad_shape, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, loss_grad->MallocData(allocator_));
    FillTyped(cfg.dtype, loss_grad, loss_grad_f);
    auto *label_tensor = new lite::Tensor(kNumberTypeInt32, {kBatch}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, label_tensor->MallocData(allocator_));
    std::memcpy(label_tensor->MutableData(), labels.data(), kBatch * sizeof(int32_t));
    auto *weight = new lite::Tensor(cfg.dtype, {kClass}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, weight->MallocData(allocator_));
    FillTyped(cfg.dtype, weight, weight_f);
    auto *tw = new lite::Tensor(cfg.dtype, {}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, tw->MallocData(allocator_));
    FillTyped(cfg.dtype, tw, {total_weight});
    inputs = {logits, loss_grad, label_tensor, weight, tw};

    auto *dx = new lite::Tensor(cfg.dtype, {kBatch, kClass}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, dx->MallocData(allocator_));
    // Prefill with 0xFF (NaN) so "unselected grad positions are zeroed" is a real
    // assertion, not a consequence of freshly allocated memory happening to be 0.
    std::memset(dx->MutableData(), 0xFF, kBatch * kClass * lite::DataTypeSize(cfg.dtype));
    outputs = {dx};

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    auto *param = static_cast<Param *>(malloc(sizeof(Param)));
    ASSERT_NE(param, nullptr);
    std::memset(param, 0, sizeof(Param));
    param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_NLLLossGrad);
    param->reduction_type_ = cfg.reduction;

    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, cfg.dtype, NHWC, schema::PrimitiveType_NLLLossGrad};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel_handle = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    ASSERT_NE(kernel_handle, nullptr);
    ASSERT_EQ(kernel_handle->Prepare(), lite::RET_OK);
    ASSERT_EQ(kernel_handle->Run(), lite::RET_OK);

    // Reference from CPU NLLLossGrad.
    std::vector<float> ref_dx(kBatch * kClass, 0.0f);
    for (int i = 0; i < kBatch; ++i) {
      const int c = labels[i];
      const int index = i * kClass + c;
      if (cfg.reduction == Reduction_Sum) {
        ref_dx[index] = -loss_grad_f[0] * weight_f[c];
      } else if (cfg.reduction == Reduction_Mean) {
        ref_dx[index] = -loss_grad_f[0] * weight_f[c] / total_weight;
      } else {
        ref_dx[index] = -loss_grad_f[i] * weight_f[c];
      }
    }
    const float err = cfg.dtype == kNumberTypeFloat32 ? 1e-4f : kErrBound;
    for (size_t i = 0; i < ref_dx.size(); ++i) {
      EXPECT_NEAR(ref_dx[i], ReadTyped(cfg.dtype, dx->data(), i), err) << "dx[" << i << "]";
    }

    delete kernel_handle;
    for (auto *t : inputs) {
      delete t;
    }
    for (auto *t : outputs) {
      delete t;
    }
    delete ctx;
    UninitDSPRuntime();
  }

  // Invalid label must be rejected by ValidateLabels (Run returns error).
  void ExpectLabelRejected(int32_t invalid_label) {
    InitDSPRuntime();
    auto *logits = new lite::Tensor(kNumberTypeFloat32, {kBatch, kClass}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, logits->MallocData(allocator_));
    std::vector<float> dummy_l(kBatch * kClass, -0.3f);
    std::memcpy(logits->MutableData(), dummy_l.data(), dummy_l.size() * sizeof(float));
    auto *loss_grad = new lite::Tensor(kNumberTypeFloat32, {}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, loss_grad->MallocData(allocator_));
    *reinterpret_cast<float *>(loss_grad->MutableData()) = 1.0f;
    auto *labels = new lite::Tensor(kNumberTypeInt32, {kBatch}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, labels->MallocData(allocator_));
    auto *label_data = reinterpret_cast<int32_t *>(labels->MutableData());
    for (int i = 0; i < kBatch; ++i) {
      label_data[i] = i == 0 ? invalid_label : i % kClass;
    }
    auto *weight = new lite::Tensor(kNumberTypeFloat32, {kClass}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, weight->MallocData(allocator_));
    std::vector<float> w(kClass, 1.0f);
    std::memcpy(weight->MutableData(), w.data(), w.size() * sizeof(float));
    auto *tw = new lite::Tensor(kNumberTypeFloat32, {}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, tw->MallocData(allocator_));
    *reinterpret_cast<float *>(tw->MutableData()) = 2.0f;
    auto *dx = new lite::Tensor(kNumberTypeFloat32, {kBatch, kClass}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, dx->MallocData(allocator_));
    std::vector<lite::Tensor *> inputs = {logits, loss_grad, labels, weight, tw};
    std::vector<lite::Tensor *> outputs = {dx};

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    auto *param = static_cast<Param *>(malloc(sizeof(Param)));
    ASSERT_NE(param, nullptr);
    std::memset(param, 0, sizeof(Param));
    param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_NLLLossGrad);
    param->reduction_type_ = Reduction_Sum;

    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_NLLLossGrad};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel_handle = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    ASSERT_NE(kernel_handle, nullptr);
    ASSERT_EQ(kernel_handle->Prepare(), lite::RET_OK);
    // Only the label is invalid; all device pointers and shapes are valid, so the
    // RET_ERROR must come from ValidateLabels' label range check.
    EXPECT_NE(kernel_handle->Run(), lite::RET_OK);

    delete kernel_handle;
    for (auto *t : inputs) {
      delete t;
    }
    for (auto *t : outputs) {
      delete t;
    }
    delete ctx;
    UninitDSPRuntime();
  }

  // CheckSpecs must reject loss_grad shape inconsistent with reduction:
  // for None, loss_grad must have exactly batch elements ([N]), not a scalar.
  void ExpectLossGradShapeRejected() {
    InitDSPRuntime();
    auto *logits = new lite::Tensor(kNumberTypeFloat32, {kBatch, kClass}, NHWC, lite::Category::CONST_TENSOR);
    auto *loss_grad = new lite::Tensor(kNumberTypeFloat32, {}, NHWC, lite::Category::CONST_TENSOR);
    auto *labels = new lite::Tensor(kNumberTypeInt32, {kBatch}, NHWC, lite::Category::CONST_TENSOR);
    auto *weight = new lite::Tensor(kNumberTypeFloat32, {kClass}, NHWC, lite::Category::CONST_TENSOR);
    auto *tw = new lite::Tensor(kNumberTypeFloat32, {}, NHWC, lite::Category::CONST_TENSOR);
    auto *dx = new lite::Tensor(kNumberTypeFloat32, {kBatch, kClass}, NHWC, lite::Category::CONST_TENSOR);
    std::vector<lite::Tensor *> inputs = {logits, loss_grad, labels, weight, tw};
    std::vector<lite::Tensor *> outputs = {dx};

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    auto *param = static_cast<Param *>(malloc(sizeof(Param)));
    ASSERT_NE(param, nullptr);
    std::memset(param, 0, sizeof(Param));
    param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_NLLLossGrad);
    param->reduction_type_ = Reduction_None;

    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_NLLLossGrad};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel_handle = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    EXPECT_EQ(kernel_handle, nullptr);

    delete kernel_handle;
    delete logits;
    delete loss_grad;
    delete labels;
    delete weight;
    delete tw;
    delete dx;
    delete ctx;
    UninitDSPRuntime();
  }
};

TEST_F(TestDSP_NLLLossGrad, Fp32_Sum) { RunCase({kNumberTypeFloat32, Reduction_Sum, {}}); }
TEST_F(TestDSP_NLLLossGrad, Fp32_Mean) { RunCase({kNumberTypeFloat32, Reduction_Mean, {}}); }
TEST_F(TestDSP_NLLLossGrad, Fp32_None) { RunCase({kNumberTypeFloat32, Reduction_None, {kBatch}}); }
TEST_F(TestDSP_NLLLossGrad, Fp16_Sum) { RunCase({kNumberTypeFloat16, Reduction_Sum, {}}); }
TEST_F(TestDSP_NLLLossGrad, Fp16_Mean) { RunCase({kNumberTypeFloat16, Reduction_Mean, {}}); }
TEST_F(TestDSP_NLLLossGrad, Fp16_None) { RunCase({kNumberTypeFloat16, Reduction_None, {kBatch}}); }
TEST_F(TestDSP_NLLLossGrad, LabelOutOfRange_Rejected) { ExpectLabelRejected(kLabelOutOfRange); }
TEST_F(TestDSP_NLLLossGrad, NegativeLabel_Rejected) { ExpectLabelRejected(-1); }
TEST_F(TestDSP_NLLLossGrad, LossGradShapeMismatch_FallsBackToCpu) { ExpectLossGradShapeRejected(); }

}  // namespace mindspore::lite::dsp::test

#endif  // SUPPORT_FT04
