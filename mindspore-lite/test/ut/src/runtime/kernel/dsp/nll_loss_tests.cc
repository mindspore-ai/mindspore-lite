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
  std::vector<int> loss_shape;
};

void FillLogits(TypeId dtype, lite::Tensor *tensor, const std::vector<float> &data) {
  if (dtype == kNumberTypeFloat32) {
    std::memcpy(tensor->MutableData(), data.data(), data.size() * sizeof(float));
    return;
  }
  auto *h = reinterpret_cast<uint16_t *>(tensor->MutableData());
  for (size_t i = 0; i < data.size(); ++i) {
    h[i] = Fp32ToFp16(data[i]);
  }
}

void Verify(TypeId dtype, lite::Tensor *loss, lite::Tensor *total_weight, const std::vector<float> &ref_loss,
            float ref_tw) {
  const float err = dtype == kNumberTypeFloat32 ? 1e-4f : kErrBound;
  if (dtype == kNumberTypeFloat32) {
    const auto *out = reinterpret_cast<const float *>(loss->MutableData());
    for (size_t i = 0; i < ref_loss.size(); ++i) {
      EXPECT_NEAR(ref_loss[i], out[i], err) << "loss[" << i << "]";
    }
    EXPECT_NEAR(ref_tw, *reinterpret_cast<const float *>(total_weight->MutableData()), err) << "total_weight";
  } else {
    const auto *out = reinterpret_cast<const uint16_t *>(loss->MutableData());
    for (size_t i = 0; i < ref_loss.size(); ++i) {
      EXPECT_NEAR(ref_loss[i], Fp16ToFp32(out[i]), err) << "loss[" << i << "]";
    }
    EXPECT_NEAR(ref_tw, Fp16ToFp32(*reinterpret_cast<const uint16_t *>(total_weight->MutableData())), err)
      << "total_weight";
  }
}
}  // namespace

class TestDSP_NLLLoss : public DSPCommonTest {
 public:
  void RunCase(const CaseConfig &cfg) {
    InitDSPRuntime();
    std::vector<float> logits_f(kBatch * kClass);
    for (int i = 0; i < kBatch * kClass; ++i) {
      logits_f[i] = -0.1f - (i % kClass) * 0.05f;
    }
    std::vector<int32_t> labels(kBatch);
    std::vector<float> weight_f(kClass);
    for (int i = 0; i < kBatch; ++i) {
      labels[i] = i % kClass;
    }
    for (int c = 0; c < kClass; ++c) {
      weight_f[c] = 0.5f + c * 0.1f;
    }

    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    auto *logits = new lite::Tensor(cfg.dtype, {kBatch, kClass}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, logits->MallocData(allocator_));
    FillLogits(cfg.dtype, logits, logits_f);
    auto *label_tensor = new lite::Tensor(kNumberTypeInt32, {kBatch}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, label_tensor->MallocData(allocator_));
    std::memcpy(label_tensor->MutableData(), labels.data(), kBatch * sizeof(int32_t));
    auto *weight = new lite::Tensor(cfg.dtype, {kClass}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, weight->MallocData(allocator_));
    FillLogits(cfg.dtype, weight, weight_f);
    inputs = {logits, label_tensor, weight};

    auto *loss = new lite::Tensor(cfg.dtype, cfg.loss_shape, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, loss->MallocData(allocator_));
    auto *total_weight = new lite::Tensor(cfg.dtype, {}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, total_weight->MallocData(allocator_));
    outputs = {loss, total_weight};

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    auto *param = static_cast<Param *>(malloc(sizeof(Param)));
    ASSERT_NE(param, nullptr);
    std::memset(param, 0, sizeof(Param));
    param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_NLLLoss);
    param->reduction_type_ = cfg.reduction;

    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, cfg.dtype, NHWC, schema::PrimitiveType_NLLLoss};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel_handle = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    ASSERT_NE(kernel_handle, nullptr);
    ASSERT_EQ(kernel_handle->Prepare(), lite::RET_OK);
    ASSERT_EQ(kernel_handle->Run(), lite::RET_OK);

    std::vector<float> ref_loss;
    float ref_tw = 0.0f;
    float total = 0.0f;
    for (int i = 0; i < kBatch; ++i) {
      const int c = labels[i];
      const float n_loss = -logits_f[i * kClass + c] * weight_f[c];
      ref_tw += weight_f[c];
      total += n_loss;
      if (cfg.reduction == Reduction_None) {
        ref_loss.push_back(n_loss);
      }
    }
    if (cfg.reduction == Reduction_Sum) {
      ref_loss.push_back(total);
    } else if (cfg.reduction == Reduction_Mean) {
      ref_loss.push_back(total / ref_tw);
    }
    Verify(cfg.dtype, loss, total_weight, ref_loss, ref_tw);

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

  // Invalid label value must be rejected by ValidateLabels (Run returns error).
  void ExpectLabelRejected() {
    InitDSPRuntime();
    auto *logits = new lite::Tensor(kNumberTypeFloat32, {kBatch, kClass}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, logits->MallocData(allocator_));
    std::vector<float> dummy(kBatch * kClass, -0.5f);
    std::memcpy(logits->MutableData(), dummy.data(), dummy.size() * sizeof(float));
    auto *labels = new lite::Tensor(kNumberTypeInt32, {kBatch}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, labels->MallocData(allocator_));
    auto *label_data = reinterpret_cast<int32_t *>(labels->MutableData());
    for (int i = 0; i < kBatch; ++i) {
      label_data[i] = i == 0 ? kLabelOutOfRange : i % kClass;
    }
    auto *weight = new lite::Tensor(kNumberTypeFloat32, {kClass}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, weight->MallocData(allocator_));
    std::vector<float> w(kClass, 1.0f);
    std::memcpy(weight->MutableData(), w.data(), w.size() * sizeof(float));
    auto *loss = new lite::Tensor(kNumberTypeFloat32, {}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, loss->MallocData(allocator_));
    auto *tw = new lite::Tensor(kNumberTypeFloat32, {}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, tw->MallocData(allocator_));
    std::vector<lite::Tensor *> inputs = {logits, labels, weight};
    std::vector<lite::Tensor *> outputs = {loss, tw};

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    auto *param = static_cast<Param *>(malloc(sizeof(Param)));
    ASSERT_NE(param, nullptr);
    std::memset(param, 0, sizeof(Param));
    param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_NLLLoss);
    param->reduction_type_ = Reduction_Sum;

    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_NLLLoss};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel_handle = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    ASSERT_NE(kernel_handle, nullptr);
    ASSERT_EQ(kernel_handle->Prepare(), lite::RET_OK);
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
  // CheckSpecs must reject a loss shape inconsistent with reduction (None needs
  // N elements); creator returning nullptr means fallback to CPU.
  void ExpectLossShapeRejected() {
    InitDSPRuntime();
    auto *logits = new lite::Tensor(kNumberTypeFloat32, {kBatch, kClass}, NHWC, lite::Category::CONST_TENSOR);
    auto *labels = new lite::Tensor(kNumberTypeInt32, {kBatch}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, labels->MallocData(allocator_));
    auto *label_data = reinterpret_cast<int32_t *>(labels->MutableData());
    for (int i = 0; i < kBatch; ++i) {
      label_data[i] = i % kClass;
    }
    auto *weight = new lite::Tensor(kNumberTypeFloat32, {kClass}, NHWC, lite::Category::CONST_TENSOR);
    // reduction=None but loss is a scalar (1 element != kBatch): CheckSpecs rejects.
    auto *loss = new lite::Tensor(kNumberTypeFloat32, {}, NHWC, lite::Category::CONST_TENSOR);
    auto *tw = new lite::Tensor(kNumberTypeFloat32, {}, NHWC, lite::Category::CONST_TENSOR);
    std::vector<lite::Tensor *> inputs = {logits, labels, weight};
    std::vector<lite::Tensor *> outputs = {loss, tw};

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    auto *param = static_cast<Param *>(malloc(sizeof(Param)));
    ASSERT_NE(param, nullptr);
    std::memset(param, 0, sizeof(Param));
    param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_NLLLoss);
    param->reduction_type_ = Reduction_None;

    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_NLLLoss};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel_handle = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    EXPECT_EQ(kernel_handle, nullptr);

    delete kernel_handle;
    delete logits;
    delete labels;
    delete weight;
    delete loss;
    delete tw;
    delete ctx;
    UninitDSPRuntime();
  }
};

TEST_F(TestDSP_NLLLoss, Fp32_None) { RunCase({kNumberTypeFloat32, Reduction_None, {kBatch}}); }
TEST_F(TestDSP_NLLLoss, Fp32_Sum) { RunCase({kNumberTypeFloat32, Reduction_Sum, {}}); }
TEST_F(TestDSP_NLLLoss, Fp32_Mean) { RunCase({kNumberTypeFloat32, Reduction_Mean, {}}); }
TEST_F(TestDSP_NLLLoss, Fp16_None) { RunCase({kNumberTypeFloat16, Reduction_None, {kBatch}}); }
TEST_F(TestDSP_NLLLoss, Fp16_Sum) { RunCase({kNumberTypeFloat16, Reduction_Sum, {}}); }
TEST_F(TestDSP_NLLLoss, Fp16_Mean) { RunCase({kNumberTypeFloat16, Reduction_Mean, {}}); }
TEST_F(TestDSP_NLLLoss, LabelOutOfRange_Rejected) { ExpectLabelRejected(); }
TEST_F(TestDSP_NLLLoss, LossShapeMismatch_FallsBackToCpu) { ExpectLossShapeRejected(); }

}  // namespace mindspore::lite::dsp::test

#endif  // SUPPORT_FT04
