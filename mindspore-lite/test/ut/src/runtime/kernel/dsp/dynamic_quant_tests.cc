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
#include "nnacl_c/dynamic_quant_parameter.h"

#ifdef SUPPORT_FT04

namespace mindspore::lite::dsp::test {

namespace {
constexpr int kLength = 256;
constexpr bool kSymmetric = false;

using QuantParam = ::DynamicQuantParameter;

// Mirror of the kernel's per-tensor scale/zp computation plus int8 quantization.
std::vector<int8_t> RefDynamicQuant(const std::vector<float> &in, bool symmetric, float *out_scale, int *out_zp) {
  const float real_min = *std::min_element(in.begin(), in.end());
  const float real_max = *std::max_element(in.begin(), in.end());
  float scale = 0.0f;
  int zp = 0;
  int min_value = -128;
  int max_value = 127;
  if (symmetric) {
    float absmax = std::max(std::fabs(real_max), std::fabs(real_min));
    if (absmax <= 0.0f) {
      absmax = 0.01f;
    }
    scale = 2.0f * absmax / 254.0f;
    zp = 0;
    min_value = -127;
  } else {
    float range = real_max - real_min;
    if (range <= 0.0f) {
      range = 0.01f;
    }
    scale = range / 255.0f;
    zp = static_cast<int>(std::round(-128.0f - real_min / scale));
  }
  *out_scale = scale;
  *out_zp = zp;
  std::vector<int8_t> out(in.size(), 0);
  const float inv_scale = 1.0f / scale;
  for (size_t i = 0; i < in.size(); ++i) {
    // The SDK quantizer multiplies by the reciprocal, not by dividing by scale;
    // these differ by one LSB at boundaries.
    float q = in[i] * inv_scale + static_cast<float>(zp);
    q = std::round(q);
    q = std::max(static_cast<float>(min_value), std::min(static_cast<float>(max_value), q));
    out[i] = static_cast<int8_t>(static_cast<int>(q));
  }
  return out;
}

void FillInput(TypeId dtype, lite::Tensor *input, const std::vector<float> &in_f) {
  if (dtype == kNumberTypeFloat32) {
    std::memcpy(input->MutableData(), in_f.data(), in_f.size() * sizeof(float));
    return;
  }
  auto *h = reinterpret_cast<uint16_t *>(input->MutableData());
  for (size_t i = 0; i < in_f.size(); ++i) {
    h[i] = Fp32ToFp16(in_f[i]);
  }
}
}  // namespace

class TestDSP_DynamicQuant : public DSPCommonTest {
 public:
  void RunCase(TypeId dtype, bool symmetric, bool all_zero = false) {
    InitDSPRuntime();
    std::vector<float> in_f(kLength);
    for (int i = 0; i < kLength; ++i) {
      in_f[i] = all_zero ? 0.0f : std::sin(static_cast<float>(i) * 0.1f) * 3.0f;
    }

    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    auto *input = new lite::Tensor(dtype, {kLength}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, input->MallocData(allocator_));
    FillInput(dtype, input, in_f);
    auto *output = new lite::Tensor(kNumberTypeInt8, {kLength}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, output->MallocData(allocator_));
    inputs.push_back(input);
    outputs.push_back(output);

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    auto *param = static_cast<QuantParam *>(malloc(sizeof(QuantParam)));
    ASSERT_NE(param, nullptr);
    std::memset(param, 0, sizeof(QuantParam));
    param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_DynamicQuant);
    param->dst_type_ = kNumberTypeInt8;
    param->symmetric_ = symmetric;
    param->axis_num_ = 0;

    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, dtype, NHWC, schema::PrimitiveType_DynamicQuant};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel_handle = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    ASSERT_NE(kernel_handle, nullptr);
    ASSERT_EQ(kernel_handle->Prepare(), lite::RET_OK);
    ASSERT_EQ(kernel_handle->Run(), lite::RET_OK);

    // The DSP kernel derives min/max from the actual fp16 values, so the reference
    // must use the fp16 round-tripped data (identical to fp32 input otherwise).
    std::vector<float> ref_in = in_f;
    if (dtype == kNumberTypeFloat16) {
      const auto *h = reinterpret_cast<const uint16_t *>(input->data());
      for (size_t i = 0; i < ref_in.size(); ++i) {
        ref_in[i] = Fp16ToFp32(h[i]);
      }
    }
    float ref_scale = 0.0f;
    int ref_zp = 0;
    const auto ref_out = RefDynamicQuant(ref_in, symmetric, &ref_scale, &ref_zp);
    const auto &qps = output->quant_params();
    ASSERT_EQ(qps.size(), 1);
    EXPECT_NEAR(qps[0].scale, ref_scale, 1e-6f);
    EXPECT_EQ(qps[0].zeroPoint, ref_zp);
    // fp16 input can land exactly on a saturation boundary (e.g. 127.5), where
    // the DSP round instruction differs from host std::round by one LSB.
    const auto *out = reinterpret_cast<const int8_t *>(output->MutableData());
    for (size_t i = 0; i < ref_out.size(); ++i) {
      if (dtype == kNumberTypeFloat32) {
        EXPECT_EQ(ref_out[i], out[i]) << "output[" << i << "]";
      } else {
        EXPECT_LE(std::abs(static_cast<int>(ref_out[i]) - static_cast<int>(out[i])), 1) << "output[" << i << "]";
      }
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

  // Per-axis quantization is not supported by the SDK kernel; CheckSpecs must
  // reject it so the scheduler falls back to CPU (creator returns nullptr).
  void ExpectAxisRejected() {
    InitDSPRuntime();
    auto *input = new lite::Tensor(kNumberTypeFloat32, {kLength}, NHWC, lite::Category::CONST_TENSOR);
    auto *output = new lite::Tensor(kNumberTypeInt8, {kLength}, NHWC, lite::Category::CONST_TENSOR);
    std::vector<lite::Tensor *> inputs = {input};
    std::vector<lite::Tensor *> outputs = {output};
    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    auto *param = static_cast<QuantParam *>(malloc(sizeof(QuantParam)));
    ASSERT_NE(param, nullptr);
    std::memset(param, 0, sizeof(QuantParam));
    param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_DynamicQuant);
    param->dst_type_ = kNumberTypeInt8;
    param->axis_num_ = 1;
    param->prefer_axes_[0] = 0;

    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_DynamicQuant};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel_handle = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    EXPECT_EQ(kernel_handle, nullptr);

    delete kernel_handle;
    delete input;
    delete output;
    delete ctx;
    UninitDSPRuntime();
  }
};

TEST_F(TestDSP_DynamicQuant, Fp32) { RunCase(kNumberTypeFloat32, kSymmetric); }
TEST_F(TestDSP_DynamicQuant, Fp16) { RunCase(kNumberTypeFloat16, kSymmetric); }
TEST_F(TestDSP_DynamicQuant, SymmetricFp32) { RunCase(kNumberTypeFloat32, true); }
TEST_F(TestDSP_DynamicQuant, AllZeroSymmetric) { RunCase(kNumberTypeFloat32, true, true); }
TEST_F(TestDSP_DynamicQuant, AxisRejected_FallsBackToCpu) { ExpectAxisRejected(); }

}  // namespace mindspore::lite::dsp::test

#endif  // SUPPORT_FT04
