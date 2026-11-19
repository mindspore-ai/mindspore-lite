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
#include "nnacl_c/group_norm_parameter.h"

#ifdef SUPPORT_FT04

namespace mindspore::lite::dsp::test {

namespace {
constexpr int kBatch = 2;
constexpr int kChannel = 4;
constexpr int kHeight = 3;
constexpr int kWidth = 3;
constexpr int kNumGroups = 2;
constexpr float kEpsilon = 1e-5f;
constexpr float kErrBound = 5e-3f;

using GroupNormParam = ::GroupNormParameter;

struct RefConfig {
  int num_groups;
  int channel;
  int unit;
  int batch;
  float epsilon;
};

// Reference: per (batch, group) mean/variance, then normalize with scale/offset.
void RefGroupNorm(const std::vector<float> &input, const std::vector<float> &scale, const std::vector<float> &offset,
                  const RefConfig &cfg, std::vector<float> *output) {
  const int num_groups = cfg.num_groups;
  const int channel = cfg.channel;
  const int unit = cfg.unit;
  const int batch = cfg.batch;
  const float epsilon = cfg.epsilon;
  const int group_channel = channel / num_groups;
  const int group_count = group_channel * unit;
  std::vector<float> mean(static_cast<size_t>(batch) * num_groups, 0.0f);
  std::vector<float> var(static_cast<size_t>(batch) * num_groups, 0.0f);
  for (int b = 0; b < batch; ++b) {
    for (int g = 0; g < num_groups; ++g) {
      float sum = 0.0f;
      for (int c = g * group_channel; c < (g + 1) * group_channel; ++c) {
        for (int u = 0; u < unit; ++u) {
          sum += input[(static_cast<size_t>(b) * unit + u) * channel + c];
        }
      }
      const float m = sum / static_cast<float>(group_count);
      float var_sum = 0.0f;
      for (int c = g * group_channel; c < (g + 1) * group_channel; ++c) {
        for (int u = 0; u < unit; ++u) {
          const float d = input[(static_cast<size_t>(b) * unit + u) * channel + c] - m;
          var_sum += d * d;
        }
      }
      mean[static_cast<size_t>(b) * num_groups + g] = m;
      var[static_cast<size_t>(b) * num_groups + g] = var_sum / static_cast<float>(group_count);
    }
  }
  output->assign(input.size(), 0.0f);
  for (int b = 0; b < batch; ++b) {
    for (int u = 0; u < unit; ++u) {
      for (int c = 0; c < channel; ++c) {
        const int g = c / group_channel;
        const float m = mean[static_cast<size_t>(b) * num_groups + g];
        const float v = var[static_cast<size_t>(b) * num_groups + g];
        const float x = input[(static_cast<size_t>(b) * unit + u) * channel + c];
        (*output)[(static_cast<size_t>(b) * unit + u) * channel + c] =
          (x - m) / std::sqrt(v + epsilon) * scale[c] + offset[c];
      }
    }
  }
}
}  // namespace

class TestDSP_GroupNormFusion : public DSPCommonTest {
 public:
  void RunCase(TypeId dtype) {
    InitDSPRuntime();
    const int unit = kHeight * kWidth;

    const std::vector<int> shape = {kBatch, kHeight, kWidth, kChannel};
    const size_t total = static_cast<size_t>(kBatch) * kChannel * kHeight * kWidth;

    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;

    auto *input = new lite::Tensor(dtype, shape, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, input->MallocData(allocator_));
    ASSERT_NE(input->MutableData(), nullptr);
    auto *scale = new lite::Tensor(dtype, {kChannel}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, scale->MallocData(allocator_));
    auto *offset = new lite::Tensor(dtype, {kChannel}, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, offset->MallocData(allocator_));
    auto *output = new lite::Tensor(dtype, shape, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, output->MallocData(allocator_));
    inputs.push_back(input);
    inputs.push_back(scale);
    inputs.push_back(offset);
    outputs.push_back(output);

    std::vector<float> in_f(total), scale_f(kChannel), offset_f(kChannel);
    for (size_t i = 0; i < total; ++i) {
      in_f[i] = static_cast<float>(i % 17) - 8.0f;
    }
    for (int c = 0; c < kChannel; ++c) {
      scale_f[c] = 0.5f + c * 0.25f;
      offset_f[c] = static_cast<float>(c) - 1.5f;
    }
    if (dtype == kNumberTypeFloat32) {
      std::memcpy(input->MutableData(), in_f.data(), total * sizeof(float));
      std::memcpy(scale->MutableData(), scale_f.data(), kChannel * sizeof(float));
      std::memcpy(offset->MutableData(), offset_f.data(), kChannel * sizeof(float));
    } else {
      auto *in_h = reinterpret_cast<uint16_t *>(input->MutableData());
      for (size_t i = 0; i < total; ++i) in_h[i] = Fp32ToFp16(in_f[i]);
      auto *s_h = reinterpret_cast<uint16_t *>(scale->MutableData());
      for (int c = 0; c < kChannel; ++c) s_h[c] = Fp32ToFp16(scale_f[c]);
      auto *o_h = reinterpret_cast<uint16_t *>(offset->MutableData());
      for (int c = 0; c < kChannel; ++c) o_h[c] = Fp32ToFp16(offset_f[c]);
    }

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    auto *param = static_cast<GroupNormParam *>(malloc(sizeof(GroupNormParam)));
    ASSERT_NE(param, nullptr);
    std::memset(param, 0, sizeof(GroupNormParam));
    param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_GroupNormFusion);
    param->epsilon_ = kEpsilon;
    param->num_groups_ = kNumGroups;
    param->affine_ = true;

    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, dtype, NHWC, schema::PrimitiveType_GroupNormFusion};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel_handle = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    ASSERT_NE(kernel_handle, nullptr);
    ASSERT_EQ(kernel_handle->Prepare(), lite::RET_OK);
    ASSERT_EQ(kernel_handle->Run(), lite::RET_OK);

    std::vector<float> ref_out;
    RefGroupNorm(in_f, scale_f, offset_f, {kNumGroups, kChannel, unit, kBatch, kEpsilon}, &ref_out);
    const float err = dtype == kNumberTypeFloat32 ? 1e-4f : kErrBound;
    if (dtype == kNumberTypeFloat32) {
      const auto *out = reinterpret_cast<const float *>(output->MutableData());
      for (size_t i = 0; i < ref_out.size(); ++i) {
        EXPECT_NEAR(ref_out[i], out[i], err) << "output[" << i << "]";
      }
    } else {
      const auto *out = reinterpret_cast<const uint16_t *>(output->MutableData());
      for (size_t i = 0; i < ref_out.size(); ++i) {
        EXPECT_NEAR(ref_out[i], Fp16ToFp32(out[i]), err) << "output[" << i << "]";
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

  // Negative cases: CheckSpecs must reject so the scheduler falls back to CPU.
  void ExpectRejected(const std::vector<int> &in_shape, TypeId scale_dtype, int num_groups, int scale_len = kChannel) {
    InitDSPRuntime();
    const std::vector<int> out_shape = {kBatch, kHeight, kWidth, kChannel};
    auto *input = new lite::Tensor(kNumberTypeFloat32, in_shape, NHWC, lite::Category::CONST_TENSOR);
    auto *scale = new lite::Tensor(scale_dtype, {scale_len}, NHWC, lite::Category::CONST_TENSOR);
    auto *offset = new lite::Tensor(kNumberTypeFloat32, {kChannel}, NHWC, lite::Category::CONST_TENSOR);
    auto *output = new lite::Tensor(kNumberTypeFloat32, out_shape, NHWC, lite::Category::CONST_TENSOR);
    std::vector<lite::Tensor *> inputs = {input, scale, offset};
    std::vector<lite::Tensor *> outputs = {output};
    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    auto *param = static_cast<GroupNormParam *>(malloc(sizeof(GroupNormParam)));
    ASSERT_NE(param, nullptr);
    std::memset(param, 0, sizeof(GroupNormParam));
    param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_GroupNormFusion);
    param->epsilon_ = kEpsilon;
    param->num_groups_ = num_groups;
    param->affine_ = true;

    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC,
                             schema::PrimitiveType_GroupNormFusion};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel_handle = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    EXPECT_EQ(kernel_handle, nullptr);

    delete kernel_handle;
    delete input;
    delete scale;
    delete offset;
    delete output;
    delete ctx;
    UninitDSPRuntime();
  }
};

TEST_F(TestDSP_GroupNormFusion, Fp32) { RunCase(kNumberTypeFloat32); }
TEST_F(TestDSP_GroupNormFusion, Fp16) { RunCase(kNumberTypeFloat16); }
TEST_F(TestDSP_GroupNormFusion, Non4D_FallsBackToCpu) {
  ExpectRejected({kBatch, kChannel, kHeight}, kNumberTypeFloat32, kNumGroups);
}
TEST_F(TestDSP_GroupNormFusion, ScaleDtypeMismatch_FallsBackToCpu) {
  ExpectRejected({kBatch, kHeight, kWidth, kChannel}, kNumberTypeFloat16, kNumGroups);
}
TEST_F(TestDSP_GroupNormFusion, GroupsNotDivisible_FallsBackToCpu) {
  ExpectRejected({kBatch, kHeight, kWidth, kChannel}, kNumberTypeFloat32, 3);
}
TEST_F(TestDSP_GroupNormFusion, ScaleLenMismatch_FallsBackToCpu) {
  ExpectRejected({kBatch, kHeight, kWidth, kChannel}, kNumberTypeFloat32, kNumGroups, kChannel - 1);
}

}  // namespace mindspore::lite::dsp::test

#endif  // SUPPORT_FT04
