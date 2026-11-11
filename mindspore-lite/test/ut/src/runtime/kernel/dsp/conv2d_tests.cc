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
#include <cstdlib>
#include <cstring>
#include <vector>
#include "ut/src/runtime/kernel/dsp/dsp_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel/dsp/dsp_subgraph.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/cpu/nnacl_c/conv_parameter.h"

#ifdef SUPPORT_FT04

namespace mindspore::lite::dsp::test {

namespace {
constexpr int kBatch = 1;
constexpr int kInputH = 4;
constexpr int kInputW = 4;
constexpr int kInChannel = 2;
constexpr int kOutChannel = 2;
constexpr int kKernelH = 3;
constexpr int kKernelW = 3;
constexpr int kStrideH = 1;
constexpr int kStrideW = 1;
constexpr int kPadH = 1;
constexpr int kPadW = 1;
constexpr int kDilationH = 1;
constexpr int kDilationW = 1;
constexpr int kGroup = 1;
constexpr int kOutputH = 4;
constexpr int kOutputW = 4;
// fp16 tolerance from on-device measurement: max abs error of a correct run is
// one ULP (0.0625 at the observed output magnitude 6..105), max rel error ~1e-3.
// Keep a 1-ULP abs bound and tighten rel to 1% so padding/bias errors are caught.
constexpr double kFp16AbsTol = 0.0625;
constexpr double kFp16RelTol = 1e-2;
constexpr float kFp32ErrBound = 1e-4f;
constexpr float kInputScale = 0.1f;
constexpr float kBiasValue = 0.5f;

struct ConvBuffers {
  const void *input;
  const void *weight;
  const void *bias;
  void *output;
  TypeId dtype;
  bool has_bias;
};

double LoadAsDouble(const void *buf, size_t i, TypeId t) {
  if (t == kNumberTypeFloat16) {
    return static_cast<double>(Fp16ToFp32(reinterpret_cast<const uint16_t *>(buf)[i]));
  }
  return static_cast<double>(reinterpret_cast<const float *>(buf)[i]);
}

void StoreFromDouble(void *buf, size_t i, TypeId t, double v) {
  if (t == kNumberTypeFloat16) {
    reinterpret_cast<uint16_t *>(buf)[i] = Fp32ToFp16(static_cast<float>(v));
  } else {
    reinterpret_cast<float *>(buf)[i] = static_cast<float>(v);
  }
}

size_t DtypeSize(TypeId t) { return t == kNumberTypeFloat16 ? sizeof(uint16_t) : sizeof(float); }

void FillData(void *buf, size_t n, TypeId t, float base, float step) {
  for (size_t i = 0; i < n; ++i) {
    StoreFromDouble(buf, i, t, static_cast<double>(base + step * static_cast<float>(i)));
  }
}

// NHWC cross-correlation reference. Input/weight/bias/output are contiguous.
void RefConv2d(const ConvBuffers &buf) {
  for (int oh = 0; oh < kOutputH; ++oh) {
    for (int ow = 0; ow < kOutputW; ++ow) {
      for (int oc = 0; oc < kOutChannel; ++oc) {
        double acc = 0.0;
        for (int kh = 0; kh < kKernelH; ++kh) {
          for (int kw = 0; kw < kKernelW; ++kw) {
            const int ih = oh * kStrideH - kPadH + kh * kDilationH;
            const int iw = ow * kStrideW - kPadW + kw * kDilationW;
            if (ih < 0 || ih >= kInputH || iw < 0 || iw >= kInputW) {
              continue;
            }
            for (int ic = 0; ic < kInChannel; ++ic) {
              const size_t in_idx = ((ih * kInputW + iw) * kInChannel + ic);
              const size_t w_idx = ((oc * kKernelH + kh) * kKernelW + kw) * kInChannel + ic;
              acc += LoadAsDouble(buf.input, in_idx, buf.dtype) * LoadAsDouble(buf.weight, w_idx, buf.dtype);
            }
          }
        }
        if (buf.has_bias) {
          acc += LoadAsDouble(buf.bias, static_cast<size_t>(oc), buf.dtype);
        }
        const size_t out_idx = ((oh * kOutputW + ow) * kOutChannel + oc);
        StoreFromDouble(buf.output, out_idx, buf.dtype, acc);
      }
    }
  }
}

ConvParameter *CreateConvParameter() {
  auto *param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (param == nullptr) {
    return nullptr;
  }
  std::memset(param, 0, sizeof(ConvParameter));
  param->op_parameter_.type_ = schema::PrimitiveType_Conv2DFusion;
  param->kernel_h_ = kKernelH;
  param->kernel_w_ = kKernelW;
  param->stride_h_ = kStrideH;
  param->stride_w_ = kStrideW;
  param->pad_u_ = kPadH;
  param->pad_d_ = kPadH;
  param->pad_l_ = kPadW;
  param->pad_r_ = kPadW;
  param->dilation_h_ = kDilationH;
  param->dilation_w_ = kDilationW;
  param->group_ = kGroup;
  param->input_channel_ = kInChannel;
  param->output_channel_ = kOutChannel;
  return param;
}

// Return nullptr on allocation failure so the caller can ASSERT_NE and stop
// before writing into a null device buffer (a raw EXPECT would only log and
// let FillData/memset dereference nullptr).
lite::Tensor *MakeTensor(TypeId dtype, const std::vector<int> &shape, const std::shared_ptr<DSPAllocator> &allocator) {
  auto *tensor = new lite::Tensor(dtype, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  if (tensor->MallocData(allocator) != lite::RET_OK || tensor->MutableData() == nullptr) {
    delete tensor;
    return nullptr;
  }
  return tensor;
}

// Match armc's fp16 verification: a mismatch requires both abs and rel error over tolerance.
void VerifyOutput(const void *actual, const void *expected, size_t num, TypeId dtype) {
  const bool is_fp16 = dtype == kNumberTypeFloat16;
  for (size_t i = 0; i < num; ++i) {
    const double a = LoadAsDouble(actual, i, dtype);
    const double e = LoadAsDouble(expected, i, dtype);
    const double abs_err = std::fabs(a - e);
    if (!is_fp16) {
      ASSERT_LE(abs_err, kFp32ErrBound) << "mismatch at element " << i;
      continue;
    }
    // |e| is >= 6 in these cases, so the relative-error denominator is stable.
    const double rel_err = abs_err / std::fabs(e);
    ASSERT_TRUE(abs_err <= kFp16AbsTol || rel_err <= kFp16RelTol)
      << "mismatch at element " << i << " abs=" << abs_err << " rel=" << rel_err;
  }
}
ConvBuffers BuildConvTensors(TypeId dtype, bool has_bias, const std::shared_ptr<DSPAllocator> &allocator,
                             std::vector<lite::Tensor *> *inputs, std::vector<lite::Tensor *> *outputs) {
  const size_t input_num = static_cast<size_t>(kBatch * kInputH * kInputW * kInChannel);
  const size_t weight_num = static_cast<size_t>(kOutChannel * kKernelH * kKernelW * kInChannel / kGroup);
  const size_t bias_num = static_cast<size_t>(kOutChannel);
  const size_t output_num = static_cast<size_t>(kBatch * kOutputH * kOutputW * kOutChannel);

  std::vector<int> input_shape = {kBatch, kInputH, kInputW, kInChannel};
  std::vector<int> weight_shape = {kOutChannel, kKernelH, kKernelW, kInChannel / kGroup};
  std::vector<int> bias_shape = {kOutChannel};
  std::vector<int> output_shape = {kBatch, kOutputH, kOutputW, kOutChannel};

  auto *input = MakeTensor(dtype, input_shape, allocator);
  if (input == nullptr) {
    return {nullptr, nullptr, nullptr, nullptr, dtype, has_bias};
  }
  inputs->push_back(input);
  FillData(input->MutableData(), input_num, dtype, 0.0f, kInputScale);

  auto *weight = MakeTensor(dtype, weight_shape, allocator);
  if (weight == nullptr) {
    return {nullptr, nullptr, nullptr, nullptr, dtype, has_bias};
  }
  inputs->push_back(weight);
  FillData(weight->MutableData(), weight_num, dtype, 0.0f, kInputScale);

  const void *bias_data = nullptr;
  if (has_bias) {
    auto *bias = MakeTensor(dtype, bias_shape, allocator);
    if (bias == nullptr) {
      return {nullptr, nullptr, nullptr, nullptr, dtype, has_bias};
    }
    inputs->push_back(bias);
    FillData(bias->MutableData(), bias_num, dtype, kBiasValue, 0.0f);
    bias_data = bias->MutableData();
  }

  auto *output = MakeTensor(dtype, output_shape, allocator);
  if (output == nullptr) {
    return {nullptr, nullptr, nullptr, nullptr, dtype, has_bias};
  }
  outputs->push_back(output);
  std::memset(output->MutableData(), 0, output_num * DtypeSize(dtype));

  return {input->MutableData(), weight->MutableData(), bias_data, output->MutableData(), dtype, has_bias};
}
}  // namespace

class TestDSP_Conv2d : public DSPCommonTest {
 public:
  // Mixed input/weight dtype must be rejected by CheckSpecs so the scheduler
  // falls back to CPU, instead of launching a DSP kernel built for one dtype.
  void ExpectMixedDtypeFallsBackToCpu() {
    InitDSPRuntime();
    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    std::vector<int> input_shape = {kBatch, kInputH, kInputW, kInChannel};
    std::vector<int> weight_shape = {kOutChannel, kKernelH, kKernelW, kInChannel / kGroup};
    std::vector<int> output_shape = {kBatch, kOutputH, kOutputW, kOutChannel};

    auto *param = CreateConvParameter();
    ASSERT_NE(param, nullptr);
    auto *input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    auto *weight = new lite::Tensor(kNumberTypeFloat16, weight_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    auto *output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    inputs.push_back(input);
    inputs.push_back(weight);
    outputs.push_back(output);

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Conv2DFusion};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    EXPECT_EQ(kernel, nullptr);

    delete ctx;
    for (auto *t : inputs) {
      delete t;
    }
    for (auto *t : outputs) {
      delete t;
    }
    delete kernel;
    UninitDSPRuntime();
  }

  void RunConvCase(TypeId dtype, bool has_bias) {
    InitDSPRuntime();
    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;

    auto *param = CreateConvParameter();
    ASSERT_NE(param, nullptr);

    const size_t output_num = static_cast<size_t>(kBatch * kOutputH * kOutputW * kOutChannel);
    ConvBuffers buf = BuildConvTensors(dtype, has_bias, allocator_, &inputs, &outputs);
    ASSERT_NE(buf.output, nullptr) << "conv2d test tensor allocation failed";

    std::vector<uint8_t> ref(output_num * DtypeSize(dtype));
    ConvBuffers ref_buf = {buf.input, buf.weight, buf.bias, ref.data(), dtype, has_bias};
    RefConv2d(ref_buf);

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());

    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, dtype, NHWC, schema::PrimitiveType_Conv2DFusion};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    ASSERT_NE(kernel, nullptr);

    ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
    ASSERT_EQ(kernel->Run(), lite::RET_OK);
    VerifyOutput(buf.output, ref.data(), output_num, dtype);

    UninitDSPRuntime();
    delete ctx;
    for (auto *t : inputs) {
      delete t;
    }
    for (auto *t : outputs) {
      delete t;
    }
    delete kernel;
  }
};

TEST_F(TestDSP_Conv2d, Fp32_WithBias) { RunConvCase(kNumberTypeFloat32, true); }
TEST_F(TestDSP_Conv2d, Fp32_NoBias) { RunConvCase(kNumberTypeFloat32, false); }
TEST_F(TestDSP_Conv2d, Fp16_WithBias) { RunConvCase(kNumberTypeFloat16, true); }
TEST_F(TestDSP_Conv2d, Fp16_NoBias) { RunConvCase(kNumberTypeFloat16, false); }

TEST_F(TestDSP_Conv2d, MixedDtype_FallsBackToCpu) { ExpectMixedDtypeFallsBackToCpu(); }

// Asymmetric padding must be rejected by CheckSpecs so the scheduler falls
// back to CPU, since conv2d_func only supports symmetric padding.
// An inconsistent group (in_channel not divisible by group) must be rejected
// by CheckSpecs so the scheduler falls back to CPU.
TEST_F(TestDSP_Conv2d, InvalidGroup_FallsBackToCpu) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  std::vector<int> input_shape = {kBatch, kInputH, kInputW, kInChannel};
  std::vector<int> weight_shape = {kOutChannel, kKernelH, kKernelW, kInChannel / kGroup};
  std::vector<int> output_shape = {kBatch, kOutputH, kOutputW, kOutChannel};

  auto *param = CreateConvParameter();
  ASSERT_NE(param, nullptr);
  param->group_ = kInChannel + 1;  // cannot divide in_channel
  auto *input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  auto *weight = new lite::Tensor(kNumberTypeFloat32, weight_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  auto *output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  inputs.push_back(input);
  inputs.push_back(weight);
  outputs.push_back(output);

  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Conv2DFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
  EXPECT_EQ(kernel, nullptr);

  delete ctx;
  for (auto *t : inputs) {
    delete t;
  }
  for (auto *t : outputs) {
    delete t;
  }
  delete kernel;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Conv2d, AsymmetricPadding_FallsBackToCpu) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  std::vector<int> input_shape = {kBatch, kInputH, kInputW, kInChannel};
  std::vector<int> weight_shape = {kOutChannel, kKernelH, kKernelW, kInChannel / kGroup};
  std::vector<int> output_shape = {kBatch, kOutputH, kOutputW, kOutChannel};

  auto *param = CreateConvParameter();
  ASSERT_NE(param, nullptr);
  param->pad_d_ = kPadH + 1;  // asymmetric: pad_u_ != pad_d_
  auto *input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  auto *weight = new lite::Tensor(kNumberTypeFloat32, weight_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  auto *output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  inputs.push_back(input);
  inputs.push_back(weight);
  outputs.push_back(output);

  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Conv2DFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
  EXPECT_EQ(kernel, nullptr);

  delete ctx;
  for (auto *t : inputs) {
    delete t;
  }
  for (auto *t : outputs) {
    delete t;
  }
  delete kernel;
  UninitDSPRuntime();
}

}  // namespace mindspore::lite::dsp::test

#endif  // SUPPORT_FT04
