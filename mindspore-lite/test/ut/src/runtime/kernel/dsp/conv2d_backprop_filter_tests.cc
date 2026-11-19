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
#include <cstdlib>
#include <cstring>
#include <vector>
#include "ut/src/runtime/kernel/dsp/dsp_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/cpu/nnacl_c/conv_parameter.h"

#ifdef SUPPORT_FT04

namespace mindspore::lite::dsp::test {

namespace {
// NHWC for the activation and the gradient. The weight the gradient produces is
// [out channel, kernel height, kernel width, in channel].
constexpr size_t kBatchDim = 0;
constexpr size_t kHeightDim = 1;
constexpr size_t kWidthDim = 2;
constexpr size_t kChannelDim = 3;
constexpr float kValueStep = 0.5f;
constexpr int kValueCycle = 5;
constexpr int kValueMiddle = 2;
constexpr float kFp32ErrBound = 1e-4f;
constexpr float kFp16ErrBound = 2e-2f;
}  // namespace

class TestDSP_Conv2DBackpropFilter : public DSPCommonTest {};

namespace {
// Small exact values, so the difference against the DSP comes from the accumulation order rather than
// from the input representation.
lite::Tensor *MakeConvTensor(TypeId dtype, const std::vector<int> &shape, int seed,
                             const std::shared_ptr<DSPAllocator> &allocator) {
  auto *tensor = new lite::Tensor(dtype, shape, NHWC, lite::Category::CONST_TENSOR);
  if (tensor->MallocData(allocator) != lite::RET_OK) {
    delete tensor;
    return nullptr;
  }
  const auto count = tensor->ElementsNum();
  auto *float_data = reinterpret_cast<float *>(tensor->MutableData());
  if (dtype == kNumberTypeFloat32) {
    for (int64_t i = 0; i < count; ++i) {
      float_data[i] =
        static_cast<float>(static_cast<int>(i + seed * kValueMiddle) % kValueCycle - kValueMiddle) * kValueStep;
    }
  } else {
    auto *half_data = reinterpret_cast<uint16_t *>(tensor->MutableData());
    for (int64_t i = 0; i < count; ++i) {
      const auto value =
        static_cast<float>(static_cast<int>(i + seed * kValueMiddle) % kValueCycle - kValueMiddle) * kValueStep;
      half_data[i] = Fp32ToFp16(value);
    }
  }
  return tensor;
}

// Lite hands the weight shape to this operator as an int32 tensor; the output tensor carries the same
// numbers after inference.
lite::Tensor *MakeWeightShapeTensor(const std::vector<int> &weight_shape,
                                    const std::shared_ptr<DSPAllocator> &allocator) {
  auto *tensor = new lite::Tensor(kNumberTypeInt32, std::vector<int>{static_cast<int>(weight_shape.size())}, NHWC,
                                  lite::Category::CONST_TENSOR);
  if (tensor->MallocData(allocator) != lite::RET_OK) {
    delete tensor;
    return nullptr;
  }
  auto *data = reinterpret_cast<int32_t *>(tensor->MutableData());
  for (size_t i = 0; i < weight_shape.size(); ++i) {
    data[i] = weight_shape[i];
  }
  return tensor;
}

// LiteKernel::~LiteKernel() releases op_parameter_ with free(), so the parameter has to come from
// malloc rather than new.
ConvParameter *MakeConvParameter(int kernel_h, int kernel_w, int stride, int pad_u, int pad_d, int pad_l, int pad_r,
                                 int dilation_h, int dilation_w, int group) {
  auto *param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (param == nullptr) {
    return nullptr;
  }
  (void)memset(param, 0, sizeof(ConvParameter));
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_Conv2DBackpropFilterFusion);
  param->kernel_h_ = kernel_h;
  param->kernel_w_ = kernel_w;
  param->stride_h_ = stride;
  param->stride_w_ = stride;
  param->dilation_h_ = dilation_h;
  param->dilation_w_ = dilation_w;
  param->pad_u_ = pad_u;
  param->pad_d_ = pad_d;
  param->pad_l_ = pad_l;
  param->pad_r_ = pad_r;
  param->group_ = group;
  return param;
}

kernel::LiteKernel *CreateKernel(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                 TypeId dtype, ConvParameter *param, lite::InnerContext *ctx) {
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, dtype, NHWC, schema::PrimitiveType_Conv2DBackpropFilterFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  if (creator == nullptr) {
    free(param);
    return nullptr;
  }
  return creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
}

std::vector<float> FloatValues(const lite::Tensor *tensor) {
  std::vector<float> values(static_cast<size_t>(tensor->ElementsNum()), 0.0f);
  if (tensor->data_type() == kNumberTypeFloat16) {
    auto *data = reinterpret_cast<const uint16_t *>(tensor->data());
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = Fp16ToFp32(data[i]);
    }
    return values;
  }
  auto *data = reinterpret_cast<const float *>(tensor->data());
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = data[i];
  }
  return values;
}

// dw[co, kh, kw, ci] accumulates dy[n, oh, ow, co] * x[n, ih, iw, ci] over the positions that reach it.
// With groups, output channel co reads only the channels of its own group, which is the slice of the
// weight the gradient fills. The pads and dilations arrive as arguments, and every case passes them the
// same way it fills the parameter, so a mis-ordered argument cannot hide behind a symmetric pad.
void ReferenceWeightGrad(const std::vector<float> &dy_values, const std::vector<float> &x_values,
                         std::vector<float> *dw, const std::vector<int> &dy_shape, const std::vector<int> &x_shape,
                         int kernel_h, int kernel_w, int stride, int pad_u, int pad_d, int pad_l, int pad_r,
                         int dilation_h, int dilation_w, int group) {
  const int batch = dy_shape[kBatchDim];
  const int out_h = dy_shape[kHeightDim];
  const int out_w = dy_shape[kWidthDim];
  const int out_c = dy_shape[kChannelDim];
  const int in_h = x_shape[kHeightDim];
  const int in_w = x_shape[kWidthDim];
  const int in_c = x_shape[kChannelDim];
  const int channels_per_group = in_c / group;
  const int out_channels_per_group = out_c / group;
  for (auto &value : *dw) {
    value = 0.0f;
  }
  for (int n = 0; n < batch; ++n) {
    for (int oh = 0; oh < out_h; ++oh) {
      for (int ow = 0; ow < out_w; ++ow) {
        for (int co = 0; co < out_c; ++co) {
          const float dy_value = dy_values[((static_cast<size_t>(n) * out_h + oh) * out_w + ow) * out_c + co];
          if (dy_value == 0.0f) {
            continue;
          }
          const int group_of_channel = co / out_channels_per_group;
          for (int kh = 0; kh < kernel_h; ++kh) {
            const int ih = oh * stride + kh * dilation_h - pad_u;
            // pad_d says how far past the input the window may reach. What lies there is zero padding,
            // which adds nothing, and it is also past the end of the buffer.
            if (ih < 0 || ih >= in_h + pad_d) {
              continue;
            }
            if (ih >= in_h) {
              continue;
            }
            for (int kw = 0; kw < kernel_w; ++kw) {
              const int iw = ow * stride + kw * dilation_w - pad_l;
              if (iw < 0 || iw >= in_w + pad_r) {
                continue;
              }
              if (iw >= in_w) {
                continue;
              }
              const size_t x_base = ((static_cast<size_t>(n) * in_h + ih) * in_w + iw) * in_c;
              for (int ci = 0; ci < channels_per_group; ++ci) {
                const int x_channel = group_of_channel * channels_per_group + ci;
                const size_t weight_index =
                  ((static_cast<size_t>(co) * kernel_h + kh) * kernel_w + kw) * channels_per_group + ci;
                (*dw)[weight_index] += dy_value * x_values[x_base + x_channel];
              }
            }
          }
        }
      }
    }
  }
}

bool MatchesWeight(const lite::Tensor *weight, const std::vector<float> &expected, float tolerance, int *bad_index) {
  const auto actual = FloatValues(weight);
  for (size_t i = 0; i < actual.size(); ++i) {
    if (std::fabs(actual[i] - expected[i]) > tolerance) {
      *bad_index = static_cast<int>(i);
      return false;
    }
  }
  *bad_index = -1;
  return true;
}
}  // namespace

TEST_F(TestDSP_Conv2DBackpropFilter, Conv2DBackpropFilter_Fp32) {
  InitDSPRuntime();
  auto *dy = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{4, 2, 2, 4}, 0, allocator_);
  auto *x = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{4, 4, 4, 4}, 3, allocator_);
  const std::vector<int> weight_shape = {4, 3, 3, 4};
  auto *shape_tensor = MakeWeightShapeTensor(weight_shape, allocator_);
  ASSERT_NE(dy, nullptr);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(shape_tensor, nullptr);
  auto *weight = new lite::Tensor(kNumberTypeFloat32, weight_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(weight->MallocData(allocator_), lite::RET_OK);
  std::memset(weight->MutableData(), 0, static_cast<size_t>(weight->ElementsNum()) * sizeof(float));
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  auto *param = MakeConvParameter(3, 3, 1, 0, 0, 0, 0, 1, 1, 1);
  ASSERT_NE(param, nullptr);
  auto *kernel = CreateKernel({dy, x, shape_tensor}, {weight}, kNumberTypeFloat32, param, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  std::vector<float> expected(static_cast<size_t>(weight->ElementsNum()), 0.0f);
  ReferenceWeightGrad(FloatValues(dy), FloatValues(x), &expected, dy->shape(), x->shape(), 3, 3, 1, 0, 0, 0, 0, 1, 1,
                      1);
  int bad_index = -1;
  ASSERT_TRUE(MatchesWeight(weight, expected, kFp32ErrBound, &bad_index)) << "weight element " << bad_index;

  delete kernel;
  delete ctx;
  delete dy;
  delete x;
  delete shape_tensor;
  delete weight;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Conv2DBackpropFilter, Conv2DBackpropFilter_Fp32_StrideAndPad) {
  InitDSPRuntime();
  auto *dy = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{1, 2, 2, 2}, 0, allocator_);
  auto *x = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{1, 4, 4, 3}, 3, allocator_);
  const std::vector<int> weight_shape = {2, 3, 3, 3};
  auto *shape_tensor = MakeWeightShapeTensor(weight_shape, allocator_);
  ASSERT_NE(dy, nullptr);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(shape_tensor, nullptr);
  auto *weight = new lite::Tensor(kNumberTypeFloat32, weight_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(weight->MallocData(allocator_), lite::RET_OK);
  std::memset(weight->MutableData(), 0, static_cast<size_t>(weight->ElementsNum()) * sizeof(float));
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // Stride 2 with one row and column of padding on every side, so a kernel that forgets the offset
  // reads positions the reference skips.
  auto *param = MakeConvParameter(3, 3, 2, 1, 1, 1, 1, 1, 1, 1);
  ASSERT_NE(param, nullptr);
  auto *kernel = CreateKernel({dy, x, shape_tensor}, {weight}, kNumberTypeFloat32, param, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  std::vector<float> expected(static_cast<size_t>(weight->ElementsNum()), 0.0f);
  ReferenceWeightGrad(FloatValues(dy), FloatValues(x), &expected, dy->shape(), x->shape(), 3, 3, 2, 1, 1, 1, 1, 1, 1,
                      1);
  int bad_index = -1;
  ASSERT_TRUE(MatchesWeight(weight, expected, kFp32ErrBound, &bad_index)) << "weight element " << bad_index;

  delete kernel;
  delete ctx;
  delete dy;
  delete x;
  delete shape_tensor;
  delete weight;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Conv2DBackpropFilter, Conv2DBackpropFilter_Fp32_AsymmetricPads) {
  InitDSPRuntime();
  auto *dy = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{1, 3, 7, 2}, 0, allocator_);
  auto *x = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{1, 4, 4, 2}, 3, allocator_);
  const std::vector<int> weight_shape = {2, 3, 3, 2};
  auto *shape_tensor = MakeWeightShapeTensor(weight_shape, allocator_);
  ASSERT_NE(dy, nullptr);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(shape_tensor, nullptr);
  auto *weight = new lite::Tensor(kNumberTypeFloat32, weight_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(weight->MallocData(allocator_), lite::RET_OK);
  std::memset(weight->MutableData(), 0, static_cast<size_t>(weight->ElementsNum()) * sizeof(float));
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // Four different pads. The shapes are the ones a forward conv with these pads would produce, so the
  // window really does reach the bottom and right padding: with every pad the same value, swapping two
  // of the arguments in the call to conv2d_backprop_filter_func would be invisible, and here each swap
  // moves the accumulation to another weight element.
  const int kernel_h = 3;
  const int kernel_w = 3;
  const int stride = 1;
  const int pad_u = 0;
  const int pad_d = 1;
  const int pad_l = 2;
  const int pad_r = 3;
  const int dilation_h = 1;
  const int dilation_w = 1;
  const int group = 1;
  auto *param =
    MakeConvParameter(kernel_h, kernel_w, stride, pad_u, pad_d, pad_l, pad_r, dilation_h, dilation_w, group);
  ASSERT_NE(param, nullptr);
  auto *kernel = CreateKernel({dy, x, shape_tensor}, {weight}, kNumberTypeFloat32, param, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  std::vector<float> expected(static_cast<size_t>(weight->ElementsNum()), 0.0f);
  ReferenceWeightGrad(FloatValues(dy), FloatValues(x), &expected, dy->shape(), x->shape(), kernel_h, kernel_w, stride,
                      pad_u, pad_d, pad_l, pad_r, dilation_h, dilation_w, group);
  int bad_index = -1;
  ASSERT_TRUE(MatchesWeight(weight, expected, kFp32ErrBound, &bad_index)) << "weight element " << bad_index;

  delete kernel;
  delete ctx;
  delete dy;
  delete x;
  delete shape_tensor;
  delete weight;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Conv2DBackpropFilter, Conv2DBackpropFilter_Fp32_Dilated) {
  InitDSPRuntime();
  auto *dy = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{1, 2, 3, 2}, 0, allocator_);
  auto *x = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{1, 6, 7, 2}, 3, allocator_);
  const std::vector<int> weight_shape = {2, 3, 3, 2};
  auto *shape_tensor = MakeWeightShapeTensor(weight_shape, allocator_);
  ASSERT_NE(dy, nullptr);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(shape_tensor, nullptr);
  auto *weight = new lite::Tensor(kNumberTypeFloat32, weight_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(weight->MallocData(allocator_), lite::RET_OK);
  std::memset(weight->MutableData(), 0, static_cast<size_t>(weight->ElementsNum()) * sizeof(float));
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // Dilation 2 on a 3x3 kernel spans five cells, so the output is two by three out of a six by seven
  // input. Every third cell of the input takes part, which no other case here reaches.
  const int kernel_h = 3;
  const int kernel_w = 3;
  const int stride = 1;
  const int pad_u = 0;
  const int pad_d = 0;
  const int pad_l = 0;
  const int pad_r = 0;
  const int dilation_h = 2;
  const int dilation_w = 2;
  const int group = 1;
  auto *param =
    MakeConvParameter(kernel_h, kernel_w, stride, pad_u, pad_d, pad_l, pad_r, dilation_h, dilation_w, group);
  ASSERT_NE(param, nullptr);
  auto *kernel = CreateKernel({dy, x, shape_tensor}, {weight}, kNumberTypeFloat32, param, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  std::vector<float> expected(static_cast<size_t>(weight->ElementsNum()), 0.0f);
  ReferenceWeightGrad(FloatValues(dy), FloatValues(x), &expected, dy->shape(), x->shape(), kernel_h, kernel_w, stride,
                      pad_u, pad_d, pad_l, pad_r, dilation_h, dilation_w, group);
  int bad_index = -1;
  ASSERT_TRUE(MatchesWeight(weight, expected, kFp32ErrBound, &bad_index)) << "weight element " << bad_index;

  delete kernel;
  delete ctx;
  delete dy;
  delete x;
  delete shape_tensor;
  delete weight;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Conv2DBackpropFilter, Conv2DBackpropFilter_Fp16) {
  InitDSPRuntime();
  auto *dy = MakeConvTensor(kNumberTypeFloat16, std::vector<int>{2, 2, 2, 2}, 0, allocator_);
  auto *x = MakeConvTensor(kNumberTypeFloat16, std::vector<int>{2, 4, 4, 2}, 3, allocator_);
  const std::vector<int> weight_shape = {2, 3, 3, 2};
  auto *shape_tensor = MakeWeightShapeTensor(weight_shape, allocator_);
  ASSERT_NE(dy, nullptr);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(shape_tensor, nullptr);
  auto *weight = new lite::Tensor(kNumberTypeFloat16, weight_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(weight->MallocData(allocator_), lite::RET_OK);
  std::memset(weight->MutableData(), 0, static_cast<size_t>(weight->ElementsNum()) * sizeof(uint16_t));
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  auto *param = MakeConvParameter(3, 3, 1, 0, 0, 0, 0, 1, 1, 1);
  ASSERT_NE(param, nullptr);
  auto *kernel = CreateKernel({dy, x, shape_tensor}, {weight}, kNumberTypeFloat16, param, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  // The reference accumulates in fp32, the way the armc half-precision case checks it.
  std::vector<float> expected(static_cast<size_t>(weight->ElementsNum()), 0.0f);
  ReferenceWeightGrad(FloatValues(dy), FloatValues(x), &expected, dy->shape(), x->shape(), 3, 3, 1, 0, 0, 0, 0, 1, 1,
                      1);
  int bad_index = -1;
  ASSERT_TRUE(MatchesWeight(weight, expected, kFp16ErrBound, &bad_index)) << "weight element " << bad_index;

  delete kernel;
  delete ctx;
  delete dy;
  delete x;
  delete shape_tensor;
  delete weight;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Conv2DBackpropFilter, Conv2DBackpropFilter_Fp32_Grouped) {
  InitDSPRuntime();
  auto *dy = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{2, 2, 2, 4}, 0, allocator_);
  auto *x = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{2, 3, 3, 4}, 3, allocator_);
  // Two groups: each filter row covers half of the input channels.
  const std::vector<int> weight_shape = {4, 2, 2, 2};
  auto *shape_tensor = MakeWeightShapeTensor(weight_shape, allocator_);
  ASSERT_NE(dy, nullptr);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(shape_tensor, nullptr);
  auto *weight = new lite::Tensor(kNumberTypeFloat32, weight_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(weight->MallocData(allocator_), lite::RET_OK);
  std::memset(weight->MutableData(), 0, static_cast<size_t>(weight->ElementsNum()) * sizeof(float));
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  auto *param = MakeConvParameter(2, 2, 1, 0, 0, 0, 0, 1, 1, 2);
  ASSERT_NE(param, nullptr);
  auto *kernel = CreateKernel({dy, x, shape_tensor}, {weight}, kNumberTypeFloat32, param, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  std::vector<float> expected(static_cast<size_t>(weight->ElementsNum()), 0.0f);
  ReferenceWeightGrad(FloatValues(dy), FloatValues(x), &expected, dy->shape(), x->shape(), 2, 2, 1, 0, 0, 0, 0, 1, 1,
                      2);
  int bad_index = -1;
  ASSERT_TRUE(MatchesWeight(weight, expected, kFp32ErrBound, &bad_index)) << "weight element " << bad_index;

  delete kernel;
  delete ctx;
  delete dy;
  delete x;
  delete shape_tensor;
  delete weight;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Conv2DBackpropFilter, Conv2DBackpropFilter_ShortWeightShapeIsRejected) {
  InitDSPRuntime();
  auto *dy = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{1, 2, 2, 2}, 0, allocator_);
  auto *x = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{1, 4, 4, 2}, 3, allocator_);
  auto *shape_tensor = MakeWeightShapeTensor(std::vector<int>{2, 3, 3}, allocator_);
  ASSERT_NE(dy, nullptr);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(shape_tensor, nullptr);
  auto *weight = new lite::Tensor(kNumberTypeFloat32, std::vector<int>{2, 3, 3, 2}, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(weight->MallocData(allocator_), lite::RET_OK);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  auto *param = MakeConvParameter(3, 3, 1, 0, 0, 0, 0, 1, 1, 1);
  ASSERT_NE(param, nullptr);
  auto *kernel = CreateKernel({dy, x, shape_tensor}, {weight}, kNumberTypeFloat32, param, ctx);
  ASSERT_EQ(kernel, nullptr);

  delete ctx;
  delete dy;
  delete x;
  delete shape_tensor;
  delete weight;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Conv2DBackpropFilter, Conv2DBackpropFilter_BatchMismatchIsRejected) {
  InitDSPRuntime();
  auto *dy = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{4, 2, 2, 2}, 0, allocator_);
  auto *x = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{3, 4, 4, 2}, 3, allocator_);
  auto *shape_tensor = MakeWeightShapeTensor(std::vector<int>{2, 3, 3, 2}, allocator_);
  ASSERT_NE(dy, nullptr);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(shape_tensor, nullptr);
  auto *weight = new lite::Tensor(kNumberTypeFloat32, std::vector<int>{2, 3, 3, 2}, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(weight->MallocData(allocator_), lite::RET_OK);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // One batch count is passed to the DSP for both tensors, so they cannot disagree.
  auto *param = MakeConvParameter(3, 3, 1, 0, 0, 0, 0, 1, 1, 1);
  ASSERT_NE(param, nullptr);
  auto *kernel = CreateKernel({dy, x, shape_tensor}, {weight}, kNumberTypeFloat32, param, ctx);
  ASSERT_EQ(kernel, nullptr);

  delete ctx;
  delete dy;
  delete x;
  delete shape_tensor;
  delete weight;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Conv2DBackpropFilter, Conv2DBackpropFilter_WeightShapeDisagreesIsRejected) {
  InitDSPRuntime();
  auto *dy = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{4, 2, 2, 4}, 0, allocator_);
  auto *x = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{4, 4, 4, 4}, 3, allocator_);
  auto *shape_tensor = MakeWeightShapeTensor(std::vector<int>{4, 3, 3, 3}, allocator_);
  ASSERT_NE(dy, nullptr);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(shape_tensor, nullptr);
  auto *weight = new lite::Tensor(kNumberTypeFloat32, std::vector<int>{4, 3, 3, 3}, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(weight->MallocData(allocator_), lite::RET_OK);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // The buffer holds three input channels while the activation has four: the DSP would read past it.
  auto *param = MakeConvParameter(3, 3, 1, 0, 0, 0, 0, 1, 1, 1);
  ASSERT_NE(param, nullptr);
  auto *kernel = CreateKernel({dy, x, shape_tensor}, {weight}, kNumberTypeFloat32, param, ctx);
  ASSERT_EQ(kernel, nullptr);

  delete ctx;
  delete dy;
  delete x;
  delete shape_tensor;
  delete weight;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Conv2DBackpropFilter, Conv2DBackpropFilter_ZeroStrideIsRejected) {
  InitDSPRuntime();
  auto *dy = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{1, 2, 2, 2}, 0, allocator_);
  auto *x = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{1, 4, 4, 2}, 3, allocator_);
  auto *shape_tensor = MakeWeightShapeTensor(std::vector<int>{2, 3, 3, 2}, allocator_);
  ASSERT_NE(dy, nullptr);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(shape_tensor, nullptr);
  auto *weight = new lite::Tensor(kNumberTypeFloat32, std::vector<int>{2, 3, 3, 2}, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(weight->MallocData(allocator_), lite::RET_OK);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  auto *param = MakeConvParameter(3, 3, 0, 0, 0, 0, 0, 1, 1, 1);
  ASSERT_NE(param, nullptr);
  auto *kernel = CreateKernel({dy, x, shape_tensor}, {weight}, kNumberTypeFloat32, param, ctx);
  ASSERT_EQ(kernel, nullptr);

  delete ctx;
  delete dy;
  delete x;
  delete shape_tensor;
  delete weight;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Conv2DBackpropFilter, Conv2DBackpropFilter_PadModeSameIsRejected) {
  InitDSPRuntime();
  auto *dy = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{1, 2, 2, 2}, 0, allocator_);
  auto *x = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{1, 4, 4, 2}, 3, allocator_);
  auto *shape_tensor = MakeWeightShapeTensor(std::vector<int>{2, 3, 3, 2}, allocator_);
  ASSERT_NE(dy, nullptr);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(shape_tensor, nullptr);
  auto *weight = new lite::Tensor(kNumberTypeFloat32, std::vector<int>{2, 3, 3, 2}, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(weight->MallocData(allocator_), lite::RET_OK);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // Under Pad_same the padding is derived while the graph runs, and the pad fields here can still say
  // zero. Accumulating with those would put weight in cells the forward pass cropped, which reads as a
  // correct result with wrong numbers, so the kernel declines and the scheduler keeps it on CPU.
  auto *param = MakeConvParameter(3, 3, 1, 0, 0, 0, 0, 1, 1, 1);
  ASSERT_NE(param, nullptr);
  param->pad_mode_ = Pad_same;
  auto *kernel = CreateKernel({dy, x, shape_tensor}, {weight}, kNumberTypeFloat32, param, ctx);
  ASSERT_EQ(kernel, nullptr);

  delete ctx;
  delete dy;
  delete x;
  delete shape_tensor;
  delete weight;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Conv2DBackpropFilter, Conv2DBackpropFilter_ResizeShrinksTheWeight) {
  InitDSPRuntime();
  auto *dy = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{2, 2, 2, 2}, 0, allocator_);
  auto *x = MakeConvTensor(kNumberTypeFloat32, std::vector<int>{2, 4, 4, 2}, 3, allocator_);
  auto *shape_tensor = MakeWeightShapeTensor(std::vector<int>{2, 3, 3, 2}, allocator_);
  ASSERT_NE(dy, nullptr);
  ASSERT_NE(x, nullptr);
  ASSERT_NE(shape_tensor, nullptr);
  auto *weight = new lite::Tensor(kNumberTypeFloat32, std::vector<int>{2, 3, 3, 2}, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(weight->MallocData(allocator_), lite::RET_OK);
  std::memset(weight->MutableData(), 0, static_cast<size_t>(weight->ElementsNum()) * sizeof(float));
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  auto *param = MakeConvParameter(3, 3, 1, 0, 0, 0, 0, 1, 1, 1);
  ASSERT_NE(param, nullptr);
  auto *kernel = CreateKernel({dy, x, shape_tensor}, {weight}, kNumberTypeFloat32, param, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Run());

  // A resize leaves CheckSpecs behind, so the weight shape has to be read again before the next run.
  weight->set_shape({2, 3, 3, 1});
  ASSERT_EQ(RET_ERROR, kernel->Prepare());
  ASSERT_EQ(RET_ERROR, kernel->Run());

  delete kernel;
  delete ctx;
  delete dy;
  delete x;
  delete shape_tensor;
  delete weight;
  UninitDSPRuntime();
}
}  // namespace mindspore::lite::dsp::test
#endif  // SUPPORT_FT04
