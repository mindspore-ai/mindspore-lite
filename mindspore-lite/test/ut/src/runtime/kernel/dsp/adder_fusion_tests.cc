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

class TestDSP_AdderFusion : public DSPCommonTest {};

TEST_F(TestDSP_AdderFusion, Fp32_Kernel1x1) {
  // 1x1 kernel, no padding: out = -|x - w| + b.
  InitDSPRuntime();
  const int batch = 1;
  const int height = 3;
  const int width = 3;
  const int in_channel = 1;
  const int out_channel = 1;
  const int kernel = 1;
  const int num = batch * height * width * in_channel;

  std::vector<int> input_shape = {batch, height, width, in_channel};
  std::vector<int> weight_shape = {out_channel, kernel, kernel, in_channel};
  std::vector<int> bias_shape = {out_channel};
  std::vector<int> output_shape = {batch, height, width, out_channel};

  auto *input = new lite::Tensor(kNumberTypeFloat32, input_shape, NHWC, lite::Category::CONST_TENSOR);
  auto *weight = new lite::Tensor(kNumberTypeFloat32, weight_shape, NHWC, lite::Category::CONST_TENSOR);
  auto *bias = new lite::Tensor(kNumberTypeFloat32, bias_shape, NHWC, lite::Category::CONST_TENSOR);
  auto *output = new lite::Tensor(kNumberTypeFloat32, output_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  ASSERT_EQ(weight->MallocData(allocator_), lite::RET_OK);
  ASSERT_EQ(bias->MallocData(allocator_), lite::RET_OK);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);

  auto *input_data = reinterpret_cast<float *>(input->MutableData());
  for (int i = 0; i < num; ++i) {
    input_data[i] = static_cast<float>(i) - 4.0f;
  }
  const float kernel_value = 2.0f;
  const float bias_value = 5.0f;
  reinterpret_cast<float *>(weight->MutableData())[0] = kernel_value;
  reinterpret_cast<float *>(bias->MutableData())[0] = bias_value;
  std::memset(output->MutableData(), 0, static_cast<size_t>(num) * sizeof(float));

  std::vector<lite::Tensor *> inputs_ = {input, weight, bias};
  std::vector<lite::Tensor *> outputs_ = {output};
  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // LiteKernel::~LiteKernel() releases op_parameter_ with free(), so the parameter
  // has to come from malloc rather than new.
  auto *param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(ConvParameter));
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_AdderFusion);
  param->kernel_h_ = kernel;
  param->kernel_w_ = kernel;
  param->stride_h_ = 1;
  param->stride_w_ = 1;
  param->dilation_h_ = 1;
  param->dilation_w_ = 1;
  param->group_ = 1;
  param->input_batch_ = batch;
  param->input_h_ = height;
  param->input_w_ = width;
  param->input_channel_ = in_channel;
  param->output_channel_ = out_channel;

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_AdderFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel_handle = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel_handle, nullptr);
  ASSERT_EQ(kernel_handle->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel_handle->Run(), lite::RET_OK);

  std::vector<float> expect(num, 0.0f);
  for (int i = 0; i < num; ++i) {
    expect[i] = -std::fabs(input_data[i] - kernel_value) + bias_value;
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(output->MutableData()), expect.data(), num, 1e-4));

  delete ctx;
  delete input;
  delete weight;
  delete bias;
  delete output;
  delete kernel_handle;
  // The runtime owns the allocator used to release device memory above.
  UninitDSPRuntime();
}

TEST_F(TestDSP_AdderFusion, Fp32_Kernel3x3_Pad1) {
  // 3x3 kernel with pad = 1; padding area counts as zero input.
  InitDSPRuntime();
  const int batch = 1;
  const int height = 3;
  const int width = 3;
  const int in_channel = 1;
  const int out_channel = 1;
  const int kernel = 3;
  const int pad = 1;
  const int num = batch * height * width * out_channel;

  std::vector<int> input_shape = {batch, height, width, in_channel};
  std::vector<int> weight_shape = {out_channel, kernel, kernel, in_channel};
  std::vector<int> bias_shape = {out_channel};
  std::vector<int> output_shape = {batch, height, width, out_channel};

  auto *input = new lite::Tensor(kNumberTypeFloat32, input_shape, NHWC, lite::Category::CONST_TENSOR);
  auto *weight = new lite::Tensor(kNumberTypeFloat32, weight_shape, NHWC, lite::Category::CONST_TENSOR);
  auto *bias = new lite::Tensor(kNumberTypeFloat32, bias_shape, NHWC, lite::Category::CONST_TENSOR);
  auto *output = new lite::Tensor(kNumberTypeFloat32, output_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  ASSERT_EQ(weight->MallocData(allocator_), lite::RET_OK);
  ASSERT_EQ(bias->MallocData(allocator_), lite::RET_OK);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);

  auto *input_data = reinterpret_cast<float *>(input->MutableData());
  for (int i = 0; i < num; ++i) {
    input_data[i] = static_cast<float>(i % 3);
  }
  auto *weight_data = reinterpret_cast<float *>(weight->MutableData());
  // Distinct weights are required to expose kernel index transposition (kh/kw swap),
  // kernel flipping and channel-order mistakes, which identical weights would hide.
  for (int i = 0; i < kernel * kernel; ++i) {
    weight_data[i] = 0.5f + static_cast<float>(i);
  }
  reinterpret_cast<float *>(bias->MutableData())[0] = 0.0f;
  std::memset(output->MutableData(), 0, static_cast<size_t>(num) * sizeof(float));

  std::vector<lite::Tensor *> inputs_ = {input, weight, bias};
  std::vector<lite::Tensor *> outputs_ = {output};
  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // LiteKernel::~LiteKernel() releases op_parameter_ with free(), so the parameter
  // has to come from malloc rather than new.
  auto *param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(ConvParameter));
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_AdderFusion);
  param->kernel_h_ = kernel;
  param->kernel_w_ = kernel;
  param->stride_h_ = 1;
  param->stride_w_ = 1;
  param->dilation_h_ = 1;
  param->dilation_w_ = 1;
  param->pad_u_ = pad;
  param->pad_d_ = pad;
  param->pad_l_ = pad;
  param->pad_r_ = pad;
  param->group_ = 1;
  param->input_batch_ = batch;
  param->input_h_ = height;
  param->input_w_ = width;
  param->input_channel_ = in_channel;
  param->output_channel_ = out_channel;

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_AdderFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel_handle = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel_handle, nullptr);
  ASSERT_EQ(kernel_handle->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel_handle->Run(), lite::RET_OK);

  std::vector<float> expect(num, 0.0f);
  for (int oh = 0; oh < height; ++oh) {
    for (int ow = 0; ow < width; ++ow) {
      float sum = 0.0f;
      for (int kh = 0; kh < kernel; ++kh) {
        for (int kw = 0; kw < kernel; ++kw) {
          int ih = oh - pad + kh;
          int iw = ow - pad + kw;
          float in_value = (ih >= 0 && ih < height && iw >= 0 && iw < width) ? input_data[ih * width + iw] : 0.0f;
          sum += std::fabs(in_value - weight_data[kh * kernel + kw]);
        }
      }
      expect[oh * width + ow] = -sum;
    }
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(output->MutableData()), expect.data(), num, 1e-4));

  delete ctx;
  delete input;
  delete weight;
  delete bias;
  delete output;
  delete kernel_handle;
  // The runtime owns the allocator used to release device memory above.
  UninitDSPRuntime();
}
TEST_F(TestDSP_AdderFusion, Fp16_Kernel3x3_Pad1) {
  // Same as Fp32_Kernel3x3_Pad1 but in fp16; covers the FT04 fp16 registration.
  InitDSPRuntime();
  const int batch = 1;
  const int height = 3;
  const int width = 3;
  const int in_channel = 1;
  const int out_channel = 1;
  const int kernel = 3;
  const int pad = 1;
  const int num = batch * height * width * out_channel;

  std::vector<int> input_shape = {batch, height, width, in_channel};
  std::vector<int> weight_shape = {out_channel, kernel, kernel, in_channel};
  std::vector<int> bias_shape = {out_channel};
  std::vector<int> output_shape = {batch, height, width, out_channel};

  auto *input = new lite::Tensor(kNumberTypeFloat16, input_shape, NHWC, lite::Category::CONST_TENSOR);
  auto *weight = new lite::Tensor(kNumberTypeFloat16, weight_shape, NHWC, lite::Category::CONST_TENSOR);
  auto *bias = new lite::Tensor(kNumberTypeFloat16, bias_shape, NHWC, lite::Category::CONST_TENSOR);
  auto *output = new lite::Tensor(kNumberTypeFloat16, output_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  ASSERT_EQ(weight->MallocData(allocator_), lite::RET_OK);
  ASSERT_EQ(bias->MallocData(allocator_), lite::RET_OK);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);

  auto *input_data = reinterpret_cast<uint16_t *>(input->MutableData());
  for (int i = 0; i < num; ++i) {
    input_data[i] = Fp32ToFp16(static_cast<float>(i % 3));
  }
  auto *weight_data = reinterpret_cast<uint16_t *>(weight->MutableData());
  // Distinct weights are required to expose kernel index transposition (kh/kw swap),
  // kernel flipping and channel-order mistakes, which identical weights would hide.
  for (int i = 0; i < kernel * kernel; ++i) {
    weight_data[i] = Fp32ToFp16(0.5f + static_cast<float>(i));
  }
  reinterpret_cast<uint16_t *>(bias->MutableData())[0] = Fp32ToFp16(0.0f);
  std::memset(output->MutableData(), 0, static_cast<size_t>(num) * sizeof(uint16_t));

  std::vector<lite::Tensor *> inputs_ = {input, weight, bias};
  std::vector<lite::Tensor *> outputs_ = {output};
  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // LiteKernel::~LiteKernel() releases op_parameter_ with free(), so the parameter
  // has to come from malloc rather than new.
  auto *param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(ConvParameter));
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_AdderFusion);
  param->kernel_h_ = kernel;
  param->kernel_w_ = kernel;
  param->stride_h_ = 1;
  param->stride_w_ = 1;
  param->dilation_h_ = 1;
  param->dilation_w_ = 1;
  param->pad_u_ = pad;
  param->pad_d_ = pad;
  param->pad_l_ = pad;
  param->pad_r_ = pad;
  param->group_ = 1;
  param->input_batch_ = batch;
  param->input_h_ = height;
  param->input_w_ = width;
  param->input_channel_ = in_channel;
  param->output_channel_ = out_channel;

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_AdderFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel_handle = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel_handle, nullptr);
  ASSERT_EQ(kernel_handle->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel_handle->Run(), lite::RET_OK);

  // Rebuild the reference in fp32 from the fp16 inputs, then compare with a relative
  // bound because fp16 keeps roughly 10 bits of mantissa.
  std::vector<float> expect(num, 0.0f);
  for (int oh = 0; oh < height; ++oh) {
    for (int ow = 0; ow < width; ++ow) {
      float sum = 0.0f;
      for (int kh = 0; kh < kernel; ++kh) {
        for (int kw = 0; kw < kernel; ++kw) {
          int ih = oh - pad + kh;
          int iw = ow - pad + kw;
          float in_value = 0.0f;
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            in_value = Fp16ToFp32(input_data[ih * width + iw]);
          }
          sum += std::fabs(in_value - Fp16ToFp32(weight_data[kh * kernel + kw]));
        }
      }
      expect[oh * width + ow] = -sum;
    }
  }
  constexpr float kFp16ErrBound = 5e-2f;
  auto *output_data = reinterpret_cast<uint16_t *>(output->MutableData());
  for (int i = 0; i < num; ++i) {
    const float tolerance = kFp16ErrBound * (1.0f + std::fabs(expect[i]));
    ASSERT_NEAR(Fp16ToFp32(output_data[i]), expect[i], tolerance);
  }

  delete ctx;
  delete input;
  delete weight;
  delete bias;
  delete output;
  delete kernel_handle;
  // The runtime owns the allocator used to release memory above.
  UninitDSPRuntime();
}

}  // namespace mindspore::lite::dsp::test
#endif
