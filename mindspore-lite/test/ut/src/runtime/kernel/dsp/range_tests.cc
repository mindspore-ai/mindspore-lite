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
#include <cstdlib>
#include <limits>
#include <vector>
#include "ut/src/runtime/kernel/dsp/dsp_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "nnacl_c/range_parameter.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel/dsp/dsp_subgraph.h"
#include "src/litert/kernel_registry.h"

namespace mindspore::lite::dsp::test {

namespace {
typedef int16_t float16;
static inline float fp16_to_fp32(float16 h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exp = (h & 0x7C00) >> 10;
  uint32_t frac = (h & 0x03FF);
  uint32_t f_exp, f_frac;
  if (exp == 0) {
    if (frac == 0) {
      f_exp = 0;
      f_frac = 0;
    } else {
      int shift = 0;
      while ((frac & 0x0200) == 0) {
        frac <<= 1;
        ++shift;
      }
      frac &= 0x03FF;
      f_exp = 127 - 15 - shift;
      f_frac = frac << 13;
    }
  } else if (exp == 0x1F) {
    f_exp = 255;
    f_frac = frac << 13;
  } else {
    f_exp = exp - 15 + 127;
    f_frac = frac << 13;
  }
  uint32_t f_bits = sign | (f_exp << 23) | f_frac;
  float result;
  std::memcpy(&result, &f_bits, sizeof(result));
  return result;
}
}  // namespace

class TestDSP_Range : public DSPCommonTest {};

#ifdef SUPPORT_FT04
TEST_F(TestDSP_Range, Range_FT04_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;

  std::vector<int> output_shape = {100};
  int num = output_shape[0];

  auto param = reinterpret_cast<RangeParameter *>(malloc(sizeof(RangeParameter)));
  ASSERT_NE(param, nullptr);
  std::memset(param, 0, sizeof(RangeParameter));
  param->start_ = 0;
  param->limit_ = 100;
  param->delta_ = 1;
  param->dtype_ = static_cast<int>(kNumberTypeFloat32);

  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs.push_back(output);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0.f);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Range};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);

  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel->Run(), lite::RET_OK);

  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = static_cast<float>(i);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0]->MutableData()), correct.data(), num));

  UninitDSPRuntime();
  delete ctx;
  for (auto t : outputs) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_Range, Range_FT04_Int32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;

  std::vector<int> output_shape = {100};
  int num = output_shape[0];

  auto param = reinterpret_cast<RangeParameter *>(malloc(sizeof(RangeParameter)));
  ASSERT_NE(param, nullptr);
  std::memset(param, 0, sizeof(RangeParameter));
  param->start_ = 0;
  param->limit_ = 100;
  param->delta_ = 1;
  param->dtype_ = static_cast<int>(kNumberTypeInt32);

  auto output = new lite::Tensor(kNumberTypeInt32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs.push_back(output);
  std::fill_n(reinterpret_cast<int32_t *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_Range};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);

  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel->Run(), lite::RET_OK);

  std::vector<int32_t> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = i;
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int32_t *>(outputs[0]->MutableData()), correct.data(), num));

  UninitDSPRuntime();
  delete ctx;
  for (auto t : outputs) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_Range, Range_FT04_Int16) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;

  std::vector<int> output_shape = {50};
  int num = output_shape[0];

  auto param = reinterpret_cast<RangeParameter *>(malloc(sizeof(RangeParameter)));
  ASSERT_NE(param, nullptr);
  std::memset(param, 0, sizeof(RangeParameter));
  param->start_ = 0;
  param->limit_ = 100;
  param->delta_ = 2;
  param->dtype_ = static_cast<int>(kNumberTypeInt16);

  auto output = new lite::Tensor(kNumberTypeInt16, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs.push_back(output);
  std::fill_n(reinterpret_cast<int16_t *>(output->MutableData()), num, static_cast<int16_t>(0));

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt16, NHWC, schema::PrimitiveType_Range};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);

  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel->Run(), lite::RET_OK);

  std::vector<int16_t> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = static_cast<int16_t>(i * 2);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int16_t *>(outputs[0]->MutableData()), correct.data(), num));

  UninitDSPRuntime();
  delete ctx;
  for (auto t : outputs) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_Range, Range_FT04_Fp16) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;

  std::vector<int> output_shape = {1024};
  int num = output_shape[0];

  auto param = reinterpret_cast<RangeParameter *>(malloc(sizeof(RangeParameter)));
  ASSERT_NE(param, nullptr);
  std::memset(param, 0, sizeof(RangeParameter));
  param->start_ = 0;
  param->limit_ = 100;
  param->delta_ = 1;
  param->dtype_ = static_cast<int>(kNumberTypeFloat16);

  auto output = new lite::Tensor(kNumberTypeFloat16, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs.push_back(output);
  std::memset(output->MutableData(), 0, static_cast<size_t>(num) * sizeof(uint16_t));

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_Range};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);

  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel->Run(), lite::RET_OK);

  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = static_cast<float>(i);
  }
  auto out_fp16 = reinterpret_cast<uint16_t *>(outputs[0]->MutableData());
  std::vector<float> actual(num);
  for (int i = 0; i < num; ++i) {
    actual[i] = fp16_to_fp32(out_fp16[i]);
  }
  ASSERT_EQ(0, CompareOutputData(actual.data(), correct.data(), num, 1e-4));

  UninitDSPRuntime();
  delete ctx;
  for (auto t : outputs) {
    delete t;
  }
  delete kernel;
}
#endif

#ifdef SUPPORT_FT78
TEST_F(TestDSP_Range, Range_FT78_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;

  std::vector<int> output_shape = {100};
  int num = output_shape[0];

  auto param = reinterpret_cast<RangeParameter *>(malloc(sizeof(RangeParameter)));
  ASSERT_NE(param, nullptr);
  std::memset(param, 0, sizeof(RangeParameter));
  param->start_ = 0;
  param->limit_ = 100;
  param->delta_ = 1;
  param->dtype_ = static_cast<int>(kNumberTypeFloat32);

  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs.push_back(output);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0.f);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Range};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);

  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel->Run(), lite::RET_OK);

  std::vector<float> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = static_cast<float>(i);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs[0]->MutableData()), correct.data(), num));

  UninitDSPRuntime();
  delete ctx;
  for (auto t : outputs) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_Range, Range_FT78_Fp64) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;

  std::vector<int> output_shape = {64};
  int num = output_shape[0];

  auto param = reinterpret_cast<RangeParameter *>(malloc(sizeof(RangeParameter)));
  ASSERT_NE(param, nullptr);
  std::memset(param, 0, sizeof(RangeParameter));
  param->start_ = -32;
  param->limit_ = 32;
  param->delta_ = 1;
  param->dtype_ = static_cast<int>(kNumberTypeFloat64);

  auto output = new lite::Tensor(kNumberTypeFloat64, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs.push_back(output);
  std::fill_n(reinterpret_cast<double *>(output->MutableData()), num, 0.0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat64, NHWC, schema::PrimitiveType_Range};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);

  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel->Run(), lite::RET_OK);

  std::vector<double> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = static_cast<double>(i - 32);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<double *>(outputs[0]->MutableData()), correct.data(), num, 1e-6));

  UninitDSPRuntime();
  delete ctx;
  for (auto t : outputs) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_Range, Range_FT78_Int32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;

  std::vector<int> output_shape = {100};
  int num = output_shape[0];

  auto param = reinterpret_cast<RangeParameter *>(malloc(sizeof(RangeParameter)));
  ASSERT_NE(param, nullptr);
  std::memset(param, 0, sizeof(RangeParameter));
  param->start_ = 0;
  param->limit_ = 100;
  param->delta_ = 1;
  param->dtype_ = static_cast<int>(kNumberTypeInt32);

  auto output = new lite::Tensor(kNumberTypeInt32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs.push_back(output);
  std::fill_n(reinterpret_cast<int32_t *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_Range};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);

  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel->Run(), lite::RET_OK);

  std::vector<int32_t> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = i;
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int32_t *>(outputs[0]->MutableData()), correct.data(), num));

  UninitDSPRuntime();
  delete ctx;
  for (auto t : outputs) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_Range, Range_FT78_Int16) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;

  std::vector<int> output_shape = {50};
  int num = output_shape[0];

  auto param = reinterpret_cast<RangeParameter *>(malloc(sizeof(RangeParameter)));
  ASSERT_NE(param, nullptr);
  std::memset(param, 0, sizeof(RangeParameter));
  param->start_ = -3;
  param->limit_ = 97;
  param->delta_ = 2;
  param->dtype_ = static_cast<int>(kNumberTypeInt16);

  auto output = new lite::Tensor(kNumberTypeInt16, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs.push_back(output);
  std::fill_n(reinterpret_cast<int16_t *>(output->MutableData()), num, static_cast<int16_t>(0));

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt16, NHWC, schema::PrimitiveType_Range};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);

  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel->Run(), lite::RET_OK);

  std::vector<int16_t> correct(num);
  int16_t value = -3;
  for (int i = 0; i < num; ++i) {
    correct[i] = value;
    value = static_cast<int16_t>(value + 2);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int16_t *>(outputs[0]->MutableData()), correct.data(), num));

  UninitDSPRuntime();
  delete ctx;
  for (auto t : outputs) {
    delete t;
  }
  delete kernel;
}

TEST_F(TestDSP_Range, Range_FT78_Int8) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;

  std::vector<int> output_shape = {40};
  int num = output_shape[0];

  auto param = reinterpret_cast<RangeParameter *>(malloc(sizeof(RangeParameter)));
  ASSERT_NE(param, nullptr);
  std::memset(param, 0, sizeof(RangeParameter));
  param->start_ = -20;
  param->limit_ = 20;
  param->delta_ = 1;
  param->dtype_ = static_cast<int>(kNumberTypeInt8);

  auto output = new lite::Tensor(kNumberTypeInt8, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs.push_back(output);
  std::fill_n(reinterpret_cast<int8_t *>(output->MutableData()), num, static_cast<int8_t>(0));

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt8, NHWC, schema::PrimitiveType_Range};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);

  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel->Run(), lite::RET_OK);

  std::vector<int8_t> correct(num);
  for (int i = 0; i < num; ++i) {
    correct[i] = static_cast<int8_t>(i - 20);
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int8_t *>(outputs[0]->MutableData()), correct.data(), num));

  UninitDSPRuntime();
  delete ctx;
  for (auto t : outputs) {
    delete t;
  }
  delete kernel;
}
#endif

}  // namespace mindspore::lite::dsp::test
