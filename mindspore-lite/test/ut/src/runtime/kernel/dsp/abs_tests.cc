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
#include <cstring>
#include <vector>
#include "ut/src/runtime/kernel/dsp/dsp_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel_registry.h"

namespace mindspore::lite::dsp::test {

class TestDSP_Abs : public DSPCommonTest {};

#ifdef SUPPORT_FT04
TEST_F(TestDSP_Abs, Abs_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = new OpParameter();
  param->type_ = static_cast<int>(schema::PrimitiveType_Abs);
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);

  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<float *>(input->MutableData()), num, -2.5f);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Abs};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> correct(num, 2.5f);
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(),
                                 outputs_[0]->ElementsNum()));
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

TEST_F(TestDSP_Abs, Abs_Int32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = new OpParameter();
  param->type_ = static_cast<int>(schema::PrimitiveType_Abs);
  auto input = new lite::Tensor(kNumberTypeInt32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);

  auto output = new lite::Tensor(kNumberTypeInt32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<int32_t *>(input->MutableData()), num, -7);
  std::fill_n(reinterpret_cast<int32_t *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_Abs};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int32_t> correct(num, 7);
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int32_t *>(outputs_[0]->MutableData()), correct.data(),
                                 outputs_[0]->ElementsNum()));
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

TEST_F(TestDSP_Abs, Abs_Int16) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = new OpParameter();
  param->type_ = static_cast<int>(schema::PrimitiveType_Abs);
  auto input = new lite::Tensor(kNumberTypeInt16, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);

  auto output = new lite::Tensor(kNumberTypeInt16, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<int16_t *>(input->MutableData()), num, static_cast<int16_t>(-1000));
  std::fill_n(reinterpret_cast<int16_t *>(output->MutableData()), num, static_cast<int16_t>(0));

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt16, NHWC, schema::PrimitiveType_Abs};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int16_t> correct(num, 1000);
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int16_t *>(outputs_[0]->MutableData()), correct.data(),
                                 outputs_[0]->ElementsNum()));
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

TEST_F(TestDSP_Abs, Abs_Cplx64) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = new OpParameter();
  param->type_ = static_cast<int>(schema::PrimitiveType_Abs);
  auto input = new lite::Tensor(kNumberTypeComplex64, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);

  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<float *>(input->MutableData()), num * 2, 1);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeComplex64, NHWC, schema::PrimitiveType_Abs};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> correct(num, 1.4142135);
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(),
                                 outputs_[0]->ElementsNum(), 1e-4));
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

TEST_F(TestDSP_Abs, Abs_Fp16) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = new OpParameter();
  param->type_ = static_cast<int>(schema::PrimitiveType_Abs);
  auto input = new lite::Tensor(kNumberTypeFloat16, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);

  auto output = new lite::Tensor(kNumberTypeFloat16, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  for (int i = 0; i < num; ++i) {
    reinterpret_cast<uint16_t *>(input->MutableData())[i] = Fp32ToFp16(-3.5f);
  }
  std::memset(output->MutableData(), 0, static_cast<size_t>(num) * sizeof(uint16_t));

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_Abs};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> correct(num, 3.5f);
  std::vector<float> actual(num);
  auto out_fp16 = reinterpret_cast<uint16_t *>(outputs_[0]->MutableData());
  for (int i = 0; i < num; ++i) {
    actual[i] = Fp16ToFp32(out_fp16[i]);
  }
  ASSERT_EQ(0, CompareOutputData(actual.data(), correct.data(), num, 1e-4));
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
#endif
}  // namespace mindspore::lite::dsp::test
