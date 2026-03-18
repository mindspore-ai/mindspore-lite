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

#include <iostream>
#include <memory>
#include <vector>
#include "ut/src/runtime/kernel/dsp/dsp_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "include/api/model.h"
#include "nnacl_c/exp_parameter.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel/dsp/dsp_subgraph.h"
#include "src/litert/kernel_registry.h"

namespace mindspore::lite::dsp::test {

constexpr int kTestArraySize = 10000;   // 100 * 100
constexpr int kTestArraySize2 = 20000;  // 100 * 100 * 2

class TestDSP_Exp : public DSPCommonTest {};

TEST_F(TestDSP_Exp, Exp_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = new ExpParameter();
  param->base_ = -1;
  param->scale_ = 1;
  param->shift_ = 0;
  auto input = new lite::Tensor(kNumberTypeFloat32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);

  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<float *>(input->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_ExpFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> correct(num, 2.7182798);
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

TEST_F(TestDSP_Exp, Exp_Int32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = new ExpParameter();
  param->base_ = -1;
  param->scale_ = 1;
  param->shift_ = 0;
  auto input = new lite::Tensor(kNumberTypeInt32, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);

  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<int32_t *>(input->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_ExpFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> correct(num, 2.7182798);
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

TEST_F(TestDSP_Exp, Exp_Cplx64) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = new ExpParameter();
  param->base_ = -1;
  param->scale_ = 1;
  param->shift_ = 0;
  auto input = new lite::Tensor(kNumberTypeComplex64, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);

  auto output = new lite::Tensor(kNumberTypeComplex64, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<float *>(input->MutableData()), num * 2, 1);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num * 2, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeComplex64, NHWC, schema::PrimitiveType_ExpFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> correct(kTestArraySize2);
  for (int i = 0; i < kTestArraySize2; i = i + 2) {
    correct[i] = 1.4686939;
    correct[i + 1] = 2.2873552;
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(),
                                 outputs_[0]->ElementsNum() * 2));
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

#ifdef SUPPORT_FT78
TEST_F(TestDSP_Exp, Exp_Fp64) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = new ExpParameter();
  param->base_ = -1;
  param->scale_ = 1;
  param->shift_ = 0;
  auto input = new lite::Tensor(kNumberTypeFloat64, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);

  auto output = new lite::Tensor(kNumberTypeFloat64, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<double *>(input->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<double *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat64, NHWC, schema::PrimitiveType_ExpFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<double> correct(kTestArraySize, 2.7182798);
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<double *>(outputs_[0]->MutableData()), correct.data(),
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

TEST_F(TestDSP_Exp, Exp_Int8) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = new ExpParameter();
  param->base_ = -1;
  param->scale_ = 1;
  param->shift_ = 0;
  auto input = new lite::Tensor(kNumberTypeInt8, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);

  auto output = new lite::Tensor(kNumberTypeInt8, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<int8_t *>(input->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<int8_t *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt8, NHWC, schema::PrimitiveType_ExpFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int8_t> correct(kTestArraySize, 2);
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int8_t *>(outputs_[0]->MutableData()), correct.data(),
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

TEST_F(TestDSP_Exp, Exp_Cplx128) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input_shape[0] * input_shape[1];

  auto param = new ExpParameter();
  param->base_ = -1;
  param->scale_ = 1;
  param->shift_ = 0;
  auto input = new lite::Tensor(kNumberTypeComplex128, input_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input->MallocData(allocator_);
  inputs_.push_back(input);

  auto output = new lite::Tensor(kNumberTypeComplex128, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<double *>(input->MutableData()), num * 2, 1);
  std::fill_n(reinterpret_cast<double *>(output->MutableData()), num * 2, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeComplex128, NHWC, schema::PrimitiveType_ExpFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<double> correct(kTestArraySize2);
  for (int i = 0; i < kTestArraySize2; i = i + 2) {
    correct[i] = 1.4686939;
    correct[i + 1] = 2.2873552;
  }
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<double *>(outputs_[0]->MutableData()), correct.data(),
                                 outputs_[0]->ElementsNum() * 2));
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
