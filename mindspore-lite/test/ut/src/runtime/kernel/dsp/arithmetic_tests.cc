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

#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include "ut/src/runtime/kernel/dsp/dsp_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "include/api/model.h"
#include "nnacl_c/arithmetic_parameter.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel/dsp/dsp_subgraph.h"
#include "src/litert/kernel_registry.h"
#include "ut/src/runtime/kernel/opencl/common.h"

namespace mindspore::lite::dsp::test {

constexpr int kTestArraySize = 10000;   // 100 * 100
constexpr int kTestArraySize2 = 20000;  // 100 * 100 * 2

class TestDSP_Arithmetic : public DSPCommonTest {};

namespace {
OpParameter *CreateParameter(schema::PrimitiveType type, const std::vector<int> &input0_shape,
                             const std::vector<int> &input1_shape,
                             schema::ActivationType act_type = schema::ActivationType_NO_ACTIVATION) {
  auto *param = opencl::test::CreateParameter<ArithmeticParameter>(type);
  int input0_size = std::accumulate(input0_shape.begin(), input0_shape.end(), 1, std::multiplies<>());
  int input1_size = std::accumulate(input1_shape.begin(), input1_shape.end(), 1, std::multiplies<>());
  if (input0_size != input1_size) {
    param->broadcasting_ = true;
  }
  param->activation_type_ = act_type;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestDSP_Arithmetic, Add_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_AddFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeFloat32, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeFloat32, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<float *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<float *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_AddFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> correct(num, 3);
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

TEST_F(TestDSP_Arithmetic, Add_Int16) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_AddFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeInt16, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeInt16, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeInt16, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<int16_t *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<int16_t *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<int16_t *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt16, NHWC, schema::PrimitiveType_AddFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int16_t> correct(kTestArraySize, 3);
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

TEST_F(TestDSP_Arithmetic, Add_Int32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_AddFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeInt32, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeInt32, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeInt32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<int32_t *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<int32_t *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<int32_t *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_AddFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int32_t> correct(kTestArraySize, 3);
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

TEST_F(TestDSP_Arithmetic, Add_Cplx64) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_AddFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeComplex64, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeComplex64, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeComplex64, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<float *>(input0->MutableData()), num * 2, 2);
  std::fill_n(reinterpret_cast<float *>(input1->MutableData()), num * 2, 1);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num * 2, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeComplex64, NHWC, schema::PrimitiveType_AddFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> correct(kTestArraySize2, 3);
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

TEST_F(TestDSP_Arithmetic, Sub_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_SubFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeFloat32, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeFloat32, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<float *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<float *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_SubFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> correct(num, 1);
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

TEST_F(TestDSP_Arithmetic, Sub_Int16) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_SubFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeInt16, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeInt16, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeInt16, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<int16_t *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<int16_t *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<int16_t *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt16, NHWC, schema::PrimitiveType_SubFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int16_t> correct(kTestArraySize, 1);
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

TEST_F(TestDSP_Arithmetic, Sub_Int32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_SubFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeInt32, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeInt32, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeInt32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<int32_t *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<int32_t *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<int32_t *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_SubFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int32_t> correct(kTestArraySize, 1);
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

TEST_F(TestDSP_Arithmetic, Sub_Cplx64) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_SubFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeComplex64, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeComplex64, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeComplex64, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<float *>(input0->MutableData()), num * 2, 2);
  std::fill_n(reinterpret_cast<float *>(input1->MutableData()), num * 2, 1);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num * 2, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeComplex64, NHWC, schema::PrimitiveType_SubFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> correct(kTestArraySize2, 1);
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

TEST_F(TestDSP_Arithmetic, Mul_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_MulFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeFloat32, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeFloat32, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<float *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<float *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_MulFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> correct(num, 2);
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

TEST_F(TestDSP_Arithmetic, Mul_Int16) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_MulFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeInt16, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeInt16, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeInt16, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<int16_t *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<int16_t *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<int16_t *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt16, NHWC, schema::PrimitiveType_MulFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int16_t> correct(kTestArraySize, 2);
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

TEST_F(TestDSP_Arithmetic, Mul_Int32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_MulFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeInt32, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeInt32, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeInt32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<int32_t *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<int32_t *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<int32_t *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_MulFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int32_t> correct(kTestArraySize, 2);
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

TEST_F(TestDSP_Arithmetic, Mul_Cplx64) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_MulFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeComplex64, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeComplex64, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeComplex64, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<float *>(input0->MutableData()), num * 2, 2);
  std::fill_n(reinterpret_cast<float *>(input1->MutableData()), num * 2, 1);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num * 2, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeComplex64, NHWC, schema::PrimitiveType_MulFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> correct(kTestArraySize2);
  for (int i = 0; i < kTestArraySize2; i += 2) {
    correct[i] = 0;
    correct[i + 1] = 4;
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

TEST_F(TestDSP_Arithmetic, Div_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_DivFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeFloat32, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeFloat32, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeFloat32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<float *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<float *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_DivFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> correct(num, 2);
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

TEST_F(TestDSP_Arithmetic, Div_Int16) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_DivFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeInt16, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeInt16, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeInt16, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<int16_t *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<int16_t *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<int16_t *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt16, NHWC, schema::PrimitiveType_DivFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int16_t> correct(kTestArraySize, 2);
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

TEST_F(TestDSP_Arithmetic, Div_Int32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_DivFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeInt32, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeInt32, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeInt32, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<int32_t *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<int32_t *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<int32_t *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_DivFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int32_t> correct(kTestArraySize, 2);
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

TEST_F(TestDSP_Arithmetic, Div_Cplx64) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_DivFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeComplex64, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeComplex64, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeComplex64, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<float *>(input0->MutableData()), num * 2, 2);
  std::fill_n(reinterpret_cast<float *>(input1->MutableData()), num * 2, 1);
  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num * 2, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeComplex64, NHWC, schema::PrimitiveType_DivFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<float> correct(kTestArraySize2);
  for (int i = 0; i < kTestArraySize2; i += 2) {
    correct[i] = 2;
    correct[i + 1] = 0;
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
TEST_F(TestDSP_Arithmetic, Add_Fp64) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_AddFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeFloat64, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeFloat64, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeFloat64, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<double *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<double *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<double *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat64, NHWC, schema::PrimitiveType_AddFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<double> correct(kTestArraySize, 3);
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

TEST_F(TestDSP_Arithmetic, Add_Int8) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_AddFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeInt8, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeInt8, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeInt8, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<int8_t *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<int8_t *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<int8_t *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt8, NHWC, schema::PrimitiveType_AddFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int8_t> correct(kTestArraySize, 3);
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

TEST_F(TestDSP_Arithmetic, Add_Cplx128) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_AddFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeComplex128, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeComplex128, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeComplex128, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<double *>(input0->MutableData()), num * 2, 2);
  std::fill_n(reinterpret_cast<double *>(input1->MutableData()), num * 2, 1);
  std::fill_n(reinterpret_cast<double *>(output->MutableData()), num * 2, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeComplex128, NHWC, schema::PrimitiveType_AddFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<double> correct(kTestArraySize2, 3);
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

TEST_F(TestDSP_Arithmetic, Sub_Fp64) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_SubFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeFloat64, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeFloat64, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeFloat64, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<double *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<double *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<double *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat64, NHWC, schema::PrimitiveType_SubFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<double> correct(kTestArraySize, 1);
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

TEST_F(TestDSP_Arithmetic, Sub_Int8) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_SubFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeInt8, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeInt8, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeInt8, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<int8_t *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<int8_t *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<int8_t *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt8, NHWC, schema::PrimitiveType_SubFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int8_t> correct(kTestArraySize, 1);
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

TEST_F(TestDSP_Arithmetic, Sub_Cplx128) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_SubFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeComplex128, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeComplex128, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeComplex128, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<double *>(input0->MutableData()), num * 2, 2);
  std::fill_n(reinterpret_cast<double *>(input1->MutableData()), num * 2, 1);
  std::fill_n(reinterpret_cast<double *>(output->MutableData()), num * 2, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeComplex128, NHWC, schema::PrimitiveType_SubFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<double> correct(kTestArraySize2, 1);
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

TEST_F(TestDSP_Arithmetic, Mul_Fp64) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_MulFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeFloat64, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeFloat64, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeFloat64, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<double *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<double *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<double *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat64, NHWC, schema::PrimitiveType_MulFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<double> correct(kTestArraySize, 2);
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

TEST_F(TestDSP_Arithmetic, Mul_Int8) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_MulFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeInt8, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeInt8, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeInt8, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<int8_t *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<int8_t *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<int8_t *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt8, NHWC, schema::PrimitiveType_MulFusion};
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

TEST_F(TestDSP_Arithmetic, Mul_Cplx128) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_MulFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeComplex128, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeComplex128, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeComplex128, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<double *>(input0->MutableData()), num * 2, 2);
  std::fill_n(reinterpret_cast<double *>(input1->MutableData()), num * 2, 1);
  std::fill_n(reinterpret_cast<double *>(output->MutableData()), num * 2, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeComplex128, NHWC, schema::PrimitiveType_MulFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<double> correct(kTestArraySize2);
  for (int i = 0; i < kTestArraySize2; i += 2) {
    correct[i] = 0;
    correct[i + 1] = 4;
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

TEST_F(TestDSP_Arithmetic, Div_Fp64) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_DivFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeFloat64, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeFloat64, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeFloat64, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<double *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<double *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<double *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat64, NHWC, schema::PrimitiveType_DivFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<double> correct(kTestArraySize, 2);
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

TEST_F(TestDSP_Arithmetic, Div_Int8) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_DivFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeInt8, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeInt8, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeInt8, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<int8_t *>(input0->MutableData()), num, 2);
  std::fill_n(reinterpret_cast<int8_t *>(input1->MutableData()), num, 1);
  std::fill_n(reinterpret_cast<int8_t *>(output->MutableData()), num, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt8, NHWC, schema::PrimitiveType_DivFusion};
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

TEST_F(TestDSP_Arithmetic, Div_Cplx128) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> input0_shape = {100, 100};
  std::vector<int> input1_shape = {100, 100};
  std::vector<int> output_shape = {100, 100};
  int num = input0_shape[0] * input0_shape[1];

  auto *param = CreateParameter(schema::PrimitiveType_DivFusion, input0_shape, input1_shape);

  auto input0 = new lite::Tensor(kNumberTypeComplex128, input0_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input0->MallocData(allocator_);
  inputs_.push_back(input0);

  auto input1 = new lite::Tensor(kNumberTypeComplex128, input1_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  input1->MallocData(allocator_);
  inputs_.push_back(input1);

  auto output = new lite::Tensor(kNumberTypeComplex128, output_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<double *>(input0->MutableData()), num * 2, 2);
  std::fill_n(reinterpret_cast<double *>(input1->MutableData()), num * 2, 1);
  std::fill_n(reinterpret_cast<double *>(output->MutableData()), num * 2, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeComplex128, NHWC, schema::PrimitiveType_DivFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<double> correct(kTestArraySize2);
  for (int i = 0; i < kTestArraySize2; i += 2) {
    correct[i] = 2;
    correct[i + 1] = 0;
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
