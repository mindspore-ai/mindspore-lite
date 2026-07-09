/**
 * Copyright 2020-2026 Huawei Technologies Co., Ltd
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
#include "schema/inner/model_generated.h"
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "nnacl_c/mul_parameter.h"
#include "src/litert/kernel_registry.h"
#include "src/executor/kernel_exec.h"
#include "src/tensor.h"
#include "nnacl_c/arithmetic_parameter.h"

namespace mindspore {

class TestMulInt8 : public mindspore::CommonTest {
 public:
  TestMulInt8() {}
};

TEST_F(TestMulInt8, Mul_quant0) {
  std::vector<int8_t> input1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> shape1 = {2, 3, 2};
  std::vector<int8_t> input2 = {1, 2, 3, 4};
  std::vector<int> shape2 = {2, 1, 2};
  std::vector<int8_t *> input(2, nullptr);
  input[0] = input1.data();
  input[1] = input2.data();

  int8_t output[12];
  std::vector<int> output_shape = {2, 3, 2};

  lite::LiteQuantParam input_quant_arg;
  input_quant_arg.scale = 1.0;
  input_quant_arg.zeroPoint = 0;
  lite::LiteQuantParam output_quant_arg;
  output_quant_arg.scale = 1.0;
  output_quant_arg.zeroPoint = 0;

  lite::Tensor *input_tensor1 = new lite::Tensor;
  TypeId tid_int8 = kNumberTypeInt8;
  input_tensor1->set_data(input1.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->AddQuantParam(input_quant_arg);
  input_tensor1->set_data_type(tid_int8);

  lite::Tensor *input_tensor2 = new lite::Tensor;
  input_tensor2->set_data(input2.data());
  input_tensor2->set_shape(shape2);
  input_tensor2->AddQuantParam(input_quant_arg);
  input_tensor2->set_data_type(tid_int8);

  std::vector<lite::Tensor *> inputs_tensor(2);
  inputs_tensor[0] = input_tensor1;
  inputs_tensor[1] = input_tensor2;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  ArithmeticParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_MulFusion;
  op_param.op_parameter_.thread_num_ = 1;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_MulFusion};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int8_t> except_result = {1, 4, 3, 8, 5, 12, 21, 32, 27, 40, 33, 48};
  PrintData("output data", output, input1.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), input1.size(), 0.000001));
  input_tensor1->set_data(nullptr);
  input_tensor2->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete input_tensor2;
  delete output0_tensor;
  delete ctx;
  kernel->set_parameter(nullptr);
  delete kernel;
}

TEST_F(TestMulInt8, Mul_quant0_thread0) {
  std::vector<int8_t> input1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  std::vector<int> shape1 = {2, 3, 3};
  std::vector<int8_t> input2 = {1, 1, 1, 1, 1, 1};
  std::vector<int> shape2 = {2, 1, 3};
  std::vector<int8_t *> input(2, nullptr);
  input[0] = input1.data();
  input[1] = input2.data();

  int8_t output[18];
  std::vector<int> output_shape = {2, 3, 3};

  lite::LiteQuantParam input_quant_arg;
  input_quant_arg.scale = 1.0;
  input_quant_arg.zeroPoint = 0;
  lite::LiteQuantParam output_quant_arg;
  output_quant_arg.scale = 1.0;
  output_quant_arg.zeroPoint = 0;

  lite::Tensor *input_tensor1 = new lite::Tensor;
  TypeId tid_int8 = kNumberTypeInt8;
  input_tensor1->set_data(input1.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->AddQuantParam(input_quant_arg);
  input_tensor1->set_data_type(tid_int8);

  lite::Tensor *input_tensor2 = new lite::Tensor;
  input_tensor2->set_data(input2.data());
  input_tensor2->set_shape(shape2);
  input_tensor2->AddQuantParam(input_quant_arg);
  input_tensor2->set_data_type(tid_int8);

  std::vector<lite::Tensor *> inputs_tensor(2);
  inputs_tensor[0] = input_tensor1;
  inputs_tensor[1] = input_tensor2;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  ArithmeticParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_MulFusion;
  op_param.op_parameter_.thread_num_ = 1;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_MulFusion};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int8_t> except_result = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  PrintData("output data", output, input1.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), input1.size(), 0.000001));
  input_tensor1->set_data(nullptr);
  input_tensor2->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete input_tensor2;
  delete output0_tensor;
  delete ctx;
  kernel->set_parameter(nullptr);
  delete kernel;
}

TEST_F(TestMulInt8, Mul_quant1) {
  std::vector<int8_t> input1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> shape1 = {2, 3, 2};
  std::vector<int8_t> input2 = {1, 2, 3, 4};
  std::vector<int> shape2 = {2, 1, 2};
  std::vector<int8_t *> input(2, nullptr);
  input[0] = input1.data();
  input[1] = input2.data();

  int8_t output[12];
  std::vector<int> output_shape = {2, 3, 2};

  lite::LiteQuantParam input_quant_arg;
  input_quant_arg.scale = 1.0;
  input_quant_arg.zeroPoint = 0;
  lite::LiteQuantParam output_quant_arg;
  output_quant_arg.scale = 2.0;
  output_quant_arg.zeroPoint = 0;

  lite::Tensor *input_tensor1 = new lite::Tensor;
  TypeId tid_int8 = kNumberTypeInt8;
  input_tensor1->set_data(input1.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->AddQuantParam(input_quant_arg);
  input_tensor1->set_data_type(tid_int8);

  lite::Tensor *input_tensor2 = new lite::Tensor;
  input_tensor2->set_data(input2.data());
  input_tensor2->set_shape(shape2);
  input_tensor2->AddQuantParam(input_quant_arg);
  input_tensor2->set_data_type(tid_int8);

  std::vector<lite::Tensor *> inputs_tensor(2);
  inputs_tensor[0] = input_tensor1;
  inputs_tensor[1] = input_tensor2;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  ArithmeticParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_MulFusion;
  op_param.op_parameter_.thread_num_ = 1;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_MulFusion};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int8_t> except_result = {1, 2, 2, 4, 3, 6, 11, 16, 14, 20, 17, 24};
  PrintData("output data", output, input1.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), input1.size(), 0.000001));
  input_tensor1->set_data(nullptr);
  input_tensor2->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete input_tensor2;
  delete output0_tensor;
  delete ctx;
  kernel->set_parameter(nullptr);
  delete kernel;
}

TEST_F(TestMulInt8, Mul_quant1_thread1) {
  std::vector<int8_t> input1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> shape1 = {2, 3, 2};
  std::vector<int8_t> input2 = {1, 2, 3, 4};
  std::vector<int> shape2 = {2, 1, 2};
  std::vector<int8_t *> input(2, nullptr);
  input[0] = input1.data();
  input[1] = input2.data();

  int8_t output[12];
  std::vector<int> output_shape = {2, 3, 2};

  lite::LiteQuantParam input_quant_arg;
  input_quant_arg.scale = 1.0;
  input_quant_arg.zeroPoint = 0;
  lite::LiteQuantParam output_quant_arg;
  output_quant_arg.scale = 2.0;
  output_quant_arg.zeroPoint = 0;

  lite::Tensor *input_tensor1 = new lite::Tensor;
  TypeId tid_int8 = kNumberTypeInt8;
  input_tensor1->set_data(input1.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->AddQuantParam(input_quant_arg);
  input_tensor1->set_data_type(tid_int8);

  lite::Tensor *input_tensor2 = new lite::Tensor;
  input_tensor2->set_data(input2.data());
  input_tensor2->set_shape(shape2);
  input_tensor2->AddQuantParam(input_quant_arg);
  input_tensor2->set_data_type(tid_int8);

  std::vector<lite::Tensor *> inputs_tensor(2);
  inputs_tensor[0] = input_tensor1;
  inputs_tensor[1] = input_tensor2;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  ArithmeticParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_MulFusion;
  op_param.op_parameter_.thread_num_ = 3;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_MulFusion};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int8_t> except_result = {1, 2, 2, 4, 3, 6, 11, 16, 14, 20, 17, 24};
  PrintData("output data", output, input1.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), input1.size(), 0.000001));
  input_tensor1->set_data(nullptr);
  input_tensor2->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete input_tensor2;
  delete output0_tensor;
  delete ctx;
  kernel->set_parameter(nullptr);
  delete kernel;
}

TEST_F(TestMulInt8, test) {
  std::vector<int8_t> input1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> shape1 = {2, 2, 3};
  std::vector<int8_t> input2 = {1, 2, 3, 4, 5, 6};
  std::vector<int> shape2 = {2, 3};
  std::vector<int8_t *> input(2, nullptr);
  input[0] = input1.data();
  input[1] = input2.data();

  int8_t output[12];
  std::vector<int> output_shape = {2, 2, 3};

  lite::LiteQuantParam input_quant_arg;
  input_quant_arg.scale = 1.0;
  input_quant_arg.zeroPoint = 0;
  lite::LiteQuantParam output_quant_arg;
  output_quant_arg.scale = 1.0;
  output_quant_arg.zeroPoint = 0;

  lite::Tensor *input_tensor1 = new lite::Tensor;
  TypeId tid_int8 = kNumberTypeInt8;
  input_tensor1->set_data(input1.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->AddQuantParam(input_quant_arg);
  input_tensor1->set_data_type(tid_int8);

  lite::Tensor *input_tensor2 = new lite::Tensor;
  input_tensor2->set_data(input2.data());
  input_tensor2->set_shape(shape2);
  input_tensor2->AddQuantParam(input_quant_arg);
  input_tensor2->set_data_type(tid_int8);

  std::vector<lite::Tensor *> inputs_tensor(2);
  inputs_tensor[0] = input_tensor1;
  inputs_tensor[1] = input_tensor2;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  ArithmeticParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_MulFusion;
  op_param.op_parameter_.thread_num_ = 2;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_MulFusion};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int8_t> except_result = {1, 4, 9, 16, 25, 36, 7, 16, 27, 40, 55, 72};
  PrintData("output data", output, input1.size());
  ASSERT_EQ(0, CompareOutputData(output, except_result.data(), input1.size(), 0.000001));
  input_tensor1->set_data(nullptr);
  input_tensor2->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete input_tensor2;
  delete output0_tensor;
  delete ctx;
  kernel->set_parameter(nullptr);
  delete kernel;
}

// NCHW [N,C,H,W] fast-broadcast regression: [2,3,4,5] * [2,3,1,1].
// This shape enters the fast broadcast path only under NCHW (C at index 1 matches,
// H=W=1 on the source side); under NHWC it correctly falls back to tile. Guards the
// FastMulNCHW path and the swap_zp == input1_hw_broadcast_ == true direction.
TEST_F(TestMulInt8, Mul_quant_nchw_hw_broadcast_in1) {
  const int N = 2, C = 3, H = 4, W = 5;
  const int hw = H * W;  // 20, elements per (n,c) block
  const int nc = N * C;  // 6, number of (n,c) scalar rows
  // input1 [N,C,1,1]: one scalar per (n,c); input0 [N,C,H,W]: each (n,c) block = 1..20
  std::vector<int8_t> input1 = {1, 2, 3, 4, 5, 6};
  std::vector<int8_t> input0(N * C * H * W);
  for (size_t i = 0; i < input0.size(); ++i) {
    input0[i] = (i % hw) + 1;
  }
  // expect: block k = (1..20) * input1[k], all <= 120 so no int8 overflow
  std::vector<int8_t> expect(N * C * H * W);
  for (int k = 0; k < nc; ++k) {
    for (int j = 0; j < hw; ++j) {
      expect[k * hw + j] = (j + 1) * input1[k];
    }
  }
  int8_t output[N * C * H * W] = {0};

  std::vector<int> shape1 = {N, C, H, W};
  std::vector<int> shape2 = {N, C, 1, 1};
  std::vector<int> output_shape = {N, C, H, W};

  lite::LiteQuantParam input_quant_arg;
  input_quant_arg.scale = 1.0;
  input_quant_arg.zeroPoint = 0;
  lite::LiteQuantParam output_quant_arg;
  output_quant_arg.scale = 1.0;
  output_quant_arg.zeroPoint = 0;

  lite::Tensor *input_tensor1 = new lite::Tensor;
  TypeId tid_int8 = kNumberTypeInt8;
  input_tensor1->set_data(input0.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->AddQuantParam(input_quant_arg);
  input_tensor1->set_data_type(tid_int8);
  input_tensor1->set_format(NCHW);

  lite::Tensor *input_tensor2 = new lite::Tensor;
  input_tensor2->set_data(input1.data());
  input_tensor2->set_shape(shape2);
  input_tensor2->AddQuantParam(input_quant_arg);
  input_tensor2->set_data_type(tid_int8);
  input_tensor2->set_format(NCHW);

  std::vector<lite::Tensor *> inputs_tensor(2);
  inputs_tensor[0] = input_tensor1;
  inputs_tensor[1] = input_tensor2;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  output0_tensor->set_format(NCHW);
  outputs_tensor[0] = output0_tensor;

  ArithmeticParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_MulFusion;
  // Required: the kernel reads thread_count_ from op_parameter_->thread_num_;
  // leaving it unset feeds ParallelLaunch a garbage task count -> flaky all-zero output
  op_param.op_parameter_.thread_num_ = 1;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_MulFusion};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  PrintData("output data", output, N * C * H * W);
  ASSERT_EQ(0, CompareOutputData(output, expect.data(), N * C * H * W, 0.000001));
  input_tensor1->set_data(nullptr);
  input_tensor2->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete input_tensor2;
  delete output0_tensor;
  delete ctx;
  // op_param is a stack object: ~LiteKernel frees op_parameter_, so freeing a stack address aborts
  kernel->set_parameter(nullptr);
  delete kernel;
}

// Reverse direction: [2,3,1,1] * [2,3,4,5] under NCHW. Same expected output, but
// input0 is now the broadcast source (input1_hw_broadcast_ == false, swap_zp == false).
TEST_F(TestMulInt8, Mul_quant_nchw_hw_broadcast_in0) {
  const int N = 2, C = 3, H = 4, W = 5;
  const int hw = H * W;
  const int nc = N * C;
  std::vector<int8_t> input0 = {1, 2, 3, 4, 5, 6};  // [N,C,1,1]
  std::vector<int8_t> input1(N * C * H * W);        // [N,C,H,W]: each (n,c) block = 1..20
  for (size_t i = 0; i < input1.size(); ++i) {
    input1[i] = (i % hw) + 1;
  }
  std::vector<int8_t> expect(N * C * H * W);
  for (int k = 0; k < nc; ++k) {
    for (int j = 0; j < hw; ++j) {
      expect[k * hw + j] = (j + 1) * input0[k];
    }
  }
  int8_t output[N * C * H * W] = {0};

  std::vector<int> shape1 = {N, C, 1, 1};
  std::vector<int> shape2 = {N, C, H, W};
  std::vector<int> output_shape = {N, C, H, W};

  lite::LiteQuantParam input_quant_arg;
  input_quant_arg.scale = 1.0;
  input_quant_arg.zeroPoint = 0;
  lite::LiteQuantParam output_quant_arg;
  output_quant_arg.scale = 1.0;
  output_quant_arg.zeroPoint = 0;

  lite::Tensor *input_tensor1 = new lite::Tensor;
  TypeId tid_int8 = kNumberTypeInt8;
  input_tensor1->set_data(input0.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->AddQuantParam(input_quant_arg);
  input_tensor1->set_data_type(tid_int8);
  input_tensor1->set_format(NCHW);

  lite::Tensor *input_tensor2 = new lite::Tensor;
  input_tensor2->set_data(input1.data());
  input_tensor2->set_shape(shape2);
  input_tensor2->AddQuantParam(input_quant_arg);
  input_tensor2->set_data_type(tid_int8);
  input_tensor2->set_format(NCHW);

  std::vector<lite::Tensor *> inputs_tensor(2);
  inputs_tensor[0] = input_tensor1;
  inputs_tensor[1] = input_tensor2;

  std::vector<lite::Tensor *> outputs_tensor(1);
  lite::Tensor *output0_tensor = new lite::Tensor;
  output0_tensor->set_data(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  output0_tensor->set_format(NCHW);
  outputs_tensor[0] = output0_tensor;

  ArithmeticParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_MulFusion;
  op_param.op_parameter_.thread_num_ = 1;
  lite::InnerContext *ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_MulFusion};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  PrintData("output data", output, N * C * H * W);
  ASSERT_EQ(0, CompareOutputData(output, expect.data(), N * C * H * W, 0.000001));
  input_tensor1->set_data(nullptr);
  input_tensor2->set_data(nullptr);
  output0_tensor->set_data(nullptr);
  delete input_tensor1;
  delete input_tensor2;
  delete output0_tensor;
  delete ctx;
  kernel->set_parameter(nullptr);
  delete kernel;
}

}  // namespace mindspore
