/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "schema/inner/model_generated.h"
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "src/litert/kernel/cpu/base/quant_dtype_cast.h"
#include "nnacl_c/int8/quant_dtype_cast_int8.h"
#include "src/litert/kernel_registry.h"
#include "src/executor/kernel_exec.h"

namespace mindspore {

class QuantDTypeCastTestFp32 : public mindspore::CommonTest {
 public:
  QuantDTypeCastTestFp32() {}
};

TEST_F(QuantDTypeCastTestFp32, QuantDTypeCastTest1) {
  const lite::LiteQuantParam quant_arg{0.21176, 5};
  // LiteKernel's destructor free()s op_parameter_, so it must be heap-allocated
  auto param = new (std::nothrow) QuantDTypeCastParameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "New param fails.";
    return;
  }
  param->srcT = kNumberTypeInt8;
  param->dstT = kNumberTypeFloat32;
  param->op_parameter_.type_ = schema::PrimitiveType_QuantDTypeCast;

  std::vector<int8_t> input = {10, 14, 29, 33, 52, 99, 19, 43, 90, 52, 19, 24, 57, 127, 76, 123};
  std::vector<int> in_shape = {1, 4, 4, 1};
  lite::Tensor input_tensor;
  input_tensor.set_data(input.data());
  input_tensor.set_shape(in_shape);
  input_tensor.set_data_type(kNumberTypeInt8);
  input_tensor.set_format(mindspore::NHWC);

  input_tensor.AddQuantParam(quant_arg);
  std::vector<lite::Tensor *> inputs_tensor;
  inputs_tensor.emplace_back(&input_tensor);

  const int out_size = 16;
  // dequant spec: real = (q - zero_point) * scale; the original expectation was computed with
  // (q + zero_point) * scale, a zero-point sign error (Test2 quantizes with round(real/scale)+zp)
  float expect_out[16] = {1.0588,  1.90584, 5.08224, 5.92928, 9.95272,  19.90544, 2.96464,  8.04688,
                          17.9996, 9.95272, 2.96464, 4.02344, 11.01152, 25.83472, 15.03496, 24.98768};
  std::vector<float> output(16);
  std::vector<int> out_shape = {1, 4, 4, 1};
  lite::Tensor output_tensor;
  output_tensor.set_data(output.data());
  output_tensor.set_shape(out_shape);
  output_tensor.set_data_type(kNumberTypeFloat32);
  // output_tensor.SetFormat(schema::Format_NHWC);
  std::vector<lite::Tensor *> outputs_tensor;
  outputs_tensor.emplace_back(&output_tensor);

  lite::InnerContext ctx;
  ctx.thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx.Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_QuantDTypeCast};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(param), &ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  for (int i = 0; i < out_size; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output.data(), expect_out, out_size, 0.000001));
  input_tensor.set_data(nullptr);
  output_tensor.set_data(nullptr);
  delete kernel;
}

TEST_F(QuantDTypeCastTestFp32, QuantDTypeCastTest2) {
  const lite::LiteQuantParam quant_arg = {0.3515625, -57};
  // LiteKernel's destructor free()s op_parameter_, so it must be heap-allocated
  auto param = new (std::nothrow) QuantDTypeCastParameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "New param fails.";
    return;
  }
  param->op_parameter_.type_ = schema::PrimitiveType_QuantDTypeCast;
  param->dstT = kNumberTypeInt8;
  param->srcT = kNumberTypeFloat32;
  std::vector<float> input = {1, 2, 5, 6, 10, -20, 3, 8, 18, 10, 3, 4, 11, 16, 15, 25};
  std::vector<int> in_shape = {1, 4, 4, 1};
  lite::Tensor input_tensor;
  input_tensor.set_data(input.data());
  input_tensor.set_shape(in_shape);
  // input_tensor.SetFormat(schema::Format_NHWC);
  input_tensor.set_data_type(kNumberTypeFloat32);
  input_tensor.AddQuantParam(quant_arg);
  std::vector<lite::Tensor *> inputs_tensor;
  inputs_tensor.emplace_back(&input_tensor);

  const int out_size = 16;
  int8_t expect_out[16] = {-54, -51, -43, -40, -29, -114, -48, -34, -6, -29, -48, -46, -26, -11, -14, 14};
  std::vector<int8_t> output(16);
  std::vector<int> out_shape = {1, 4, 4, 1};
  lite::Tensor output_tensor;
  output_tensor.set_data(output.data());
  output_tensor.set_shape(out_shape);
  output_tensor.set_format(mindspore::NHWC);
  output_tensor.set_data_type(kNumberTypeInt8);
  // Prepare() rejects an int8 output tensor without quant params; inited=false keeps the
  // kernel extracting the quant params from the input tensor
  output_tensor.AddQuantParam(quant_arg);
  std::vector<lite::Tensor *> outputs_tensor;
  outputs_tensor.emplace_back(&output_tensor);

  lite::InnerContext ctx;
  ctx.thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx.Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_QuantDTypeCast};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(param), &ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  for (int i = 0; i < out_size; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output.data(), expect_out, out_size, 0.000001));
  input_tensor.set_data(nullptr);
  output_tensor.set_data(nullptr);
  delete kernel;
}
}  // namespace mindspore
