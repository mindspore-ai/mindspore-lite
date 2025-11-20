
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
#include "schema/inner/model_generated.h"
#include "src/litert/kernel_registry.h"

namespace mindspore::lite::dsp::test {

class TestDSP_Assert : public DSPCommonTest {};

TEST_F(TestDSP_Assert, Assert_Int32_True) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  // input: bool scalar true
  auto t_cond = new lite::Tensor(kNumberTypeInt32, {1}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_cond->MallocData(allocator_);
  *reinterpret_cast<int32_t *>(t_cond->MutableData()) = 1;
  inputs_.push_back(t_cond);

  // output: bool scalar
  auto t_out = new lite::Tensor(kNumberTypeInt32, {1}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_out->MallocData(allocator_);
  outputs_.push_back(t_out);
  *reinterpret_cast<int32_t *>(t_out->MutableData()) = 0;

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_Assert};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto *param = new OpParameter();
  param->type_ = static_cast<int>(schema::PrimitiveType_Assert);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int32_t res = *reinterpret_cast<int32_t *>(outputs_[0]->MutableData());
  ASSERT_EQ(1, res);

  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  delete kernel;
}

TEST_F(TestDSP_Assert, Assert_Int32_False) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  // input: bool scalar false
  auto t_cond = new lite::Tensor(kNumberTypeInt32, {1}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_cond->MallocData(allocator_);
  *reinterpret_cast<int32_t *>(t_cond->MutableData()) = 0;
  inputs_.push_back(t_cond);

  auto t_out = new lite::Tensor(kNumberTypeInt32, {1}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_out->MallocData(allocator_);
  outputs_.push_back(t_out);
  *reinterpret_cast<int32_t *>(t_out->MutableData()) = 1;

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_Assert};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto *param = new OpParameter();
  param->type_ = static_cast<int>(schema::PrimitiveType_Assert);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  int32_t res = *reinterpret_cast<int32_t *>(outputs_[0]->MutableData());
  ASSERT_EQ(0, res);

  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  delete kernel;
}

}  // namespace mindspore::lite::dsp::test
