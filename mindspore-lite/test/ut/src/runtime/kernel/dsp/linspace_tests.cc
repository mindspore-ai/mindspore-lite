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
#include <cmath>
#include <cstdint>
#include <cstring>
#include "ut/src/runtime/kernel/dsp/dsp_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "include/api/model.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel/dsp/dsp_subgraph.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/cpu/nnacl_c/op_base.h"

namespace mindspore::lite::dsp::test {

class TestDSP_Linspace : public DSPCommonTest {};

TEST_F(TestDSP_Linspace, Linspace_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  const int num = 100;
  const float start = 0.0f;
  const float end = 99.0f;  // inclusive

  // inputs: start (fp32), end (fp32), num (i32)
  auto in_start = new lite::Tensor(kNumberTypeFloat32, {1}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  in_start->MallocData(allocator_);
  *reinterpret_cast<float *>(in_start->MutableData()) = start;
  inputs_.push_back(in_start);

  auto in_end = new lite::Tensor(kNumberTypeFloat32, {1}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  in_end->MallocData(allocator_);
  *reinterpret_cast<float *>(in_end->MutableData()) = end;
  inputs_.push_back(in_end);

  auto in_num = new lite::Tensor(kNumberTypeInt32, {1}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  in_num->MallocData(allocator_);
  *reinterpret_cast<int32_t *>(in_num->MutableData()) = num;
  inputs_.push_back(in_num);

  // output: fp32 length=num
  auto output = new lite::Tensor(kNumberTypeFloat32, {num}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::fill_n(reinterpret_cast<float *>(output->MutableData()), num, 0.0f);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_LinSpace};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  // Minimal OpParameter for kernel construction
  auto *param = new OpParameter();
  param->type_ = static_cast<int>(schema::PrimitiveType_LinSpace);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  // expected: evenly spaced from start to end inclusive
  std::vector<float> correct(num);
  if (num == 1) {
    correct[0] = start;
  } else {
    const float step = (end - start) / static_cast<float>(num - 1);
    for (int i = 0; i < num; ++i) correct[i] = start + step * static_cast<float>(i);
  }

  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct.data(),
                                 outputs_[0]->ElementsNum()));

  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  delete kernel;
}

#ifdef SUPPORT_FT04
TEST_F(TestDSP_Linspace, Linspace_Fp16) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  const int num = 257;
  const float start = -1.0f;
  const float end = 1.0f;  // inclusive

  // inputs: start (fp32), end (fp32), num (i32)
  auto in_start = new lite::Tensor(kNumberTypeFloat32, {1}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  in_start->MallocData(allocator_);
  *reinterpret_cast<float *>(in_start->MutableData()) = start;
  inputs_.push_back(in_start);

  auto in_end = new lite::Tensor(kNumberTypeFloat32, {1}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  in_end->MallocData(allocator_);
  *reinterpret_cast<float *>(in_end->MutableData()) = end;
  inputs_.push_back(in_end);

  auto in_num = new lite::Tensor(kNumberTypeInt32, {1}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  in_num->MallocData(allocator_);
  *reinterpret_cast<int32_t *>(in_num->MutableData()) = num;
  inputs_.push_back(in_num);

  // output: fp16 length=num
  auto output = new lite::Tensor(kNumberTypeFloat16, {num}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  output->MallocData(allocator_);
  outputs_.push_back(output);

  std::memset(output->MutableData(), 0, static_cast<size_t>(num) * sizeof(uint16_t));

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_LinSpace};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto *param = new OpParameter();
  param->type_ = static_cast<int>(schema::PrimitiveType_LinSpace);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  // expected: quantized to fp16 grid (via fp32->fp16->fp32)
  std::vector<float> correct(num);
  if (num == 1) {
    correct[0] = Fp16ToFp32(Fp32ToFp16(start));
  } else {
    const float step = (end - start) / static_cast<float>(num - 1);
    for (int i = 0; i < num; ++i) {
      float v = start + step * static_cast<float>(i);
      correct[i] = Fp16ToFp32(Fp32ToFp16(v));
    }
  }

  auto out_fp16 = reinterpret_cast<uint16_t *>(outputs_[0]->MutableData());
  std::vector<float> actual(num);
  for (int i = 0; i < num; ++i) actual[i] = Fp16ToFp32(out_fp16[i]);

  ASSERT_EQ(0, CompareOutputData(actual.data(), correct.data(), outputs_[0]->ElementsNum(), 1e-4));

  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  delete kernel;
}
#endif

}  // namespace mindspore::lite::dsp::test
