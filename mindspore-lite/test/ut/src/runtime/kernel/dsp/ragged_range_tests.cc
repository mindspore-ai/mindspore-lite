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

namespace mindspore::lite::dsp::test {

class TestDSP_RaggedRange : public DSPCommonTest {};

TEST_F(TestDSP_RaggedRange, RaggedRange_Fp32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  // Larger dataset: rows=5
  // starts=[0,10,-5,100,7], limits=[50,60,5,110,27], deltas=[1,2,3,1,4]
  std::vector<int> vec5 = {5};
  auto t_starts = new lite::Tensor(kNumberTypeFloat32, vec5, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_starts->MallocData(allocator_);
  auto t_limits = new lite::Tensor(kNumberTypeFloat32, vec5, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_limits->MallocData(allocator_);
  auto t_deltas = new lite::Tensor(kNumberTypeFloat32, vec5, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_deltas->MallocData(allocator_);
  inputs_.push_back(t_starts);
  inputs_.push_back(t_limits);
  inputs_.push_back(t_deltas);

  auto starts_data = reinterpret_cast<float *>(t_starts->MutableData());
  auto limits_data = reinterpret_cast<float *>(t_limits->MutableData());
  auto deltas_data = reinterpret_cast<float *>(t_deltas->MutableData());
  float starts_host[5] = {0.f, 10.f, -5.f, 100.f, 7.f};
  float limits_host[5] = {50.f, 60.f, 5.f, 110.f, 27.f};
  float deltas_host[5] = {1.f, 2.f, 3.f, 1.f, 4.f};
  std::memcpy(starts_data, starts_host, sizeof(starts_host));
  std::memcpy(limits_data, limits_host, sizeof(limits_host));
  std::memcpy(deltas_data, deltas_host, sizeof(deltas_host));

  // outputs (splits size rows+1, values computed below)
  auto t_splits = new lite::Tensor(kNumberTypeInt32, {6}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_splits->MallocData(allocator_);
  // rough upper bound for values, we'll only compare first computed_len elements
  auto t_values = new lite::Tensor(kNumberTypeFloat32, {200}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_values->MallocData(allocator_);
  outputs_.push_back(t_splits);
  outputs_.push_back(t_values);

  std::fill_n(reinterpret_cast<int32_t *>(t_splits->MutableData()), 6, 0);
  std::fill_n(reinterpret_cast<float *>(t_values->MutableData()), 200, 0.0f);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_RaggedRange};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto *param = new OpParameter();
  param->type_ = static_cast<int>(schema::PrimitiveType_RaggedRange);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  // build expected
  std::vector<int32_t> expect_splits(6, 0);
  std::vector<float> expect_values;
  int32_t acc = 0;
  for (int r = 0; r < 5; ++r) {
    expect_splits[r] = acc;
    for (float v = starts_host[r]; v < limits_host[r]; v += deltas_host[r]) {
      expect_values.push_back(v);
    }
    acc = static_cast<int32_t>(expect_values.size());
  }
  expect_splits[5] = acc;

  // compare splits
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int32_t *>(outputs_[0]->MutableData()), expect_splits.data(), 6));
  // compare first acc values
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[1]->MutableData()), expect_values.data(), acc));

  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  delete kernel;
}

TEST_F(TestDSP_RaggedRange, RaggedRange_Int32) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  // Larger dataset: rows=4
  std::vector<int> vec4 = {4};
  auto t_starts = new lite::Tensor(kNumberTypeInt32, vec4, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_starts->MallocData(allocator_);
  auto t_limits = new lite::Tensor(kNumberTypeInt32, vec4, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_limits->MallocData(allocator_);
  auto t_deltas = new lite::Tensor(kNumberTypeInt32, vec4, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_deltas->MallocData(allocator_);
  inputs_.push_back(t_starts);
  inputs_.push_back(t_limits);
  inputs_.push_back(t_deltas);

  auto starts_data = reinterpret_cast<int32_t *>(t_starts->MutableData());
  auto limits_data = reinterpret_cast<int32_t *>(t_limits->MutableData());
  auto deltas_data = reinterpret_cast<int32_t *>(t_deltas->MutableData());
  int32_t starts_host[4] = {0, -100, 5, 1000};
  int32_t limits_host[4] = {200, -50, 50, 1010};
  int32_t deltas_host[4] = {2, 5, 3, 1};
  std::memcpy(starts_data, starts_host, sizeof(starts_host));
  std::memcpy(limits_data, limits_host, sizeof(limits_host));
  std::memcpy(deltas_data, deltas_host, sizeof(deltas_host));

  auto t_splits = new lite::Tensor(kNumberTypeInt32, {5}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_splits->MallocData(allocator_);
  auto t_values = new lite::Tensor(kNumberTypeInt32, {300}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_values->MallocData(allocator_);
  outputs_.push_back(t_splits);
  outputs_.push_back(t_values);

  std::fill_n(reinterpret_cast<int32_t *>(t_splits->MutableData()), 5, 0);
  std::fill_n(reinterpret_cast<int32_t *>(t_values->MutableData()), 300, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_RaggedRange};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto *param = new OpParameter();
  param->type_ = static_cast<int>(schema::PrimitiveType_RaggedRange);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int32_t> expect_splits(5, 0);
  std::vector<int32_t> expect_values;
  int32_t acc = 0;
  for (int r = 0; r < 4; ++r) {
    expect_splits[r] = acc;
    for (int32_t v = starts_host[r]; v < limits_host[r]; v += deltas_host[r]) {
      expect_values.push_back(v);
    }
    acc = static_cast<int32_t>(expect_values.size());
  }
  expect_splits[4] = acc;

  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int32_t *>(outputs_[0]->MutableData()), expect_splits.data(), 5));
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int32_t *>(outputs_[1]->MutableData()), expect_values.data(), acc));

  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  delete kernel;
}

#ifdef SUPPORT_FT04
TEST_F(TestDSP_RaggedRange, RaggedRange_Fp16) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  // Larger dataset with fp32 inputs and fp16 outputs
  std::vector<int> vec3 = {3};
  auto t_starts = new lite::Tensor(kNumberTypeFloat32, vec3, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_starts->MallocData(allocator_);
  auto t_limits = new lite::Tensor(kNumberTypeFloat32, vec3, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_limits->MallocData(allocator_);
  auto t_deltas = new lite::Tensor(kNumberTypeFloat32, vec3, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_deltas->MallocData(allocator_);
  inputs_.push_back(t_starts);
  inputs_.push_back(t_limits);
  inputs_.push_back(t_deltas);

  auto starts_f = reinterpret_cast<float *>(t_starts->MutableData());
  auto limits_f = reinterpret_cast<float *>(t_limits->MutableData());
  auto deltas_f = reinterpret_cast<float *>(t_deltas->MutableData());
  float starts_host[3] = {-10.f, 0.f, 1.5f};
  float limits_host[3] = {0.f, 50.f, 6.f};
  float deltas_host[3] = {0.5f, 1.f, 1.25f};
  std::memcpy(starts_f, starts_host, sizeof(starts_host));
  std::memcpy(limits_f, limits_host, sizeof(limits_host));
  std::memcpy(deltas_f, deltas_host, sizeof(deltas_host));

  // outputs
  auto t_splits = new lite::Tensor(kNumberTypeInt32, {4}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_splits->MallocData(allocator_);
  auto t_values = new lite::Tensor(kNumberTypeFloat16, {200}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_values->MallocData(allocator_);
  outputs_.push_back(t_splits);
  outputs_.push_back(t_values);

  std::fill_n(reinterpret_cast<int32_t *>(t_splits->MutableData()), 4, 0);
  std::memset(t_values->MutableData(), 0, 200 * sizeof(uint16_t));

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_RaggedRange};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto *param = new OpParameter();
  param->type_ = static_cast<int>(schema::PrimitiveType_RaggedRange);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  // expected
  std::vector<int32_t> expect_splits(4, 0);
  std::vector<float> expect_values;
  int32_t acc = 0;
  for (int r = 0; r < 3; ++r) {
    expect_splits[r] = acc;
    for (float v = starts_host[r]; v < limits_host[r]; v += deltas_host[r]) {
      expect_values.push_back(v);
    }
    acc = static_cast<int32_t>(expect_values.size());
  }
  expect_splits[3] = acc;

  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int32_t *>(outputs_[0]->MutableData()), expect_splits.data(), 4));

  auto out_fp16 = reinterpret_cast<uint16_t *>(outputs_[1]->MutableData());
  std::vector<float> actual(acc);
  for (int i = 0; i < acc; ++i) actual[i] = Fp16ToFp32(out_fp16[i]);
  std::vector<float> correct(acc);
  for (int i = 0; i < acc; ++i) correct[i] = Fp16ToFp32(Fp32ToFp16(expect_values[i]));
  ASSERT_EQ(0, CompareOutputData(actual.data(), correct.data(), acc, 1e-3));

  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  delete kernel;
}

TEST_F(TestDSP_RaggedRange, RaggedRange_Int16) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  // Larger dataset with int32 inputs and int16 outputs
  std::vector<int> vec3 = {3};
  auto t_starts = new lite::Tensor(kNumberTypeInt32, vec3, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_starts->MallocData(allocator_);
  auto t_limits = new lite::Tensor(kNumberTypeInt32, vec3, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_limits->MallocData(allocator_);
  auto t_deltas = new lite::Tensor(kNumberTypeInt32, vec3, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_deltas->MallocData(allocator_);
  inputs_.push_back(t_starts);
  inputs_.push_back(t_limits);
  inputs_.push_back(t_deltas);

  auto starts_d32 = reinterpret_cast<int32_t *>(t_starts->MutableData());
  auto limits_d32 = reinterpret_cast<int32_t *>(t_limits->MutableData());
  auto deltas_d32 = reinterpret_cast<int32_t *>(t_deltas->MutableData());
  int32_t starts_host[3] = {-10, 0, 100};
  int32_t limits_host[3] = {10, 100, 110};
  int32_t deltas_host[3] = {2, 3, 1};
  std::memcpy(starts_d32, starts_host, sizeof(starts_host));
  std::memcpy(limits_d32, limits_host, sizeof(limits_host));
  std::memcpy(deltas_d32, deltas_host, sizeof(deltas_host));

  auto t_splits = new lite::Tensor(kNumberTypeInt32, {4}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_splits->MallocData(allocator_);
  auto t_values = new lite::Tensor(kNumberTypeInt16, {300}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_values->MallocData(allocator_);
  outputs_.push_back(t_splits);
  outputs_.push_back(t_values);

  std::fill_n(reinterpret_cast<int32_t *>(t_splits->MutableData()), 4, 0);
  std::fill_n(reinterpret_cast<int16_t *>(t_values->MutableData()), 300, 0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt16, NHWC, schema::PrimitiveType_RaggedRange};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto *param = new OpParameter();
  param->type_ = static_cast<int>(schema::PrimitiveType_RaggedRange);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int32_t> expect_splits(4, 0);
  std::vector<int16_t> expect_values;
  int32_t acc = 0;
  for (int r = 0; r < 3; ++r) {
    expect_splits[r] = acc;
    for (int32_t v = starts_host[r]; v < limits_host[r]; v += deltas_host[r]) {
      expect_values.push_back(static_cast<int16_t>(v));
    }
    acc = static_cast<int32_t>(expect_values.size());
  }
  expect_splits[3] = acc;

  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int32_t *>(outputs_[0]->MutableData()), expect_splits.data(), 4));
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int16_t *>(outputs_[1]->MutableData()), expect_values.data(), acc));

  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  delete kernel;
}
#endif

#ifdef SUPPORT_FT78
TEST_F(TestDSP_RaggedRange, RaggedRange_Int16) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  std::vector<int> vec3 = {3};
  auto t_starts = new lite::Tensor(kNumberTypeInt16, vec3, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_starts->MallocData(allocator_);
  auto t_limits = new lite::Tensor(kNumberTypeInt16, vec3, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_limits->MallocData(allocator_);
  auto t_deltas = new lite::Tensor(kNumberTypeInt16, vec3, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_deltas->MallocData(allocator_);
  inputs_.push_back(t_starts);
  inputs_.push_back(t_limits);
  inputs_.push_back(t_deltas);

  int16_t starts_host[3] = {-12, 0, 90};
  int16_t limits_host[3] = {-2, 30, 100};
  int16_t deltas_host[3] = {3, 5, 2};
  std::memcpy(t_starts->MutableData(), starts_host, sizeof(starts_host));
  std::memcpy(t_limits->MutableData(), limits_host, sizeof(limits_host));
  std::memcpy(t_deltas->MutableData(), deltas_host, sizeof(deltas_host));

  auto t_splits = new lite::Tensor(kNumberTypeInt32, {4}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_splits->MallocData(allocator_);
  auto t_values = new lite::Tensor(kNumberTypeInt16, {256}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_values->MallocData(allocator_);
  outputs_.push_back(t_splits);
  outputs_.push_back(t_values);

  std::fill_n(reinterpret_cast<int32_t *>(t_splits->MutableData()), 4, 0);
  std::fill_n(reinterpret_cast<int16_t *>(t_values->MutableData()), 256, static_cast<int16_t>(0));

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt16, NHWC, schema::PrimitiveType_RaggedRange};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto *param = new OpParameter();
  param->type_ = static_cast<int>(schema::PrimitiveType_RaggedRange);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int32_t> expect_splits(4, 0);
  std::vector<int16_t> expect_values;
  int32_t acc = 0;
  for (int r = 0; r < 3; ++r) {
    expect_splits[r] = acc;
    for (int v = static_cast<int>(starts_host[r]);
         deltas_host[r] > 0 ? v < static_cast<int>(limits_host[r]) : v > static_cast<int>(limits_host[r]);
         v += static_cast<int>(deltas_host[r])) {
      expect_values.push_back(static_cast<int16_t>(v));
    }
    acc = static_cast<int32_t>(expect_values.size());
  }
  expect_splits[3] = acc;

  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int32_t *>(outputs_[0]->MutableData()), expect_splits.data(), 4));
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<int16_t *>(outputs_[1]->MutableData()), expect_values.data(), acc));

  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  delete kernel;
}

TEST_F(TestDSP_RaggedRange, RaggedRange_Fp64) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> vec4 = {4};
  auto t_starts = new lite::Tensor(kNumberTypeFloat64, vec4, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_starts->MallocData(allocator_);
  auto t_limits = new lite::Tensor(kNumberTypeFloat64, vec4, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_limits->MallocData(allocator_);
  auto t_deltas = new lite::Tensor(kNumberTypeFloat64, vec4, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_deltas->MallocData(allocator_);
  inputs_.push_back(t_starts);
  inputs_.push_back(t_limits);
  inputs_.push_back(t_deltas);

  double starts_host[4] = {-5.0, -5.0, -5.0, -5.0};
  double limits_host[4] = {0.0, 0.0, 0.0, 0.0};
  double deltas_host[4] = {0.25, 0.25, 0.25, 0.25};
  std::memcpy(t_starts->MutableData(), starts_host, sizeof(starts_host));
  std::memcpy(t_limits->MutableData(), limits_host, sizeof(limits_host));
  std::memcpy(t_deltas->MutableData(), deltas_host, sizeof(deltas_host));

  auto t_splits = new lite::Tensor(kNumberTypeInt32, {5}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_splits->MallocData(allocator_);
  auto t_values = new lite::Tensor(kNumberTypeFloat64, {512}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_values->MallocData(allocator_);
  outputs_.push_back(t_splits);
  outputs_.push_back(t_values);

  std::fill_n(reinterpret_cast<int32_t *>(t_splits->MutableData()), 5, 0);
  std::fill_n(reinterpret_cast<double *>(t_values->MutableData()), 512, 0.0);

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat64, NHWC, schema::PrimitiveType_RaggedRange};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto *param = new OpParameter();
  param->type_ = static_cast<int>(schema::PrimitiveType_RaggedRange);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int32_t> expect_splits(5, 0);
  std::vector<double> expect_values;
  int32_t acc = 0;
  for (int r = 0; r < 4; ++r) {
    expect_splits[r] = acc;
    for (double v = starts_host[r]; deltas_host[r] > 0 ? v < limits_host[r] : v > limits_host[r]; v += deltas_host[r]) {
      expect_values.push_back(v);
    }
    acc = static_cast<int32_t>(expect_values.size());
  }
  expect_splits[4] = acc;

  auto actual_splits_ptr = reinterpret_cast<int32_t *>(outputs_[0]->MutableData());
  std::vector<int32_t> actual_splits(actual_splits_ptr, actual_splits_ptr + 5);
  for (size_t i = 0; i < actual_splits.size(); ++i) {
    EXPECT_EQ(expect_splits[i], actual_splits[i]) << "split index " << i;
  }

  auto actual_values_ptr = reinterpret_cast<double *>(outputs_[1]->MutableData());
  std::vector<double> actual_values(actual_values_ptr, actual_values_ptr + acc);
  for (int i = 0; i < acc; ++i) {
    EXPECT_NEAR(expect_values[i], actual_values[i], 1e-6) << "value index " << i;
  }

  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  delete kernel;
}

TEST_F(TestDSP_RaggedRange, RaggedRange_Int8) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;

  std::vector<int> vec4 = {4};
  auto t_starts = new lite::Tensor(kNumberTypeInt8, vec4, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_starts->MallocData(allocator_);
  auto t_limits = new lite::Tensor(kNumberTypeInt8, vec4, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_limits->MallocData(allocator_);
  auto t_deltas = new lite::Tensor(kNumberTypeInt8, vec4, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_deltas->MallocData(allocator_);
  inputs_.push_back(t_starts);
  inputs_.push_back(t_limits);
  inputs_.push_back(t_deltas);

  int8_t starts_host[4] = {-20, -10, 5, 100};
  int8_t limits_host[4] = {-5, 10, 20, 110};
  int8_t deltas_host[4] = {3, 4, 5, 1};
  std::memcpy(t_starts->MutableData(), starts_host, sizeof(starts_host));
  std::memcpy(t_limits->MutableData(), limits_host, sizeof(limits_host));
  std::memcpy(t_deltas->MutableData(), deltas_host, sizeof(deltas_host));

  auto t_splits = new lite::Tensor(kNumberTypeInt32, {5}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_splits->MallocData(allocator_);
  auto t_values = new lite::Tensor(kNumberTypeInt8, {256}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  t_values->MallocData(allocator_);
  outputs_.push_back(t_splits);
  outputs_.push_back(t_values);

  std::fill_n(reinterpret_cast<int32_t *>(t_splits->MutableData()), 5, 0);
  std::fill_n(reinterpret_cast<int8_t *>(t_values->MutableData()), 256, static_cast<int8_t>(0));

  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt8, NHWC, schema::PrimitiveType_RaggedRange};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  auto *param = new OpParameter();
  param->type_ = static_cast<int>(schema::PrimitiveType_RaggedRange);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int32_t> expect_splits(5, 0);
  std::vector<int8_t> expect_values;
  int32_t acc = 0;
  for (int r = 0; r < 4; ++r) {
    expect_splits[r] = acc;
    for (int v = static_cast<int>(starts_host[r]);
         deltas_host[r] > 0 ? v < static_cast<int>(limits_host[r]) : v > static_cast<int>(limits_host[r]);
         v += static_cast<int>(deltas_host[r])) {
      expect_values.push_back(static_cast<int8_t>(v));
    }
    acc = static_cast<int32_t>(expect_values.size());
  }
  expect_splits[4] = acc;

  auto actual_splits_ptr = reinterpret_cast<int32_t *>(outputs_[0]->MutableData());
  std::vector<int32_t> actual_splits(actual_splits_ptr, actual_splits_ptr + 5);
  for (size_t i = 0; i < actual_splits.size(); ++i) {
    EXPECT_EQ(expect_splits[i], actual_splits[i]) << "split index " << i;
  }

  auto actual_values_ptr = reinterpret_cast<int8_t *>(outputs_[1]->MutableData());
  std::vector<int8_t> actual_values(actual_values_ptr, actual_values_ptr + acc);
  for (int i = 0; i < acc; ++i) {
    EXPECT_EQ(expect_values[i], actual_values[i]) << "value index " << i;
  }

  UninitDSPRuntime();
  delete ctx;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  delete kernel;
}
#endif

}  // namespace mindspore::lite::dsp::test
