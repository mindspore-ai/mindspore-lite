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

#ifdef SUPPORT_FT04

namespace mindspore::lite::dsp::test {

class TestDSP_AddGrad : public DSPCommonTest {};

TEST_F(TestDSP_AddGrad, AddGrad_Fp32) {
  // Same shape: dx1 == dx2 == dy. The graph mirrors Lite's AddGrad contract
  // (dy, x1, x2): the kernel only forwards dy's gradient, but it requires the same
  // three inputs a real graph carries.
  {
    InitDSPRuntime();
    const int num = 100;
    std::vector<int> shape = {num};

    auto *dy = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
    auto *x1 = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
    auto *x2 = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
    auto *dx1 = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
    auto *dx2 = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(dy->MallocData(allocator_), lite::RET_OK);
    ASSERT_EQ(x1->MallocData(allocator_), lite::RET_OK);
    ASSERT_EQ(x2->MallocData(allocator_), lite::RET_OK);
    ASSERT_EQ(dx1->MallocData(allocator_), lite::RET_OK);
    ASSERT_EQ(dx2->MallocData(allocator_), lite::RET_OK);

    auto *dy_data = reinterpret_cast<float *>(dy->MutableData());
    auto *x1_data = reinterpret_cast<float *>(x1->MutableData());
    auto *x2_data = reinterpret_cast<float *>(x2->MutableData());
    for (int i = 0; i < num; ++i) {
      dy_data[i] = 0.5f * static_cast<float>(i) - 3.0f;
      x1_data[i] = 1.0f;
      x2_data[i] = 2.0f;
    }
    std::memset(dx1->MutableData(), 0, static_cast<size_t>(num) * sizeof(float));
    std::memset(dx2->MutableData(), 0, static_cast<size_t>(num) * sizeof(float));

    std::vector<lite::Tensor *> inputs_ = {dy, x1, x2};
    std::vector<lite::Tensor *> outputs_ = {dx1, dx2};
    auto ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());

    // LiteKernel::~LiteKernel() releases op_parameter_ with free(), so the parameter
    // has to come from malloc rather than new.
    auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
    ASSERT_NE(param, nullptr);
    (void)memset(param, 0, sizeof(OpParameter));
    param->type_ = static_cast<int>(schema::PrimitiveType_AddGrad);

    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_AddGrad};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto kernel_handle = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
    ASSERT_NE(kernel_handle, nullptr);
    ASSERT_EQ(kernel_handle->Prepare(), lite::RET_OK);
    ASSERT_EQ(kernel_handle->Run(), lite::RET_OK);

    std::vector<float> expect(dy_data, dy_data + num);
    ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(dx1->MutableData()), expect.data(), num, 1e-4));
    ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(dx2->MutableData()), expect.data(), num, 1e-4));

    delete ctx;
    delete dy;
    delete x1;
    delete x2;
    delete dx1;
    delete dx2;
    delete kernel_handle;
    // The runtime owns the allocator used to release device memory above.
    UninitDSPRuntime();
  }

  // Broadcast to scalar: dx1 == dx2 == sum(dy).
  {
    InitDSPRuntime();
    const int num = 100;
    std::vector<int> dy_shape = {num};
    std::vector<int> dx_shape = {1};

    auto *dy = new lite::Tensor(kNumberTypeFloat32, dy_shape, NHWC, lite::Category::CONST_TENSOR);
    auto *x1 = new lite::Tensor(kNumberTypeFloat32, dx_shape, NHWC, lite::Category::CONST_TENSOR);
    auto *x2 = new lite::Tensor(kNumberTypeFloat32, dx_shape, NHWC, lite::Category::CONST_TENSOR);
    auto *dx1 = new lite::Tensor(kNumberTypeFloat32, dx_shape, NHWC, lite::Category::CONST_TENSOR);
    auto *dx2 = new lite::Tensor(kNumberTypeFloat32, dx_shape, NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(dy->MallocData(allocator_), lite::RET_OK);
    ASSERT_EQ(x1->MallocData(allocator_), lite::RET_OK);
    ASSERT_EQ(x2->MallocData(allocator_), lite::RET_OK);
    ASSERT_EQ(dx1->MallocData(allocator_), lite::RET_OK);
    ASSERT_EQ(dx2->MallocData(allocator_), lite::RET_OK);

    auto *dy_data = reinterpret_cast<float *>(dy->MutableData());
    float sum = 0.0f;
    for (int i = 0; i < num; ++i) {
      dy_data[i] = 0.01f * static_cast<float>(i + 1);
      sum += dy_data[i];
    }
    reinterpret_cast<float *>(x1->MutableData())[0] = 1.0f;
    reinterpret_cast<float *>(x2->MutableData())[0] = 2.0f;
    reinterpret_cast<float *>(dx1->MutableData())[0] = 0.0f;
    reinterpret_cast<float *>(dx2->MutableData())[0] = 0.0f;

    std::vector<lite::Tensor *> inputs_ = {dy, x1, x2};
    std::vector<lite::Tensor *> outputs_ = {dx1, dx2};
    auto ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());

    auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
    ASSERT_NE(param, nullptr);
    (void)memset(param, 0, sizeof(OpParameter));
    param->type_ = static_cast<int>(schema::PrimitiveType_AddGrad);

    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_AddGrad};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto kernel_handle = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
    ASSERT_NE(kernel_handle, nullptr);
    ASSERT_EQ(kernel_handle->Prepare(), lite::RET_OK);
    ASSERT_EQ(kernel_handle->Run(), lite::RET_OK);

    ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(dx1->MutableData()), &sum, 1, 1e-4));
    ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(dx2->MutableData()), &sum, 1, 1e-4));

    delete ctx;
    delete dy;
    delete x1;
    delete x2;
    delete dx1;
    delete dx2;
    delete kernel_handle;
    UninitDSPRuntime();
  }
}
TEST_F(TestDSP_AddGrad, AddGrad_Fp16) {
  // Same-shape fp16 case: dx1 == dx2 == dy. Covers the FT04 fp16 registration.
  InitDSPRuntime();
  const int num = 100;
  std::vector<int> shape = {num};

  auto *dy = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::CONST_TENSOR);
  auto *x1 = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::CONST_TENSOR);
  auto *x2 = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::CONST_TENSOR);
  auto *dx1 = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::CONST_TENSOR);
  auto *dx2 = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(dy->MallocData(allocator_), lite::RET_OK);
  ASSERT_EQ(x1->MallocData(allocator_), lite::RET_OK);
  ASSERT_EQ(x2->MallocData(allocator_), lite::RET_OK);
  ASSERT_EQ(dx1->MallocData(allocator_), lite::RET_OK);
  ASSERT_EQ(dx2->MallocData(allocator_), lite::RET_OK);

  auto *dy_data = reinterpret_cast<uint16_t *>(dy->MutableData());
  auto *x1_data = reinterpret_cast<uint16_t *>(x1->MutableData());
  auto *x2_data = reinterpret_cast<uint16_t *>(x2->MutableData());
  for (int i = 0; i < num; ++i) {
    dy_data[i] = Fp32ToFp16(0.5f * static_cast<float>(i) - 3.0f);
    x1_data[i] = Fp32ToFp16(1.0f);
    x2_data[i] = Fp32ToFp16(2.0f);
  }
  std::memset(dx1->MutableData(), 0, static_cast<size_t>(num) * sizeof(uint16_t));
  std::memset(dx2->MutableData(), 0, static_cast<size_t>(num) * sizeof(uint16_t));

  std::vector<lite::Tensor *> inputs_ = {dy, x1, x2};
  std::vector<lite::Tensor *> outputs_ = {dx1, dx2};
  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // LiteKernel::~LiteKernel() releases op_parameter_ with free(), so the parameter
  // has to come from malloc rather than new.
  auto *param = static_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  ASSERT_NE(param, nullptr);
  (void)memset(param, 0, sizeof(OpParameter));
  param->type_ = static_cast<int>(schema::PrimitiveType_AddGrad);

  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_AddGrad};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel_handle = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel_handle, nullptr);
  ASSERT_EQ(kernel_handle->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel_handle->Run(), lite::RET_OK);

  // fp16 keeps roughly 10 bits of mantissa, so compare with a relative bound instead of
  // the absolute 1e-4 used by the fp32 cases.
  constexpr float kFp16ErrBound = 5e-2f;
  auto *dx1_data = reinterpret_cast<uint16_t *>(dx1->MutableData());
  auto *dx2_data = reinterpret_cast<uint16_t *>(dx2->MutableData());
  for (int i = 0; i < num; ++i) {
    const float expect = Fp16ToFp32(dy_data[i]);
    const float tolerance = kFp16ErrBound * (1.0f + std::fabs(expect));
    ASSERT_NEAR(Fp16ToFp32(dx1_data[i]), expect, tolerance);
    ASSERT_NEAR(Fp16ToFp32(dx2_data[i]), expect, tolerance);
  }

  delete ctx;
  delete dy;
  delete x1;
  delete x2;
  delete dx1;
  delete dx2;
  delete kernel_handle;
  // The runtime owns the allocator used to release device memory above.
  UninitDSPRuntime();
}

}  // namespace mindspore::lite::dsp::test
#endif
