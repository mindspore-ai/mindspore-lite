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
#include "src/litert/kernel/cpu/nnacl_c/clip_parameter.h"

#ifdef SUPPORT_FT04

namespace mindspore::lite::dsp::test {

namespace {
constexpr int kClipLength = 100;
constexpr int kValueCenter = kClipLength / 2;
constexpr float kClipMin = -5.0f;
constexpr float kClipMax = 5.0f;
// Half of the samples fall outside the bounds and the rest stay inside, so a kernel that clamps too
// early and one that never clamps are both caught.
constexpr float kFpStep = 0.45f;
constexpr int kIntStep = 25;
constexpr float kFpErrBound = 1e-3f;
constexpr int kScalarShapeSize = 1;
constexpr int kResizedLength = kClipLength * 2;
}  // namespace

class TestDSP_Clip : public DSPCommonTest {};

namespace {
// LiteKernel::~LiteKernel() releases op_parameter_ with free(), so the parameter has to come from
// malloc rather than new.
ClipParameter *MakeClipParameter(float min_value, float max_value) {
  auto *param = static_cast<ClipParameter *>(malloc(sizeof(ClipParameter)));
  if (param == nullptr) {
    return nullptr;
  }
  (void)memset(param, 0, sizeof(ClipParameter));
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_Clip);
  param->min_val_ = min_value;
  param->max_val_ = max_value;
  return param;
}

// The bounds come in as arguments rather than from the constants above, so a case hands the reference
// and MakeClipParameter the same pair. A case that later checks one-sided or extreme bounds cannot
// then drift away from a reference that still clamps to -5 and 5.
float ClipRef(float value, float lower, float upper) {
  if (value < lower) {
    return lower;
  }
  return (value > upper) ? upper : value;
}
}  // namespace

TEST_F(TestDSP_Clip, Clip_Fp32) {
  InitDSPRuntime();
  const std::vector<int> shape = {kClipLength};
  auto *input = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  auto *input_data = reinterpret_cast<float *>(input->MutableData());
  for (int j = 0; j < kClipLength; ++j) {
    input_data[j] = static_cast<float>(j - kValueCenter) * kFpStep;
  }
  auto *output = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(kClipLength) * sizeof(float));

  auto *param = MakeClipParameter(kClipMin, kClipMax);
  ASSERT_NE(param, nullptr);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Clip};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({input}, {output}, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<float *>(output->MutableData());
  for (int j = 0; j < kClipLength; ++j) {
    ASSERT_NEAR(out_data[j], ClipRef(input_data[j], kClipMin, kClipMax), kFpErrBound);
  }

  delete kernel;
  delete ctx;
  delete input;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Clip, Clip_Fp16) {
  InitDSPRuntime();
  const std::vector<int> shape = {kClipLength};
  auto *input = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  auto *input_data = reinterpret_cast<uint16_t *>(input->MutableData());
  for (int j = 0; j < kClipLength; ++j) {
    input_data[j] = Fp32ToFp16(static_cast<float>(j - kValueCenter) * kFpStep);
  }
  auto *output = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(kClipLength) * sizeof(uint16_t));

  auto *param = MakeClipParameter(kClipMin, kClipMax);
  ASSERT_NE(param, nullptr);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_Clip};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({input}, {output}, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  // Both bounds are representable in fp16, so a clamped element is expected to match bit for bit.
  auto *out_data = reinterpret_cast<uint16_t *>(output->MutableData());
  for (int j = 0; j < kClipLength; ++j) {
    ASSERT_NEAR(Fp16ToFp32(out_data[j]), ClipRef(Fp16ToFp32(input_data[j]), kClipMin, kClipMax), kFpErrBound);
  }

  delete kernel;
  delete ctx;
  delete input;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Clip, Clip_Int16) {
  InitDSPRuntime();
  const std::vector<int> shape = {kClipLength};
  auto *input = new lite::Tensor(kNumberTypeInt16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  auto *input_data = reinterpret_cast<int16_t *>(input->MutableData());
  for (int j = 0; j < kClipLength; ++j) {
    input_data[j] = static_cast<int16_t>((j - kValueCenter) * kIntStep);
  }
  auto *output = new lite::Tensor(kNumberTypeInt16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(kClipLength) * sizeof(int16_t));

  auto *param = MakeClipParameter(kClipMin, kClipMax);
  ASSERT_NE(param, nullptr);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt16, NHWC, schema::PrimitiveType_Clip};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({input}, {output}, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<int16_t *>(output->MutableData());
  for (int j = 0; j < kClipLength; ++j) {
    ASSERT_EQ(out_data[j], static_cast<int16_t>(ClipRef(static_cast<float>(input_data[j]), kClipMin, kClipMax)));
  }

  delete kernel;
  delete ctx;
  delete input;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Clip, Clip_Int32) {
  InitDSPRuntime();
  const std::vector<int> shape = {kClipLength};
  auto *input = new lite::Tensor(kNumberTypeInt32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  auto *input_data = reinterpret_cast<int32_t *>(input->MutableData());
  for (int j = 0; j < kClipLength; ++j) {
    input_data[j] = (j - kValueCenter) * kIntStep;
  }
  auto *output = new lite::Tensor(kNumberTypeInt32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(kClipLength) * sizeof(int32_t));

  auto *param = MakeClipParameter(kClipMin, kClipMax);
  ASSERT_NE(param, nullptr);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_Clip};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({input}, {output}, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<int32_t *>(output->MutableData());
  for (int j = 0; j < kClipLength; ++j) {
    ASSERT_EQ(out_data[j], static_cast<int32_t>(ClipRef(static_cast<float>(input_data[j]), kClipMin, kClipMax)));
  }

  delete kernel;
  delete ctx;
  delete input;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Clip, Clip_TensorBoundsAreRejected) {
  InitDSPRuntime();
  const std::vector<int> shape = {kClipLength};
  const std::vector<int> scalar_shape = {kScalarShapeSize};
  auto *input = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  auto *min_tensor = new lite::Tensor(kNumberTypeFloat32, scalar_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(min_tensor->MallocData(allocator_), lite::RET_OK);
  *reinterpret_cast<float *>(min_tensor->MutableData()) = kClipMin;
  auto *max_tensor = new lite::Tensor(kNumberTypeFloat32, scalar_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(max_tensor->MallocData(allocator_), lite::RET_OK);
  *reinterpret_cast<float *>(max_tensor->MutableData()) = kClipMax;
  auto *output = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);

  // The DSP entry passes the bounds to clip_func as host floats, so a graph that carries them as
  // inputs has to stay on the CPU kernel instead of guessing from the attributes.
  auto *param = MakeClipParameter(kClipMin, kClipMax);
  ASSERT_NE(param, nullptr);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Clip};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({input, min_tensor, max_tensor}, {output}, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_EQ(kernel, nullptr);

  delete ctx;
  delete input;
  delete min_tensor;
  delete max_tensor;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Clip, Clip_MinNotBelowMaxIsRejected) {
  InitDSPRuntime();
  const std::vector<int> shape = {kClipLength};
  auto *input = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  auto *output = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);

  // The CPU kernel refuses the same parameter, so the DSP kernel must not quietly pick a side.
  auto *param = MakeClipParameter(kClipMax, kClipMin);
  ASSERT_NE(param, nullptr);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Clip};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({input}, {output}, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_EQ(kernel, nullptr);

  delete ctx;
  delete input;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Clip, Clip_OutputDtypeMismatchIsRejected) {
  InitDSPRuntime();
  const std::vector<int> shape = {kClipLength};
  auto *input = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  // clip_func writes back in the dtype it read, so an fp16 buffer would be overrun element by element.
  auto *output = new lite::Tensor(kNumberTypeFloat16, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);

  auto *param = MakeClipParameter(kClipMin, kClipMax);
  ASSERT_NE(param, nullptr);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Clip};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({input}, {output}, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_EQ(kernel, nullptr);

  delete ctx;
  delete input;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Clip, Clip_ResizeMakesSizesDiffer) {
  InitDSPRuntime();
  const std::vector<int> shape = {kClipLength};
  auto *input = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(input->MallocData(allocator_), lite::RET_OK);
  auto *output = new lite::Tensor(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);

  auto *param = MakeClipParameter(kClipMin, kClipMax);
  ASSERT_NE(param, nullptr);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Clip};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({input}, {output}, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Run());

  // A resize leaves CheckSpecs behind, so Prepare has to notice that the buffers no longer match.
  input->set_shape({kResizedLength});
  ASSERT_EQ(RET_ERROR, kernel->Prepare());
  ASSERT_EQ(RET_ERROR, kernel->Run());

  delete kernel;
  delete ctx;
  delete input;
  delete output;
  UninitDSPRuntime();
}
}  // namespace mindspore::lite::dsp::test
#endif  // SUPPORT_FT04
