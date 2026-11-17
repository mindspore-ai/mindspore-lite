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

#include <cstdlib>
#include <cstring>
#include <vector>
#include "ut/src/runtime/kernel/dsp/dsp_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel/cpu/nnacl_c/crop_parameter.h"
#include "src/litert/kernel_registry.h"

#ifdef SUPPORT_FT04

namespace mindspore::lite::dsp::test {

namespace {
constexpr int kCropBatch = 2;
constexpr int kCropHeight = 6;
constexpr int kCropWidth = 6;
constexpr int kCropChannels = 4;
constexpr int kCropOutHeight = 5;
constexpr int kCropOutWidth = 5;
constexpr int kCropOutChannels = 3;
constexpr int kCropAxis = 1;
constexpr int kSingleOffsetAxis = 2;
constexpr int kHeightOffset = 0;
constexpr int kWidthOffset = 1;
constexpr int kChannelOffset = 1;
constexpr int kRepeatedOffset = 1;
constexpr int kOffsetCount = 3;
constexpr int kSingleOffsetCount = 1;
// One past the last dimension of a 4D input, which crop_infer.c and the kernel both reject.
constexpr int kInvalidAxis = 4;
// With axis 1 these offsets stop at dimension 2 instead of reaching the last dimension.
constexpr int kEarlyOffsetCount = 2;
constexpr size_t kRank = 4;
constexpr size_t kShapeSize = 4;
constexpr float kValueScale = 0.1f;

// Fills a tensor with values that are all different, so a misplaced copy shows up immediately.
void FillCropInput(lite::Tensor *value) {
  auto *data = reinterpret_cast<float *>(value->MutableData());
  const auto elements = static_cast<size_t>(value->ElementsNum());
  for (size_t i = 0; i < elements; ++i) {
    data[i] = static_cast<float>(i) * kValueScale;
  }
}

// A rejected geometry never reaches the DSP, so these cases only need the shapes, the tensor that
// carries the output shape and the parameter. The caller owns the tensors it gets back.
void MakeRejectCaseTensors(const std::shared_ptr<DSPAllocator> &allocator, const std::vector<int> &in_shape,
                           const std::vector<int> &out_shape, lite::Tensor **value, lite::Tensor **shape_tensor,
                           lite::Tensor **output) {
  *value = new lite::Tensor(kNumberTypeFloat32, in_shape, NHWC, lite::Category::CONST_TENSOR);
  *shape_tensor = new lite::Tensor(kNumberTypeInt32, std::vector<int>{static_cast<int>(kShapeSize)}, NHWC,
                                   lite::Category::CONST_TENSOR);
  *output = new lite::Tensor(kNumberTypeFloat32, out_shape, NHWC, lite::Category::CONST_TENSOR);
  if ((*shape_tensor)->MallocData(allocator) != lite::RET_OK) {
    return;
  }
  auto *shape_data = reinterpret_cast<int32_t *>((*shape_tensor)->MutableData());
  for (size_t dim = 0; dim < kRank; ++dim) {
    shape_data[dim] = static_cast<int32_t>(out_shape[dim]);
  }
}

// Fills the parameter the way crop_populate.cc leaves it: one OpParameter head plus the Crop fields.
CropParameter *MakeCropParameter(int64_t axis, int offset_size, int64_t offset0, int64_t offset1, int64_t offset2) {
  auto *param = static_cast<CropParameter *>(malloc(sizeof(CropParameter)));
  if (param == nullptr) {
    return nullptr;
  }
  (void)memset(param, 0, sizeof(CropParameter));
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_Crop);
  param->axis_ = axis;
  param->offset_size_ = offset_size;
  param->offset_[0] = offset0;
  param->offset_[1] = offset1;
  param->offset_[2] = offset2;
  return param;
}
}  // namespace

class TestDSP_Crop : public DSPCommonTest {};

TEST_F(TestDSP_Crop, Crop_Fp32) {
  InitDSPRuntime();
  const std::vector<int> in_shape = {kCropBatch, kCropHeight, kCropWidth, kCropChannels};
  const std::vector<int> out_shape = {kCropBatch, kCropOutHeight, kCropOutWidth, kCropOutChannels};
  auto *value = new lite::Tensor(kNumberTypeFloat32, in_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(value->MallocData(allocator_), lite::RET_OK);
  FillCropInput(value);
  auto *value_data = reinterpret_cast<float *>(value->MutableData());
  auto *shape_tensor = new lite::Tensor(kNumberTypeInt32, std::vector<int>{static_cast<int>(kShapeSize)}, NHWC,
                                        lite::Category::CONST_TENSOR);
  ASSERT_EQ(shape_tensor->MallocData(allocator_), lite::RET_OK);
  auto *shape_data = reinterpret_cast<int32_t *>(shape_tensor->MutableData());
  for (size_t dim = 0; dim < kRank; ++dim) {
    shape_data[dim] = static_cast<int32_t>(out_shape[dim]);
  }
  auto *output = new lite::Tensor(kNumberTypeFloat32, out_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(output->ElementsNum()) * sizeof(float));

  // LiteKernel::~LiteKernel() releases op_parameter_ with free(), so the parameter has to
  // come from malloc rather than new.
  auto *param = MakeCropParameter(kCropAxis, kOffsetCount, kHeightOffset, kWidthOffset, kChannelOffset);
  ASSERT_NE(param, nullptr);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Crop};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({value, shape_tensor}, {output}, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<float *>(output->MutableData());
  for (int n = 0; n < kCropBatch; ++n) {
    for (int h = 0; h < kCropOutHeight; ++h) {
      for (int w = 0; w < kCropOutWidth; ++w) {
        for (int c = 0; c < kCropOutChannels; ++c) {
          const int src = ((n * kCropHeight + h + kHeightOffset) * kCropWidth + w + kWidthOffset) * kCropChannels + c +
                          kChannelOffset;
          const int dst = ((n * kCropOutHeight + h) * kCropOutWidth + w) * kCropOutChannels + c;
          ASSERT_FLOAT_EQ(out_data[dst], value_data[src]);
        }
      }
    }
  }

  delete kernel;
  delete ctx;
  delete value;
  delete shape_tensor;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Crop, Crop_Fp16) {
  InitDSPRuntime();
  const std::vector<int> in_shape = {kCropBatch, kCropHeight, kCropWidth, kCropChannels};
  const std::vector<int> out_shape = {kCropBatch, kCropOutHeight, kCropOutWidth, kCropOutChannels};
  auto *value = new lite::Tensor(kNumberTypeFloat16, in_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(value->MallocData(allocator_), lite::RET_OK);
  auto *value_data = reinterpret_cast<uint16_t *>(value->MutableData());
  const auto in_elements = static_cast<size_t>(value->ElementsNum());
  for (size_t i = 0; i < in_elements; ++i) {
    value_data[i] = Fp32ToFp16(static_cast<float>(i) * kValueScale);
  }
  auto *shape_tensor = new lite::Tensor(kNumberTypeInt32, std::vector<int>{static_cast<int>(kShapeSize)}, NHWC,
                                        lite::Category::CONST_TENSOR);
  ASSERT_EQ(shape_tensor->MallocData(allocator_), lite::RET_OK);
  auto *shape_data = reinterpret_cast<int32_t *>(shape_tensor->MutableData());
  for (size_t dim = 0; dim < kRank; ++dim) {
    shape_data[dim] = static_cast<int32_t>(out_shape[dim]);
  }
  auto *output = new lite::Tensor(kNumberTypeFloat16, out_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  std::memset(output->MutableData(), 0, static_cast<size_t>(output->ElementsNum()) * sizeof(uint16_t));

  auto *param = MakeCropParameter(kCropAxis, kOffsetCount, kHeightOffset, kWidthOffset, kChannelOffset);
  ASSERT_NE(param, nullptr);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_Crop};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({value, shape_tensor}, {output}, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<uint16_t *>(output->MutableData());
  for (int n = 0; n < kCropBatch; ++n) {
    for (int h = 0; h < kCropOutHeight; ++h) {
      for (int w = 0; w < kCropOutWidth; ++w) {
        for (int c = 0; c < kCropOutChannels; ++c) {
          const int src = ((n * kCropHeight + h + kHeightOffset) * kCropWidth + w + kWidthOffset) * kCropChannels + c +
                          kChannelOffset;
          const int dst = ((n * kCropOutHeight + h) * kCropOutWidth + w) * kCropOutChannels + c;
          ASSERT_FLOAT_EQ(Fp16ToFp32(out_data[dst]), Fp16ToFp32(value_data[src]));
        }
      }
    }
  }

  delete kernel;
  delete ctx;
  delete value;
  delete shape_tensor;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Crop, Crop_Fp32_RepeatedOffset) {
  InitDSPRuntime();
  // One offset with axis 2 stands for "shift every dimension from 2 on by the same amount", so the
  // output keeps the full height here.
  const std::vector<int> in_shape = {kCropBatch, kCropHeight, kCropWidth, kCropChannels};
  const std::vector<int> out_shape = {kCropBatch, kCropHeight, kCropOutWidth, kCropOutChannels};
  auto *value = new lite::Tensor(kNumberTypeFloat32, in_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(value->MallocData(allocator_), lite::RET_OK);
  FillCropInput(value);
  auto *value_data = reinterpret_cast<float *>(value->MutableData());
  auto *shape_tensor = new lite::Tensor(kNumberTypeInt32, std::vector<int>{static_cast<int>(kShapeSize)}, NHWC,
                                        lite::Category::CONST_TENSOR);
  ASSERT_EQ(shape_tensor->MallocData(allocator_), lite::RET_OK);
  auto *shape_data = reinterpret_cast<int32_t *>(shape_tensor->MutableData());
  for (size_t dim = 0; dim < kRank; ++dim) {
    shape_data[dim] = static_cast<int32_t>(out_shape[dim]);
  }
  auto *output = new lite::Tensor(kNumberTypeFloat32, out_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  // The cleared span follows the tensor, so a case that keeps the full height clears all 180 of its
  // elements and an unwritten one can never pass by holding a stale value.
  std::memset(output->MutableData(), 0, static_cast<size_t>(output->ElementsNum()) * sizeof(float));

  auto *param = MakeCropParameter(kSingleOffsetAxis, kSingleOffsetCount, kRepeatedOffset, 0, 0);
  ASSERT_NE(param, nullptr);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Crop};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({value, shape_tensor}, {output}, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());

  auto *out_data = reinterpret_cast<float *>(output->MutableData());
  for (int n = 0; n < kCropBatch; ++n) {
    for (int h = 0; h < kCropHeight; ++h) {
      for (int w = 0; w < kCropOutWidth; ++w) {
        for (int c = 0; c < kCropOutChannels; ++c) {
          const int src =
            ((n * kCropHeight + h) * kCropWidth + w + kRepeatedOffset) * kCropChannels + c + kRepeatedOffset;
          const int dst = ((n * kCropHeight + h) * kCropOutWidth + w) * kCropOutChannels + c;
          ASSERT_FLOAT_EQ(out_data[dst], value_data[src]);
        }
      }
    }
  }

  delete kernel;
  delete ctx;
  delete value;
  delete shape_tensor;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Crop, Crop_Fp32_ResizeMakesOffsetInvalid) {
  InitDSPRuntime();
  const std::vector<int> in_shape = {kCropBatch, kCropHeight, kCropWidth, kCropChannels};
  const std::vector<int> out_shape = {kCropBatch, kCropOutHeight, kCropOutWidth, kCropOutChannels};
  auto *value = new lite::Tensor(kNumberTypeFloat32, in_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(value->MallocData(allocator_), lite::RET_OK);
  auto *shape_tensor = new lite::Tensor(kNumberTypeInt32, std::vector<int>{static_cast<int>(kShapeSize)}, NHWC,
                                        lite::Category::CONST_TENSOR);
  ASSERT_EQ(shape_tensor->MallocData(allocator_), lite::RET_OK);
  auto *shape_data = reinterpret_cast<int32_t *>(shape_tensor->MutableData());
  for (size_t dim = 0; dim < kRank; ++dim) {
    shape_data[dim] = static_cast<int32_t>(out_shape[dim]);
  }
  auto *output = new lite::Tensor(kNumberTypeFloat32, out_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);

  auto *param = MakeCropParameter(kCropAxis, kOffsetCount, kHeightOffset, kWidthOffset, kChannelOffset);
  ASSERT_NE(param, nullptr);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Crop};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({value, shape_tensor}, {output}, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());

  // A resized input that no longer holds what the offsets read has to be refused by Prepare, which
  // is the only hook DSPKernel::ReSize re-runs after CheckSpecs was done at creation time.
  const std::vector<int> resized_in_shape = {kCropBatch, kCropHeight, kCropWidth - 2, kCropChannels};
  value->set_shape(resized_in_shape);
  ASSERT_EQ(RET_ERROR, kernel->Prepare());

  delete kernel;
  delete ctx;
  delete value;
  delete shape_tensor;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Crop, Crop_Fp32_AxisOutOfRange) {
  InitDSPRuntime();
  const std::vector<int> in_shape = {kCropBatch, kCropHeight, kCropWidth, kCropChannels};
  const std::vector<int> out_shape = {kCropBatch, kCropOutHeight, kCropOutWidth, kCropOutChannels};
  lite::Tensor *value = nullptr;
  lite::Tensor *shape_tensor = nullptr;
  lite::Tensor *output = nullptr;
  MakeRejectCaseTensors(allocator_, in_shape, out_shape, &value, &shape_tensor, &output);
  ASSERT_NE(value, nullptr);
  ASSERT_NE(shape_tensor->MutableData(), nullptr);
  // The offsets would be fine for axis 1, but an axis of 4 has no dimension to start from.
  auto *param = MakeCropParameter(kInvalidAxis, kOffsetCount, kHeightOffset, kWidthOffset, kChannelOffset);
  ASSERT_NE(param, nullptr);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Crop};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  // The creator releases the kernel it built, and with it op_parameter_, so nothing is freed here.
  auto *kernel = creator({value, shape_tensor}, {output}, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_EQ(kernel, nullptr);

  delete ctx;
  delete value;
  delete shape_tensor;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Crop, Crop_Fp32_OffsetStopsBeforeLastDim) {
  InitDSPRuntime();
  const std::vector<int> in_shape = {kCropBatch, kCropHeight, kCropWidth, kCropChannels};
  const std::vector<int> out_shape = {kCropBatch, kCropOutHeight, kCropOutWidth, kCropOutChannels};
  lite::Tensor *value = nullptr;
  lite::Tensor *shape_tensor = nullptr;
  lite::Tensor *output = nullptr;
  MakeRejectCaseTensors(allocator_, in_shape, out_shape, &value, &shape_tensor, &output);
  ASSERT_NE(value, nullptr);
  ASSERT_NE(shape_tensor->MutableData(), nullptr);
  // crop_infer.c accepts a list of offsets only when it reaches down to the last dimension.
  auto *param = MakeCropParameter(kCropAxis, kEarlyOffsetCount, kWidthOffset, kChannelOffset, 0);
  ASSERT_NE(param, nullptr);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Crop};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({value, shape_tensor}, {output}, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_EQ(kernel, nullptr);

  delete ctx;
  delete value;
  delete shape_tensor;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Crop, Crop_Fp32_OffsetReadsPastInput) {
  InitDSPRuntime();
  const std::vector<int> in_shape = {kCropBatch, kCropHeight, kCropWidth, kCropChannels};
  // The channel offset of 1 leaves 3 channels to read, so an output of 4 channels would run past
  // the input; the check has to refuse it before any DSP launch.
  const std::vector<int> out_shape = {kCropBatch, kCropOutHeight, kCropOutWidth, kCropChannels};
  lite::Tensor *value = nullptr;
  lite::Tensor *shape_tensor = nullptr;
  lite::Tensor *output = nullptr;
  MakeRejectCaseTensors(allocator_, in_shape, out_shape, &value, &shape_tensor, &output);
  ASSERT_NE(value, nullptr);
  ASSERT_NE(shape_tensor->MutableData(), nullptr);
  auto *param = MakeCropParameter(kCropAxis, kOffsetCount, kHeightOffset, kWidthOffset, kChannelOffset);
  ASSERT_NE(param, nullptr);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Crop};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator({value, shape_tensor}, {output}, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_EQ(kernel, nullptr);

  delete ctx;
  delete value;
  delete shape_tensor;
  delete output;
  UninitDSPRuntime();
}
}  // namespace mindspore::lite::dsp::test
#endif  // SUPPORT_FT04
