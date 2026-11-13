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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "ut/src/runtime/kernel/dsp/dsp_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel/dsp/dsp_subgraph.h"
#include "src/litert/kernel_registry.h"
#include "nnacl_c/gather_parameter.h"

#ifdef SUPPORT_FT04

namespace mindspore::lite::dsp::test {

namespace {
constexpr int kDim0 = 2;
constexpr int kDim1 = 4;
constexpr int kDim2 = 3;
const std::vector<int> kShape = {kDim0, kDim1, kDim2};
const size_t kCells = kDim0 * kDim1 * kDim2;

OpParameter *CreateGatherDParam() {
  auto *param = reinterpret_cast<GatherParameter *>(malloc(sizeof(GatherParameter)));
  if (param == nullptr) {
    return nullptr;
  }
  std::memset(param, 0, sizeof(GatherParameter));
  param->op_parameter_.type_ = schema::PrimitiveType_GatherD;
  return reinterpret_cast<OpParameter *>(param);
}

// NHWC-style flat index for a 3D coordinate.
size_t FlatIndex(int i0, int i1, int i2) { return (static_cast<size_t>(i0) * kDim1 + i1) * kDim2 + i2; }

// Byte-level reference: gather along axis, out[i] = x[..., index[i], ...].
void RefGatherD(const void *x, const int32_t *index, void *out, size_t elem_size, int axis) {
  auto *src = reinterpret_cast<const uint8_t *>(x);
  auto *dst = reinterpret_cast<uint8_t *>(out);
  for (int i0 = 0; i0 < kDim0; ++i0) {
    for (int i1 = 0; i1 < kDim1; ++i1) {
      for (int i2 = 0; i2 < kDim2; ++i2) {
        const size_t out_idx = FlatIndex(i0, i1, i2);
        int gather_at = index[out_idx];
        size_t in_idx = out_idx;
        if (axis == 0) {
          in_idx = FlatIndex(gather_at, i1, i2);
        } else if (axis == 1) {
          in_idx = FlatIndex(i0, gather_at, i2);
        } else if (axis == 2) {
          in_idx = FlatIndex(i0, i1, gather_at);
        }
        std::memcpy(dst + out_idx * elem_size, src + in_idx * elem_size, elem_size);
      }
    }
  }
}
}  // namespace

class TestDSP_GatherD : public DSPCommonTest {
 public:
  void RunGatherDCase(TypeId dtype, int axis) {
    InitDSPRuntime();
    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    const size_t elem_size = lite::DataTypeSize(dtype);

    auto *param = CreateGatherDParam();
    ASSERT_NE(param, nullptr);

    auto *x = new lite::Tensor(dtype, kShape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, x->MallocData(allocator_));
    ASSERT_NE(x->MutableData(), nullptr);
    // Per-byte pattern so every cell is distinguishable.
    auto *x_bytes = reinterpret_cast<uint8_t *>(x->MutableData());
    for (size_t i = 0; i < kCells * elem_size; ++i) {
      x_bytes[i] = static_cast<uint8_t>((i * 7 + 3) & 0xFF);
    }
    inputs.push_back(x);

    auto *axis_tensor = new lite::Tensor(kNumberTypeInt32, {1}, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, axis_tensor->MallocData(allocator_));
    ASSERT_NE(axis_tensor->MutableData(), nullptr);
    *reinterpret_cast<int32_t *>(axis_tensor->MutableData()) = axis;
    inputs.push_back(axis_tensor);

    auto *indices = new lite::Tensor(kNumberTypeInt32, kShape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, indices->MallocData(allocator_));
    ASSERT_NE(indices->MutableData(), nullptr);
    auto *idx_data = reinterpret_cast<int32_t *>(indices->MutableData());
    int norm_axis = axis < 0 ? axis + 3 : axis;
    const int max_index = kShape[norm_axis];
    for (size_t i = 0; i < kCells; ++i) {
      idx_data[i] = static_cast<int32_t>(i % max_index);  // valid range [0, kShape[axis])
    }
    inputs.push_back(indices);

    auto *output = new lite::Tensor(dtype, kShape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, output->MallocData(allocator_));
    ASSERT_NE(output->MutableData(), nullptr);
    outputs.push_back(output);
    std::memset(output->MutableData(), 0, kCells * elem_size);

    std::vector<uint8_t> ref(kCells * elem_size);
    RefGatherD(x->MutableData(), idx_data, ref.data(), elem_size, norm_axis);

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, dtype, NHWC, schema::PrimitiveType_GatherD};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    ASSERT_NE(kernel, nullptr);

    ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
    ASSERT_EQ(kernel->Run(), lite::RET_OK);
    EXPECT_EQ(0, std::memcmp(output->MutableData(), ref.data(), kCells * elem_size))
      << "gather_d mismatch (dtype=" << static_cast<int>(dtype) << " axis=" << axis << ")";

    delete kernel;
    for (auto *t : inputs) {
      delete t;
    }
    for (auto *t : outputs) {
      delete t;
    }
    delete ctx;
    UninitDSPRuntime();
  }

  // Negative: indices dtype must be int32; mismatch falls back to CPU.
  void ExpectRejected(TypeId x_dtype, TypeId indices_dtype) {
    InitDSPRuntime();
    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    auto *param = CreateGatherDParam();
    ASSERT_NE(param, nullptr);
    auto *x = new lite::Tensor(x_dtype, kShape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    auto *axis_tensor = new lite::Tensor(kNumberTypeInt32, {1}, mindspore::NHWC, lite::Category::CONST_TENSOR);
    auto *indices = new lite::Tensor(indices_dtype, kShape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    auto *output = new lite::Tensor(x_dtype, kShape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, x->MallocData(allocator_));
    ASSERT_EQ(lite::RET_OK, axis_tensor->MallocData(allocator_));
    ASSERT_EQ(lite::RET_OK, indices->MallocData(allocator_));
    ASSERT_EQ(lite::RET_OK, output->MallocData(allocator_));
    inputs.push_back(x);
    inputs.push_back(axis_tensor);
    inputs.push_back(indices);
    outputs.push_back(output);

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, x_dtype, NHWC, schema::PrimitiveType_GatherD};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    EXPECT_EQ(kernel, nullptr);

    delete kernel;
    for (auto *t : inputs) {
      delete t;
    }
    for (auto *t : outputs) {
      delete t;
    }
    delete ctx;
    UninitDSPRuntime();
  }
};

TEST_F(TestDSP_GatherD, Fp32_Axis0) { RunGatherDCase(kNumberTypeFloat32, 0); }
TEST_F(TestDSP_GatherD, Fp32_Axis1) { RunGatherDCase(kNumberTypeFloat32, 1); }
TEST_F(TestDSP_GatherD, Fp32_Axis2) { RunGatherDCase(kNumberTypeFloat32, 2); }
TEST_F(TestDSP_GatherD, Fp32_NegAxisTo0) { RunGatherDCase(kNumberTypeFloat32, -3); }
TEST_F(TestDSP_GatherD, Fp32_NegAxisTo2) { RunGatherDCase(kNumberTypeFloat32, -1); }
TEST_F(TestDSP_GatherD, Fp16_Axis1) { RunGatherDCase(kNumberTypeFloat16, 1); }
TEST_F(TestDSP_GatherD, Int16_Axis1) { RunGatherDCase(kNumberTypeInt16, 1); }
TEST_F(TestDSP_GatherD, Int32_Axis1) { RunGatherDCase(kNumberTypeInt32, 1); }
TEST_F(TestDSP_GatherD, C64_Axis1) { RunGatherDCase(kNumberTypeComplex64, 1); }

TEST_F(TestDSP_GatherD, WrongIndicesDtype_FallsBackToCpu) { ExpectRejected(kNumberTypeFloat32, kNumberTypeFloat32); }

// Out-of-range indices must be rejected by CheckSpecs/Run and fall back to CPU.
TEST_F(TestDSP_GatherD, IndexOutOfRange_FallsBackToCpu) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  auto *param = CreateGatherDParam();
  ASSERT_NE(param, nullptr);
  auto *x = new lite::Tensor(kNumberTypeFloat32, kShape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  auto *axis_tensor = new lite::Tensor(kNumberTypeInt32, {1}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  auto *indices = new lite::Tensor(kNumberTypeInt32, kShape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  auto *output = new lite::Tensor(kNumberTypeFloat32, kShape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(lite::RET_OK, x->MallocData(allocator_));
  ASSERT_EQ(lite::RET_OK, axis_tensor->MallocData(allocator_));
  ASSERT_EQ(lite::RET_OK, indices->MallocData(allocator_));
  ASSERT_EQ(lite::RET_OK, output->MallocData(allocator_));
  *reinterpret_cast<int32_t *>(axis_tensor->MutableData()) = 1;
  auto *idx_data = reinterpret_cast<int32_t *>(indices->MutableData());
  for (size_t i = 0; i < kCells; ++i) {
    idx_data[i] = kDim1;  // out of range: valid is [0, kDim1)
  }
  inputs.push_back(x);
  inputs.push_back(axis_tensor);
  inputs.push_back(indices);
  outputs.push_back(output);

  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_GatherD};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  EXPECT_NE(kernel->Run(), lite::RET_OK);  // rejected -> RET_ERROR before DSP launch

  delete kernel;
  for (auto *t : inputs) {
    delete t;
  }
  for (auto *t : outputs) {
    delete t;
  }
  delete ctx;
  UninitDSPRuntime();
}

}  // namespace mindspore::lite::dsp::test

#endif  // SUPPORT_FT04
