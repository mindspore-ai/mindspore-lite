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
#include "nnacl_c/fill_parameter.h"

#ifdef SUPPORT_FT04

namespace mindspore::lite::dsp::test {

namespace {
constexpr int kDim0 = 8;
constexpr int kDim1 = 4;
constexpr size_t kFillCells = kDim0 * kDim1;

OpParameter *CreateFillParam(schema::PrimitiveType prim) {
  auto *param = reinterpret_cast<FillParameter *>(malloc(sizeof(FillParameter)));
  if (param == nullptr) {
    return nullptr;
  }
  std::memset(param, 0, sizeof(FillParameter));
  param->op_parameter_.type_ = prim;
  return reinterpret_cast<OpParameter *>(param);
}

// Fill the value tensor with a per-byte pattern (values need not be meaningful,
// the op only copies the whole scalar cell to every output element).
void FillValue(void *buf, size_t elem_size) {
  auto *p = reinterpret_cast<uint8_t *>(buf);
  for (size_t b = 0; b < elem_size; ++b) {
    p[b] = static_cast<uint8_t>((b + 7) & 0xFF);
  }
}
}  // namespace

class TestDSP_Fill : public DSPCommonTest {
 public:
  void RunFillCase(schema::PrimitiveType prim, TypeId dtype) {
    InitDSPRuntime();
    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    const size_t elem_size = lite::DataTypeSize(dtype);

    auto *param = CreateFillParam(prim);
    ASSERT_NE(param, nullptr);

    auto *shape_tensor = new lite::Tensor(kNumberTypeInt32, {2}, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, shape_tensor->MallocData(allocator_));
    ASSERT_NE(shape_tensor->MutableData(), nullptr);
    auto *shape_data = reinterpret_cast<int32_t *>(shape_tensor->MutableData());
    shape_data[0] = kDim0;
    shape_data[1] = kDim1;

    auto *value_tensor = new lite::Tensor(dtype, {1}, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, value_tensor->MallocData(allocator_));
    ASSERT_NE(value_tensor->MutableData(), nullptr);
    FillValue(value_tensor->MutableData(), elem_size);

    // Fill: in[0]=value, in[1]=shape; FillV2: in[0]=shape, in[1]=value.
    if (prim == schema::PrimitiveType_FillV2) {
      inputs.push_back(shape_tensor);
      inputs.push_back(value_tensor);
    } else {
      inputs.push_back(value_tensor);
      inputs.push_back(shape_tensor);
    }

    std::vector<int> out_shape = {kDim0, kDim1};
    auto *output = new lite::Tensor(dtype, out_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, output->MallocData(allocator_));
    ASSERT_NE(output->MutableData(), nullptr);
    outputs.push_back(output);
    std::memset(output->MutableData(), 0, kFillCells * elem_size);

    std::vector<uint8_t> ref(kFillCells * elem_size);
    for (size_t i = 0; i < kFillCells; ++i) {
      std::memcpy(ref.data() + i * elem_size, value_tensor->MutableData(), elem_size);
    }

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, dtype, NHWC, prim};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    ASSERT_NE(kernel, nullptr);

    ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
    ASSERT_EQ(kernel->Run(), lite::RET_OK);
    EXPECT_EQ(0, std::memcmp(output->MutableData(), ref.data(), kFillCells * elem_size))
      << "fill output mismatch (prim=" << static_cast<int>(prim) << " dtype=" << static_cast<int>(dtype) << ")";

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

  // Negative: value tensor dtype mismatch must be rejected by CheckSpecs.
  void ExpectRejected(schema::PrimitiveType prim, TypeId out_dtype, TypeId value_dtype, int value_elems = 1) {
    InitDSPRuntime();
    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    auto *param = CreateFillParam(prim);
    ASSERT_NE(param, nullptr);
    auto *shape_tensor = new lite::Tensor(kNumberTypeInt32, {2}, mindspore::NHWC, lite::Category::CONST_TENSOR);
    auto *value_tensor = new lite::Tensor(value_dtype, {value_elems}, mindspore::NHWC, lite::Category::CONST_TENSOR);
    auto *output = new lite::Tensor(out_dtype, {kDim0, kDim1}, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, shape_tensor->MallocData(allocator_));
    ASSERT_EQ(lite::RET_OK, value_tensor->MallocData(allocator_));
    ASSERT_EQ(lite::RET_OK, output->MallocData(allocator_));
    if (prim == schema::PrimitiveType_FillV2) {
      inputs.push_back(shape_tensor);
      inputs.push_back(value_tensor);
    } else {
      inputs.push_back(value_tensor);
      inputs.push_back(shape_tensor);
    }
    outputs.push_back(output);

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, out_dtype, NHWC, prim};
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

// Both PrimitiveType_Fill and PrimitiveType_FillV2 share the same kernel.
TEST_F(TestDSP_Fill, Fill_Fp32) { RunFillCase(schema::PrimitiveType_Fill, kNumberTypeFloat32); }
TEST_F(TestDSP_Fill, Fill_Fp16) { RunFillCase(schema::PrimitiveType_Fill, kNumberTypeFloat16); }
TEST_F(TestDSP_Fill, Fill_Int16) { RunFillCase(schema::PrimitiveType_Fill, kNumberTypeInt16); }
TEST_F(TestDSP_Fill, Fill_Int32) { RunFillCase(schema::PrimitiveType_Fill, kNumberTypeInt32); }
TEST_F(TestDSP_Fill, Fill_C64) { RunFillCase(schema::PrimitiveType_Fill, kNumberTypeComplex64); }
TEST_F(TestDSP_Fill, FillV2_Fp32) { RunFillCase(schema::PrimitiveType_FillV2, kNumberTypeFloat32); }
TEST_F(TestDSP_Fill, FillV2_Fp16) { RunFillCase(schema::PrimitiveType_FillV2, kNumberTypeFloat16); }
TEST_F(TestDSP_Fill, FillV2_Int16) { RunFillCase(schema::PrimitiveType_FillV2, kNumberTypeInt16); }
TEST_F(TestDSP_Fill, FillV2_Int32) { RunFillCase(schema::PrimitiveType_FillV2, kNumberTypeInt32); }
TEST_F(TestDSP_Fill, FillV2_C64) { RunFillCase(schema::PrimitiveType_FillV2, kNumberTypeComplex64); }

TEST_F(TestDSP_Fill, Fill_ValueDtypeMismatch_FallsBackToCpu) {
  ExpectRejected(schema::PrimitiveType_Fill, kNumberTypeFloat32, kNumberTypeFloat16);
}
TEST_F(TestDSP_Fill, FillV2_ValueNotScalar_FallsBackToCpu) {
  ExpectRejected(schema::PrimitiveType_FillV2, kNumberTypeFloat32, kNumberTypeFloat32, 2);
}

// Wrong layout for Fill (FillV2 style: shape@0, value@1) must be rejected,
// proving the kernel picks indices by primitive type.
TEST_F(TestDSP_Fill, Fill_WrongLayout_FallsBackToCpu) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  auto *param = CreateFillParam(schema::PrimitiveType_Fill);
  ASSERT_NE(param, nullptr);
  auto *shape_tensor = new lite::Tensor(kNumberTypeInt32, {2}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  auto *value_tensor = new lite::Tensor(kNumberTypeFloat32, {1}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  auto *output = new lite::Tensor(kNumberTypeFloat32, {kDim0, kDim1}, mindspore::NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(lite::RET_OK, shape_tensor->MallocData(allocator_));
  ASSERT_EQ(lite::RET_OK, value_tensor->MallocData(allocator_));
  ASSERT_EQ(lite::RET_OK, output->MallocData(allocator_));
  inputs.push_back(shape_tensor);  // wrong: Fill expects value at in[0]
  inputs.push_back(value_tensor);
  outputs.push_back(output);

  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Fill};
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

}  // namespace mindspore::lite::dsp::test

#endif  // SUPPORT_FT04
