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
#include "nnacl_c/triu_tril_parameter.h"

#ifdef SUPPORT_FT04

namespace mindspore::lite::dsp::test {

namespace {
constexpr int kBatch = 2;
constexpr int kHeight = 4;
constexpr int kWidth = 4;

// Fill each matrix cell with a per-cell byte pattern (values need not be meaningful,
// the op only copies or zeroes whole cells).
void FillInput(void *buf, size_t cells, size_t elem_size) {
  auto *p = reinterpret_cast<uint8_t *>(buf);
  for (size_t i = 0; i < cells; ++i) {
    for (size_t b = 0; b < elem_size; ++b) {
      p[i * elem_size + b] = static_cast<uint8_t>((i + b) & 0xFF);
    }
  }
}

// Byte-level reference: keep cell if (tril: j <= i+k) / (triu: j >= i+k), else zero.
void RefTriuTril(const void *in, void *out, int k, size_t elem_size, int num_matrices, bool is_tril) {
  const auto *src = reinterpret_cast<const uint8_t *>(in);
  auto *dst = reinterpret_cast<uint8_t *>(out);
  const size_t matrix_bytes = static_cast<size_t>(kHeight) * kWidth * elem_size;
  for (int m = 0; m < num_matrices; ++m) {
    const uint8_t *s = src + m * matrix_bytes;
    uint8_t *d = dst + m * matrix_bytes;
    for (int i = 0; i < kHeight; ++i) {
      for (int j = 0; j < kWidth; ++j) {
        const size_t off = (static_cast<size_t>(i) * kWidth + j) * elem_size;
        const bool keep = is_tril ? (j <= i + k) : (j >= i + k);
        if (keep) {
          std::memcpy(d + off, s + off, elem_size);
        } else {
          std::memset(d + off, 0, elem_size);
        }
      }
    }
  }
}

OpParameter *CreateParam(bool is_tril) {
  OpParameter *param = nullptr;
  if (is_tril) {
    auto *p = reinterpret_cast<TrilParameter *>(malloc(sizeof(TrilParameter)));
    if (p == nullptr) {
      return nullptr;
    }
    std::memset(p, 0, sizeof(TrilParameter));
    p->op_parameter_.type_ = schema::PrimitiveType_Tril;
    param = reinterpret_cast<OpParameter *>(p);
  } else {
    auto *p = reinterpret_cast<TriuParameter *>(malloc(sizeof(TriuParameter)));
    if (p == nullptr) {
      return nullptr;
    }
    std::memset(p, 0, sizeof(TriuParameter));
    p->op_parameter_.type_ = schema::PrimitiveType_Triu;
    param = reinterpret_cast<OpParameter *>(p);
  }
  return param;
}
}  // namespace

class TestDSP_TriuTril : public DSPCommonTest {
 public:
  // has_k=false -> single data input (k defaults to 0); has_k=true -> data + int k tensor.
  void RunCase(bool is_tril, TypeId dtype, int k, bool has_k, TypeId k_dtype = kNumberTypeInt32,
               std::vector<int> shape = {kBatch, kHeight, kWidth}) {
    InitDSPRuntime();
    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    size_t cells = 1;
    for (int d : shape) {
      cells *= static_cast<size_t>(d);
    }
    const size_t elem_size = lite::DataTypeSize(dtype);
    const auto prim = is_tril ? schema::PrimitiveType_Tril : schema::PrimitiveType_Triu;

    auto *param = CreateParam(is_tril);
    ASSERT_NE(param, nullptr);

    auto *input = new lite::Tensor(dtype, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, input->MallocData(allocator_));
    ASSERT_NE(input->MutableData(), nullptr);
    inputs.push_back(input);
    FillInput(input->MutableData(), cells, elem_size);

    if (has_k) {
      auto *k_tensor = new lite::Tensor(k_dtype, {1}, mindspore::NHWC, lite::Category::CONST_TENSOR);
      ASSERT_EQ(lite::RET_OK, k_tensor->MallocData(allocator_));
      ASSERT_NE(k_tensor->MutableData(), nullptr);
      if (k_dtype == kNumberTypeInt64) {
        *reinterpret_cast<int64_t *>(k_tensor->MutableData()) = k;
      } else {
        *reinterpret_cast<int32_t *>(k_tensor->MutableData()) = k;
      }
      inputs.push_back(k_tensor);
    }

    auto *output = new lite::Tensor(dtype, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, output->MallocData(allocator_));
    ASSERT_NE(output->MutableData(), nullptr);
    outputs.push_back(output);
    std::memset(output->MutableData(), 0, cells * elem_size);

    std::vector<uint8_t> ref(cells * elem_size);
    const int num_matrices = static_cast<int>(cells) / (kHeight * kWidth);
    RefTriuTril(input->MutableData(), ref.data(), has_k ? k : 0, elem_size, num_matrices, is_tril);

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, dtype, NHWC, prim};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    ASSERT_NE(kernel, nullptr);

    ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
    ASSERT_EQ(kernel->Run(), lite::RET_OK);
    EXPECT_EQ(0, std::memcmp(output->MutableData(), ref.data(), cells * elem_size))
      << (is_tril ? "tril" : "triu") << " output mismatch dtype=" << static_cast<int>(dtype) << " k=" << k;

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

  // Negative case: creator must reject (return nullptr) so the op falls back to CPU.
  void ExpectRejected(bool is_tril, const std::vector<int> &shape, TypeId k_dtype, int k_elems = 1) {
    InitDSPRuntime();
    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    const auto prim = is_tril ? schema::PrimitiveType_Tril : schema::PrimitiveType_Triu;
    auto *param = CreateParam(is_tril);
    ASSERT_NE(param, nullptr);
    auto *input = new lite::Tensor(kNumberTypeFloat32, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    auto *k_tensor = new lite::Tensor(k_dtype, {k_elems}, mindspore::NHWC, lite::Category::CONST_TENSOR);
    auto *output = new lite::Tensor(kNumberTypeFloat32, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    inputs.push_back(input);
    inputs.push_back(k_tensor);
    outputs.push_back(output);

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, prim};
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

// ---- Tril ----
TEST_F(TestDSP_TriuTril, Tril_Fp32_K0) { RunCase(true, kNumberTypeFloat32, 0, true); }
TEST_F(TestDSP_TriuTril, Tril_Fp32_KPos) { RunCase(true, kNumberTypeFloat32, 1, true); }
TEST_F(TestDSP_TriuTril, Tril_Fp32_KNeg) { RunCase(true, kNumberTypeFloat32, -1, true); }
TEST_F(TestDSP_TriuTril, Tril_Fp32_NoK) { RunCase(true, kNumberTypeFloat32, 0, false); }
TEST_F(TestDSP_TriuTril, Tril_Fp16_K0) { RunCase(true, kNumberTypeFloat16, 0, true); }
TEST_F(TestDSP_TriuTril, Tril_Int16_K0) { RunCase(true, kNumberTypeInt16, 0, true); }
TEST_F(TestDSP_TriuTril, Tril_Int32_K0) { RunCase(true, kNumberTypeInt32, 0, true); }
TEST_F(TestDSP_TriuTril, Tril_C64_K0) { RunCase(true, kNumberTypeComplex64, 0, true); }
TEST_F(TestDSP_TriuTril, Tril_Fp32_K64_Pos) { RunCase(true, kNumberTypeFloat32, 1, true, kNumberTypeInt64); }
TEST_F(TestDSP_TriuTril, Tril_Fp32_K64_Neg) { RunCase(true, kNumberTypeFloat32, -1, true, kNumberTypeInt64); }
TEST_F(TestDSP_TriuTril, Tril_Fp32_Rank2) {
  RunCase(true, kNumberTypeFloat32, 0, true, kNumberTypeInt32, {kHeight, kWidth});
}
TEST_F(TestDSP_TriuTril, Tril_InvalidShape_FallsBackToCpu) {
  ExpectRejected(true, {kHeight * kWidth}, kNumberTypeInt32);
}
TEST_F(TestDSP_TriuTril, Tril_InvalidKDtype_FallsBackToCpu) {
  ExpectRejected(true, {kBatch, kHeight, kWidth}, kNumberTypeFloat32);
}
TEST_F(TestDSP_TriuTril, Tril_InvalidKElems_FallsBackToCpu) {
  ExpectRejected(true, {kBatch, kHeight, kWidth}, kNumberTypeInt32, 2);
}

// ---- Triu ----
TEST_F(TestDSP_TriuTril, Triu_Fp32_K0) { RunCase(false, kNumberTypeFloat32, 0, true); }
TEST_F(TestDSP_TriuTril, Triu_Fp32_KPos) { RunCase(false, kNumberTypeFloat32, 1, true); }
TEST_F(TestDSP_TriuTril, Triu_Fp32_KNeg) { RunCase(false, kNumberTypeFloat32, -1, true); }
TEST_F(TestDSP_TriuTril, Triu_Fp32_NoK) { RunCase(false, kNumberTypeFloat32, 0, false); }
TEST_F(TestDSP_TriuTril, Triu_Fp16_K0) { RunCase(false, kNumberTypeFloat16, 0, true); }
TEST_F(TestDSP_TriuTril, Triu_Int16_K0) { RunCase(false, kNumberTypeInt16, 0, true); }
TEST_F(TestDSP_TriuTril, Triu_Int32_K0) { RunCase(false, kNumberTypeInt32, 0, true); }
TEST_F(TestDSP_TriuTril, Triu_C64_K0) { RunCase(false, kNumberTypeComplex64, 0, true); }
TEST_F(TestDSP_TriuTril, Triu_Fp32_K64_Pos) { RunCase(false, kNumberTypeFloat32, 1, true, kNumberTypeInt64); }
TEST_F(TestDSP_TriuTril, Triu_Fp32_K64_Neg) { RunCase(false, kNumberTypeFloat32, -1, true, kNumberTypeInt64); }
TEST_F(TestDSP_TriuTril, Triu_Fp32_Rank2) {
  RunCase(false, kNumberTypeFloat32, 0, true, kNumberTypeInt32, {kHeight, kWidth});
}
TEST_F(TestDSP_TriuTril, Triu_InvalidShape_FallsBackToCpu) {
  ExpectRejected(false, {kHeight * kWidth}, kNumberTypeInt32);
}
TEST_F(TestDSP_TriuTril, Triu_InvalidKDtype_FallsBackToCpu) {
  ExpectRejected(false, {kBatch, kHeight, kWidth}, kNumberTypeFloat32);
}
TEST_F(TestDSP_TriuTril, Triu_InvalidKElems_FallsBackToCpu) {
  ExpectRejected(false, {kBatch, kHeight, kWidth}, kNumberTypeInt32, 2);
}

}  // namespace mindspore::lite::dsp::test

#endif  // SUPPORT_FT04
