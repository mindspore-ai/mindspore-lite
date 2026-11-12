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

#ifdef SUPPORT_FT04

namespace mindspore::lite::dsp::test {

namespace {
constexpr int kCastLength = 1024;
constexpr int kComplexComponents = 2;
constexpr int kImagOffset = 1;
constexpr int kInputValueModulus = 41;
constexpr int kInputValueCenter = 20;
constexpr double kFractionalOffset = 0.5;
// DSP Cast follows C explicit-conversion semantics: integer narrowing truncates
// (implementation-defined for out-of-range), fp16 uses round-to-nearest-even.
// Reference values below are given explicitly for boundary cases; do not rely
// on static_cast of out-of-range doubles.
// fp16 tolerance: 1 ULP of the half value, expressed via Fp16ToFp32 differ.
constexpr float kFp32ErrBound = 1e-4f;
// Boundary inputs for fp32 -> fp16. Values are chosen so the IEEE
// round-to-nearest-even result is unambiguous; reference values are given
// explicitly via Fp32ToFp16, not via out-of-range static_cast (which is UB).
constexpr int kBoundaryLength = 8;
const float kBoundaryInputs[kBoundaryLength] = {0.0f, -1.0f, 1.0f, -0.5f, 0.5f, -100.0f, 100.0f, 65504.0f};
constexpr int kUnsupportedDstBool = 30;
constexpr int kUnsupportedDstInt64 = 35;

bool IsComplex(TypeId t) { return t == kNumberTypeComplex64; }

double LoadAsDouble(const void *buf, size_t i, TypeId t) {
  switch (t) {
    case kNumberTypeInt16:
      return static_cast<double>(reinterpret_cast<const int16_t *>(buf)[i]);
    case kNumberTypeInt32:
      return static_cast<double>(reinterpret_cast<const int32_t *>(buf)[i]);
    case kNumberTypeFloat16:
      return static_cast<double>(Fp16ToFp32(reinterpret_cast<const uint16_t *>(buf)[i]));
    case kNumberTypeFloat32:
      return static_cast<double>(reinterpret_cast<const float *>(buf)[i]);
    case kNumberTypeComplex64:
      return static_cast<double>(reinterpret_cast<const float *>(buf)[i * kComplexComponents]);
    default:
      return 0.0;
  }
}

void StoreFromDouble(void *buf, size_t i, TypeId t, double v) {
  switch (t) {
    case kNumberTypeInt16:
      reinterpret_cast<int16_t *>(buf)[i] = static_cast<int16_t>(v);
      break;
    case kNumberTypeInt32:
      reinterpret_cast<int32_t *>(buf)[i] = static_cast<int32_t>(v);
      break;
    case kNumberTypeFloat16:
      reinterpret_cast<uint16_t *>(buf)[i] = Fp32ToFp16(static_cast<float>(v));
      break;
    case kNumberTypeFloat32:
      reinterpret_cast<float *>(buf)[i] = static_cast<float>(v);
      break;
    case kNumberTypeComplex64: {
      auto *p = reinterpret_cast<float *>(buf);
      p[i * kComplexComponents] = static_cast<float>(v);
      p[i * kComplexComponents + kImagOffset] = 0.0f;
      break;
    }
    default:
      break;
  }
}

void FillInput(void *buf, size_t n, TypeId t) {
  for (size_t i = 0; i < n; ++i) {
    double v = static_cast<double>(static_cast<int>(i % kInputValueModulus) - kInputValueCenter);
    if (t == kNumberTypeFloat16 || t == kNumberTypeFloat32 || t == kNumberTypeComplex64) {
      v += kFractionalOffset;
    }
    StoreFromDouble(buf, i, t, v);
    if (IsComplex(t)) {
      reinterpret_cast<float *>(buf)[i * kComplexComponents + kImagOffset] = static_cast<float>(-v);
    }
  }
}

void RefCast(const void *in, void *out, size_t n, TypeId src, TypeId dst) {
  if (src == kNumberTypeFloat32 && dst == kNumberTypeComplex64) {
    const auto *s = reinterpret_cast<const float *>(in);
    auto *d = reinterpret_cast<float *>(out);
    for (size_t i = 0; i < n; ++i) {
      d[i * kComplexComponents] = s[i];
      d[i * kComplexComponents + kImagOffset] = 0.0f;
    }
    return;
  }
  if (src == kNumberTypeComplex64 && dst == kNumberTypeFloat32) {
    const auto *s = reinterpret_cast<const float *>(in);
    auto *d = reinterpret_cast<float *>(out);
    for (size_t i = 0; i < n; ++i) {
      d[i] = s[i * kComplexComponents];
    }
    return;
  }
  for (size_t i = 0; i < n; ++i) {
    StoreFromDouble(out, i, dst, LoadAsDouble(in, i, src));
  }
}

// One ULP tolerance for fp16: compare the decoded floats, allowing a single
// half-precision step difference (the DSP may accumulate in half precision).
bool Fp16WithinOneUlp(uint16_t got, uint16_t ref) {
  if (got == ref) {
    return true;
  }
  const float g = Fp16ToFp32(got);
  const float r = Fp16ToFp32(ref);
  const float next = Fp16ToFp32(static_cast<uint16_t>(ref + 1));
  const float prev = Fp16ToFp32(static_cast<uint16_t>(ref - 1));
  return g == r || g == next || g == prev;
}

int CompareCast(const void *got, const void *ref, size_t n, TypeId dst) {
  for (size_t i = 0; i < n; ++i) {
    if (dst == kNumberTypeFloat16) {
      const auto g = reinterpret_cast<const uint16_t *>(got)[i];
      const auto r = reinterpret_cast<const uint16_t *>(ref)[i];
      if (!Fp16WithinOneUlp(g, r)) {
        return 1;
      }
      continue;
    }
    double g = LoadAsDouble(got, i, dst);
    double r = LoadAsDouble(ref, i, dst);
    if (std::fabs(g - r) > kFp32ErrBound) {
      return 1;
    }
  }
  if (IsComplex(dst)) {
    const auto *g = reinterpret_cast<const float *>(got);
    const auto *r = reinterpret_cast<const float *>(ref);
    for (size_t i = 0; i < n; ++i) {
      if (std::fabs(g[i * kComplexComponents + kImagOffset] - r[i * kComplexComponents + kImagOffset]) >
          kFp32ErrBound) {
        return 1;
      }
    }
  }
  return 0;
}
}  // namespace

class TestDSP_Cast : public DSPCommonTest {
 public:
  // An unsupported (src,dst) pair must be rejected by CheckSpecs so the
  // scheduler falls back to CPU (creator returns nullptr), instead of trying
  // to launch a non-existent DSP kernel.
  void ExpectCpuFallback(TypeId src, TypeId dst) {
    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    std::vector<int> shape = {kCastLength};
    auto *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
    ASSERT_NE(param, nullptr);
    std::memset(param, 0, sizeof(OpParameter));
    param->type_ = schema::PrimitiveType_Cast;

    auto *input = new lite::Tensor(src, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    auto *output = new lite::Tensor(dst, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    inputs.push_back(input);
    outputs.push_back(output);

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, src, NHWC, schema::PrimitiveType_Cast};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    EXPECT_EQ(kernel, nullptr);

    delete ctx;
    for (auto *t : inputs) {
      delete t;
    }
    for (auto *t : outputs) {
      delete t;
    }
    delete kernel;
  }

  // fp32 -> fp16 boundary values with an explicit reference.
  void RunBoundaryCase() {
    InitDSPRuntime();
    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    std::vector<int> shape = {kBoundaryLength};
    const size_t n = kBoundaryLength;

    auto *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
    ASSERT_NE(param, nullptr);
    std::memset(param, 0, sizeof(OpParameter));
    param->type_ = schema::PrimitiveType_Cast;

    auto *input = new lite::Tensor(kNumberTypeFloat32, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, input->MallocData(allocator_));
    inputs.push_back(input);
    auto *in_data = reinterpret_cast<float *>(input->MutableData());
    for (size_t i = 0; i < n; ++i) {
      in_data[i] = kBoundaryInputs[i];
    }

    auto *output = new lite::Tensor(kNumberTypeFloat16, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, output->MallocData(allocator_));
    outputs.push_back(output);

    std::vector<uint16_t> ref(n);
    for (size_t i = 0; i < n; ++i) {
      ref[i] = Fp32ToFp16(kBoundaryInputs[i]);
    }

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Cast};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
    ASSERT_EQ(kernel->Run(), lite::RET_OK);
    EXPECT_EQ(0, CompareCast(output->MutableData(), ref.data(), n, kNumberTypeFloat16));

    UninitDSPRuntime();
    delete ctx;
    for (auto *t : inputs) {
      delete t;
    }
    for (auto *t : outputs) {
      delete t;
    }
    delete kernel;
  }

  void RunCastCase(TypeId src, TypeId dst) {
    InitDSPRuntime();
    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    std::vector<int> shape = {kCastLength};
    size_t n = kCastLength;

    auto *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
    ASSERT_NE(param, nullptr);
    std::memset(param, 0, sizeof(OpParameter));
    param->type_ = schema::PrimitiveType_Cast;

    auto *input = new lite::Tensor(src, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, input->MallocData(allocator_));
    ASSERT_NE(input->MutableData(), nullptr);
    inputs.push_back(input);
    FillInput(input->MutableData(), n, src);

    auto *output = new lite::Tensor(dst, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, output->MallocData(allocator_));
    ASSERT_NE(output->MutableData(), nullptr);
    outputs.push_back(output);
    std::memset(output->MutableData(), 0, n * lite::DataTypeSize(dst));

    std::vector<uint8_t> ref(n * lite::DataTypeSize(dst));
    RefCast(input->MutableData(), ref.data(), n, src, dst);

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());

    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, src, NHWC, schema::PrimitiveType_Cast};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    ASSERT_NE(kernel, nullptr);

    ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
    ASSERT_EQ(kernel->Run(), lite::RET_OK);
    EXPECT_EQ(0, CompareCast(output->MutableData(), ref.data(), n, dst));

    UninitDSPRuntime();
    delete ctx;
    for (auto *t : inputs) {
      delete t;
    }
    for (auto *t : outputs) {
      delete t;
    }
    delete kernel;
  }
};

// Common conversions (int16 / int32 / fp32 mutual and fp32 <-> complex64).
TEST_F(TestDSP_Cast, I16_To_I32) { RunCastCase(kNumberTypeInt16, kNumberTypeInt32); }
TEST_F(TestDSP_Cast, I16_To_Fp32) { RunCastCase(kNumberTypeInt16, kNumberTypeFloat32); }
TEST_F(TestDSP_Cast, I32_To_I16) { RunCastCase(kNumberTypeInt32, kNumberTypeInt16); }
TEST_F(TestDSP_Cast, I32_To_Fp32) { RunCastCase(kNumberTypeInt32, kNumberTypeFloat32); }
TEST_F(TestDSP_Cast, Fp32_To_I16) { RunCastCase(kNumberTypeFloat32, kNumberTypeInt16); }
TEST_F(TestDSP_Cast, Fp32_To_I32) { RunCastCase(kNumberTypeFloat32, kNumberTypeInt32); }
TEST_F(TestDSP_Cast, Fp32_To_C64) { RunCastCase(kNumberTypeFloat32, kNumberTypeComplex64); }
TEST_F(TestDSP_Cast, C64_To_Fp32) { RunCastCase(kNumberTypeComplex64, kNumberTypeFloat32); }

// FT04 only: fp16 related conversions.
TEST_F(TestDSP_Cast, I16_To_Fp16) { RunCastCase(kNumberTypeInt16, kNumberTypeFloat16); }
TEST_F(TestDSP_Cast, I32_To_Fp16) { RunCastCase(kNumberTypeInt32, kNumberTypeFloat16); }
TEST_F(TestDSP_Cast, Fp16_To_I16) { RunCastCase(kNumberTypeFloat16, kNumberTypeInt16); }
TEST_F(TestDSP_Cast, Fp16_To_I32) { RunCastCase(kNumberTypeFloat16, kNumberTypeInt32); }
TEST_F(TestDSP_Cast, Fp16_To_Fp32) { RunCastCase(kNumberTypeFloat16, kNumberTypeFloat32); }
TEST_F(TestDSP_Cast, Fp32_To_Fp16) { RunCastCase(kNumberTypeFloat32, kNumberTypeFloat16); }

// Negative cases: combinations outside the supported (src,dst) matrix.
TEST_F(TestDSP_Cast, Fp32_To_Bool_FallsBackToCpu) {
  ExpectCpuFallback(kNumberTypeFloat32, static_cast<TypeId>(kUnsupportedDstBool));
}
TEST_F(TestDSP_Cast, I32_To_Int64_FallsBackToCpu) {
  ExpectCpuFallback(kNumberTypeInt32, static_cast<TypeId>(kUnsupportedDstInt64));
}

// Boundary values with explicit reference (no UB from out-of-range static_cast).
TEST_F(TestDSP_Cast, Fp32_To_Fp16_Boundary) { RunBoundaryCase(); }

}  // namespace mindspore::lite::dsp::test

#endif  // SUPPORT_FT04
