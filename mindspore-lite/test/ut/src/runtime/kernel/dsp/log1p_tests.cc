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

#include <cassert>
#include <cmath>
#include <cstdint>
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

namespace {
constexpr int kLog1pLength = 1024;

// Load input element i as double, for any supported input dtype.
double LoadAsDouble(const void *buf, size_t i, TypeId t) {
  switch (t) {
    case kNumberTypeFloat16:
      return static_cast<double>(Fp16ToFp32(reinterpret_cast<const uint16_t *>(buf)[i]));
    case kNumberTypeFloat32:
      return static_cast<double>(reinterpret_cast<const float *>(buf)[i]);
    case kNumberTypeInt16:
      return static_cast<double>(reinterpret_cast<const int16_t *>(buf)[i]);
    case kNumberTypeInt32:
      return static_cast<double>(reinterpret_cast<const int32_t *>(buf)[i]);
    default:
      assert(false && "log1p test: unsupported input dtype");
      return 0.0;
  }
}

void StoreInput(void *buf, size_t i, TypeId t, double v) {
  switch (t) {
    case kNumberTypeFloat16:
      reinterpret_cast<uint16_t *>(buf)[i] = Fp32ToFp16(static_cast<float>(v));
      break;
    case kNumberTypeFloat32:
      reinterpret_cast<float *>(buf)[i] = static_cast<float>(v);
      break;
    case kNumberTypeInt16:
      reinterpret_cast<int16_t *>(buf)[i] = static_cast<int16_t>(v);
      break;
    case kNumberTypeInt32:
      reinterpret_cast<int32_t *>(buf)[i] = static_cast<int32_t>(v);
      break;
    default:
      assert(false && "log1p test: unsupported input dtype");
      break;
  }
}

// Reference: log(1+x). Caller asserts out_dtype is fp16 or fp32, so the ref
// buffer (allocated by DataTypeSize(out_dtype)) is always written with the
// matching element width.
void RefLog1p(const void *in, void *out, size_t n, TypeId in_dtype, TypeId out_dtype) {
  for (size_t i = 0; i < n; ++i) {
    const double expected = std::log1p(LoadAsDouble(in, i, in_dtype));
    if (out_dtype == kNumberTypeFloat16) {
      reinterpret_cast<uint16_t *>(out)[i] = Fp32ToFp16(static_cast<float>(expected));
    } else {
      assert(out_dtype == kNumberTypeFloat32 && "log1p test: output must be fp16/fp32");
      reinterpret_cast<float *>(out)[i] = static_cast<float>(expected);
    }
  }
}
}  // namespace

class TestDSP_Log1p : public DSPCommonTest {
 public:
  void RunLog1pCase(TypeId in_dtype, TypeId out_dtype) {
    ASSERT_TRUE(out_dtype == kNumberTypeFloat16 || out_dtype == kNumberTypeFloat32)
      << "log1p test: only fp16/fp32 output is supported by the reference";
    InitDSPRuntime();
    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    std::vector<int> shape = {kLog1pLength};
    const size_t n = kLog1pLength;

    auto *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
    ASSERT_NE(param, nullptr);
    std::memset(param, 0, sizeof(OpParameter));
    param->type_ = schema::PrimitiveType_Log1p;

    auto *input = new lite::Tensor(in_dtype, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, input->MallocData(allocator_));
    ASSERT_NE(input->MutableData(), nullptr);
    inputs.push_back(input);
    // Keep x in [0, 100] so log(1+x) is well-defined and precision is stable.
    for (size_t i = 0; i < n; ++i) {
      StoreInput(input->MutableData(), i, in_dtype, static_cast<double>(i % 100));
    }

    auto *output = new lite::Tensor(out_dtype, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, output->MallocData(allocator_));
    ASSERT_NE(output->MutableData(), nullptr);
    outputs.push_back(output);
    std::memset(output->MutableData(), 0, n * lite::DataTypeSize(out_dtype));

    std::vector<uint8_t> ref(n * lite::DataTypeSize(out_dtype));
    RefLog1p(input->MutableData(), ref.data(), n, in_dtype, out_dtype);

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, in_dtype, NHWC, schema::PrimitiveType_Log1p};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    ASSERT_NE(kernel, nullptr);

    ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
    ASSERT_EQ(kernel->Run(), lite::RET_OK);

    // Tolerance from on-device behavior: fp16 needs a wider bound, fp32 ~1e-4.
    const double abs_tol = (out_dtype == kNumberTypeFloat16) ? 5e-2 : 1e-4;
    for (size_t i = 0; i < n; ++i) {
      const double got =
        (out_dtype == kNumberTypeFloat16)
          ? static_cast<double>(Fp16ToFp32(reinterpret_cast<const uint16_t *>(output->MutableData())[i]))
          : static_cast<double>(reinterpret_cast<const float *>(output->MutableData())[i]);
      const double exp = (out_dtype == kNumberTypeFloat16)
                           ? static_cast<double>(Fp16ToFp32(reinterpret_cast<const uint16_t *>(ref.data())[i]))
                           : static_cast<double>(reinterpret_cast<const float *>(ref.data())[i]);
      ASSERT_LE(std::fabs(got - exp), abs_tol) << "log1p mismatch at " << i;
    }

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

  // Negative: output dtype must match the primitive's contract; mismatch falls back to CPU.
  void ExpectRejected(TypeId in_dtype, TypeId out_dtype) {
    InitDSPRuntime();
    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    std::vector<int> shape = {kLog1pLength};
    auto *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
    ASSERT_NE(param, nullptr);
    std::memset(param, 0, sizeof(OpParameter));
    param->type_ = schema::PrimitiveType_Log1p;
    auto *input = new lite::Tensor(in_dtype, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    auto *output = new lite::Tensor(out_dtype, shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, input->MallocData(allocator_));
    ASSERT_EQ(lite::RET_OK, output->MallocData(allocator_));
    inputs.push_back(input);
    outputs.push_back(output);

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, in_dtype, NHWC, schema::PrimitiveType_Log1p};
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

TEST_F(TestDSP_Log1p, Fp32) { RunLog1pCase(kNumberTypeFloat32, kNumberTypeFloat32); }
TEST_F(TestDSP_Log1p, Fp16) { RunLog1pCase(kNumberTypeFloat16, kNumberTypeFloat16); }
TEST_F(TestDSP_Log1p, Int16_ToFp32) { RunLog1pCase(kNumberTypeInt16, kNumberTypeFloat32); }
TEST_F(TestDSP_Log1p, Int32_ToFp32) { RunLog1pCase(kNumberTypeInt32, kNumberTypeFloat32); }

TEST_F(TestDSP_Log1p, Int16_WrongOutDtype_FallsBackToCpu) { ExpectRejected(kNumberTypeInt16, kNumberTypeInt16); }
TEST_F(TestDSP_Log1p, Fp32_WrongOutDtype_FallsBackToCpu) { ExpectRejected(kNumberTypeFloat32, kNumberTypeInt32); }

}  // namespace mindspore::lite::dsp::test

#endif  // SUPPORT_FT04
