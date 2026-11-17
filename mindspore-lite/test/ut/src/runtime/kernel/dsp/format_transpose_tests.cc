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
#include "src/litert/kernel/dsp/kernel/format_transpose.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/cpu/nnacl_c/format_transpose_parameter.h"

#ifdef SUPPORT_FT04

namespace mindspore::lite::dsp::test {

namespace {
constexpr int kBatch = 2;
constexpr int kHeight = 3;
constexpr int kWidth = 4;
constexpr int kChannel = 5;
constexpr int kPlane = kHeight * kWidth;
constexpr size_t kElementCount = kBatch * kChannel * kPlane;
constexpr int kFormatNchw = static_cast<int>(Format_NCHW);
constexpr int kFormatNhwc = static_cast<int>(Format_NHWC);
constexpr uint8_t kByteSeed = 11;
constexpr uint8_t kByteStride = 37;
constexpr uint8_t kOutputSentinel = 0xAB;
// Sentinel for an out-of-range format value used in negative cases.
constexpr int kUnsupportedFormat = 13;

struct TransposeCase {
  TypeId dtype;
  int src_format;
  int dst_format;
};

std::vector<int> SrcShape(int src_format) {
  if (src_format == kFormatNhwc) {
    return {kBatch, kHeight, kWidth, kChannel};
  }
  return {kBatch, kChannel, kHeight, kWidth};
}

std::vector<int> DstShape(int dst_format) {
  if (dst_format == kFormatNhwc) {
    return {kBatch, kHeight, kWidth, kChannel};
  }
  return {kBatch, kChannel, kHeight, kWidth};
}

mindspore::Format ToFormat(int format) { return format == kFormatNhwc ? mindspore::NHWC : mindspore::NCHW; }

FormatTransposeParameter *CreateParam(int src_format, int dst_format) {
  auto *param = reinterpret_cast<FormatTransposeParameter *>(malloc(sizeof(FormatTransposeParameter)));
  if (param == nullptr) {
    return nullptr;
  }
  std::memset(param, 0, sizeof(FormatTransposeParameter));
  param->op_parameter_.type_ = schema::PrimitiveType_FormatTranspose;
  param->src_format_ = static_cast<FormatC>(src_format);
  param->dst_format_ = static_cast<FormatC>(dst_format);
  return param;
}

// The DSP kernel only permutes whole elements, so the reference is a pure
// byte-level gather. Fill every element with a distinct byte pattern so any
// wrong (batch, channel, plane) mapping is detected.
void FillInput(void *buf, size_t elem_size) {
  auto *bytes = reinterpret_cast<uint8_t *>(buf);
  for (size_t i = 0; i < kElementCount; ++i) {
    for (size_t b = 0; b < elem_size; ++b) {
      bytes[i * elem_size + b] = static_cast<uint8_t>(kByteSeed + i * kByteStride + b);
    }
  }
}

void BuildReference(const void *in, void *ref, size_t elem_size, int src_format) {
  const auto *src = reinterpret_cast<const uint8_t *>(in);
  auto *dst = reinterpret_cast<uint8_t *>(ref);
  for (int n = 0; n < kBatch; ++n) {
    for (int c = 0; c < kChannel; ++c) {
      for (int p = 0; p < kPlane; ++p) {
        size_t si = 0;
        size_t di = 0;
        if (src_format == kFormatNhwc) {
          si = (static_cast<size_t>(n) * kPlane + p) * kChannel + c;
          di = (static_cast<size_t>(n) * kChannel + c) * kPlane + p;
        } else {
          si = (static_cast<size_t>(n) * kChannel + c) * kPlane + p;
          di = (static_cast<size_t>(n) * kPlane + p) * kChannel + c;
        }
        std::memcpy(dst + di * elem_size, src + si * elem_size, elem_size);
      }
    }
  }
}
}  // namespace

class TestDSP_FormatTranspose : public DSPCommonTest {
 public:
  void RunTransposeCase(const TransposeCase &tc) {
    InitDSPRuntime();
    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    const size_t elem_size = lite::DataTypeSize(tc.dtype);

    auto *param = CreateParam(tc.src_format, tc.dst_format);
    ASSERT_NE(param, nullptr);

    auto *input =
      new lite::Tensor(tc.dtype, SrcShape(tc.src_format), ToFormat(tc.src_format), lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, input->MallocData(allocator_));
    ASSERT_NE(input->MutableData(), nullptr);
    FillInput(input->MutableData(), elem_size);
    inputs.push_back(input);

    auto *output =
      new lite::Tensor(tc.dtype, DstShape(tc.dst_format), ToFormat(tc.dst_format), lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, output->MallocData(allocator_));
    ASSERT_NE(output->MutableData(), nullptr);
    std::memset(output->MutableData(), kOutputSentinel, kElementCount * elem_size);
    outputs.push_back(output);

    std::vector<uint8_t> ref(kElementCount * elem_size);
    BuildReference(input->MutableData(), ref.data(), elem_size, tc.src_format);

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, tc.dtype, NHWC, schema::PrimitiveType_FormatTranspose};
    auto creator = KernelRegistry::GetInstance()->GetCreator(key);
    ASSERT_NE(creator, nullptr);
    auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
    ASSERT_NE(kernel, nullptr);

    ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
    ASSERT_EQ(kernel->Run(), lite::RET_OK);
    EXPECT_EQ(0, std::memcmp(output->MutableData(), ref.data(), kElementCount * elem_size))
      << "format_transpose output mismatch (dtype=" << static_cast<int>(tc.dtype) << " src=" << tc.src_format
      << " dst=" << tc.dst_format << ")";

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

  // Unsupported format pairs / dtypes must be rejected by CheckSpecs so the
  // creator returns nullptr (scheduler then falls back to CPU). This only
  // verifies the rejection; the actual CPU fallback is a framework path.
  void ExpectRejectedByCheckSpecs(int src_format, int dst_format, TypeId dtype) {
    InitDSPRuntime();
    std::vector<lite::Tensor *> inputs;
    std::vector<lite::Tensor *> outputs;
    auto *param = CreateParam(src_format, dst_format);
    ASSERT_NE(param, nullptr);

    auto *input = new lite::Tensor(dtype, SrcShape(src_format), ToFormat(src_format), lite::Category::CONST_TENSOR);
    auto *output = new lite::Tensor(dtype, DstShape(dst_format), ToFormat(dst_format), lite::Category::CONST_TENSOR);
    ASSERT_EQ(lite::RET_OK, input->MallocData(allocator_));
    ASSERT_EQ(lite::RET_OK, output->MallocData(allocator_));
    inputs.push_back(input);
    outputs.push_back(output);

    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());
    kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, dtype, NHWC, schema::PrimitiveType_FormatTranspose};
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

  // Bypass DSPKernelCreator (whose -1-shape early return skips CheckSpecs) and
  // construct the kernel directly, to cover the Run() side re-validation.
  // Run() calls CheckSpecs() before touching device memory, so no DSP runtime
  // or device buffer is needed for these rejection paths.
  void ExpectRunRejected(int src_format, int dst_format, const std::vector<int> &in_shape,
                         const std::vector<int> &out_shape) {
    auto *param = CreateParam(src_format, dst_format);
    ASSERT_NE(param, nullptr);
    std::vector<lite::Tensor *> inputs = {
      new lite::Tensor(kNumberTypeFloat32, in_shape, ToFormat(src_format), lite::Category::CONST_TENSOR)};
    std::vector<lite::Tensor *> outputs = {
      new lite::Tensor(kNumberTypeFloat32, out_shape, ToFormat(dst_format), lite::Category::CONST_TENSOR)};
    auto *ctx = new lite::InnerContext;
    ASSERT_EQ(lite::RET_OK, ctx->Init());

    kernel::FormatTransposeDSPKernel kernel(reinterpret_cast<OpParameter *>(param), inputs, outputs, ctx);
    EXPECT_EQ(kernel.Run(), lite::RET_ERROR);

    for (auto *t : inputs) {
      delete t;
    }
    for (auto *t : outputs) {
      delete t;
    }
    delete ctx;
  }
};

// FT04 supports fp32, int16, int32, complex64 and fp16, in both directions.
TEST_F(TestDSP_FormatTranspose, Fp32_NhwcToNchw) { RunTransposeCase({kNumberTypeFloat32, kFormatNhwc, kFormatNchw}); }
TEST_F(TestDSP_FormatTranspose, Fp32_NchwToNhwc) { RunTransposeCase({kNumberTypeFloat32, kFormatNchw, kFormatNhwc}); }
TEST_F(TestDSP_FormatTranspose, Fp16_NhwcToNchw) { RunTransposeCase({kNumberTypeFloat16, kFormatNhwc, kFormatNchw}); }
TEST_F(TestDSP_FormatTranspose, Fp16_NchwToNhwc) { RunTransposeCase({kNumberTypeFloat16, kFormatNchw, kFormatNhwc}); }
TEST_F(TestDSP_FormatTranspose, I16_NhwcToNchw) { RunTransposeCase({kNumberTypeInt16, kFormatNhwc, kFormatNchw}); }
TEST_F(TestDSP_FormatTranspose, I16_NchwToNhwc) { RunTransposeCase({kNumberTypeInt16, kFormatNchw, kFormatNhwc}); }
TEST_F(TestDSP_FormatTranspose, I32_NhwcToNchw) { RunTransposeCase({kNumberTypeInt32, kFormatNhwc, kFormatNchw}); }
TEST_F(TestDSP_FormatTranspose, I32_NchwToNhwc) { RunTransposeCase({kNumberTypeInt32, kFormatNchw, kFormatNhwc}); }
TEST_F(TestDSP_FormatTranspose, C64_NhwcToNchw) { RunTransposeCase({kNumberTypeComplex64, kFormatNhwc, kFormatNchw}); }
TEST_F(TestDSP_FormatTranspose, C64_NchwToNhwc) { RunTransposeCase({kNumberTypeComplex64, kFormatNchw, kFormatNhwc}); }

// Negative: identical formats and out-of-range enum values are no-ops in the
// DSP kernel, so CheckSpecs must reject them (creator returns nullptr).
TEST_F(TestDSP_FormatTranspose, IdenticalFormat_RejectedByCheckSpecs) {
  ExpectRejectedByCheckSpecs(kFormatNhwc, kFormatNhwc, kNumberTypeFloat32);
}
TEST_F(TestDSP_FormatTranspose, UnsupportedFormat_RejectedByCheckSpecs) {
  ExpectRejectedByCheckSpecs(kUnsupportedFormat, kFormatNchw, kNumberTypeFloat32);
}

// Run() side re-validation (CheckSpecs is skipped by DSPKernelCreator when the
// output shape is not inferred). Construct the kernel directly to reach these
// defensive branches without a device runtime.
TEST_F(TestDSP_FormatTranspose, Run_RejectsIdenticalFormatPair) {
  ExpectRunRejected(kFormatNhwc, kFormatNhwc, SrcShape(kFormatNhwc), DstShape(kFormatNchw));
}
TEST_F(TestDSP_FormatTranspose, Run_RejectsNon4DShape) {
  const std::vector<int> rank3 = {kBatch, kHeight, kWidth};
  ExpectRunRejected(kFormatNhwc, kFormatNchw, rank3, rank3);
}
TEST_F(TestDSP_FormatTranspose, Run_RejectsUninferredShape) {
  const std::vector<int> dynamic_shape = {-1, kHeight, kWidth, kChannel};
  ExpectRunRejected(kFormatNhwc, kFormatNchw, SrcShape(kFormatNhwc), dynamic_shape);
}

}  // namespace mindspore::lite::dsp::test

#endif  // SUPPORT_FT04
