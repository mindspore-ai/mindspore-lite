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
#include "src/litert/kernel/cpu/nnacl_c/concat_parameter.h"
#include "src/litert/kernel/dsp/kernel/concat.h"

#ifdef SUPPORT_FT04

namespace mindspore::lite::dsp::test {

namespace {
constexpr int kValueScale = 64;
constexpr size_t kComplexComponents = 2;
constexpr size_t kRealComponent = 0;
constexpr size_t kImagComponent = 1;
constexpr int kNoMismatch = -1;
// A helper met a dtype it does not know how to read or write. Reported instead of quietly falling back
// to one element size, which would make expected and actual wrong in the same way and pass anyway.
constexpr int kUnsupportedDtype = -2;
// The largest input count the kernel commits to, matching kMaxConcatInputs in concat.cc.
constexpr int kConcatInputCount = 8;
constexpr int kRejectedInputCount = kConcatInputCount + 1;
constexpr int kRankOfTwo = 2;
constexpr int kColumnOfSix = 6;
// Quantization parameters for the int8 cases. The kernel requires an exact match, so the two scales are
// far apart rather than close.
constexpr double kQuantScale = 1.0;
constexpr double kOtherQuantScale = 2.0;
constexpr int32_t kQuantZeroPoint = 0;
constexpr int32_t kOtherQuantZeroPoint = 5;
}  // namespace

class TestDSP_Concat : public DSPCommonTest {};

namespace {
// Every element of every input gets its own value, so a misplaced block, a swapped input or an axis
// that was taken literally instead of normalized shows up as a mismatch.
lite::Tensor *MakeTensor(TypeId dtype, const std::vector<int> &shape, int input_index,
                         const std::shared_ptr<DSPAllocator> &allocator) {
  auto *tensor = new lite::Tensor(dtype, shape, NHWC, lite::Category::CONST_TENSOR);
  if (tensor->MallocData(allocator) != lite::RET_OK) {
    delete tensor;
    return nullptr;
  }
  int element = 0;
  const auto base = static_cast<float>(input_index * kValueScale);
  if (dtype == kNumberTypeComplex64) {
    auto *data = reinterpret_cast<float *>(tensor->MutableData());
    for (size_t i = 0; i < tensor->ElementsNum(); ++i) {
      data[i * kComplexComponents + kRealComponent] = base + static_cast<float>(element);
      data[i * kComplexComponents + kImagComponent] = base + static_cast<float>(element) + 0.5f;
      ++element;
    }
  } else if (dtype == kNumberTypeFloat32) {
    auto *data = reinterpret_cast<float *>(tensor->MutableData());
    for (size_t i = 0; i < tensor->ElementsNum(); ++i) {
      data[i] = base + static_cast<float>(element++);
    }
  } else if (dtype == kNumberTypeFloat16) {
    auto *data = reinterpret_cast<uint16_t *>(tensor->MutableData());
    for (size_t i = 0; i < tensor->ElementsNum(); ++i) {
      data[i] = Fp32ToFp16(base + static_cast<float>(element++));
    }
  } else if (dtype == kNumberTypeInt32) {
    auto *data = reinterpret_cast<int32_t *>(tensor->MutableData());
    for (size_t i = 0; i < tensor->ElementsNum(); ++i) {
      data[i] = static_cast<int32_t>(base + static_cast<float>(element++));
    }
  } else if (dtype == kNumberTypeInt16) {
    auto *data = reinterpret_cast<int16_t *>(tensor->MutableData());
    for (size_t i = 0; i < tensor->ElementsNum(); ++i) {
      data[i] = static_cast<int16_t>(base + static_cast<float>(element++));
    }
  } else {
    // Filling a buffer of another element size would make the expected values and the output wrong in
    // the same way, and the comparison below would still pass. An unlisted dtype has to fail instead.
    delete tensor;
    return nullptr;
  }
  return tensor;
}

std::vector<float> TensorValues(const lite::Tensor *tensor) {
  const bool listed = tensor->data_type() == kNumberTypeComplex64 || tensor->data_type() == kNumberTypeFloat32 ||
                      tensor->data_type() == kNumberTypeFloat16 || tensor->data_type() == kNumberTypeInt32 ||
                      tensor->data_type() == kNumberTypeInt16;
  if (!listed) {
    // Reading with the wrong element size would agree with the values MakeTensor wrote the same way,
    // so the caller gets an empty vector and VerifyConcat reports kUnsupportedDtype.
    return {};
  }
  const auto count = tensor->ElementsNum();
  std::vector<float> values(static_cast<size_t>(count) * kComplexComponents);
  if (tensor->data_type() == kNumberTypeComplex64) {
    auto *data = reinterpret_cast<const float *>(tensor->data());
    for (int64_t i = 0; i < count; ++i) {
      values[i * kComplexComponents] = data[i * kComplexComponents];
      values[i * kComplexComponents + kImagComponent] = data[i * kComplexComponents + kImagComponent];
    }
  } else if (tensor->data_type() == kNumberTypeFloat32) {
    auto *data = reinterpret_cast<const float *>(tensor->data());
    for (int64_t i = 0; i < count; ++i) {
      values[i] = data[i];
    }
  } else if (tensor->data_type() == kNumberTypeFloat16) {
    auto *data = reinterpret_cast<const uint16_t *>(tensor->data());
    for (int64_t i = 0; i < count; ++i) {
      values[i] = Fp16ToFp32(data[i]);
    }
  } else if (tensor->data_type() == kNumberTypeInt32) {
    auto *data = reinterpret_cast<const int32_t *>(tensor->data());
    for (int64_t i = 0; i < count; ++i) {
      values[i] = static_cast<float>(data[i]);
    }
  } else {
    // kNumberTypeInt16, the only dtype left by the check above.
    auto *data = reinterpret_cast<const int16_t *>(tensor->data());
    for (int64_t i = 0; i < count; ++i) {
      values[i] = static_cast<float>(data[i]);
    }
  }
  return values;
}

// Walks the output in row-major order and reports the first element that does not come from the input
// and position the concatenation calls for, kNoMismatch when the whole output is right, and
// kUnsupportedDtype when a tensor holds a dtype the helpers above do not know how to read.
int VerifyConcat(const lite::Tensor *output, const std::vector<lite::Tensor *> &inputs, int axis) {
  const auto &out_shape = output->shape();
  const auto rank = static_cast<int>(out_shape.size());
  const int normalized_axis = (axis < 0) ? axis + rank : axis;
  const auto components = (output->data_type() == kNumberTypeComplex64) ? kComplexComponents : 1;
  std::vector<int> prefixes(inputs.size(), 0);
  for (size_t i = 1; i < inputs.size(); ++i) {
    prefixes[i] = prefixes[i - 1] + inputs[i - 1]->shape()[normalized_axis];
  }
  const auto out_values = TensorValues(output);
  std::vector<std::vector<float>> in_values(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    in_values[i] = TensorValues(inputs[i]);
  }
  const auto out_span = static_cast<size_t>(output->ElementsNum()) * components;
  if (out_values.size() < out_span) {
    return kUnsupportedDtype;
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (in_values[i].size() < static_cast<size_t>(inputs[i]->ElementsNum()) * components) {
      return kUnsupportedDtype;
    }
  }
  std::vector<int> coords(rank, 0);
  for (int64_t out_index = 0; out_index < output->ElementsNum(); ++out_index) {
    int rest = static_cast<int>(out_index);
    for (int dim = rank - 1; dim >= 0; --dim) {
      coords[dim] = rest % out_shape[dim];
      rest /= out_shape[dim];
    }
    size_t source = 0;
    for (size_t i = 1; i < inputs.size(); ++i) {
      if (coords[normalized_axis] >= prefixes[i]) {
        source = i;
      }
    }
    const auto &in_shape = inputs[source]->shape();
    coords[normalized_axis] -= prefixes[source];
    int in_index = 0;
    for (int dim = 0; dim < rank; ++dim) {
      in_index = in_index * in_shape[dim] + coords[dim];
    }
    for (size_t part = 0; part < components; ++part) {
      const float expected = in_values[source][in_index * components + part];
      if (out_values[static_cast<size_t>(out_index) * components + part] != expected) {
        return static_cast<int>(out_index);
      }
    }
    coords[normalized_axis] += prefixes[source];
  }
  return kNoMismatch;
}

// LiteKernel::~LiteKernel() releases op_parameter_ with free(), so the parameter has to come from
// malloc rather than new. Ownership passes to the kernel, on both the created and the refused path.
ConcatParameter *MakeConcatParameter(int axis) {
  auto *param = static_cast<ConcatParameter *>(malloc(sizeof(ConcatParameter)));
  if (param == nullptr) {
    return nullptr;
  }
  (void)memset(param, 0, sizeof(ConcatParameter));
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_Concat);
  param->axis_ = axis;
  return param;
}

kernel::LiteKernel *CreateConcatKernel(const std::vector<lite::Tensor *> &inputs,
                                       const std::vector<lite::Tensor *> &outputs, TypeId dtype, int axis,
                                       lite::InnerContext *ctx) {
  auto *param = MakeConcatParameter(axis);
  if (param == nullptr) {
    return nullptr;
  }
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, dtype, NHWC, schema::PrimitiveType_Concat};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  if (creator == nullptr) {
    free(param);
    return nullptr;
  }
  return creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx, key);
}

// An int8 tensor with the quantization parameters a real graph attaches, or without them when the case
// is about them being missing. No data is needed: the rule under test is decided from the parameters.
lite::Tensor *MakeQuantTensor(const std::vector<int> &shape, bool with_quant_params, double scale, int32_t zero_point) {
  auto *tensor = new lite::Tensor(kNumberTypeInt8, shape, NHWC, lite::Category::CONST_TENSOR);
  if (with_quant_params) {
    const LiteQuantParam quant_param{scale, zero_point};
    tensor->AddQuantParam(quant_param);
  }
  return tensor;
}

// int8 only reaches the DSP through the SUPPORT_FT78 registrations, so these cases build the kernel
// directly and test the decision it makes before a launch rather than a launch itself. The concrete
// type is what makes CheckSpecs visible; LiteKernel does not expose it.
kernel::ConcatDSPKernel *CreateQuantKernel(const std::vector<lite::Tensor *> &inputs,
                                           const std::vector<lite::Tensor *> &outputs, ConcatParameter *param,
                                           lite::InnerContext *ctx) {
  return new kernel::ConcatDSPKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs, ctx);
}
}  // namespace

TEST_F(TestDSP_Concat, Concat_Fp32_Axis0) {
  InitDSPRuntime();
  const std::vector<int> shape = {4, 8};
  auto *first = MakeTensor(kNumberTypeFloat32, shape, 0, allocator_);
  auto *second = MakeTensor(kNumberTypeFloat32, shape, 1, allocator_);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  const std::vector<int> out_shape = {8, 8};
  auto *output = new lite::Tensor(kNumberTypeFloat32, out_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  std::vector<lite::Tensor *> inputs = {first, second};
  auto *kernel = CreateConcatKernel(inputs, {output}, kNumberTypeFloat32, 0, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());
  ASSERT_EQ(kNoMismatch, VerifyConcat(output, inputs, 0));

  delete kernel;
  delete ctx;
  delete first;
  delete second;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Concat, Concat_Fp32_NegativeAxis) {
  InitDSPRuntime();
  const std::vector<int> first_shape = {2, 4};
  const std::vector<int> second_shape = {2, 6};
  auto *first = MakeTensor(kNumberTypeFloat32, first_shape, 0, allocator_);
  auto *second = MakeTensor(kNumberTypeFloat32, second_shape, 1, allocator_);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  const std::vector<int> out_shape = {2, 10};
  auto *output = new lite::Tensor(kNumberTypeFloat32, out_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // -1 has to reach armc as the last dimension, the way the CPU kernel reads it.
  std::vector<lite::Tensor *> inputs = {first, second};
  auto *kernel = CreateConcatKernel(inputs, {output}, kNumberTypeFloat32, -1, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());
  ASSERT_EQ(kNoMismatch, VerifyConcat(output, inputs, -1));

  delete kernel;
  delete ctx;
  delete first;
  delete second;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Concat, Concat_Fp16_Axis1) {
  InitDSPRuntime();
  const std::vector<int> first_shape = {2, 3};
  const std::vector<int> second_shape = {2, 5};
  auto *first = MakeTensor(kNumberTypeFloat16, first_shape, 0, allocator_);
  auto *second = MakeTensor(kNumberTypeFloat16, second_shape, 1, allocator_);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  const std::vector<int> out_shape = {2, 8};
  auto *output = new lite::Tensor(kNumberTypeFloat16, out_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  std::vector<lite::Tensor *> inputs = {first, second};
  auto *kernel = CreateConcatKernel(inputs, {output}, kNumberTypeFloat16, 1, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());
  ASSERT_EQ(kNoMismatch, VerifyConcat(output, inputs, 1));

  delete kernel;
  delete ctx;
  delete first;
  delete second;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Concat, Concat_Int16_Axis0) {
  InitDSPRuntime();
  const std::vector<int> shape = {3, 4};
  auto *first = MakeTensor(kNumberTypeInt16, shape, 0, allocator_);
  auto *second = MakeTensor(kNumberTypeInt16, shape, 1, allocator_);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  const std::vector<int> out_shape = {6, 4};
  auto *output = new lite::Tensor(kNumberTypeInt16, out_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  std::vector<lite::Tensor *> inputs = {first, second};
  auto *kernel = CreateConcatKernel(inputs, {output}, kNumberTypeInt16, 0, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());
  ASSERT_EQ(kNoMismatch, VerifyConcat(output, inputs, 0));

  delete kernel;
  delete ctx;
  delete first;
  delete second;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Concat, Concat_Int32_ThreeInputs) {
  InitDSPRuntime();
  const std::vector<int> first_shape = {2, 2};
  const std::vector<int> wide_shape = {2, 3};
  auto *first = MakeTensor(kNumberTypeInt32, first_shape, 0, allocator_);
  auto *second = MakeTensor(kNumberTypeInt32, wide_shape, 1, allocator_);
  auto *third = MakeTensor(kNumberTypeInt32, first_shape, 2, allocator_);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(third, nullptr);
  const std::vector<int> out_shape = {2, 7};
  auto *output = new lite::Tensor(kNumberTypeInt32, out_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  std::vector<lite::Tensor *> inputs = {first, second, third};
  auto *kernel = CreateConcatKernel(inputs, {output}, kNumberTypeInt32, 1, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());
  ASSERT_EQ(kNoMismatch, VerifyConcat(output, inputs, 1));

  delete kernel;
  delete ctx;
  delete first;
  delete second;
  delete third;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Concat, Concat_Complex64_Axis0) {
  InitDSPRuntime();
  const std::vector<int> shape = {1, 3};
  auto *first = MakeTensor(kNumberTypeComplex64, shape, 0, allocator_);
  auto *second = MakeTensor(kNumberTypeComplex64, shape, 1, allocator_);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  const std::vector<int> out_shape = {2, 3};
  auto *output = new lite::Tensor(kNumberTypeComplex64, out_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // Both halves of every element have to land together, so the check reads them component by
  // component rather than trusting the element count.
  std::vector<lite::Tensor *> inputs = {first, second};
  auto *kernel = CreateConcatKernel(inputs, {output}, kNumberTypeComplex64, 0, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());
  ASSERT_EQ(kNoMismatch, VerifyConcat(output, inputs, 0));

  delete kernel;
  delete ctx;
  delete first;
  delete second;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Concat, Concat_Fp32_EightInputs) {
  InitDSPRuntime();
  const std::vector<int> shape = {1, 4};
  std::vector<lite::Tensor *> inputs;
  for (int i = 0; i < kConcatInputCount; ++i) {
    auto *tensor = MakeTensor(kNumberTypeFloat32, shape, i, allocator_);
    ASSERT_NE(tensor, nullptr);
    inputs.push_back(tensor);
  }
  const std::vector<int> out_shape = {8, 4};
  auto *output = new lite::Tensor(kNumberTypeFloat32, out_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  auto *kernel = CreateConcatKernel(inputs, {output}, kNumberTypeFloat32, 0, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());
  ASSERT_EQ(RET_OK, kernel->Run());
  ASSERT_EQ(kNoMismatch, VerifyConcat(output, inputs, 0));

  delete kernel;
  delete ctx;
  for (auto *tensor : inputs) {
    delete tensor;
  }
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Concat, Concat_RankMismatchIsRejected) {
  InitDSPRuntime();
  auto *first = MakeTensor(kNumberTypeFloat32, std::vector<int>{2, 4}, 0, allocator_);
  auto *second = MakeTensor(kNumberTypeFloat32, std::vector<int>{2, 2, 4}, 1, allocator_);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  auto *output = new lite::Tensor(kNumberTypeFloat32, std::vector<int>{4, 4}, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // Concatenating tensors of different rank would need a reshape the DSP is not told about.
  std::vector<lite::Tensor *> inputs = {first, second};
  auto *kernel = CreateConcatKernel(inputs, {output}, kNumberTypeFloat32, 0, ctx);
  ASSERT_EQ(kernel, nullptr);

  delete ctx;
  delete first;
  delete second;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Concat, Concat_AxisOutOfRangeIsRejected) {
  InitDSPRuntime();
  auto *first = MakeTensor(kNumberTypeFloat32, std::vector<int>{2, 4}, 0, allocator_);
  auto *second = MakeTensor(kNumberTypeFloat32, std::vector<int>{2, 4}, 1, allocator_);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  auto *output = new lite::Tensor(kNumberTypeFloat32, std::vector<int>{4, 4}, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  std::vector<lite::Tensor *> inputs = {first, second};
  auto *kernel = CreateConcatKernel(inputs, {output}, kNumberTypeFloat32, kRankOfTwo, ctx);
  ASSERT_EQ(kernel, nullptr);

  delete ctx;
  delete first;
  delete second;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Concat, Concat_OutputTooSmallIsRejected) {
  InitDSPRuntime();
  auto *first = MakeTensor(kNumberTypeFloat32, std::vector<int>{2, 4}, 0, allocator_);
  auto *second = MakeTensor(kNumberTypeFloat32, std::vector<int>{2, 4}, 1, allocator_);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  // Half the joined extent: the DSP would write past the buffer.
  auto *output = new lite::Tensor(kNumberTypeFloat32, std::vector<int>{2, 4}, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  std::vector<lite::Tensor *> inputs = {first, second};
  auto *kernel = CreateConcatKernel(inputs, {output}, kNumberTypeFloat32, 0, ctx);
  ASSERT_EQ(kernel, nullptr);

  delete ctx;
  delete first;
  delete second;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Concat, Concat_SingleInputIsRejected) {
  InitDSPRuntime();
  auto *first = MakeTensor(kNumberTypeFloat32, std::vector<int>{2, 4}, 0, allocator_);
  ASSERT_NE(first, nullptr);
  auto *output = new lite::Tensor(kNumberTypeFloat32, std::vector<int>{2, 4}, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  std::vector<lite::Tensor *> inputs = {first};
  auto *kernel = CreateConcatKernel(inputs, {output}, kNumberTypeFloat32, 0, ctx);
  ASSERT_EQ(kernel, nullptr);

  delete ctx;
  delete first;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Concat, Concat_MoreThanEightInputsIsRejected) {
  InitDSPRuntime();
  std::vector<lite::Tensor *> inputs;
  for (int i = 0; i < kRejectedInputCount; ++i) {
    auto *tensor = MakeTensor(kNumberTypeFloat32, std::vector<int>{1, 4}, i, allocator_);
    ASSERT_NE(tensor, nullptr);
    inputs.push_back(tensor);
  }
  auto *output =
    new lite::Tensor(kNumberTypeFloat32, std::vector<int>{kRejectedInputCount, 4}, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // One input past the envelope armc's own case covers. Handing concat_func a row count it was never
  // checked against would let it read or copy beyond the buffers, so the kernel has to refuse and leave
  // the join to the CPU kernel. The eight input side is accepted by Concat_Fp32_EightInputs, which
  // keeps the boundary from drifting to the wrong side.
  auto *kernel = CreateConcatKernel(inputs, {output}, kNumberTypeFloat32, 0, ctx);
  ASSERT_EQ(kernel, nullptr);

  for (auto *tensor : inputs) {
    delete tensor;
  }
  delete ctx;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Concat, Concat_Int8_SharedQuantParamsIsAccepted) {
  InitDSPRuntime();
  const std::vector<int> shape = {2, 4};
  auto *first = MakeQuantTensor(shape, true, kQuantScale, kQuantZeroPoint);
  auto *second = MakeQuantTensor(shape, true, kQuantScale, kQuantZeroPoint);
  auto *output = MakeQuantTensor(std::vector<int>{4, 4}, true, kQuantScale, kQuantZeroPoint);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // One scale and one zero point across the join: copying bytes is already the right answer.
  auto *param = MakeConcatParameter(0);
  ASSERT_NE(param, nullptr);
  std::vector<lite::Tensor *> inputs = {first, second};
  auto *kernel = CreateQuantKernel(inputs, {output}, param, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Prepare());

  delete kernel;
  delete ctx;
  delete first;
  delete second;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Concat, Concat_Int8_MismatchedScaleIsRejected) {
  InitDSPRuntime();
  const std::vector<int> shape = {2, 4};
  auto *first = MakeQuantTensor(shape, true, kQuantScale, kQuantZeroPoint);
  auto *second = MakeQuantTensor(shape, true, kOtherQuantScale, kQuantZeroPoint);
  auto *output = MakeQuantTensor(std::vector<int>{4, 4}, true, kQuantScale, kQuantZeroPoint);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // The second input would be copied into a buffer read with half its scale.
  auto *param = MakeConcatParameter(0);
  ASSERT_NE(param, nullptr);
  std::vector<lite::Tensor *> inputs = {first, second};
  auto *kernel = CreateQuantKernel(inputs, {output}, param, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_ERROR, kernel->CheckSpecs());
  ASSERT_EQ(RET_ERROR, kernel->Prepare());
  ASSERT_EQ(RET_ERROR, kernel->Run());

  delete kernel;
  delete ctx;
  delete first;
  delete second;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Concat, Concat_Int8_MismatchedZeroPointIsRejected) {
  InitDSPRuntime();
  const std::vector<int> shape = {2, 4};
  auto *first = MakeQuantTensor(shape, true, kQuantScale, kQuantZeroPoint);
  auto *second = MakeQuantTensor(shape, true, kQuantScale, kOtherQuantZeroPoint);
  auto *output = MakeQuantTensor(std::vector<int>{4, 4}, true, kQuantScale, kQuantZeroPoint);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  auto *param = MakeConcatParameter(0);
  ASSERT_NE(param, nullptr);
  std::vector<lite::Tensor *> inputs = {first, second};
  auto *kernel = CreateQuantKernel(inputs, {output}, param, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_ERROR, kernel->CheckSpecs());

  delete kernel;
  delete ctx;
  delete first;
  delete second;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Concat, Concat_Int8_MissingQuantParamsIsRejected) {
  InitDSPRuntime();
  const std::vector<int> shape = {2, 4};
  auto *first = MakeQuantTensor(shape, false, kQuantScale, kQuantZeroPoint);
  auto *second = MakeQuantTensor(shape, false, kQuantScale, kQuantZeroPoint);
  auto *output = MakeQuantTensor(std::vector<int>{4, 4}, false, kQuantScale, kQuantZeroPoint);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // Without parameters there is nothing to compare, and the CPU int8 kernel refuses these too.
  auto *param = MakeConcatParameter(0);
  ASSERT_NE(param, nullptr);
  std::vector<lite::Tensor *> inputs = {first, second};
  auto *kernel = CreateQuantKernel(inputs, {output}, param, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_ERROR, kernel->CheckSpecs());

  delete kernel;
  delete ctx;
  delete first;
  delete second;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Concat, Concat_Int8_MismatchedParameterQuantIsRejected) {
  InitDSPRuntime();
  const std::vector<int> shape = {2, 4};
  auto *first = MakeQuantTensor(shape, false, kQuantScale, kQuantZeroPoint);
  auto *second = MakeQuantTensor(shape, false, kQuantScale, kQuantZeroPoint);
  auto *output = MakeQuantTensor(std::vector<int>{4, 4}, false, kQuantScale, kQuantZeroPoint);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  // The other way quant parameters reach this kernel: a filled ConcatQuantArg, which takes precedence
  // over the tensor parameters the way the CPU int8 kernel fills it in.
  static QuantArg in_args[] = {{static_cast<float>(kQuantScale), kQuantZeroPoint},
                               {static_cast<float>(kOtherQuantScale), kQuantZeroPoint}};
  auto *param = MakeConcatParameter(0);
  ASSERT_NE(param, nullptr);
  param->quant_arg_.in_args_ = in_args;
  param->quant_arg_.out_args_.scale_ = static_cast<float>(kQuantScale);
  param->quant_arg_.out_args_.zp_ = kQuantZeroPoint;
  std::vector<lite::Tensor *> inputs = {first, second};
  auto *kernel = CreateQuantKernel(inputs, {output}, param, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_ERROR, kernel->CheckSpecs());

  delete kernel;
  delete ctx;
  delete first;
  delete second;
  delete output;
  UninitDSPRuntime();
}

TEST_F(TestDSP_Concat, Concat_ResizeChangesAnInputShape) {
  InitDSPRuntime();
  const std::vector<int> shape = {2, 4};
  auto *first = MakeTensor(kNumberTypeFloat32, shape, 0, allocator_);
  auto *second = MakeTensor(kNumberTypeFloat32, shape, 1, allocator_);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  const std::vector<int> out_shape = {4, 4};
  auto *output = new lite::Tensor(kNumberTypeFloat32, out_shape, NHWC, lite::Category::CONST_TENSOR);
  ASSERT_EQ(output->MallocData(allocator_), lite::RET_OK);
  auto *ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  std::vector<lite::Tensor *> inputs = {first, second};
  auto *kernel = CreateConcatKernel(inputs, {output}, kNumberTypeFloat32, 0, ctx);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(RET_OK, kernel->Run());

  // A resize reaches the kernel through LiteKernel::ReSize. DSPKernel::ReSize stops as soon as
  // InferShapeDone() reports that the output holds no unknown dimension (lite_kernel.h:93-97,
  // dsp_kernel.cc:79-82); it never asks whether the inputs still agree with each other, and Prepare is
  // not re-run. Widening the second input is accepted there while the shape table cached at creation
  // goes stale, which leaves Run as the only place able to refuse.
  second->set_shape({2, kColumnOfSix});
  ASSERT_EQ(RET_OK, kernel->ReSize());
  ASSERT_EQ(RET_ERROR, kernel->Run());

  delete kernel;
  delete ctx;
  delete first;
  delete second;
  delete output;
  UninitDSPRuntime();
}
}  // namespace mindspore::lite::dsp::test
#endif  // SUPPORT_FT04
