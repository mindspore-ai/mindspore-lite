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

#define USE_DEPRECATED_API
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "common/common_test.h"
#include "include/errorcode.h"
#include "src/tensor.h"
#include "tools/converter/micro/coder/allocator/allocator.h"
#include "tools/converter/micro/coder/context.h"
#include "tools/converter/micro/coder/generator/component/component.h"
#include "tools/converter/micro/coder/opcoders/parallel.h"
#include "tools/converter/micro/coder/opcoders/nnacl/int8/add_int8_coder.h"
#include "nnacl_c/arithmetic_parameter.h"

namespace mindspore {
namespace lite {
namespace micro {
namespace {
LiteQuantParam MakeQuantParam(double scale, int32_t zero_point) {
  LiteQuantParam param;
  param.scale = scale;
  param.zeroPoint = zero_point;
  param.inited = true;
  return param;
}

std::string JoinCodeBlocks(const CoderContext &context) {
  std::string code;
  for (const auto &block : context.code_blocks()) {
    code += block;
  }
  return code;
}

std::vector<int> BroadcastShape(const std::vector<int> &a, const std::vector<int> &b) {
  std::vector<int> out;
  size_t ndim = std::max(a.size(), b.size());
  for (size_t i = 0; i < ndim; i++) {
    int da = i < ndim - a.size() ? 1 : a[i - (ndim - a.size())];
    int db = i < ndim - b.size() ? 1 : b[i - (ndim - b.size())];
    out.push_back(std::max(da, db));
  }
  return out;
}
}  // namespace

// Guards the kernel selection of AddInt8Coder::DoCode. AddBroadcastInt8Run only implements
// the bias-add-like pattern: one operand covers the whole output and the other holds exactly
// one chunk of in_size_ elements. Feeding it any other broadcast shape (e.g. (2,3,4,5)x(2,1,1,1)
// or (4,1)x(1,8)) applies the wrong broadcast index and can read past the smaller operand, so
// those shapes must fall back to TileDimensionsInt8 + AddInt8, the same strategy as the litert
// QuantizedAddCPUKernel. These cases pin the generated kernel calls per shape combination.
class AddInt8CoderBroadcastPatternTest : public mindspore::CommonTest {
 public:
  AddInt8CoderBroadcastPatternTest() = default;

  static void SetUpTestCase() {
    // CoderContext and the memory allocator build std::strings from the component prefix
    // globals (kInputPrefixName, kBufferPrefixNameAdd, ...). In lite-test-converter these
    // default to nullptr until InitGlobalVariable runs, and assigning std::string from a
    // null char* segfaults. Same setup as TestSoftmaxInt8Coder.
    InitGlobalVariable(0);
  }

  // Drives the real AddInt8Coder Prepare+DoCode on an int8 AddFusion node with the given
  // input shapes and returns the generated net code.
  std::string GenerateCode(const std::vector<int> &shape0, const std::vector<int> &shape1) {
    auto output_shape = BroadcastShape(shape0, shape1);

    auto *input0 = new Tensor(kNumberTypeInt8, shape0);
    input0->AddQuantParam(MakeQuantParam(0.03, -3));
    auto *input1 = new Tensor(kNumberTypeInt8, shape1);
    input1->AddQuantParam(MakeQuantParam(0.02, 1));
    auto *output = new Tensor(kNumberTypeInt8, output_shape);
    output->AddQuantParam(MakeQuantParam(0.04, 2));

    std::vector<Tensor *> inputs{input0, input1};
    std::vector<Tensor *> outputs{output};
    auto coder = std::make_unique<nnacl::AddInt8Coder>(inputs, outputs, nullptr, 0, kX86);
    auto *arith_param = static_cast<ArithmeticParameter *>(calloc(1, sizeof(ArithmeticParameter)));
    // In the real flow ArithmeticInferShape (nnacl_c/infer/arithmetic_infer.c) fills the shape
    // arrays before the coder runs: both operands are right-aligned to ndim_ and padded with 1.
    // ReSize() only re-copies them when they still contain -1 (unpopulated), so pre-filling them
    // here mirrors the graph-inferred parameter the coder framework hands over.
    arith_param->ndim_ = output_shape.size();
    int offset0 = static_cast<int>(output_shape.size() - shape0.size());
    int offset1 = static_cast<int>(output_shape.size() - shape1.size());
    for (size_t i = 0; i < arith_param->ndim_; i++) {
      arith_param->in_shape0_[i] = static_cast<int>(i) >= offset0 ? shape0[i - offset0] : 1;
      arith_param->in_shape1_[i] = static_cast<int>(i) >= offset1 ? shape1[i - offset1] : 1;
      arith_param->out_shape_[i] = output_shape[i];
    }
    coder->set_parameter(reinterpret_cast<OpParameter *>(arith_param));  // freed by ~OperatorCoder

    // The session initializes gThreadNum ("m0_thread_num") before DoCode; without it the
    // OpParameter serializer writes a null char* which sets the stream badbit and silently
    // truncates all generated code after the first struct.
    InitThread(0);

    // Mirror CoderSession: assign tensor addresses first (graph inputs get input0/input1...,
    // intermediates get buffer offsets), then Prepare, then DoCode. Without Assign, GenCode
    // exit(1)s on tensor pointers that the allocator has no runtime address for.
    std::vector<std::unique_ptr<OperatorCoder>> nodes;
    nodes.push_back(std::move(coder));
    std::vector<Tensor *> all_tensors{input0, input1, output};
    EXPECT_EQ(MemoryAllocator::GetInstance()->Assign(inputs, outputs, nodes, all_tensors), RET_OK);

    CoderContext context(0);
    EXPECT_EQ(nodes[0]->Prepare(&context), RET_OK);
    EXPECT_EQ(nodes[0]->DoCode(&context), RET_OK);
    // Prepare() mallocs workspace buffers through the allocator singleton; release them
    // so consecutive tests do not leak.
    MemoryAllocator::GetInstance()->Free();
    FreeThread();
    std::string code = JoinCodeBlocks(context);

    delete input0;
    delete input1;
    delete output;
    return code;
  }
};

TEST_F(AddInt8CoderBroadcastPatternTest, GeneralBroadcastMustTile) {
  // The regression: one operand is full-size but the other spans 2 elements while the
  // broadcast chunk is 1, so AddBroadcastInt8Run would add input1[0] everywhere.
  auto code = GenerateCode({2, 3, 4, 5}, {2, 1, 1, 1});
  EXPECT_NE(code.find("TileDimensionsInt8"), std::string::npos);
  EXPECT_NE(code.find("AddInt8"), std::string::npos);
  EXPECT_EQ(code.find("AddBroadcastInt8Run"), std::string::npos);
}

TEST_F(AddInt8CoderBroadcastPatternTest, CrossBroadcastMustTile) {
  // Neither operand is full-size: (4,1)x(1,8) also cannot use the chunked runner.
  auto code = GenerateCode({4, 1}, {1, 8});
  EXPECT_NE(code.find("TileDimensionsInt8"), std::string::npos);
  EXPECT_EQ(code.find("AddBroadcastInt8Run"), std::string::npos);
}

TEST_F(AddInt8CoderBroadcastPatternTest, MidDimBroadcastMustTile) {
  auto code = GenerateCode({2, 1, 4}, {2, 3, 4});
  EXPECT_NE(code.find("TileDimensionsInt8"), std::string::npos);
  EXPECT_EQ(code.find("AddBroadcastInt8Run"), std::string::npos);
}

TEST_F(AddInt8CoderBroadcastPatternTest, BiasLikeBroadcastKeepsFastPath) {
  // The fullyconnected bias-add shape the fast path was introduced for: the second
  // operand is exactly one chunk of the last dimensions and stays in place.
  auto code = GenerateCode({16, 32}, {32});
  EXPECT_NE(code.find("AddBroadcastInt8Run"), std::string::npos);
  EXPECT_EQ(code.find("TileDimensionsInt8"), std::string::npos);
}

TEST_F(AddInt8CoderBroadcastPatternTest, FullOperandOnRightKeepsFastPath) {
  auto code = GenerateCode({32}, {16, 32});
  EXPECT_NE(code.find("AddBroadcastInt8Run"), std::string::npos);
  EXPECT_EQ(code.find("TileDimensionsInt8"), std::string::npos);
}

TEST_F(AddInt8CoderBroadcastPatternTest, SameShapeTakesPlainRun) {
  auto code = GenerateCode({2, 3}, {2, 3});
  EXPECT_NE(code.find("AddInt8Run"), std::string::npos);
  EXPECT_EQ(code.find("AddBroadcastInt8Run"), std::string::npos);
  EXPECT_EQ(code.find("TileDimensionsInt8"), std::string::npos);
}
}  // namespace micro
}  // namespace lite
}  // namespace mindspore
