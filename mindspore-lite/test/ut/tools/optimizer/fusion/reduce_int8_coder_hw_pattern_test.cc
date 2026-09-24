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
#include "tools/converter/micro/coder/context.h"
#include "tools/converter/micro/coder/opcoders/nnacl/int8/reduce_int8_coder.h"
#include "nnacl_c/reduce_parameter.h"

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
}  // namespace

// Guards the kernel selection of ReduceInt8Coder::DoCode: the ReduceMeanHW fast path
// (NCHW pack + plane reduction) is only valid for reducing exactly the H/W axes {1, 2}
// of a 4D input. It used to be picked by a sum-of-axes test (sum == 3), which also
// captured axes {0, 3} (N+C) and produced garbage output (only n*c of the outputs were
// written). These cases pin the generated kernel calls per axes/mode/shape.
class ReduceInt8CoderHWPatternTest : public mindspore::CommonTest {
 public:
  ReduceInt8CoderHWPatternTest() = default;

  // Drives the real ReduceInt8Coder Prepare+DoCode on an int8 1x16x16x3-shaped graph
  // with the given axes and returns the generated net code.
  std::string GenerateCode(const std::vector<int> &axes, int mode = static_cast<int>(schema::ReduceMode_ReduceMean),
                           const std::vector<int> &input_shape = {1, 16, 16, 3}) {
    std::vector<int> normalized_axes = axes;
    for (auto &axis : normalized_axes) {
      if (axis < 0) {
        axis += static_cast<int>(input_shape.size());
      }
    }
    std::vector<int> output_shape = input_shape;
    for (auto axis : normalized_axes) {
      output_shape[axis] = 1;
    }

    auto *input = new Tensor(kNumberTypeInt8, input_shape);
    input->AddQuantParam(MakeQuantParam(0.039, -1));
    auto *axes_tensor = new Tensor(kNumberTypeInt32, {static_cast<int>(axes.size())});
    axes_tensor->MallocData();
    memcpy(axes_tensor->data(), axes.data(), axes.size() * sizeof(int));
    auto *output = new Tensor(kNumberTypeInt8, output_shape);
    output->AddQuantParam(MakeQuantParam(0.035, 5));

    std::vector<Tensor *> inputs{input, axes_tensor};
    std::vector<Tensor *> outputs{output};
    nnacl::ReduceInt8Coder coder(inputs, outputs, nullptr, 0, kX86);
    auto *reduce_param = static_cast<ReduceParameter *>(calloc(1, sizeof(ReduceParameter)));
    reduce_param->mode_ = mode;
    reduce_param->keep_dims_ = true;
    reduce_param->reduce_to_end_ = false;
    // ReduceParameter composes (not inherits) OpParameter; the coder reinterprets the
    // pointer back, mirroring how the coder framework passes populated parameters.
    coder.set_parameter(reinterpret_cast<OpParameter *>(reduce_param));  // freed by ~OperatorCoder

    CoderContext context(0);
    EXPECT_EQ(coder.Prepare(&context), RET_OK);
    EXPECT_EQ(coder.DoCode(&context), RET_OK);
    // Prepare() mallocs workspace buffers through the allocator singleton; release them
    // so consecutive tests do not leak.
    MemoryAllocator::GetInstance()->Free();
    std::string code = JoinCodeBlocks(context);

    delete input;
    delete axes_tensor;
    delete output;
    return code;
  }
};

TEST_F(ReduceInt8CoderHWPatternTest, AxisNAndCMustNotTakeHWFastPath) {
  // The regression: axes {0, 3} also sums to 3 and used to be mistaken for H/W.
  auto code = GenerateCode({0, 3});
  EXPECT_EQ(code.find("ReduceMeanHW"), std::string::npos);
  EXPECT_EQ(code.find("PackNHWCToNCHWInt8"), std::string::npos);
  EXPECT_NE(code.find("ReduceMeanInt8"), std::string::npos);
  EXPECT_NE(code.find("ReduceMeanLastAxis"), std::string::npos);
}

TEST_F(ReduceInt8CoderHWPatternTest, ReversedAxisNAndCMustNotTakeHWFastPath) {
  auto code = GenerateCode({3, 0});
  EXPECT_EQ(code.find("ReduceMeanHW"), std::string::npos);
  EXPECT_NE(code.find("ReduceMeanLastAxis"), std::string::npos);
}

TEST_F(ReduceInt8CoderHWPatternTest, AxisHWKeepsFastPath) {
  auto code = GenerateCode({1, 2});
  EXPECT_NE(code.find("PackNHWCToNCHWInt8"), std::string::npos);
  EXPECT_NE(code.find("ReduceMeanHW"), std::string::npos);
  EXPECT_EQ(code.find("ReduceMeanInt8"), std::string::npos);
}

TEST_F(ReduceInt8CoderHWPatternTest, ReversedAxisHWKeepsFastPath) {
  auto code = GenerateCode({2, 1});
  EXPECT_NE(code.find("ReduceMeanHW"), std::string::npos);
  EXPECT_EQ(code.find("ReduceMeanInt8"), std::string::npos);
}

TEST_F(ReduceInt8CoderHWPatternTest, NegativeHWAxesNormalizedToFastPath) {
  // [-3, -2] is H/W on a rank-4 input; the pattern is decided after ReSize() so
  // negative axes are already normalized and still take the fast path.
  auto code = GenerateCode({-3, -2});
  EXPECT_NE(code.find("ReduceMeanHW"), std::string::npos);
  EXPECT_EQ(code.find("ReduceMeanInt8"), std::string::npos);
}

TEST_F(ReduceInt8CoderHWPatternTest, Rank3InputTakesGenericPath) {
  auto code = GenerateCode({1, 2}, static_cast<int>(schema::ReduceMode_ReduceMean), {16, 16, 3});
  EXPECT_EQ(code.find("ReduceMeanHW"), std::string::npos);
  EXPECT_NE(code.find("ReduceMeanLastAxis"), std::string::npos);
}

TEST_F(ReduceInt8CoderHWPatternTest, NonMeanModeTakesGenericPath) {
  // The fast path kernel is ReduceMean-specific; other reduce modes must not take it.
  auto code = GenerateCode({1, 2}, static_cast<int>(schema::ReduceMode_ReduceSum));
  EXPECT_EQ(code.find("ReduceMeanHW"), std::string::npos);
  EXPECT_NE(code.find("ReduceSumInt8"), std::string::npos);
  EXPECT_NE(code.find("ReduceSumLastAxis"), std::string::npos);
}
}  // namespace micro
}  // namespace lite
}  // namespace mindspore
