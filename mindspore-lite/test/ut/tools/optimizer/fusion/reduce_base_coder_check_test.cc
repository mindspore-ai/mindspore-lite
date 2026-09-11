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
#include <memory>
#include <vector>
#include "common/common_test.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"
#include "tools/converter/micro/coder/opcoders/base/reduce_base_coder.h"

namespace mindspore {
namespace lite {
namespace micro {
namespace {
// Exposes the protected validation entry so the axes checks can be driven directly.
class CheckableReduceBaseCoder : public ReduceBaseCoder {
 public:
  CheckableReduceBaseCoder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors)
      : ReduceBaseCoder(in_tensors, out_tensors, nullptr, 0, Target::kX86) {}
  int CheckParameters() { return ReduceBaseCoder::CheckParameters(); }
  void SetAxes(const std::vector<int> &axes) {
    num_axes_ = static_cast<int>(axes.size());
    for (size_t i = 0; i < axes.size(); i++) {
      axes_[i] = axes[i];
    }
  }
  // Unused pure-virtual entries of OperatorCoder; only CheckParameters is exercised here.
  int Prepare(CoderContext *const context) override { return RET_OK; }
  int DoCode(CoderContext *const context) override { return RET_OK; }
};
}  // namespace

class ReduceBaseCoderCheckParametersTest : public mindspore::CommonTest {
 public:
  ReduceBaseCoderCheckParametersTest() = default;

  std::unique_ptr<CheckableReduceBaseCoder> MakeCoder(const std::vector<int> &shape) {
    auto input = new Tensor(kNumberTypeFloat32, shape);
    auto output = new Tensor(kNumberTypeFloat32, {1});
    std::vector<Tensor *> inputs{input};
    std::vector<Tensor *> outputs{output};
    return std::make_unique<CheckableReduceBaseCoder>(inputs, outputs);
  }
};

// Valid axes: single, multi and negative forms must pass.
TEST_F(ReduceBaseCoderCheckParametersTest, ValidAxesPass) {
  auto coder = MakeCoder({2, 3, 4});
  coder->SetAxes({1, 2});
  ASSERT_EQ(coder->CheckParameters(), RET_OK);
}

TEST_F(ReduceBaseCoderCheckParametersTest, ValidNegativeAxisPass) {
  auto coder = MakeCoder({2, 3, 4});
  coder->SetAxes({-1});
  ASSERT_EQ(coder->CheckParameters(), RET_OK);
}

// Empty axes (reduce-all) is normalized internally to the full axis list.
TEST_F(ReduceBaseCoderCheckParametersTest, EmptyAxesReduceAllPass) {
  auto coder = MakeCoder({2, 3, 4});
  coder->SetAxes({});
  ASSERT_EQ(coder->CheckParameters(), RET_OK);
}

// num of axes larger than input rank (duplicates inflate the count): rejected.
TEST_F(ReduceBaseCoderCheckParametersTest, AxesCountExceedsRankRejected) {
  auto coder = MakeCoder({1, 4, 4, 3});
  coder->SetAxes({0, 1, 1, 2, 0});
  ASSERT_EQ(coder->CheckParameters(), RET_ERROR);
}

// Axis value outside the input rank: rejected.
TEST_F(ReduceBaseCoderCheckParametersTest, OutOfRangeAxisRejected) {
  auto coder = MakeCoder({1, 4, 4, 3});
  coder->SetAxes({0, 1, 2, 9});
  ASSERT_EQ(coder->CheckParameters(), RET_ERROR);
}

// Duplicates with count <= rank (e.g. [0,1,1] on a rank-3 input) used to silently reduce the
// same axis twice; they must be rejected explicitly.
TEST_F(ReduceBaseCoderCheckParametersTest, DuplicateAxisRejected) {
  auto coder = MakeCoder({2, 3, 4});
  coder->SetAxes({0, 1, 1});
  ASSERT_EQ(coder->CheckParameters(), RET_ERROR);
}

// A negative axis and its positive equivalent are duplicates after normalization ([-1,2] on
// rank 3 both map to axis 2) and must be rejected.
TEST_F(ReduceBaseCoderCheckParametersTest, NegativeDuplicateRejected) {
  auto coder = MakeCoder({2, 3, 4});
  coder->SetAxes({-1, 2});
  ASSERT_EQ(coder->CheckParameters(), RET_ERROR);
}
}  // namespace micro
}  // namespace lite
}  // namespace mindspore
