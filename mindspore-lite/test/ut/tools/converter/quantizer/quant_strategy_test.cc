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
#include <gtest/gtest.h>
#include <memory>
#include <set>
#include <string>
#include "common/common_test.h"
#include "infer/eltwise.h"
#include "ir/func_graph.h"
#include "ir/manager.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/lite_ops.h"
#include "mindspore/ops/op_def/math_ops.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_e.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_f.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_o.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_q.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_u.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "tools/converter/quantizer/quant_strategy.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace lite::quant {
namespace {
// Mirror of the FULL_QUANT default whitelist in full_quant_quantizer.cc:
// note that Eltwise is deliberately NOT in this set.
std::set<PrimitivePtr> BuildSupportInt8Ops() {
  return {prim::kPrimConv2DFusion, prim::kPrimFullConnection, prim::kPrimMatMulFusion, prim::kPrimReshape,
          prim::kPrimTranspose,    prim::kPrimAddFusion,      prim::kPrimSubFusion,    prim::kPrimMulFusion,
          prim::kPrimMaximum,      prim::kPrimMinimum,        prim::kPrimActivation,   prim::kPrimSoftmax,
          prim::kPrimReduceFusion, prim::kPrimConstantOfShape};
}

// Mirror of the whitelist with enable_all_ops = true (set in the quant config by
// micro codegen; litert never sets it).
std::set<PrimitivePtr> BuildSupportInt8OpsAllOps() {
  auto ops = BuildSupportInt8Ops();
  (void)ops.emplace(prim::kPrimEltwise);
  return ops;
}
}  // namespace

class QuantStrategyTest : public ::testing::Test {
 public:
  QuantStrategyTest() : strategy_(0, 0, {}, TargetDevice::CPU) {}

  // Build graph: output = prim(in_0, ..., in_{n-1}), all fp32 [3,5] parameters.
  CNodePtr BuildVariadicNode(const PrimitivePtr &prim, size_t data_input_num) {
    graph_ = std::make_shared<FuncGraph>();
    std::vector<AnfNodePtr> inputs{NewValueNode(prim)};
    for (size_t i = 0; i < data_input_num; i++) {
      auto param = graph_->add_parameter();
      param->set_name("input_" + std::to_string(i));
      param->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>{3, 5}));
      inputs.push_back(param);
    }
    auto cnode = graph_->NewCNode(inputs);
    cnode->set_fullname_with_scope("quant_strategy_test_node");
    cnode->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>{3, 5}));
    manager_ = Manage(graph_, true);
    return cnode;
  }

  FuncGraphPtr graph_ = nullptr;
  FuncGraphManagerPtr manager_ = nullptr;
  QuantStrategy strategy_;
};

// Guard (default whitelist, i.e. litert): Eltwise must NOT pass the full-quant
// whitelist. Its output is often the cancellation residual of same-range inputs
// (|a+b| << |a|,|b|), so int8 output quant steps are on the order of the signal itself
// and drop accuracy below the threshold (eltsum regression).
TEST_F(QuantStrategyTest, EltwiseNotFullQuantizedByDefault) {
  auto sum_prim = std::make_shared<ops::Eltwise>();
  sum_prim->set_mode(mindspore::EltwiseMode::SUM);
  auto cnode = BuildVariadicNode(sum_prim->GetPrim(), 2);
  ASSERT_NE(cnode, nullptr);
  ASSERT_NE(manager_, nullptr);
  EXPECT_FALSE(strategy_.CanOpFullQuantized(manager_, cnode, BuildSupportInt8Ops(), {}, {}));
}

// Guard (enable_all_ops = true, i.e. micro quant config): Eltwise IS quantized -
// required for int8_genuine micro models (verified 33/33 in the Sum feature
// verification). The gate lives in the whitelist built by full_quant_quantizer,
// not in QuantStrategy itself.
TEST_F(QuantStrategyTest, EltwiseFullQuantizedWithEnableAllOps) {
  auto sum_prim = std::make_shared<ops::Eltwise>();
  sum_prim->set_mode(mindspore::EltwiseMode::SUM);
  auto cnode = BuildVariadicNode(sum_prim->GetPrim(), 2);
  ASSERT_NE(cnode, nullptr);
  ASSERT_NE(manager_, nullptr);
  EXPECT_TRUE(strategy_.CanOpFullQuantized(manager_, cnode, BuildSupportInt8OpsAllOps(), {}, {}));
}

// Positive control: AddFusion (in the whitelist) is quantized under the same setup.
TEST_F(QuantStrategyTest, AddFusionIsFullQuantized) {
  auto cnode = BuildVariadicNode(prim::kPrimAddFusion, 2);
  ASSERT_NE(cnode, nullptr);
  auto manager = Manage(graph_, true);
  EXPECT_TRUE(strategy_.CanOpFullQuantized(manager, cnode, BuildSupportInt8Ops(), {}, {}));
}
}  // namespace lite::quant
}  // namespace mindspore
