/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/mapper/fused_batchnorm_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "src/common/log_util.h"
#include "ops_utils/op_utils.h"
#include "common/common_test.h"
#include "infer/return.h"
#include "infer/make_tuple.h"
#include "mindspore/ops/op_def/nn_optimizer_ops.h"
#include "mindspore/ops/infer/fused_batch_norm.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "tools/converter/anf_transform.h"
#include "mindapi/ir/tensor.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/core/include/ir/dtype/number.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
class FusedBatchNormMapperTest : public mindspore::CommonTest {
 public:
  FusedBatchNormMapperTest() = default;
};

namespace {
CNodePtr AddReturn(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &return_inputs) {
  if (return_inputs.empty()) {
    MS_LOG(ERROR) << "return node's input is empty!";
    return nullptr;
  }
  if (graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  AnfNodePtr return_input;
  if (return_inputs.size() == 1) {
    return_input = return_inputs.front();
  } else {
    auto make_tuple_prim_ptr = std::make_shared<ops::MakeTuple>();
    if (make_tuple_prim_ptr == nullptr) {
      MS_LOG(ERROR) << "new MakeTuple failed";
      return nullptr;
    }
    auto prim_c = make_tuple_prim_ptr->GetPrim();
    if (prim_c == nullptr) {
      MS_LOG(ERROR) << "prim_c is nullptr!";
      return nullptr;
    }
    auto return_input_cnode = graph->NewCNode(prim_c, return_inputs);
    if (return_input_cnode == nullptr) {
      MS_LOG(ERROR) << "new make tuple cnode failed";
      return nullptr;
    }
    return_input_cnode->set_fullname_with_scope("return_tuple");
    return_input = return_input_cnode;
  }
  auto return_prim = std::make_shared<ops::Return>();
  if (return_prim == nullptr) {
    MS_LOG(ERROR) << "create return primitive failed!";
    return nullptr;
  }
  auto return_prim_c = return_prim->GetPrim();
  if (return_prim_c == nullptr) {
    MS_LOG(ERROR) << "return_prim_c is nullptr!";
    return nullptr;
  }
  auto return_cnode = graph->NewCNode(return_prim_c, {return_input});
  if (return_cnode == nullptr) {
    MS_LOG(ERROR) << "create Return failed";
    return nullptr;
  }
  return_cnode->set_fullname_with_scope("ReturnNode");
  graph->set_return(return_cnode);
  return return_cnode;
}

CNodePtr InitFusedBatchNormNodeWithInput(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  auto prim = std::make_shared<ops::FusedBatchNorm>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "FusedBatchNorm prim is nullptr!";
    return nullptr;
  }
  prim->set_epsilon(0.1);
  prim->set_momentum(0.2);
  auto prim_c = prim->GetPrim();
  if (prim_c == nullptr) {
    MS_LOG(ERROR) << "get prim_c failed, FusedBatchNorm node prim_c is nullptr!";
    return nullptr;
  }
  AnfNodePtrList inputs = {};
  auto cnode = func_graph->NewCNode(prim_c, inputs);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "create FusedBatchNorm node failed, FusedBatchNorm node node is nullptr!";
    return nullptr;
  }
  cnode->set_fullname_with_scope("fused_batch_norm");
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{});
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "create FusedBatchNorm node node abstract failed, abstract is nullptr!";
    return nullptr;
  }
  cnode->set_abstract(abstract);
  auto ret = AddReturn(func_graph, {cnode});
  if (ret == nullptr) {
    MS_LOG(ERROR) << "add return node failed!";
    return nullptr;
  }
  return cnode;
}
}  //  namespace

TEST_F(FusedBatchNormMapperTest, InitFusedBatchNormNodeWithInput) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildIntValueParameterNode(func_graph, 0, "input_data");
  ASSERT_NE(data_param, nullptr);
  auto cnode = InitFusedBatchNormNodeWithInput(func_graph);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), 1);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameFusedBatchNorm);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), 1);
  auto cnode_input_0 = cnode->input(0);
  auto input_value_node = utils::isa<ValueNodePtr>(cnode_input_0);
  ASSERT_EQ(input_value_node, true);
  const auto &origin_prim = GetCNodePrimitive(cnode);
  auto prim_name = origin_prim->name();
  ASSERT_EQ(prim_name, "FusedBatchNorm");
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  auto new_prim = GetValueNode<PrimitivePtr>(value_node);
  auto attr_size = new_prim->attrs().size();
  ASSERT_EQ(attr_size, 5);
  MS_LOG(INFO) << "PASS";
}
}  // namespace mindspore
