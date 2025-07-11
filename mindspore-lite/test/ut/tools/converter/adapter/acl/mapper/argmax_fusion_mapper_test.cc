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

#include <memory>
#include "tools/converter/adapter/acl/mapper/argmax_fusion_mapper.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "src/common/log_util.h"
#include "ops_utils/op_utils.h"
#include "common/common_test.h"
#include "infer/return.h"
#include "infer/make_tuple.h"
#include "mindspore/ops/op_def/nn_optimizer_ops.h"
#include "mindspore/ops/infer/cxx_api/arg_max_fusion.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "tools/converter/anf_transform.h"
#include "mindapi/ir/tensor.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/core/include/ir/dtype/number.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace {
constexpr int kNumInputNum1 = 1;
constexpr int kNumInputNum2 = 2;
constexpr int kNumInputNum3 = 3;
constexpr int kNumInputIndex0 = 0;
constexpr int kNumInputIndex1 = 1;
constexpr int kArgMinFusionAttrSize5 = 5;
}  // namespace
class ArgmaxFusionMapperTest : public mindspore::CommonTest {
 public:
  ArgmaxFusionMapperTest() = default;
};

namespace {
CNodePtr AddReturn(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &return_inputs) {
  if (graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  if (return_inputs.empty()) {
    MS_LOG(ERROR) << "return node's input is empty!";
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
    return_input_cnode->set_fullname_with_scope("return tuple");
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
  return_cnode->set_fullname_with_scope("Return");
  graph->set_return(return_cnode);
  return return_cnode;
}

CNodePtr InitArgmaxFusionNodeWithInputSize1(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  auto prim = std::make_shared<ops::ArgMaxFusion>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "argmax fusion prim is nullptr!";
    return nullptr;
  }
  prim->set_axis(1);
  prim->set_keep_dims(true);
  auto prim_c = prim->GetPrim();
  if (prim_c == nullptr) {
    MS_LOG(ERROR) << "get prim_c failed, argmax fusion node prim_c is nullptr!";
    return nullptr;
  }
  AnfNodePtrList inputs = {};
  auto cnode = func_graph->NewCNode(prim_c, inputs);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "create argmax fusion node failed, argmax fusion node node is nullptr!";
    return nullptr;
  }
  cnode->set_fullname_with_scope("argmax_fusion_node");
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{});
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "create argmax fusion node node abstract failed, abstract is nullptr!";
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

CNodePtr InitArgmaxFusionNodeWithInputSize2(const FuncGraphPtr &func_graph, const ParameterPtr &data) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  if (data == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  auto prim = std::make_shared<ops::ArgMaxFusion>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "argmax fusion prim is nullptr!";
    return nullptr;
  }
  prim->set_axis(1);
  prim->set_keep_dims(true);
  auto prim_c = prim->GetPrim();
  if (prim_c == nullptr) {
    MS_LOG(ERROR) << "get prim_c failed, argmax fusion node prim_c is nullptr!";
    return nullptr;
  }
  AnfNodePtrList inputs = {data};
  auto cnode = func_graph->NewCNode(prim_c, inputs);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "create argmax fusion node failed, argmax fusion node node is nullptr!";
    return nullptr;
  }
  cnode->set_fullname_with_scope("argmax_fusion_node");
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{});
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "create argmax fusion node node abstract failed, abstract is nullptr!";
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

CNodePtr InitArgmaxFusionNodeWithInputSize3(const FuncGraphPtr &func_graph, const ParameterPtr &input0,
                                            const ParameterPtr &input1) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  if (input0 == nullptr) {
    MS_LOG(ERROR) << "input0 is nullptr!";
    return nullptr;
  }
  if (input1 == nullptr) {
    MS_LOG(ERROR) << "input1 is nullptr!";
    return nullptr;
  }
  auto prim = std::make_shared<ops::ArgMaxFusion>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "argmax fusion prim is nullptr!";
    return nullptr;
  }
  prim->set_axis(1);
  prim->set_keep_dims(true);
  auto prim_c = prim->GetPrim();
  if (prim_c == nullptr) {
    MS_LOG(ERROR) << "get prim_c failed, argmax fusion node prim_c is nullptr!";
    return nullptr;
  }
  AnfNodePtrList inputs = {input0, input1};
  auto cnode = func_graph->NewCNode(prim_c, inputs);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "create argmax fusion node failed, argmax fusion node node is nullptr!";
    return nullptr;
  }
  cnode->set_fullname_with_scope("argmax_fusion_node");
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{});
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "create argmax fusion node node abstract failed, abstract is nullptr!";
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

CNodePtr InitArgmaxFusionNodeWithoutKeepDims(const FuncGraphPtr &func_graph, const ParameterPtr &data) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  if (data == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  auto prim = std::make_shared<ops::ArgMaxFusion>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "argmax fusion prim is nullptr!";
    return nullptr;
  }
  prim->set_axis(1);
  auto prim_c = prim->GetPrim();
  if (prim_c == nullptr) {
    MS_LOG(ERROR) << "get prim_c failed, argmax fusion node prim_c is nullptr!";
    return nullptr;
  }
  AnfNodePtrList inputs = {data};
  auto cnode = func_graph->NewCNode(prim_c, inputs);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "create argmax fusion node failed, argmax fusion node node is nullptr!";
    return nullptr;
  }
  cnode->set_fullname_with_scope("argmax_fusion_node");
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{});
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "create argmax fusion node node abstract failed, abstract is nullptr!";
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

TEST_F(ArgmaxFusionMapperTest, ArgmaxFusionNodeMapperWithInputSize1) {
  // create funcgraph
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  // init argmax cnode
  auto cnode = InitArgmaxFusionNodeWithInputSize1(func_graph);
  // check init cnode
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputNum1);
  // UT for func
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameArgMaxFusion);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  // check result
  ASSERT_EQ(status, lite::RET_ERROR);
  ASSERT_EQ(cnode->inputs().size(), kNumInputNum1);
  MS_LOG(INFO) << "PASS";
}

TEST_F(ArgmaxFusionMapperTest, ArgmaxFusionNodeMapperWithInputSize2) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildIntValueParameterNode(func_graph, 0, "input_data");
  ASSERT_NE(data_param, nullptr);
  auto cnode = InitArgmaxFusionNodeWithInputSize2(func_graph, data_param);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputNum2);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameArgMaxFusion);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), 4);
  auto cnode_input_0 = cnode->input(kNumInputIndex0);
  auto input_value_node = utils::isa<ValueNodePtr>(cnode_input_0);
  ASSERT_EQ(input_value_node, true);
  auto cnode_input_1 = cnode->input(kNumInputIndex1);
  auto input_param = utils::isa<ParameterPtr>(cnode_input_1);
  ASSERT_EQ(input_param, true);
  const auto &origin_prim = GetCNodePrimitive(cnode);
  auto prim_name = origin_prim->name();
  ASSERT_EQ(prim_name, "ArgMaxWithValue");
  auto value_node = cnode->input(kNumInputIndex0)->cast<ValueNodePtr>();
  auto new_prim = GetValueNode<PrimitivePtr>(value_node);
  auto attr_size = new_prim->attrs().size();
  ASSERT_EQ(attr_size, 4);
  auto output_type = new_prim->GetAttr("output_type");
  ASSERT_EQ(output_type, nullptr);
  MS_LOG(INFO) << "PASS";
}

TEST_F(ArgmaxFusionMapperTest, ArgmaxFusionNodeMapperWithInputSize3) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildIntValueParameterNode(func_graph, 0, "input_data");
  ASSERT_NE(data_param, nullptr);
  auto data_param1 = opt::BuildIntValueParameterNode(func_graph, 1, "input_data1");
  ASSERT_NE(data_param1, nullptr);
  auto cnode = InitArgmaxFusionNodeWithInputSize3(func_graph, data_param, data_param1);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputNum3);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameArgMaxFusion);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_ERROR);
  ASSERT_EQ(cnode->inputs().size(), kNumInputNum3);
  MS_LOG(INFO) << "PASS";
}

TEST_F(ArgmaxFusionMapperTest, InitArgmaxFusionNodeWithoutKeepDims) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildIntValueParameterNode(func_graph, 0, "input_data");
  ASSERT_NE(data_param, nullptr);
  auto cnode = InitArgmaxFusionNodeWithoutKeepDims(func_graph, data_param);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), 2);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameArgMaxFusion);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), 2);
  auto cnode_input_0 = cnode->input(kNumInputIndex0);
  auto input_value_node = utils::isa<ValueNodePtr>(cnode_input_0);
  ASSERT_EQ(input_value_node, true);
  auto cnode_input_1 = cnode->input(kNumInputIndex1);
  auto input_param = utils::isa<ParameterPtr>(cnode_input_1);
  ASSERT_EQ(input_param, true);
  const auto &origin_prim = GetCNodePrimitive(cnode);
  auto prim_name = origin_prim->name();
  ASSERT_EQ(prim_name, "ArgMaxV2");
  auto value_node = cnode->input(kNumInputIndex0)->cast<ValueNodePtr>();
  auto new_prim = GetValueNode<PrimitivePtr>(value_node);
  auto attr_size = new_prim->attrs().size();
  ASSERT_EQ(attr_size, 4);
  auto output_type = new_prim->GetAttr("output_type");
  ASSERT_NE(output_type, nullptr);
  MS_LOG(INFO) << "PASS";
}

}  // namespace mindspore
