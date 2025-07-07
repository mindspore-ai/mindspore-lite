/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "common/common_test.h"
#include "infer/return.h"
#include "infer/make_tuple.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "tools/converter/anf_transform.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "infer/cxx_api/add_fusion.h"
#include "infer/cxx_api/mul_fusion.h"
#include "infer/cxx_api/div_fusion.h"
#include "infer/cxx_api/sub_fusion.h"
#include "infer/cxx_api/exp_fusion.h"
#include "infer/ops_func_impl/tan.h"
#include "infer/cxx_api/pow_fusion.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_e.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_p.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace {
constexpr int kNumInputSize = 3;
}  // namespace
class ArithmeticMapperTest : public mindspore::CommonTest {
 public:
  ArithmeticMapperTest() = default;
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
      MS_LOG(ERROR) << "create return node failed";
      return nullptr;
    }
    return_input_cnode->set_fullname_with_scope("return_node");
    return_input = return_input_cnode;
  }
  auto return_prim = std::make_shared<ops::Return>();
  if (return_prim == nullptr) {
    MS_LOG(ERROR) << "create return primitive failed!";
    return nullptr;
  }
  auto return_prim_c = return_prim->GetPrim();
  if (return_prim_c == nullptr) {
    MS_LOG(ERROR) << "prim_c is nullptr!";
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

CNodePtr InitAddFusionCNode(const FuncGraphPtr &func_graph, const ParameterPtr &input1, const ParameterPtr &input2) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  if (input1 == nullptr || input2 == nullptr) {
    MS_LOG(ERROR) << "AddFusion cnode's input is nullptr!";
    return nullptr;
  }
  auto prim = std::make_unique<ops::AddFusion>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "AddFusion prim is nullptr!";
    return nullptr;
  }
  auto prim_c = prim->GetPrim();
  if (prim_c == nullptr) {
    MS_LOG(ERROR) << "get prim_c failed, AddFusion node prim_c is nullptr!";
    return nullptr;
  }

  AnfNodePtrList inputs = {input1, input2};
  auto cnode = func_graph->NewCNode(prim_c, inputs);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "create AddFusion node failed, AddFusion node is nullptr!";
    return nullptr;
  }
  cnode->set_fullname_with_scope("Add_fusion_node");
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{});
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "create AddFusion node abstract failed, abstract is nullptr!";
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

CNodePtr InitDivFusionCNode(const FuncGraphPtr &func_graph, const ParameterPtr &input1, const ParameterPtr &input2) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  if (input1 == nullptr || input2 == nullptr) {
    MS_LOG(ERROR) << "DivFusion cnode's input is nullptr!";
    return nullptr;
  }
  auto prim = std::make_unique<ops::DivFusion>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "DivFusion prim is nullptr!";
    return nullptr;
  }
  auto prim_c = prim->GetPrim();
  if (prim_c == nullptr) {
    MS_LOG(ERROR) << "get prim_c failed, DivFusion node prim_c is nullptr!";
    return nullptr;
  }
  prim->AddAttr(ops::kOriginalOpName, api::MakeValue(std::string(ops::kNameRealDiv)));
  AnfNodePtrList inputs = {input1, input2};
  auto cnode = func_graph->NewCNode(prim_c, inputs);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "create DivFusion node failed, DivFusion node is nullptr!";
    return nullptr;
  }
  cnode->set_fullname_with_scope("div_fusion_node");
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{});
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "create DivFusion node abstract failed, abstract is nullptr!";
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
CNodePtr InitMulFusionCNode(const FuncGraphPtr &func_graph, const ParameterPtr &input1, const ParameterPtr &input2) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  if (input1 == nullptr || input2 == nullptr) {
    MS_LOG(ERROR) << "MulFusion cnode's input is nullptr!";
    return nullptr;
  }
  auto prim = std::make_unique<ops::MulFusion>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "MulFusion prim is nullptr!";
    return nullptr;
  }
  auto prim_c = prim->GetPrim();
  if (prim_c == nullptr) {
    MS_LOG(ERROR) << "get prim_c failed, MulFusion node prim_c is nullptr!";
    return nullptr;
  }

  AnfNodePtrList inputs = {input1, input2};
  auto cnode = func_graph->NewCNode(prim_c, inputs);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "create MulFusion node failed, MulFusion node is nullptr!";
    return nullptr;
  }
  cnode->set_fullname_with_scope("mul_fusion_node");
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{});
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "create clip node abstract failed, abstract is nullptr!";
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

CNodePtr InitPowFusionCNode(const FuncGraphPtr &func_graph, const ParameterPtr &input1, const ParameterPtr &input2) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  if (input1 == nullptr || input2 == nullptr) {
    MS_LOG(ERROR) << "PowFusion cnode's input is nullptr!";
    return nullptr;
  }
  auto prim = std::make_unique<ops::PowFusion>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "PowFusion prim is nullptr!";
    return nullptr;
  }
  auto prim_c = prim->GetPrim();
  if (prim_c == nullptr) {
    MS_LOG(ERROR) << "get prim_c failed, relu node prim_c is nullptr!";
    return nullptr;
  }

  AnfNodePtrList inputs = {input1, input2};
  auto cnode = func_graph->NewCNode(prim_c, inputs);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "create PowFusion node failed, PowFusion node is nullptr!";
    return nullptr;
  }
  cnode->set_fullname_with_scope("pow_fusion_node");
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{});
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "create PowFusion node abstract failed, abstract is nullptr!";
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

CNodePtr InitSubFusionCNode(const FuncGraphPtr &func_graph, const ParameterPtr &input1, const ParameterPtr &input2) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  if (input1 == nullptr || input2 == nullptr) {
    MS_LOG(ERROR) << "SubFusion cnode's input is nullptr!";
    return nullptr;
  }
  auto prim = std::make_unique<ops::SubFusion>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "SubFusion prim is nullptr!";
    return nullptr;
  }
  auto prim_c = prim->GetPrim();
  if (prim_c == nullptr) {
    MS_LOG(ERROR) << "get prim_c failed, SubFusion node prim_c is nullptr!";
    return nullptr;
  }

  AnfNodePtrList inputs = {input1, input2};
  auto cnode = func_graph->NewCNode(prim_c, inputs);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "create SubFusion node failed, SubFusion node is nullptr!";
    return nullptr;
  }
  cnode->set_fullname_with_scope("sub_fusion_node");
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{});
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "create SubFusion node abstract failed, abstract is nullptr!";
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

CNodePtr InitExpFusionCNode(const FuncGraphPtr &func_graph, const ParameterPtr &input1, const ParameterPtr &input2) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  if (input1 == nullptr || input2 == nullptr) {
    MS_LOG(ERROR) << "ExpFusion cnode's input is nullptr!";
    return nullptr;
  }
  auto prim = std::make_unique<ops::ExpFusion>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "ExpFusion prim is nullptr!";
    return nullptr;
  }
  auto prim_c = prim->GetPrim();
  if (prim_c == nullptr) {
    MS_LOG(ERROR) << "get prim_c failed, ExpFusion node prim_c is nullptr!";
    return nullptr;
  }

  AnfNodePtrList inputs = {input1, input2};
  auto cnode = func_graph->NewCNode(prim_c, inputs);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "create ExpFusion node failed, ExpFusion node is nullptr!";
    return nullptr;
  }
  cnode->set_fullname_with_scope("Add_fusion_node");
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{});
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "create ExpFusion node abstract failed, abstract is nullptr!";
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

TEST_F(ArithmeticMapperTest, TestAddFusion) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto input1 = opt::BuildFloatValueParameterNode(func_graph, 0.5, "input_param_1");
  ASSERT_NE(input1, nullptr);
  auto input2 = opt::BuildFloatValueParameterNode(func_graph, 0.5, "input_param_2");
  ASSERT_NE(input2, nullptr);
  auto cnode = InitAddFusionCNode(func_graph, input1, input2);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputSize);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameAddFusion);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), kNumInputSize);
  const auto &new_prim = GetCNodePrimitive(cnode);
  auto prim_name = new_prim->name();
  ASSERT_EQ(prim_name, "Add");
}

TEST_F(ArithmeticMapperTest, TestDivFusion) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto input1 = opt::BuildFloatValueParameterNode(func_graph, 0.1, "input_param_1");
  ASSERT_NE(input1, nullptr);
  auto input2 = opt::BuildFloatValueParameterNode(func_graph, 0.2, "input_param_2");
  ASSERT_NE(input2, nullptr);

  auto cnode = InitDivFusionCNode(func_graph, input1, input2);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputSize);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameDivFusion);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), kNumInputSize);
  const auto &new_prim = GetCNodePrimitive(cnode);
  auto prim_name = new_prim->name();
  ASSERT_EQ(prim_name, "RealDiv");
}

TEST_F(ArithmeticMapperTest, TestMulFusion) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto input1 = opt::BuildFloatValueParameterNode(func_graph, 0.1, "input_param_1");
  ASSERT_NE(input1, nullptr);
  auto input2 = opt::BuildFloatValueParameterNode(func_graph, 0.2, "input_param_2");
  ASSERT_NE(input2, nullptr);

  auto cnode = InitMulFusionCNode(func_graph, input1, input2);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputSize);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameMulFusion);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), kNumInputSize);
  const auto &new_prim = GetCNodePrimitive(cnode);
  auto prim_name = new_prim->name();
  ASSERT_EQ(prim_name, "Mul");
}

TEST_F(ArithmeticMapperTest, TestPowFusion) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto input1 = opt::BuildFloatValueParameterNode(func_graph, 0.1, "input_param_1");
  ASSERT_NE(input1, nullptr);
  auto input2 = opt::BuildFloatValueParameterNode(func_graph, 0.2, "input_param_2");
  ASSERT_NE(input2, nullptr);

  auto cnode = InitPowFusionCNode(func_graph, input1, input2);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputSize);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNamePowFusion);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), kNumInputSize);
  const auto &new_prim = GetCNodePrimitive(cnode);
  auto prim_name = new_prim->name();
  ASSERT_EQ(prim_name, "Pow");
}

TEST_F(ArithmeticMapperTest, TestSubFusion) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto input1 = opt::BuildFloatValueParameterNode(func_graph, 0.1, "input_param_1");
  ASSERT_NE(input1, nullptr);
  auto input2 = opt::BuildFloatValueParameterNode(func_graph, 0.2, "input_param_2");
  ASSERT_NE(input2, nullptr);

  auto cnode = InitSubFusionCNode(func_graph, input1, input2);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputSize);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameSubFusion);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), kNumInputSize);
  const auto &new_prim = GetCNodePrimitive(cnode);
  auto prim_name = new_prim->name();
  ASSERT_EQ(prim_name, "Sub");
}

TEST_F(ArithmeticMapperTest, TestExpFusion) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto input1 = opt::BuildFloatValueParameterNode(func_graph, 0.1, "input_param_1");
  ASSERT_NE(input1, nullptr);
  auto input2 = opt::BuildFloatValueParameterNode(func_graph, 0.2, "input_param_2");
  ASSERT_NE(input2, nullptr);

  auto cnode = InitExpFusionCNode(func_graph, input1, input2);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputSize);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameExpFusion);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), kNumInputSize);
  const auto &new_prim = GetCNodePrimitive(cnode);
  auto prim_name = new_prim->name();
  ASSERT_EQ(prim_name, "Exp");
}
}  // namespace mindspore
