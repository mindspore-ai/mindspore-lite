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
#include "infer/leaky_relu.h"
#include "infer/softsign.h"
#include "infer/softplus.h"
#include "infer/selu.h"
#include "infer/ops_func_impl/hswish.h"
#include "infer/ops_func_impl/sign.h"
#include "infer/return.h"
#include "infer/make_tuple.h"
#include "infer/cxx_api/activation.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "tools/converter/anf_transform.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "mindapi/ir/tensor.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/ops/op_def/nn_optimizer_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_e.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_h.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace {
constexpr int kNumInputIndex0 = 0;
constexpr int kNumInputIndex1 = 1;
constexpr int kNumInputIndex2 = 2;
constexpr int kNumInputIndex3 = 3;
}  // namespace
class ActivationMapperTest : public mindspore::CommonTest {
 public:
  ActivationMapperTest() = default;
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

CNodePtr InitActivationCNode(const FuncGraphPtr &func_graph, const ParameterPtr &data, ActivationType activation_type) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  if (data == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr!";
    return nullptr;
  }
  auto prim = std::make_shared<ops::Activation>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "relu prim is nullptr!";
    return nullptr;
  }
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  if (activation_type == mindspore::ActivationType::ELU) {
    prim->set_activation_type(mindspore::ActivationType::ELU);
    prim->set_alpha(0.1);
  } else if (activation_type == mindspore::ActivationType::GELU) {
    prim->set_activation_type(mindspore::ActivationType::GELU);
  } else if (activation_type == mindspore::ActivationType::RELU) {
    prim->set_activation_type(mindspore::ActivationType::RELU);
    prim->set_min_val(0);
    prim->set_max_val(0.5);
  } else if (activation_type == mindspore::ActivationType::RELU6) {
    prim->set_activation_type(mindspore::ActivationType::RELU6);
    prim->set_min_val(0);
    prim->set_max_val(0.5);
  } else if (activation_type == mindspore::ActivationType::SIGMOID) {
    prim->set_activation_type(mindspore::ActivationType::SIGMOID);
  } else if (activation_type == mindspore::ActivationType::HSIGMOID) {
    prim->set_activation_type(mindspore::ActivationType::HSIGMOID);
    prim->set_alpha(0.1);
  } else if (activation_type == mindspore::ActivationType::ABS) {
    // not UT test
    prim->set_activation_type(mindspore::ActivationType::ABS);
  } else if (activation_type == mindspore::ActivationType::SOFTSIGN) {
    // not UT test
    prim->set_activation_type(mindspore::ActivationType::SOFTSIGN);
  } else if (activation_type == mindspore::ActivationType::SOFTPLUS) {
    prim->set_activation_type(mindspore::ActivationType::SOFTPLUS);
  } else if (activation_type == mindspore::ActivationType::SELU) {
    prim->set_activation_type(mindspore::ActivationType::SELU);
  } else if (activation_type == mindspore::ActivationType::HSWISH) {
    // not UT test
    prim->set_activation_type(mindspore::ActivationType::HSWISH);
  } else if (activation_type == mindspore::ActivationType::SIGN) {
    // not UT test
    prim->set_activation_type(mindspore::ActivationType::SIGN);
  } else if (activation_type == mindspore::ActivationType::TANH) {
    prim->set_activation_type(mindspore::ActivationType::TANH);
  } else if (activation_type == mindspore::ActivationType::LEAKY_RELU) {
    prim->set_activation_type(mindspore::ActivationType::LEAKY_RELU);
    prim->set_alpha(0.1);
  } else if (activation_type == mindspore::ActivationType::HSWISH) {
    prim->set_activation_type(mindspore::ActivationType::HSWISH);
  } else {
    return nullptr;
  }
  auto prim_c = prim->GetPrim();
  if (prim_c == nullptr) {
    MS_LOG(ERROR) << "get prim_c failed, relu node prim_c is nullptr!";
    return nullptr;
  }

  AnfNodePtrList inputs = {data};
  auto cnode = func_graph->NewCNode(prim_c, inputs);
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "create relu node failed, relu node is nullptr!";
    return nullptr;
  }
  cnode->set_fullname_with_scope("relu_cnode");
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
}  //  namespace

TEST_F(ActivationMapperTest, ELUNodeMapperTest) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildFloatValueParameterNode(func_graph, 0.5, "input_data");
  ASSERT_NE(data_param, nullptr);
  auto cnode = InitActivationCNode(func_graph, data_param, mindspore::ActivationType::ELU);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameActivation);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto cnode_input_1 = cnode->input(kNumInputIndex1);
  auto input_param = utils::isa<ParameterPtr>(cnode_input_1);
  ASSERT_EQ(input_param, true);
  auto cnode_input_0 = cnode->input(0);
  auto input_value_node = utils::isa<ValueNodePtr>(cnode_input_0);
  ASSERT_EQ(input_value_node, true);
  auto is_true = opt::CheckPrimitiveType(cnode, prim::kPrimElu);
  ASSERT_EQ(is_true, true);
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  auto new_prim = GetValueNode<PrimitivePtr>(value_node);
  auto attr_size = new_prim->attrs().size();
  ASSERT_EQ(attr_size, 2);
}

TEST_F(ActivationMapperTest, GELUNodeMapperTest) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildFloatValueParameterNode(func_graph, 0.5, "input_data");
  ASSERT_NE(data_param, nullptr);
  auto cnode = InitActivationCNode(func_graph, data_param, mindspore::ActivationType::GELU);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameActivation);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto cnode_input_1 = cnode->input(kNumInputIndex1);
  auto input_param = utils::isa<ParameterPtr>(cnode_input_1);
  ASSERT_EQ(input_param, true);
  auto cnode_input_0 = cnode->input(0);
  auto input_value_node = utils::isa<ValueNodePtr>(cnode_input_0);
  ASSERT_EQ(input_value_node, true);
  auto is_true = opt::CheckPrimitiveType(cnode, prim::kPrimGeLU);
  ASSERT_EQ(is_true, true);
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  auto new_prim = GetValueNode<PrimitivePtr>(value_node);
  auto attr_size = new_prim->attrs().size();
  ASSERT_EQ(attr_size, 1);
}

TEST_F(ActivationMapperTest, RELUNodeMapperTest) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildFloatValueParameterNode(func_graph, 0.5, "input_data");
  ASSERT_NE(data_param, nullptr);
  auto cnode = InitActivationCNode(func_graph, data_param, mindspore::ActivationType::RELU);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameActivation);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto cnode_input_1 = cnode->input(1);
  auto input_param = utils::isa<ParameterPtr>(cnode_input_1);
  ASSERT_EQ(input_param, true);
  auto cnode_input_0 = cnode->input(0);
  auto input_value_node = utils::isa<ValueNodePtr>(cnode_input_0);
  ASSERT_EQ(input_value_node, true);
  auto is_true = opt::CheckPrimitiveType(cnode, prim::kPrimReLU);
  ASSERT_EQ(is_true, true);
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  auto new_prim = GetValueNode<PrimitivePtr>(value_node);
  auto attr_size = new_prim->attrs().size();
  ASSERT_EQ(attr_size, kNumInputIndex3);
}

TEST_F(ActivationMapperTest, RELU6NodeMapperTest) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildFloatValueParameterNode(func_graph, 0.5, "input_data");
  ASSERT_NE(data_param, nullptr);
  auto cnode = InitActivationCNode(func_graph, data_param, mindspore::ActivationType::RELU6);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameActivation);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto cnode_input_1 = cnode->input(kNumInputIndex1);
  auto input_param = utils::isa<ParameterPtr>(cnode_input_1);
  ASSERT_EQ(input_param, true);
  auto cnode_input_0 = cnode->input(0);
  auto input_value_node = utils::isa<ValueNodePtr>(cnode_input_0);
  ASSERT_EQ(input_value_node, true);
  auto is_true = opt::CheckPrimitiveType(cnode, prim::kPrimReLU6);
  ASSERT_EQ(is_true, true);
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  auto new_prim = GetValueNode<PrimitivePtr>(value_node);
  auto attr_size = new_prim->attrs().size();
  ASSERT_EQ(attr_size, kNumInputIndex3);
}

TEST_F(ActivationMapperTest, SIGMOIDNodeMapperTest) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildFloatValueParameterNode(func_graph, 0.5, "input_data");
  ASSERT_NE(data_param, nullptr);
  auto cnode = InitActivationCNode(func_graph, data_param, mindspore::ActivationType::SIGMOID);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameActivation);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto cnode_input_1 = cnode->input(kNumInputIndex1);
  auto input_param = utils::isa<ParameterPtr>(cnode_input_1);
  ASSERT_EQ(input_param, true);
  auto cnode_input_0 = cnode->input(0);
  auto input_value_node = utils::isa<ValueNodePtr>(cnode_input_0);
  ASSERT_EQ(input_value_node, true);
  auto is_true = opt::CheckPrimitiveType(cnode, prim::kPrimSigmoid);
  ASSERT_EQ(is_true, true);
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  auto new_prim = GetValueNode<PrimitivePtr>(value_node);
  auto attr_size = new_prim->attrs().size();
  ASSERT_EQ(attr_size, kNumInputIndex1);
}

TEST_F(ActivationMapperTest, HSIGMOIDNodeMapperTest) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildFloatValueParameterNode(func_graph, 0.5, "input_data");
  ASSERT_NE(data_param, nullptr);
  auto cnode = InitActivationCNode(func_graph, data_param, mindspore::ActivationType::HSIGMOID);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameActivation);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto cnode_input_1 = cnode->input(kNumInputIndex1);
  auto input_param = utils::isa<ParameterPtr>(cnode_input_1);
  ASSERT_EQ(input_param, true);
  auto cnode_input_0 = cnode->input(0);
  auto input_value_node = utils::isa<ValueNodePtr>(cnode_input_0);
  ASSERT_EQ(input_value_node, true);
  auto is_true = opt::CheckPrimitiveType(cnode, prim::kPrimHSigmoid);
  ASSERT_EQ(is_true, true);
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  auto new_prim = GetValueNode<PrimitivePtr>(value_node);
  auto attr_size = new_prim->attrs().size();
  ASSERT_EQ(attr_size, kNumInputIndex2);
}

TEST_F(ActivationMapperTest, SOFTPLUSNodeMapperTest) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildFloatValueParameterNode(func_graph, 0.5, "input_data");
  ASSERT_NE(data_param, nullptr);
  auto cnode = InitActivationCNode(func_graph, data_param, mindspore::ActivationType::SOFTPLUS);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameActivation);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto cnode_input_1 = cnode->input(kNumInputIndex1);
  auto input_param = utils::isa<ParameterPtr>(cnode_input_1);
  ASSERT_EQ(input_param, true);
  auto cnode_input_0 = cnode->input(0);
  auto input_value_node = utils::isa<ValueNodePtr>(cnode_input_0);
  ASSERT_EQ(input_value_node, true);
  auto is_true = opt::CheckPrimitiveType(cnode, prim::kPrimSoftplus);
  ASSERT_EQ(is_true, true);
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  auto new_prim = GetValueNode<PrimitivePtr>(value_node);
  auto attr_size = new_prim->attrs().size();
  ASSERT_EQ(attr_size, kNumInputIndex3);
}

TEST_F(ActivationMapperTest, SELUNodeMapperTest) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildFloatValueParameterNode(func_graph, 0.5, "input_data");
  ASSERT_NE(data_param, nullptr);
  auto cnode = InitActivationCNode(func_graph, data_param, mindspore::ActivationType::SELU);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameActivation);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto cnode_input_1 = cnode->input(kNumInputIndex1);
  auto input_param = utils::isa<ParameterPtr>(cnode_input_1);
  ASSERT_EQ(input_param, true);
  auto cnode_input_0 = cnode->input(0);
  auto input_value_node = utils::isa<ValueNodePtr>(cnode_input_0);
  ASSERT_EQ(input_value_node, true);
  auto is_true = opt::CheckPrimitiveType(cnode, prim::kPrimSeLU);
  ASSERT_EQ(is_true, true);
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  auto new_prim = GetValueNode<PrimitivePtr>(value_node);
  auto attr_size = new_prim->attrs().size();
  ASSERT_EQ(attr_size, kNumInputIndex3);
}

TEST_F(ActivationMapperTest, LEAKY_RELUNodeMapperTest) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildFloatValueParameterNode(func_graph, 0.5, "input_data");
  ASSERT_NE(data_param, nullptr);
  auto cnode = InitActivationCNode(func_graph, data_param, mindspore::ActivationType::LEAKY_RELU);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameActivation);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto cnode_input_1 = cnode->input(kNumInputIndex1);
  auto input_param = utils::isa<ParameterPtr>(cnode_input_1);
  ASSERT_EQ(input_param, true);
  auto cnode_input_0 = cnode->input(0);
  auto input_value_node = utils::isa<ValueNodePtr>(cnode_input_0);
  ASSERT_EQ(input_value_node, true);
  auto is_true = opt::CheckPrimitiveType(cnode, prim::kPrimLeakyRelu);
  ASSERT_EQ(is_true, true);
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  auto new_prim = GetValueNode<PrimitivePtr>(value_node);
  auto attr_size = new_prim->attrs().size();
  ASSERT_EQ(attr_size, kNumInputIndex2);
}

TEST_F(ActivationMapperTest, TANHNodeMapperTest) {
  auto func_graph = std::make_shared<FuncGraph>();
  ASSERT_NE(func_graph, nullptr);
  auto manager = MakeManager();
  ASSERT_NE(manager, nullptr);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  auto data_param = opt::BuildFloatValueParameterNode(func_graph, 0.5, "input_data");
  ASSERT_NE(data_param, nullptr);
  auto cnode = InitActivationCNode(func_graph, data_param, mindspore::ActivationType::TANH);
  ASSERT_NE(cnode, nullptr);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto mapper = lite::PrimitiveMapperRegister::GetInstance().GetPrimitiveMapper(ops::kNameActivation);
  ASSERT_NE(mapper, nullptr);
  auto status = mapper->Mapper(cnode);
  ASSERT_EQ(status, lite::RET_OK);
  ASSERT_EQ(cnode->inputs().size(), kNumInputIndex2);
  auto cnode_input_1 = cnode->input(kNumInputIndex1);
  auto input_param = utils::isa<ParameterPtr>(cnode_input_1);
  ASSERT_EQ(input_param, true);
  auto cnode_input_0 = cnode->input(0);
  auto input_value_node = utils::isa<ValueNodePtr>(cnode_input_0);
  ASSERT_EQ(input_value_node, true);
  auto is_true = opt::CheckPrimitiveType(cnode, prim::kPrimTanh);
  ASSERT_EQ(is_true, true);
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  auto new_prim = GetValueNode<PrimitivePtr>(value_node);
  auto attr_size = new_prim->attrs().size();
  ASSERT_EQ(attr_size, 1);
}
}  // namespace mindspore
