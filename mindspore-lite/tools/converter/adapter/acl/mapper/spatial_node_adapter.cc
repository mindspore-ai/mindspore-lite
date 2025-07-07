/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "tools/converter/adapter/acl/mapper/spatial_node_adapter.h"
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "base/base.h"
#include "include/errorcode.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "infer/fused_batch_norm.h"
#include "infer/cxx_api/layer_norm_fusion.h"
#include "infer/cxx_api/gegluv2.h"
#include "infer/instance_norm.h"
#include "infer/add_layernorm.h"
#include "infer/custom.h"
#include "mindspore/ops/op_def/nn_op_name.h"
#include "infer/stack.h"
#include "mindspore/ops/op_def/op_name.h"
#include "infer/tuple_get_item.h"
#include "infer/make_tuple.h"
#include "infer/group_norm_silu.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "include/registry/converter_context.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kCnodeInputMinNum = 2;
constexpr auto kAnfPrimitiveIndex = 0;
constexpr auto kNamewiEltwise = "Eltwise";
const std::set<std::string> kCNodeWithMultiOutputs = {
  kBatchNormOpName,          ops::kNameFusedBatchNorm,  ops::kNameInstanceNorm, ops::kNameLayerNorm,
  ops::kNameLayerNormFusion, ops::kNameArgMaxWithValue, ops::kNameGeGluV2,      ops::kNameGroupNormSilu};

const std::set<std::string> kCNodeWithDynamicInput = {kNamewiEltwise, ops::kNameConcat, ops::kNameStack,
                                                      acl::kNameConcatV2};
}  // namespace

static STATUS AdapteNodeWithDynamicInput(const CNodePtr &cnode) {
  // For the 3rd model, the input of the multi-input operator has been expanded in the conversion tool, and no special
  // processing is required here.
  auto prim = GetCNodePrimitive(cnode);
  CHECK_NULL_RETURN(prim);
  std::string cnode_func_name = GetCNodeFuncName(cnode);
  if (kCNodeWithDynamicInput.find(cnode_func_name) == kCNodeWithDynamicInput.end()) {
    return lite::RET_OK;
  }
  auto in_node = cnode->inputs()[1];
  if (!utils::isa<CNodePtr>(in_node)) {
    return RET_OK;
  }
  auto in_node_prim = GetCNodePrimitive(in_node);
  CHECK_NULL_RETURN(in_node_prim);
  if (in_node_prim->name() != ops::kNameMakeTuple) {
    MS_LOG(INFO) << "Only the inputs of a multi-input operator whose input is MakeTuple needs to be expanded.";
    return lite::RET_OK;
  }
  auto tuple_node = in_node->cast<CNodePtr>();
  CHECK_NULL_RETURN(tuple_node);
  std::vector<AnfNodePtr> new_inputs = {cnode->inputs()[0]};
  new_inputs.insert(new_inputs.end(), tuple_node->inputs().begin() + 1, tuple_node->inputs().end());
  cnode->set_inputs(new_inputs);
  // add kAttrDynInputSizes for multi-input operator.
  int64_t input_num = tuple_node->size() - 1;
  auto dst_prim = prim->Clone();
  dst_prim->AddAttr(kAttrDynInputSizes, MakeValue(std::vector<int64_t>{input_num, -1}));
  ValueNodePtr value_node = cnode->input(0)->cast<ValueNodePtr>();
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}

STATUS AdapteMuitiInputNode(const FuncGraphPtr &func_graph) {
  CHECK_NULL_RETURN(func_graph);
  auto cnodes = func_graph->GetOrderedCnodes();
  for (const auto &cnode : cnodes) {
    MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_ERROR, "Cnode is nullptr.");
    if (AdapteNodeWithDynamicInput(cnode) != lite::RET_OK) {
      MS_LOG(ERROR) << "Adapter node with multioutput failed.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

CNodePtr CreateTupleGetItemNode(const FuncGraphPtr &func_graph, const CNodePtr &input_cnode) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, nullptr, "func graph failed");
  MS_CHECK_TRUE_MSG(input_cnode != nullptr, nullptr, "input cnode failed");
  CNodePtr get_item_cnode = nullptr;
  auto tuple_get_item_prim_ptr = std::make_shared<ops::TupleGetItem>();
  MS_CHECK_TRUE_MSG(tuple_get_item_prim_ptr != nullptr, nullptr, "New TupleGetItem failed");
  auto tuple_get_item_prim_c = tuple_get_item_prim_ptr->GetPrim();
  auto tuple_get_item_prim = NewValueNode(tuple_get_item_prim_c);
  MS_CHECK_TRUE_MSG(tuple_get_item_prim != nullptr, nullptr, "tuple_prim is nullptr.");
  auto get_item_value = NewValueNode(MakeValue<int64_t>(0));
  MS_CHECK_TRUE_MSG(get_item_value != nullptr, nullptr, "item_value is nullptr.");
  AnfNodePtrList inputs{tuple_get_item_prim, input_cnode, get_item_value};
  get_item_cnode = func_graph->NewCNode(inputs);
  MS_CHECK_TRUE_MSG(get_item_cnode != nullptr, nullptr, "New get item cnode failed.");

  std::vector<int64_t> shape;
  if (acl::GetShapeVectorFromCNode(input_cnode, &shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get shape failed.";
    return nullptr;
  }
  TypeId type = acl::GetTypeFromNode(input_cnode);
  auto tensor_abstract = CreateTensorAbstract(shape, type);
  MS_CHECK_TRUE_MSG(tensor_abstract != nullptr, nullptr, "tensor_abstract is nullptr!");
  MS_CHECK_TRUE_MSG(input_cnode->abstract() != nullptr, nullptr, "input_cnode->abstract() is nullptr.");
  std::string abstract_name = input_cnode->abstract()->name();
  tensor_abstract->set_name(abstract_name);
  MS_CHECK_TRUE_MSG(tensor_abstract != nullptr, nullptr, "Create tensor abstract failed.");
  get_item_cnode->set_abstract(tensor_abstract);
  get_item_cnode->set_fullname_with_scope(input_cnode->fullname_with_scope() + "_getitem");
  AbstractBasePtrList abstract_list = {tensor_abstract};
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  MS_CHECK_TRUE_MSG(abstract_tuple != nullptr, nullptr, "Create abstract Tuple failed.");
  input_cnode->set_abstract(abstract_tuple);
  return get_item_cnode;
}

static STATUS AdapteNodeWithMultiOutputs(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                         const FuncGraphManagerPtr &manager) {
  std::string cnode_func_name = GetCNodeFuncName(cnode);
  if (cnode_func_name == prim::kPrimTupleGetItem->name()) {
    return lite::RET_OK;
  }

  for (size_t i = 1; i < cnode->size(); ++i) {
    auto input = cnode->input(i);
    MS_CHECK_TRUE_MSG(input != nullptr, lite::RET_ERROR, "input is nullptr.");
    if (!utils::isa<CNode>(input)) {
      continue;
    }
    auto input_cnode = input->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(input_cnode != nullptr, lite::RET_ERROR, "input_cnode is nullptr.");
    std::string input_func_name = GetCNodeFuncName(input_cnode);
    bool custom_node_has_muilt_output = false;
    // When the cnode name is custom, and output_num attr is greater than 1, this var will be set to true
    if (input_func_name == kCustomTypeCustom) {
      auto prim = GetCNodePrimitive(input_cnode);
      if (prim->HasAttr(kAttrOutputNum) && GetValue<int64_t>(prim->GetAttr(kAttrOutputNum)) > 1) {
        custom_node_has_muilt_output = true;
      }
    }
    if (kCNodeWithMultiOutputs.find(input_func_name) != kCNodeWithMultiOutputs.end() || custom_node_has_muilt_output) {
      MS_LOG(DEBUG) << "Input " << input_func_name << " of cnode " << cnode_func_name << " has multioutputs";
      CNodePtr get_item_cnode = CreateTupleGetItemNode(func_graph, input_cnode);
      if (get_item_cnode == nullptr) {
        MS_LOG(ERROR) << "Create tuple item for " << input_func_name << " of " << cnode_func_name << " failed.";
        return lite::RET_ERROR;
      }
      if (!manager->Replace(input_cnode, get_item_cnode)) {
        MS_LOG(ERROR) << "Replace " << input_func_name << " of " << cnode_func_name << " failed.";
        return lite::RET_ERROR;
      }
    }
  }
  return lite::RET_OK;
}

STATUS AdapteMuitiOutputNode(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  CHECK_NULL_RETURN(func_graph);
  auto cnodes = func_graph->GetOrderedCnodes();
  for (const auto &cnode : cnodes) {
    MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_ERROR, "Cnode is nullptr.");
    if (AdapteNodeWithMultiOutputs(func_graph, cnode, manager) != lite::RET_OK) {
      MS_LOG(ERROR) << "Adapter node with multioutput failed.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}
}  // namespace lite
}  // namespace mindspore
