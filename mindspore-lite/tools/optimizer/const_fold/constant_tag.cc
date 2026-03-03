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
#include "tools/optimizer/const_fold/constant_tag.h"
#include "mindspore/core/include/utils/anf_utils.h"
#include "common/common.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace opt {
STATUS ConstantTag::PostProcess(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  if (!CheckCanFold(func_graph, cnode)) {
    return lite::RET_OK;
  }
  if (const_fold_processor_ == nullptr) {
    const_fold_processor_ = std::make_shared<ConstantTagProcessor>(fmk_type_, train_flag_);
  }
  MS_CHECK_TRUE_MSG(const_fold_processor_ != nullptr, lite::RET_NULL_PTR, "const fold processor is nullptr");
  auto status = const_fold_processor_->DoConstantFold(func_graph, cnode);
  if (status != lite::RET_OK) {
    MS_LOG(WARNING) << "do constant fold failed, the node is " << cnode->fullname_with_scope();
  }
  return status;
}

bool ConstantTag::CheckAllConstInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, false, "func_graph is nullptr.");
  MS_CHECK_TRUE_MSG(cnode != nullptr, false, "cnode is nullptr.");
  auto inputs = cnode->inputs();
  auto graph_inputs =
    sub_inputs_map_.find(func_graph) != sub_inputs_map_.end() ? sub_inputs_map_[func_graph] : func_graph->get_inputs();
  return std::all_of(inputs.begin(), inputs.end(), [&graph_inputs](const AnfNodePtr &node) {
    auto cnode = node->cast<CNodePtr>();
    bool cnode_has_value = cnode && cnode->HasAttr(lite::kNameCNodeValueAttr);

    return (node->isa<ValueNode>() && !IsValueNode<FuncGraph>(node)) ||
           (node->isa<Parameter>() && node->cast<ParameterPtr>()->has_default() &&
            std::find(graph_inputs.begin(), graph_inputs.end(), node) == graph_inputs.end()) ||
           cnode_has_value;
  });
}
bool ConstantTag::Run(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, false, "func_graph is nullptr.");
  if (!InferShapePass::Run(func_graph)) {
    MS_LOG(ERROR) << "ConstantTag Run failed, reason: infer shape failed.";
    return false;
  }
  auto nodes = FuncGraph::TopoSort(func_graph->get_return());
  for (auto &n : nodes) {
    if (!n->isa<CNode>()) {
      continue;
    }
    auto cnode = n->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(cnode != nullptr, false, "cnode is nullptr.");
    auto ret = UpdateShapeValueAttr(cnode);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "UpdateShapeValueAttr for " << cnode->fullname_with_scope() << " failed.";
      return false;
    }
    ret = UpdateDynamicShapeAttr(cnode);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "UpdateDynamicShapeAttr for " << cnode->fullname_with_scope() << " failed.";
      return false;
    }
  }
  return true;
}

STATUS ConstantTag::UpdateDynamicShapeAttr(const CNodePtr &cnode) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_NULL_PTR, "cnode is nullptr");
  auto abstract = cnode->abstract();
  MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_NULL_PTR, "abstract is nullptr");
  auto shapes_attr = cnode->GetAttr(lite::kNameShapeAttr);
  std::vector<ValuePtr> shapes;
  if (shapes_attr != nullptr) {
    if (!shapes_attr->isa<ValueSequence>()) {
      MS_LOG(ERROR) << "attr: " << lite::kNameShapeAttr << " should be a ValueSequence";
      return lite::RET_ERROR;
    }

    shapes = GetValue<std::vector<ValuePtr>>(shapes_attr);
  }
  ValuePtr ptr;
  auto ret = ConvertAbstractToValue(abstract, &ptr);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, ret, "convert cnode abstract to value failed.");
  shapes.push_back(ptr);
  cnode->AddAttr(lite::kNameShapeAttr, MakeValue(shapes));
  return lite::RET_OK;
}

STATUS ConstantTag::ConvertAbstractToValue(const std::shared_ptr<abstract::AbstractBase> &abstract, ValuePtr *value) {
  MS_CHECK_TRUE_MSG(value != nullptr, lite::RET_NULL_PTR, "value is nullptr.");
  if (abstract->isa<abstract::AbstractTensor>()) {
    auto abstract_tensor = abstract->cast<abstract::AbstractTensorPtr>();
    MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, lite::RET_NULL_PTR, "abstract_tensor is nullptr.");
    auto element = abstract_tensor->element();
    MS_CHECK_TRUE_MSG(element != nullptr, lite::RET_NULL_PTR, "element is nullptr.");
    auto type = element->GetType();
    MS_CHECK_TRUE_MSG(type != nullptr, lite::RET_NULL_PTR, "type is nullptr.");
    auto shape = abstract_tensor->GetShape();
    MS_CHECK_TRUE_MSG(shape != nullptr, lite::RET_NULL_PTR, "shape is nullptr.");

    *value = std::make_shared<tensor::MetaTensor>(type->type_id(), shape->GetShapeVector());
  } else if (abstract->isa<abstract::AbstractScalar>()) {
    auto abstract_scalar = abstract->cast<abstract::AbstractScalarPtr>();
    MS_CHECK_TRUE_MSG(abstract_scalar != nullptr, lite::RET_NULL_PTR, "abstract_scalar is nullptr.");
    auto type = abstract_scalar->GetType();
    MS_CHECK_TRUE_MSG(type != nullptr, lite::RET_NULL_PTR, "type is nullptr.");

    *value = std::make_shared<tensor::MetaTensor>(type->type_id(), ShapeVector());
  } else if (abstract->isa<abstract::AbstractTuple>()) {
    auto abstract_tuple = abstract->cast<abstract::AbstractTuplePtr>();
    MS_CHECK_TRUE_MSG(abstract_tuple != nullptr, lite::RET_NULL_PTR, "abstract_tuple is nullptr.");
    std::vector<ValuePtr> shapes_tuple;
    for (auto &e : abstract_tuple->elements()) {
      ValuePtr ptr;
      auto ret = ConvertAbstractToValue(e, &ptr);
      MS_CHECK_TRUE_MSG(ret == lite::RET_OK, ret, "convert elements failed.");
      MS_CHECK_TRUE_MSG(ptr != nullptr, lite::RET_NULL_PTR, "convert elements failed. value ptr is nullptr.");
      MS_CHECK_TRUE_MSG(ptr->isa<tensor::MetaTensor>(), lite::RET_ERROR, "abstract tuple should contains a MetaTensor");
      shapes_tuple.push_back(ptr);
    }
    *value = MakeValue(shapes_tuple);
  }
  return lite::RET_OK;
}

STATUS ConstantTag::UpdateShapeValueAttr(const CNodePtr &cnode) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_NULL_PTR, "cnode is nullptr");
  auto kernel_name = AnfUtils::GetCNodeName(cnode);
  if (kernel_name == ops::kNameShape) {
    cnode->AddAttr(lite::kNameShapeValueAttr, MakeValue(true));
    return lite::RET_OK;
  }

  auto inputs = cnode->inputs();
  for (size_t i = 1; i < inputs.size(); i++) {
    auto input = inputs[i];
    MS_CHECK_TRUE_MSG(input != nullptr, lite::RET_NULL_PTR, "input is nullptr.");
    if (input->isa<Parameter>()) {
      continue;
    }
    if (input->isa<CNode>()) {
      auto input_cnode = input->cast<CNodePtr>();
      MS_CHECK_TRUE_MSG(input_cnode != nullptr, lite::RET_NULL_PTR, "input_cnode is nullptr.");
      auto depend_value = input_cnode->GetAttr(lite::kNameShapeValueAttr);
      if (!depend_value) {
        continue;
      }
      if (input_cnode->HasAttr(lite::kNameCNodeValueAttr)) {
        cnode->AddAttr(lite::kNameShapeValueAttr, MakeValue(true));
        return lite::RET_OK;
      }
    }
  }
  return lite::RET_OK;
}

bool ConstantTag::CheckCanFold(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  if (IsSpecialType(cnode) || CheckPrimitiveType(cnode, prim::kPrimCustom) || IsMarkedTrainOp(cnode)) {
    return false;
  }
  if (CheckAllConstInput(func_graph, cnode)) {
    return true;
  }
  auto prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    MS_LOG(DEBUG) << "prim is nullptr.";
    return false;
  }
  auto is_inferred = prim->GetAttr(kInferDone) != nullptr && GetValue<bool>(prim->GetAttr(kInferDone));
  if (!is_inferred) {
    MS_LOG(DEBUG) << "is_inferred is false.";
    return false;
  }
  if (CheckPrimitiveType(cnode, prim::kPrimShape)) {
    return true;
  }
  return false;
}

std::shared_ptr<NodeInferShape> ConstantTag::CreateNodeInferShape() {
  return std::make_shared<ConstantTagNodeInfer>(fmk_type_, train_flag_);
}

}  // namespace opt
}  // namespace mindspore
