/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "tools/common/graph_util.h"
#include <algorithm>
#include <functional>
#include <ctime>
#include <utility>
#include <set>
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "tools/common/meta_graph_utils.h"
#include "schema/inner/model_generated.h"
#include "tools/common/tensor_util.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#include "nnacl/op_base.h"
#include "infer/make_tuple.h"
#include "tools/converter/converter_context.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/common/string_util.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_u.h"

namespace mindspore {
namespace lite {
namespace {
const int kZeroPointGap = 128;
constexpr size_t kTupleGetItemFirstInputIdx = 1;
constexpr size_t kDependInputNum = 3;
constexpr size_t kDependFirstInputIdx = 1;
constexpr size_t kSequenceCodeGetItemInputSize = 3;
constexpr size_t kSecondIndex = 1;
constexpr size_t kInvalidSize = SIZE_MAX;
constexpr auto kMakeTuple = "MakeTuple";
constexpr auto kMakeList = "make_list";
constexpr size_t kEncMaxLen = 16;
}  // namespace

static STATUS GetAbstractfromSequenceCodeGetItem(const CNodePtr &cnode, AbstractBasePtr *abstract, size_t *idx) {
  MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_ERROR, "Abstract is nullptr.");
  MS_CHECK_TRUE_MSG(idx != nullptr, lite::RET_ERROR, "idx is nullptr.");
  auto SequenceCode_inputs = cnode->inputs();
  MS_CHECK_TRUE_MSG(SequenceCode_inputs.size() == kSequenceCodeGetItemInputSize, lite::RET_ERROR,
                    "The node must have 3 inputs!");
  auto get_item_input_cnode = SequenceCode_inputs.at(kSecondIndex);
  MS_CHECK_TRUE_MSG(get_item_input_cnode != nullptr, lite::RET_ERROR, "input node is nullptr.");

  AbstractBasePtrList abstract_list;
  if (opt::CheckPrimitiveType(cnode, prim::kPrimTupleGetItem)) {
    *idx = opt::GetTupleGetItemOutIndex(cnode);
    if (!mindspore::utils::isa<mindspore::abstract::AbstractTuplePtr>(get_item_input_cnode->abstract())) {
      MS_LOG(ERROR) << "TupleGetItem's abstract is not AbstractTuple, cnode name: "
                    << get_item_input_cnode->fullname_with_scope();
      return lite::RET_ERROR;
    }
    auto input_node_abstract = utils::cast<abstract::AbstractTuplePtr>(get_item_input_cnode->abstract());
    abstract_list = input_node_abstract->elements();
  } else {
    *idx = opt::GetListGetItemOutIndex(cnode);
    if (!mindspore::utils::isa<mindspore::abstract::AbstractListPtr>(get_item_input_cnode->abstract())) {
      MS_LOG(ERROR) << "ListGetItem's abstract is not AbstractTuple, cnode name: "
                    << get_item_input_cnode->fullname_with_scope();
      return lite::RET_ERROR;
    }
    auto input_node_abstract = utils::cast<abstract::AbstractListPtr>(get_item_input_cnode->abstract());
    abstract_list = input_node_abstract->elements();
  }

  if (abstract_list.size() <= *idx) {
    MS_LOG(ERROR) << "Abstract's size is smaller than expect";
    return lite::RET_ERROR;
  }
  *abstract = abstract_list[*idx];
  return lite::RET_OK;
}

STATUS GetShapeVectorFromParameter(const mindspore::ParameterPtr &param_node, std::vector<int64_t> *shape_vector) {
  MS_CHECK_TRUE_MSG(shape_vector != nullptr, RET_ERROR, "shape vector is nullptr.");
  auto abstract_base = param_node->abstract();
  if (abstract_base == nullptr) {
    MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << param_node->name();
    return RET_ERROR;
  }

  if (!abstract_base->isa<abstract::AbstractTensor>()) {
    MS_LOG(ERROR) << "Abstract of parameter should be abstract tensor, " << param_node->name();
    return lite::RET_ERROR;
  }
  auto abstract_tensor = abstract_base->cast<abstract::AbstractTensorPtr>();
  MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, RET_ERROR, "Cast to abstract tensor failed!");
  *shape_vector = abstract_tensor->shape()->shape();
  return lite::RET_OK;
}

STATUS GetShapeVectorAndIdxFromCNode(const CNodePtr &cnode, std::vector<int64_t> *shape_vector, size_t *idx) {
  MS_CHECK_TRUE_MSG(shape_vector != nullptr, lite::RET_ERROR, "shape is nullptr");
  MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_ERROR, "cnode is nullptr");
  AbstractBasePtr cnode_abstract = nullptr;
  if ((opt::CheckPrimitiveType(cnode, prim::kPrimTupleGetItem)) ||
      (opt::CheckPrimitiveType(cnode, prim::kPrimListGetItem))) {
    // idx is only used when cnode is type of kPrimTupleGetItem or kPrimListGetItem.
    MS_CHECK_TRUE_MSG(idx != nullptr, lite::RET_ERROR, "idx is nullptr");
    if (GetAbstractfromSequenceCodeGetItem(cnode, &cnode_abstract, idx) != lite::RET_OK) {
      MS_LOG(ERROR) << "Get abstract from tuple get item failed.";
      return lite::RET_ERROR;
    }
  } else {
    cnode_abstract = cnode->abstract();
  }
  // the control flow model may be nullptr
  if (cnode_abstract == nullptr) {
    *shape_vector = std::vector<int64_t>();
    return lite::RET_OK;
  }
  if (cnode_abstract->BuildShape() == mindspore::abstract::kNoShape) {
    *shape_vector = std::vector<int64_t>();
    return lite::RET_OK;
  }
  if (!utils::isa<mindspore::abstract::AbstractTensorPtr>(cnode_abstract)) {
    MS_LOG(ERROR) << "Abstract is not abstract tensor. " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  auto cnode_abstract_tensor = cnode_abstract->cast<mindspore::abstract::AbstractTensorPtr>();
  CHECK_NULL_RETURN(cnode_abstract_tensor);
  if (!utils::isa<mindspore::abstract::ShapePtr>(cnode_abstract_tensor->BuildShape())) {
    MS_LOG(ERROR) << "Shape of abstract tensor should be ShapePtr. " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  auto shape_ptr = utils::cast<mindspore::abstract::ShapePtr>(cnode_abstract_tensor->BuildShape());
  CHECK_NULL_RETURN(shape_ptr);
  if (shape_ptr->shape().empty()) {
    MS_LOG(WARNING) << "Shape is empty " << cnode->fullname_with_scope();
  }
  *shape_vector = shape_ptr->shape();
  return lite::RET_OK;
}

STATUS GetCNodeOrParameterShapeVec(const AnfNodePtr &anf_node, std::vector<int> *shape) {
  auto int64_t_to_int_func = [](int64_t x) -> int { return static_cast<int>(x); };
  std::vector<int64_t> in_shape;
  if (anf_node->isa<CNode>()) {
    auto status = GetShapeVectorAndIdxFromCNode(anf_node->cast<CNodePtr>(), &in_shape);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Get shape from CNode failed.";
      return status;
    }
  } else if (anf_node->isa<Parameter>()) {
    auto param_node = anf_node->cast<ParameterPtr>();
    auto status = GetShapeVectorFromParameter(param_node, &in_shape);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Get shape from Parameter failed.";
      return status;
    }
  } else {
    MS_LOG(ERROR) << "Node type is not recognized.";
    return RET_ERROR;
  }
  shape->resize(in_shape.size());
  (void)std::transform(in_shape.begin(), in_shape.end(), shape->begin(), int64_t_to_int_func);
  return RET_OK;
}

static STATUS TraceOutput(const AnfNodePtr &node, std::vector<std::pair<AnfNodePtr, int64_t>> *outputs,
                          std::vector<std::string> *output_names, std::vector<std::vector<int64_t>> *output_dims) {
  static size_t iter = 0;
  CHECK_NULL_RETURN(node);
  if (utils::isa<ParameterPtr>(node) || utils::isa<ValueNode>(node)) {
    MS_LOG(INFO) << "Name of graph output value node is : " << node->fullname_with_scope();
    outputs->emplace_back(std::pair<AnfNodePtr, int64_t>(node, 0));
    output_names->push_back(node->fullname_with_scope());
    output_dims->emplace_back(std::vector<int64_t>());
    return lite::RET_OK;
  }
  AnfNodePtr cur_node = node;
  CNodePtr pre_node = nullptr;
  while (cur_node->isa<CNode>() && IsPrimitiveCNode(cur_node, prim::kPrimTupleGetItem)) {
    auto tmp = cur_node->cast<CNodePtr>();
    CHECK_NULL_RETURN(tmp);
    pre_node = tmp;
    cur_node = tmp->input(kTupleGetItemFirstInputIdx);
    CHECK_NULL_RETURN(cur_node);
  }
  auto cnode = cur_node->cast<CNodePtr>();
  CHECK_NULL_RETURN(cnode);
  std::string name = GetCNodeFuncName(cnode);
  iter++;
  MS_LOG(INFO) << "Func name of cnode " << name << " ,trace iter: " << iter;
  if ((name == kMakeTuple) || (name == kMakeList)) {
    for (size_t i = 1; i < cnode->size(); ++i) {
      auto make_tuple_input = cnode->input(i);
      if (opt::CheckPrimitiveType(make_tuple_input, prim::kPrimUpdateState) ||
          opt::CheckPrimitiveType(make_tuple_input, prim::kPrimLoad)) {
        continue;
      }
      if (TraceOutput(make_tuple_input, outputs, output_names, output_dims) != lite::RET_OK) {
        MS_LOG(ERROR) << "The input[ " << i << "]"
                      << " trace output failed, name: " << name;
        return lite::RET_ERROR;
      }
    }
  } else if (name == prim::kPrimDepend->name()) {
    if (cnode->size() < kDependInputNum) {
      MS_LOG(ERROR) << "Length of inputs is " << cnode->size() << ", which is less than three.";
      return lite::RET_ERROR;
    }
    if (TraceOutput(cnode->input(kDependFirstInputIdx), outputs, output_names, output_dims) != lite::RET_OK) {
      MS_LOG(ERROR) << "Depend node trace output failed.";
      return lite::RET_ERROR;
    }
  } else {
    MS_LOG(INFO) << "Name of graph output node is " << cnode->fullname_with_scope();
    std::string node_name = cnode->fullname_with_scope();
    std::vector<int64_t> dims;
    size_t idx = -1;
    STATUS ret;
    if (pre_node != nullptr && IsPrimitiveCNode(pre_node, prim::kPrimTupleGetItem)) {
      ret = GetShapeVectorAndIdxFromCNode(pre_node, &dims, &idx);
      node_name = node_name + "_" + std::to_string(idx);
    } else {
      ret = GetShapeVectorAndIdxFromCNode(cnode, &dims, &idx);
    }
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "Get node shape failed.";
      return lite::RET_ERROR;
    }
    outputs->emplace_back(std::pair<AnfNodePtr, int64_t>(cnode, idx));
    output_names->emplace_back(node_name);
    output_dims->emplace_back(dims);
  }
  return lite::RET_OK;
}

void AdjustDuplicateNodeName(const FuncGraphPtr &func_graph) {
  std::set<std::string> nodes_name;
  auto nodes_list = TopoSort(func_graph->get_return());
  int index = 0;
  for (auto &node : nodes_list) {
    if (node == nullptr) {
      continue;
    }
    auto name = node->fullname_with_scope();
    if (nodes_name.find(name) == nodes_name.end()) {
      (void)nodes_name.insert(name);
      continue;
    }
    auto sole_name = name + "_" + std::to_string(index);
    while (nodes_name.find(sole_name) != nodes_name.end()) {
      ++index;
      sole_name = name + "_" + std::to_string(index);
    }
    ++index;
    if (utils::isa<CNode>(node)) {
      node->cast<CNodePtr>()->set_fullname_with_scope(sole_name);
      (void)nodes_name.insert(sole_name);
      continue;
    }
    if (utils::isa<Parameter>(node)) {
      node->cast<ParameterPtr>()->set_name(sole_name);
      (void)nodes_name.insert(sole_name);
    }
  }
}

int SetFuncGraphOutput(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &outputs) {
  if (graph == nullptr || outputs.empty()) {
    MS_LOG(DEBUG) << "Input graph is nullptr or outputs is empty";
    return RET_INPUT_PARAM_INVALID;
  }
  if (outputs.size() == 1) {
    graph->set_output(outputs.front(), false);
    return RET_OK;
  }
  auto make_tuple_prim_ptr = std::make_shared<ops::MakeTuple>();
  if (make_tuple_prim_ptr == nullptr) {
    MS_LOG(DEBUG) << "new MakeTuple failed";
    return lite::RET_NULL_PTR;
  }
  auto make_tuple_prim_c = make_tuple_prim_ptr->GetPrim();
  MS_CHECK_TRUE_MSG(make_tuple_prim_c != nullptr, lite::RET_NULL_PTR, "make_tuple_prim_c is nullptr");
  auto make_tuple_cnode = graph->NewCNode(make_tuple_prim_c, outputs);
  if (make_tuple_cnode == nullptr) {
    MS_LOG(DEBUG) << "new cnode failed";
    return lite::RET_NULL_PTR;
  }
  make_tuple_cnode->set_fullname_with_scope("return tuple");
  graph->set_output(make_tuple_cnode, false);
  return RET_OK;
}

OpDefCopyer GetSimpleOpCopyer() {
  return [](const CNodeT &inCNode) -> std::unique_ptr<CNodeT> {
    std::unique_ptr<CNodeT> newCNode = std::make_unique<CNodeT>();
    if (newCNode == nullptr) {
      return nullptr;
    }

    newCNode->name = inCNode.name;
    newCNode->quantType = inCNode.quantType;
    newCNode->primitive = std::make_unique<schema::PrimitiveT>();
    newCNode->primitive->value.type = inCNode.primitive->value.type;
    return newCNode;
  };
}

STATUS AddTensor2Node(schema::MetaGraphT *graphT, uint32_t nodeIdx, std::unique_ptr<TensorT> tensor,
                      InsertPlace place) {
  MS_CHECK_TRUE_MSG(graphT != nullptr, RET_NULL_PTR, "graphT is nullptr");
  if (nodeIdx >= graphT->nodes.size()) {
    MS_LOG(ERROR) << "nodeIdx out of range: " << nodeIdx;
    return RET_PARAM_INVALID;
  }
  graphT->allTensors.emplace_back(std::move(tensor));
  uint32_t newTensorIdx = graphT->allTensors.size() - 1;
  auto node = graphT->nodes.at(nodeIdx).get();
  MS_CHECK_TRUE_MSG(node != nullptr, RET_NULL_PTR, "node is nullptr");
  if (place == kBefore) {
    node->inputIndex.emplace_back(newTensorIdx);
  } else {
    node->outputIndex.emplace_back(newTensorIdx);
  }
  return RET_OK;
}

STATUS ReplaceTensorOfNode(schema::MetaGraphT *graphT, uint32_t nodeIdx, uint32_t inTensorIdx,
                           std::unique_ptr<TensorT> tensor) {
  MS_CHECK_TRUE_MSG(graphT != nullptr, RET_ERROR, "graphT is nullptr!");
  if (nodeIdx >= graphT->nodes.size()) {
    MS_LOG(ERROR) << "nodeIdx out of range: " << nodeIdx;
    return RET_PARAM_INVALID;
  }
  auto node = graphT->nodes.at(nodeIdx).get();
  MS_CHECK_TRUE_MSG(node != nullptr, RET_NULL_PTR, "node is nullptr");
  if (inTensorIdx >= graphT->allTensors.size()) {
    MS_LOG(ERROR) << "inTensorIdx out of range: " << nodeIdx;
    return RET_PARAM_INVALID;
  }
  if (!IsContain(node->inputIndex, inTensorIdx)) {
    MS_LOG(ERROR) << "inTensorIdx(" << inTensorIdx << ") is not a inputIdx of node(" << nodeIdx << ")";
    return RET_PARAM_INVALID;
  }
  graphT->allTensors.at(inTensorIdx).swap(tensor);
  return RET_OK;
}

NodeIter InsertNode(schema::MetaGraphT *graphT, uint32_t existNodeIdx, InsertPlace place, size_t inoutIndex,
                    std::unique_ptr<CNodeT> toAddNode, STATUS *errorCode, int *insert_num,
                    const OpDefCopyer &opDefCopyer) {
  std::vector<std::unique_ptr<schema::CNodeT>>::iterator it;
  MS_CHECK_TRUE_RET(graphT != nullptr, it);
  MS_CHECK_TRUE_RET(errorCode != nullptr, it);
  if (existNodeIdx >= graphT->nodes.size()) {
    MS_LOG(ERROR) << "nodeIdx out of range: " << existNodeIdx;
    return graphT->nodes.end();
  }
  auto node_iter = graphT->nodes.begin() + existNodeIdx;
  MS_CHECK_TRUE_RET(node_iter != graphT->nodes.begin(), it);
  MS_CHECK_TRUE_RET((*node_iter) != nullptr, it);
  return InsertNode(graphT, node_iter, place, inoutIndex, std::move(toAddNode), errorCode, insert_num);
}

NodeIter InsertNode(schema::MetaGraphT *graphT, NodeIter existNodeIter, InsertPlace place, size_t inoutIndexIdx,
                    std::unique_ptr<CNodeT> toAddNode, STATUS *errorCode, int *insert_num,
                    const OpDefCopyer &opDefCopyer) {
  std::vector<std::unique_ptr<schema::CNodeT>>::iterator it;
  MS_CHECK_TRUE_RET(graphT != nullptr, it);
  MS_CHECK_TRUE_RET(errorCode != nullptr, it);
  if (place == kBefore) {
    return InsertNodeBefore(graphT, existNodeIter, inoutIndexIdx, std::move(toAddNode), errorCode, insert_num,
                            opDefCopyer);
  } else if (place == kAfter) {
    return InsertNodeAfter(graphT, existNodeIter, inoutIndexIdx, std::move(toAddNode), errorCode, insert_num,
                           opDefCopyer);
  } else {
    MS_LOG(ERROR) << "Invalid InsertPlace : " << place;
    return graphT->nodes.end();
  }
}

NodeIter InsertNodeBefore(schema::MetaGraphT *graphT, NodeIter existNodeIter, size_t inputIndexIdx,
                          std::unique_ptr<CNodeT> toAddNodeIn, STATUS *errorCode, int *insert_num,
                          const OpDefCopyer &opDefCopyer) {
  std::vector<std::unique_ptr<schema::CNodeT>>::iterator it;
  MS_CHECK_TRUE_RET(graphT != nullptr, it);
  MS_CHECK_TRUE_RET(errorCode != nullptr, it);
  auto &existNode = *existNodeIter;
  MS_CHECK_TRUE_RET(existNode != nullptr, it);
  MS_CHECK_TRUE_RET(existNode->inputIndex.size() > inputIndexIdx, it);
  MS_CHECK_TRUE_RET(toAddNodeIn != nullptr, it);
  auto preTensorIdx = existNode->inputIndex.at(inputIndexIdx);
  MS_CHECK_TRUE_RET(graphT->allTensors.size() > preTensorIdx, it);

  auto preNodeIdxes = GetInputNodeIdx(*graphT, *(existNode), inputIndexIdx);
  size_t insert_node_num = preNodeIdxes.empty() ? 1 : preNodeIdxes.size();
  std::vector<std::unique_ptr<CNodeT>> toAddNodes;
  for (size_t i = 0; i < insert_node_num; ++i) {
    auto &preTensor = graphT->allTensors.at(preTensorIdx);
    MS_CHECK_TRUE_RET(preTensor != nullptr, it);
    auto toAddTensor = CopyTensorDefT(preTensor);
    if (toAddTensor == nullptr) {
      *errorCode = RET_NULL_PTR;
      MS_LOG(ERROR) << "Copy Tensor failed";
      return graphT->nodes.end();
    }
    toAddTensor->nodeType = NodeType_CNode;
    toAddTensor->refCount = 0;
    toAddTensor->data.clear();
    MS_CHECK_TRUE_RET(toAddNodeIn->primitive != nullptr, it);
    if (toAddNodeIn->primitive->value.type == schema::PrimitiveType_QuantDTypeCast) {
      auto prim = toAddNodeIn->primitive->value.AsQuantDTypeCast();
      MS_CHECK_TRUE_RET(prim != nullptr, it);
      if (prim->src_t == TypeId::kNumberTypeUInt8) {
        if (preTensor->dataType == TypeId::kNumberTypeUInt8) {
          toAddTensor->quantParams.front()->zeroPoint -= kZeroPointGap;
        } else {
          preTensor->quantParams.front()->zeroPoint += kZeroPointGap;
        }
      } else if (prim->dst_t == TypeId::kNumberTypeUInt8) {
        if (preTensor->dataType == TypeId::kNumberTypeInt8) {
          toAddTensor->quantParams.front()->zeroPoint += kZeroPointGap;
        } else {
          preTensor->quantParams.front()->zeroPoint -= kZeroPointGap;
        }
      }
      preTensor->dataType = prim->src_t;
      toAddTensor->dataType = prim->dst_t;
    }
    graphT->allTensors.emplace_back(std::move(toAddTensor));
    size_t toAddTensorIdx = graphT->allTensors.size() - 1;
    auto toAddNode = opDefCopyer(*toAddNodeIn);
    if (toAddNode == nullptr) {
      MS_LOG(ERROR) << "copy toAddNodeIn failed";
      *errorCode = RET_NULL_PTR;
      return graphT->nodes.end();
    }
    if (!preNodeIdxes.empty()) {
      toAddNode->name = toAddNodeIn->name + "_" + std::to_string(i);
    }
    toAddNode->inputIndex.clear();
    toAddNode->inputIndex.push_back(preTensorIdx);
    toAddNode->outputIndex.clear();
    toAddNode->outputIndex.push_back(toAddTensorIdx);
    for (auto iter = existNode->inputIndex.begin(); iter != existNode->inputIndex.end(); iter++) {
      if (*iter == preTensorIdx) {
        *iter = toAddTensorIdx;
        break;
      }
    }
    toAddNodes.emplace_back(std::move(toAddNode));
  }
  for (auto &toAddNode : toAddNodes) {
    existNodeIter = graphT->nodes.insert(existNodeIter, std::move(toAddNode));
    existNodeIter++;
    *insert_num += 1;
  }
  *errorCode = RET_OK;
  return existNodeIter;
}

NodeIter InsertNodeAfter(schema::MetaGraphT *graphT, NodeIter existNodeIter, size_t outputIndexIdx,
                         std::unique_ptr<schema::CNodeT> toAddNodeIn, STATUS *errorCode, int *insert_num,
                         const OpDefCopyer &opDefCopyer) {
  std::vector<std::unique_ptr<schema::CNodeT>>::iterator it;
  MS_CHECK_TRUE_RET(graphT != nullptr, it);
  MS_CHECK_TRUE_RET(errorCode != nullptr, it);
  auto &existNode = *existNodeIter;
  MS_CHECK_TRUE_RET(existNode != nullptr, it);
  MS_CHECK_TRUE_RET(existNode->outputIndex.size() > outputIndexIdx, it);
  MS_CHECK_TRUE_RET(toAddNodeIn != nullptr, it);
  auto postTensorIdx = existNode->outputIndex.at(outputIndexIdx);
  MS_CHECK_TRUE_RET(graphT->allTensors.size() > postTensorIdx, it);
  auto postNodeIdxes = GetOutputNodeIdx(*graphT, *(existNode), outputIndexIdx);
  bool is_output_index = IsContain(graphT->outputIndex, postTensorIdx);
  size_t insert_node_num = (postNodeIdxes.empty() || is_output_index) ? postNodeIdxes.size() + 1 : postNodeIdxes.size();
  bool has_insert_for_graph_out = postNodeIdxes.empty() || is_output_index;
  std::vector<std::unique_ptr<schema::CNodeT>> toAddNodes;
  for (size_t i = 0; i < insert_node_num; ++i) {
    auto &postTensor = graphT->allTensors.at(postTensorIdx);
    MS_CHECK_TRUE_RET(postTensor != nullptr, it);
    auto toAddTensor = CopyTensorDefT(postTensor);
    if (toAddTensor == nullptr) {
      MS_LOG(ERROR) << "Copy TensorT failed";
      *errorCode = RET_NULL_PTR;
      return graphT->nodes.end();
    }
    toAddTensor->nodeType = NodeType_CNode;
    MS_CHECK_TRUE_RET(toAddNodeIn->primitive != nullptr, it);
    if (toAddNodeIn->primitive->value.type == schema::PrimitiveType_QuantDTypeCast) {
      auto prim = toAddNodeIn->primitive->value.AsQuantDTypeCast();
      MS_CHECK_TRUE_RET(prim != nullptr, it);
      if (prim->dst_t == TypeId::kNumberTypeUInt8) {
        if (postTensor->dataType == TypeId::kNumberTypeUInt8) {
          postTensor->quantParams.front()->zeroPoint -= kZeroPointGap;
        } else {
          toAddTensor->quantParams.front()->zeroPoint += kZeroPointGap;
        }
      } else if (prim->src_t == TypeId::kNumberTypeUInt8) {
        if (postTensor->dataType == TypeId::kNumberTypeUInt8) {
          toAddTensor->quantParams.front()->zeroPoint -= kZeroPointGap;
        } else {
          postTensor->quantParams.front()->zeroPoint += kZeroPointGap;
        }
      }
      postTensor->dataType = prim->src_t;
      toAddTensor->dataType = prim->dst_t;
    }
    graphT->allTensors.emplace_back(std::move(toAddTensor));
    size_t toAddTensorIdx = graphT->allTensors.size() - 1;
    auto toAddNode = opDefCopyer(*toAddNodeIn);
    if (toAddNode == nullptr) {
      MS_LOG(ERROR) << "copy toAddNodeIn failed";
      *errorCode = RET_NULL_PTR;
      return graphT->nodes.end();
    }
    toAddNode->inputIndex.clear();
    toAddNode->inputIndex.push_back(postTensorIdx);
    toAddNode->outputIndex.clear();
    toAddNode->outputIndex.push_back(toAddTensorIdx);
    if (!postNodeIdxes.empty()) {
      toAddNode->name = toAddNodeIn->name + "_" + std::to_string(i);
    }
    if (has_insert_for_graph_out) {
      ReplaceOutput(postTensorIdx, toAddTensorIdx, graphT);
      has_insert_for_graph_out = false;
    } else {
      auto &postNode = graphT->nodes.at(postNodeIdxes[is_output_index ? i - 1 : i]);
      for (auto iter = postNode->inputIndex.begin(); iter != postNode->inputIndex.end(); iter++) {
        if (*iter == postTensorIdx) {
          *iter = toAddTensorIdx;
        }
      }
    }
    toAddNodes.emplace_back(std::move(toAddNode));
  }
  for (auto &toAddNode : toAddNodes) {
    existNodeIter = graphT->nodes.insert(existNodeIter, std::move(toAddNode));
    existNodeIter++;
    *insert_num += 1;
  }
  *errorCode = RET_OK;
  return existNodeIter;
}

STATUS ValidateFileStr(const std::string &modelFile, const std::string &fileType) {
  if (modelFile.size() > fileType.size() && modelFile.substr(modelFile.size() - fileType.size()) == fileType) {
    return RET_OK;
  } else {
    return RET_ERROR;
  }
}

void SetSubgraphTensorIndices(schema::MetaGraphT *meta_graphT) {
  if (meta_graphT == nullptr) {
    MS_LOG(ERROR) << "meta_graphT is nullptr.";
    return;
  }
  for (auto &subgraph : meta_graphT->subGraph) {
    std::vector<uint32_t> subgraph_indices{};
    subgraph_indices.assign(subgraph->inputIndices.begin(), subgraph->inputIndices.end());
    subgraph_indices.assign(subgraph->outputIndices.begin(), subgraph->outputIndices.end());
    for (auto &node_idx : subgraph->nodeIndices) {
      auto &node = meta_graphT->nodes.at(node_idx);
      for (auto &input_idx : node->inputIndex) {
        if (IsContain(subgraph_indices, input_idx)) {
          continue;
        } else {
          subgraph_indices.push_back(input_idx);
        }
      }
      for (auto &output_idx : node->outputIndex) {
        if (IsContain(subgraph_indices, output_idx)) {
          continue;
        } else {
          subgraph_indices.push_back(output_idx);
        }
      }
    }
    subgraph->tensorIndices.assign(subgraph_indices.begin(), subgraph_indices.end());
  }
}

std::string GetModelName(const std::string &modelFile) {
  std::string modelName = modelFile;
  modelName = modelName.substr(modelName.find_last_of('/') + 1);
  modelName = modelName.substr(0, modelName.find_last_of('.'));
  return modelName;
}

std::vector<int> GetTransposePerm(MetaGraphT *graph, const std::unique_ptr<CNodeT> &cnode) {
  MS_CHECK_TRUE_MSG(graph != nullptr, {}, "graph is nullptr!");
  MS_CHECK_TRUE_MSG(cnode != nullptr, {}, "cnode is nullptr!");
  std::vector<int> perm;
  if (cnode->primitive->value.type != schema::PrimitiveType_Transpose) {
    return perm;
  }
  if (cnode->inputIndex.size() < 2) {
    MS_LOG(ERROR) << "transpose node input size is less than 2.";
    return perm;
  }
  MS_CHECK_TRUE_RET(cnode->outputIndex.at(1) < graph->allTensors.size(), {});
  auto &perm_tensor = graph->allTensors.at(cnode->inputIndex.at(1));
  if (perm_tensor->data.empty()) {
    return perm;
  }
  MS_CHECK_TRUE_RET(perm_tensor->dims.size() != 0, {});
  perm.resize(perm_tensor->dims[0]);
  if (memcpy_s(perm.data(), perm_tensor->dims[0] * sizeof(int), perm_tensor->data.data(),
               perm_tensor->dims[0] * sizeof(int)) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
    return {};
  }
  return perm;
}

TypeId GetAbstractTensorDtype(const abstract::AbstractTensorPtr &tensor) {
  if (tensor == nullptr || tensor->element() == nullptr) {
    MS_LOG(ERROR) << "abstract_tensor or abstract_tensor->element() is nullptr";
    return kTypeUnknown;
  }
  auto type_ptr = tensor->element()->GetTypeTrack();
  MS_CHECK_TRUE_MSG(type_ptr != nullptr, kTypeUnknown, "type_ptr is nullptr");
  return type_ptr->type_id();
}

TypeId GetParameterDtype(const ParameterPtr &param_node) {
  MS_CHECK_TRUE_MSG(param_node != nullptr, kTypeUnknown, "param_node is nullptr");
  auto abstract_base = param_node->abstract();
  MS_CHECK_TRUE_MSG(abstract_base != nullptr, kTypeUnknown, "abstract_base is nullptr");
  auto abstract_tensor = abstract_base->cast<abstract::AbstractTensorPtr>();
  MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, kTypeUnknown, "Cast to abstract tensor failed!");
  auto type_ptr = abstract_tensor->element()->GetTypeTrack();
  MS_CHECK_TRUE_MSG(type_ptr != nullptr, kTypeUnknown, "type_ptr is nullptr");
  return type_ptr->type_id();
}

STATUS UpdateFuncGraphInputsAndOutputsDtype(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, RET_ERROR, "func_graph is nullptr!");
  // update graph inputs dtype
  size_t idx = 0;
  for (auto &input : func_graph->get_inputs()) {
    TypeId type = GetParameterDtype(input->cast<ParameterPtr>());
    ConverterInnerContext::GetInstance()->UpdateGraphInputDType(idx, type);
    idx++;
  }
  // update graph outputs dtype
  auto graph_return = func_graph->get_return();
  idx = 0;
  for (auto &input : graph_return->inputs()) {
    if (input->isa<CNode>()) {
      if (utils::isa<abstract::AbstractTuple>(input->abstract())) {
        auto tuple = std::reinterpret_pointer_cast<abstract::AbstractTuple>(input->abstract());
        if (tuple == nullptr) {
          MS_LOG(ERROR) << "tuple is nullptr";
          return RET_ERROR;
        }
        for (const auto &tuple_item : tuple->elements()) {
          MS_CHECK_TRUE_MSG(tuple_item != nullptr, RET_ERROR, "tuple_item is nullptr!");
          if (utils::isa<abstract::AbstractTuple>(tuple_item)) {
            continue;
          }
          TypeId type = GetAbstractTensorDtype(tuple_item->cast<abstract::AbstractTensorPtr>());
          ConverterInnerContext::GetInstance()->UpdateGraphOutputDType(idx, type);
          idx++;
        }
      } else if (utils::isa<abstract::AbstractTensor>(input->abstract())) {
        TypeId type = GetAbstractTensorDtype(input->abstract()->cast<abstract::AbstractTensorPtr>());
        ConverterInnerContext::GetInstance()->UpdateGraphOutputDType(idx, type);
        idx++;
      } else {
        ConverterInnerContext::GetInstance()->UpdateGraphOutputDType(idx, kTypeUnknown);
        idx++;
      }
    }
  }
  return RET_OK;
}

STATUS GetFuncGraphOutputsInfo(const FuncGraphPtr &func_graph, std::vector<std::pair<AnfNodePtr, int64_t>> *outputs,
                               std::vector<std::string> *output_names, std::vector<std::vector<int64_t>> *output_dims) {
  MS_CHECK_TRUE_MSG(outputs != nullptr, lite::RET_ERROR, "Output is nullptr.");
  MS_CHECK_TRUE_MSG(output_names != nullptr, lite::RET_ERROR, "Output names is nullptr.");
  MS_CHECK_TRUE_MSG(output_dims != nullptr, lite::RET_ERROR, "Output dims is nullptr.");
  AnfNodePtr return_input = func_graph->output();
  CHECK_NULL_RETURN(return_input);
  if (TraceOutput(return_input, outputs, output_names, output_dims) != lite::RET_OK) {
    MS_LOG(ERROR) << "Trace output failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS UpdateGraphOutputName(schema::MetaGraphT *meta_graph) {
  MS_CHECK_TRUE_MSG(meta_graph != nullptr, RET_NULL_PTR, "meta_graph is nullptr");
  auto output_names = ConverterInnerContext::GetInstance()->GetGraphOutputTensorNames();
  if (output_names.size() > meta_graph->outputIndex.size()) {
    MS_LOG(ERROR) << "the num of setting output_names is greater than actual, " << output_names.size() << " > "
                  << meta_graph->outputIndex.size() << ".";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return RET_ERROR;
  }
  for (size_t idx = 0; idx < output_names.size(); idx++) {
    auto &tensor = meta_graph->allTensors.at(meta_graph->outputIndex.at(idx));
    tensor->name = output_names.at(idx);
  }
  return RET_OK;
}

int TransferMetaGraph(const schema::MetaGraphT &graph, void **model_buf, size_t *size) {
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "input model_buf invalid";
    return RET_ERROR;
  }
  if (size == nullptr) {
    MS_LOG(ERROR) << "input size invalid";
    return RET_ERROR;
  }

  /* model_buf malloc here, free outside */
  if (*model_buf != nullptr) {
    MS_LOG(ERROR) << "input model_buf must be nullptr";
    return RET_ERROR;
  }
  flatbuffers::FlatBufferBuilder builder(MAX_GRAPH_SIZE);
  auto offset = schema::MetaGraph::Pack(builder, &graph);
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  *size = builder.GetSize();
  auto content = builder.GetBufferPointer();
  if (content == nullptr) {
    MS_LOG(ERROR) << "GetBufferPointer nullptr";
    return RET_ERROR;
  }
  *model_buf = new (std::nothrow) char[*size];
  if (*model_buf == nullptr) {
    MS_LOG(ERROR) << "malloc model_buf failed";
    return RET_ERROR;
  }
  return memcpy_s(*model_buf, *size, content, *size);
}

int InitEncryptKey(const std::shared_ptr<ConverterPara> &param, unsigned char *encKey, size_t *keyLen) {
  if (!param->enable_encryption) {
    return RET_OK;
  }
  if (param->encrypt_key.empty()) {
    MS_LOG(ERROR) << "param->encrypt_key is empty.";
    return RET_INPUT_PARAM_INVALID;
  }
  *keyLen = lite::Hex2ByteArray(param->encrypt_key, encKey, kEncMaxLen);
  if (*keyLen != kEncMaxLen) {
    MS_LOG(ERROR) << "enc_key must expressed in hexadecimal characters "
                  << " and only support AES-GCM method and the key length is " << kEncMaxLen;
    return RET_INPUT_PARAM_INVALID;
  }

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
