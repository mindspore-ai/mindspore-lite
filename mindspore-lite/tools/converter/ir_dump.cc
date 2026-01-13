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
#include "tools/converter/ir_dump.h"
#include <sys/stat.h>
#if defined(_WIN32) || defined(_WIN64)
#include <stdlib.h>
#endif
#include <list>
#include <utility>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include "ir/func_graph.h"
#include "ir/graph_utils.h"
#include "ir/value.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/file_utils.h"
namespace mindspore::lite {
namespace {
static int kNumGraphIndex = 0;
constexpr char kNameDumpGraphLevel0 = '0';
constexpr char kNameDumpGraphLevel1 = '1';
}  // namespace

Status PrintNodeOutputType(std::ostringstream &buffer, const AnfNodePtr &node) {
  MS_CHECK_TRUE_MSG(node != nullptr, kLiteError, "node is nullptr");
  // for dtype
  TypePtr type = dyn_cast<Type>(node->Type());
  if (type == nullptr) {
    buffer << "<null>";
    return kSuccess;
  }
  buffer << "<" << type;
  // for shape
  abstract::BaseShapePtr shape = dyn_cast<abstract::BaseShape>(node->Shape());
  if (shape != nullptr) {
    buffer << ", ";
    shape->ToStringWithBuffer(buffer);
  }
  // for value
  ValuePtr tensor_value = nullptr;
  auto abstract = node->abstract();
  if (abstract != nullptr) {
    if (abstract->isa<abstract::AbstractTensor>()) {
      tensor_value = abstract->BuildValue();
    }
  }
  if (tensor_value != nullptr && tensor_value != kValueAny) {
    buffer << ", value=...";
  }
  buffer << ">";
  return kSuccess;
}

Status DumpParameterNodes(const FuncGraphPtr &graph, std::ostringstream &buffer,
                          std::map<AnfNodePtr, int32_t> *para_map, int32_t *para_num) {
  MS_CHECK_TRUE_MSG(graph != nullptr, kLiteError, "graph is nullptr.");
  MS_CHECK_TRUE_MSG(para_map != nullptr, kLiteError, "para_map is nullptr.");

  std::vector<AnfNodePtr> parameters = graph->parameters();
  buffer << "# Total params size: " << parameters.size() << std::endl;

  if (parameters.empty()) {
    return kSuccess;
  }
  buffer << "# Params:" << std::endl;
  // Dump parameters
  for (const auto &param : parameters) {
    if (param == nullptr) {
      continue;
    }
    auto parameter_ptr = param->cast<ParameterPtr>();
    if (parameter_ptr == nullptr) {
      MS_LOG(WARNING) << "param cannot cast to ParameterPtr";
      continue;
    }
    buffer << "%param" << *para_num << "_" << parameter_ptr->name() << ": ";
    // Print parameters' type and shape
    auto status = PrintNodeOutputType(buffer, param);
    MS_CHECK_TRUE_MSG(status == kSuccess, kLiteError, "PrintNodeOutputType failed.");
    if (parameter_ptr->has_default()) {
      buffer << " : has_default";
    }
    buffer << std::endl;
    (*para_map)[param] = (*para_num)++;
  }
  return kSuccess;
}

Status PrintNodeInputType(std::ostringstream &buffer, const AnfNodePtr &node) {
  MS_CHECK_TRUE_MSG(node != nullptr, kLiteError, "node is nullptr");
  const auto &inputs = GetInputs(node);
  size_t len = inputs.size();
  if (len > 1) {
    // Skip inputs[0] which is Primitive value node
    for (size_t i = 1; i < len; ++i) {
      AnfNodePtr in = inputs[i];
      if (i != 1) {
        buffer << ", ";
      }
      auto status = PrintNodeOutputType(buffer, in);
      MS_CHECK_TRUE_MSG(status == kSuccess, kLiteError, "PrintNodeOutputType failed.");
    }
  }
  return kSuccess;
}

void PrintNodeOutputSymbolicInfo(std::ostringstream &buffer, const AnfNodePtr &node) {
  if (node == nullptr) {
    return;
  }
  auto abstract = node->abstract();
  if (abstract == nullptr) {
    buffer << "<null>";
    return;
  }
  auto shape = abstract->GetSymbolicShape();
  auto value = abstract->GetSymbolicValue();
  if (shape != nullptr || value != nullptr) {
    if (shape != nullptr) {
      buffer << "S:" << shape->ToString();
    }
    if (value != nullptr) {
      buffer << "V:" << value->ToString();
    }
  } else {
    buffer << "<null>";
  }
}

void PrintNodeInputSymbolicInfo(std::ostringstream &buffer, const AnfNodePtr &node) {
  if (node == nullptr) {
    return;
  }
  const auto &inputs = GetInputs(node);
  if (inputs.size() <= 1) {
    return;
  }
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (i != 1) {
      buffer << ", ";
    }
    PrintNodeOutputSymbolicInfo(buffer, inputs[i]);
  }
}

Status DumpSymbolicInfo(const AnfNodePtr &node, const FuncGraphPtr &fg, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  MS_CHECK_TRUE_MSG(node != nullptr, kLiteError, "node is nullptr");
  MS_CHECK_TRUE_MSG(fg != nullptr, kLiteError, "graph is nullptr");
  MS_CHECK_TRUE_MSG(gsub != nullptr, kLiteError, "subgraph is nullptr");
  if (fg->symbol_engine() == nullptr) {
    return kSuccess;
  }
  if (node != fg->get_return()) {
    gsub->buffer << "      : (";
    PrintNodeInputSymbolicInfo(gsub->buffer, node);
    gsub->buffer << ") -> (";
    PrintNodeOutputSymbolicInfo(gsub->buffer, node);
    gsub->buffer << ")";
  } else {
    gsub->buffer << "      : (";
    PrintNodeInputSymbolicInfo(gsub->buffer, node);
    gsub->buffer << ")";
  }
  gsub->buffer << std::endl;
  return kSuccess;
}

Status DumpGraphInfo(const FuncGraphPtr &graph, std::ostringstream &buffer, size_t sub_graphs_size) {
  MS_CHECK_TRUE_MSG(graph != nullptr, kLiteError, "graph is nullptr");

  buffer << "# IR entry: @" << graph->ToString() << std::endl;
  buffer << "# Total subgraphs: " << sub_graphs_size << std::endl;
  buffer << std::endl;
  if (!graph->attrs().empty()) {
    buffer << "# Attrs:" << std::endl;
    for (const auto &attr : graph->attrs()) {
      buffer << attr.first << ": ";
      if (attr.second->isa<BoolImm>()) {
        buffer << GetValue<bool>(attr.second);
      } else if (attr.second->isa<StringImm>()) {
        buffer << GetValue<std::string>(attr.second);
      }
      buffer << std::endl;
    }
    buffer << std::endl;
  }
  return kSuccess;
}

void DumpParameterOperator(const AnfNodePtr &node, const std::shared_ptr<SubGraphIRInfo> &gsub, const AnfNodePtr &op) {
  if (op->func_graph() != nullptr && op->func_graph() != node->func_graph()) {
    gsub->buffer << "$(@" << op->func_graph()->ToString() << ":";
  }
  gsub->buffer << op->ToString();
  if (op->func_graph() != nullptr && op->func_graph() != node->func_graph()) {
    gsub->buffer << ")";
  }
}

Status DumpOperator(const AnfNodePtr &node, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  MS_CHECK_TRUE_MSG(gsub != nullptr, kLiteError, "gsub is nullptr");

  auto cnode = dyn_cast<CNode>(node);
  MS_CHECK_TRUE_MSG(cnode != nullptr, kLiteError, "cnode is nullptr");

  AnfNodePtr op = cnode->input(0);
  MS_CHECK_TRUE_MSG(op != nullptr, kLiteError, "op is nullptr");

  if (IsValueNode<FuncGraph>(op)) {
    FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(op);
    if (fg != nullptr) {
      gsub->buffer << "call @" << fg->ToString();
    }
  } else if (op->isa<CNode>()) {
    if (gsub->local_var_map.find(op) != gsub->local_var_map.end()) {
      gsub->buffer << "%" << gsub->local_var_map[op];
    } else {
      auto input = op->cast<CNodePtr>();
      MS_CHECK_TRUE_MSG(input != nullptr, kLiteError, "input is nullptr");
      auto fg = input->func_graph();
      MS_CHECK_TRUE_MSG(fg != nullptr, kLiteError, "fg is nullptr");
      gsub->buffer << "$(@" << fg->ToString() << ":" << input->ToString() << ")";
    }
  } else if (op->isa<ValueNode>()) {
    auto value = GetValueNode(op);
    if (value != nullptr) {
      if (value->isa<Primitive>()) {
        gsub->buffer << value->ToString();
      }
    }
  } else {
    DumpParameterOperator(node, gsub, op);
  }
  return kSuccess;
}

Status DumpParameterInOperand(const AnfNodePtr &node, const AnfNodePtr &in,
                              const std::map<AnfNodePtr, int32_t> &para_map,
                              const std::shared_ptr<SubGraphIRInfo> &gsub) {
  MS_CHECK_TRUE_MSG(node != nullptr, kLiteError, "node is nullptr");
  MS_CHECK_TRUE_MSG(node->func_graph() != nullptr, kLiteError, "func graph is nullptr");
  MS_CHECK_TRUE_MSG(in != nullptr, kLiteError, "in is nullptr");
  MS_CHECK_TRUE_MSG(gsub != nullptr, kLiteError, "gsub is nullptr");

  if (in->func_graph() == nullptr) {
    MS_LOG(INFO) << "Parameter should belong to a func graph. Check func graph: " << node->func_graph();
  }
  if (in->func_graph() != nullptr && in->func_graph() != node->func_graph()) {
    gsub->buffer << "$(@" << in->func_graph()->ToString() << ":";
  } else {
    gsub->buffer << "%";
  }
  auto iter = para_map.find(in);
  if (iter == para_map.end()) {
    gsub->buffer << "para_" << in->ToString();
  } else {
    gsub->buffer << "para" << iter->second << "_" << in->ToString();
  }
  if (in->func_graph() != nullptr && in->func_graph() != node->func_graph()) {
    gsub->buffer << ")";
  }
  return kSuccess;
}

Status DumpOperands(const AnfNodePtr &node, const std::map<AnfNodePtr, int32_t> &para_map,
                    const std::shared_ptr<SubGraphIRInfo> &gsub) {
  MS_CHECK_TRUE_MSG(node != nullptr, kLiteError, "node is nullptr.");
  MS_CHECK_TRUE_MSG(gsub != nullptr, kLiteError, "gsub is nullptr");

  gsub->buffer << "(";
  const auto &inputs = GetInputs(node);
  size_t len = inputs.size();
  if (len <= 1) {
    gsub->buffer << ")";
    return kLiteError;
  }
  for (size_t i = 1; i < len; ++i) {
    AnfNodePtr in = inputs[i];
    MS_EXCEPTION_IF_NULL(in);
    if (i != 1) {
      gsub->buffer << ", ";
    }
    if (in->isa<Parameter>()) {
      DumpParameterInOperand(node, in, para_map, gsub);
    } else if (in->isa<CNode>()) {
      auto iter = gsub->local_var_map.find(in);
      if (iter != gsub->local_var_map.end()) {
        gsub->buffer << "%" << iter->second;
      } else {
        auto input = in->cast<CNodePtr>();
        MS_CHECK_TRUE_MSG(input != nullptr, kLiteError, "input is nullptr");
        auto fg = input->func_graph();
        MS_CHECK_TRUE_MSG(fg != nullptr, kLiteError, "fg is nullptr");
        gsub->buffer << "$(@" << fg->ToString() << ":" << input->ToString() << ")";
      }
    } else if (in->isa<ValueNode>() && !IsValueNode<FuncGraph>(in)) {
      auto value = GetValueNode(in);
      MS_CHECK_TRUE_MSG(value != nullptr, kLiteError, "value is nullptr");
      if (IsPrimitiveCNode(node, prim::kPrimVirtualViewGrad) && value->isa<Primitive>()) {
        gsub->buffer << value->ToString();
      }
    } else if (IsValueNode<FuncGraph>(in)) {
      FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(in);
      MS_CHECK_TRUE_MSG(fg != nullptr, kLiteError, "fg is nullptr");
      gsub->buffer << "@" << fg->ToString();
    } else {
      gsub->buffer << in->ToString();
    }
  }
  gsub->buffer << ")";
  return kSuccess;
}

Status DumpAttrs(const mindspore::HashMap<std::string, ValuePtr> &attrs, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  int i = 0;
  for (const auto &attr : attrs) {
    if (i++ != 0) {
      gsub->buffer << ", ";
    }
    gsub->buffer << attr.first << ": ";
    if (attr.second == nullptr) {
      gsub->buffer << "null";
    } else {
      gsub->buffer << attr.second->ToString();
    }
  }
  return kSuccess;
}

Status DumpOperateAttrs(const AnfNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  MS_CHECK_TRUE_MSG(op != nullptr, kLiteError, "op is nullptr");
  MS_CHECK_TRUE_MSG(gsub != nullptr, kLiteError, "gsub is nullptr");
  if (IsValueNode<Primitive>(op)) {
    PrimitivePtr primitive = GetValueNode<PrimitivePtr>(op);
    MS_EXCEPTION_IF_NULL(primitive);
    if (!primitive->instance_name().empty()) {
      gsub->buffer << " {";
      gsub->buffer << "instance name: ";
      gsub->buffer << primitive->instance_name();
      gsub->buffer << "}";
    }
    auto attrs = primitive->attrs();
    if (!attrs.empty()) {
      gsub->buffer << " primitive_attrs: {";
      DumpAttrs(attrs, gsub);
      gsub->buffer << "}";
    }
  }
  return kSuccess;
}

Status DumpCNodeAttrs(const CNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  MS_CHECK_TRUE_MSG(op != nullptr, kLiteError, "op is nullptr");
  MS_CHECK_TRUE_MSG(gsub != nullptr, kLiteError, "gsub is nullptr");

  if (op->attrs().empty()) {
    return kSuccess;
  }
  auto attrs = op->attrs();
  gsub->buffer << " cnode_attrs: {";
  DumpAttrs(attrs, gsub);
  gsub->buffer << "}";
  return kSuccess;
}

Status DumpCNodePrimalAttrs(const CNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  MS_CHECK_TRUE_MSG(op != nullptr, kLiteError, "op is nullptr");
  MS_CHECK_TRUE_MSG(gsub != nullptr, kLiteError, "gsub is nullptr");

  if (op->primal_attrs().empty()) {
    return kSuccess;
  }
  auto primal_attrs = op->primal_attrs();
  gsub->buffer << " cnode_primal_attrs: {";
  DumpAttrs(primal_attrs, gsub);
  gsub->buffer << "}";
  return kSuccess;
}

Status DumpShape(const AnfNodePtr &node, const FuncGraphPtr &sub_graph, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  MS_CHECK_TRUE_MSG(node != nullptr, kLiteError, "node is nullptr");
  MS_CHECK_TRUE_MSG(sub_graph != nullptr, kLiteError, "subgraph is nullptr");
  MS_CHECK_TRUE_MSG(gsub != nullptr, kLiteError, "gsub is nullptr");

  gsub->buffer << std::endl;
  if (node != sub_graph->get_return()) {
    gsub->buffer << "      : (";
    PrintNodeInputType(gsub->buffer, node);
    gsub->buffer << ") -> (";
    auto status = PrintNodeOutputType(gsub->buffer, node);
    MS_CHECK_TRUE_MSG(status == kSuccess, kLiteError, "PrintNodeOutputType failed.");
    gsub->buffer << ")";
  } else {
    gsub->buffer << "      : (";
    PrintNodeInputType(gsub->buffer, node);
    gsub->buffer << ")";
  }
  gsub->buffer << std::endl;
  return kSuccess;
}

Status DumpParameters(const FuncGraphPtr &func_graph, std::ostringstream &oss) {
  std::vector<AnfNodePtr> parameters = func_graph->parameters();
  oss << "# Parameters: " << parameters.size() << ", (";
  if (parameters.size() == 1) {
    MS_CHECK_TRUE_MSG(parameters[0] != nullptr, kLiteError, "parameters[0] is nullptr");
    auto status = PrintNodeOutputType(oss, parameters[0]);
    MS_CHECK_TRUE_MSG(status == kSuccess, kLiteError, "PrintNodeOutputType failed");
  } else if (parameters.size() > 1) {
    for (size_t idx = 0; idx < parameters.size() - 1; idx++) {
      MS_CHECK_TRUE_MSG(parameters[idx] != nullptr, kLiteError, "parameters[idx] is nullptr");
      auto status = PrintNodeOutputType(oss, parameters[idx]);
      MS_CHECK_TRUE_MSG(status == kSuccess, kLiteError, "PrintNodeOutputType failed.");
      oss << ", ";
    }
    MS_CHECK_TRUE_MSG(parameters[parameters.size() - 1] != nullptr, kLiteError,
                      "parameters[parameters.size() - 1] is nullptr");
    auto status = PrintNodeOutputType(oss, parameters[parameters.size() - 1]);
    MS_CHECK_TRUE_MSG(status == kSuccess, kLiteError, "PrintNodeOutputType failed.");
  }
  oss << ")\n";
  return kSuccess;
}

Status DumpCNode(const CNodePtr &node, const FuncGraphPtr &sub_graph, const std::map<AnfNodePtr, int32_t> &para_map,
                 const std::shared_ptr<SubGraphIRInfo> &gsub) {
  MS_CHECK_TRUE_MSG(node != nullptr, kLiteError, "node is nullptr");
  MS_CHECK_TRUE_MSG(sub_graph != nullptr, kLiteError, "sub graph is nullptr");
  MS_CHECK_TRUE_MSG(gsub != nullptr, kLiteError, "gsub is nullptr");

  if (node != sub_graph->get_return()) {
    gsub->buffer << "  %" << gsub->node_index << "(" << node->ToString() << ") = ";
    gsub->local_var_map[node] = gsub->node_index++;
  } else {
    gsub->buffer << "  ";
  }

  if (node->weak_inputs().empty()) {
    MS_LOG(ERROR) << "Input of CNode is empty";
    return kLiteError;
  }
  auto status = DumpOperator(node, gsub);
  MS_CHECK_TRUE_MSG(status == kSuccess, kLiteError, "DumpOperator failed.");

  status = DumpOperands(node, para_map, gsub);
  MS_CHECK_TRUE_MSG(status == kSuccess, kLiteError, "DumpOperands failed.");

  AnfNodePtr op = node->input(0);
  status = DumpOperateAttrs(op, gsub);
  MS_CHECK_TRUE_MSG(status == kSuccess, kLiteError, "DumpOperateAttrs failed.");

  status = DumpCNodeAttrs(node, gsub);
  MS_CHECK_TRUE_MSG(status == kSuccess, kLiteError, "DumpCNodeAttrs failed.");

  status = DumpCNodePrimalAttrs(node, gsub);
  MS_CHECK_TRUE_MSG(status == kSuccess, kLiteError, "DumpCNodePrimalAttrs failed.");

  if (node == sub_graph->get_return()) {
    status = DumpShape(node, sub_graph, gsub);
    MS_CHECK_TRUE_MSG(status == kSuccess, kLiteError, "DumpShape failed.");

    status = DumpSymbolicInfo(node, sub_graph, gsub);
    MS_CHECK_TRUE_MSG(status == kSuccess, kLiteError, "DumpSymbolicInfo failed.");
  } else {
    gsub->buffer << std::endl;
  }
  gsub->buffer << "      # Fullname with scope: (" << node->fullname_with_scope() << ")" << std::endl;
  return kSuccess;
}

Status OutputOrderList(const FuncGraphPtr &sub_graph, std::ostringstream &oss) {
  auto &order_list = sub_graph->order_list();
  if (order_list.empty()) {
    return kLiteError;
  }
  constexpr int width = 4;
  oss << "# Order:\n";
  int i = 1;
  for (auto &weak_node : order_list) {
    const auto &node = weak_node.lock();
    if (node != nullptr) {
      oss << '#' << std::setw(width) << i << ": " << node->DebugString() << '\n';
    }
    ++i;
  }
  return kSuccess;
}

Status CollectGraphInfo(const std::vector<AnfNodePtr> &nodes, std::map<AnfNodePtr, int32_t> *para_map,
                        std::map<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> *const sub_graphs, int32_t total_para,
                        DumpGraphLevel dump_level) {
  MS_CHECK_TRUE_MSG(para_map != nullptr, kLiteError, "para_map is nullptr");
  MS_CHECK_TRUE_MSG(sub_graphs != nullptr, kLiteError, "sub graph is nullptr");
  for (const auto &node : nodes) {
    MS_CHECK_TRUE_MSG(node != nullptr, kLiteError, "node is nullptr");
    FuncGraphPtr sub_graph = node->func_graph();
    if (sub_graph == nullptr) {
      MS_LOG(INFO) << "Node[" << node->ToString() << "] belongs to no graph!";
      continue;
    }
    std::shared_ptr<SubGraphIRInfo> gsub;
    auto iter = sub_graphs->find(sub_graph);
    if (iter == sub_graphs->end()) {
      gsub = std::make_shared<SubGraphIRInfo>();
      gsub->node_index = 0;
      (void)sub_graphs->emplace(sub_graph, gsub);
      const std::vector<AnfNodePtr> &parameters = sub_graph->parameters();
      for (size_t idx = 0; idx < parameters.size(); ++idx) {
        MS_CHECK_TRUE_MSG(parameters[idx] != nullptr, kLiteError, "parameters[idx]  is nullptr");
        if ((*para_map).find(parameters[idx]) == para_map->end()) {
          (void)(*para_map).emplace(parameters[idx], total_para);
          ++total_para;
        }
      }
    } else {
      gsub = iter->second;
    }
    if (!node->isa<Parameter>()) {
      if (node->isa<CNode>()) {
        // Print and record output of operator if it is not 'Return'
        gsub->cnode_num++;
        auto status = DumpCNode(node->cast<CNodePtr>(), sub_graph, *para_map, gsub);
        MS_CHECK_TRUE_MSG(status == kSuccess, kLiteError, "DumpCNode failed.");
      } else {
        gsub->buffer << "  " << node->ToString() << std::endl;
      }
    }
  }
  return kSuccess;
}

Status DumpSubgraph(const std::map<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> *sub_graphs,
                    const FuncGraphPtr &graph, std::map<AnfNodePtr, int32_t> *para_map, std::ostringstream &oss,
                    DumpGraphLevel dump_level) {
  MS_CHECK_TRUE_MSG(sub_graphs != nullptr, kLiteError, "sub graph is nullptr.");
  MS_CHECK_TRUE_MSG(graph != nullptr, kLiteError, "graph is nullptr");

  for (const auto &sg : *sub_graphs) {
    MS_CHECK_TRUE_MSG(sg.first != nullptr, kLiteError, "sg is nullptr");
    oss << "subgraph @" << sg.first->ToString();
    oss << "(";
    if (sg.first != graph) {
      auto status = DumpParameters(sg.first, oss);
      MS_CHECK_TRUE_MSG(status == kSuccess, kLiteError, "DumpParameters failed.");

      std::vector<AnfNodePtr> parameters = sg.first->parameters();
      if (parameters.size() == 1) {
        MS_CHECK_TRUE_MSG(parameters[0] != nullptr, kLiteError, "parameters[0] is nullptr");
        oss << "%para" << (*para_map)[parameters[0]] << "_" << parameters[0]->ToString();
      } else if (parameters.size() > 1) {
        for (size_t idx = 0; idx < parameters.size() - 1; idx++) {
          MS_CHECK_TRUE_MSG(parameters[idx] != nullptr, kLiteError, "parameters[idx] is nullptr");
          oss << "%para" << (*para_map)[parameters[idx]] << "_" << parameters[idx]->ToString();
          oss << ", ";
        }
        MS_CHECK_TRUE_MSG(parameters[parameters.size() - 1] != nullptr, kLiteError,
                          "parameters[parameters.size() - 1] is nullptr");
        oss << "%para" << (*para_map)[parameters[parameters.size() - 1]] << "_"
            << parameters[parameters.size() - 1]->ToString();
      }
    }
    oss << ") {" << std::endl;
    // dump subgraph node
    MS_CHECK_TRUE_MSG(sg.second != nullptr, kLiteError, "sg.second is nullptr");
    oss << sg.second->buffer.str();
    oss << "}" << std::endl;
    OutputOrderList(sg.first, oss);
    oss << std::endl;
    oss << std::endl;
  }
  return kSuccess;
}

Status DumpNodeInfo(const std::map<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> *sub_graphs, std::ostringstream &oss,
                    const std::vector<AnfNodePtr> &nodes) {
  oss << "Node counting information:" << std::endl;
  oss << "Total number of nodes: " << nodes.size() << std::endl;
  auto total_cnode =
    std::accumulate((*sub_graphs).cbegin(), (*sub_graphs).cend(), 0,
                    [](const int32_t previous, const std::pair<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> &pair) {
                      return previous + pair.second->cnode_num;
                    });
  oss << "Total number of cnodes: " << total_cnode << std::endl;
  oss << std::endl;
  return kSuccess;
}

Status DumpIr(std::ostringstream &out_oss, const FuncGraphPtr &graph, DumpGraphLevel dump_level) {
  MS_CHECK_TRUE_MSG(graph != nullptr, kLiteError, "graph is nullptr");
  auto nodes = TopoSort(graph->get_return());
  // Dump for param
  std::map<AnfNodePtr, int32_t> param_map;
  int32_t total_para = 0;
  std::ostringstream param_oss;
  auto status = DumpParameterNodes(graph, param_oss, &param_map, &total_para);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "DumpParameterNodes failed.";
    return kLiteError;
  }
  std::map<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> sub_graphs;
  status = CollectGraphInfo(nodes, &param_map, &sub_graphs, total_para, dump_level);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "CollectGraphInfo failed.";
    return kLiteError;
  }
  status = DumpGraphInfo(graph, out_oss, sub_graphs.size());
  if (status != kSuccess) {
    MS_LOG(ERROR) << "DumpGraphInfo failed.";
    return kLiteError;
  }
  if (dump_level == DumpGraphLevel::kLevel0) {
    out_oss << param_oss.str();
  }
  out_oss << std::endl;
  status = DumpNodeInfo(&sub_graphs, out_oss, nodes);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "DumpNodeInfo failed.";
    return kLiteError;
  }
  status = DumpSubgraph(&sub_graphs, graph, &param_map, out_oss, dump_level);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "DumpSubgraph failed.";
    return kLiteError;
  }
  return kSuccess;
}

Status SaveIrFile(const std::string &dump_file_path, const std::string &str) {
  ChangeFileMode(dump_file_path, S_IWUSR);
  std::ofstream fout(dump_file_path);
  if (!fout.is_open()) {
    MS_LOG(ERROR) << "Open dump file: " << dump_file_path << " failed.";
    return kLiteError;
  }
  fout << str;
  fout.close();
  ChangeFileMode(dump_file_path, S_IRUSR);
  return kSuccess;
}

Status DumpGraph(const std::string &pass_name, const FuncGraphPtr &graph) {
  MS_CHECK_TRUE_MSG(graph != nullptr, kLiteError, "graph is nullptr");
  auto dump_level_env = std::getenv("MSLITE_DUMP_GRAPH_LEVEL");  // MSLITE_DUMP_GRAPH_PATH
  if (dump_level_env == nullptr) {
    MS_LOG(DEBUG) << "not dump graph.";
    return kSuccess;
  }
  DumpGraphLevel level = DumpGraphLevel::kLevel1;
  if (dump_level_env[0] == kNameDumpGraphLevel0) {
    level = DumpGraphLevel::kLevel0;
  } else if (dump_level_env[0] == kNameDumpGraphLevel1) {
    level = DumpGraphLevel::kLevel1;
  } else {
    MS_LOG(ERROR) << "not only support set MSLITE_DUMP_GRAPH_LEVEL to '0' or '1' ";
    return kLiteError;
  }
  std::string path = "./";
  auto dump_path_env = std::getenv("MSLITE_DUMP_GRAPH_PATH");
  if (dump_path_env != nullptr) {
    path = std::string(dump_path_env) + "/";
  }
  std::string real_path_str = RealPath(path.c_str());
  if (real_path_str.empty()) {
    MS_LOG(ERROR) << "dump graph ir path is not valid.";
    return kLiteError;
  }
  std::ostringstream out_oss;
  auto status = DumpIr(out_oss, graph, level);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "DumpIr failed.";
    return kLiteError;
  }
  std::string dump_file_path = real_path_str + "/" + std::to_string(kNumGraphIndex++) + "_" + pass_name + ".ir";
  MS_LOG(INFO) << "dump file path : " << dump_file_path << "dump graph level: " << level;
  status = SaveIrFile(dump_file_path, out_oss.str());
  if (status != kSuccess) {
    MS_LOG(ERROR) << "SaveIrFile failed.";
    return kLiteError;
  }
  return kSuccess;
}
}  // namespace mindspore::lite
