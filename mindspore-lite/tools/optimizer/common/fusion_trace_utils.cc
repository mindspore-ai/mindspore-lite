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
#include "tools/optimizer/common/fusion_trace_utils.h"
#include <algorithm>
#include <queue>
#include <set>
#include <string>
#include <vector>
#include "base/base.h"
#include "ir/dtype/type_id.h"
#include "ir/func_graph.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "utils/log_adapter.h"
#include "src/common/log_adapter.h"
#include "src/common/ops/anf_utils.h"

namespace mindspore::opt {
namespace {
PrimitivePtr GetPrimitiveFromCNode(const CNodePtr &cnode) {
  if (cnode == nullptr || cnode->size() == 0 || cnode->input(0) == nullptr) {
    return nullptr;
  }
  return GetValueNode<PrimitivePtr>(cnode->input(0));
}

std::vector<std::string> SplitSemicolon(const std::string &str) {
  std::vector<std::string> items;
  if (str.empty()) {
    return items;
  }
  size_t start = 0;
  while (start <= str.size()) {
    size_t pos = str.find(';', start);
    if (pos == std::string::npos) {
      auto item = str.substr(start);
      if (!item.empty()) items.push_back(item);
      break;
    }
    auto item = str.substr(start, pos - start);
    if (!item.empty()) items.push_back(item);
    start = pos + 1;
  }
  return items;
}

std::string JoinSemicolon(const std::vector<std::string> &items) {
  std::string result;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i != 0) {
      result += ";";
    }
    result += items[i];
  }
  return result;
}

// Merge `extra` into the semicolon-separated attribute `attr_name` of `prim`, order preserved, dedup.
void MergeAttr(const PrimitivePtr &prim, const std::string &attr_name, const std::vector<std::string> &extra) {
  if (prim == nullptr || extra.empty()) {
    return;
  }
  std::vector<std::string> merged;
  auto attr = prim->GetAttr(attr_name);
  if (attr != nullptr) {
    auto current = SplitSemicolon(GetValue<std::string>(attr));
    merged.insert(merged.end(), current.begin(), current.end());
  }
  for (const auto &item : extra) {
    if (std::find(merged.begin(), merged.end(), item) == merged.end()) {
      merged.push_back(item);
    }
  }
  prim->AddAttr(attr_name, MakeValue(JoinSemicolon(merged)));
}

std::string GetAttrString(const AnfNodePtr &node, const std::string &attr_name) {
  auto cnode = dyn_cast<CNode>(node);
  if (cnode == nullptr) {
    return "";
  }
  auto prim = GetPrimitiveFromCNode(cnode);
  if (prim == nullptr) {
    return "";
  }
  auto attr = prim->GetAttr(attr_name);
  if (attr == nullptr) {
    return "";
  }
  return GetValue<std::string>(attr);
}

void CollectAllGraphs(const FuncGraphPtr &root, std::set<FuncGraphPtr> *all_graphs) {
  if (root == nullptr || all_graphs == nullptr || all_graphs->find(root) != all_graphs->end()) {
    return;
  }
  all_graphs->insert(root);
  std::queue<FuncGraphPtr> todo;
  todo.push(root);
  while (!todo.empty()) {
    auto graph = todo.front();
    todo.pop();
    for (auto &cnode : graph->GetOrderedCnodes()) {
      for (auto &input : cnode->inputs()) {
        auto sub_graph = (input == nullptr) ? nullptr : GetValueNode<FuncGraphPtr>(input);
        if (sub_graph == nullptr) {
          continue;
        }
        if (all_graphs->insert(sub_graph).second) {
          todo.push(sub_graph);
        }
      }
    }
  }
}
}  // namespace

void TraceOriginInfo(const CNodePtr &node, const std::string &origin_name, const std::string &origin_type,
                     const std::string &origin_tensors) {
  auto prim = GetPrimitiveFromCNode(node);
  if (prim == nullptr) {
    return;
  }
  if (!origin_name.empty()) {
    MergeAttr(prim, kOriginNamesAttr, {origin_name});
  }
  if (!origin_type.empty()) {
    MergeAttr(prim, kOriginOpTypesAttr, {origin_type});
  }
  if (!origin_tensors.empty() && prim->GetAttr(kOriginTensorsAttr) == nullptr) {
    prim->AddAttr(kOriginTensorsAttr, MakeValue(origin_tensors));
  }
}

void TraceAbsorbNames(const CNodePtr &survivor, const AnfNodePtr &absorbed) {
  if (survivor == nullptr || absorbed == nullptr) {
    return;
  }
  auto survivor_prim = GetPrimitiveFromCNode(survivor);
  if (survivor_prim == nullptr) {
    return;
  }
  std::vector<std::string> names;
  std::vector<std::string> types;
  auto absorbed_cnode = dyn_cast<CNode>(absorbed);
  if (absorbed_cnode != nullptr) {
    // Carry the absorbed node's ORIGINAL names. Its current fullname is only used when it has no
    // origin records yet (e.g. pass-created helper nodes have synthetic names, which would leak
    // into the origin domain and pollute per-layer matching).
    auto absorbed_names = SplitSemicolon(GetAttrString(absorbed_cnode, kOriginNamesAttr));
    if (absorbed_names.empty()) {
      const auto &current_name = absorbed_cnode->fullname_with_scope();
      if (!current_name.empty()) {
        names.push_back(current_name);
      }
    } else {
      names.insert(names.end(), absorbed_names.begin(), absorbed_names.end());
    }
    auto absorbed_types = SplitSemicolon(GetAttrString(absorbed_cnode, kOriginOpTypesAttr));
    types.insert(types.end(), absorbed_types.begin(), absorbed_types.end());
    // Keep the TFLite anchor: the survivor keeps its input side and always takes the absorbed
    // output side (absorbed is downstream, so its outputs are the fusion's external outputs).
    // A survivor with no anchor of its own (e.g. ONNX nodes never record one) adopts the
    // absorbed anchor whole instead of splicing onto an empty in-side.
    {
      auto absorbed_anchor = GetAttrString(absorbed_cnode, kOriginTensorsAttr);
      if (!absorbed_anchor.empty()) {
        auto survivor_anchor = GetAttrString(survivor, kOriginTensorsAttr);
        if (survivor_anchor.empty()) {
          survivor_prim->AddAttr(kOriginTensorsAttr, MakeValue(absorbed_anchor));
        } else {
          auto in_side = survivor_anchor.substr(0, survivor_anchor.find(" -> out:"));
          auto out_side = absorbed_anchor.substr(absorbed_anchor.find(" -> out:"));
          survivor_prim->AddAttr(kOriginTensorsAttr, MakeValue(in_side + out_side));
        }
      }
    }
  }
  MergeAttr(survivor_prim, kOriginNamesAttr, names);
  MergeAttr(survivor_prim, kOriginOpTypesAttr, types);
}

void TraceNewFusionNode(const CNodePtr &new_node, const std::vector<AnfNodePtr> &parts) {
  for (const auto &part : parts) {
    TraceAbsorbNames(new_node, part);
  }
}

std::vector<std::string> GetSelfAndOriginNames(const AnfNodePtr &node) {
  std::vector<std::string> names;
  if (node == nullptr) {
    return names;
  }
  auto cnode = dyn_cast<CNode>(node);
  if (cnode == nullptr) {
    return names;
  }
  names.push_back(cnode->fullname_with_scope());
  for (const auto &name : SplitSemicolon(GetAttrString(cnode, kOriginNamesAttr))) {
    if (std::find(names.begin(), names.end(), name) == names.end()) {
      names.push_back(name);
    }
  }
  return names;
}

std::vector<std::string> GetOriginOpTypes(const AnfNodePtr &node) {
  return SplitSemicolon(GetAttrString(node, kOriginOpTypesAttr));
}

std::string GetOriginTensors(const AnfNodePtr &node) { return GetAttrString(node, kOriginTensorsAttr); }

void StripOriginNames(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return;
  }
  std::set<FuncGraphPtr> all_graphs;
  CollectAllGraphs(func_graph, &all_graphs);
  // Subgraphs referenced through primitive attributes (control flow) are not reachable via cnode
  // inputs; the manager tracks every managed graph, take the union so no record leaks into export.
  auto manager = func_graph->manager();
  if (manager != nullptr) {
    for (auto &graph : manager->func_graphs()) {
      all_graphs.insert(graph);
    }
  }
  for (const auto &graph : all_graphs) {
    for (auto &cnode : graph->GetOrderedCnodes()) {
      auto prim = GetPrimitiveFromCNode(cnode);
      if (prim == nullptr) {
        continue;
      }
      prim->EraseAttr(kOriginNamesAttr);
      prim->EraseAttr(kOriginOpTypesAttr);
      prim->EraseAttr(kOriginTensorsAttr);
    }
  }
}
}  // namespace mindspore::opt
