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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_FUSION_TRACE_UTILS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_FUSION_TRACE_UTILS_H_

#include <string>
#include <vector>
#include "ir/anf.h"
#include "ir/func_graph.h"

namespace mindspore::opt {
// Fusion tracing attributes. They live only in converter memory and are stripped before export
// (see StripOriginNames), so the exported .ms keeps byte-for-byte compatibility.
constexpr char kOriginNamesAttr[] = "origin_names";       // semicolon separated original op names
constexpr char kOriginOpTypesAttr[] = "origin_op_types";  // semicolon separated original op types

// Record origin info right after a frontend parser creates a node. Called once per node at import.
void TraceOriginInfo(const CNodePtr &node, const std::string &origin_name, const std::string &origin_type);

// In-place fusion: `survivor` absorbs `absorbed` (the node being removed).
// MUST be called before the absorbed node is replaced/dropped, after the survivor's primitive type
// is finalized. Carries over all records (including records the absorbed node itself accumulated).
void TraceAbsorbNames(const CNodePtr &survivor, const AnfNodePtr &absorbed);

// New-node fusion: `new_node` is created from `parts`. Call after the node is built, before replace.
void TraceNewFusionNode(const CNodePtr &new_node, const std::vector<AnfNodePtr> &parts);

// The node's current name plus every original name it carries (deduplicated, current name first).
std::vector<std::string> GetSelfAndOriginNames(const AnfNodePtr &node);

// Every ORIGINAL op name the node carries (its own plus absorbed ones), without the current name.
// The whitelist's name domain: converted (framework) names must not whitelist a layer.
std::vector<std::string> GetOriginNames(const AnfNodePtr &node);

// Every original op type the node carries (its own plus absorbed ones). May be empty (e.g. MindIR).
std::vector<std::string> GetOriginOpTypes(const AnfNodePtr &node);

// Remove all tracing attributes from every node of the graph (including subgraphs).
// Unconditional: fusion happens even without quantization, so this must not hang off the quant branch.
void StripOriginNames(const FuncGraphPtr &func_graph);

// ---- tracing switch (refined plan-2) ----
// Tracing (parser recording + every transport) is meaningful only when a per-layer list or the
// decision report needs the origin domain; the converter turns it off otherwise so plain
// conversions carry zero overhead. Default on for unit tests and direct API users.
void SetFusionTraceEnabled(bool enabled);
bool IsFusionTraceEnabled();

// ---- pass-level graph-diff transport (for the Pass-framework fusion passes) ----
// Ungated primitives used by the diff transport in LitePassManager::Run.
std::set<CNodePtr> SnapshotRecordedCnodes(const FuncGraphPtr &func_graph);
void CarryOriginRecords(const CNodePtr &survivor, const AnfNodePtr &absorbed);
bool HasOriginRecords(const AnfNodePtr &node);
}  // namespace mindspore::opt

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_FUSION_TRACE_UTILS_H_
