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
#include "tools/converter/quantizer/layer_decision_reporter.h"
#include <algorithm>
#include <fstream>
#include <queue>
#include <unordered_map>
#include "ir/dtype/type_id.h"
#include "infer/custom.h"
#include "ops/op_def_utils.h"
#include "src/common/log_adapter.h"
#include "src/common/ops/anf_utils.h"
#include "tools/converter/micro/coder/opcoders/op_coder_register.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/converter/quantizer/quant_params.h"
#include "ir/quantization_param.h"
#include "ir/value.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/optimizer/common/fusion_trace_utils.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::lite::quant {
namespace {
constexpr size_t kCsvColumnCount = 12;
// One decision-report row as named fields, mirroring the frozen 12-column layout (spec 2.4);
// Cells() serializes the fields in declaration order. The CSV-facing vector<string> stays the
// interchange format of BuildCsvContent/WriteFile — this struct is what the writers populate, so
// every cell's meaning is readable at the write site. Status rows reuse node_type for the status
// value and matched_config for the "key=entry" string, leaving every other field empty.
struct ReportRow {
  std::string origin_names;
  std::string origin_op_types;
  std::string node_name;
  std::string node_type;
  std::string origin_data_type;
  std::string quant_decision;
  std::string reason;
  std::string matched_config;
  std::string micro_coder_available;
  std::string scale;
  std::string zero_point;
  std::string bit_num;

  // Serialization order == declaration order == the frozen CSV column order (spec 2.4).
  std::vector<std::string> Cells() const {
    std::vector<std::string> cells;
    cells.reserve(kCsvColumnCount);
    cells.push_back(origin_names);
    cells.push_back(origin_op_types);
    cells.push_back(node_name);
    cells.push_back(node_type);
    cells.push_back(origin_data_type);
    cells.push_back(quant_decision);
    cells.push_back(reason);
    cells.push_back(matched_config);
    cells.push_back(micro_coder_available);
    cells.push_back(scale);
    cells.push_back(zero_point);
    cells.push_back(bit_num);
    return cells;
  }
};
constexpr size_t kMaxCandidateHints = 3;

// Mirror of the micro coder target map (coder.cc kTargetMap); kept in sync by TC-CON cases.
const std::unordered_map<std::string, mindspore::lite::micro::Target> kMicroTargetMap = {
  {"x86", micro::Target::kX86},           {"ARM64", micro::Target::kARM64}, {"ARM32", micro::Target::kARM32},
  {"Cortex-M", micro::Target::kCortex_M}, {"RISCV", micro::Target::kRiscV}, {"All", micro::Target::kAllTargets}};

// Graphs in discovery (BFS from root) order, so report rows keep a deterministic, execution-like
// order across subgraphs instead of a pointer-ordered set.
void CollectAllGraphs(const FuncGraphPtr &root, std::vector<FuncGraphPtr> *all_graphs) {
  if (root == nullptr || all_graphs == nullptr) {
    return;
  }
  std::set<FuncGraphPtr> seen;
  std::queue<FuncGraphPtr> todo;
  todo.push(root);
  seen.insert(root);
  while (!todo.empty()) {
    auto graph = todo.front();
    todo.pop();
    all_graphs->push_back(graph);
    for (auto &cnode : graph->GetOrderedCnodes()) {
      for (auto &input : cnode->inputs()) {
        auto sub_graph = (input == nullptr) ? nullptr : GetValueNode<FuncGraphPtr>(input);
        if (sub_graph == nullptr) {
          continue;
        }
        if (seen.insert(sub_graph).second) {
          todo.push(sub_graph);
        }
      }
    }
  }
}

std::string JoinVector(const std::vector<std::string> &items, const std::string &sep) {
  std::string out;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i != 0) {
      out += sep;
    }
    out += items[i];
  }
  return out;
}

size_t EditDistance(const std::string &a, const std::string &b) {
  std::vector<size_t> prev(b.size() + 1), curr(b.size() + 1);
  for (size_t j = 0; j <= b.size(); ++j) {
    prev[j] = j;
  }
  for (size_t i = 1; i <= a.size(); ++i) {
    curr[0] = i;
    for (size_t j = 1; j <= b.size(); ++j) {
      size_t substitute = prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
      curr[j] = std::min({curr[j - 1] + 1, prev[j] + 1, substitute});
    }
    std::swap(prev, curr);
  }
  return prev[b.size()];
}

// Up to three nearest pool entries for an unmatched config entry, "a (dist 1), b (dist 2)" style.
std::string NearestCandidates(const std::string &entry, const std::set<std::string> &pool) {
  std::vector<std::pair<size_t, std::string>> ranked;
  std::transform(pool.begin(), pool.end(), std::back_inserter(ranked),
                 [&entry](const std::string &item) { return std::make_pair(EditDistance(entry, item), item); });
  std::sort(ranked.begin(), ranked.end());
  std::string hint;
  size_t shown = 0;
  for (const auto &item : ranked) {
    if (shown == kMaxCandidateHints || item.second == entry) {
      break;
    }
    if (shown != 0) {
      hint += ", ";
    }
    hint += item.second + " (edit distance " + std::to_string(item.first) + ")";
    ++shown;
  }
  return hint.empty() ? "" : ", nearest candidates: " + hint;
}

}  // namespace

std::set<std::string> LayerDecisionRecorder::ConsumedEntries() const {
  std::set<std::string> consumed;
  for (const auto &item : decisions_) {
    if (!item.second.matched_config.empty()) {
      consumed.insert(item.second.matched_config);
    }
  }
  consumed.insert(consumed_entries_.begin(), consumed_entries_.end());
  return consumed;
}

std::set<std::string> LayerDecisionRecorder::SkippedNodeNames() const {
  std::set<std::string> skipped;
  for (const auto &item : decisions_) {
    if (!item.second.quantized &&
        (item.second.reason == QuantReportReason::kSkipNode || item.second.reason == QuantReportReason::kSkipOpType)) {
      skipped.insert(item.first);
    }
  }
  return skipped;
}

std::vector<std::string> LayerDecisionReporter::CsvHeader() {
  return {"OriginNames",   "OriginOpTypes",       "NodeName", "NodeType",  "OriginDataType", "QuantDecision", "Reason",
          "MatchedConfig", "MicroCoderAvailable", "Scale",    "ZeroPoint", "BitNum"};
}

std::string LayerDecisionReporter::EscapeCsvField(const std::string &field) {
  std::string value = field;
  // CSV formula injection: a model-controlled field starting with these characters would be
  // interpreted as a formula by spreadsheet software; neutralize with a leading apostrophe.
  if (!value.empty() && (value[0] == '=' || value[0] == '+' || value[0] == '-' || value[0] == '@')) {
    value = "'" + value;
  }
  if (value.find(',') != std::string::npos || value.find('"') != std::string::npos ||
      value.find('\n') != std::string::npos) {
    std::string escaped;
    for (char ch : value) {
      if (ch == '"') {
        escaped += "\"\"";
      } else {
        escaped += ch;
      }
    }
    value = "\"" + escaped + "\"";
  }
  return value;
}

std::string LayerDecisionReporter::BuildCsvContent(const std::vector<std::vector<std::string>> &rows) {
  std::ostringstream content;
  for (const auto &row : rows) {
    std::vector<std::string> cells;
    cells.reserve(kCsvColumnCount);
    for (size_t i = 0; i < kCsvColumnCount; ++i) {
      cells.push_back(i < row.size() ? EscapeCsvField(row[i]) : std::string());
    }
    content << JoinVector(cells, ",") << "\n";
  }
  return content.str();
}

std::string LayerDecisionReporter::InsertModelName(const std::string &path, const std::string &model_name) {
  if (model_name.empty() || path.empty() || path.back() == '/') {
    // No model to disambiguate (or the path is a directory — the caller already warned).
    return path;
  }
  auto clean = model_name;
  std::replace_if(clean.begin(), clean.end(), [](char ch) { return ch == '/' || ch == '\\'; }, '_');
  auto slash = path.find_last_of("/\\");
  std::string dir = (slash == std::string::npos) ? "" : path.substr(0, slash + 1);
  std::string file = (slash == std::string::npos) ? path : path.substr(slash + 1);
  auto dot = file.find_last_of('.');
  if (dot == std::string::npos || dot == 0) {
    return dir + file + "_" + clean;
  }
  return dir + file.substr(0, dot) + "_" + clean + file.substr(dot);
}

bool LayerDecisionReporter::ValidatePerLayerConfig(const ConverterPara &param,
                                                   const std::set<std::string> &origin_names,
                                                   const std::set<std::string> &origin_types) {
  const std::vector<std::pair<std::string, const std::set<std::string> *>> lists = {
    {"quant_node", &param.commonQuantParam.quant_node},
    {"skip_quant_node", &param.commonQuantParam.skip_quant_node},
    {"quant_op_type", &param.commonQuantParam.quant_op_type},
    {"skip_quant_op_type", &param.commonQuantParam.skip_quant_op_type}};
  bool all_matched = true;
  for (const auto &list : lists) {
    const bool by_name = list.first == "quant_node" || list.first == "skip_quant_node";
    const std::set<std::string> &pool = by_name ? origin_names : origin_types;
    for (const auto &entry : *list.second) {
      if (pool.find(entry) != pool.end()) {
        continue;
      }
      all_matched = false;
      MS_LOG(WARNING) << "per-layer quant config entry matches no node of the original model: " << list.first << "="
                      << entry << NearestCandidates(entry, pool);
    }
  }
  if (!all_matched && param.commonQuantParam.layer_decision_strict) {
    MS_LOG(ERROR) << "layer_decision_strict is enabled: failing the conversion on unmatched per-layer entries.";
    return false;
  }
  return true;
}

std::string LayerDecisionReporter::PredictCoderAvailable(const CNodePtr &cnode) const {
  if (param_ == nullptr || !param_->microParam.enable_micro) {
    return "N/A";
  }
  auto target_it = kMicroTargetMap.find(param_->microParam.target);
  if (target_it == kMicroTargetMap.end()) {
    MS_LOG(WARNING) << "layer decision report: unknown micro target " << param_->microParam.target;
    return "N/A";
  }
  // Mirror the coder selection on the EXPORTED .ms graph: the dtype codegen sees is the
  // post-quant tensor type. A QUANT_ALL node's activation inputs become int8 after export
  // (the ANF abstract stays fp32), so read the quant type FIRST; unquantized fp32 layers fold
  // into fp16 under weight_fp16. ConstantOfShape is registered by its OUTPUT dtype.
  TypeId dtype = kTypeUnknown;
  auto prim_for_quant = (cnode->size() > 0) ? GetValueNode<PrimitivePtr>(cnode->input(0)) : nullptr;
  bool quantized =
    prim_for_quant != nullptr && prim_for_quant->GetAttr(quant::kQuantType) != nullptr &&
    static_cast<quant::QuantType>(GetValue<int>(prim_for_quant->GetAttr(quant::kQuantType))) == quant::QUANT_ALL;
  if (quantized) {
    dtype = kNumberTypeInt8;  // full-quant activations are int8 on every target_device config
  } else {
    AnfNodePtr dtype_probe = (cnode->size() > 1) ? cnode->input(1) : nullptr;
    if (NodePrimitiveType(cnode) == "ConstantOfShape") {
      dtype_probe = cnode;
    }
    if (dtype_probe == nullptr || opt::GetDataTypeFromAnfNode(dtype_probe, &dtype) != RET_OK) {
      return "N/A";
    }
    if (param_->weight_fp16 && dtype == kNumberTypeFloat32) {
      dtype = kNumberTypeFloat16;
    }
  }
  auto prim_t = (cnode->size() > 0) ? GetPrimitiveT(cnode->input(0)) : nullptr;
  if (prim_t == nullptr) {
    return "N/A";
  }
  auto op_type = static_cast<int>(prim_t->value.type);
  std::string builtin_custom_type;
  if (op_type == schema::PrimitiveType_Custom) {
    auto custom_op = mindspore::ops::GetOperator<mindspore::ops::Custom>(cnode->input(0));
    if (custom_op != nullptr) {
      builtin_custom_type = custom_op->get_type();
    }
  }
  micro::CoderKey key(target_it->second, dtype, op_type, builtin_custom_type);
  auto creator = micro::OpCoderFactory::GetInstance()->FindOpCoder(key);
  return creator != nullptr ? "true" : "false";
}

namespace {
// Fills Scale/ZeroPoint/BitNum for a quantized layer: full quant stores per-tensor params as the
// quant_param attr (a ValueList of QuantizationParam); the legacy QuantParamHolder is the fallback
// for other quant paths.
void FillQuantParamSummary(const CNodePtr &cnode, int bit_num, ReportRow *row) {
  auto prim = (cnode->size() > 0) ? GetValueNode<PrimitivePtr>(cnode->input(0)) : nullptr;
  if (prim != nullptr) {
    auto quant_attr = std::dynamic_pointer_cast<mindspore::ValueList>(prim->GetAttr(quant::kQuantParam));
    if (quant_attr != nullptr && !quant_attr->value().empty()) {
      auto quantization = std::dynamic_pointer_cast<mindspore::QuantizationParam>(quant_attr->value().front());
      if (quantization != nullptr) {
        auto first_value = [&quantization](const std::string &key) -> ValuePtr {
          auto list = std::dynamic_pointer_cast<mindspore::ValueList>(quantization->GetAttr(key));
          if (list == nullptr || list->value().empty()) {
            return nullptr;
          }
          return list->value().front();
        };
        auto scale = first_value(quant::kScaleList);
        auto zero_point = first_value(quant::kZeroPointList);
        if (scale != nullptr && zero_point != nullptr) {
          try {
            row->scale = std::to_string(GetValue<double>(scale));
            row->zero_point = std::to_string(GetValue<int32_t>(zero_point));
          } catch (const std::exception &) {
            // Imm subtypes vary by write path; fall back to the canonical string form.
            row->scale = scale->ToString();
            row->zero_point = zero_point->ToString();
          }
          row->bit_num = std::to_string(bit_num);
          return;
        }
      }
    }
  }
  auto holder = GetCNodeQuantHolder(cnode);
  if (holder != nullptr) {
    const auto &input_params = holder->get_input_quant_params();
    if (!input_params.empty() && !input_params[0].empty()) {
      row->scale = std::to_string(input_params[0][0].scale);
      row->zero_point = std::to_string(input_params[0][0].zeroPoint);
      row->bit_num = std::to_string(bit_num);
    }
  }
}
}  // namespace

std::vector<std::string> LayerDecisionReporter::BuildLayerRow(const CNodePtr &cnode) const {
  const auto &recorder = LayerDecisionRecorder::GetInstance();
  auto node_name = cnode->fullname_with_scope();
  ReportRow row;

  row.origin_names = JoinVector(opt::GetSelfAndOriginNames(cnode), ";");
  row.origin_op_types = JoinVector(opt::GetOriginOpTypes(cnode), ";");
  row.node_name = node_name;
  row.node_type = NodePrimitiveType(cnode);

  TypeId origin_dtype = kTypeUnknown;
  if (opt::GetDataTypeFromAnfNode(cnode, &origin_dtype) == RET_OK) {
    row.origin_data_type = TypeIdLabel(origin_dtype);
  }

  LayerDecisionInfo info;
  bool recorded = recorder.Get(node_name, &info);
  row.quant_decision = (recorded && info.quantized) ? "QUANTIZED" : "NOT_QUANTIZED";
  // Every layer row carries a reason: nodes decided outside the per-layer marking (e.g. quant
  // boundary casts) report that instead of an empty Reason.
  row.reason = recorded ? info.reason : QuantReportReason::kOutsideMarkPass;
  row.matched_config = recorded ? info.matched_config : "";
  if (!recorded) {
    // Nodes decided outside the per-layer marking (e.g. quant boundary casts) still show up.
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim != nullptr && prim->GetAttr(quant::kQuantType) != nullptr &&
        static_cast<quant::QuantType>(GetValue<int>(prim->GetAttr(quant::kQuantType))) == quant::QUANT_ALL) {
      row.quant_decision = "QUANTIZED";
    }
  }

  row.micro_coder_available = PredictCoderAvailable(cnode);

  if (row.quant_decision == "QUANTIZED") {
    FillQuantParamSummary(cnode, param_->commonQuantParam.bit_num, &row);
  }
  return row.Cells();
}

void LayerDecisionReporter::AppendStatusRows(std::vector<std::vector<std::string>> *rows) const {
  if (param_ == nullptr || rows == nullptr) {
    return;
  }
  const auto &recorder = LayerDecisionRecorder::GetInstance();
  const auto consumed = recorder.ConsumedEntries();
  const std::vector<std::pair<std::string, const std::set<std::string> &>> lists = {
    {"quant_node", param_->commonQuantParam.quant_node},
    {"quant_op_type", param_->commonQuantParam.quant_op_type},
    {"skip_quant_node", param_->commonQuantParam.skip_quant_node},
    {"skip_quant_op_type", param_->commonQuantParam.skip_quant_op_type}};
  for (const auto &list : lists) {
    for (const auto &entry : list.second) {
      if (consumed.find(list.first + "=" + entry) != consumed.end()) {
        continue;  // hit a layer, shown in that layer's MatchedConfig; no status row
      }
      auto status =
        recorder.InPreOptSnapshot(entry) ? QuantReportReason::kFusedInvalid : QuantReportReason::kConfigUnmatched;
      // Status rows describe a config entry, not a layer: all layer columns stay empty, the config
      // string lives in MatchedConfig as "key=entry", the status lives in NodeType.
      ReportRow row;
      row.node_type = status;
      row.matched_config = list.first + "=" + entry;
      rows->push_back(row.Cells());
      MS_LOG(WARNING) << "layer decision report: config entry not effective: " << row.matched_config
                      << ", status: " << status;
    }
  }
}

int LayerDecisionReporter::WriteFile(const std::vector<std::vector<std::string>> &rows) const {
  auto path = param_->commonQuantParam.layer_decision_report;
  auto slash = param_->model_file.find_last_of("/\\");
  auto model_name = (slash == std::string::npos) ? param_->model_file : param_->model_file.substr(slash + 1);
  auto dot = model_name.find_last_of('.');
  if (dot != std::string::npos) {
    model_name = model_name.substr(0, dot);
  }
  auto final_path = InsertModelName(path, model_name);
  std::ofstream out(final_path, std::ios::trunc);
  if (!out.is_open()) {
    // The report is an auxiliary feature: a write failure warns and never aborts the conversion.
    MS_LOG(WARNING) << "layer decision report: cannot open output file " << final_path;
    return RET_OK;
  }
  out << BuildCsvContent(rows);
  out.close();
  if (out.fail()) {
    MS_LOG(WARNING) << "layer decision report: failed writing file " << final_path;
    return RET_OK;
  }
  MS_LOG(INFO) << "layer decision report written to " << final_path;
  return RET_OK;
}

int LayerDecisionReporter::Generate(const FuncGraphPtr &func_graph, bool include_status_rows) {
  if (func_graph == nullptr || param_ == nullptr) {
    return RET_ERROR;
  }
  std::vector<std::vector<std::string>> rows;
  rows.push_back(CsvHeader());
  std::vector<FuncGraphPtr> all_graphs;
  CollectAllGraphs(func_graph, &all_graphs);
  for (const auto &graph : all_graphs) {
    auto cnodes = graph->GetOrderedCnodes();
    std::transform(cnodes.begin(), cnodes.end(), std::back_inserter(rows),
                   [this](const CNodePtr &cnode) { return BuildLayerRow(cnode); });
  }
  // Status rows need the marking stage's consumed-entry accounting; dry-run (report-only, no
  // quantization) has no decisions, so every entry would false-fire and they are skipped.
  if (include_status_rows) {
    AppendStatusRows(&rows);
  }
  return WriteFile(rows);
}
}  // namespace mindspore::lite::quant
