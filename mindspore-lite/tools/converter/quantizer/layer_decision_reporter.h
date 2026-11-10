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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_LAYER_DECISION_REPORTER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_LAYER_DECISION_REPORTER_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "tools/converter/cxx_api/converter_para.h"

namespace mindspore::lite::quant {
// Reason codes of the layer decision report (why a layer was NOT quantized).
struct QuantReportReason {
  static constexpr const char *kWhitelistNotMatched = "WHITELIST_NOT_MATCHED";
  static constexpr const char *kSkipNode = "SKIP_NODE";
  static constexpr const char *kSkipOpType = "SKIP_OP_TYPE";
  static constexpr const char *kQuantAll = "QUANT_ALL";
  static constexpr const char *kNotFp32 = "NOT_FP32";
  static constexpr const char *kOpNotInSupportSet = "OP_NOT_IN_SUPPORT_SET";
  static constexpr const char *kActivationNotSupported = "ACTIVATION_NOT_SUPPORTED";
  static constexpr const char *kStructural = "STRUCTURAL";
  static constexpr const char *kAscendSpec = "ASCEND_SPEC";
  static constexpr const char *kFetchDtypeFailed = "FETCH_DTYPE_FAILED";
  static constexpr const char *kOutsideMarkPass = "OUTSIDE_MARK_PASS";
  static constexpr const char *kConfigUnmatched = "CONFIG_UNMATCHED";
  static constexpr const char *kFusedInvalid = "FUSED_INVALID";
};

// Per-node decision recorded while the quantizer marks nodes ("record while judging"); the report
// consumes these records instead of re-judging.
struct LayerDecisionInfo {
  bool quantized = false;
  std::string reason;          // quantized layers carry kQuantAll or the whitelist-reject reason
  std::string matched_config;  // "key=entry", e.g. skip_quant_node=BN-8; empty when nothing hit
};

// Converter-process-wide decision store. The converter is one-shot per process, a singleton is the
// lightest way to carry decisions from the marking stage to the report stage.
// Converter runs one FuncGraph at a time within a process; the recorder is a process-level
// singleton whose lifecycle is bounded by Clear() at each conversion entry (see
// QuantizationOptimizer::Run). Multi-threaded conversion is not a supported mode today;
// if it ever becomes one, promote this to a per-conversion instance passed from the
// quantizer (the API below is already instance-shaped for that migration).
class LayerDecisionRecorder {
 public:
  static LayerDecisionRecorder &GetInstance() {
    static LayerDecisionRecorder instance;
    return instance;
  }

  void Record(const std::string &node_name, const LayerDecisionInfo &info) { decisions_[node_name] = info; }

  // A layer may hit several list entries (name domain and type domain both match); all of them are
  // consumed so AppendStatusRows never reports a false CONFIG_UNMATCHED for an actually-hit entry.
  void RecordConsumed(const std::set<std::string> &entries) {
    consumed_entries_.insert(entries.begin(), entries.end());
  }

  bool Get(const std::string &node_name, LayerDecisionInfo *info) const {
    auto it = decisions_.find(node_name);
    if (it == decisions_.end() || info == nullptr) {
      return false;
    }
    *info = it->second;
    return true;
  }

  void SetPreOptSnapshot(const std::set<std::string> &names) { pre_opt_snapshot_ = names; }

  bool InPreOptSnapshot(const std::string &name) const {
    return pre_opt_snapshot_.find(name) != pre_opt_snapshot_.end();
  }

  // Entries of the per-layer config lists that actually hit some node.
  std::set<std::string> ConsumedEntries() const;

  // Converted (current) names of layers skipped by the per-layer blacklist — for consumers that
  // run after origin attributes are stripped (e.g. the bias-correction runtime callbacks).
  std::set<std::string> SkippedNodeNames() const;

  // Clears per-conversion decision records only: the pre-opt snapshot is (re)set by
  // AnfTransform::RunPass at each conversion's entry and must survive the quant entry Clear(),
  // otherwise FUSED_INVALID status rows can never be produced.
  void Clear() {
    decisions_.clear();
    consumed_entries_.clear();
  }

 private:
  LayerDecisionRecorder() = default;
  ~LayerDecisionRecorder() = default;
  std::map<std::string, LayerDecisionInfo> decisions_;
  std::set<std::string> consumed_entries_;
  std::set<std::string> pre_opt_snapshot_;
};

// Layer decision report: a 12-column CSV listing EVERY layer of the converted graph (whether it hit
// a list or not, quantized or not) plus status rows for config entries that did not take effect.
// Column order is frozen (see the feature document 2.4): the origin domain comes first.
class LayerDecisionReporter {
 public:
  explicit LayerDecisionReporter(const std::shared_ptr<ConverterPara> &param) : param_(param) {}
  ~LayerDecisionReporter() = default;

  int Generate(const FuncGraphPtr &func_graph, bool include_status_rows = true);

  // Static helpers, unit-test friendly.
  static std::string BuildCsvContent(const std::vector<std::vector<std::string>> &rows);
  static std::string EscapeCsvField(const std::string &field);
  static std::string InsertModelName(const std::string &path, const std::string &model_name);
  static std::vector<std::string> CsvHeader();

  // Pre-optimization validation of per-layer entries: name-domain entries must exist in the
  // original graph (the pre-opt snapshot), type-domain entries in its origin type records.
  // Unmatched entries are always logged with up to three nearest candidates; returns false only
  // in strict mode (layer_decision_strict) so the caller can fail the conversion.
  static bool ValidatePerLayerConfig(const ConverterPara &param, const std::set<std::string> &origin_names,
                                     const std::set<std::string> &origin_types);

  std::string PredictCoderAvailable(const CNodePtr &cnode) const;

 private:
  std::vector<std::string> BuildLayerRow(const CNodePtr &cnode) const;
  void AppendStatusRows(std::vector<std::vector<std::string>> *rows) const;
  int WriteFile(const std::vector<std::vector<std::string>> &rows) const;

  const std::shared_ptr<ConverterPara> param_{nullptr};
};
}  // namespace mindspore::lite::quant

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_LAYER_DECISION_REPORTER_H_
