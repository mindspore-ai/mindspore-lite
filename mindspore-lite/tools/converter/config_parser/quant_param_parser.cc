/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tools/converter/config_parser/quant_param_parser.h"
#include <tuple>
#include <vector>
#include "src/common/log_adapter.h"
#include "tools/common/string_util.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kQuantBitNumInt16 = 16;
constexpr int kQuantBitNumInt8 = 8;
constexpr int kMinSize = 0;
constexpr int kMaxSize = 65535;
constexpr size_t kMaxNodeListEntries = 1000;    // name lists: quant_node / skip_quant_node
constexpr size_t kMaxOpTypeListEntries = 200;   // op-type lists: quant_op_type / skip_quant_op_type
constexpr size_t kMaxEntryStringLength = 1024;  // single list entry length cap

// Split one list key, validate it, and insert the entries. The entry count is the raw
// comma-separated field count checked BEFORE set deduplication (a list of 1001 duplicates stays
// illegal even though it deduplicates to one entry) and every entry is length-capped; oversized
// lists are almost always paste/generation mistakes, reject them up front. Empty fields
// ("a,,b" style) stay tolerated as before — they are not inserted — but still count toward the
// field limit.
bool InsertSplitEntries(const std::string &key, const std::string &entries, size_t max_entries,
                        std::set<std::string> *target) {
  const auto items = SplitStringToVector(entries, ',');
  if (items.size() > max_entries) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: " << key << " accepts at most " << max_entries << " entries, got " << items.size()
                  << ".";
    return false;
  }
  // Validate every entry BEFORE touching the target: both failure paths (count over, entry too
  // long) then leave the target set untouched, so a rejected key never leaves a half-filled list
  // behind for callers that still inspect the parsed param.
  std::set<std::string> validated;
  for (const auto &item : items) {
    if (item.size() > kMaxEntryStringLength) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: " << key << " has an entry longer than " << kMaxEntryStringLength
                    << " characters.";
      return false;
    }
    if (!item.empty()) {  // "a,,b" style lists must not put an empty entry into the list
      validated.insert(item);
    }
  }
  target->insert(validated.begin(), validated.end());
  return true;
}

// Whitelist (quant_node/quant_op_type) and blacklist (skip_quant_node/skip_quant_op_type) are
// mutually exclusive: any entry on both sides is an invalid config, fail fast instead of partial.
bool ListsNotBothActive(const quant::CommonQuantParam &common_quant) {
  bool whitelist_active = !common_quant.quant_node.empty() || !common_quant.quant_op_type.empty();
  bool blacklist_active = !common_quant.skip_quant_node.empty() || !common_quant.skip_quant_op_type.empty();
  if (whitelist_active && blacklist_active) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: quant_node/quant_op_type (whitelist) and "
                     "skip_quant_node/skip_quant_op_type (blacklist) are mutually exclusive. "
                     "Please keep only one side.";
    return false;
  }
  return true;
}

// New list keys are scoped to the full-quant path this release (frozen by the design): the
// whitelist pair and the type blacklist key. Only the legacy name blacklist (skip_quant_node)
// keeps its three-path behavior. Pairing any new key with another quant_type would silently
// ignore the config, so reject it up front (same fail-fast contract as the mutual-exclusion
// check — one aligned strategy for every new key).
bool NewKeysRequireFullQuant(const quant::CommonQuantParam &common_quant) {
  if (common_quant.quant_type == quant::QUANT_ALL) {
    return true;
  }
  if (!common_quant.quant_node.empty() || !common_quant.quant_op_type.empty() ||
      !common_quant.skip_quant_op_type.empty()) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: quant_node/quant_op_type/skip_quant_op_type only take effect "
                     "with quant_type=FULL_QUANT.";
    return false;
  }
  return true;
}
}  // namespace
int QuantParamParser::ParseFilter(const CommonQuantString &common_quant_string, quant::CommonQuantParam *common_quant) {
  MS_ASSERT(common_quant != nullptr);
  if (!common_quant_string.min_quant_weight_size.empty()) {
    if (!ConvertIntNum(common_quant_string.min_quant_weight_size, &common_quant->min_quant_weight_size)) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: min_quant_weight_size should be a valid number.";
      return RET_INPUT_PARAM_INVALID;
    }
    if (common_quant->min_quant_weight_size < kMinSize || common_quant->min_quant_weight_size > kMaxSize) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: min_quant_weight_size should in [0,65535]." << std::endl;
      return RET_INPUT_PARAM_INVALID;
    }
  }
  if (!common_quant_string.min_quant_weight_channel.empty()) {
    if (!ConvertIntNum(common_quant_string.min_quant_weight_channel, &common_quant->min_quant_weight_channel)) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: min_quant_weight_channel should be a valid number.";
      return RET_INPUT_PARAM_INVALID;
    }

    if (common_quant->min_quant_weight_channel < kMinSize || common_quant->min_quant_weight_channel > kMaxSize) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: min_quant_weight_channel should be in the range [0,65535]." << std::endl;
      return RET_INPUT_PARAM_INVALID;
    }
  }
  // Table-driven so ParseFilter stays well under the lizard CCN ceiling; the per-key blocks
  // only differ in source string, key name, cap and target set.
  const std::vector<std::tuple<const std::string &, const char *, size_t, std::set<std::string> *>> list_keys = {
    {common_quant_string.skip_quant_node, "skip_quant_node", kMaxNodeListEntries, &common_quant->skip_quant_node},
    {common_quant_string.quant_node, "quant_node", kMaxNodeListEntries, &common_quant->quant_node},
    {common_quant_string.quant_op_type, "quant_op_type", kMaxOpTypeListEntries, &common_quant->quant_op_type},
    {common_quant_string.skip_quant_op_type, "skip_quant_op_type", kMaxOpTypeListEntries,
     &common_quant->skip_quant_op_type}};
  for (const auto &[entries, key, max_entries, target] : list_keys) {
    if (!entries.empty() && !InsertSplitEntries(key, entries, max_entries, target)) {
      return RET_INPUT_PARAM_INVALID;
    }
  }
  common_quant->layer_decision_report = common_quant_string.layer_decision_report;
  if (!common_quant_string.layer_decision_strict.empty() &&
      !ConvertBool(common_quant_string.layer_decision_strict, &common_quant->layer_decision_strict)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: layer_decision_strict should be true or false.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!ListsNotBothActive(*common_quant)) {
    return RET_INPUT_PARAM_INVALID;
  }
  if (!NewKeysRequireFullQuant(*common_quant)) {
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int QuantParamParser::ParseBitNum(const CommonQuantString &common_quant_string, quant::CommonQuantParam *common_quant) {
  MS_CHECK_TRUE_MSG(common_quant != nullptr, RET_ERROR, "common_quant is nullptr.");
  if (!common_quant_string.bit_num.empty() && !ConvertIntNum(common_quant_string.bit_num, &common_quant->bit_num)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: bit_num should be a valid number.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (common_quant->quant_type == quant::QUANT_WEIGHT) {
    if (common_quant->bit_num < 0 || common_quant->bit_num > kQuantBitNumInt16) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: bit_num should be [0,16].";
      return RET_INPUT_PARAM_INVALID;
    }
  } else if (common_quant->quant_type == quant::QUANT_ALL) {
    if (common_quant->bit_num != kQuantBitNumInt8) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: bit_num should be 8.";
      return RET_INPUT_PARAM_INVALID;
    }
  } else if (common_quant->quant_type == quant::QUANT_DYNAMIC) {
    if (common_quant->bit_num != kQuantBitNumInt8) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: bit_num should be 8.";
      return RET_INPUT_PARAM_INVALID;
    }
  }
  return RET_OK;
}

int QuantParamParser::ParseEnableEncode(const CommonQuantString &common_quant_string,
                                        quant::CommonQuantParam *common_quant) {
  MS_CHECK_TRUE_MSG(common_quant != nullptr, RET_ERROR, "common_quant is nullptr.");
  if (!common_quant_string.enable_encode.empty() &&
      !ConvertBool(common_quant_string.enable_encode, &common_quant->enable_encode)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: enable_encode should be true or false.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (common_quant->quant_type == quant::QUANT_WEIGHT &&
      (common_quant->bit_num != kQuantBitNumInt8 && common_quant->bit_num != kQuantBitNumInt16)) {
    if (!common_quant->enable_encode) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: enable_encode should be true when parameter bit_num belongs to [0,7] or [9,15].";
      return RET_INPUT_PARAM_INVALID;
    }
  }
  return RET_OK;
}

int QuantParamParser::ParseCommonQuant(const CommonQuantString &common_quant_string,
                                       quant::CommonQuantParam *common_quant) {
  MS_ASSERT(common_quant != nullptr);
  if (!common_quant_string.quant_type.empty()) {
    auto ret = ParseQuantType(common_quant_string.quant_type, &common_quant->quant_type);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Parse quant_type failed.";
      return ret;
    }
  }

  auto ret = ParseBitNum(common_quant_string, common_quant);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse bit num failed.";
    return ret;
  }

  ret = ParseEnableEncode(common_quant_string, common_quant);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse enable encode failed.";
    return ret;
  }

  ret = ParseFilter(common_quant_string, common_quant);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse filter failed.";
    return ret;
  }

  common_quant->debug_info_save_path = common_quant_string.debug_info_save_path;
  if (!common_quant->debug_info_save_path.empty()) {
    common_quant->is_debug = true;
  }

  // this is required only for model larger than 2G
  common_quant->workspace = common_quant_string.workspace;
  return RET_OK;
}

int QuantParamParser::ParseMixedBitWeightQuant(const MixedBitWeightQuantString &mixed_bit_weight_quant_string,
                                               quant::MixedBitWeightQuantParam *mixed_bit_weight_quant) {
  MS_CHECK_TRUE_MSG(mixed_bit_weight_quant != nullptr, RET_ERROR, "mixed_bit_weight_quant is nullptr.");
  if (!mixed_bit_weight_quant_string.init_scale.empty() &&
      !ConvertDoubleNum(mixed_bit_weight_quant_string.init_scale, &mixed_bit_weight_quant->init_scale)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: init_scale should be a valid number.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (mixed_bit_weight_quant->init_scale <= 0 || mixed_bit_weight_quant->init_scale >= 1) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: init_scale should at (0,1)";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!mixed_bit_weight_quant_string.use_cv_data.empty() &&
      !ConvertBool(mixed_bit_weight_quant_string.use_cv_data, &mixed_bit_weight_quant->use_cv_data)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: use_cv_data should be true or false.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!mixed_bit_weight_quant_string.max_iterations.empty()) {
    if (!ConvertIntNum(mixed_bit_weight_quant_string.max_iterations, &mixed_bit_weight_quant->max_iterations)) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: max_iterations should be a valid number.";
      return RET_INPUT_PARAM_INVALID;
    }
    if (mixed_bit_weight_quant->max_iterations < quant::kMinIterations ||
        mixed_bit_weight_quant->max_iterations > kMaxSize) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: max_iterations should be in the range [40,65535]." << std::endl;
      return RET_INPUT_PARAM_INVALID;
    }
  }

  if (!mixed_bit_weight_quant_string.auto_tune.empty() &&
      !ConvertBool(mixed_bit_weight_quant_string.auto_tune, &mixed_bit_weight_quant->auto_tune)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: auto_tune should be true or false.";
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int QuantParamParser::ParseFullQuant(const FullQuantString &full_quant_string, quant::FullQuantParam *full_quant) {
  MS_CHECK_TRUE_MSG(full_quant != nullptr, RET_ERROR, "full_quant is nullptr.");
  if (!full_quant_string.activation_quant_method.empty() &&
      ParseActivationQuantizedMethod(full_quant_string.activation_quant_method, &full_quant->activation_quant_method) !=
        RET_OK) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: Parse activation_quant_method failed.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!full_quant_string.bias_correction.empty() &&
      !ConvertBool(full_quant_string.bias_correction, &full_quant->bias_correction)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: bias_correction should be true or false.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!full_quant_string.target_device.empty()) {
    auto ret = ParseTargetDevice(full_quant_string.target_device, &full_quant->target_device);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Parse device failed.";
      return ret;
    }
  }
  if (!full_quant_string.per_channel.empty() && !ConvertBool(full_quant_string.per_channel, &full_quant->per_channel)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: per_channel should be true or false.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!full_quant_string.smooth_alpha.empty() &&
      !ConvertDoubleNum(full_quant_string.smooth_alpha, &full_quant->smooth_alpha)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: smooth_alpha should be a valid number.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!full_quant_string.enable_smooth_shift.empty() &&
      !ConvertBool(full_quant_string.enable_smooth_shift, &full_quant->enable_smooth_shift)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: enable_smooth_shift should be true or false.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!full_quant_string.enable_all_ops.empty() &&
      !ConvertBool(full_quant_string.enable_all_ops, &full_quant->enable_all_ops)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: enable_all_ops should be true or false.";
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int QuantParamParser::ParseQuantType(const std::string &quant_type_str, quant::QuantType *quant_type) {
  if (quant_type_str == "WEIGHT_QUANT") {
    (*quant_type) = quant::QUANT_WEIGHT;
    return RET_OK;
  } else if (quant_type_str == "FULL_QUANT") {
    (*quant_type) = quant::QUANT_ALL;
    return RET_OK;
  } else if (quant_type_str == "DYNAMIC_QUANT") {
    (*quant_type) = quant::QUANT_DYNAMIC;
    return RET_OK;
  } else if (quant_type_str.empty()) {
    (*quant_type) = quant::QUANT_NONE;
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "INPUT ILLEGAL: quant_type must be WEIGHT_QUANT|FULL_QUANT|DYNAMIC_QUANT.";
    return RET_INPUT_PARAM_INVALID;
  }
}

int QuantParamParser::ParseTargetDevice(const std::string &target_device_str, quant::TargetDevice *target_device) {
  MS_CHECK_TRUE_MSG(target_device != nullptr, RET_ERROR, "target_device is nullptr.");
  if (target_device_str == "KIRIN") {
    (*target_device) = quant::KIRIN;
    return RET_OK;
  } else if (target_device_str == "NVGPU") {
    (*target_device) = quant::NVGPU;
    return RET_OK;
  } else if (target_device_str == "DSP") {
    (*target_device) = quant::DSP;
    return RET_OK;
  } else if (target_device_str == "ASCEND") {
    (*target_device) = quant::ASCEND;
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "INPUT ILLEGAL: target_device must be KIRIN|NVGPU|DSP|ASCEND.";
    return RET_INPUT_PARAM_INVALID;
  }
}

int QuantParamParser::ParseActivationQuantizedMethod(const std::string &activation_quant_method_str,
                                                     quant::ActivationQuantizedMethod *activation_quant_method) {
  MS_CHECK_TRUE_MSG(activation_quant_method != nullptr, RET_ERROR, "activation_quant_method is nullptr.");
  if (activation_quant_method_str == "MAX_MIN") {
    (*activation_quant_method) = quant::MAX_MIN;
    return RET_OK;
  } else if (activation_quant_method_str == "KL") {
    (*activation_quant_method) = quant::KL;
    return RET_OK;
  } else if (activation_quant_method_str == "REMOVAL_OUTLIER") {
    (*activation_quant_method) = quant::REMOVAL_OUTLIER;
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "INPUT ILLEGAL: activation_quant_method must be MAX_MIN|KL|REMOVAL_OUTLIER.";
    return RET_INPUT_PARAM_INVALID;
  }
}

int QuantParamParser::ParseWeightQuant(const WeightQuantString &weight_quant_string,
                                       quant::WeightQuantParam *weight_quant) {
  MS_CHECK_TRUE_MSG(weight_quant != nullptr, RET_ERROR, "weight_quant is nullptr.");
  if (!weight_quant_string.dequant_strategy.empty()) {
    if (weight_quant_string.dequant_strategy == "ON_THE_FLY") {
      weight_quant->dequant_strategy = quant::ON_THE_FLY;
    } else {
      MS_LOG(ERROR) << "INPUT ILLEGAL: dequant_strategy must be ON_THE_FLY.";
      return RET_INPUT_PARAM_INVALID;
    }
  }
  if (!weight_quant_string.quant_strategy.empty()) {
    if (weight_quant_string.quant_strategy == "GPTQ") {
      weight_quant->quant_strategy = quant::GPTQ_ALGORITHM;
    } else if (weight_quant_string.quant_strategy == "MAX_MIN") {
      weight_quant->quant_strategy = quant::MAX_MIN_ALGORITHM;
    } else {
      MS_LOG(ERROR) << "INPUT ILLEGAL: quant_strategy must be GPTQ or MAX_MIN.";
      return RET_INPUT_PARAM_INVALID;
    }
  }
  if (!weight_quant_string.update_mindir.empty() &&
      !ConvertBool(weight_quant_string.update_mindir, &weight_quant->update_mindir)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: update_mindir should be true or false.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!weight_quant_string.max_segments.empty() &&
      !ConvertIntNum(weight_quant_string.max_segments, &weight_quant->max_segments)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: decode_threads should be a number.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!weight_quant_string.per_channel.empty() &&
      !ConvertBool(weight_quant_string.per_channel, &weight_quant->per_channel)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: per_channel should be true or false.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!weight_quant_string.bias_correction.empty() &&
      !ConvertBool(weight_quant_string.bias_correction, &weight_quant->bias_correction)) {
    MS_LOG(ERROR) << "INPUT ILLEGAL: bias_correction should be true or false.";
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int QuantParamParser::ParseExportPrecisionMode(const std::string &precision_modeL_str,
                                               quant::PrecisionMode *precision_mode) {
  if (precision_modeL_str == "QUANT") {
    (*precision_mode) = quant::QUANT;
    return RET_OK;
  } else if (precision_modeL_str == "FLOAT32") {
    (*precision_mode) = quant::FLOAT32;
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "INPUT ILLEGAL: export_precision_mode must be QUANT or FLOAT32.";
    return RET_INPUT_PARAM_INVALID;
  }
}

int QuantParamParser::ParseTransformQuant(const TransformQuantString &transform_quant_string,
                                          quant::TransformQuantParam *transform_quant) {
  MS_CHECK_TRUE_MSG(transform_quant != nullptr, RET_ERROR, "transform_quant is nullptr.");
  if (!transform_quant_string.export_precision_mode.empty()) {
    auto ret = ParseExportPrecisionMode(transform_quant_string.export_precision_mode, &transform_quant->precision_mode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Parse precision mode failed.";
      return ret;
    }
  }
  return RET_OK;
}

int QuantParamParser::ParseDynamicQuant(const DynamicQuantString &dynamic_quant_string,
                                        quant::DynamicQuantParam *dynamic_quant) {
  MS_CHECK_TRUE_MSG(dynamic_quant != nullptr, RET_ERROR, "dynamic_quant is nullptr.");
  if (!dynamic_quant_string.quant_strategy.empty()) {
    auto ret = ParseDynamicQuantStrategy(dynamic_quant_string.quant_strategy, &dynamic_quant->quant_strategy);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Parse dynamic quant strategy failed.";
      return ret;
    }
  }
  return RET_OK;
}

int QuantParamParser::ParseDynamicQuantStrategy(const std::string &dynamic_quant_strategy_str,
                                                quant::DynamicQuantStrategy *dynamic_strategy) {
  if (dynamic_quant_strategy_str == "ALWC") {
    (*dynamic_strategy) = quant::ACTIVATION_LAYER_WEIGHT_CHANNEL;
  } else if (dynamic_quant_strategy_str == "ACWL") {
    (*dynamic_strategy) = quant::ACTIVATION_CHANNEL_WEIGHT_LAYER;
  } else {
    MS_LOG(ERROR) << "INPUT ILLEGAL: dynamic_quant_strategy must be ALWC or ACWL.";
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
