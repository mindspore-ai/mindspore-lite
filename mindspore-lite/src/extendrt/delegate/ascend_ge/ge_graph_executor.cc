/**
 * Copyright 2022-2026 Huawei Technologies Co., Ltd
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

#include "extendrt/delegate/ascend_ge/ge_graph_executor.h"

#include <algorithm>
#include <fstream>
#include <future>
#include <utility>
#include <nlohmann/json.hpp>
#include "extendrt/delegate/factory.h"
#include "include/utils/scoped_long_running.h"
#include "src/common/common.h"
#include "src/common/file_utils.h"
#include "cxx_api/acl_utils.h"
#include "tools/common/graph_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/graph/remove_load_pass.h"
#include "src/extendrt/utils/func_graph_utils.h"
#include "external/ge_common/ge_api_error_codes.h"
#include "src/extendrt/delegate/ascend_ge/aoe_api_tune_process.h"
#include "extendrt/delegate/ascend_ge/ge_utils.h"
#include "extendrt/delegate/ascend_ge/ge_dynamic_utils.h"
#include "tools/common/string_util.h"
#include "src/extendrt/cxx_api/file_utils.h"
#include "mindspore/ops/infer/custom.h"
#include "tools/common/custom_ascend_utils.h"
#include "op_proto/inc/array_ops.h"
#include "op_proto/inc/elewise_calculation_ops.h"
#include "tools/optimizer/graph/attr_to_args_pass.h"
#include "utils/ms_utils_secure.h"
#include "src/extendrt/utils/tensor_default_impl.h"
#include "mindspore/core/include/utils/misc.h"
namespace mindspore {
namespace {
constexpr auto kProviderGe = "ge";
constexpr auto kDump = "dump";
constexpr auto kDumpMode = "dump_mode";
constexpr auto kProfiling = "profiler";
constexpr auto kRefModeNone = "none";
constexpr auto kRefModeVariable = "variable";
constexpr auto kRefModeAll = "all";
constexpr auto kIsAdapted = "is_adapted";
constexpr size_t kAlignRefData = 32;
constexpr size_t kErrorSize = 0;
std::mutex g_compile_graph_mutex;
size_t ALIGN_UP_REF_DATA(size_t size) {
  return ((size + kMemAlignSize + kAlignRefData - 1) / kMemAlignSize) * kMemAlignSize;
}

backend::ge_backend::DfGraphPtr GenExampleGraph(const std::string &name) {
  MS_LOG(INFO) << "Gen fake graph name is " << name;
  auto graph = std::make_shared<backend::ge_backend::DfGraph>(name);
  auto shape_data = std::vector<int64_t>({1, 1, 1, 1});
  backend::ge_backend::GeTensorDesc desc_data(ge::Shape(shape_data), ge::FORMAT_ND, ge::DT_FLOAT16);
  auto data = ge::op::Data("data");
  data.set_attr_index(0);
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);
  auto add = ge::op::Add("add").set_input_x1(data).set_input_x2(data);
  std::vector<backend::ge_backend::Operator> inputs{data};
  std::vector<backend::ge_backend::Operator> outputs{add};
  graph->SetInputs(inputs);
  graph->SetOutputs(outputs);
  return graph;
}

bool UpdateOmCacheIdxFile(const std::string &idx_file_name) {
  std::ifstream ifs(idx_file_name);
  if (!ifs.good() || !ifs.is_open()) {
    MS_LOG(INFO) << "model cache idx json not exists, idx file: " << idx_file_name << ", skip create small ge graph";
    return false;
  }
  nlohmann::json dump_cfg_json;
  try {
    dump_cfg_json = nlohmann::json::parse(ifs);
    const std::string cache_file_list = "cache_file_list";
    const std::string var_desc_file_name = "var_desc_file_name";
    auto &cache_file_config = dump_cfg_json[cache_file_list];
    if (cache_file_config == nullptr || cache_file_config[0] == nullptr) {
      MS_LOG(WARNING) << "model cache idx json content invalid, idx file: " << idx_file_name
                      << ", skip create small ge graph";
      return false;
    }
    auto &config = cache_file_config[0];
    if (config[var_desc_file_name] != nullptr) {
      config.erase(var_desc_file_name);
      auto new_json_str = dump_cfg_json.dump(4);
      ifs.close();
      std::ofstream ofs(idx_file_name, std::ios::out);
      if (!ofs.is_open()) {
        MS_LOG(WARNING) << "Failed to open model cache idx file for write, idx file: " << idx_file_name
                        << ", skip create small ge graph";
        return false;
      }
      ofs << new_json_str;
      ofs.close();
#ifndef _MSC_VER
      chmod(idx_file_name.c_str(), S_IRUSR);
#endif
      MS_LOG(INFO) << "Erase option " << var_desc_file_name;
    }
    return true;
  } catch (const std::exception &error) {
    MS_LOG(WARNING) << "parse model cache idx json failed, idx file: " << idx_file_name
                    << ", skip create small ge graph";
    return false;
  }
}
}  // namespace

std::atomic_uint32_t GeGraphExecutor::global_graph_idx_ = 0;
uint32_t GeGraphExecutor::GetNextGraphIdx() { return global_graph_idx_++; }

GeGraphExecutor::~GeGraphExecutor() {
  if (ge_session_) {
    for (auto graph_id : compute_graph_id_list_) {
      ge_session_->RemoveGraph(graph_id);
      auto session_context = GeSessionManager::GetGeSessionContext(session_id_);
      if (session_context != nullptr) {
        (void)session_context->feature_graph_ids.erase(graph_id);
      }
    }
    ge_session_ = nullptr;
    GeSessionManager::TryReleaseGeSessionContext(session_id_);
  }
}

bool GeGraphExecutor::SetGeTensorShape(GeTensor *ge_tensor, ShapeVector shape) {
  MS_CHECK_TRUE_RET(ge_tensor != nullptr, RET_ERROR);
  auto ge_desc = ge_tensor->GetTensorDesc();
  ge::Shape new_ge_shape(shape);
  ge_desc.Update(new_ge_shape);
  ge_desc.SetOriginShape(new_ge_shape);
  ge_tensor->SetTensorDesc(ge_desc);
  MS_LOG(INFO) << "In SetGeTensorShape update ge shape to :" << shape;
  return true;
}

bool GeGraphExecutor::InitInputDeviceTensor(const FuncGraphPtr &anf_graph) {
  MS_LOG(INFO) << "Call InitInputDeviceTensor start.";
  auto inputs = anf_graph->get_inputs();
  inputs_buffer_infos_.resize(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    auto &input_info = inputs_buffer_infos_[i];
    auto shape = FuncGraphUtils::GetTensorShape({inputs[i], 0});

    /* set max_batch size and max_seq_len for dyn shape */
    std::vector<int64_t> new_shape;
    for (size_t j = 0; j < shape.size(); j++) {
      if (shape[j] == abstract::Shape::kShapeDimAny) {
        new_shape.push_back(dyn_kv_cache_info_.max_seq_len_size);
      } else {
        new_shape.push_back(shape[j]);
      }
    }

    MS_LOG(INFO) << "Init input_" << i << " buffer for ge, change shape: " << shape << " -> " << new_shape;
    auto dtype = static_cast<TypeId>(FuncGraphUtils::GetTensorDataType({inputs[i], 0}));
    if (!InitInOutDeviceBuffer("Input " + std::to_string(i), new_shape, dtype, &input_info)) {
      return false;
    }
  }
  return true;
}

bool GeGraphExecutor::InitOutputDeviceTensor(const FuncGraphPtr &anf_graph, uint32_t graph_id) {
  MS_LOG(INFO) << "Call GE GetCompiledGraphSummary start, graph id " << graph_id;
  auto graph_summary = ge_session_->GetCompiledGraphSummary(graph_id);
  if (graph_summary == nullptr) {
    MS_LOG(ERROR) << "Failed to call GE GetCompiledGraphSummary, graph id " << graph_id
                  << ", error: " << ge::GEGetErrorMsg();
    return false;
  }
  MS_LOG(INFO) << "Call GE GetCompiledGraphSummary end, graph id " << graph_id;
  dyn_kv_cache_info_.is_ge_graph_static_ = graph_summary->IsStatic();
  MS_LOG(INFO) << "GE graph is static :" << dyn_kv_cache_info_.is_ge_graph_static_ << ", graph id: " << graph_id;
  std::vector<AnfWithOutIndex> outputs;
  if (!FuncGraphUtils::GetFuncGraphOutputs(anf_graph, &outputs)) {
    MS_LOG(ERROR) << "Failed to get func graph outputs";
    return false;
  }
  outputs_buffer_infos_.resize(outputs.size());
  if (dyn_kv_cache_info_.is_ge_graph_static_) {
    std::vector<::ge::Shape> ge_shapes;
    auto ge_status = graph_summary->GetOutputShapes(ge_shapes);
    if (ge_status != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call GetOutputShapes, status: " << ge_status;
      return false;
    }
    if (outputs.size() != ge_shapes.size()) {
      MS_LOG(ERROR) << "Output count got from graph " << outputs.size() << " != that " << ge_shapes.size()
                    << " got from GE";
      return false;
    }
    for (size_t i = 0; i < outputs.size(); i++) {
      auto &output_info = outputs_buffer_infos_[i];
      auto shape = ge_shapes[i].GetDims();
      auto dtype = static_cast<TypeId>(FuncGraphUtils::GetTensorDataType(outputs[i]));
      if (!InitInOutDeviceBuffer("Output " + std::to_string(i), shape, dtype, &output_info)) {
        return false;
      }
    }
  }
  return true;
}

void GeGraphExecutor::SetRefShape(std::vector<int64_t> *ref_shape, bool dyn, std::string tensor_name) {
  MS_CHECK_TRUE_RET_VOID(ref_shape != nullptr);
  if (!dyn_kv_cache_info_.dynamic_kv_cache) {
    return;
  }
  size_t b_index = kDim0;
  size_t s_index = kDim2;
  if (dyn_kv_cache_info_.kv_cache_layout == lite::kKVCacheLayoutBSH) {
    s_index = kDim1;
  }
  if (dyn) {
    if (dyn_kv_cache_info_.batch_size_dyn) {
      (*ref_shape)[b_index] = abstract::Shape::kShapeDimAny;
      MS_LOG(INFO) << "for " << tensor_name << " update batch size to dyn(-1) for ge_option.";
    }
    if (dyn_kv_cache_info_.seq_length_dyn) {
      (*ref_shape)[s_index] = abstract::Shape::kShapeDimAny;
      MS_LOG(INFO) << "for " << tensor_name << " update seq length size to dyn(-1) for ge_option.";
    }
  } else {
    if (dyn_kv_cache_info_.batch_size_dyn) {
      (*ref_shape)[b_index] = dyn_kv_cache_info_.real_batch_size;
      MS_LOG(INFO) << "for " << tensor_name << " update batch size to " << dyn_kv_cache_info_.real_batch_size
                   << " for ge_option.";
    }
    if (dyn_kv_cache_info_.seq_length_dyn) {
      (*ref_shape)[s_index] = dyn_kv_cache_info_.real_seq_len_size;
      MS_LOG(INFO) << "for " << tensor_name << " update seq length size to " << dyn_kv_cache_info_.real_seq_len_size
                   << " for ge_option.";
    }
  }
}

void GeGraphExecutor::UpdateOutputShapeInfo(std::vector<::ge::Tensor> *ge_outputs) {
  MS_CHECK_TRUE_RET_VOID(ge_outputs != nullptr);
  MS_CHECK_TRUE_RET_VOID(ge_outputs->size() >= outputs_buffer_infos_.size());
  MS_LOG(INFO) << "Update output dtype and shape.";
  for (size_t i = 0; i < outputs_buffer_infos_.size(); i++) {
    auto &output_info = outputs_buffer_infos_[i];
    auto &ge_output = ge_outputs->at(i);
    auto ge_tensor_desc = ge_output.GetTensorDesc();
    output_info.shape = device::ascend::TransformUtil::ConvertGeShape(ge_tensor_desc.GetShape());
    output_info.dtype = device::ascend::TransformUtil::ConvertGeDataType(ge_tensor_desc.GetDataType());
    output_info.max_size = SizeOf(output_info.shape) * GetDataTypeSize(output_info.dtype);
    auto out_device = ge_output.GetData();
    if (dyn_kv_cache_info_.is_ge_graph_static_ && out_device != output_info.device_addr) {
      MS_LOG(WARNING) << "GE output device address not equal malloc device memory when graph is static";
    }
    output_info.device_addr = out_device;
    MS_LOG(INFO) << "Update output_" << i << " dtype: " << output_info.dtype << ", shape: " << output_info.shape;
  }
  return;
}

bool GeGraphExecutor::CheckRefDataInfo() {
  if (!dyn_kv_cache_info_.dynamic_kv_cache) {
    return true;
  }
  auto &ref_shape = ref_data_infos_.front().shape;
  for (size_t i = 0; i < ref_data_infos_.size(); i++) {
    auto &ref_data_info = ref_data_infos_[i];
    auto &para_name = ref_data_info.name;
    if (dyn_kv_cache_info_.kv_cache_layout == lite::kKVCacheLayoutBSH) {
      if (ref_data_info.shape.size() != kShape3dDims) {
        MS_LOG(ERROR) << "KVCache shape size is not " << kShape3dDims << ", while KVCache layout is "
                      << dyn_kv_cache_info_.kv_cache_layout << ", KVCache param " << para_name << ", shape "
                      << ref_data_info.shape;
        return false;
      }
    } else if (dyn_kv_cache_info_.kv_cache_layout == lite::kKVCacheLayoutBNSD) {
      if (ref_data_info.shape.size() != kShape4dDims) {
        MS_LOG(ERROR) << "KVCache shape size is not " << kShape4dDims << ", while KVCache layout is "
                      << dyn_kv_cache_info_.kv_cache_layout << ", KVCache param " << para_name << ", shape "
                      << ref_data_info.shape;
        return false;
      }
    } else {
      MS_LOG(ERROR) << "Unsupported KVCache layout " << dyn_kv_cache_info_.kv_cache_layout;
      return false;
    }
    if (ref_shape != ref_data_info.shape) {
      MS_LOG(ERROR) << "KVCache shape " << ref_data_info.shape << " of " << para_name << " != KVCache shape "
                    << ref_shape << " of " << ref_data_infos_.front().name;
      return false;
    }
  }
  return true;
}

bool GeGraphExecutor::InitMaxShapeParam() {
  if (ref_data_infos_.empty()) {
    MS_LOG(INFO) << "RefData count is empty";
    return true;
  }
  if (!CheckRefDataInfo()) {
    return false;
  }
  auto &ref_shape = ref_data_infos_.front().shape;
  size_t b_index = kDim0;
  size_t s_index = kDim2;
  if (ref_shape.size() == kShape3dDims) {  // BSH
    s_index = kDim1;
  } else if (ref_shape.size() == kShape4dDims) {  // BNSD
    s_index = kDim2;
  } else {
    MS_LOG(WARNING) << "RefData dim count is unexpected, shape " << ref_shape << ", name "
                    << ref_data_infos_.front().name;
    return true;
  }
  std::string max_batch_size;
  if (GetConfigOption("ascend_context", "max_batch_size", &max_batch_size)) {
    MS_LOG(INFO) << "Get max batch size from config file, ascend_context, max_batch_size";
    dyn_kv_cache_info_.max_batch_size = std::stoi(max_batch_size);
  } else {
    MS_LOG(INFO) << "Get max batch size from ref data shape : " << ref_shape;
    dyn_kv_cache_info_.max_batch_size = ref_shape[b_index];
  }

  std::string max_seq_length;
  if (GetConfigOption("ascend_context", "max_seq_length", &max_seq_length)) {
    MS_LOG(INFO) << "Get max seq length from config file, ascend_context, max_seq_length";
    dyn_kv_cache_info_.max_seq_len_size = std::stoi(max_seq_length);
  } else {
    MS_LOG(INFO) << "Get max seq length from ref data shape : " << ref_shape;
    dyn_kv_cache_info_.max_seq_len_size = ref_shape[s_index];
  }

  MS_LOG(INFO) << "set dynamic max shape, max batch size : " << dyn_kv_cache_info_.max_batch_size
               << ", max seq length: " << dyn_kv_cache_info_.max_seq_len_size;
  return true;
}

bool GeGraphExecutor::InitRealShapeParam(const std::vector<mindspore::MSTensor> &inputs) {
  if (!dyn_kv_cache_info_.dynamic_kv_cache) {
    return true;
  }
  auto input_0_shape = inputs[0].Shape();
  if (input_0_shape.size() != kShape2dDims) {
    MS_LOG(ERROR) << "Expected input 0 shape to be [bs, seq_length], but got " << input_0_shape;
    return false;
  }
  dyn_kv_cache_info_.real_batch_size = input_0_shape.at(Index0);
  MS_LOG(INFO) << "Real batch size : " << dyn_kv_cache_info_.real_batch_size;
  dyn_kv_cache_info_.real_seq_len_size = input_0_shape.at(Index1);
  MS_LOG(INFO) << "Real seq length size : " << dyn_kv_cache_info_.real_seq_len_size;
  return true;
}

bool GeGraphExecutor::GetConfigOption(const std::string &section_name, const std::string &option_name,
                                      std::string *option_val) {
  if (option_val == nullptr) {
    MS_LOG(ERROR) << "Input argument option_val is nullptr";
    return false;
  }
  auto config_it = config_infos_.find(section_name);
  if (config_it == config_infos_.end()) {
    return false;
  }
  auto &options = config_it->second;
  auto option_it = options.find(option_name);
  if (option_it == options.end()) {
    return false;
  }
  *option_val = option_it->second;
  return true;
}

uint32_t GeGraphExecutor::GetRankID() const {
  auto ascend_info = GeUtils::GetAscendDeviceInfo(context_);
  if (ascend_info == nullptr) {
    MS_LOG(ERROR) << "Can not find ascend device context.";
    return 0;
  }
  return ascend_info->GetRankID();
}

uint32_t GeGraphExecutor::GetDeviceID() const {
  auto ascend_info = GeUtils::GetAscendDeviceInfo(context_);
  if (ascend_info == nullptr) {
    MS_LOG(ERROR) << "Can not find ascend device context.";
    return 0;
  }
  return ascend_info->GetDeviceID();
}

bool GeGraphExecutor::Init() {
  ge_global_context_ = GeDeviceContext::InitGlobalContext(context_, config_infos_);
  if (ge_global_context_ == nullptr) {
    MS_LOG(ERROR) << "Failed to Init global context";
    return false;
  }
  if (!InitRefModeConfig()) {
    return false;
  }
  std::string model_cache_mode;
  (void)GetConfigOption(lite::kAscendContextSection, lite::kModelCacheMode, &model_cache_mode);
  if (!model_cache_mode.empty()) {
    cache_mode_ = model_cache_mode;
    MS_LOG(INFO) << "Set set model cache mode " << model_cache_mode;
  }
  return true;
}

bool GeGraphExecutor::InitRefModeConfig() {
  std::string ref_mode;
  (void)GetConfigOption(lite::kAscendContextSection, lite::kParameterAsRefData, &ref_mode);
  if (!ref_mode.empty()) {
    ref_mode = lite::StringTolower(ref_mode);
    if (ref_mode != kRefModeNone && ref_mode != kRefModeVariable && ref_mode != kRefModeAll) {
      MS_LOG(ERROR) << "Only " << kRefModeNone << ", " << kRefModeVariable << " or " << kRefModeAll
                    << " is supported for " << lite::kParameterAsRefData;
      return false;
    }
    if (ref_mode == kRefModeAll) {
      ref_mode_flag_ = backend::ge_backend::RefModeFlag::kRefModeAll;
    } else if (ref_mode == kRefModeVariable) {
      ref_mode_flag_ = backend::ge_backend::RefModeFlag::kRefModeVariable;
    } else {
      ref_mode_flag_ = backend::ge_backend::RefModeFlag::kRefModeNone;
    }
    MS_LOG(INFO) << "Set parameter ref mode ok.";
  } else {
    ref_mode_flag_ = backend::ge_backend::RefModeFlag::kRefModeNone;
  }
  return true;
}

void GeGraphExecutor::GetGeSessionOptions(std::map<std::string, std::string> *ge_options_ptr) {
  MS_EXCEPTION_IF_NULL(ge_options_ptr);
  auto &ge_options = *ge_options_ptr;
  ge_options["ge.trainFlag"] = "0";
  ge_options["ge.enablePrintOpPass"] = "0";
  ge_options["ge.exec.device_id"] = std::to_string(GetDeviceID());
  ge_options["ge.exec.staticMemoryPolicy"] = "2";
  if (ref_mode_flag_ != backend::ge_backend::RefModeFlag::kRefModeNone) {
    ge_options["ge.constLifecycle"] = "graph";
  }
  auto config_it = config_infos_.find(lite::kGeSessionOptionsSection);
  if (config_it != config_infos_.end()) {
    for (auto &item : config_it->second) {
      ge_options[item.first] = item.second;
    }
  }
  config_it = config_infos_.find(lite::kAscendContextSection);
  if (config_it != config_infos_.end()) {
    GetGeSessionOptionsFromAscendContext(config_it->second, ge_options_ptr);
  }
}

bool GeGraphExecutor::SetModelCacheDir(std::map<std::string, std::string> *session_options_ptr) {
  MS_CHECK_TRUE_RET(session_options_ptr != nullptr, false);
  auto &ge_options = *session_options_ptr;
  auto build_cache_dir = "model_build_cache_" + std::to_string(GetRankID());
  if (lite::CreateDir(build_cache_dir) != RET_OK) {
    MS_LOG(ERROR) << "Failed to create build cache dir " << build_cache_dir;
    return false;
  }
  ge_options[lite::kGeGraphCompilerCacheDir] = build_cache_dir;
  MS_LOG(INFO) << "Update session attr " << lite::kGeGraphCompilerCacheDir << " to " << build_cache_dir;
  return true;
}

bool GeGraphExecutor::SetOfflineBuildModelCacheDir(std::map<std::string, std::string> *session_options_ptr) {
  MS_CHECK_TRUE_RET(session_options_ptr != nullptr, false);
  std::string build_cache_dir;
  auto &ge_options = *session_options_ptr;
  bool build_cache_enabled = false;
  std::string output_file;
  (void)GetConfigOption(lite::kConverterParams, lite::kConverterOutputFile, &output_file);
  std::string output_dir = "./";
  if (output_file.find("/") != std::string::npos) {
    output_dir = output_file.substr(0, output_file.rfind("/") + 1);
  }
  session_id_ = GetSessionId();
  auto ge_session_context = GeSessionManager::GetGeSessionContext(session_id_);
  if (ge_session_context) {
    const auto &last_ge_options = ge_session_context->session_options;
    if (auto it = last_ge_options.find(lite::kGeGraphCompilerCacheDir); it != last_ge_options.end()) {
      build_cache_dir = it->second;
      build_cache_enabled = true;
    }
  }
  if (!build_cache_enabled) {
    std::string mindir_postfix = ".mindir";
    auto ops = output_file.find(mindir_postfix);
    if (ops != std::string::npos && ops == output_file.size() - mindir_postfix.size()) {
      output_file = output_file.substr(0, output_file.size() - mindir_postfix.size());
    }
    if (output_file.empty()) {
      MS_LOG(ERROR) << "Converter output file cannot be empty";
      return false;
    }
    build_cache_dir = output_file + "_variables";
  }
  if (lite::CreateDir(build_cache_dir) != RET_OK) {
    MS_LOG(ERROR) << "Failed to create build cache dir " << build_cache_dir;
    return false;
  }
  ge_options[lite::kGeGraphCompilerCacheDir] = build_cache_dir;
  MS_LOG(INFO) << "Update session attr " << lite::kGeGraphCompilerCacheDir << " to " << build_cache_dir;
  if (build_cache_dir.find(output_dir) == 0) {
    build_cache_relative_dir_ = "./" + build_cache_dir.substr(output_dir.size());
  }
  return true;
}

void GeGraphExecutor::SetGeDumpOptions(const std::map<std::string, std::string> &config,
                                       std::map<std::string, std::string> *ge_options_ptr) {
  MS_EXCEPTION_IF_NULL(ge_options_ptr);
  auto &ge_options = *ge_options_ptr;
  auto option_id = config.find(lite::kDumpPathKey);
  if (option_id == config.end()) {
    return;
  }
  auto dump_path = option_id->second;
  auto real_path = lite::RealPath(dump_path.c_str());
  std::ifstream ifs(real_path);
  if (!ifs.good() || !ifs.is_open()) {
    MS_LOG(EXCEPTION) << "The dump config file is not exit or open failed.";
  }
  nlohmann::json dump_cfg_json;
  try {
    dump_cfg_json = nlohmann::json::parse(ifs);
  } catch (const nlohmann::json::parse_error &error) {
    MS_LOG(EXCEPTION) << "parse json failed, please check the file.";
  }
  if (dump_cfg_json[kDump] != nullptr && dump_cfg_json[kDump][kDumpMode] != nullptr) {
    ge_options["ge.exec.enableDump"] = "1";
    ge_options["ge.exec.dumpMode"] = dump_cfg_json[kDump][kDumpMode].get<std::string>();
  }
}

void GeGraphExecutor::SetGeProfilingOptions(const std::map<std::string, std::string> &config,
                                            std::map<std::string, std::string> *ge_options_ptr) {
  MS_EXCEPTION_IF_NULL(ge_options_ptr);
  auto &ge_options = *ge_options_ptr;
  auto option_id = config.find(lite::kProfilingPathKey);
  if (option_id == config.end()) {
    return;
  }
  auto profiling_path = option_id->second;
  auto real_path = lite::RealPath(profiling_path.c_str());
  std::ifstream ifs(real_path);
  if (!ifs.good() || !ifs.is_open()) {
    MS_LOG(EXCEPTION) << "The profiling_path config file is not exit or open failed.";
  }
  nlohmann::json profiling_cfg_json;
  try {
    profiling_cfg_json = nlohmann::json::parse(ifs);
  } catch (const nlohmann::json::parse_error &error) {
    MS_LOG(EXCEPTION) << "parse json failed, please check the file.";
  }
  if (profiling_cfg_json[kProfiling] != nullptr) {
    ge_options["ge.exec.profilingMode"] = "1";
    ge_options["ge.exec.profilingOptions"] = profiling_cfg_json[kProfiling].dump();
  }
}

void GeGraphExecutor::GetGeSessionOptionsFromAscendContext(const std::map<std::string, std::string> &config,
                                                           std::map<std::string, std::string> *ge_options_ptr) {
  MS_EXCEPTION_IF_NULL(ge_options_ptr);
  auto &ge_options = *ge_options_ptr;
  SetGeDumpOptions(config, ge_options_ptr);
  SetGeProfilingOptions(config, ge_options_ptr);
  auto option_id = config.find(lite::kGeVariableMemoryMaxSize);
  if (option_id != config.end()) {
    ge_options["ge.variableMemoryMaxSize"] = option_id->second;
  }
  option_id = config.find(lite::kGeGraphMemoryMaxSize);
  if (option_id != config.end()) {
    ge_options["ge.graphMemoryMaxSize"] = option_id->second;
  }
  option_id = config.find(lite::kGraphCompilerCacheDirKey);
  if (option_id != config.end()) {
    ge_options[lite::kGeGraphCompilerCacheDir] = option_id->second;
  }
}

void GeGraphExecutor::GetGeGraphOptions(const FuncGraphPtr &anf_graph,
                                        std::map<std::string, std::string> *ge_options_ptr) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  MS_EXCEPTION_IF_NULL(ge_options_ptr);
  auto &ge_options = *ge_options_ptr;
  auto ascend_device_info = GeUtils::GetAscendDeviceInfo(context_);
  if (ascend_device_info == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to get graph session options, can not find ascend device context.";
  }
  uint32_t rank_id = ascend_device_info->GetRankID();
  graph_name_ = std::to_string(rank_id) + "_" + std::to_string(global_graph_idx_) + "_" + anf_graph->ToString();
  for (auto &c : graph_name_) {
    if (c == '.') {
      c = '_';
    }
  }
  ge_options[lite::kGeGraphKey] = graph_name_;
  auto config_it = config_infos_.find(lite::kGeGraphOptionsSection);
  if (config_it != config_infos_.end()) {
    for (auto &item : config_it->second) {
      ge_options[item.first] = item.second;
    }
  }

  auto precision_mode = ascend_device_info->GetPrecisionMode();
  if (!precision_mode.empty()) {
    ge_options["ge.exec.precision_mode"] = TransforPrecisionToAcl(precision_mode);
  }
  config_it = config_infos_.find(lite::kAscendContextSection);
  if (config_it == config_infos_.end()) {
    return;
  }
  auto config = config_it->second;
  auto option_id = config.find(lite::kModifyMixList);
  if (option_id != config.end()) {
    ge_options["ge.exec.modify_mixlist"] = option_id->second;
  }
}

int64_t GeGraphExecutor::GetSessionId() {
  std::string inner_group_id;
  (void)GetConfigOption(lite::kLiteInnerGroupSection, lite::kLiteInnerGroupId, &inner_group_id);
  if (inner_group_id.empty()) {
    return lite::kUnkonwnSessionId;
  }
  int64_t session_id = lite::kUnkonwnSessionId;
  if (!lite::ConvertStrToInt(inner_group_id, &session_id)) {
    MS_LOG(WARNING) << "Failed to parse session_id to int64_t";
    return lite::kUnkonwnSessionId;
  }
  return session_id;
}

bool GeGraphExecutor::CreateSession(const std::map<std::string, std::string> &extra_options) {
  if (ge_session_ != nullptr) {
    MS_LOG(INFO) << "Ge session has already been created";
    return true;
  }
  session_id_ = GetSessionId();
  (void)setenv("GE_TRAIN", "0", 1);
  std::map<std::string, std::string> session_options = extra_options;
  GetGeSessionOptions(&session_options);
  if (auto option_id = session_options.find(lite::kGeGraphCompilerCacheDir); option_id != session_options.end()) {
    build_cache_dir_ = option_id->second;
  }
  session_options_ = session_options;
  ge_session_ = GeSessionManager::CreateGeSession(session_id_, session_options);
  if (ge_session_ == nullptr) {
    MS_LOG(ERROR) << "Failed to create ge session";
    return false;
  }
  return true;
}

bool GeGraphExecutor::AddGraph(const backend::ge_backend::DfGraphPtr &graph,
                               const std::map<std::string, std::string> &options, uint32_t *graph_id_ret) {
  if (ge_session_ == nullptr) {
    MS_LOG(ERROR) << "Failed to add graph, ge session cannot be nullptr";
    return false;
  }
  auto graph_id = GetNextGraphIdx();
  auto ge_status = ge_session_->AddGraph(static_cast<uint32_t>(graph_id), *(graph), options);
  if (ge_status != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE AddGraph Failed: " << ge::GEGetErrorMsg();
    return false;
  }
  *graph_id_ret = graph_id;
  return true;
}

void GeGraphExecutor::GetParams(const FuncGraphPtr &anf_graph, backend::ge_backend::TensorOrderMap *param_tensors) {
  MS_EXCEPTION_IF_NULL(anf_graph);

  backend::ge_backend::TensorOrderMap res;
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      auto value = para->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
      MS_EXCEPTION_IF_NULL(tensor);
      auto para_name = para->name();
      res.emplace(para_name, tensor);
    }
  }
  if (session_id_ != lite::kUnkonwnSessionId) {
    std::vector<std::string> graph_params;
    std::transform(res.begin(), res.end(), std::back_inserter(graph_params),
                   [](const auto &item) { return item.first; });
    auto new_params_set = GeSessionManager::UpdateSessionVariables(session_id_, graph_params);
    for (auto &item : res) {
      // parameters not in new_params_set has been init by other graph
      if (new_params_set.find(item.first) == new_params_set.end()) {
        item.second->set_init_flag(true);
      }
    }
  }
  *param_tensors = res;
}

bool GeGraphExecutor::UpdateGraphInputs(const FuncGraphPtr &graph) {
  std::string input_shape_str;
  std::vector<GeDynamicShapeInfo> input_shapes;
  if (!GeDynamicUtils::GetGraphInputShapes(context_, config_infos_, &input_shapes, &input_shape_str)) {
    MS_LOG(ERROR) << "Failed to get input shape from AscendDeviceInfo or config file";
    return false;
  }
  if (input_shapes.empty()) {
    MS_LOG(INFO) << "Not found input shape in AscendDeviceInfo or config file";
    return true;
  }
  auto inputs = graph->get_inputs();
  if (inputs.size() != input_shapes.size()) {
    MS_LOG(ERROR) << "FuncGraph input size " << inputs.size() << " != input size " << input_shapes.size()
                  << " in AscendDeviceInfo or config file " << input_shapes.size();
    return false;
  }
  for (size_t i = 0; i < input_shapes.size(); i++) {
    auto node = inputs[i];
    MS_CHECK_TRUE_RET(node != nullptr, false);
    auto input_shape = input_shapes[i];
    auto para = node->cast<ParameterPtr>();
    if (para == nullptr) {
      MS_LOG(ERROR) << "Cast input to Parameter failed";
      return false;
    }
    MS_LOG(INFO) << "Func graph input_" << i << " " << para->name()
                 << ", shape: " << FuncGraphUtils::GetTensorShape({node, 0});

    auto it = std::find_if(input_shapes.begin(), input_shapes.end(),
                           [&para](const auto &item) { return item.name == para->name(); });
    if (it == input_shapes.end()) {
      MS_LOG(ERROR) << "Failed to find input " << para->name() << " in input_shape " << input_shape_str;
      return false;
    }
    auto abstract = para->abstract();
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "Get input abstract failed";
      return false;
    }
    ShapeVector shape;
    std::transform(it->shape.begin(), it->shape.end(), std::back_inserter(shape), [](auto &dim) { return dim.dim; });
    MS_LOG(INFO) << "Update shape of input_" << i << " " << para->name() << " to " << shape;
    abstract->set_shape(std::make_shared<abstract::Shape>(shape));
  }
  return true;
}

bool GeGraphExecutor::InitRefDataList(const std::vector<std::pair<std::string, tensor::TensorPtr>> &ref_data_tensors) {
  for (auto &item : ref_data_tensors) {
    auto para_name = item.first;
    auto &tensor = item.second;
    RefDataInfo ref_data_info;
    ref_data_info.name = para_name;
    ref_data_info.shape = tensor->shape_c();
    ref_data_info.dtype = static_cast<TypeId>(tensor->data_type_c());
    ref_data_info.host_data = item.second;
    MS_LOG(INFO) << "Init ref data info[" << ref_data_infos_.size() << "] :" << ref_data_info.name
                 << ", dtype:" << ref_data_info.dtype << ", shape:" << ref_data_info.shape;
    ref_data_infos_.push_back(ref_data_info);
  }
  return true;
}

bool GeGraphExecutor::InitMemoryContextManager() {
  auto session_context = GeSessionManager::GetGeSessionContext(session_id_);
  if (session_context != nullptr) {
    memory_manager_ = session_context->memory_manager.lock();
    context_manager_ = session_context->context_manager.lock();
  }
  if (memory_manager_ == nullptr) {
    memory_manager_ = std::make_shared<GeMemoryManager>();
    if (memory_manager_ == nullptr) {
      MS_LOG(ERROR) << "Failed to create memory manager";
      return false;
    }
    if (session_context != nullptr) {
      session_context->memory_manager = memory_manager_;
    }
  }
  if (context_manager_ == nullptr) {
    context_manager_ = std::make_shared<GeContextManager>();
    if (context_manager_ == nullptr) {
      MS_LOG(ERROR) << "Failed to create context manager";
      return false;
    }
    if (!context_manager_->InitContext(GetDeviceID())) {
      MS_LOG(ERROR) << "Failed to init device";
      return false;
    }
    if (session_context != nullptr) {
      session_context->context_manager = context_manager_;
    }
  }
  if (!context_manager_->SetContext()) {
    MS_LOG(ERROR) << "Failed to set ge context";
    return false;
  }
  return true;
}

bool GeGraphExecutor::InitRefDataDeviceTensor() {
  MS_LOG(INFO) << "InitRefDataDeviceTensor start.";
  if (ref_data_infos_.empty()) {
    MS_LOG(INFO) << "There is not ref data, no need to init ref data device data";
    return true;
  }
  std::map<std::string, RefDataInfo> session_ref_data_map;
  auto session_context = GeSessionManager::GetGeSessionContext(session_id_);
  if (session_context != nullptr) {
    session_ref_data_map = session_context->ref_data_map_;
  }

  size_t ref_data_total_size = 0;
  std::map<std::string, tensor::TensorPtr> new_param_tensor_map;
  for (size_t i = 0; i < ref_data_infos_.size(); i++) {
    auto &item = ref_data_infos_[i];
    auto tensor = item.host_data;
    item.size = tensor->DataSize();
    item.host_data = nullptr;  // release host memory
    ShapeVector ref_data_shape = tensor->shape_c();
    SetRefShape(&ref_data_shape, true, item.name);
    auto desc = device::ascend::TransformUtil::GetGeTensorDesc(
      ref_data_shape, static_cast<TypeId>(tensor->data_type_c()), kOpFormat_NCHW);
    if (desc == nullptr) {
      MS_LOG(ERROR) << "Failed to get Tensor Desc";
      return false;
    }
    desc->SetPlacement(::ge::kPlacementDevice);
    auto ret = item.ge_tensor.SetTensorDesc(*desc);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call ge::Tensor::SetTensorDesc, ret " << ret;
      return false;
    }
    if (auto ref_it = session_ref_data_map.find(item.name); ref_it != session_ref_data_map.end()) {
      auto &org_item = ref_it->second;
      MS_LOG(INFO) << "Find RefData " << item.name << ", shape " << org_item.shape << ", size " << org_item.size;
      if (org_item.size != item.size) {
        MS_LOG(ERROR) << "RefData " << item.name << " data size != the size in pre graph, current shape " << item.shape
                      << ", size " << item.size << ", pre shape " << org_item.shape << ", pre size " << org_item.size;
        return false;
      }
      auto dst_addr = ref_it->second.ge_tensor.GetData();
      ret = item.ge_tensor.SetData(dst_addr, item.size, [](uint8_t *) -> void {});
      if (ret != ge::GRAPH_SUCCESS) {
        MS_LOG(ERROR) << "Failed to call ge::Tensor SetData(uint8_t*, size, DeleteFunc), data size " << item.size;
        return false;
      }
    } else {
      item.offset = ref_data_total_size;
      ref_data_total_size += ALIGN_UP_REF_DATA(tensor->DataSize());
      new_param_tensor_map[item.name] = tensor;
    }
  }
  if (ref_data_total_size != 0) {
    auto device_memory = memory_manager_->MallocDeviceMemory("RefData input", ref_data_total_size);
    if (device_memory == nullptr) {
      return false;
    }
    for (auto &item : ref_data_infos_) {
      auto it = new_param_tensor_map.find(item.name);
      if (it == new_param_tensor_map.end()) {
        continue;
      }
      auto &tensor_val = it->second;
      auto dst_addr = device_memory + item.offset;
      if (!memory_manager_->MemcpyHost2Device(dst_addr, item.size, tensor_val->data_c(), tensor_val->DataSize())) {
        MS_LOG(ERROR) << "Failed to memory copy host data to device";
        return false;
      }
      auto ret = item.ge_tensor.SetData(dst_addr, item.size, [](uint8_t *) -> void {});
      if (ret != ge::GRAPH_SUCCESS) {
        MS_LOG(ERROR) << "Failed to call ge::Tensor SetData(uint8_t*, size, DeleteFunc), data size " << item.size;
        return false;
      }
      if (session_context != nullptr) {
        session_context->ref_data_map_[item.name] = item;
      }
    }
  }
  return true;
}

bool GeGraphExecutor::InitInOutDeviceBuffer(const std::string &name, const ShapeVector &shape, TypeId dtype,
                                            InOutBufferInfo *buffer_info) {
  MS_CHECK_TRUE_RET(buffer_info != nullptr, false);
  auto &info = *buffer_info;
  auto desc = device::ascend::TransformUtil::GetGeTensorDesc(shape, dtype, kOpFormat_NCHW);
  if (desc == nullptr) {
    MS_LOG(ERROR) << "Failed to get Tensor Desc";
    return false;
  }
  auto tensor_size = SizeOf(shape) * GetDataTypeSize(dtype);
  if (tensor_size <= 0) {
    MS_LOG(INFO) << "Failed to calculate " << name << " tensor size, shape " << ShapeVectorToStr(shape)
                 << ", date type " << dtype;
    return false;
  }
  desc->SetPlacement(::ge::kPlacementDevice);
  auto ret = info.ge_tensor.SetTensorDesc(*desc);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Failed to call ge::Tensor::SetTensorDesc, ret " << ret;
    return false;
  }
  info.device_addr = memory_manager_->MallocDeviceMemory(name, tensor_size);
  if (info.device_addr == nullptr) {
    MS_LOG(ERROR) << "Failed to malloc device memory for " << name << ", memory size " << tensor_size
                  << ", tensor shape " << shape;
    return false;
  }
  ret = info.ge_tensor.SetData(reinterpret_cast<uint8_t *>(info.device_addr), tensor_size, [](uint8_t *) -> void {});
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Failed to call ge::Tensor SetData(uint8_t*, size, DeleteFunc), data size " << tensor_size;
    return false;
  }
  info.max_size = tensor_size;
  info.shape = shape;
  info.dtype = dtype;
  return true;
}

bool GeGraphExecutor::UpdateInputShapeOption(
  const FuncGraphPtr &func_graph, const std::vector<std::pair<std::string, tensor::TensorPtr>> &ref_data_tensors,
  std::map<std::string, std::string> *ge_options_ptr) {
  if (ge_options_ptr == nullptr) {
    MS_LOG(ERROR) << "Input argument ge_options_ptr cannot be nullptr";
    return false;
  }
  std::string input_shape_str;
  std::vector<GeDynamicShapeInfo> input_shapes;
  if (!GeDynamicUtils::GetGraphInputShapes(context_, config_infos_, &input_shapes, &input_shape_str)) {
    MS_LOG(ERROR) << "Failed to get input shape from AscendDeviceInfo or config file";
    return false;
  }
  std::map<std::string, std::string> shape_map;
  if (input_shapes.empty()) {
    MS_LOG(INFO) << "Not found input shape in AscendDeviceInfo or config file";
    if (!dyn_kv_cache_info_.dynamic_kv_cache) {
      return true;
    }
    auto inputs = func_graph->get_inputs();
    bool dyn_input = false;
    for (auto &item : inputs) {
      auto shape = FuncGraphUtils::GetTensorShape({item, 0});
      if (std::any_of(shape.begin(), shape.end(), [](auto dim) { return dim < 0; })) {
        dyn_input = true;
      }
      shape_map[item->fullname_with_scope()] = lite::VectorToStrJoin(shape, ",");
    }
    if (!dyn_input) {
      MS_LOG(INFO) << "Current model has no dynamic inputs and there is no ge.inputShape set in config, skip update "
                      "ge.inputShape option for dynamic KVCache";
      return true;
    }
  } else {
    for (auto &item : input_shapes) {
      shape_map[item.name] = item.shape_str;
    }
  }
  for (auto &item : ref_data_tensors) {
    ShapeVector ref_dyn_shape = item.second->shape_c();
    SetRefShape(&ref_dyn_shape, true, item.first);
    shape_map[item.first] = lite::VectorToStrJoin(ref_dyn_shape, ",");
  }
  std::string new_input_shape_str = lite::MapToStrJoin(shape_map, ":", ";");
  GeDynamicUtils::UpdateGraphInputShapes(context_, &config_infos_, new_input_shape_str);
  (*ge_options_ptr)["ge.inputShape"] = new_input_shape_str;
  MS_LOG(INFO) << "Update ge.inputShape to " << new_input_shape_str;
  return true;
}

bool GeGraphExecutor::InitRefDataContext(const FuncGraphPtr &func_graph,
                                         const std::vector<std::pair<std::string, tensor::TensorPtr>> &ref_data_tensors,
                                         std::map<std::string, std::string> *ge_options_ptr) {
  if (!UpdateInputShapeOption(func_graph, ref_data_tensors, ge_options_ptr)) {
    MS_LOG(ERROR) << "Failed to update input shape option";
    return false;
  }
  if (!InitRefDataList(ref_data_tensors)) {
    MS_LOG(ERROR) << "Failed to init ref data list";
    return false;
  }
  if (!InitMaxShapeParam()) {
    MS_LOG(ERROR) << "Failed to init max shape size";
    return false;
  }
  return true;
}

backend::ge_backend::DfGraphPtr GeGraphExecutor::CreateFakeGraph(const std::map<std::string, std::string> &ge_options) {
  if (build_cache_dir_.empty()) {
    MS_LOG(INFO) << "Option model_cache_mode " << cache_mode_ << " is not mem_opt and not load offline model or "
                 << lite::kGeGraphCompilerCacheDir << " is empty, skip create small ge graph";
    return nullptr;
  }
  auto graph_it = ge_options.find(lite::kGeGraphKey);
  if (graph_it == ge_options.end()) {
    MS_LOG(INFO) << "Cannot find option " << lite::kGeGraphKey << ", skip create small ge graph";
    return nullptr;
  }
  auto graph_key = graph_it->second;
  auto idx_file_name = build_cache_dir_ + "/" + graph_key + ".idx";
  if (!UpdateOmCacheIdxFile(idx_file_name)) {
    return nullptr;
  }
  auto df_graph = GenExampleGraph(graph_key);
  if (df_graph == nullptr) {
    MS_LOG(WARNING) << "Failed to create small ge graph for graph " << graph_key << ", skip create small ge graph";
    return nullptr;
  }
  MS_LOG(INFO) << "Create small  ge graph for graph " << graph_key;
  return df_graph;
}

backend::ge_backend::DfGraphPtr GeGraphExecutor::CreateGeGraphOnline(
  const FuncGraphPtr &anf_graph, std::map<std::string, std::string> *ge_options_ptr) {
  MS_CHECK_TRUE_RET(ge_options_ptr != nullptr, nullptr);
  backend::ge_backend::TensorOrderMap params_vals;
  GetParams(anf_graph, &params_vals);
  backend::ge_backend::SetDynRefDataFunc dyn_ref_data_func = nullptr;
  if (dyn_kv_cache_info_.dynamic_kv_cache) {
    dyn_ref_data_func = [this](const AnfNodePtr &node, const ShapeVector &org_shape) -> ShapeVector {
      return SetKVCacheShape(dyn_kv_cache_info_.batch_size_dyn, dyn_kv_cache_info_.seq_length_dyn,
                             dyn_kv_cache_info_.kv_cache_layout, org_shape);
    };
  }

  auto converter = std::make_shared<backend::ge_backend::DfGraphConvertor>(anf_graph, "", ref_mode_flag_);
  // TODO(yefeng): backend::ge_backend::BuildGraph
  backend::ge_backend::BuildGraph(graph_name_, converter, params_vals);
  auto err_code = backend::ge_backend::ErrCode(converter);
  if (err_code != 0) {
    backend::ge_backend::ClearGraph();
    MS_LOG(ERROR) << "Convert df graph failed, err:" << err_code;
    return nullptr;
  }
  auto init_graph = backend::ge_backend::GetInitGraph(converter);
  if (init_graph != nullptr) {
    uint32_t init_graph_id = 0;
    if (!AddGraph(init_graph, {}, &init_graph_id)) {
      MS_LOG(ERROR) << "Failed to add init graph, graph name " << anf_graph->ToString();
      return nullptr;
    }
    auto init_data_names = converter->GetInitDataNames();
    // copy init weight to device
    if (!RunGeInitGraph(init_graph_id, init_data_names, params_vals)) {
      MS_LOG(ERROR) << "Failed to run init graph for " << anf_graph->ToString();
      return nullptr;
    }
    ge_session_->RemoveGraph(init_graph_id);
  } else {
    MS_LOG(INFO) << "There is no init graph for graph " << anf_graph->ToString();
  }
  if (ref_mode_flag_ != backend::ge_backend::RefModeFlag::kRefModeNone) {
    auto ref_data_names = converter->GetRefDataNames();
    std::vector<std::pair<std::string, tensor::TensorPtr>> ref_datas;
    std::transform(ref_data_names.begin(), ref_data_names.end(), std::back_inserter(ref_datas),
                   [&params_vals](auto &item) { return std::make_pair(item, params_vals.at(item)); });
    if (!InitRefDataContext(anf_graph, ref_datas, ge_options_ptr)) {
      MS_LOG(ERROR) << "Failed to init refdata context";
      return nullptr;
    }
  }
  auto df_graph = backend::ge_backend::GetComputeGraph(converter);
  return df_graph;
}

void GeGraphExecutor::SetOptionsIntoOfflineModel(const std::map<std::string, std::string> &graph_options,
                                                 std::map<std::string, ValuePtr> *attr_map_ptr) {
  auto &attr_map = *attr_map_ptr;

  if (!build_cache_relative_dir_.empty()) {
    attr_map[lite::kNameAttrWeightDir] = MakeValue(build_cache_relative_dir_);
    MS_LOG(INFO) << "Set graph attr " << lite::kNameAttrWeightDir << " to " << build_cache_relative_dir_;
  }
  // ge session options
  auto find_set_option = [](const std::map<std::string, std::string> &from_options,
                            std::vector<std::string> *to_options, const std::string &option) {
    auto config_it = from_options.find(option);
    if (config_it != from_options.end()) {
      to_options->push_back(option);
      to_options->push_back(config_it->second);
    }
  };
  std::vector<std::string> session_save_options;
  find_set_option(session_options_, &session_save_options, "ge.externalWeight");
  attr_map[lite::kGeSessionOptionsSection] = MakeValue(session_save_options);

  std::vector<std::string> graph_save_options;
  find_set_option(graph_options, &graph_save_options, "ge.inputShape");
  find_set_option(graph_options, &graph_save_options, "ge.dynamicDims");
  find_set_option(graph_options, &graph_save_options, "ge.dynamicNodeType");
  attr_map[lite::kGeGraphOptionsSection] = MakeValue(graph_save_options);
}

bool GeGraphExecutor::LoadOnlineGraph(const FuncGraphPtr &anf_graph, uint32_t *graph_id) {
  std::map<std::string, std::string> extra_session_options;
  if (!cache_mode_.empty()) {
    if (!SetModelCacheDir(&extra_session_options)) {
      return false;
    }
  }
  if (!CreateSession(extra_session_options)) {
    MS_LOG(ERROR) << "Failed to create ge session";
    return false;
  }
  std::map<std::string, std::string> ge_options;
  GetGeGraphOptions(anf_graph, &ge_options);
  auto df_graph = CompileGraphCommon(anf_graph, &ge_options);
  if (df_graph == nullptr) {
    MS_LOG(ERROR) << "Input param graph is nullptr.";
    return false;
  }
  if (cache_mode_ == "mem_opt") {
    auto fake_df_graph = CreateFakeGraph(ge_options);
    if (fake_df_graph != nullptr) {
      df_graph = fake_df_graph;
    }
  }
  if (!AddGraph(df_graph, ge_options, graph_id)) {
    MS_LOG(ERROR) << "Failed to add compute graph, graph name " << anf_graph->ToString();
    return false;
  }
  return true;
}

backend::ge_backend::DfGraphPtr GeGraphExecutor::CompileGraphCommon(
  const FuncGraphPtr &anf_graph, std::map<std::string, std::string> *ge_options_ptr) {
  if (anf_graph == nullptr) {
    MS_LOG(ERROR) << "Input param graph is nullptr.";
    return nullptr;
  }
  lite::AdjustDuplicateNodeName(anf_graph);
  if (ge_options_ptr == nullptr) {
    MS_LOG(ERROR) << "Input param ge_options_ptr is nullptr.";
    return nullptr;
  }
  auto remove_load_pass = std::make_shared<opt::RemoveLoadPass>();
  remove_load_pass->Run(anf_graph);

  if (!UpdateGraphInputs(anf_graph)) {
    MS_LOG(ERROR) << "Failed to update graph inputs";
    return nullptr;
  }

  opt::UpdateManager(anf_graph);

  // Convert mindir attributes to inputs because of dynamic_shape operator.
  // For the transformed operators, the GE adapter only supports inputs but not attributes.
  auto args_to_attr_pass = std::make_shared<opt::AttrToArgsPass>();
  if (args_to_attr_pass == nullptr) {
    MS_LOG(ERROR) << "create AttrToArgsPass failed";
    return nullptr;
  }
  if (!args_to_attr_pass->Run(anf_graph)) {
    MS_LOG(ERROR) << "convert args to attr pass failed";
    return nullptr;
  }

  backend::ge_backend::DfGraphPtr df_graph = nullptr;
  df_graph = CreateGeGraphOnline(anf_graph, ge_options_ptr);
  return df_graph;
}

Status GeGraphExecutor::CompileGraph(const FuncGraphPtr &anf_graph, const std::map<string, string> &,
                                     uint32_t *graph_id) {
  MS_CHECK_TRUE_RET(anf_graph != nullptr, Status(kLiteNullptr, "anf_graph is nullptr."));
  MS_CHECK_TRUE_RET(graph_id != nullptr, Status(kLiteNullptr, "graph_id is nullptr"));

  std::string compile_graph_parallel;
  GetConfigOption(lite::kCommonContextSection, lite::kCompileGraphParallel, &compile_graph_parallel);
  if (compile_graph_parallel == lite::kEnableValue) {
    MS_LOG(WARNING) << lite::kCompileGraphParallel << " does not support ge provider";
  }

  std::lock_guard lock(g_compile_graph_mutex);

  bool is_adapted = anf_graph->has_attr(kIsAdapted);
  if (!is_adapted) {
    auto ret = GeUtils::AdaptGraph(anf_graph);
    MS_CHECK_TRUE_MSG(ret == kSuccess, kLiteError, "Adapt graph failed");
    anf_graph->set_attr(kIsAdapted, MakeValue(true));
  }
  uint32_t compute_graph_id = 0;
  if (CustomAscendUtils::IsCustomFuncGraph(anf_graph)) {
    MS_LOG(ERROR) << "Offline converted MindIR is not supported currently";
    return Status(kLiteError, "Offline converted MindIR is not supported currently");
  } else {
    auto ret = LoadOnlineGraph(anf_graph, &compute_graph_id);
    if (!ret) {
      MS_LOG(ERROR) << "Failed to load online model";
      return Status(kLiteError, "Failed to load online model");
    }
  }
  compute_graph_id_list_.push_back(compute_graph_id);
  *graph_id = compute_graph_id;

  // Record whether the graph has dynamic input dimensions at compile time.
  bool has_dynamic_dim = false;
  auto graph_inputs = anf_graph->get_inputs();
  for (const auto &node : graph_inputs) {
    auto shape = FuncGraphUtils::GetTensorShape({node, 0});
    if (std::any_of(shape.begin(), shape.end(), [](auto dim) { return dim == -1; })) {
      has_dynamic_dim = true;
      break;
    }
  }
  graph_has_dynamic_dim_[*graph_id] = has_dynamic_dim;
  if (ref_mode_flag_ != backend::ge_backend::RefModeFlag::kRefModeNone) {
    if (!BuildGraphRefMode(anf_graph, compute_graph_id)) {
      MS_LOG(ERROR) << "Failed to build ge graph with refdata";
      return Status(kLiteError, "Failed to build ge graph with refdata");
    }
  } else {
    if (!InitMemoryContextManager()) {
      MS_LOG(WARNING) << "Failed to init memory context manager in CompileGraph (non-RefMode)";
    }
  }
  std::vector<MSTensorPtr> orig_output;
  std::vector<std::string> output_names;
  FuncGraphUtils::GetFuncGraphOutputsInfo(anf_graph, &orig_output, &output_names);
  original_graph_outputs_[*graph_id] = orig_output;
  return kSuccess;
}

bool GeGraphExecutor::GetOneRealInputs(const FuncGraphPtr &anf_graph, std::vector<ge::Tensor> *ge_tensors_ptr) {
  std::vector<std::pair<std::string, ShapeVector>> input_shapes_configs;
  std::string input_shape_str;
  if (!GeDynamicUtils::GetGraphOneRealShapes(context_, config_infos_, &input_shapes_configs, &input_shape_str)) {
    MS_LOG(ERROR) << "Failed to get one real input shape";
    return false;
  }
  std::vector<MSTensorPtr> inputs;
  std::vector<std::string> input_names;
  FuncGraphUtils::GetFuncGraphInputsInfo(anf_graph, &inputs, &input_names);
  if (!input_shapes_configs.empty() && input_shapes_configs.size() != inputs.size()) {
    MS_LOG(ERROR) << "Input count " << input_shapes_configs.size()
                  << " get from input_shape of AscendDeviceInfo or config file != input count " << inputs.size()
                  << " got from graph";
    return false;
  }
  std::vector<::ge::Tensor> ge_inputs;
  // cppcheck-suppress cppcheckError
  for (size_t i = 0; i < inputs.size(); i++) {
    auto &input = inputs[i];
    auto input_name = input_names[i];
    if (!input_shapes_configs.empty()) {
      auto it = std::find_if(input_shapes_configs.begin(), input_shapes_configs.end(),
                             [&input_name](const auto &item) { return input_name == item.first; });
      if (it == input_shapes_configs.end()) {
        MS_LOG(ERROR) << "Cannot find input " << input_name << " in input_shape " << input_shape_str;
        return false;
      }
      input->SetShape(it->second);
    } else if (GeDynamicUtils::IsDynamicInputShapes({input->Shape()})) {
      MS_LOG(ERROR) << "Input " << i << " is dynamic shape " << input->Shape()
                    << ", but there is no input shape specified in AscendDeviceInfo or config file";
      return false;
    }
    MS_LOG(INFO) << "Input " << i << " shape " << input->Shape() << ", datatype " << input->DataType();
    auto ge_tensor = ConvertMSTensor(input, kOpFormat_NCHW);
    if (ge_tensor == nullptr) {
      MS_LOG(ERROR) << "Failed to converter input " << i << " ME Tensor to GE Tensor";
      return false;
    }
    ge_inputs.emplace_back(*ge_tensor);
  }
  *ge_tensors_ptr = ge_inputs;
  return true;
}

bool GeGraphExecutor::AoeTuning(const FuncGraphPtr &anf_graph) {
  if (!CreateSession({})) {
    MS_LOG(ERROR) << "Failed to create ge session";
    return false;
  }
  std::map<std::string, std::string> ge_options;
  GetGeGraphOptions(anf_graph, &ge_options);
  auto df_graph = CompileGraphCommon(anf_graph, &ge_options);
  if (df_graph == nullptr) {
    MS_LOG(ERROR) << "Input param graph is nullptr.";
    return false;
  }
  std::vector<::ge::Tensor> ge_inputs;
  if (!GetOneRealInputs(anf_graph, &ge_inputs)) {
    MS_LOG(ERROR) << "Failed to get one real inputs";
    return false;
  }
  AoeApiTuning tuning;
  auto status = tuning.AoeTurningGraph(ge_session_, df_graph, ge_inputs, context_, config_infos_);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Failed to call AoeTurningGraph";
    return false;
  }
  return true;
}

bool GeGraphExecutor::RunGeInitGraph(uint32_t init_graph_id, const std::vector<std::string> &init_data_names,
                                     const backend::ge_backend::TensorOrderMap &params_vals) {
  std::vector<tensor::TensorPtr> init_data_tensors;
  for (auto &item : init_data_names) {
    auto it = params_vals.find(item);
    if (it == params_vals.end()) {
      MS_LOG(ERROR) << "Cannot find parameter " << item << " in parameter map";
      return false;
    }
    init_data_tensors.push_back(it->second);
  }
  MS_LOG(DEBUG) << "ExecInitGraph start.";
  std::vector<::ge::Tensor> ge_inputs;
  for (size_t i = 0; i < init_data_tensors.size(); i++) {
    auto &input = init_data_tensors[i];
    auto ge_tensor = device::ascend::TransformUtil::ConvertTensor(input, kOpFormat_NCHW, false);
    if (ge_tensor == nullptr) {
      MS_LOG(ERROR) << "Failed to converter input " << i << " ME Tensor to GE Tensor";
      return false;
    }
    ge_inputs.emplace_back(*ge_tensor);
  }
  std::vector<::ge::Tensor> ge_outputs;
  auto ge_status = ge_session_->RunGraph(init_graph_id, ge_inputs, ge_outputs);
  if (ge_status != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Exec init graph failed, graph id " << init_graph_id;
    return false;
  }
  MS_LOG(INFO) << "Exec init graph success, graph id " << init_graph_id;
  return true;
}

bool GeGraphExecutor::RunGeGraphAsync(uint32_t graph_id, const std::vector<::ge::Tensor> &inputs,
                                      std::vector<::ge::Tensor> *outputs) {
  bool is_finished = false;
  bool end_of_sequence = false;
  std::promise<void> promise;
  auto call_back = [outputs, &is_finished, &end_of_sequence, &promise](ge::Status ge_status,
                                                                       const std::vector<ge::Tensor> &ge_outputs) {
    if (ge_status == ge::GRAPH_SUCCESS) {
      *outputs = ge_outputs;
      is_finished = true;
    } else if (ge_status == ge::END_OF_SEQUENCE) {
      MS_LOG(ERROR) << "RunAsync out of range: End of sequence.";
      end_of_sequence = true;
    } else {
      MS_LOG(ERROR) << "RunAsync failed." << ge::GEGetErrorMsg();
    }
    promise.set_value();
    return;
  };
  if (ge_session_ == nullptr) {
    MS_LOG(ERROR) << "The GE session is null, can't run the graph!";
    return false;
  }
  ge::Status ret = ge_session_->RunGraphAsync(graph_id, inputs, call_back);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE RunGraphAsync Failed: " << ge::GEGetErrorMsg();
    return false;
  }
  auto future = promise.get_future();
  future.wait();
  if (end_of_sequence) {
    MS_LOG(ERROR) << "Failed to call GE RunGraphAsync: End of sequence";
    return false;
  }
  return is_finished;
}

bool GeGraphExecutor::PrepareInputTensors(const std::vector<mindspore::MSTensor> &inputs,
                                          std::vector<::ge::Tensor> *ge_inputs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    auto &input = inputs[i];
    MS_LOG(INFO) << "Input " << i << " shape " << tensor::ShapeToString(input.Shape()) << ", datatype "
                 << input.DataType();
    auto tensor_size = input.DataSize();
    auto &input_info = inputs_buffer_infos_[i];
    if (input_info.max_size < tensor_size) {
      MS_LOG(ERROR) << "Input " << i << " data size invalid, graph size " << input_info.max_size << ", given size "
                    << tensor_size;
      return false;
    }
    if (!memory_manager_->MemcpyHost2Device(input_info.device_addr, input_info.max_size, input.Data().get(),
                                            tensor_size)) {
      return false;
    }

    SetGeTensorShape(&input_info.ge_tensor, input.Shape());
    ge_inputs->push_back(input_info.ge_tensor);
  }
  return true;
}

bool GeGraphExecutor::PrepareRefDataInputs(std::vector<::ge::Tensor> *ge_inputs) {
  if (ge_inputs == nullptr) {
    MS_LOG(ERROR) << "ge_inputs is nullptr.";
    return false;
  }
  for (auto &item : ref_data_infos_) {
    if (dyn_kv_cache_info_.dynamic_kv_cache) {
      ShapeVector ref_real_shape =
        device::ascend::TransformUtil::ConvertGeShape(item.ge_tensor.GetTensorDesc().GetShape());
      SetRefShape(&ref_real_shape, false, item.name);
      SetGeTensorShape(&item.ge_tensor, ref_real_shape);
      MS_LOG(INFO) << "Update RefData Input " << item.name << " shape to " << tensor::ShapeToString(ref_real_shape);
    }
    ge_inputs->push_back(item.ge_tensor);
  }
  return true;
}

bool GeGraphExecutor::PrepareOutputTensors(std::vector<::ge::Tensor> *ge_outputs) {
  if (!dyn_kv_cache_info_.is_ge_graph_static_) {
    ge_outputs->resize(outputs_buffer_infos_.size());
    for (auto &ge_tensor : *ge_outputs) {
      auto ret = ge_tensor.SetData(nullptr, 0U, [](uint8_t *) -> void {});
      if (ret != ge::GRAPH_SUCCESS) {
        MS_LOG(ERROR) << "Failed to call ge::Tensor SetData(nullptr, 0, DeleteFunc) for output";
        return false;
      }
    }
  } else {
    for (auto &output : outputs_buffer_infos_) {
      ge_outputs->push_back(output.ge_tensor);
    }
  }
  return true;
}

bool GeGraphExecutor::InitInputDataTensor(const std::vector<mindspore::MSTensor> &inputs,
                                          std::vector<::ge::Tensor> *ge_inputs, std::vector<::ge::Tensor> *ge_outputs) {
  if (inputs_buffer_infos_.size() != inputs.size()) {
    MS_LOG(ERROR) << "Input data info size " << inputs_buffer_infos_.size() << " != inputs size " << inputs.size();
    return false;
  }
  if (memory_manager_ == nullptr) {
    MS_LOG(ERROR) << "Memory manager or context manager is nullptr";
    return false;
  }
  if (!PrepareInputTensors(inputs, ge_inputs)) {
    return false;
  }
  if (!PrepareRefDataInputs(ge_inputs)) {
    return false;
  }
  if (!PrepareOutputTensors(ge_outputs)) {
    return false;
  }
  return true;
}

bool GeGraphExecutor::BuildGraphRefMode(const FuncGraphPtr &anf_graph, uint32_t graph_id) {
  MS_LOG(INFO) << "Call GE CompileGraph start, graph id " << graph_id;
  ge::Status ret = ge_session_->CompileGraph(graph_id);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE CompileGraph Failed: " << ge::GEGetErrorMsg();
    return false;
  }
  MS_LOG(INFO) << "Call GE CompileGraph end, graph id " << graph_id;
  if (!InitMemoryContextManager()) {
    return false;
  }

  if (!InitRefDataDeviceTensor()) {
    MS_LOG(ERROR) << "Failed to init ref data device data";
    return false;
  }

  // ref data input memories have been allocated
  // for input data memory
  if (!InitInputDeviceTensor(anf_graph)) {
    MS_LOG(ERROR) << "Failed to init input data device data";
    return false;
  }

  // for output memory
  if (!InitOutputDeviceTensor(anf_graph, graph_id)) {
    MS_LOG(ERROR) << "Failed to init input data device data";
    return false;
  }
  return true;
}

bool GeGraphExecutor::RunGraphRefMode(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
                                      std::vector<mindspore::MSTensor> *outputs) {
  MS_LOG(INFO) << "RunGraphRefMode begin";
  std::vector<::ge::Tensor> ge_inputs;
  std::vector<::ge::Tensor> ge_outputs;
  if (!InitRealShapeParam(inputs)) {
    return false;
  }
  if (!InitInputDataTensor(inputs, &ge_inputs, &ge_outputs)) {
    MS_LOG(ERROR) << "Init input tensor failed in run graph.";
    return false;
  }
  auto stream = context_manager_->GetDefaultStream();
  if (!RunGraphWithStreamAsync(graph_id, stream, ge_inputs, &ge_outputs)) {
    MS_LOG(ERROR) << "Failed in run graph with stream async.";
    return false;
  }
  if (!SyncDeviceOutputsToHost(outputs, &ge_outputs)) {
    MS_LOG(ERROR) << "Failed in sync device output to host.";
    return false;
  }
  MS_LOG(INFO) << "RunGraphRefMode end";
  return true;
}

bool GeGraphExecutor::SyncDeviceOutputsToHost(std::vector<mindspore::MSTensor> *outputs,
                                              std::vector<::ge::Tensor> *ge_outputs) {
  UpdateOutputShapeInfo(ge_outputs);

  size_t output_size = outputs_buffer_infos_.size();
  if (!outputs->empty()) {
    if (outputs->size() != output_size) {
      MS_LOG(ERROR) << "Invalid output size, outputs' size " << outputs->size() << "ge tensor size " << output_size;
      return false;
    }
    // cppcheck-suppress cppcheckError
    for (size_t i = 0; i < output_size; ++i) {
      auto &output_info = outputs_buffer_infos_[i];
      auto &output = (*outputs)[i];
      if (output.DataSize() < output_info.max_size) {
        MS_LOG(EXCEPTION) << "Output node " << i << "'s mem size " << output.DataSize()
                          << " is less than actual output size " << output_info.max_size;
      }
      if ((*outputs)[i].Data() == nullptr) {
        MS_LOG(ERROR) << "Output data ptr is nullptr.";
        return false;
      }
      auto mem_ret =
        memory_manager_->MemcpyDevice2Host(reinterpret_cast<uint8_t *>(output.MutableData()), output.DataSize(),
                                           output_info.device_addr, output_info.max_size);
      if (!mem_ret) {
        MS_LOG(ERROR) << "Failed to copy output data, dst size: " << output.DataSize()
                      << ", src size: " << output_info.max_size;
        return false;
      }
      MS_LOG(INFO) << "Output " << i << " shape " << tensor::ShapeToString(output_info.shape) << ", datatype "
                   << output_info.dtype;
    }
  } else {
    for (size_t i = 0; i < output_size; i++) {
      auto &output_info = outputs_buffer_infos_[i];
      auto ms_tensor = MSTensor("", static_cast<DataType>(output_info.dtype), output_info.shape, nullptr, 0);
      auto mem_ret =
        memory_manager_->MemcpyDevice2Host(reinterpret_cast<uint8_t *>(ms_tensor.MutableData()), ms_tensor.DataSize(),
                                           output_info.device_addr, output_info.max_size);
      if (!mem_ret) {
        MS_LOG(ERROR) << "Failed to copy output data, dst size: " << ms_tensor.DataSize()
                      << ", src size: " << output_info.max_size;
        return false;
      }
      MS_LOG(INFO) << "Output " << i << " shape " << tensor::ShapeToString(output_info.shape) << ", datatype "
                   << output_info.dtype;
      outputs->push_back(ms_tensor);
    }
  }
  return true;
}

bool GeGraphExecutor::RunGraphWithStreamAsync(uint32_t graph_id, void *stream, const std::vector<GeTensor> &inputs,
                                              std::vector<GeTensor> *outputs) {
  MS_CHECK_TRUE_RET(outputs != nullptr, false);
  for (auto ge_input : inputs) {
    MS_LOG(INFO) << "In ge graph " << graph_id << ", input for RunGraphWithStreamAsync : "
                 << tensor::ShapeToString(
                      device::ascend::TransformUtil::ConvertGeShape(ge_input.GetTensorDesc().GetShape()));
  }
  MS_LOG(INFO) << "Run the graph in GE with " << inputs.size() << " inputs";
  struct timeval start_time;
  (void)gettimeofday(&start_time, nullptr);

  MS_CHECK_TRUE_RET(ge_session_ != nullptr, false);
  ge::Status ret = ge_session_->RunGraphWithStreamAsync(graph_id, stream, inputs, *outputs);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE RunGraphWithStreamAsync Failed, ret is: " << ret;
    return false;
  }
  if (!context_manager_->SyncStream(stream)) {
    MS_LOG(ERROR) << "Sync stream for RunGraphWithStreamAsync failed";
    return false;
  }
  struct timeval end_time;
  (void)gettimeofday(&end_time, nullptr);
  const uint64_t kUSecondInSecond = 1000000;
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "Call GE RunGraphWithStreamAsync Success in " << cost << " us, GE outputs num: " << outputs->size()
               << ", graph id: " << graph_id;

  return true;
}

Status GeGraphExecutor::RunGraph(uint32_t graph_id, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                                 const std::map<string, string> & /* compile_options */) {
  if (outputs == nullptr) {
    MS_LOG(ERROR) << " Input param is nullptr.";
    return kLiteError;
  }

  MS_LOG(INFO) << "Run ge graph [" << graph_id << "] with " << inputs.size() << " ms_tensor_inputs";
  for (size_t i = 0; i < inputs.size(); i++) {
    auto &ms_tensor_input = inputs[i];
    MS_LOG(INFO) << "Input " << i << " shape " << ms_tensor_input.Shape() << ", datatype "
                 << ms_tensor_input.DataType();
  }
  if (ref_mode_flag_ != backend::ge_backend::RefModeFlag::kRefModeNone) {
    auto ret = RunGraphRefMode(graph_id, inputs, outputs);
    if (!ret) {
      return kLiteError;
    }
    return kSuccess;
  }
  // Zero-copy: if any input or output is a device tensor, use RunGraphWithStreamAsync
  // with pre-allocated device buffers. Only after first inference (graph must be
  // lazily compiled by RunGeGraphAsync first).
  bool zero_copy = !is_first_inference_;
  if (zero_copy) {
    bool has_device = false;
    for (const auto &in : inputs) {
      if (in.IsDevice()) {
        has_device = true;
        break;
      }
    }
    if (!has_device && !outputs->empty()) {
      for (const auto &out : *outputs) {
        if (out.IsDevice()) {
          has_device = true;
          break;
        }
      }
    }
    zero_copy = has_device;
  }
  if (is_first_inference_) {
    // First inference: use RunGeGraphAsync to trigger GE lazy compilation.
    is_first_inference_ = false;
    zero_copy = false;
  }

  std::vector<::ge::Tensor> ge_outputs;
  auto time_start = std::chrono::system_clock::now();
  Status status;
  if (zero_copy) {
    status = RunGraphZeroCopy(graph_id, inputs, outputs, &ge_outputs);
  } else {
    status = RunGraphNormal(graph_id, inputs, outputs, &ge_outputs);
  }
  if (status != kSuccess) {
    return status;
  }
  auto time_cost =
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - time_start).count();
  MS_LOG(INFO) << "Call GE RunGraph Success in " << time_cost << " us, graph id " << graph_id
               << " the GE outputs num is: " << ge_outputs.size();
  return kSuccess;
}

bool GeGraphExecutor::IsDynamicBucketOrStatic(uint32_t graph_id) const {
  auto dyn_it = graph_has_dynamic_dim_.find(graph_id);
  bool has_dynamic_dim = (dyn_it != graph_has_dynamic_dim_.end()) ? dyn_it->second : false;
  if (!has_dynamic_dim) {
    return true;
  }
  auto config_it = config_infos_.find(lite::kGeGraphOptionsSection);
  if (config_it != config_infos_.end()) {
    auto option_it = config_it->second.find("ge.dynamicDims");
    if (option_it != config_it->second.end() && !option_it->second.empty()) {
      return true;
    }
  }
  return false;
}

Status GeGraphExecutor::RunGraphNormal(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
                                       std::vector<mindspore::MSTensor> *outputs, std::vector<GeTensor> *ge_outputs) {
  std::vector<::ge::Tensor> ge_inputs;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto ge_tensor = ConvertMSTensor(std::make_shared<MSTensor>(inputs[i]), kOpFormat_NCHW, false);
    if (ge_tensor == nullptr) {
      MS_LOG(ERROR) << "Failed to converter input " << i << " ME Tensor to GE Tensor";
      return kLiteError;
    }
    ge_inputs.emplace_back(*ge_tensor);
  }
  auto ref_data_infos = ref_data_infos_;
  std::transform(ref_data_infos.begin(), ref_data_infos.end(), std::back_inserter(ge_inputs),
                 [](const auto &item) { return item.ge_tensor; });
  bool exec_ok = RunGeGraphAsync(graph_id, ge_inputs, ge_outputs);
  if (!exec_ok) {
    MS_LOG(ERROR) << "Exec compute graph failed, graph id " << graph_id;
    return kLiteError;
  }
  return HandleNormalOutputs(graph_id, inputs, outputs, ge_outputs);
}

Status GeGraphExecutor::HandleNormalOutputs(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
                                            std::vector<mindspore::MSTensor> *outputs,
                                            std::vector<GeTensor> *ge_outputs) {
  if (!outputs->empty()) {
    if (outputs->size() != ge_outputs->size()) {
      MS_LOG(ERROR) << "Invalid output size, outputs' size " << outputs->size() << "ge tensor size "
                    << ge_outputs->size();
      return kLiteError;
    }
    for (size_t i = 0; i < outputs->size(); ++i) {
      const auto &tensor = (*ge_outputs)[i];
      auto &output = (*outputs)[i];
      if (output.DataSize() < LongToSize(UlongToLong(tensor.GetSize()))) {
        MS_LOG(EXCEPTION) << "Output node " << i << "'s mem size " << output.DataSize()
                          << " is less than actual output size " << tensor.GetSize();
      }
      if (tensor.GetData() == nullptr) {
        MS_LOG(ERROR) << "GE output " << i << " data ptr is nullptr.";
        return kLiteError;
      }
      // User device output (first inference / non-zero-copy path): device tensor's
      // Data()/MutableData() is null, cannot use host memcpy; copy to user's device
      // buffer via D2D or H2D based on GE output placement.
      void *user_device_buf = output.GetDeviceData();
      bool ge_out_is_device = (tensor.GetTensorDesc().GetPlacement() == ::ge::kPlacementDevice);
      if (user_device_buf != nullptr || ge_out_is_device) {
        if (memory_manager_ == nullptr && !InitMemoryContextManager()) {
          MS_LOG(ERROR) << "Failed to init memory context manager for device output copy.";
          return kLiteError;
        }
        auto src_size = LongToSize(UlongToLong(tensor.GetSize()));
        bool copy_ok = false;
        if (user_device_buf != nullptr) {
          copy_ok =
            ge_out_is_device
              ? memory_manager_->MemcpyDevice2Device(user_device_buf, output.DataSize(), tensor.GetData(), src_size)
              : memory_manager_->MemcpyHost2Device(user_device_buf, output.DataSize(), tensor.GetData(), src_size);
        } else {
          // User host buffer + GE device output -> D2H copy
          if (output.MutableData() == nullptr) {
            MS_LOG(ERROR) << "Output host data ptr is nullptr.";
            return kLiteError;
          }
          copy_ok =
            memory_manager_->MemcpyDevice2Host(output.MutableData(), output.DataSize(), tensor.GetData(), src_size);
        }
        if (!copy_ok) {
          MS_LOG(ERROR) << "Failed to copy output data " << i << ", dst size: " << output.DataSize()
                        << ", src size: " << src_size;
          return kLiteError;
        }
        MS_LOG(INFO) << "Output " << i << " copied, size " << src_size;
        continue;
      }
      if (output.Data() == nullptr) {
        MS_LOG(ERROR) << "Output data ptr is nullptr.";
        return kLiteError;
      }
      auto mem_ret = common::huge_memcpy(static_cast<uint8_t *>(output.MutableData()), output.DataSize(),
                                         tensor.GetData(), tensor.GetSize());
      if (mem_ret != EOK) {
        MS_LOG(ERROR) << "Failed to copy output data, dst size: " << output.DataSize()
                      << ", src size: " << tensor.GetSize();
        return kLiteError;
      }
    }
  } else {
    for (size_t i = 0; i < ge_outputs->size(); i++) {
      auto &ge_tensor = (*ge_outputs)[i];
      auto ms_tensor = ConvertGeTensorNoCopy(&ge_tensor, graph_id, i);
      if (ms_tensor == nullptr) {
        MS_LOG(ERROR) << "Failed to converter output " << i << " GE Tensor to ME Tensor";
        return kLiteError;
      }
      MS_LOG(INFO) << "Output " << i << " shape " << tensor::ShapeToString(ms_tensor->Shape()) << ", datatype "
                   << ms_tensor->DataType();
      outputs->push_back(*ms_tensor);
    }
  }
  graph_inputs_[graph_id] = inputs;
  graph_outputs_[graph_id] = *outputs;
  MS_LOG(INFO) << "GE run graph " << graph_id << " end.";
  return kSuccess;
}

Status GeGraphExecutor::RunGraphZeroCopy(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
                                         std::vector<mindspore::MSTensor> *outputs, std::vector<GeTensor> *ge_outputs) {
  if (!IsDynamicBucketOrStatic(graph_id)) {
    MS_LOG(ERROR) << "Zero-copy is not supported for purely dynamic graphs, graph id " << graph_id
                  << ". Set ge.dynamicDims for dynamic-bucket mode.";
    return kLiteError;
  }
  // Zero-copy path: use pre-allocated device buffers (inputs_buffer_infos_ /
  // outputs_buffer_infos_). Device inputs/outputs use user's buffer directly;
  // host inputs H2D to pre-allocated; host outputs D2H from pre-allocated.
  if (memory_manager_ == nullptr && !InitMemoryContextManager()) {
    MS_LOG(ERROR) << "Failed to init memory context manager for zero-copy.";
    return kLiteError;
  }
  std::vector<::ge::Tensor> ge_inputs;
  // Prepare inputs: device → zero-copy bind user buffer, host → H2D to cached device buffer.
  // (RefMode's PrepareInputTensors depends on inputs_buffer_infos_; non-RefMode does
  // not initialize it, so it cannot be reused.)
  if (!PrepareZeroCopyInputs(inputs, &ge_inputs)) {
    MS_LOG(ERROR) << "Failed to prepare input tensors for zero-copy";
    return kLiteError;
  }
  auto ref_data_infos = ref_data_infos_;
  std::transform(ref_data_infos.begin(), ref_data_infos.end(), std::back_inserter(ge_inputs),
                 [](const auto &item) { return item.ge_tensor; });
  // Prepare outputs: device → bind user's buffer (zero-copy); host → cached temp
  // device buffer (GE writes, then D2H copy out), symmetric with input side.
  std::vector<bool> output_is_device;
  if (!BindZeroCopyOutputs(outputs, ge_outputs, &output_is_device)) {
    return kLiteError;
  }
  auto stream = context_manager_->GetDefaultStream();
  bool exec_ok = RunGraphWithStreamAsync(graph_id, stream, ge_inputs, ge_outputs);
  if (!exec_ok) {
    MS_LOG(ERROR) << "RunGraphWithStreamAsync failed for zero-copy, graph id " << graph_id;
    return kLiteError;
  }
  Status status = HandleZeroCopyOutputs(graph_id, outputs, ge_outputs, output_is_device);
  if (status != kSuccess) {
    return status;
  }
  graph_inputs_[graph_id] = inputs;
  graph_outputs_[graph_id] = *outputs;
  return kSuccess;
}

uint8_t *GeGraphExecutor::GetCachedDeviceBuffer(const std::string &name, size_t size) {
  auto it = cached_temp_buffers_.find(name);
  if (it != cached_temp_buffers_.end() && it->second.second >= size) {
    return static_cast<uint8_t *>(it->second.first);
  }
  // Need a new (or larger) buffer; free the old one if it exists.
  if (it != cached_temp_buffers_.end()) {
    memory_manager_->FreeDeviceMemory(it->second.first);
  }
  auto buf = memory_manager_->MallocDeviceMemory(name, size);
  if (buf == nullptr) {
    MS_LOG(ERROR) << "Failed to malloc cached device memory for " << name << ", size " << size;
    return nullptr;
  }
  cached_temp_buffers_[name] = {buf, size};
  return buf;
}

bool GeGraphExecutor::PrepareZeroCopyInputs(const std::vector<mindspore::MSTensor> &inputs,
                                            std::vector<GeTensor> *ge_inputs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    auto &input = inputs[i];
    if (input.IsDevice()) {
      // Device input: zero-copy bind user's device buffer with kPlacementDevice.
      auto desc = device::ascend::TransformUtil::GetGeTensorDesc(input.Shape(), static_cast<TypeId>(input.DataType()),
                                                                 kOpFormat_NCHW);
      if (desc == nullptr) {
        MS_LOG(ERROR) << "Failed to get tensor desc for device input " << i;
        return false;
      }
      desc->SetPlacement(::ge::kPlacementDevice);
      ge::Tensor ge_tensor;
      if (ge_tensor.SetTensorDesc(*desc) != ACL_SUCCESS) {
        MS_LOG(ERROR) << "Failed to set tensor desc for device input " << i;
        return false;
      }
      auto device_data = const_cast<MSTensor &>(input).GetDeviceData();
      auto data_size = input.DataSize();
      if (device_data == nullptr || data_size == 0) {
        MS_LOG(ERROR) << "Device input " << i << " has null device data or zero size";
        return false;
      }
      if (ge_tensor.SetData(reinterpret_cast<uint8_t *>(device_data), data_size, [](uint8_t *) -> void {}) !=
          ge::GRAPH_SUCCESS) {
        MS_LOG(ERROR) << "Failed to bind device data to ge input " << i;
        return false;
      }
      ge_inputs->push_back(ge_tensor);
      continue;
    }
    // Host input: H2D to a temp device buffer (RunGraphWithStreamAsync requires device inputs).
    auto tensor_size = input.DataSize();
    if (tensor_size == 0 || input.Data().get() == nullptr) {
      MS_LOG(ERROR) << "Input " << i << " host data is invalid, size " << tensor_size;
      return false;
    }
    // Reuse cached device buffer (no per-inference alloc/free).
    auto device_buf = GetCachedDeviceBuffer("input_" + std::to_string(i), tensor_size);
    if (device_buf == nullptr) {
      MS_LOG(ERROR) << "Failed to get cached device buffer for H2D input, size " << tensor_size;
      return false;
    }
    if (!memory_manager_->MemcpyHost2Device(device_buf, tensor_size, input.Data().get(), tensor_size)) {
      MS_LOG(ERROR) << "Failed to H2D copy input " << i << ", size " << tensor_size;
      return false;
    }
    auto desc = device::ascend::TransformUtil::GetGeTensorDesc(input.Shape(), static_cast<TypeId>(input.DataType()),
                                                               kOpFormat_NCHW);
    if (desc == nullptr) {
      MS_LOG(ERROR) << "Failed to get tensor desc for H2D input " << i;
      return false;
    }
    desc->SetPlacement(::ge::kPlacementDevice);
    ge::Tensor ge_tensor;
    if (ge_tensor.SetTensorDesc(*desc) != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Failed to set tensor desc for H2D input " << i;
      return false;
    }
    if (ge_tensor.SetData(device_buf, tensor_size, [](uint8_t *) -> void {}) != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Failed to bind H2D device data to ge input " << i << ", size " << tensor_size;
      return false;
    }
    ge_inputs->push_back(ge_tensor);
  }
  return true;
}

Status GeGraphExecutor::HandleZeroCopyOutputs(uint32_t graph_id, std::vector<mindspore::MSTensor> *outputs,
                                              std::vector<GeTensor> *ge_outputs,
                                              const std::vector<bool> &output_is_device) {
  // Handle outputs after execution.
  if (outputs->empty()) {
    // No user outputs: create MSTensors from GE output tensors.
    for (size_t i = 0; i < ge_outputs->size(); i++) {
      auto &ge_tensor = (*ge_outputs)[i];
      auto ms_tensor = ConvertGeTensorNoCopy(&ge_tensor, graph_id, i);
      if (ms_tensor == nullptr) {
        MS_LOG(ERROR) << "Failed to converter output " << i << " GE Tensor to ME Tensor";
        return kLiteError;
      }
      outputs->push_back(*ms_tensor);
    }
    return kSuccess;
  }
  for (size_t i = 0; i < outputs->size(); i++) {
    auto out_shape = device::ascend::TransformUtil::ConvertGeShape((*ge_outputs)[i].GetTensorDesc().GetShape());
    (*outputs)[i].SetShape(out_shape);
    if (output_is_device[i]) {
      continue;
    }
    // Host output: D2H copy from the cached temp device buffer (bound to ge_outputs[i]) to user host buffer.
    auto src_size = static_cast<size_t>((*ge_outputs)[i].GetSize());
    auto src_ptr = (*ge_outputs)[i].GetData();
    if (src_ptr == nullptr) {
      MS_LOG(ERROR) << "Temp output device ptr is nullptr for index " << i;
      return kLiteError;
    }
    if ((*outputs)[i].MutableData() == nullptr) {
      MS_LOG(ERROR) << "Output " << i << " host data ptr is nullptr.";
      return kLiteError;
    }
    if (!memory_manager_->MemcpyDevice2Host((*outputs)[i].MutableData(), (*outputs)[i].DataSize(), src_ptr, src_size)) {
      MS_LOG(ERROR) << "Failed to D2H copy output " << i;
      return kLiteError;
    }
  }
  return kSuccess;
}

bool GeGraphExecutor::BindZeroCopyOutputs(std::vector<mindspore::MSTensor> *outputs, std::vector<GeTensor> *ge_outputs,
                                          std::vector<bool> *output_is_device) {
  size_t output_count = !outputs->empty() ? outputs->size() : 0;
  for (size_t i = 0; i < output_count; i++) {
    auto &output = (*outputs)[i];
    bool is_device = output.IsDevice();
    output_is_device->push_back(is_device);
    auto desc = device::ascend::TransformUtil::GetGeTensorDesc(output.Shape(), static_cast<TypeId>(output.DataType()),
                                                               kOpFormat_NCHW);
    if (desc == nullptr) {
      MS_LOG(ERROR) << "Failed to get tensor desc for zero-copy output " << i;
      return false;
    }
    desc->SetPlacement(::ge::kPlacementDevice);
    ge::Tensor ge_tensor;
    auto ret = ge_tensor.SetTensorDesc(*desc);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Failed to set tensor desc for zero-copy output " << i << ", ret " << ret;
      return false;
    }
    if (is_device) {
      // Device output: bind user's device buffer, GE writes directly (zero-copy).
      ret = ge_tensor.SetData(reinterpret_cast<uint8_t *>(output.GetDeviceData()), output.DataSize(),
                              [](uint8_t *) -> void {});
      if (ret != ge::GRAPH_SUCCESS) {
        MS_LOG(ERROR) << "Failed to bind device data to ge output " << i;
        return false;
      }
    } else {
      // Host output: cached device buffer, GE writes, then D2H copy out.
      if (output.DataSize() == 0) {
        MS_LOG(ERROR) << "Host output " << i << " size is 0";
        return false;
      }
      auto device_buf = GetCachedDeviceBuffer("output_" + std::to_string(i), output.DataSize());
      if (device_buf == nullptr) {
        MS_LOG(ERROR) << "Failed to get cached device buffer for host output " << i << ", size " << output.DataSize();
        return false;
      }
      ret = ge_tensor.SetData(device_buf, output.DataSize(), [](uint8_t *) -> void {});
      if (ret != ge::GRAPH_SUCCESS) {
        MS_LOG(ERROR) << "Failed to bind temp device data to ge output " << i;
        return false;
      }
    }
    ge_outputs->push_back(ge_tensor);
  }
  return true;
}

std::vector<mindspore::MSTensor> GeGraphExecutor::GetInputInfos(uint32_t graph_id) {
  return graph_inputs_.find(graph_id) != graph_inputs_.end() ? graph_inputs_.at(graph_id)
                                                             : std::vector<mindspore::MSTensor>();
}

MSTensorPtr GeGraphExecutor::ConvertGeTensorNoCopy(::ge::Tensor *ge_tensor_ptr, uint32_t graph_id, size_t idx) {
  MS_CHECK_TRUE_RET(ge_tensor_ptr != nullptr, nullptr);
  auto &ge_tensor = *ge_tensor_ptr;
  auto ge_tensor_desc = ge_tensor.GetTensorDesc();
  auto me_shape = device::ascend::TransformUtil::ConvertGeShape(ge_tensor_desc.GetShape());
  if (original_graph_outputs_.find(graph_id) == original_graph_outputs_.end()) {
    MS_LOG(ERROR) << "Graph original outputs with the given graph id is not found.";
    return nullptr;
  }
  auto original_outputs = original_graph_outputs_[graph_id];
  if (idx >= original_outputs.size()) {
    MS_LOG(ERROR) << "Graph output index is out of range.";
    return nullptr;
  }
  // Use GE output's actual dtype first; fall back to ANF graph dtype if unknown
  auto type_id = static_cast<TypeId>(original_outputs[idx]->DataType());
  if (type_id == kTypeUnknown) {
    MS_LOG(ERROR) << "Could not convert Ge Tensor because of unsupported data type: "
                  << static_cast<int>(ge_tensor_desc.GetDataType());
    return nullptr;
  }
  // Support both HOST and DEVICE placements
  bool is_device_placement = (ge_tensor_desc.GetPlacement() == ::ge::kPlacementDevice);
  auto &&ge_data_uni = ge_tensor.ResetData();
  auto deleter = ge_data_uni.get_deleter();
  auto ge_data = ge_data_uni.release();
  if (ge_data == nullptr) {
    MS_LOG(ERROR) << "Ge data cannot be nullptr";
    return nullptr;
  }
  constexpr int64_t kTensorAlignBytes = 64;
  if (reinterpret_cast<uintptr_t>(ge_data) % kTensorAlignBytes != 0) {
    MS_LOG(ERROR) << "Skip zero-copy ge tensor , because of bytes not aligned with expected.";
    return nullptr;
  }
  int64_t elem_num = 1;
  for (size_t i = 0; i < me_shape.size(); ++i) {
    elem_num *= me_shape[i];
  }
  auto type_byte = GetTypeByte(TypeIdToType(type_id));
  if (type_byte * elem_num != ge_tensor.GetSize()) {
    MS_LOG(ERROR) << "Output datatype error! expected=" << (type_byte * elem_num)
                  << ", ge_size=" << ge_tensor.GetSize();
    return nullptr;
  }
  auto tensor_impl = std::make_shared<TensorDefaultImpl>("", static_cast<DataType>(type_id), me_shape);
  MS_CHECK_TRUE_RET(tensor_impl != nullptr, nullptr);
  tensor_impl->SetDeleter(deleter);
  if (is_device_placement) {
    tensor_impl->SetDeviceId(static_cast<int>(GetDeviceID()));
    tensor_impl->SetDeviceData(static_cast<void *>(ge_data));
  } else {
    tensor_impl->SetData(static_cast<void *>(ge_data), true);
  }
  auto tensor = std::make_shared<MSTensor>(tensor_impl);
  MS_CHECK_TRUE_RET(tensor != nullptr, nullptr);
  return tensor;
}

std::vector<mindspore::MSTensor> GeGraphExecutor::GetOutputInfos(uint32_t graph_id) {
  return graph_outputs_.find(graph_id) != graph_outputs_.end() ? graph_outputs_.at(graph_id)
                                                               : std::vector<mindspore::MSTensor>();
}

bool GeGraphExecutor::CreateAsCustomFuncGraph(const FuncGraphPtr &func_graph,
                                              const std::map<std::string, std::string> &graph_options) {
  Buffer buffer;
  auto files = ReadFileNames(build_cache_dir_);
  for (auto &file : files) {
    if (file.find(".om") != std::string::npos && file.find(graph_name_) != std::string::npos) {
      auto om_path = build_cache_dir_ + "/" + file;
      buffer = ReadFile(om_path);
      break;
    }
  }
  if (buffer.DataSize() == 0 || buffer.Data() == nullptr) {
    MS_LOG(ERROR) << "Failed to read model buffer file, model cache " << build_cache_dir_;
    return false;
  }
  std::map<std::string, ValuePtr> attr_map;
  SetOptionsIntoOfflineModel(session_options_, &attr_map);
  std::vector<std::string> ref_datas;
  std::transform(ref_data_infos_.begin(), ref_data_infos_.end(), std::back_inserter(ref_datas),
                 [](auto &item) { return item.name; });
  DynKVCacheSaveInfo save_info;
  save_info.seq_length_dyn = dyn_kv_cache_info_.seq_length_dyn;
  save_info.batch_size_dyn = dyn_kv_cache_info_.batch_size_dyn;
  save_info.kv_cache_layout = dyn_kv_cache_info_.kv_cache_layout;

  if (!CustomAscendUtils::CreateCustomFuncGraph(func_graph, buffer, graph_name_, attr_map, ref_datas, save_info)) {
    MS_LOG(ERROR) << "Create custom func graph failed";
    return false;
  }
  return true;
}

bool GeGraphExecutor::OfflineBuildGraph(const FuncGraphPtr &graph) {
  if (ref_mode_flag_ == backend::ge_backend::RefModeFlag::kRefModeNone) {
    MS_LOG(INFO) << "parameter_as_refdata in ascend_context is none, skip offline build graph";
    return true;
  }
  MS_LOG(INFO) << "Set offline mode";
  std::map<std::string, std::string> extra_session_options;
  if (!SetOfflineBuildModelCacheDir(&extra_session_options)) {
    return false;
  }
  if (!CreateSession(extra_session_options)) {
    MS_LOG(ERROR) << "Failed to create ge session";
    return false;
  }

  uint32_t graph_id = 0;
  std::map<std::string, std::string> ge_options;
  GetGeGraphOptions(graph, &ge_options);
  auto df_graph = CompileGraphCommon(graph, &ge_options);
  if (df_graph == nullptr) {
    MS_LOG(ERROR) << "Input param graph is nullptr.";
    return false;
  }
  if (!AddGraph(df_graph, ge_options, &graph_id)) {
    MS_LOG(ERROR) << "Failed to add compute graph.";
    return false;
  }
  compute_graph_id_list_.push_back(graph_id);
  MS_LOG(INFO) << "Call GE CompileGraph start, graph id " << graph_id;
  ge::Status ret = ge_session_->CompileGraph(graph_id);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE CompileGraph Failed: " << ge::GEGetErrorMsg();
    return false;
  }
  MS_LOG(INFO) << "Call GE CompileGraph end, graph id " << graph_id;
  if (!CreateAsCustomFuncGraph(graph, ge_options)) {
    MS_LOG(ERROR) << "Failed to CreateAsCustomFuncGraph";
    return false;
  }
  return true;
}

std::shared_ptr<GeTensor> GeGraphExecutor::ConvertMSTensor(const std::shared_ptr<MSTensor> &tensor,
                                                           const std::string &format, bool copy) {
  // get tensor data type size
  MS_EXCEPTION_IF_NULL(tensor);
  auto data_type = tensor->DataType();
  if (data_type == DataType::kObjectTypeString) {
    MS_LOG(ERROR) << "The MSTensor data type \'string\' is unsupported.";
  }
  size_t type_size = GetDataTypeSize(static_cast<const TypeId>(data_type));
  if (type_size == kErrorSize) {
    MS_LOG(ERROR) << "The MSTensor data type size is wrong, type size is: " << type_size;
    return nullptr;
  }

  // get tensor buff size
  size_t data_buff_size = tensor->DataSize();
  if (data_buff_size == 0) {
    MS_LOG(INFO) << "The MSTensor data buff size is 0.";
  }
  // create ge tensor
  auto desc = device::ascend::TransformUtil::GetGeTensorDesc(tensor->Shape(),
                                                             static_cast<const TypeId>(tensor->DataType()), format);
  if (desc == nullptr) {
    MS_LOG(ERROR) << "Failed to get Tensor Desc";
    return nullptr;
  }
  void *device_data = tensor->GetDeviceData();
  if (device_data != nullptr) {
    desc->SetPlacement(::ge::kPlacementDevice);
  }
  std::shared_ptr<GeTensor> tensor_ptr = std::make_shared<GeTensor>(*desc);
  if (tensor_ptr == nullptr) {
    MS_LOG(ERROR) << "Failed to convert MSTensor to Ge Tensor!";
    return nullptr;
  }
  if (device_data != nullptr) {
    MSTensorRel rel(tensor);
    auto ret =
      tensor_ptr->SetData(static_cast<uint8_t *>(device_data), data_buff_size, [rel](uint8_t *) -> void { rel.Rel(); });
    if (ret != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Failed to bind device data to ge tensor (zero-copy), data size " << data_buff_size;
      return nullptr;
    }
    return tensor_ptr;
  }
  if (copy) {
    auto ret = tensor_ptr->SetData(static_cast<uint8_t *>(tensor->MutableData()), data_buff_size);
    if (ret != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call ge::Tensor SetData(const uint8_t*, size), data size " << data_buff_size;
      return nullptr;
    }
  } else {
    MSTensorRel rel(tensor);
    auto ret = tensor_ptr->SetData(static_cast<uint8_t *>(tensor->MutableData()), data_buff_size,
                                   [rel](uint8_t *) -> void { rel.Rel(); });
    if (ret != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call ge::Tensor SetData(uint8_t*, size, DeleteFunc), data size " << data_buff_size;
      return nullptr;
    }
  }
  MS_LOG(DEBUG) << "Convert MSTensor to Ge Tensor success!";
  return tensor_ptr;
}

static std::shared_ptr<LiteGraphExecutor> GeGraphExecutorCreator(const std::shared_ptr<Context> &ctx,
                                                                 const ConfigInfos &config_infos) {
  auto ge_executor = std::make_shared<GeGraphExecutor>(ctx, config_infos);
  MS_CHECK_TRUE_RET(ge_executor != nullptr, nullptr);
  auto ret = ge_executor->Init();
  if (!ret) {
    MS_LOG(ERROR) << "Failed to init GeGraphExecutor";
    return nullptr;
  }
  return ge_executor;
}

REG_DELEGATE(kAscend, kProviderGe, GeGraphExecutorCreator)
}  // namespace mindspore
