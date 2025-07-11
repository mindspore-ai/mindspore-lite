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
#include "mindspore-lite/minddata/dataset/engine/consumers/tree_consumer.h"
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mindspore-lite/minddata/dataset/engine/datasetops/data_queue_op.h"
#include "mindspore-lite/minddata/dataset/engine/opt/pre/getter_pass.h"
#include "mindspore-lite/minddata/dataset/engine/perf/auto_tune.h"
#include "mindspore-lite/minddata/dataset/engine/perf/info_collector.h"
#include "mindspore-lite/minddata/dataset/engine/perf/profiling.h"
#include "mindspore-lite/minddata/dataset/engine/tree_adapter.h"
#ifndef ENABLE_ANDROID
#include "mindspore-lite/minddata/dataset/kernels/data/data_utils.h"
#include "minddata/mindrecord/include/shard_header.h"
#include "minddata/mindrecord/include/shard_index_generator.h"
#include "minddata/mindrecord/include/shard_writer.h"
#endif
#ifdef WITH_BACKEND
#include "utils/ms_context.h"
#endif

namespace mindspore {
namespace dataset {
using ProfilingRegistrationState = ProfilingManager::ProfilingRegistrationState;
// TreeConsumer
TreeConsumer::TreeConsumer() : TreeConsumer(1) {}

TreeConsumer::TreeConsumer(int32_t num_epochs) : num_epochs_(num_epochs) {
  tree_adapter_ = std::make_unique<TreeAdapter>();
}

Status TreeConsumer::Init(const std::shared_ptr<DatasetNode> &root) {
  RETURN_IF_NOT_OK(tree_adapter_->Compile(root));
  profiling_manager_ = GlobalContext::profiling_manager();
  RETURN_IF_NOT_OK(RegisterProfilingManager());
  return Status::OK();
}

Status TreeConsumer::Init(const std::shared_ptr<DatasetNode> &root, int64_t global_step, int64_t dataset_size) {
  MS_LOG(WARNING) << "TreeConsumer does not support initializing from intermediate epoch or step, change to "
                     "initialize from the beginning.";
  return Init(root);
}

Status TreeConsumer::Terminate() {
  if (tree_adapter_->AllTasks() != nullptr) {
    return tree_adapter_->AllTasks()->ServiceStop();
  }
  return Status::OK();
}

Status IteratorConsumer::RegisterProfilingManager() {
  ExecutionTree *tree = nullptr;
#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
  if (tree_adapter_->tree_) {
#endif
    tree = tree_adapter_->tree_.get();
#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
  } else {
    tree = tree_adapter_->receive_tree_.get();
  }
#endif
  auto profiler_state = profiling_manager_->GetProfilerTreeState(tree);
  // This should never happen
  CHECK_FAIL_RETURN_UNEXPECTED(profiler_state != ProfilingManager::kEnabledTreeRegistered,
                               "Something went wrong. Current tree is already registered with the MD Profiler");
  if (profiler_state == ProfilingManager::kEnabledDifferentTreeRegistered && profiling_manager_->IsProfiling()) {
    MS_LOG(WARNING) << "Dataset Profiling is already enabled for a different data pipeline.";
  } else if (profiler_state == ProfilingManager::kEnabledDifferentTreeRegistered &&
             profiling_manager_->IsAutotuning()) {
    MS_LOG(WARNING) << "AutoTune for dataset is already enabled for a different data pipeline.";
  } else if (profiler_state == ProfilingRegistrationState::kEnabledTreeNotRegistered) {
    // Profiling infrastructures need to be initialized before Op launching
    // Setup profiling manager
    RETURN_IF_NOT_OK(profiling_manager_->RegisterTree(tree));
    // dataset_iterator node is used for graph mode
    std::shared_ptr<Tracing> iterator_tracing = std::make_shared<DatasetIteratorTracing>();
    RETURN_IF_NOT_OK(profiling_manager_->RegisterTracingNode(iterator_tracing));
    RETURN_IF_NOT_OK(tree_adapter_->SetProfilingManagerPtr(profiling_manager_, iterator_tracing));
    // Launch Monitor Thread
    RETURN_IF_NOT_OK(profiling_manager_->LaunchMonitor());
  } else {
    MS_LOG(INFO) << "Unable to register this tree with ProfilingManager.";
  }
  return Status::OK();
}

Status ToDevice::RegisterProfilingManager() {
  ExecutionTree *tree = nullptr;
#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
  if (tree_adapter_->tree_) {
#endif
    tree = tree_adapter_->tree_.get();
#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
  } else {
    tree = tree_adapter_->receive_tree_.get();
  }
#endif
  auto profiler_state = profiling_manager_->GetProfilerTreeState(tree);
  // This should never happen
  CHECK_FAIL_RETURN_UNEXPECTED(profiler_state != ProfilingManager::kEnabledTreeRegistered,
                               "Something went wrong. Current tree is already registered with the MD Profiler");
  if (profiler_state == ProfilingManager::kEnabledDifferentTreeRegistered && profiling_manager_->IsProfiling()) {
    MS_LOG(WARNING) << "Dataset Profiling is already enabled for a different data pipeline.";
  } else if (profiler_state == ProfilingManager::kEnabledDifferentTreeRegistered &&
             profiling_manager_->IsAutotuning()) {
    MS_LOG(WARNING) << "AutoTune for dataset is already enabled for a different data pipeline.";
  } else if (profiler_state == ProfilingRegistrationState::kEnabledTreeNotRegistered) {
    // Profiling infrastructures need to be initialized before Op launching
    // Setup profiling manager
    RETURN_IF_NOT_OK(profiling_manager_->RegisterTree(tree));
    // device_queue node is used for graph mode
    std::shared_ptr<Tracing> device_queue_tracing = std::make_shared<DeviceQueueTracing>();
    RETURN_IF_NOT_OK(profiling_manager_->RegisterTracingNode(device_queue_tracing));
    RETURN_IF_NOT_OK(tree_adapter_->SetProfilingManagerPtr(profiling_manager_));
    // Launch Monitor Thread
    RETURN_IF_NOT_OK(profiling_manager_->LaunchMonitor());
  } else {
    MS_LOG(INFO) << "Unable to register this tree with ProfilingManager.";
  }
  return Status::OK();
}

Status TreeConsumer::RegisterProfilingManager() {
  if (profiling_manager_->IsProfiling()) {
    return {StatusCode::kMDUnexpectedError, "Dataset Profiling is not supported for this kind of dataset."};
  }
  return Status::OK();
}

Status TreeConsumer::InitAutoTune() {
  auto profiler_state = profiling_manager_->GetProfilerTreeState(tree_adapter_->tree_.get());
  if (profiler_state == ProfilingRegistrationState::kNotEnabled) {
    // Init ProfilingManager to `Enable` it.
    RETURN_IF_NOT_OK(profiling_manager_->Init(true));
    // Register this tree
    RETURN_IF_NOT_OK(RegisterProfilingManager());
    // Start Profiler
    RETURN_IF_NOT_OK(profiling_manager_->Start());
    // AutoTune object and thread init
    autotune_ = std::make_unique<AutoTune>(this->tree_adapter_.get(), GetProfilingManager());
    RETURN_IF_NOT_OK(autotune_->LaunchThread());
  } else if (profiler_state == ProfilingManager::kEnabledDifferentTreeRegistered && profiling_manager_->IsProfiling()) {
    MS_LOG(WARNING) << "Cannot enable AutoTune for the current data pipeline as Dataset Profiling is enabled for "
                       "another data pipeline.";
  } else if (profiler_state == ProfilingManager::kEnabledDifferentTreeRegistered &&
             profiling_manager_->IsAutotuning()) {
    MS_LOG(WARNING)
      << "Cannot enable AutoTune for the current data pipeline as it is already enabled for another data pipeline.";
  } else if (profiler_state == ProfilingManager::kEnabledTreeRegistered && profiling_manager_->IsProfiling()) {
    MS_LOG(WARNING)
      << "Cannot enable AutoTune for the current data pipeline as Dataset Profiling is already enabled for the "
         "current data pipeline.";
  } else {
    MS_LOG(WARNING) << "Cannot enable AutoTune for the current data pipeline.";
  }
  return Status::OK();
}

std::string TreeConsumer::GetOffload() { return (tree_adapter_->GetOffloadJson()).dump(); }

// IteratorConsumer
Status IteratorConsumer::Init(const std::shared_ptr<DatasetNode> &root, int64_t global_step, int64_t dataset_size) {
  if (global_step != 0) {
    tree_adapter_ = std::make_unique<TreeAdapter>(TreeAdapter::UsageFlag::kDeReset);
  }
  RETURN_IF_NOT_OK(tree_adapter_->Compile(root, num_epochs_, global_step, dataset_size, false));
  profiling_manager_ = GlobalContext::profiling_manager();
  if (profiling_manager_->IsProfiling()) {
    // Init has been called already
    RETURN_IF_NOT_OK(RegisterProfilingManager());
  }
  if (GlobalContext::config_manager()->enable_autotune()) {
    RETURN_IF_NOT_OK(InitAutoTune());
  }
  return Status::OK();
}

Status IteratorConsumer::GetNextAsVector(std::vector<TensorPtr> *const out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  uint64_t start_time = GetSyscnt();
  out->clear();
  TensorRow res;
  RETURN_IF_NOT_OK(tree_adapter_->GetNext(&res));

  // Return empty vector if there's no data
  if (res.empty()) {
    RETURN_IF_NOT_OK(
      CollectPipelineInfo("IteratorConsumer", "GetNextAsVector", start_time, {{"TensorRowFlags", res.FlagName()}}));
    return Status::OK();
  }

  if (res.Timer()->Enabled()) {
    // VL_MD is 10900
    VLOG_MD(res.Timer()->Summary());
  }

  // Filter meta column
  std::vector<size_t> to_keep_indices;
  for (const auto &colMap : tree_adapter_->GetColumnNameMap()) {
    std::string column_name = colMap.first;
    // Need to filter meta column start with kDftMetaColumnPrefix
    size_t pos = column_name.find(kDftMetaColumnPrefix);
    if (pos != std::string::npos && pos == 0) {
      continue;
    }
    to_keep_indices.push_back(colMap.second);
  }
  if (to_keep_indices.empty()) {
    std::string err_msg = "No effective column found, maybe all columns are meta column and will be filtered. ";
    err_msg += "If you want to output meta column please rename column name to a new one which is not start with ";
    err_msg += "\"" + std::string(kDftMetaColumnPrefix) + "\"";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::sort(to_keep_indices.begin(), to_keep_indices.end());
  (void)std::transform(to_keep_indices.begin(), to_keep_indices.end(), std::back_inserter(*out),
                       [&res](const auto &it) { return std::move(res[it]); });
  RETURN_IF_NOT_OK(CollectPipelineInfo("IteratorConsumer", "GetNextAsVector", start_time));
  return Status::OK();
}

Status IteratorConsumer::GetNextAsMap(std::unordered_map<std::string, TensorPtr> *const out_map) {
  RETURN_UNEXPECTED_IF_NULL(out_map);
  uint64_t start_time = GetSyscnt();

  out_map->clear();
  TensorRow res;
  RETURN_IF_NOT_OK(tree_adapter_->GetNext(&res));

  // Return empty map if there's no data
  if (res.empty()) {
    RETURN_IF_NOT_OK(
      CollectPipelineInfo("IteratorConsumer", "GetNextAsMap", start_time, {{"TensorRowFlags", res.FlagName()}}));
    return Status::OK();
  }

  if (res.Timer()->Enabled()) {
    // VL_MD is 10900
    VLOG_MD(res.Timer()->Summary());
  }

  // Populate the out map from the row and return it
  for (const auto &colMap : tree_adapter_->GetColumnNameMap()) {
    std::string column_name = colMap.first;
    // Need to filter meta column start with kDftMetaColumnPrefix
    size_t pos = column_name.find(kDftMetaColumnPrefix);
    if (pos != std::string::npos && pos == 0) {
      continue;
    }
    (*out_map)[colMap.first] = std::move(res[colMap.second]);
  }
  if (out_map->empty()) {
    std::string err_msg = "No effective column found, maybe all columns are meta column and will be filtered. ";
    err_msg += "If you want to output meta column please rename column name to a new one which is not start with ";
    err_msg += "\"" + std::string(kDftMetaColumnPrefix) + "\"";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  RETURN_IF_NOT_OK(CollectPipelineInfo("IteratorConsumer", "GetNextAsMap", start_time));
  return Status::OK();
}

Status IteratorConsumer::GetNextAsOrderedPair(std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> *const vec) {
  CHECK_FAIL_RETURN_UNEXPECTED(vec != nullptr && vec->empty(), "vec is null or non-empty.");
  uint64_t start_time = GetSyscnt();

  TensorRow curr_row;

  RETURN_IF_NOT_OK(tree_adapter_->GetNext(&curr_row));

  // Return empty pair if there's no data
  if (curr_row.empty()) {
    RETURN_IF_NOT_OK(CollectPipelineInfo("IteratorConsumer", "GetNextAsOrderedPair", start_time,
                                         {{"TensorRowFlags", curr_row.FlagName()}}));
    return Status::OK();
  }

  size_t num_cols = curr_row.size();  // num_cols is non-empty.
  // order the column names according to their ids
  if (column_order_.empty()) {
    for (const auto &itr : tree_adapter_->GetColumnNameMap()) {
      int32_t ind = itr.second;
      CHECK_FAIL_RETURN_UNEXPECTED(ind < num_cols && ind >= 0, "column id out of bounds.");
      // Need to filter meta column start with kDftMetaColumnPrefix
      size_t pos = itr.first.find(kDftMetaColumnPrefix);
      if (pos != std::string::npos && pos == 0) {
        continue;
      }
      column_order_[ind] = itr.first;
    }
  }

  if (column_order_.empty()) {
    std::string err_msg = "No effective column found, maybe all columns are meta column and will be filtered. ";
    err_msg += "If you want to output meta column please rename column name to a new one which is not start with ";
    err_msg += "\"" + std::string(kDftMetaColumnPrefix) + "\"";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  vec->reserve(column_order_.size());

  std::transform(column_order_.begin(), column_order_.end(), std::back_inserter(*vec),
                 [curr_row](const auto &col) { return std::make_pair(col.second, curr_row[col.first]); });
  RETURN_IF_NOT_OK(CollectPipelineInfo("IteratorConsumer", "GetNextAsOrderedPair", start_time));
  return Status::OK();
}

// ToDevice
Status ToDevice::Init(const std::shared_ptr<DatasetNode> &root, int64_t global_step, int64_t dataset_size) {
  if (global_step != 0) {
    tree_adapter_ = std::make_unique<TreeAdapter>(TreeAdapter::UsageFlag::kDeReset);
  }
  RETURN_IF_NOT_OK(tree_adapter_->Compile(root, num_epochs_, global_step, dataset_size, true));
  profiling_manager_ = GlobalContext::profiling_manager();
  if (profiling_manager_->IsProfiling()) {
    // Init has been called already
    RETURN_IF_NOT_OK(RegisterProfilingManager());
  }
  if (GlobalContext::config_manager()->enable_autotune()) {
    RETURN_IF_NOT_OK(InitAutoTune());
  }
  return Status::OK();
}

Status ToDevice::Send() {
  RETURN_IF_NOT_OK(tree_adapter_->Launch());
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
  CHECK_FAIL_RETURN_UNEXPECTED(root != nullptr, "Root is a nullptr.");
  return Status::OK();
}

Status ToDevice::Continue() {
  // tree_.root() must be DataQueueOp
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
  CHECK_FAIL_RETURN_UNEXPECTED(root != nullptr, "Root is a nullptr.");
  DataQueueOp *op = dynamic_cast<DataQueueOp *>(root.get());
  CHECK_FAIL_RETURN_UNEXPECTED(op != nullptr, "ContinueSend only supported by DataQueueOp");
  op->ContinueSend();
  return Status::OK();
}

Status ToDevice::Stop() {
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
  CHECK_FAIL_RETURN_UNEXPECTED(root != nullptr, "Root is a nullptr.");
  DataQueueOp *op = dynamic_cast<DataQueueOp *>(root.get());
  CHECK_FAIL_RETURN_UNEXPECTED(op != nullptr, "StopSend only supported by DataQueueOp");
  op->StopSend();

  return Status::OK();
}

Status ToDevice::GetDataInfo(std::vector<DataType> *const types, std::vector<TensorShape> *const shapes) {
  RETURN_UNEXPECTED_IF_NULL(types);
  RETURN_UNEXPECTED_IF_NULL(shapes);
  // tree_.root() must be DataQueueOp
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
  CHECK_FAIL_RETURN_UNEXPECTED(root != nullptr, "Root is a nullptr.");
  DataQueueOp *op = dynamic_cast<DataQueueOp *>(root.get());
  CHECK_FAIL_RETURN_UNEXPECTED(op != nullptr, "GetDataInfo only supported by DataQueueOp");
  DATA_INFO data_info;
  RETURN_IF_NOT_OK(op->GetDataInfo(&data_info));
  for (auto el : data_info) {
    types->push_back(el.first);
    shapes->push_back(el.second);
  }
  return Status::OK();
}

Status ToDevice::GetMbufQueueSize(size_t *queue_size) {
  RETURN_UNEXPECTED_IF_NULL(queue_size);
  // tree_.root() must be DataQueueOp
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
  CHECK_FAIL_RETURN_UNEXPECTED(root != nullptr, "Root is a nullptr.");
  DataQueueOp *op = dynamic_cast<DataQueueOp *>(root.get());
  CHECK_FAIL_RETURN_UNEXPECTED(op != nullptr, "GetMbufQueueSize only supported by DataQueueOp");
  RETURN_IF_NOT_OK(op->GetMbufQueueSize(queue_size));
  return Status::OK();
}

Status ToDevice::GetSendInfo(std::vector<std::vector<double>> *send_info) {
  RETURN_UNEXPECTED_IF_NULL(send_info);
  // tree_.root() must be DataQueueOp
  std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
  CHECK_FAIL_RETURN_UNEXPECTED(root != nullptr, "Root is a nullptr.");
  DataQueueOp *op = dynamic_cast<DataQueueOp *>(root.get());
  CHECK_FAIL_RETURN_UNEXPECTED(op != nullptr, "GetSendInfo only supported by DataQueueOp");
  DATA_INFO data_info;
  *send_info = op->GetSendInfo();
  return Status::OK();
}

Status ToDevice::Terminate() {
#ifdef WITH_BACKEND
  RETURN_UNEXPECTED_IF_NULL(MsContext::GetInstance());
  if (MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
    CHECK_FAIL_RETURN_UNEXPECTED(root != nullptr, "Root is a nullptr.");
    DataQueueOp *op = dynamic_cast<DataQueueOp *>(root.get());
    CHECK_FAIL_RETURN_UNEXPECTED(op != nullptr, "StopSend only supported by DataQueueOp");
    op->StopWaiting();
  }
#endif
  return TreeConsumer::Terminate();
}

Status TreeConsumer::Reset(int64_t step, const int64_t dataset_size) {
  MS_LOG(INFO) << "Resetting TreeConsumer";

  std::string unique_id = "";
#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
  if (tree_adapter_->tree_) {
#endif
    unique_id = tree_adapter_->tree_->GetUniqueId();
#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
  } else {
    unique_id = tree_adapter_->receive_tree_->GetUniqueId();
  }
#endif
  MS_LOG(INFO) << "Terminating pipeline with UUID:" << unique_id;
  std::shared_ptr<DatasetNode> old_root = tree_adapter_->input_ir_;
  RETURN_IF_NOT_OK(this->Stop());
  RETURN_IF_NOT_OK(this->Terminate());
#ifdef WITH_BACKEND
  RETURN_UNEXPECTED_IF_NULL(MsContext::GetInstance());
  if (MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice) {
    // clear the device if GPU is used.
    std::shared_ptr<DatasetOp> root = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
    CHECK_FAIL_RETURN_UNEXPECTED(root != nullptr, "Root is a nullptr.");
    DataQueueOp *op = dynamic_cast<DataQueueOp *>(root.get());
    if (op != nullptr) {
      MS_LOG(INFO) << "Clearing the GPU device";
      RETURN_IF_NOT_OK(op->ClearDevice());
    }
  }
#endif
  tree_adapter_ = std::make_unique<TreeAdapter>(TreeAdapter::UsageFlag::kDeReset);
  RETURN_IF_NOT_OK(tree_adapter_->Compile(old_root, num_epochs_, step, dataset_size, true));
  RETURN_IF_NOT_OK(tree_adapter_->Launch());
  MS_LOG(INFO) << "Launched a new pipeline after reset. UUID: " << tree_adapter_->tree_->GetUniqueId();
  std::shared_ptr<DatasetOp> root2 = std::shared_ptr<DatasetOp>(tree_adapter_->GetRoot());
  CHECK_FAIL_RETURN_UNEXPECTED(root2 != nullptr, "Root is a nullptr.");
  return Status::OK();
}

#ifndef ENABLE_ANDROID
// SaveToDisk
Status SaveToDisk::ValidateParams() {
  if (dataset_path_.empty()) {
    std::string err = "SaveToDisk failed, dataset_path must not be empty";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err);
  }
  Path dir(dataset_path_);
  if (dir.IsDirectory()) {
    std::string err = "SaveToDisk failed, dataset_path must not be a directory";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err);
  }
  std::string real_path;
  if (dir.ParentPath().empty()) {
    dir = Path(".") / dir;
  }
  if (Path::RealPath(dir.ParentPath(), real_path).IsError()) {
    std::string err_msg = "SaveToDisk failed, can not get real dataset path: " + dir.ParentPath();
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (access(dir.ParentPath().c_str(), R_OK) == -1) {
    std::string err_msg = "SaveToDisk failed, no access to specified dataset path: " + dataset_path_;
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (num_files_ <= 0 || num_files_ > 1000) {
    std::string err = "SaveToDisk failed, num_files must between 1 and 1000, but got " + std::to_string(num_files_);
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err);
  }
  if (dataset_type_ != "mindrecord") {
    std::string err = "SaveToDisk failed, only \"mindrecord\" dataset format is supported, but got " + dataset_type_;
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err);
  }
  return Status::OK();
}

Status SaveToDisk::Save() {
  uint64_t start_time = GetSyscnt();
  std::vector<std::string> file_names;
  if (num_files_ == 1) {
    file_names.push_back(dataset_path_);
  } else {
    for (int32_t i = 0; i < num_files_; i++) {
      file_names.push_back(dataset_path_ + std::to_string(i));
    }
  }

  auto mr_header = std::make_shared<mindrecord::ShardHeader>();
  auto mr_writer = std::make_unique<mindrecord::ShardWriter>();
  std::vector<std::string> blob_fields;
  RETURN_IF_NOT_OK(mindrecord::ShardWriter::Initialize(&mr_writer, file_names));

  std::unordered_map<std::string, int32_t> column_name_id_map;
  for (auto el : tree_adapter_->GetColumnNameMap()) {
    std::string column_name = el.first;
    (void)std::transform(column_name.begin(), column_name.end(), column_name.begin(),
                         [](unsigned char c) { return ispunct(c) ? '_' : c; });
    column_name_id_map[column_name] = el.second;
  }

  TensorRow row;
  uint64_t mr_schema_id = 0;
  bool first_loop = true;  // build schema in first loop
  auto PreTensorRowShapes = std::map<std::string, std::vector<int>>();

  do {
    nlohmann::json row_raw_data;
    std::map<std::string, std::unique_ptr<std::vector<uint8_t>>> row_bin_data;
    RETURN_IF_NOT_OK(tree_adapter_->GetNext(&row));
    if (row.empty()) {
      break;
    }
    RETURN_IF_NOT_OK(CheckTensorRowShapes(column_name_id_map, row, &PreTensorRowShapes));
    if (first_loop) {
      nlohmann::json mr_json;
      std::vector<std::string> index_fields;
      RETURN_IF_NOT_OK(FetchMetaFromTensorRow(column_name_id_map, row, &mr_json, &index_fields));
      MS_LOG(INFO) << "Schema of saved mindrecord: " << mr_json.dump();
      RETURN_IF_NOT_OK(
        mindrecord::ShardHeader::Initialize(&mr_header, mr_json, index_fields, blob_fields, mr_schema_id));
      RETURN_IF_NOT_OK(mr_writer->SetShardHeader(mr_header));
      first_loop = false;
    }
    // construct data
    if (!row.empty()) {  // write data
      RETURN_IF_NOT_OK(FetchDataFromTensorRow(row, column_name_id_map, &row_raw_data, &row_bin_data));
      std::shared_ptr<std::vector<uint8_t>> output_bin_data;
      RETURN_IF_NOT_OK(mr_writer->MergeBlobData(blob_fields, row_bin_data, &output_bin_data));
      std::map<std::uint64_t, std::vector<nlohmann::json>> raw_data;
      raw_data.insert(
        std::pair<uint64_t, std::vector<nlohmann::json>>(mr_schema_id, std::vector<nlohmann::json>{row_raw_data}));
      std::vector<std::vector<uint8_t>> bin_data;
      if (output_bin_data != nullptr) {
        bin_data.emplace_back(*output_bin_data);
      }
      RETURN_IF_NOT_OK(mr_writer->WriteRawData(raw_data, bin_data));
    }
  } while (!row.empty());

  RETURN_IF_NOT_OK(mr_writer->Commit());
  RETURN_IF_NOT_OK(mindrecord::ShardIndexGenerator::Finalize(file_names));
  RETURN_IF_NOT_OK(CollectPipelineInfo("SaveToDisk", "Save", start_time));
  return Status::OK();
}

template <typename T>
bool SaveToDisk::map_compare(T const &lhs, T const &rhs) {
  return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

Status SaveToDisk::CheckTensorRowShapes(const std::unordered_map<std::string, int32_t> &column_name_id_map,
                                        const TensorRow &row,
                                        std::map<std::string, std::vector<int>> *PreTensorRowShapes_ptr) {
  std::map<std::string, std::vector<int>> CurrTensorRowShapes;
  for (auto &col : column_name_id_map) {
    auto idx = col.second;
    auto column_name = col.first;
    auto &tensor = row[idx];
    auto column_type = tensor->type();
    auto column_shape = tensor->shape();

    auto shapes = column_shape.AsVector();
    std::vector<int> mr_shape(shapes.begin(), shapes.end());

    if (mr_shape.empty() || mr_shape.size() == 1) {
      continue;  // ignore scalar and one dimension tensor
    }
    std::string mr_type;
    std::string el = column_type.ToString();
    if (mindrecord::kTypesMap.find(el) == mindrecord::kTypesMap.end()) {
      std::string err_msg("Invalid type, unsupported data type: " + el);
      RETURN_STATUS_UNEXPECTED(err_msg);
    } else {
      mr_type = mindrecord::kTypesMap.at(el);
    }
    if (mr_type == "bytes" || mr_type == "string") {
      continue;
    }
    mr_shape.erase(mr_shape.begin());  // ignore the first dimension
    CurrTensorRowShapes[column_name] = mr_shape;
  }
  if (PreTensorRowShapes_ptr->empty()) {
    *PreTensorRowShapes_ptr = CurrTensorRowShapes;
    return Status::OK();
  }
  auto res = map_compare(*PreTensorRowShapes_ptr, CurrTensorRowShapes);
  CHECK_FAIL_RETURN_UNEXPECTED(res,
                               "Tensor with dynamic shape do not currently support saving. Except for the shape of "
                               "dimension 0, the other dimension shapes must be fixed. "
                               "You can reshape the Tensor to a fixed shape before saving.");
  return Status::OK();
}

Status SaveToDisk::FetchMetaFromTensorRow(const std::unordered_map<std::string, int32_t> &column_name_id_map,
                                          const TensorRow &row, nlohmann::json *schema,
                                          std::vector<std::string> *index_fields) {
  if (schema == nullptr) {
    RETURN_STATUS_UNEXPECTED("schema can not be nullptr.");
  }
  if (index_fields == nullptr) {
    RETURN_STATUS_UNEXPECTED("index_fields can not be nullptr.");
  }
  if (column_name_id_map.empty()) {
    RETURN_STATUS_UNEXPECTED("column_name_id_map can not be nullptr..");
  }
  nlohmann::json dataset_schema;
  for (auto &col : column_name_id_map) {
    auto idx = col.second;
    auto column_name = col.first;
    auto &tensor = row[idx];
    auto column_type = tensor->type();
    auto column_shape = tensor->shape();

    std::string mr_type;
    auto shapes = column_shape.AsVector();
    std::vector<int> mr_shape(shapes.begin(), shapes.end());
    std::string el = column_type.ToString();
    dataset_schema[column_name] = el;
    if (mindrecord::kTypesMap.find(el) == mindrecord::kTypesMap.end()) {
      std::string err_msg("Invalid type, unsupported data type: " + el);
      RETURN_STATUS_UNEXPECTED(err_msg);
    } else {
      mr_type = mindrecord::kTypesMap.at(el);
    }
    if (mr_shape.empty()) {
      (*schema)[column_name] = {{"type", mr_type}};
    } else {
      if (mr_type == "string") {  // mindrecord can not support string with shape.
        std::string err_msg("Invalid data, mindrecord can not support multi-dimensional string tensor.");
        RETURN_STATUS_UNEXPECTED(err_msg);
      }
      if (mr_type == "bytes") {  // ignore shape of bytes in minrecord
        (*schema)[column_name] = {{"type", mr_type}};
      } else {
        mr_shape[0] = -1;  // make first dimension -1
        (*schema)[column_name] = {{"type", mr_type}, {"shape", mr_shape}};
      }
    }
    if (mr_type == "bytes" || !mr_shape.empty()) {
      continue;
    }
    index_fields->emplace_back(column_name);  // candidate of index fields
  }
  MS_LOG(DEBUG) << "Schema of dataset: " << dataset_schema.dump();
  return Status::OK();
}

inline Status ValidateInputParams(nlohmann::json *row_raw_data,
                                  std::map<std::string, std::unique_ptr<std::vector<uint8_t>>> *row_bin_data,
                                  const std::unordered_map<std::string, int32_t> &column_name_id_map) {
  if (row_raw_data == nullptr) {
    RETURN_STATUS_UNEXPECTED("row_raw_data can not be nullptr.");
  }
  if (row_bin_data == nullptr) {
    RETURN_STATUS_UNEXPECTED("row_bin_data can not be nullptr.");
  }
  if (column_name_id_map.empty()) {
    RETURN_STATUS_UNEXPECTED("column_name_id_map can not be nullptr.");
  }
  return Status::OK();
}

Status SaveToDisk::FetchIntData(std::shared_ptr<Tensor> tensor, std::string column_name, nlohmann::json *row_raw_data,
                                std::unique_ptr<std::vector<uint8_t>> *data_ptr) {
  RETURN_UNEXPECTED_IF_NULL(row_raw_data);
  RETURN_UNEXPECTED_IF_NULL(data_ptr);
  auto column_type = tensor->type();
  Status s;
  if (column_type == DataType::DE_INT8) {
    std::unique_ptr<int32_t> data;
    std::unique_ptr<int8_t> dummy;
    RETURN_IF_NOT_OK(
      TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, data_ptr, &dummy, true));
    if (data != nullptr) {
      (*row_raw_data)[column_name] = std::move(*data);
    }
  } else if (column_type == DataType::DE_UINT8) {
    std::unique_ptr<int32_t> data;
    std::unique_ptr<uint8_t> dummy;
    RETURN_IF_NOT_OK(
      TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, data_ptr, &dummy, true));
    if (data != nullptr) {
      (*row_raw_data)[column_name] = std::move(*data);
    }
  } else if (column_type == DataType::DE_INT16) {
    std::unique_ptr<int32_t> data;
    std::unique_ptr<int16_t> dummy;
    RETURN_IF_NOT_OK(
      TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, data_ptr, &dummy, true));
    if (data != nullptr) {
      (*row_raw_data)[column_name] = std::move(*data);
    }
  } else if (column_type == DataType::DE_UINT16) {
    std::unique_ptr<int32_t> data;
    std::unique_ptr<uint16_t> dummy;
    RETURN_IF_NOT_OK(
      TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, data_ptr, &dummy, true));
    if (data != nullptr) {
      (*row_raw_data)[column_name] = std::move(*data);
    }
  } else if (column_type == DataType::DE_INT32) {
    std::unique_ptr<int32_t> data, dummy;
    RETURN_IF_NOT_OK(TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, data_ptr, &dummy));
    if (data != nullptr) {
      (*row_raw_data)[column_name] = std::move(*data);
    }
  } else if (column_type == DataType::DE_UINT32) {
    std::unique_ptr<int64_t> data;
    std::unique_ptr<uint32_t> dummy;
    RETURN_IF_NOT_OK(
      TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, data_ptr, &dummy, true));
    if (data != nullptr) {
      (*row_raw_data)[column_name] = std::move(*data);
    }
  } else if (column_type == DataType::DE_INT64 || column_type == DataType::DE_UINT64) {
    std::unique_ptr<int64_t> data, dummy;
    RETURN_IF_NOT_OK(TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, data_ptr, &dummy));
    if (data != nullptr) {
      (*row_raw_data)[column_name] = std::move(*data);
    }
  }

  return Status::OK();
}

Status SaveToDisk::FetchFloatData(std::shared_ptr<Tensor> tensor, std::string column_name, nlohmann::json *row_raw_data,
                                  std::unique_ptr<std::vector<uint8_t>> *data_ptr) {
  RETURN_UNEXPECTED_IF_NULL(row_raw_data);
  RETURN_UNEXPECTED_IF_NULL(data_ptr);
  auto column_type = tensor->type();
  Status s;
  if (column_type == DataType::DE_FLOAT16) {
    std::unique_ptr<float> data, dummy;
    std::shared_ptr<Tensor> out_tensor;
    RETURN_IF_NOT_OK(TypeCast(tensor, &out_tensor, DataType("float32")));
    RETURN_IF_NOT_OK(
      TransformTensor(out_tensor->GetBuffer(), out_tensor->shape(), out_tensor->Size(), &data, data_ptr, &dummy));
    if (data != nullptr) {
      (*row_raw_data)[column_name] = std::move(*data);
    }
  } else if (column_type == DataType::DE_FLOAT32) {
    std::unique_ptr<float> data, dummy;
    RETURN_IF_NOT_OK(TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, data_ptr, &dummy));
    if (data != nullptr) {
      (*row_raw_data)[column_name] = std::move(*data);
    }
  } else if (column_type == DataType::DE_FLOAT64) {
    std::unique_ptr<double> data, dummy;
    RETURN_IF_NOT_OK(TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, data_ptr, &dummy));
    if (data != nullptr) {
      (*row_raw_data)[column_name] = std::move(*data);
    }
  }
  return Status::OK();
}

Status SaveToDisk::FetchItemData(std::shared_ptr<Tensor> tensor, std::string column_name, nlohmann::json *row_raw_data,
                                 std::map<std::string, std::unique_ptr<std::vector<uint8_t>>> *row_bin_data) {
  RETURN_UNEXPECTED_IF_NULL(tensor);
  RETURN_UNEXPECTED_IF_NULL(row_raw_data);
  RETURN_UNEXPECTED_IF_NULL(row_bin_data);
  auto column_type = tensor->type();
  Status s;
  std::unique_ptr<std::vector<uint8_t>> data_ptr;
  if (column_type == DataType::DE_BOOL) {
    std::unique_ptr<int32_t> data;
    std::unique_ptr<int8_t> dummy;
    RETURN_IF_NOT_OK(
      TransformTensor(tensor->GetBuffer(), tensor->shape(), tensor->Size(), &data, &data_ptr, &dummy, true));
    if (data != nullptr) {
      (*row_raw_data)[column_name] = std::move(*data);
    }
  } else if (column_type == DataType::DE_INT8 || column_type == DataType::DE_UINT8 ||
             column_type == DataType::DE_INT16 || column_type == DataType::DE_UINT16 ||
             column_type == DataType::DE_INT32 || column_type == DataType::DE_UINT32 ||
             column_type == DataType::DE_INT64 || column_type == DataType::DE_UINT64) {
    s = FetchIntData(tensor, column_name, row_raw_data, &data_ptr);
    RETURN_IF_NOT_OK(s);
  } else if (column_type == DataType::DE_FLOAT16 || column_type == DataType::DE_FLOAT32 ||
             column_type == DataType::DE_FLOAT64) {
    s = FetchFloatData(tensor, column_name, row_raw_data, &data_ptr);
    RETURN_IF_NOT_OK(s);
  } else if (column_type == DataType::DE_BYTES) {
    std::unique_ptr<char> data;
    std::unique_ptr<char> dummy;
    CHECK_FAIL_RETURN_UNEXPECTED(tensor->shape().Rank() == 1 || tensor->shape().Rank() == 0,
                                 "Currently, multi-dimensional bytes cannot be converted to MindRecord.");
    if (tensor->shape().Rank() == 1) {
      CHECK_FAIL_RETURN_UNEXPECTED(tensor->shape().AsVector()[0] == 1,
                                   "Currently, multi-dimensional bytes cannot be converted to MindRecord.");
    }
    // current only support one bytes to mindrecord field
    uint32_t string_length = 0;
    RETURN_IF_NOT_OK(tensor->GetStringLength(&string_length));
    RETURN_IF_NOT_OK(TransformTensor(tensor->GetBuffer() + kOffsetSize * 2, TensorShape({1}), string_length, &data,
                                     &data_ptr, &dummy));
  } else if (column_type.IsString()) {
    std::string_view sv;
    RETURN_IF_NOT_OK(tensor->GetItemAt(&sv, {}));  // assume scalar string tensor
    std::string ss(sv);
    (*row_raw_data)[column_name] = std::move(ss);
  } else {
    RETURN_STATUS_UNEXPECTED("Invalid dtype, got unexpected type when casting data: " + column_type.ToString());
  }
  if (data_ptr != nullptr) {
    (*row_bin_data)[column_name] = std::move(data_ptr);
  }
  return Status::OK();
}

Status SaveToDisk::FetchDataFromTensorRow(const TensorRow &row,
                                          const std::unordered_map<std::string, int32_t> &column_name_id_map,
                                          nlohmann::json *row_raw_data,
                                          std::map<std::string, std::unique_ptr<std::vector<uint8_t>>> *row_bin_data) {
  RETURN_UNEXPECTED_IF_NULL(row_raw_data);
  RETURN_UNEXPECTED_IF_NULL(row_bin_data);
  Status s;
  s = ValidateInputParams(row_raw_data, row_bin_data, column_name_id_map);
  if (s.IsError()) {
    return s;
  }
  for (auto &col : column_name_id_map) {
    auto idx = col.second;
    auto column_name = col.first;
    auto &tensor = row[idx];
    s = FetchItemData(tensor, column_name, row_raw_data, row_bin_data);
    RETURN_IF_NOT_OK(s);
  }
  return Status::OK();
}

template <typename T, typename S>
Status SaveToDisk::TransformTensor(const unsigned char *src, const TensorShape &shape, const int64_t num_of_elements,
                                   std::unique_ptr<T> *data, std::unique_ptr<std::vector<uint8_t>> *data_ptr,
                                   std::unique_ptr<S> *s, bool need_convert) {
  // No need to check src since we support some scenarios that src is nullptr and num_of_elements is 0.
  RETURN_UNEXPECTED_IF_NULL(data);
  RETURN_UNEXPECTED_IF_NULL(data_ptr);
  RETURN_UNEXPECTED_IF_NULL(s);

  *data_ptr = std::make_unique<std::vector<uint8_t>>(num_of_elements * sizeof(T));
  if (need_convert) {
    auto tmp_ptr = std::make_unique<std::vector<uint8_t>>(num_of_elements * sizeof(S));
    (void)std::copy(src, src + sizeof(S) * num_of_elements, tmp_ptr->begin());
    auto s_ptr = reinterpret_cast<S *>(&(*(tmp_ptr->begin())));
    auto el = std::make_unique<T>();
    for (uint32_t i = 0; i < num_of_elements; ++i) {
      *el = *(s_ptr + i);
      auto t_ptr = reinterpret_cast<uint8_t *>(el.get());
      for (uint32_t j = 0; j < sizeof(T); ++j) {
        *((*data_ptr)->begin() + i * sizeof(T) + j) = *(t_ptr + j);
      }
    }
  } else {
    (void)std::copy(src, src + sizeof(T) * num_of_elements, (*data_ptr)->begin());
  }
  if (shape.empty()) {
    *data = std::make_unique<T>();
    auto t_ptr = reinterpret_cast<uint8_t *>((*data).get());
    for (uint32_t i = 0; i < sizeof(T); ++i) {
      *(t_ptr + i) = *((*data_ptr)->begin() + i);
    }
  }
  return Status::OK();
}
#endif

Status BuildVocabConsumer::Init(const std::shared_ptr<DatasetNode> &root) { return tree_adapter_->Compile(root, 1); }

Status BuildVocabConsumer::Start() {
  uint64_t start_time = GetSyscnt();
  // Getting one row would trigger building the vocab
  TensorRow row;
  RETURN_IF_NOT_OK(tree_adapter_->GetNext(&row));
  // The returned row would EOE which is an empty row
  CHECK_FAIL_RETURN_UNEXPECTED(row.empty(), "BuildVocab: The fetched row from BuildVocab should be an EOE.");
  RETURN_IF_NOT_OK(CollectPipelineInfo("BuildVocabConsumer", "Start", start_time));
  return Status::OK();
}
Status DatasetSizeGetter::GetDatasetSize(int64_t *size, bool estimate) {
  if (dataset_size_ == -1) {
    RETURN_IF_NOT_OK(root_->GetDatasetSize(shared_from_this(), estimate, size));
    dataset_size_ = *size;  // save the previous result
  }

  *size = dataset_size_;
  return Status::OK();
}

Status DatasetSizeGetter::Init(const std::shared_ptr<DatasetNode> &root) {
  root_ = root;
  return Status::OK();
}

Status DatasetSizeGetter::DryRun(const std::shared_ptr<DatasetNode> &ir_node, int64_t *dataset_size) {
  RETURN_UNEXPECTED_IF_NULL(dataset_size);
  std::shared_ptr<TreeAdapter> tree_adapter = std::make_shared<TreeAdapter>(TreeAdapter::UsageFlag::kDeGetter);
  tree_adapters_.push_back(tree_adapter);
  RETURN_IF_NOT_OK(tree_adapter->Compile(ir_node, 1));
  TensorRow row;
  RETURN_IF_NOT_OK(GetRow(tree_adapter, &row));
  int64_t row_cnt = 0;
  while (!row.empty()) {
    ++row_cnt;
    RETURN_IF_NOT_OK(GetRow(tree_adapter, &row));
  }
  *dataset_size = row_cnt;
  return Status::OK();
}

Status DatasetSizeGetter::GetRow(const std::shared_ptr<TreeAdapter> &tree_adapter, TensorRow *row) {
  RETURN_UNEXPECTED_IF_NULL(row);
  return tree_adapter->GetNext(row);
}

Status DatasetSizeGetter::Terminate() {
  for (const auto &tree : tree_adapters_) {
    RETURN_UNEXPECTED_IF_NULL(tree);
    RETURN_UNEXPECTED_IF_NULL(tree->AllTasks());
    RETURN_IF_NOT_OK(tree->AllTasks()->ServiceStop());
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
