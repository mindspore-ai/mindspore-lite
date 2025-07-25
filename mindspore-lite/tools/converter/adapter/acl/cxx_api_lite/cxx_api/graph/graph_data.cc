/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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
#include "cxx_api/graph/graph_data.h"
#include "src/common/log_adapter.h"
#include "utils/ms_context.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
Graph::GraphData::GraphData(const FuncGraphPtr &func_graph, enum ModelType model_type)
    : func_graph_(nullptr), om_data_(), model_type_(ModelType::kUnknownType), data_graph_({}) {
  if (model_type != ModelType::kMindIR) {
    MS_LOG(EXCEPTION) << "Invalid ModelType " << model_type;
  }
  func_graph_ = func_graph;
  model_type_ = model_type;
}

Graph::GraphData::GraphData(const Buffer &om_data, enum ModelType model_type)
    : func_graph_(nullptr), om_data_(om_data), model_type_(model_type), data_graph_({}) {
  if (model_type_ != ModelType::kOM) {
    MS_LOG(EXCEPTION) << "Invalid ModelType " << model_type_;
  }
}

Graph::GraphData::~GraphData() {}

FuncGraphPtr Graph::GraphData::GetFuncGraph() const {
  if (model_type_ != ModelType::kMindIR) {
    MS_LOG(ERROR) << "Invalid ModelType " << model_type_;
    return nullptr;
  }

  return func_graph_;
}

Buffer Graph::GraphData::GetOMData() const {
  if (model_type_ != ModelType::kOM) {
    MS_LOG(ERROR) << "Invalid ModelType " << model_type_;
    return Buffer();
  }

  return om_data_;
}

void Graph::GraphData::SetPreprocess(const std::vector<std::shared_ptr<dataset::Execute>> &data_graph) {
  data_graph_ = data_graph;
}
}  // namespace mindspore
