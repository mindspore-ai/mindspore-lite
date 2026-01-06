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

#include "src/litert/delegate/pnna/pnna_subgraph.h"
#include "src/litert/delegate/pnna/pnna_utils.h"
#include "include/errorcode.h"

namespace mindspore::lite {
PNNASubGraph::~PNNASubGraph() {
  for (auto op : ops_) {
    if (op != nullptr) {
      delete op;
    }
  }
}

std::shared_ptr<pnna::Tensor> PNNASubGraph::GetMappedTensor(MSTensor *operand) {
  if (tensors_.empty()) return nullptr;
  auto it = std::find_if(tensors_.begin(), tensors_.end(), [&](const auto &p) { return *(p.first) == *operand; });
  if (it != tensors_.end()) {
    return it->second.back();
  }
  return nullptr;
}

void PNNASubGraph::UpdateTensorMap(MSTensor *operand, std::shared_ptr<pnna::Tensor> tensor) {
  if (tensors_.empty()) {
    tensors_.insert(std::make_pair(operand, std::vector<std::shared_ptr<pnna::Tensor>>{tensor}));
  } else {
    bool is_found = false;
    for (auto it : tensors_) {
      if (*it.first == *operand) {
        it.second.push_back(tensor);
        is_found = true;
      }
    }
    if (!is_found) {
      tensors_.insert(std::make_pair(operand, std::vector<std::shared_ptr<pnna::Tensor>>{tensor}));
    }
  }
}

std::shared_ptr<pnna::Tensor> PNNASubGraph::AddTensor(MSTensor *operand) {
  pnna::TensorAttribute tensor_attr;
  std::function<bool(MSTensor *)> is_graph_input = [&](MSTensor *operand) {
    return std::find(inputs_.begin(), inputs_.end(), *operand) != inputs_.end();
  };
  std::function<bool(MSTensor *)> is_graph_output = [&](MSTensor *operand) {
    return std::find(outputs_.begin(), outputs_.end(), *operand) != outputs_.end();
  };
  if (operand->IsConst() && operand->Data().get()) {
    tensor_attr = pnna::TensorAttribute::CONSTANT;
  } else if (is_graph_input(operand)) {
    tensor_attr = pnna::TensorAttribute::INPUT;
  } else if (is_graph_output(operand)) {
    tensor_attr = pnna::TensorAttribute::OUTPUT;
  } else {
    tensor_attr = pnna::TensorAttribute::TRANSIENT;
  }
  auto tensor = CreatePnnaTensor(graph_.get(), operand, tensor_attr);
  MS_ASSERT(tensor);
  return tensor;
}

std::shared_ptr<pnna::Tensor> PNNASubGraph::ConvertOperand(MSTensor *operand) {
  auto tensor = AddTensor(operand);
  UpdateTensorMap(operand, tensor);
  return tensor;
}

int PNNASubGraph::Init() {
  graph_ = ctx_->CreateGraph();
  if (!graph_) {
    MS_LOG(ERROR) << "Failed to create the pnna graph!";
    return RET_ERROR;
  }
  return RET_OK;
}

int PNNASubGraph::CreatePNNAModel() {
  for (auto op : ops_) {
    if (op->AddOpToPNNAModel(this) != RET_OK) {
      MS_LOG(ERROR) << "Add pnna op to model failed: " << op->name();
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int PNNASubGraph::CompilePNNAModel() {
  if (!graph_->Compile()) {
    MS_LOG(ERROR) << "Compile pnna graph failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int PNNASubGraph::Prepare() { return RET_OK; }

int PNNASubGraph::PreProcess() {
  // set input data.
  for (int idx = 0; idx < static_cast<int>(inputs_.size()); idx++) {
    MS_CHECK_TRUE_RET(idx < static_cast<int>(inputs_.size()), RET_ERROR);
    auto tensor = inputs_.at(idx);
    auto origin_tensor = origin_inputs_.at(idx);
    // malloc data for tensor
    auto data = tensor.MutableData();
    MS_CHECK_TRUE_RET(data != nullptr, RET_ERROR);
    for (auto it : tensors_) {
      if (*it.first == origin_tensor) {
        MS_ASSERT(it.second.empty() == false);
        if (!it.second.front()->CopyDataToTensor(data, tensor.DataSize())) {
          MS_LOG(ERROR) << "Failed to copy data to pnna tensor.";
          return RET_ERROR;
        }
      }
    }
  }
  // set output data.
  for (int idx = 0; idx < static_cast<int>(outputs_.size()); idx++) {
    MS_CHECK_TRUE_RET(idx < static_cast<int>(outputs_.size()), RET_ERROR);
    auto tensor = outputs_.at(idx);
    auto data = tensor.MutableData();
    MS_CHECK_TRUE_RET(data != nullptr, RET_ERROR);
  }
  return RET_OK;
}

int PNNASubGraph::PostProcess() {
  for (int idx = 0; idx < static_cast<int>(outputs_.size()); idx++) {
    auto tensor = outputs_.at(idx);
    auto origin_tensor = origin_outputs_.at(idx);
    for (auto it : tensors_) {
      if (*it.first == origin_tensor) {
        MS_ASSERT(it.second.empty() == false);
        if (!it.second.front()->CopyDataFromTensor(const_cast<void *>(tensor.Data().get()))) {
          MS_LOG(ERROR) << "Failed to copy data from pnna tensor.";
          return RET_ERROR;
        }
      }
    }
  }
  return RET_OK;
}

int PNNASubGraph::Execute() {
  if (PreProcess() != RET_OK) {
    MS_LOG(ERROR) << "PreProcess failed.";
    return RET_ERROR;
  }
  if (!graph_->Run()) {
    MS_LOG(ERROR) << "Run pnna graph failed.";
    return RET_ERROR;
  }
  if (PostProcess() != RET_OK) {
    MS_LOG(ERROR) << "PostProcess failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite
