/**
 * copy from https://gitcode.com/mindspore/mindspore/blob/master/mindspore/ccsrc/backend/ge_backend/graph_ir/types.h
 *
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_BACKEND_GE_BACKEND_GRAPH_IR_TYPES_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_BACKEND_GE_BACKEND_GRAPH_IR_TYPES_H_

#include <string>
#include <vector>
#include <map>
#include <memory>

#include <unordered_map>
#include "ir/anf.h"
#include "ir/tensor.h"
#include "graph/tensor.h"
#include "ge/ge_api.h"

using GeTensor = ::ge::Tensor;

namespace mindspore::backend::ge_backend {
enum Status : int { SUCCESS = 0, FAILED, INVALID_ARGUMENT, ALREADY_EXISTS, NOT_FOUND };

using MeTensor = mindspore::tensor::Tensor;
using MeTensorPtr = std::shared_ptr<MeTensor>;
using MeDataType = mindspore::TypeId;
using GeDataType = ::ge::DataType;
using GeFormat = ::ge::Format;
using GeShape = ::ge::Shape;
using GeTensorPtr = std::shared_ptr<GeTensor>;
using GeTensorDesc = ::ge::TensorDesc;
using AnfGraph = FuncGraph;
using AnfGraphPtr = FuncGraphPtr;
using Operator = ::ge::Operator;
using OperatorPtr = std::shared_ptr<::ge::Operator>;
using DfGraph = ::ge::Graph;
using DfGraphPtr = std::shared_ptr<DfGraph>;
using TensorMap = std::unordered_map<std::string, std::shared_ptr<MeTensor>>;
using OptionMap = std::map<std::string, std::string>;
using TensorOrderMap = std::map<std::string, std::shared_ptr<tensor::Tensor>>;
using GeAllocatorPtr = ::ge::AllocatorPtr;

struct DfGraphWrapper {
 public:
  DfGraphWrapper(const std::string &name, const int &id, const DfGraphPtr &graph_ptr, const OptionMap &options);
  ~DfGraphWrapper() {}

  std::string name_;
  int id_;
  int times_{};
  DfGraphPtr graph_ptr_;
  OptionMap options_ = {};
  bool is_added_to_ge_session_ = false;
  bool export_air_ = false;
  std::mutex mutex_;
};

using DfGraphWrapperPtr = std::shared_ptr<DfGraphWrapper>;

struct DfGraphConfig {
 public:
  DfGraphConfig(const OptionMap &options = {}, const bool &is_cloud = false, const bool &need_aoe = false,
                const bool &export_air = false)
      : options_(options), is_cloud_(is_cloud), need_aoe_(need_aoe), export_air_(export_air) {}
  ~DfGraphConfig() {}
  OptionMap options_;
  bool is_cloud_;
  bool need_aoe_;
  bool export_air_;
};

struct ControlEdge {
  OperatorPtr src_op;
  OperatorPtr dest_op;
};

using SessionOptions = std::map<std::string, std::string>;

struct GraphRunnerOptions {
  std::string target{"default_graph_runner"};
  SessionOptions options;
  // if sess_ptr is nullptr, GraphRunner will create a new ge session
  std::shared_ptr<::ge::Session> sess_ptr{nullptr};
};

struct RunOptions {
  // graph's name
  std::string name;
};
}  // namespace mindspore::backend::ge_backend
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_BACKEND_GE_BACKEND_GRAPH_IR_TYPES_H_
