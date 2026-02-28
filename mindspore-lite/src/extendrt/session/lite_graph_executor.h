/**
 * Copyright 2019-2026 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_EXTENDRT_SESSION_LITE_GRAPH_EXECUTOR_H_
#define MINDSPORE_LITE_EXTENDRT_SESSION_LITE_GRAPH_EXECUTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <map>

#include "include/api/types.h"
#include "include/api/status.h"
#include "mindspore/mindspore/core/include/base/base.h"
#include "src/common/log_adapter.h"
#include "mindspore/mindspore/core/include/mindapi/base/type_id.h"

namespace mindspore {
/// \brief Adaptive Graph Executor for cloud Graph Executor to solve interface conflicts.
class LiteGraphExecutor {
 public:
  LiteGraphExecutor() = default;
  virtual ~LiteGraphExecutor() = default;

  virtual void Initialize() { return; }

  virtual Status CompileGraph(const std::shared_ptr<FuncGraph> &graph,
                              const std::map<std::string, std::string> &compile_options, uint32_t *graph_id) {
    return kLiteNotSupport;
  }

  virtual Status CompileGraph(const void *model_data, size_t data_size,
                              const std::map<std::string, std::string> &compile_options, uint32_t *graph_id) {
    return kLiteNotSupport;
  }

  // form base class
  virtual Status RunGraph(const std::shared_ptr<FuncGraph> &graph, const std::vector<MSTensor> &inputs,
                          std::vector<MSTensor> *outputs, const std::map<std::string, std::string> &compile_options) {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
    return kLiteNotSupport;
  }

  virtual bool CompileGraph(const std::shared_ptr<FuncGraph> &graph,
                            const std::map<std::string, std::string> &compile_options) {
    return true;
  }

  virtual bool UpdateWeights(const std::vector<std::vector<mindspore::MSTensor>> &weights) {
    MS_LOG(ERROR) << "UpdateWeights failed.";
    return false;
  }

  virtual Status RunGraph(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
                          std::vector<mindspore::MSTensor> *outputs,
                          const std::map<std::string, std::string> &compile_options) {
    (void)graph_id;
    (void)inputs;
    (void)outputs;
    (void)compile_options;
    return kLiteNotSupport;
  }

  virtual Status Resize(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
                        const std::vector<std::vector<int64_t>> &new_shapes) {
    (void)graph_id;
    (void)inputs;
    (void)new_shapes;
    return kLiteNotSupport;
  }
  virtual std::vector<mindspore::MSTensor> GetInputInfos(uint32_t graph_id) {
    (void)graph_id;
    MS_LOG(WARNING) << "Getting graph input info is not supported.";
    return {};
  }
  virtual std::vector<mindspore::MSTensor> GetOutputInfos(uint32_t graph_id) {
    (void)graph_id;
    MS_LOG(WARNING) << "Getting graph output info is not supported.";
    return {};
  }

  virtual uint64_t GetShareableHandle() { return sharable_handle_; }

  virtual const std::vector<TypeId> GetOutputDataType() { return {}; }
  void SetBefore(const MSKernelCallBack &before) { before_ = before; }

  void SetAfter(const MSKernelCallBack &after) { after_ = after; }

 protected:
  MSKernelCallBack before_;
  MSKernelCallBack after_;
  uint64_t sharable_handle_;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_EXTENDRT_SESSION_LITE_GRAPH_EXECUTOR_H_
