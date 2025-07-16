/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "runtime/hardware/device_context.h"
#include "include/api/status.h"

namespace mindspore {
/// \brief Adaptive Graph Executor for cloud Graph Executor to solve interface conflicts.
class LiteGraphExecutor {
 public:
  LiteGraphExecutor() = default;
  virtual ~LiteGraphExecutor() = default;

  virtual void Initialize() { return; }
  virtual void Finalize() { return; }

  virtual bool CompileGraph(const std::shared_ptr<FuncGraph> &graph,
                            const std::map<std::string, std::string> &compile_options, uint32_t *graph_id) {
    return false;
  }

  virtual bool CompileGraph(const void *model_data, size_t data_size,
                            const std::map<std::string, std::string> &compile_options, uint32_t *graph_id) {
    return false;
  }

  // form base class
  virtual bool RunGraph(const std::shared_ptr<FuncGraph> &graph, const std::vector<MSTensor> &inputs,
                        std::vector<MSTensor> *outputs, const std::map<std::string, std::string> &compile_options) {
    MS_LOG(EXCEPTION) << "Unimplemented interface.";
  }

  virtual bool CompileGraph(const std::shared_ptr<FuncGraph> &graph,
                            const std::map<std::string, std::string> &compile_options) {
    return true;
  }

  virtual bool UpdateWeights(const std::vector<std::vector<std::shared_ptr<mindspore::MSTensor>>> &weights) {
    MS_LOG(ERROR) << "UpdateWeights failed.";
    return false;
  }

  virtual bool RunGraph(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
                        std::vector<mindspore::MSTensor> *outputs,
                        const std::map<std::string, std::string> &compile_options) {
    (void)graph_id;
    (void)inputs;
    (void)outputs;
    (void)compile_options;
    return false;
  }

  virtual bool Resize(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
                      const std::vector<std::vector<int64_t>> &new_shapes) {
    (void)graph_id;
    (void)inputs;
    (void)new_shapes;
    return true;
  }
  virtual std::vector<mindspore::MSTensor> GetInputInfos(uint32_t graph_id) {
    (void)graph_id;
    return {};
  }
  virtual std::vector<mindspore::MSTensor> GetOutputInfos(uint32_t graph_id) {
    (void)graph_id;
    return {};
  }

  virtual const std::vector<TypeId> GetOutputDataType() { return {}; }
  void SetBefore(const MSKernelCallBack &before) { before_ = before; }

  void SetAfter(const MSKernelCallBack &after) { after_ = after; }

 protected:
  MSKernelCallBack before_;
  MSKernelCallBack after_;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_EXTENDRT_SESSION_LITE_GRAPH_EXECUTOR_H_
