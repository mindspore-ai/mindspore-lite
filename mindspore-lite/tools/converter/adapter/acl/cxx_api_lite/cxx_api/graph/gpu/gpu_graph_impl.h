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
#ifndef MINDSPORE_CCSRC_CXX_API_GRAPH_MS_GPU_GRAPH_IMPL_H
#define MINDSPORE_CCSRC_CXX_API_GRAPH_MS_GPU_GRAPH_IMPL_H
#include <string>
#include <vector>
#include <utility>
#include <memory>
#include "include/api/status.h"
#include "include/api/graph.h"
#include "cxx_api/graph/graph_impl.h"
#include "ir/anf.h"
#include "cxx_api/model/model_impl.h"

namespace mindspore {
class GPUGraphImpl : public GraphCell::GraphImpl {
 public:
  GPUGraphImpl();
  ~GPUGraphImpl() override = default;

  Status Run(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) override;
  Status Load(uint32_t device_id) override;
  std::vector<MSTensor> GetInputs() override;
  std::vector<MSTensor> GetOutputs() override;

  bool CheckDeviceSupport(mindspore::DeviceType device_type) override;

 private:
  Status InitEnv();
  Status FinalizeEnv();
  Status CompileGraph(const std::shared_ptr<FuncGraph> &func_graph);
  Status CheckModelInputs(const std::vector<tensor::TensorPtr> &inputs) const;
  std::vector<tensor::TensorPtr> RunGraph(const std::vector<tensor::TensorPtr> &inputs);
  Status ExecuteModel(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs);

  std::string device_type_;
  bool init_flag_;
  bool set_device_id_flag_;

  // tensor-rt
  uint32_t batch_size_{0};
  uint32_t workspace_size_{0};
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXX_API_GRAPH_MS_GPU_GRAPH_IMPL_H
