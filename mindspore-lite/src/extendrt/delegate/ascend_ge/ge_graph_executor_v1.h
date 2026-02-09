/**
 * Copyright 2025-2026 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_GRAPH_EXECUTOR_V1_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_GRAPH_EXECUTOR_V1_H_

#include <vector>
#include <string>
#include <memory>
#include <map>
#include "include/api/context.h"
#include "backend/ge_backend/graph_ir/types.h"
#include "common/config_infos.h"
#include "extendrt/session/lite_graph_executor.h"
#include "extendrt/delegate/ascend_ge/ge_device_context.h"
#include "extendrt/delegate/ascend_ge/ge_memory_manager.h"
#include "extendrt/delegate/ascend_ge/ge_context_manager.h"
#include "extendrt/delegate/ascend_ge/ge_options_container.h"
#include "extendrt/delegate/ascend_ge/ge_graph_compiler.h"

namespace mindspore {
using MSTensorPtr = std::shared_ptr<MSTensor>;

class GeGraphExecutorV1 : public LiteGraphExecutor {
 public:
  GeGraphExecutorV1(const std::shared_ptr<mindspore::Context> &context, const ConfigInfos &config_infos)
      : context_(context), config_infos_(config_infos) {}
  ~GeGraphExecutorV1();

  bool Init();
  bool CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options,
                    uint32_t *graph_id) override;

  bool RunGraph(uint32_t graph_id, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                const std::map<string, string> &compile_options) override;
  bool Resize(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
              const std::vector<ShapeVector> &dims) override {
    return true;
  }

  std::vector<mindspore::MSTensor> GetInputInfos(uint32_t graph_id) override;
  std::vector<mindspore::MSTensor> GetOutputInfos(uint32_t graph_id) override;

 private:
  bool CheckParallelCompile();
  bool InitGEResource();
  bool InitMsTensor(const FuncGraphPtr &graph, uint32_t graph_id);
  bool InitGeTensor(uint32_t graph_id);
  bool PrepareGeInputs(const std::vector<MSTensor> &inputs, std::vector<GeTensor> *ge_inputs, uint32_t graph_id);
  bool PrepareGeOutputsForStatic(std::vector<MSTensor> *outputs, std::vector<GeTensor> *ge_outputs, uint32_t graph_id);
  bool RunStaticGraph(uint32_t graph_id, const std::vector<GeTensor> &ge_inputs, std::vector<MSTensor> *outputs);
  bool RunDynamicGraph(uint32_t graph_id, const std::vector<GeTensor> &ge_inputs, std::vector<MSTensor> *outputs);
  bool MallocDeviceMem(std::pair<void *, size_t> &tensor_mem_info, void *&device_addr, size_t size);
  bool PostProcessOutputsForStatic(std::vector<MSTensor> *outputs, uint32_t graph_id);
  bool PostProcessOutputsForDynamic(std::vector<MSTensor> *outputs, uint32_t graph_id,
                                    const std::vector<GeTensor> &outputs_ge_tensors);
  bool IsDynamical(const std::vector<MSTensor> &outputs, uint32_t graph_id);
  GeOptionsContainer ge_options_container_;
  GeGraphCompiler ge_graph_compiler_;
  const std::shared_ptr<mindspore::Context> context_;
  ConfigInfos config_infos_;
  GeSessionInfo ge_session_info_;
  std::shared_ptr<GeDeviceContext> ge_global_context_{nullptr};
  std::shared_ptr<GeMemoryManager> memory_manager_{nullptr};
  std::shared_ptr<GeContextManager> context_manager_{nullptr};
  // {graph_id, {tensor, {allocate_address_by_us, size}}}. delayed free.
  // Used to defer releasing device memory allocated for GE input tensors
  std::map<uint32_t, std::vector<std::pair<GeTensor, std::pair<void *, size_t>>>> ge_inputs_;
  // {graph_id, {tensor, {allocate_address_by_us, size}}}. delayed free.
  // Used to defer releasing device memory allocated for GE output tensors
  std::map<uint32_t, std::vector<std::pair<GeTensor, std::pair<void *, size_t>>>> ge_outputs_;
  std::map<uint32_t, std::vector<mindspore::MSTensor>> ms_inputs_;
  std::map<uint32_t, std::vector<mindspore::MSTensor>> ms_outputs_;
  std::map<uint32_t, std::pair<uint32_t, uint32_t>> graph_id_group_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_GRAPH_EXECUTOR_V1_H_
