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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_GRAPH_EXECUTOR_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_GRAPH_EXECUTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <set>
#include <utility>

#include "include/api/context.h"
#include "include/model.h"
#include "tools/converter/adapter/acl/backend/ge_backend/graph_ir/types.h"
#include "extendrt/session/lite_graph_executor.h"
#include "common/config_infos.h"
#include "tools/converter/adapter/acl/backend/ge_backend/graph_ir/utils.h"
#include "extendrt/delegate/ascend_ge/ge_device_context.h"
#include "extendrt/delegate/ascend_ge/ge_memory_manager.h"
#include "extendrt/delegate/ascend_ge/ge_context_manager.h"
#include "extendrt/delegate/ascend_ge/ge_session_manager.h"
#include "src/common/common.h"

namespace mindspore {
using MSTensorPtr = std::shared_ptr<MSTensor>;

class MSTensorRel {
 public:
  explicit MSTensorRel(const MSTensorPtr &tensor) : tensor_(tensor) {}
  ~MSTensorRel() = default;
  void Rel() const { tensor_ = nullptr; }

 private:
  mutable MSTensorPtr tensor_;
};

struct InOutBufferInfo {
  ShapeVector shape;
  TypeId dtype = kTypeUnknown;
  void *device_addr = nullptr;
  size_t max_size = 0;
  GeTensor ge_tensor;
};

struct OutputInfo {
  ShapeVector shape;
  TypeId dtype = kTypeUnknown;
};

struct GraphRuntimeInfo {
  void *const_addr = nullptr;
  size_t const_size = 0;
  void *feature_addr = nullptr;
  size_t feature_size = 0;
  std::vector<ShapeVector> output_shapes;
};

struct DynKVCacheInfo {
  bool dynamic_kv_cache = false;
  bool batch_size_dyn = false;
  bool seq_length_dyn = false;
  bool is_ge_graph_static_ = false;
  int64_t real_batch_size = -1;
  int64_t real_seq_len_size = -1;
  int64_t max_batch_size = 32;
  int64_t max_seq_len_size = 4096;
  std::vector<std::vector<int64_t>> dynamic_kv_cache_dims;
  std::string kv_cache_layout = lite::kKVCacheLayoutBNSD;
};

class GeGraphExecutor : public LiteGraphExecutor {
 public:
  GeGraphExecutor(const std::shared_ptr<mindspore::Context> &context, const ConfigInfos &config_infos)
      : context_(context), config_infos_(config_infos) {}
  ~GeGraphExecutor();

  Status CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options,
                      uint32_t *graph_id) override;

  Status RunGraph(uint32_t graph_id, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                  const std::map<string, string> &compile_options) override;

  Status Resize(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
                const std::vector<ShapeVector> &dims) override {
    return kSuccess;
  }

  std::vector<mindspore::MSTensor> GetInputInfos(uint32_t graph_id) override;
  std::vector<mindspore::MSTensor> GetOutputInfos(uint32_t graph_id) override;
  bool Init();
  bool AoeTuning(const FuncGraphPtr &graph);
  bool OfflineBuildGraph(const FuncGraphPtr &graph);

 private:
  const std::shared_ptr<mindspore::Context> context_;
  ConfigInfos config_infos_;
  std::shared_ptr<ge::Session> ge_session_ = nullptr;
  std::map<std::string, std::string> session_options_;
  int64_t session_id_ = -1;
  std::vector<uint32_t> init_graph_id_list_;
  std::vector<uint32_t> compute_graph_id_list_;
  backend::ge_backend::RefModeFlag ref_mode_flag_ = backend::ge_backend::RefModeFlag::kRefModeNone;
  std::string cache_mode_;
  std::vector<RefDataInfo> ref_data_infos_;
  std::vector<InOutBufferInfo> inputs_buffer_infos_;
  std::vector<InOutBufferInfo> outputs_buffer_infos_;

  std::shared_ptr<GeMemoryManager> memory_manager_ = nullptr;
  std::shared_ptr<GeContextManager> context_manager_ = nullptr;
  // Cache of temporary device buffers for the zero-copy path (used by host inputs/outputs).
  // Keyed by tensor name+size so that different tensors never share the same buffer.
  // Reused across inferences; all buffers are released in the destructor.
  std::map<std::string, std::pair<void *, size_t>> cached_temp_buffers_;

  std::shared_ptr<GeDeviceContext> ge_global_context_ = nullptr;
  std::string graph_name_;
  std::string build_cache_dir_;
  std::string build_cache_relative_dir_;

  std::map<uint32_t, std::vector<mindspore::MSTensor>> graph_inputs_;
  std::map<uint32_t, std::vector<mindspore::MSTensor>> graph_outputs_;
  std::map<uint32_t, std::vector<MSTensorPtr>> original_graph_outputs_;
  // Records whether the graph has dynamic input dimensions (-1) at compile time.
  std::map<uint32_t, bool> graph_has_dynamic_dim_;
  DynKVCacheInfo dyn_kv_cache_info_;

  std::shared_ptr<AscendDeviceInfo> GetAscendDeviceInfo();
  uint32_t GetRankID() const;
  uint32_t GetDeviceID() const;
  void GetGeGraphOptions(const FuncGraphPtr &anf_graph, std::map<std::string, std::string> *ge_options);
  void GetGeSessionOptions(std::map<std::string, std::string> *ge_options);
  void GetGeSessionOptionsFromAscendContext(const std::map<std::string, std::string> &config,
                                            std::map<std::string, std::string> *ge_options_ptr);
  void SetGeDumpOptions(const std::map<std::string, std::string> &config,
                        std::map<std::string, std::string> *ge_options_ptr);
  void SetGeProfilingOptions(const std::map<std::string, std::string> &config,
                             std::map<std::string, std::string> *ge_options_ptr);
  bool CreateSession(const std::map<std::string, std::string> &extra_options);
  int64_t GetSessionId();
  void GetParams(const FuncGraphPtr &anf_graph, backend::ge_backend::TensorOrderMap *param_tensors);

  bool AddGraph(const backend::ge_backend::DfGraphPtr &graph, const std::map<std::string, std::string> &options,
                uint32_t *graph_id);
  bool RunGeInitGraph(uint32_t init_graph_id, const std::vector<std::string> &init_data_names,
                      const backend::ge_backend::TensorOrderMap &params_vals);
  MSTensorPtr ConvertGeTensorNoCopy(::ge::Tensor *ge_tensor_ptr, uint32_t graph_id, size_t idx);

  bool RunGraphWithStreamAsync(uint32_t graph_id, void *stream, const std::vector<GeTensor> &inputs,
                               std::vector<GeTensor> *outputs);
  // Sub-steps extracted from RunGraph (to keep cyclomatic complexity/function length in check):
  //   RunGraphNormal: non-zero-copy path (input conversion + RunGeGraphAsync + output handling);
  //   HandleNormalOutputs: non-zero-copy output copy (device/host buffer x GE placement);
  //   RunGraphZeroCopy: zero-copy path (input/output device buffers + RunGraphWithStreamAsync);
  //   BindZeroCopyOutputs: zero-copy output binding (device -> user buffer, host -> preallocated buffer).
  Status RunGraphNormal(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
                        std::vector<mindspore::MSTensor> *outputs, std::vector<GeTensor> *ge_outputs);
  Status HandleNormalOutputs(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
                             std::vector<mindspore::MSTensor> *outputs, std::vector<GeTensor> *ge_outputs);
  // Check whether the graph supports zero-copy: dynamic-bucket (ge.dynamicNodeType=1)
  // or static graphs are allowed; purely dynamic graphs are not.
  bool IsDynamicBucketOrStatic(uint32_t graph_id) const;

  Status RunGraphZeroCopy(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
                          std::vector<mindspore::MSTensor> *outputs, std::vector<GeTensor> *ge_outputs);
  // Prepare inputs for the zero-copy path: device inputs are bound to user buffers without copy,
  // host inputs are copied H2D into temporary device buffers (RunGraphWithStreamAsync requires
  // all-device inputs). The temporary buffers are released by the caller after execution.
  bool PrepareZeroCopyInputs(const std::vector<mindspore::MSTensor> &inputs, std::vector<GeTensor> *ge_inputs);
  // Get (or reuse) a cached device buffer for the given tensor name; allocates a new one
  // if not yet cached or existing buffer is smaller than needed. Keyed by name so that
  // different tensors never share the same buffer.
  uint8_t *GetCachedDeviceBuffer(const std::string &name, size_t size);
  // Prepare outputs for the zero-copy path: device outputs are bound to user buffers (GE writes
  // directly), host outputs use cached temporary device buffers (D2H after GE writes). This is
  // symmetric with the input side and does not depend on RefMode's outputs_buffer_infos_.
  bool BindZeroCopyOutputs(std::vector<mindspore::MSTensor> *outputs, std::vector<GeTensor> *ge_outputs,
                           std::vector<bool> *output_is_device);
  Status HandleZeroCopyOutputs(uint32_t graph_id, std::vector<mindspore::MSTensor> *outputs,
                               std::vector<GeTensor> *ge_outputs, const std::vector<bool> &output_is_device);
  bool InitRefDataList(const std::vector<std::pair<std::string, tensor::TensorPtr>> &ref_data_tensors);
  bool InitRefDataContext(const FuncGraphPtr &func_graph,
                          const std::vector<std::pair<std::string, tensor::TensorPtr>> &ref_data_tensors,
                          std::map<std::string, std::string> *ge_options_ptr);
  bool InitRefDataDeviceTensor();
  bool InitConstantFeatureDeviceMemory(uint32_t graph_id);
  bool InitInOutDeviceBuffer(const std::string &name, const ShapeVector &shape, TypeId dtype,
                             InOutBufferInfo *buffer_info);
  bool InitInputDataTensor(const std::vector<mindspore::MSTensor> &inputs, std::vector<::ge::Tensor> *ge_inputs,
                           std::vector<::ge::Tensor> *ge_outputs);
  bool PrepareInputTensors(const std::vector<mindspore::MSTensor> &inputs, std::vector<::ge::Tensor> *ge_inputs);
  bool PrepareRefDataInputs(std::vector<::ge::Tensor> *ge_inputs);
  bool PrepareOutputTensors(std::vector<::ge::Tensor> *ge_outputs);
  bool InitMemoryContextManager();

  bool BuildGraphRefMode(const FuncGraphPtr &anf_graph, uint32_t graph_id);
  bool RunGraphRefMode(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
                       std::vector<mindspore::MSTensor> *outputs);
  bool SyncDeviceOutputsToHost(std::vector<mindspore::MSTensor> *outputs, std::vector<::ge::Tensor> *ge_outputs);

  bool UpdateInputShapeOption(const FuncGraphPtr &func_graph,
                              const std::vector<std::pair<std::string, tensor::TensorPtr>> &ref_data_tensors,
                              std::map<std::string, std::string> *ge_options_ptr);

  static std::atomic_uint32_t global_graph_idx_;
  static uint32_t GetNextGraphIdx();

  bool RunGeGraphAsync(uint32_t graph_id, const std::vector<::ge::Tensor> &inputs, std::vector<::ge::Tensor> *outputs);

  backend::ge_backend::DfGraphPtr CompileGraphCommon(const FuncGraphPtr &graph,
                                                     std::map<std::string, std::string> *ge_options_ptr);

  backend::ge_backend::DfGraphPtr CreateGeGraphOnline(const FuncGraphPtr &anf_graph,
                                                      std::map<std::string, std::string> *ge_options_ptr);
  backend::ge_backend::DfGraphPtr CreateFakeGraph(const std::map<std::string, std::string> &ge_options);

  void SetOptionsIntoOfflineModel(const std::map<std::string, std::string> &graph_options,
                                  std::map<std::string, ValuePtr> *attr_map);

  bool LoadOnlineGraph(const FuncGraphPtr &anf_graph, uint32_t *graph_id);
  bool UpdateGraphInputs(const FuncGraphPtr &graph);

  bool GetOneRealInputs(const FuncGraphPtr &func_graph, std::vector<ge::Tensor> *ge_tensors);
  bool CreateAsCustomFuncGraph(const FuncGraphPtr &func_graph, const std::map<std::string, std::string> &graph_options);
  bool SetModelCacheDir(std::map<std::string, std::string> *session_options_ptr);
  bool SetOfflineBuildModelCacheDir(std::map<std::string, std::string> *session_options_ptr);
  bool GetConfigOption(const std::string &section_name, const std::string &option_name, std::string *option_val);

  bool SetGeTensorShape(GeTensor *ge_tensor, ShapeVector shape);
  void UpdateOutputShapeInfo(std::vector<::ge::Tensor> *ge_outputs);
  bool InitRefModeConfig();
  bool InitRealShapeParam(const std::vector<mindspore::MSTensor> &inputs);
  bool CheckRefDataInfo();
  bool InitMaxShapeParam();
  void SetRefShape(std::vector<int64_t> *ref_shape, bool dyn, std::string tensor_name);
  bool InitInputDeviceTensor(const FuncGraphPtr &anf_graph);
  bool InitOutputDeviceTensor(const FuncGraphPtr &anf_graph, uint32_t graph_id);
  std::shared_ptr<GeTensor> ConvertMSTensor(const std::shared_ptr<MSTensor> &tensor, const std::string &format,
                                            bool copy = true);
  bool is_first_inference_ = true;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_GRAPH_EXECUTOR_H_
