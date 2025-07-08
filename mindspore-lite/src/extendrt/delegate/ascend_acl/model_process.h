/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef delegate_MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_MODEL_MODEL_PROCESS_H_
#define delegate_MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_MODEL_MODEL_PROCESS_H_

#include <vector>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <functional>
#include <memory>
#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include "include/api/types.h"
#include "include/errorcode.h"
#include "extendrt/delegate/ascend_acl/acl_model_options.h"
#include "extendrt/delegate/ascend_acl/dyn_shape_process.h"
#include "extendrt/delegate/ascend_acl/acl_mem_manager.h"
#include "mindspore/core/include/mindapi/base/type_id.h"

namespace mindspore {
struct AclTensorInfo {
  void *cur_device_data;
  void *device_data;
  size_t buffer_size;
  size_t malloc_buffer_size;
  aclDataType data_type;
  std::vector<int64_t> dims;
  std::string name;
  aclTensorDesc *dynamic_acl_tensor_desc = nullptr;
  aclDataBuffer *dynamic_acl_data_buffer = nullptr;
};

class ModelProcess {
 public:
  explicit ModelProcess(const std::shared_ptr<AclModelOptions> &options)
      : options_(options), device_id_(options->device_id) {}
  ~ModelProcess();

  bool Load(const void *om_data, size_t om_data_size);
  bool UnLoad();
  bool PredictFromHost(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> *outputs);
  bool UpdateWeights(const std::vector<MSTensor> &inputs);

  // override this method to avoid request/reply data copy
  void SetIsDevice(bool is_device) { is_run_on_device_ = is_device; }

  std::set<uint64_t> GetDynamicBatch();
  std::set<std::pair<uint64_t, uint64_t>> GetDynamicImage();
  std::pair<aclmdlIODims *, size_t> GetDynamicDims();
  std::vector<Format> GetInputFormat();
  const std::vector<ShapeVector> GetOutputShape();
  const std::vector<ShapeVector> GetInputShape();
  const std::vector<TypeId> GetInputDataType();
  const std::vector<TypeId> GetOutputDataType();
  std::vector<Format> GetOutputFormat();

  bool Resize(const std::vector<ShapeVector> &new_shapes);

 private:
  bool PreInitModelResource();

  bool InitInputsBuffer();
  bool InitOutputsBuffer();
  void DestroyInputsBuffer();
  void DestroyOutputsBuffer();
  bool CreateDataBuffer(void **data_mem_buffer, size_t buffer_size, aclmdlDataset *dataset);

  bool CheckAndInitInput(const std::vector<MSTensor> &inputs);
  bool CheckAndInitOutput(const std::vector<MSTensor> *outputs);
  void CheckAndInitDynOutputDeviceBuf(const MSTensor output, const AclTensorInfo &output_info,
                                      void **output_device_buffer, size_t *output_buf_size, size_t output_idx);
  bool CheckInputTensors(const std::vector<MSTensor> &inputs);
  bool CheckOutputTensors(const std::vector<MSTensor> *outputs);
  bool CheckAndSetDynFlag();
  bool GetOutputs(const std::vector<MSTensor> *outputs);

  bool ResetInputSize(const std::vector<ShapeVector> &new_shapes);
  bool ResetOutputSize();
  bool IsDynamicShape();
  bool IsDynamicBatchSize();
  bool IsDynamicImageSize();
  bool IsDynamicDims();
  bool ResetDynamicOutputTensor(const std::vector<MSTensor> *outputs);
  bool ResizeDynamicInputShape(const std::vector<ShapeVector> &new_shapes);
  bool ResizeDynamicInputShapeRange(const std::vector<ShapeVector> &new_shapes);
  bool ResizeDynamicBatchAndImageSize(const std::vector<ShapeVector> &new_shapes);
  void FreeResourceInput(std::vector<AclTensorInfo> acl_tensor_info);
  void FreeResourceOutput(std::vector<AclTensorInfo> *acl_tensor_info, const std::vector<MSTensor> *outputs);
  aclError AclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind);
  bool PrepareMutiModelShare(const void *om_data, size_t om_data_size);
  bool InitUpdateWeightBuffer(const std::vector<MSTensor> &kernel_inputs);
  bool CreateWeightsInput(const std::vector<MSTensor> &inputs);
  void DestoryUpdateWeightBuffer();
  bool ShareWeightspaceProcess(const size_t &work_size);
  bool ShareWorkspaceProcess(const size_t &work_size, const size_t &weight_size);
  bool ShareWorkspaceAndWeightspaceProcess(const size_t &work_size);

  std::shared_ptr<AclModelOptions> options_;
  uint32_t model_id_ = UINT32_MAX;
  // if run one device(AICPU), there is no need to alloc device memory and copy inputs to(/outputs from) device
  bool is_run_on_device_ = false;
  aclmdlDesc *model_desc_ = nullptr;
  aclmdlDataset *inputs_ = nullptr;
  aclmdlDataset *outputs_ = nullptr;
  aclmdlDataset *weight_inputs_ = nullptr;
  aclmdlDataset *weight_outputs_ = nullptr;
  aclmdlDesc *model_weight_desc_ = nullptr;

  bool loaded_ = false;
  bool inited_weights_ = false;

  size_t data_input_num_ = 0;
  std::vector<AclTensorInfo> input_infos_;
  std::vector<AclTensorInfo> output_infos_;
  std::vector<AclTensorInfo> weight_input_infos_;

  AclDynamicShapeOptions dynamic_shape_options_;
  DynShapeProcess dyn_shape_proc_;
  std::vector<ShapeVector> cur_input_shapes_;
  bool is_dynamic_output_ = false;
  bool is_dynamic_input_ = false;
  bool is_dynamic_resize_input_ = false;
  bool is_dynamic_shape_range_ = false;
  aclmdlIODims *dynamic_dims_ = nullptr;
  void *weight_ptr_ = nullptr;
  void *work_ptr_ = nullptr;
  std::vector<bool> user_defined_output_buf_;
  std::set<void *> dyn_out_sys_buf_addr_;
  bool is_sharing_workspace_ = false;
  int32_t device_id_ = 0;
  uint32_t infer_id_ = 0;
  uint32_t update_id_ = 0;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_MODEL_MODEL_PROCESS_H_
