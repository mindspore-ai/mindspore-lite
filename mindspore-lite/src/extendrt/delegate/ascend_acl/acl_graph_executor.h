/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_ACL_ACL_GRAPH_EXECUTOR_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_ACL_ACL_GRAPH_EXECUTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <map>

#include "include/api/context.h"
#include "include/api/status.h"
#include "include/api/types.h"
#include "extendrt/session/lite_graph_executor.h"
#include "common/config_infos.h"
#include "src/common/common.h"
#include "extendrt/delegate/ascend_acl/model_infer.h"

namespace mindspore {
class AclGraphExecutor : public LiteGraphExecutor {
 public:
  AclGraphExecutor(const std::shared_ptr<mindspore::Context> &context, const ConfigInfos &config_info) {
    context_ = context;
    config_info_ = config_info;
  }
  ~AclGraphExecutor();

  bool CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options,
                    uint32_t *graph_id) override;
  bool CompileGraph(const void *model_data, size_t data_size, const std::map<string, string> &compile_options,
                    uint32_t *graph_id) override;
  bool RunGraph(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
                std::vector<mindspore::MSTensor> *outputs, const std::map<string, string> &compile_options) override;

  bool Resize(uint32_t graph_id, const std::vector<mindspore::MSTensor> &inputs,
              const std::vector<ShapeVector> &dims) override;

  std::vector<mindspore::MSTensor> GetInputInfos(uint32_t graph_id) override;

  std::vector<mindspore::MSTensor> GetOutputInfos(uint32_t graph_id) override;

  const std::vector<TypeId> GetOutputDataType() { return model_infer_->GetOutputDataType(); }

  bool UpdateWeights(const std::vector<std::vector<MSTensor>> &inputs);

  Status Init();
  std::string GetConfigOption(const std::string &section_name, const std::string &option_name);

 private:
  Status BuildCustomAscendKernel(const CNodePtr &cnode);
  void GetShareMemInfos(std::shared_ptr<AclModelOptions> acl_options_ptr);
  Status GetExecConfig(const std::shared_ptr<AclModelOptions> &acl_options_ptr);
  std::shared_ptr<AclModelOptions> GenAclOptions();
  bool GetDeviceID(int32_t *device_id);
  Status GetOutputTensors(const std::vector<std::string> &output_names, std::vector<MSTensor> *output_tensors);
  std::shared_ptr<mindspore::Context> context_ = nullptr;
  ConfigInfos config_info_;
  std::shared_ptr<ModelInfer> model_infer_;
  std::shared_ptr<Primitive> primitive_ = nullptr;
  std::map<uint32_t, std::vector<mindspore::MSTensor>> graph_outputs_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  bool load_model_ = false;
};

}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_ACL_ACL_GRAPH_EXECUTOR_H_
