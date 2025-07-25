/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_PYTORCH_PYTORCH_MODEL_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_PYTORCH_PYTORCH_MODEL_PARSER_H_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "torch/script.h"
#include "include/securec.h"
#include "include/registry/model_parser.h"
#include "include/registry/model_parser_registry.h"
#include "tools/converter/parser/pytorch/pytorch_node_parser_registry.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
class PytorchModelParser : public converter::ModelParser {
 public:
  PytorchModelParser() = default;

  ~PytorchModelParser() override = default;

  api::FuncGraphPtr Parse(const converter::ConverterParameters &flag) override;

 private:
  STATUS InitOriginModel(const std::string &model_file);
  STATUS ConvertTorchGraph(const FuncGraphPtr &anf_graph);
  STATUS ConvertGraphInputs(const FuncGraphPtr &anf_graph);
  STATUS ConvertGraphOutputs(const FuncGraphPtr &anf_graph);
  STATUS ConvertNodes(const FuncGraphPtr &anf_graph);
  STATUS TorchModelAdjust(const FuncGraphPtr &anf_graph);
  STATUS ConvertConstNode(const torch::jit::Node *torch_node, const FuncGraphPtr &anf_graph,
                          std::unordered_map<std::string, AnfNodePtr> *anf_nodes_map);
  static tensor::TensorPtr ConvertTorchTensor(const at::Tensor &torch_tensor);

  std::shared_ptr<torch::jit::Graph> torch_model_;
  std::vector<FuncGraphPtr> all_subgraphs_{};
  std::unordered_map<std::string, AnfNodePtr> anf_nodes_map_{};
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_PYTORCH_PYTORCH_MODEL_PARSER_H_
