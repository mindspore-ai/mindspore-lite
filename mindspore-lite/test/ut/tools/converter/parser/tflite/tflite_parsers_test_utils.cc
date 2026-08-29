/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ut/tools/converter/parser/tflite/tflite_parsers_test_utils.h"
#include <string>
#include "ir/func_graph.h"
#include "src/common/log_adapter.h"
#include "include/registry/model_parser.h"
#include "include/registry/model_parser_registry.h"
#include "tools/lite_exporter/anf_exporter.h"
#include "schema/inner/model_generated.h"

namespace mindspore {

schema::MetaGraphT *TestTfliteParser::LoadAndConvert(const std::string &model_path, const std::string &weight_path) {
  auto model_parser = registry::ModelParserRegistry::GetModelParser(converter::kFmkTypeTflite);
  if (model_parser == nullptr) {
    MS_LOG(ERROR) << "tflite model parser is nullptr";
    return nullptr;
  }
  converter::ConverterParameters flags;
  flags.fmk = converter::kFmkTypeTflite;
  flags.model_file = model_path;
  flags.weight_file = weight_path;
  auto func_graph_base = model_parser->Parse(flags);
  delete model_parser;
  if (func_graph_base == nullptr) {
    MS_LOG(ERROR) << "parse tflite model failed: " << model_path;
    return nullptr;
  }
  auto func_graph = std::dynamic_pointer_cast<FuncGraph>(func_graph_base->impl());
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "convert api func graph to func graph failed";
    return nullptr;
  }
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manage func graph failed";
    return nullptr;
  }
  // Const tensors declared as tflite subgraph inputs (e.g. block_shape/crops) are folded into
  // primitive attrs by TfliteInputsAdjust, leaving graph inputs without consumers that
  // AnfExporter::SetMetaGraphInput rejects. Prune them before export.
  AnfNodePtrList used_params;
  const auto &node_users = manager->node_users();
  for (const auto &param : func_graph->parameters()) {
    auto iter = node_users.find(param);
    if (iter != node_users.end() && !iter->second.empty()) {
      used_params.push_back(param);
    }
  }
  func_graph->set_parameters(std::move(used_params));
  return lite::Export(func_graph, false, false, false);
}

void TestTfliteParser::TearDown() { delete meta_graph; }

}  // namespace mindspore
