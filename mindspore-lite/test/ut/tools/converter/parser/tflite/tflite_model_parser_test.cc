/**
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
// Expand the core logging headers (complete MS_LOG family) first: later converter
// headers pull in lite's src/common/log.h whose narrower MS_LOG macro would
// otherwise break the inline functions of already-open core headers.
#if defined(_WIN32)
#include <process.h>
#define MS_UT_GETPID _getpid
#else
#include <unistd.h>
#define MS_UT_GETPID getpid
#endif
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include "src/common/log_adapter.h"
#include "ir/func_graph.h"
#include "mindspore/core/include/ir/graph_utils.h"
#include "common/common_test.h"
// Both flatbuffer schemas share the basename model_generated.h: the mindspore inner schema
// (schema:: namespace, needed by tensor_util.h pulled in by the parser header) must come first,
// then the tflite schema (tflite:: namespace) providing ModelT for the in-code fixture.
#include "schema/inner/model_generated.h"
#include "schema/model_generated.h"
#include "include/registry/converter_context.h"
#include "tools/converter/parser/tflite/tflite_model_parser.h"
#include "tools/optimizer/common/fusion_trace_utils.h"

namespace mindspore {
namespace {
// A minimal one-operator tflite model built in code: two fp32 graph inputs, one ADD operator
// without a builtin_options table (the parser tolerates the missing union entry), one output.
// Serializing through the flatbuffers object API keeps the fixture data-free.
std::string WriteMinimalAddModel() {
  auto op = std::make_unique<tflite::OperatorT>();
  op->opcode_index = 0;
  op->inputs = {0, 1};
  op->outputs = {2};

  auto subgraph = std::make_unique<tflite::SubGraphT>();
  auto make_tensor = [](const std::string &name) {
    auto tensor = std::make_unique<tflite::TensorT>();
    tensor->name = name;
    tensor->shape = {1, 4};
    tensor->type = tflite::TensorType_FLOAT32;
    tensor->buffer = 0;
    return tensor;
  };
  subgraph->tensors.push_back(make_tensor("in0"));
  subgraph->tensors.push_back(make_tensor("in1"));
  subgraph->tensors.push_back(make_tensor("out"));
  subgraph->operators.push_back(std::move(op));
  subgraph->inputs = {0, 1};
  subgraph->outputs = {2};

  auto model = std::make_unique<tflite::ModelT>();
  model->version = 3;
  auto code = std::make_unique<tflite::OperatorCodeT>();
  code->deprecated_builtin_code = tflite::BuiltinOperator_ADD;
  code->builtin_code = tflite::BuiltinOperator_ADD;
  model->operator_codes.push_back(std::move(code));
  model->subgraphs.push_back(std::move(subgraph));
  model->description = "ms";
  model->buffers.push_back(std::make_unique<tflite::BufferT>());  // buffer 0; nothing is const

  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(tflite::CreateModel(builder, model.get()), tflite::ModelIdentifier());
  const std::string path = "/tmp/ms_tflite_origin_ut_" + std::to_string(MS_UT_GETPID()) + ".tflite";
  FILE *file = fopen(path.c_str(), "wb");
  if (file == nullptr) {
    return "";  // caller asserts on the empty path and fails with a clear message
  }
  (void)fwrite(builder.GetBufferPointer(), 1, builder.GetSize(), file);
  (void)fclose(file);
  return path;
}
}  // namespace

class TfliteModelParserTest : public mindspore::CommonTest {
 public:
  TfliteModelParserTest() = default;
};

namespace {
// The fusion-trace switch is process-global. Restore it (and remove the temp model) even when
// the parse path throws (MS_EXCEPTION_IF_NULL) or an assertion aborts the test body early,
// so later tests in this binary never observe a leaked switch or a leftover file.
class TraceSwitchGuard {
 public:
  explicit TraceSwitchGuard(std::string path) : path_(std::move(path)) { opt::SetFusionTraceEnabled(true); }
  ~TraceSwitchGuard() {
    opt::SetFusionTraceEnabled(false);
    if (!path_.empty()) {
      remove(path_.c_str());
    }
  }

 private:
  std::string path_;
};
}  // namespace

// Guards the TFLite origin-name identifier: the underscore spelling must carry the
// operators-array index (what Netron shows as the node's identifier property) - the same
// number as the dash current name. The recording used to read op_idx AFTER the per-operator
// increment, so every TYPE_identifier entry was one larger than the documented value and
// doc-recipe configs (quant_node=TYPE_<netron identifier>) failed with CONFIG_UNMATCHED.
TEST_F(TfliteModelParserTest, OriginNameCarriesOperatorsArrayIndex) {
  const auto model_path = WriteMinimalAddModel();
  ASSERT_FALSE(model_path.empty());  // temp file could not be written
  TraceSwitchGuard trace_guard(model_path);
  converter::ConverterParameters params;
  params.fmk = converter::kFmkTypeTflite;
  params.model_file = model_path;
  lite::TfliteModelParser parser;
  auto api_graph = parser.Parse(params);
  ASSERT_NE(api_graph, nullptr);
  auto func_graph = std::dynamic_pointer_cast<FuncGraph>(api_graph->impl());
  ASSERT_NE(func_graph, nullptr);

  // Exactly one node carries origin records: the ADD itself, recorded as ADD_0 (index 0).
  int nodes_with_origin = 0;
  for (auto &node : mindspore::TopoSort(func_graph->get_return())) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    const auto origins = opt::GetOriginNames(cnode);
    if (origins.empty()) {
      continue;
    }
    nodes_with_origin++;
    ASSERT_EQ(origins.size(), 1u);
    EXPECT_EQ(origins[0], "ADD_0");
    EXPECT_EQ(cnode->fullname_with_scope(), "ADD-0");
  }
  EXPECT_EQ(nodes_with_origin, 1);
}
}  // namespace mindspore
