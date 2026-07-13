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

#include <gtest/gtest.h>
#include <vector>
#include "flatbuffers/flatbuffers.h"
#include "schema/model_generated.h"
#include "src/common/common.h"
#include "src/litert/lite_model.h"

namespace mindspore {
namespace lite {

TEST(FlatBuffersVerifierTest, RejectModelExceedingTableLimit) {
  flatbuffers::FlatBufferBuilder builder;

  schema::TensorBuilder tensor_builder(builder);
  auto tensor = tensor_builder.Finish();

  // Reusing one table offset keeps the malicious buffer compact. The verifier
  // still visits and counts the table for every vector element.
  std::vector<flatbuffers::Offset<schema::Tensor>> tensors(FLATBUFFERS_MAX_TABLES, tensor);
  auto all_tensors = builder.CreateVector(tensors);

  schema::MetaGraphBuilder graph_builder(builder);
  graph_builder.add_allTensors(all_tensors);
  schema::FinishMetaGraphBuffer(builder, graph_builder.Finish());

  // The root MetaGraph and its tensor references make the verifier visit
  // FLATBUFFERS_MAX_TABLES + 1 tables in total.
  flatbuffers::Verifier relaxed_verifier(builder.GetBufferPointer(), builder.GetSize(), FLATBUFFERS_MAX_DEPTH,
                                         FLATBUFFERS_MAX_TABLES + 1);
  ASSERT_TRUE(schema::VerifyMetaGraphBuffer(relaxed_verifier));

  LiteModel model;
  EXPECT_EQ(model.ConstructModel(reinterpret_cast<const char *>(builder.GetBufferPointer()), builder.GetSize(), true),
            RET_ERROR);
}

}  // namespace lite
}  // namespace mindspore
