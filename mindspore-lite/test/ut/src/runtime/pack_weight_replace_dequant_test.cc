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
#ifdef SHARING_MODEL_WEIGHT
#include <string>
#include "common/common_test.h"
#include "src/litert/pack_weight.h"

namespace mindspore {
namespace {
constexpr char kTestModelId[] = "ut_model_id";
constexpr int kTestNumaId = 0;
constexpr size_t kTestModelSize = 8;
constexpr size_t kTestDataSize = 4;
}  // namespace

class PackWeightReplaceDequantTest : public mindspore::CommonTest {
 public:
  PackWeightReplaceDequantTest() = default;
};

TEST_F(PackWeightReplaceDequantTest, ReplaceDequantDataUpdatesMapping) {
  mindspore::lite::PackWeight pack_weight;
  char model_buf[kTestModelSize] = {0};
  ASSERT_EQ(pack_weight.InitPackWeight(model_buf, kTestModelSize, kTestModelId, kTestNumaId, true),
            mindspore::lite::RET_OK);
  char *shared_buf = pack_weight.GetSharedModelBuf(kTestModelId, kTestNumaId);
  ASSERT_NE(shared_buf, nullptr);

  int origin_data = 1;
  int new_data = 2;
  ASSERT_EQ(pack_weight.StoreOriginTensorData(shared_buf, &origin_data), mindspore::lite::RET_OK);

  ASSERT_EQ(pack_weight.ReplaceDequantData(&origin_data, &new_data), mindspore::lite::RET_OK);

  bool is_packed = false;
  void *packed_data = pack_weight.GetPackData(&new_data, kTestDataSize, &is_packed);
  ASSERT_NE(packed_data, nullptr);
  ASSERT_FALSE(is_packed);
  ASSERT_EQ(pack_weight.GetPackData(&origin_data, kTestDataSize, &is_packed), nullptr);
}

TEST_F(PackWeightReplaceDequantTest, ReplaceDequantDataNullArgumentReturnsError) {
  mindspore::lite::PackWeight pack_weight;
  char model_buf[kTestModelSize] = {0};
  ASSERT_EQ(pack_weight.InitPackWeight(model_buf, kTestModelSize, kTestModelId, kTestNumaId, true),
            mindspore::lite::RET_OK);

  int new_data = 2;
  int origin_data = 1;
  ASSERT_EQ(pack_weight.ReplaceDequantData(nullptr, &new_data), mindspore::lite::RET_ERROR);
  ASSERT_EQ(pack_weight.ReplaceDequantData(&origin_data, nullptr), mindspore::lite::RET_ERROR);
}

TEST_F(PackWeightReplaceDequantTest, ReplaceDequantDataNotFoundReturnsOk) {
  mindspore::lite::PackWeight pack_weight;
  char model_buf[kTestModelSize] = {0};
  ASSERT_EQ(pack_weight.InitPackWeight(model_buf, kTestModelSize, kTestModelId, kTestNumaId, true),
            mindspore::lite::RET_OK);

  int origin_data = 1;
  int new_data = 2;
  ASSERT_EQ(pack_weight.ReplaceDequantData(&origin_data, &new_data), mindspore::lite::RET_OK);
}
}  // namespace mindspore
#endif  // SHARING_MODEL_WEIGHT
