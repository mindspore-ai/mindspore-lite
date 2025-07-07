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

#define USE_DEPRECATED_API
#include <memory>
#include <map>
#include <string>
#include "include/registry/pass_registry.h"
#include "include/registry/pass_base.h"
#include "common/common_test.h"
#include "src/common/log_adapter.h"

using mindspore::registry::POSITION_ASCEND;
namespace mindspore {
class PassRegistryPositionAscendTest : public mindspore::CommonTest {
 public:
  PassRegistryPositionAscendTest() = default;
  api::FuncGraphPtr func_graph_ = nullptr;
};

namespace opt {
class PassTutorial : public registry::PassBase {
 public:
  PassTutorial() : PassBase("PassTutorial") {}

  bool Execute(const api::FuncGraphPtr &func_graph) override {
    MS_LOG(INFO) << "PassTutorial Execute";
    return true;
  }
};

REG_PASS(PassTutorial, opt::PassTutorial)
REG_SCHEDULED_PASS(POSITION_ASCEND, {"PassTutorial"})
}  // namespace opt

TEST_F(PassRegistryPositionAscendTest, RunPassAtPositionAscend) {
  auto schedule_task = registry::PassRegistry::GetOuterScheduleTask(POSITION_ASCEND);
  ASSERT_EQ(schedule_task.size(), 1);
  auto pass = registry::PassRegistry::GetPassFromStoreRoom("PassTutorial");
  ASSERT_NE(pass, nullptr);
  auto ret = pass->Execute(func_graph_);
  ASSERT_EQ(ret, true);
  MS_LOG(INFO) << "PASS";
}
}  // namespace mindspore
