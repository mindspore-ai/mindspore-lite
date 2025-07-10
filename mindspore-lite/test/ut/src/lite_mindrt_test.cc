/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "actor/actor.h"
#include "actor/op_actor.h"
#include "async/uuid_base.h"
#include "async/future.h"
#include "src/litert/lite_mindrt.h"
#include "thread/hqueue.h"
#include "thread/actor_threadpool.h"
#include "common/common_test.h"
#include "schema/model_generated.h"

namespace mindspore {
class LiteMindRtTest : public mindspore::CommonTest {
 public:
  LiteMindRtTest() {}
};

class TestActor : public ActorBase {
 public:
  explicit TestActor(const std::string &nm, ActorThreadPool *pool, const int i) : ActorBase(nm, pool), data(i) {}
  int Fn1(int *val) {
    if (val) {
      (*val)++;
    }
    return 0;
  }
  int Fn2(int *val) { return data + (*val); }

 public:
  int data = 0;
};

TEST_F(LiteMindRtTest, ActorThreadPoolTest) {
  Initialize("", "", "", "", 40);
  auto pool = ActorThreadPool::CreateThreadPool(40);
  std::vector<AID> actors;
  for (size_t i = 0; i < 200; i++) {
    AID t1 = Spawn(ActorReference(new TestActor(std::to_string(i), pool, i)));
    actors.emplace_back(t1);
  }

  std::vector<int *> vv;
  std::vector<Future<int>> fv;
  std::vector<int> expected;
  size_t sz = 300;
  for (size_t i = 0; i < sz; i++) {
    int data = 0;
    for (auto a : actors) {
      int *val = new int(i);
      vv.emplace_back(val);
      Future<int> ret;
      ret = Async(a, &TestActor::Fn1, val)
              .Then(Defer(a, &TestActor::Fn2, val), ret);
      fv.emplace_back(ret);
      expected.emplace_back(data + i + 1);
      data++;
    }
  }
  for (size_t i = 0; i < vv.size(); i++) {
    int ret = fv[i].Get();
    ASSERT_EQ(ret, expected[i]);
  }
  Finalize();

  for (size_t i = 0; i < vv.size(); i++) {
    delete vv[i];
  }
}

}  // namespace mindspore
