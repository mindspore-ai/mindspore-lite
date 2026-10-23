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
#include <cstdlib>
#include <cstring>
#include <vector>
#include "common/common_test.h"
#include "nnacl_c/errorcode.h"
#include "nnacl_c/kernel.h"
#include "nnacl_c/kernel/where.h"

extern "C" int WhereRunWithTripleInputs(WhereStruct *where);

namespace mindspore {
namespace {
void *TestAlloc(void *allocator, size_t size) {
  (void)allocator;
  return malloc(size);
}

void TestFree(void *allocator, void *ptr) {
  (void)allocator;
  free(ptr);
}

int TestParallelLaunch(void *thread_pool, void *task, void *param, int task_num) {
  (void)thread_pool;
  auto run = reinterpret_cast<int (*)(void *, int, float, float)>(task);
  for (int i = 0; i < task_num; ++i) {
    int ret = run(param, i, 0.0f, 1.0f);
    if (ret != NNACL_OK) {
      return ret;
    }
  }
  return NNACL_OK;
}
}  // namespace

class WhereBroadcastFp32Test : public mindspore::CommonTest {
 public:
  WhereBroadcastFp32Test() = default;

 protected:
  void SetShape(TensorC *tensor, const std::vector<int> &shape) {
    tensor->shape_size_ = shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
      tensor->shape_[i] = shape[i];
    }
  }

  void InitWhere(const std::vector<int> &condition_shape, const std::vector<int> &x_shape,
                 const std::vector<int> &y_shape, const std::vector<int> &output_shape) {
    memset(&env_, 0, sizeof(env_));
    env_.Alloc = TestAlloc;
    env_.Free = TestFree;
    env_.ParallelLaunch = TestParallelLaunch;

    memset(&where_, 0, sizeof(where_));
    where_.base_.env_ = &env_;
    where_.base_.thread_nr_ = 1;
    where_.data_type_ = kNumberTypeFloat32;

    memset(&condition_, 0, sizeof(condition_));
    memset(&x_, 0, sizeof(x_));
    memset(&y_, 0, sizeof(y_));
    memset(&output_, 0, sizeof(output_));
    condition_.data_type_ = kNumberTypeFloat32;
    x_.data_type_ = kNumberTypeFloat32;
    y_.data_type_ = kNumberTypeFloat32;
    output_.data_type_ = kNumberTypeFloat32;
    SetShape(&condition_, condition_shape);
    SetShape(&x_, x_shape);
    SetShape(&y_, y_shape);
    SetShape(&output_, output_shape);

    inputs_[0] = &condition_;
    inputs_[1] = &x_;
    inputs_[2] = &y_;
    outputs_[0] = &output_;
    where_.base_.in_ = inputs_;
    where_.base_.in_size_ = 3;
    where_.base_.out_ = outputs_;
    where_.base_.out_size_ = 1;
  }

  ExecEnv env_;
  WhereStruct where_;
  TensorC condition_;
  TensorC x_;
  TensorC y_;
  TensorC output_;
  TensorC *inputs_[3];
  TensorC *outputs_[1];
};

TEST_F(WhereBroadcastFp32Test, BroadcastableShapesRunSuccessfully) {
  InitWhere({2, 1}, {1, 3}, {2, 3}, {2, 3});
  bool condition_data[] = {true, false};
  float x_data[] = {1.0f, 2.0f, 3.0f};
  float y_data[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
  float output_data[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  where_.base_.in_[0]->data_ = condition_data;
  where_.base_.in_[1]->data_ = x_data;
  where_.base_.in_[2]->data_ = y_data;
  where_.base_.out_[0]->data_ = output_data;

  ASSERT_EQ(WhereRunWithTripleInputs(&where_), NNACL_OK);
  ASSERT_EQ(where_.args_.condition_num_, 6);
  ASSERT_EQ(where_.args_.x_num_, 6);
  ASSERT_EQ(where_.args_.y_num_, 6);
  ASSERT_EQ(where_.args_.max_num_, 6);
  const float expect[] = {1.0f, 2.0f, 3.0f, 40.0f, 50.0f, 60.0f};
  ASSERT_EQ(CompareOutputData(output_data, expect, 6), 0);
}

TEST_F(WhereBroadcastFp32Test, NonBroadcastableShapeReturnsInvalid) {
  InitWhere({2}, {3}, {4}, {4});
  bool condition_data[] = {true, false};
  float x_data[] = {1.0f, 2.0f, 3.0f};
  float y_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
  float output_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
  where_.base_.in_[0]->data_ = condition_data;
  where_.base_.in_[1]->data_ = x_data;
  where_.base_.in_[2]->data_ = y_data;
  where_.base_.out_[0]->data_ = output_data;

  ASSERT_EQ(WhereRunWithTripleInputs(&where_), NNACL_WHERE_CONDITION_NUM_INVALID);
}

TEST_F(WhereBroadcastFp32Test, InputRankLargerThanOutputRankReturnsInvalid) {
  InitWhere({6}, {6}, {3, 1}, {6});
  bool condition_data[] = {true, false, true, false, true, false};
  float x_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float y_data[] = {10.0f, 20.0f, 30.0f};
  float output_data[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  where_.base_.in_[0]->data_ = condition_data;
  where_.base_.in_[1]->data_ = x_data;
  where_.base_.in_[2]->data_ = y_data;
  where_.base_.out_[0]->data_ = output_data;

  ASSERT_EQ(WhereRunWithTripleInputs(&where_), NNACL_WHERE_CONDITION_NUM_INVALID);
}
}  // namespace mindspore
