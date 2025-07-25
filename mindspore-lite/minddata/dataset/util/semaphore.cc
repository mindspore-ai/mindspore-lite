/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "mindspore-lite/minddata/dataset/util/semaphore.h"
#include "mindspore-lite/minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
Status Semaphore::P() {
  std::unique_lock<std::mutex> lck(mutex_);
  RETURN_IF_NOT_OK(wait_cond_.Wait(&lck, [this]() { return value_ > 0; }));
  --value_;
  return Status::OK();
}
void Semaphore::V() {
  std::unique_lock<std::mutex> lck(mutex_);
  ++value_;
  wait_cond_.NotifyOne();
}
int Semaphore::Peek() const { return value_; }
Status Semaphore::Register(TaskGroup *vg) { return wait_cond_.Register(vg->GetIntrpService()); }
Status Semaphore::Deregister() { return (wait_cond_.Deregister()); }
void Semaphore::ResetIntrpState() { wait_cond_.ResetIntrpState(); }

}  // namespace dataset
}  // namespace mindspore
