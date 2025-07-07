/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SIG_HANDLER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SIG_HANDLER_H_

#include <cstdint>
#include <set>

namespace mindspore::dataset {
/// \brief Register the custom signal handlers.
extern void RegisterHandlers();

/// \brief Register signal handlers for main process.
extern void RegisterMainHandlers();

/// \brief Register signal handlers for worker process.
extern void RegisterWorkerHandlers();

/// \brief Register workers to be monitored by the watch dog.
extern void RegisterWorkerPIDs(int64_t id, const std::set<int> &pids);

/// \brief Deregister workers to be monitored by the watch dog.
extern void DeregisterWorkerPIDs(int64_t id);
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_SIG_HANDLER_H_
