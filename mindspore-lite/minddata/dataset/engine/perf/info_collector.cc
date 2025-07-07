/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "mindspore-lite/minddata/dataset/engine/perf/info_collector.h"
#include "debug/profiler/profiling.h"

namespace mindspore::dataset {

uint64_t GetSyscnt() {
  uint64_t time_cnt = 0;
#if !defined(ENABLE_ANDROID)
  time_cnt = profiler::GetClockSyscnt();
#endif
  return time_cnt;
}

double GetMilliTimeStamp() {
  auto now = std::chrono::high_resolution_clock::now();
  int64_t us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
  return static_cast<double>(us) / 1000.;
}

Status CollectPipelineInfo(const std::string &event, const std::string &stage, const uint64_t &start_time,
                           const std::map<std::string, std::string> &custom_info) {
#if !defined(ENABLE_ANDROID)
  (void)profiler::CollectHostInfo("Dataset", event, stage, start_time, profiler::GetClockSyscnt(), InfoLevel::kUser,
                                  custom_info);
#endif
  return Status::OK();
}

Status CollectOpInfo(const std::string &event, const std::string &stage, const uint64_t &start_time,
                     const std::map<std::string, std::string> &custom_info) {
#if !defined(ENABLE_ANDROID)
  (void)profiler::CollectHostInfo("Dataset", event, stage, start_time, profiler::GetClockSyscnt(),
                                  InfoLevel::kDeveloper, custom_info);
#endif
  return Status::OK();
}
}  // namespace mindspore::dataset
