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
#ifndef MSLLM_NNRT_LOG_H
#define MSLLM_NNRT_LOG_H

/// @brief Lightweight MS_LOG shim for NNRT executor files.
///
/// When building as part of the full mindspore-lite tree (OHOS), the real
/// common/log.h is available and this file is NOT used.  For standalone
/// compilation this provides a zero-dependency fallback mapping MS_LOG to
/// a simple ostream wrapper.

#if !defined(MSLITE_LLM_NNRT_USE_REAL_LOG)

#include <iostream>

namespace mslite {
namespace backend {
namespace nnrt {

struct NnrtLogLine {
  std::ostream &os;
  ~NnrtLogLine() { os << '\n'; }
  template <typename T>
  NnrtLogLine &operator<<(const T &v) {
    os << v;
    return *this;
  }
};

}  // namespace nnrt
}  // namespace backend
}  // namespace mslite

// Chain:  MS_LOG(ERROR) → MS_LOG_ERROR → MSLOG_IF(::mslite::backend::nnrt::ERROR)
// But NNRT source calls MS_LOG(ERROR), which the real header expands to MS_LOG_ERROR.
// We short-circuit that by defining the per-level macros directly.
//
// The NNRT .cc files write:  MS_LOG(ERROR) << "msg";
// mindspore-lite expands MS_LOG(level) to MS_LOG_##level, then e.g. MS_LOG_ERROR to MSLOG_IF(...).
// We define MS_LOG(level) to expand to a streaming expression directly.

#define MS_LOG_DEBUG ::mslite::backend::nnrt::NnrtLogLine{std::cerr} << "[DEBUG] nnrt: "
#define MS_LOG_INFO ::mslite::backend::nnrt::NnrtLogLine{std::cerr} << "[INFO]  nnrt: "
#define MS_LOG_WARNING ::mslite::backend::nnrt::NnrtLogLine{std::cerr} << "[WARN]  nnrt: "
#define MS_LOG_ERROR ::mslite::backend::nnrt::NnrtLogLine{std::cerr} << "[ERROR] nnrt: "

#define MS_LOG(level) MS_LOG_##level

#else
// Use the real mindspore-lite logging — common/log.h must be in the include path.
#include "common/log.h"
#endif

#endif  // MSLLM_NNRT_LOG_H
