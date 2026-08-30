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
#include <algorithm>
#include <cstdint>
#include <vector>

namespace mslite_llm {

int32_t GreedySample(const std::vector<float> &logits) {
  if (logits.empty()) {
    return 0;
  }
  int32_t argmax = 0;
  float max_val = logits[0];
  for (size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > max_val) {
      max_val = logits[i];
      argmax = static_cast<int32_t>(i);
    }
  }
  return argmax;
}

}  // namespace mslite_llm
