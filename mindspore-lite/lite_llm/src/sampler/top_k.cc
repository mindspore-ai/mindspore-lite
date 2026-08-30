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
#include <numeric>
#include <vector>

namespace mslite_llm {

void ApplyTopKFilter(std::vector<float> &probs, int32_t top_k) {
  if (top_k <= 0 || top_k >= static_cast<int32_t>(probs.size())) {
    return;
  }

  std::vector<size_t> indices(probs.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                    [&probs](size_t a, size_t b) { return probs[a] > probs[b]; });

  std::vector<size_t> keep(indices.begin(), indices.begin() + top_k);
  std::sort(keep.begin(), keep.end());

  std::vector<float> filtered(probs.size(), 0.0f);
  for (auto idx : keep) {
    filtered[idx] = probs[idx];
  }
  probs = std::move(filtered);

  float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
  if (sum > 0.0f) {
    std::transform(probs.begin(), probs.end(), probs.begin(), [sum](float v) { return v / sum; });
  }
}

}  // namespace mslite_llm
