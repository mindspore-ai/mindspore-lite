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

#ifndef CHUNK_GATED_DELTA_RULE_TILING_DATA_H
#define CHUNK_GATED_DELTA_RULE_TILING_DATA_H

#include "kernel_operator.h"  // NOLINT(build/include_subdir)

namespace cgdr {
#pragma pack(push, 8)
struct alignas(8) ChunkGatedDeltaRuleTilingData {
  uint32_t vectorCoreNum;
  uint32_t ubCalSize;
  uint32_t ubRestBytes;
  uint32_t t;    // total sequence length
  uint32_t hqk;  // number of q/k heads
  uint32_t dk;   // key head dim
  uint32_t hv;   // number of value heads
  uint32_t dv;   // value head dim
  uint32_t chunkSize;
  uint32_t numChunks;
  uint32_t b;  // batch size
  uint32_t padSize;
  uint32_t hasInitialState;
  float scaleValue;
  uint32_t vStep;  // tile size for V dimension in state processing
  uint32_t debug;  // debug print flag
};
#pragma pack(pop)
}  // namespace cgdr

#endif  // CHUNK_GATED_DELTA_RULE_TILING_DATA_H
