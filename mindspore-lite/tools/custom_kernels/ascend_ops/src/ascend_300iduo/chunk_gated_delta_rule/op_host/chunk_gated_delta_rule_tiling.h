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

#ifndef CHUNK_GATED_DELTA_RULE_TILING_H
#define CHUNK_GATED_DELTA_RULE_TILING_H

#include <cstdint>

// ChunkGatedDeltaRuleTilingData MUST stay global (mirror of the kernel-side
// chunk_gated_delta_rule_tiling_data.h): the two structs must be byte-identical
// and the CANN tiling macros require a global struct. The byte stream the host
// writes here is read by the device via GET_TILING_DATA.
// kTilingDataAlign mirrors the device-side constant; both #pragma pack and
// alignas must use the same value.
constexpr int kTilingDataAlign = 8;
#pragma pack(push, 8)
struct alignas(kTilingDataAlign) ChunkGatedDeltaRuleTilingData {
  uint32_t vectorCoreNum;
  uint32_t ubCalSize;
  uint32_t ubRestBytes;
  uint32_t t;          // total sequence length (padded)
  uint32_t hqk;        // number of q/k heads
  uint32_t dk;         // key head dim
  uint32_t hv;         // number of value heads
  uint32_t dv;         // value head dim
  uint32_t chunkSize;  // chunk size (e.g., 64)
  uint32_t numChunks;  // number of chunks
  uint32_t b;          // batch size
  uint32_t padSize;    // padding size
  uint32_t hasInitialState;
  float scaleValue;
  uint32_t vStep;
  uint32_t debug;
};
#pragma pack(pop)

#endif  // CHUNK_GATED_DELTA_RULE_TILING_H
