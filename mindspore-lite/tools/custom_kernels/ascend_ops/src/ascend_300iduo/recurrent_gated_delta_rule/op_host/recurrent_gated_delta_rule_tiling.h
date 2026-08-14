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

#ifndef RECURRENT_GATED_DELTA_RULE_TILING_H
#define RECURRENT_GATED_DELTA_RULE_TILING_H

#include <cstdint>

#pragma pack(push, 8)
struct alignas(8) RecurrentGatedDeltaRuleTilingData {
  uint32_t vectorCoreNum;
  uint32_t ubCalSize;
  uint32_t ubRestBytes;
  uint32_t t;
  uint32_t nk;
  uint32_t dk;
  uint32_t nv;
  uint32_t dv;
  uint32_t sBlockNum;
  uint32_t b;
  uint32_t vStep;
  uint32_t stateOutBufferNum;
  uint32_t attnOutBufferNum;
  float scale;
  uint32_t hasGama;
  uint32_t hasGamaK;
  uint32_t hasAcceptedTokens;
  uint32_t gamaKScalar;        // 1: gamaK is per-head scalar [T, NV], broadcast to DK; 0: per-head vector [T, NV*DK]
  uint32_t cuSeqlensIsPrefix;  // 1: cu_seqlens style [B+1]; 0: actual_seq_lengths style [B]
  uint32_t cuSeqlensIsInt64;   // 1: input dtype is int64; 0: int32
  uint32_t ssmStateIndicesIsInt64;  // 1: input dtype is int64; 0: int32
  uint32_t reserved;
};
#pragma pack(pop)

#endif  // RECURRENT_GATED_DELTA_RULE_TILING_H
