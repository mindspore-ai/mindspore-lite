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

/* fused_causal_conv_silu_tiling.cpp Host tiling implementation for FusedCausalConvSilu. */

#include "fused_causal_conv_silu_tiling.h"  // NOLINT(build/include_subdir): AscendC op headers are included relative to the op directory
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

namespace {
constexpr int64_t K_TAPS = 4;
constexpr int64_t VAL_ELEMS = 8;
constexpr int64_t ALIGN16 = 16;
constexpr int64_t INPUT_X_DIM_NUM = 3;   // x: [B, T, S]
constexpr int64_t INPUT_W_DIM_NUM = 2;   // w: [K, C]
constexpr size_t INPUT_DESC_NUM = 4;     // x, weight, bias, seq_len
constexpr size_t INPUT_IDX_SEQ_LEN = 3;  // seq_len is the 4th input
constexpr size_t OUTPUT_DESC_NUM = 2;    // y, conv_state

ge::graphStatus TilingFunc(gert::TilingContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }

  auto shapeX = context->GetInputShape(0);
  auto shapeW = context->GetInputShape(1);
  auto shapeB = context->GetInputShape(2);
  auto shapeVal = context->GetInputShape(3);
  if (shapeX == nullptr || shapeW == nullptr || shapeB == nullptr || shapeVal == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto &xO = shapeX->GetOriginShape();
  const auto &wO = shapeW->GetOriginShape();
  const auto &bO = shapeB->GetOriginShape();
  const auto &valO = shapeVal->GetOriginShape();

  if (xO.GetDimNum() != INPUT_X_DIM_NUM || wO.GetDimNum() != INPUT_W_DIM_NUM || bO.GetDimNum() != 1 ||
      valO.GetDimNum() != 1) {
    return ge::GRAPH_FAILED;
  }
  const int64_t B = xO.GetDim(0);
  const int64_t T = xO.GetDim(1);
  const int64_t S = xO.GetDim(2);
  const int64_t K = wO.GetDim(0);
  const int64_t C = wO.GetDim(1);
  if (B != 1 || T < 1) {
    return ge::GRAPH_FAILED;
  }
  if (K != K_TAPS) {
    return ge::GRAPH_FAILED;
  }
  if (C < ALIGN16 || C % ALIGN16 != 0 || C % FUSED_CONV_TILE_C != 0) {
    return ge::GRAPH_FAILED;
  }
  if (S < C || S % ALIGN16 != 0) {
    return ge::GRAPH_FAILED;
  }
  if (bO.GetDim(0) != C || valO.GetDim(0) != VAL_ELEMS) {
    return ge::GRAPH_FAILED;
  }

  for (size_t i = 0; i < INPUT_DESC_NUM; i++) {
    auto desc = context->GetInputDesc(i);
    if (desc == nullptr) {
      return ge::GRAPH_FAILED;
    }
    if (i == INPUT_IDX_SEQ_LEN) {
      if (desc->GetDataType() != ge::DT_INT32) {
        return ge::GRAPH_FAILED;
      }
    } else if (desc->GetDataType() != ge::DT_BF16) {
      return ge::GRAPH_FAILED;
    }
  }
  for (size_t i = 0; i < OUTPUT_DESC_NUM; i++) {
    auto desc = context->GetOutputDesc(i);
    if (desc == nullptr || desc->GetDataType() != ge::DT_BF16) {
      return ge::GRAPH_FAILED;
    }
  }

  auto platformInfoPtr = context->GetPlatformInfo();
  if (platformInfoPtr == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
  if (aivNum == 0) {
    return ge::GRAPH_FAILED;
  }
  int64_t usedCoreNum = static_cast<int64_t>(aivNum);

  auto td = context->GetTilingData<FusedCausalConvSiluTilingData>();
  if (td == nullptr) {
    return ge::GRAPH_FAILED;
  }
  td->T = T;
  td->C = C;
  td->S = S;
  td->K = K;
  td->usedCoreNum = usedCoreNum;

  context->SetTilingKey(0);
  context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
  size_t *workspaces = context->GetWorkspaceSizes(1);
  if (workspaces == nullptr) {
    return ge::GRAPH_FAILED;
  }
  workspaces[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}  // namespace

ge::graphStatus FusedCausalConvSiluTilingFunc(gert::TilingContext *context) { return TilingFunc(context); }

IMPL_OP_OPTILING(FusedCausalConvSilu).Tiling(FusedCausalConvSiluTilingFunc, sizeof(FusedCausalConvSiluTilingData));

}  // namespace optiling
