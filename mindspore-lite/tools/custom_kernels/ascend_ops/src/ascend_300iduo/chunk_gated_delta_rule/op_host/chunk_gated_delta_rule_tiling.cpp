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

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "chunk_gated_delta_rule_tiling.h"  // NOLINT(build/include_subdir)

using namespace ge;    // NOLINT(build/namespaces)
using namespace gert;  // NOLINT(build/namespaces)

// ChunkGatedDeltaRuleTilingData is global (see chunk_gated_delta_rule_tiling.h),
// so no using-declaration is needed here.

namespace {

constexpr uint32_t FP16_NUM_PER_BLOCK = 16;
constexpr uint32_t FP32_NUM_PER_BLOCK = 8;
// Default chunk size (tokens per chunk) when the op attr is unset.
constexpr uint32_t kDefaultChunkSize = 64;
// Axis index of the head-dim within the [T, H, D] query/value layouts.
constexpr int32_t kHeadDimAxis = 2;
// chunkKFp32 & kCumdecayFp32 (and chunkVFp32 & chunkAttnOutFp32) each share the
// same layout, so their byte cost is counted twice when sizing tmpBuff.
constexpr uint32_t kPairedTileCount = 2;
// Fixed workspace size requested for the op (32 MiB).
constexpr uint64_t kWorkspaceBytes = 32ULL * 1024 * 1024;
// Input indices for the CGDR operator.
constexpr uint32_t kQueryInput = 0;
constexpr uint32_t kValueInput = 2;
// Shape dimension indices within the [T, H, D] layout.
constexpr uint32_t kDimT = 0;
constexpr uint32_t kDimH = 1;
// Minimum block dimension for AIV core dispatch.
constexpr uint32_t kMinBlockDim = 1;
// Number of workspace slots requested from the runtime.
constexpr uint32_t kWorkspaceCount = 1;

uint32_t CeilAlign(uint32_t val, uint32_t align) { return (val + align - 1) / align * align; }

// Collected query/value shape dimensions used across the tiling helpers.
struct ShapeDims {
  uint32_t t;    // total sequence length
  uint32_t hqk;  // number of q/k heads
  uint32_t dk;   // key head dim
  uint32_t hv;   // number of value heads
  uint32_t dv;   // value head dim
};

// Query the platform for the AIV core count and UB size. Returns false on any null handle.
bool GetPlatformInfo(TilingContext *context, uint32_t &aivNum, int64_t &ubSize) {
  if (context == nullptr) {
    return false;
  }
  auto platformInfo = context->GetPlatformInfo();
  if (platformInfo == nullptr) {
    return false;
  }
  platform_ascendc::PlatformAscendC platform(platformInfo);
  aivNum = platform.GetCoreNumAiv();
  if (aivNum == 0) {
    aivNum = platform.GetCoreNum();
  }
  uint64_t ubSize64 = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize64);
  ubSize = static_cast<int64_t>(ubSize64);
  return true;
}

// Extract the [T, Hqk, Dk] query and [*, Hv, Dv] value dims. Returns false on a null shape handle.
bool ParseShapeDims(TilingContext *context, ShapeDims &dims) {
  auto queryShape = context->GetInputShape(kQueryInput);
  auto valueShape = context->GetInputShape(kValueInput);
  if (queryShape == nullptr || valueShape == nullptr) {
    return false;
  }
  const auto &qDims = queryShape->GetStorageShape();
  const auto &vDims = valueShape->GetStorageShape();
  dims.t = qDims.GetDim(kDimT);
  dims.hqk = qDims.GetDim(kDimH);
  dims.dk = qDims.GetDim(kHeadDimAxis);
  dims.hv = vDims.GetDim(kDimH);
  dims.dv = vDims.GetDim(kHeadDimAxis);
  return true;
}

// Read optional attrs chunk_size (idx 0) and scale_value (idx 1), falling back to defaults.
void ParseAttrs(TilingContext *context, int64_t &chunkSizeAttr, float &scaleValueAttr) {
  chunkSizeAttr = kDefaultChunkSize;
  scaleValueAttr = 1.0f;
  auto attrs = context->GetAttrs();
  if (attrs == nullptr || attrs->GetAttrNum() == 0) {
    return;
  }
  const int64_t *chunkPtr = attrs->GetAttrPointer<int64_t>(0);
  if (chunkPtr != nullptr) {
    chunkSizeAttr = *chunkPtr;
  }
  if (attrs->GetAttrNum() > 1) {
    const float *scalePtr = attrs->GetAttrPointer<float>(1);
    if (scalePtr != nullptr) {
      scaleValueAttr = *scalePtr;
    }
  }
}

// Total tmpBuff bytes (all FP32 tiles) for a candidate vStep. chunkKFp32 & kCumdecayFp32
// share the [cs, alignK] layout; chunkVFp32 & chunkAttnOutFp32 share [cs, avFp32] — each
// counted twice. chunkScoresFp32 and stateInFp32 overlap, so only the max is reserved.
uint32_t ComputeTmpBuffBytes(uint32_t vs, uint32_t chunkSize, uint32_t dk, uint32_t alignK, uint32_t avFp32) {
  uint32_t avStepAligned = CeilAlign(vs, FP32_NUM_PER_BLOCK);
  uint32_t cs = chunkSize;
  uint32_t kTileBytes = cs * alignK * sizeof(float);  // chunkKFp32 + kCumdecayFp32
  uint32_t vTileBytes = cs * avFp32 * sizeof(float);  // chunkVFp32 + chunkAttnOutFp32
  uint32_t tDecay = cs * cs * sizeof(float);          // decayMaskFp32
  uint32_t tGcum = cs * sizeof(float);                // gCumsumFp32
  uint32_t stateStrideK = CeilAlign(dk, FP16_NUM_PER_BLOCK);
  uint32_t tDelta = ((stateStrideK > cs) ? stateStrideK : cs) * sizeof(float);  // deltaFp32
  uint32_t tExpGcum = cs * sizeof(float);                                       // expGCumFp32
  uint32_t tScores = cs * cs * sizeof(float);                                   // chunkScoresFp32 (attn matrix)
  uint32_t tState = dk * avStepAligned * sizeof(float);                         // stateInFp32 [DK, vStep]
  uint32_t tOverlap = (tScores > tState) ? tScores : tState;
  return kPairedTileCount * kTileBytes + tDecay + kPairedTileCount * vTileBytes + tGcum + tDelta + tExpGcum + tOverlap;
}

// stateOutQueue bytes: max of the state tile [dk, avStepAligned] and the chunk attn output [cs, avFp32].
uint32_t ComputeOutQueueBytes(uint32_t vs, uint32_t dk, uint32_t chunkSize, uint32_t avFp32) {
  uint32_t avStepAligned = CeilAlign(vs, FP32_NUM_PER_BLOCK);
  uint32_t stateBytes = dk * avStepAligned * sizeof(uint16_t);
  uint32_t chunkBytes = chunkSize * avFp32 * sizeof(uint16_t);
  return (stateBytes > chunkBytes) ? stateBytes : chunkBytes;
}

// Find the largest vStep (FP32-block aligned, <= dv) whose tmpBuff + outQueue fits in UB.
// Also returns the resulting buffer sizes for tiling/debug.
void SolveVStep(uint32_t dv, int64_t ubSize, uint32_t chunkSize, uint32_t dk, uint32_t alignK, uint32_t avFp32,
                uint32_t &vStepVal, uint32_t &tbufTotal, uint32_t &outQueueMax, uint32_t &restBytes) {
  uint32_t maxVStep = CeilAlign(dv, FP16_NUM_PER_BLOCK);
  vStepVal = FP16_NUM_PER_BLOCK;
  for (uint32_t vs = maxVStep; vs >= FP16_NUM_PER_BLOCK; vs -= FP16_NUM_PER_BLOCK) {
    if (ComputeTmpBuffBytes(vs, chunkSize, dk, alignK, avFp32) + ComputeOutQueueBytes(vs, dk, chunkSize, avFp32) <=
        static_cast<uint32_t>(ubSize)) {
      vStepVal = vs;
      break;
    }
  }
  if (vStepVal > dv) {
    vStepVal = CeilAlign(dv, FP16_NUM_PER_BLOCK);
  }
  tbufTotal = ComputeTmpBuffBytes(vStepVal, chunkSize, dk, alignK, avFp32);
  outQueueMax = ComputeOutQueueBytes(vStepVal, dk, chunkSize, avFp32);
  restBytes = (ubSize > static_cast<int64_t>(outQueueMax)) ? static_cast<uint32_t>(ubSize - outQueueMax) : 0;
}

// Write the resolved scalar parameters into the device-visible tiling struct.
void FillTilingData(ChunkGatedDeltaRuleTilingData *td, const ShapeDims &dims, uint32_t aivNum, int64_t ubSize,
                    uint32_t chunkSize, uint32_t numChunks, uint32_t padSize, float scaleValue, uint32_t vStepVal,
                    uint32_t restBytes) {
  td->vectorCoreNum = aivNum;
  td->ubCalSize = static_cast<uint32_t>(ubSize);
  td->ubRestBytes = restBytes;
  td->t = dims.t;
  td->hqk = dims.hqk;
  td->dk = dims.dk;
  td->hv = dims.hv;
  td->dv = dims.dv;
  td->chunkSize = chunkSize;
  td->numChunks = numChunks;
  td->b = 1;
  td->padSize = padSize;
  td->hasInitialState = 1;
  td->scaleValue = scaleValue;
  td->vStep = vStepVal;
  td->debug = 0;
}

// Optional stderr dump, gated by CGDR_DEBUG_LOG=1. No-op otherwise.
void LogTiling(const ShapeDims &dims, uint32_t chunkSize, uint32_t numChunks, float scaleValue, int64_t ubSize,
               uint32_t vStepVal, uint32_t blockDim, uint32_t outQueueMax, uint32_t tbufTotal, uint32_t restBytes) {
  const char *tilingDbg = std::getenv("CGDR_DEBUG_LOG");
  if (tilingDbg == nullptr || tilingDbg[0] != '1') {
    return;
  }
  fprintf(stderr,
          "ChunkGatedDeltaRule tiling: T=%u, chunkSize=%u, numChunks=%u, hqk=%u, dk=%u, hv=%u, dv=%u, "
          "scaleValue=%f, ubSize=%ld, vStep=%u, blockDim=%u, outQueue=%u, tbuf=%u, restBytes=%u\n",
          dims.t, chunkSize, numChunks, dims.hqk, dims.dk, dims.hv, dims.dv, scaleValue, ubSize, vStepVal, blockDim,
          outQueueMax, tbufTotal, restBytes);
  fflush(stderr);
}

static uint32_t ChunkGatedDeltaRuleTilingFunc(TilingContext *context) {
  uint32_t aivNum = 0;
  int64_t ubSize = 0;
  if (!GetPlatformInfo(context, aivNum, ubSize)) {
    return GRAPH_FAILED;
  }

  ShapeDims dims;
  if (!ParseShapeDims(context, dims)) {
    return GRAPH_FAILED;
  }

  int64_t chunkSizeAttr = 0;
  float scaleValueAttr = 1.0f;
  ParseAttrs(context, chunkSizeAttr, scaleValueAttr);
  uint32_t chunkSize = static_cast<uint32_t>(chunkSizeAttr);

  // Pad T up to a multiple of chunkSize and pre-compute the FP16/FP32 block alignments.
  uint32_t padSize = (chunkSize - dims.t % chunkSize) % chunkSize;
  uint32_t numChunks = (dims.t + padSize) / chunkSize;
  uint32_t alignK = CeilAlign(dims.dk, FP16_NUM_PER_BLOCK);
  uint32_t avFp32 = CeilAlign(dims.dv, FP32_NUM_PER_BLOCK);

  uint32_t vStepVal = 0;
  uint32_t tbufTotal = 0;
  uint32_t outQueueMax = 0;
  uint32_t restBytes = 0;
  SolveVStep(dims.dv, ubSize, chunkSize, dims.dk, alignK, avFp32, vStepVal, tbufTotal, outQueueMax, restBytes);

  auto td = context->GetTilingData<ChunkGatedDeltaRuleTilingData>();
  if (td == nullptr) {
    return GRAPH_FAILED;
  }
  FillTilingData(td, dims, aivNum, ubSize, chunkSize, numChunks, padSize, scaleValueAttr, vStepVal, restBytes);

  // One block per value head (b=1); cap at the available AIV cores.
  uint32_t blockDim = (dims.hv < aivNum) ? dims.hv : aivNum;
  if (blockDim == 0) {
    blockDim = kMinBlockDim;
  }
  context->SetBlockDim(blockDim);
  context->SetTilingKey(0);

  LogTiling(dims, chunkSize, numChunks, scaleValueAttr, ubSize, vStepVal, blockDim, outQueueMax, tbufTotal, restBytes);

  size_t *ws = context->GetWorkspaceSizes(kWorkspaceCount);
  if (ws != nullptr) {
    ws[0] = kWorkspaceBytes;
  }
  return GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(ChunkGatedDeltaRule).Tiling(ChunkGatedDeltaRuleTilingFunc, sizeof(ChunkGatedDeltaRuleTilingData));
