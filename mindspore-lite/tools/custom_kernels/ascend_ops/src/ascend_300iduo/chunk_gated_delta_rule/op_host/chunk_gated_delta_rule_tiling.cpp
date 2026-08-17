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
// Chunk size is an internal implementation detail, matching the 910B operator.
constexpr uint32_t kDefaultChunkSize = 64;
// Axis index of the head-dim within the [T, H, D] query/value layouts.
constexpr int32_t kHeadDimAxis = 2;
// chunkKFp32 & kCumdecayFp32 (and chunkVFp32 & chunkAttnOutFp32) each share the
// same layout, so their byte cost is counted twice when sizing tmpBuff.
constexpr uint32_t kPairedTileCount = 2;
// Fixed workspace size requested for the op (32 MiB).
constexpr uint64_t kWorkspaceBytes = 32ULL * 1024 * 1024;
constexpr uint32_t kMatmulM = 64;
constexpr uint32_t kMatmulK = 128;
constexpr uint32_t kMatmulN = 128;
constexpr uint32_t kSmallCubeDk = 64;
constexpr uint32_t kMediumCubeDk = 96;
constexpr uint32_t kLargeCubeDk = 128;
constexpr uint32_t kCubeStageSlotCount = 2;
constexpr uint64_t kRawMatmulStageBytesPerCore =
  kCubeStageSlotCount * static_cast<uint64_t>(2 * kMatmulM * kMatmulK + kMatmulK * kMatmulN) * sizeof(uint16_t);
// Input indices for the CGDR operator.
constexpr uint32_t kQueryInput = 0;
constexpr uint32_t kValueInput = 2;
constexpr uint32_t kInitialStateInput = 4;
constexpr uint32_t kGammaInput = 6;
// Shape dimension indices within the [T, H, D] layout.
constexpr uint32_t kDimT = 0;
constexpr uint32_t kDimH = 1;
// Minimum block dimension for AIV core dispatch.
constexpr uint32_t kMinBlockDim = 1;
// Number of workspace slots requested from the runtime.
constexpr uint32_t kWorkspaceCount = 1;

uint32_t CeilAlign(uint32_t val, uint32_t align) { return (val + align - 1) / align * align; }
uint32_t DivCeil(uint32_t val, uint32_t divisor) { return (val + divisor - 1) / divisor; }

// Collected query/value shape dimensions used across the tiling helpers.
struct ShapeDims {
  uint32_t t;      // total sequence length
  uint32_t hqk;    // number of q/k heads
  uint32_t dk;     // key head dim
  uint32_t hv;     // number of value heads
  uint32_t dv;     // value head dim
  uint32_t batch;  // batch size from initial_state
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
  auto initialStateShape = context->GetInputShape(kInitialStateInput);
  if (queryShape == nullptr || valueShape == nullptr || initialStateShape == nullptr) {
    return false;
  }
  const auto &qDims = queryShape->GetStorageShape();
  const auto &vDims = valueShape->GetStorageShape();
  const auto &stateDims = initialStateShape->GetStorageShape();
  dims.t = qDims.GetDim(kDimT);
  dims.hqk = qDims.GetDim(kDimH);
  dims.dk = qDims.GetDim(kHeadDimAxis);
  dims.hv = vDims.GetDim(kDimH);
  dims.dv = vDims.GetDim(kHeadDimAxis);
  dims.batch = stateDims.GetDim(kDimT);
  return true;
}

// Read the optional scale_value attr (idx 0), falling back to the public default.
void ParseAttrs(TilingContext *context, float &scaleValueAttr) {
  scaleValueAttr = 1.0f;
  auto attrs = context->GetAttrs();
  if (attrs == nullptr || attrs->GetAttrNum() == 0) {
    return;
  }
  const float *scalePtr = attrs->GetAttrPointer<float>(0);
  if (scalePtr != nullptr) {
    scaleValueAttr = *scalePtr;
  }
}

bool HasGamma(TilingContext *context) {
  return context->GetOptionalInputDesc(kGammaInput) != nullptr &&
         context->GetOptionalInputTensor(kGammaInput) != nullptr &&
         context->GetOptionalInputShape(kGammaInput) != nullptr;
}

// Total tmpBuff bytes for a candidate vStep. V temporaries are compact tiles.
// With head*V-tile core splitting, each core handles only one tile, so scores
// are dead before that tile loads state and the two buffers may overlap.
uint32_t ComputeTmpBuffBytes(uint32_t vs, uint32_t dv, uint32_t chunkSize, uint32_t alignK,
                             bool allowScoresStateOverlap) {
  uint32_t avStepAligned = CeilAlign(vs, FP32_NUM_PER_BLOCK);
  uint32_t cs = chunkSize;
  uint32_t kTileBytes = cs * alignK * sizeof(float);         // chunkKFp32 + kCumdecayFp32
  uint32_t vTileBytes = cs * avStepAligned * sizeof(float);  // chunkVFp32 + chunkAttnOutFp32
  uint32_t tDecay = cs * cs * sizeof(float);                 // decayMaskFp32
  uint32_t tGcum = cs * sizeof(float);                       // gCumsumFp32
  uint32_t stateStrideK = alignK;
  uint32_t dotProductElem = (stateStrideK > cs) ? stateStrideK : cs;
  uint32_t deltaElem = (dotProductElem > avStepAligned) ? dotProductElem : avStepAligned;
  uint32_t tDelta = deltaElem * sizeof(float);                     // deltaFp32
  uint32_t tDotProduct = dotProductElem * sizeof(float);           // dotProductFp32
  uint32_t tExpGcum = cs * sizeof(float);                          // expGCumFp32
  uint32_t tBeta = tExpGcum;                                       // betaFp32
  uint32_t tScores = cs * cs * sizeof(float);                      // chunkScoresFp32 (attn matrix)
  uint32_t tState = stateStrideK * avStepAligned * sizeof(float);  // stateInFp32 [DK, vStep]
  bool overlapScoresState = vs >= dv || allowScoresStateOverlap;
  uint32_t tScoresState = overlapScoresState ? ((tScores > tState) ? tScores : tState) : (tScores + tState);
  return kPairedTileCount * kTileBytes + tDecay + kPairedTileCount * vTileBytes + tGcum + tDelta + tDotProduct +
         tExpGcum + tBeta + tScoresState;
}

// stateOutQueue bytes: max of the state tile and compact chunk output.
uint32_t ComputeOutQueueBytes(uint32_t vs, uint32_t stateStrideK, uint32_t chunkSize) {
  uint32_t avStepAligned = CeilAlign(vs, FP32_NUM_PER_BLOCK);
  uint32_t stateBytes = stateStrideK * avStepAligned * sizeof(uint16_t);
  uint32_t chunkBytes = chunkSize * avStepAligned * sizeof(uint16_t);
  return (stateBytes > chunkBytes) ? stateBytes : chunkBytes;
}

// Find the largest vStep (FP32-block aligned, <= dv) whose tmpBuff + outQueue fits in UB.
// Also returns the resulting buffer sizes for tiling/debug.
bool SolveVStep(uint32_t dv, int64_t ubSize, uint32_t chunkSize, uint32_t alignK, bool allowScoresStateOverlap,
                uint32_t preferredVStep, uint32_t &vStepVal, uint32_t &tbufTotal, uint32_t &outQueueMax,
                uint32_t &restBytes) {
  if (ubSize <= 0) {
    return false;
  }
  uint32_t maxVStep = CeilAlign(dv, FP16_NUM_PER_BLOCK);
  bool found = false;
  // Prefer the regular Cube width even when Dv needs several tiles. This keeps
  // every full/tail tile on the same 64/96/128 fused implementation.
  if (preferredVStep != 0) {
    uint64_t preferredBytes =
      static_cast<uint64_t>(ComputeTmpBuffBytes(preferredVStep, dv, chunkSize, alignK, allowScoresStateOverlap)) +
      ComputeOutQueueBytes(preferredVStep, alignK, chunkSize);
    if (preferredBytes <= static_cast<uint64_t>(ubSize)) {
      vStepVal = preferredVStep;
      found = true;
    }
  }
  for (uint32_t vs = maxVStep; !found && vs >= FP16_NUM_PER_BLOCK; vs -= FP16_NUM_PER_BLOCK) {
    uint64_t requiredBytes =
      static_cast<uint64_t>(ComputeTmpBuffBytes(vs, dv, chunkSize, alignK, allowScoresStateOverlap)) +
      ComputeOutQueueBytes(vs, alignK, chunkSize);
    if (requiredBytes <= static_cast<uint64_t>(ubSize)) {
      vStepVal = vs;
      found = true;
      break;
    }
  }
  if (!found) {
    return false;
  }
  tbufTotal = ComputeTmpBuffBytes(vStepVal, dv, chunkSize, alignK, allowScoresStateOverlap);
  outQueueMax = ComputeOutQueueBytes(vStepVal, alignK, chunkSize);
  restBytes = (ubSize > static_cast<int64_t>(outQueueMax)) ? static_cast<uint32_t>(ubSize - outQueueMax) : 0;
  return true;
}

// Select the smallest compiled Cube tile that can hold Dk. The kernel pads
// the tail to this width, so this is a shape-range policy rather than a
// one-off specialization for a benchmark dimension.
uint32_t SelectCubeDk(uint32_t dk) {
  if (dk <= kSmallCubeDk) {
    return kSmallCubeDk;
  }
  if (dk <= kMediumCubeDk) {
    return kMediumCubeDk;
  }
  if (dk <= kLargeCubeDk) {
    return kLargeCubeDk;
  }
  return 0;
}

// Write the resolved scalar parameters into the device-visible tiling struct.
void FillTilingData(ChunkGatedDeltaRuleTilingData *td, const ShapeDims &dims, uint32_t aivNum, int64_t ubSize,
                    uint32_t chunkSize, uint32_t numChunks, uint32_t padSize, float scaleValue, uint32_t vStepVal,
                    uint32_t restBytes, bool hasGamma) {
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
  td->b = dims.batch;
  td->padSize = padSize;
  td->hasGamma = hasGamma ? 1 : 0;
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

  float scaleValueAttr = 1.0f;
  ParseAttrs(context, scaleValueAttr);
  uint32_t chunkSize = kDefaultChunkSize;
  bool hasGamma = HasGamma(context);

  // Pad T up to a multiple of chunkSize and pre-compute the FP16/FP32 block alignments.
  uint32_t padSize = (chunkSize - dims.t % chunkSize) % chunkSize;
  uint32_t numChunks = (dims.t + padSize) / chunkSize;
  uint32_t cubeDk = SelectCubeDk(dims.dk);
  uint32_t alignK = (cubeDk == 0) ? CeilAlign(dims.dk, FP16_NUM_PER_BLOCK) : cubeDk;

  uint32_t vStepVal = 0;
  uint32_t tbufTotal = 0;
  uint32_t outQueueMax = 0;
  uint32_t restBytes = 0;
  // A V tile is an independent work item. Once the attention matrix has been
  // staged for its Cube product, the state tile can safely reuse the same UB
  // region, including when a head spans multiple V tiles.
  bool allowScoresStateOverlap = true;
  // Keep V and padded K on the same regular Cube tile. This enables the
  // fused path for every Dk in the 64/96/128 ranges, including non-aligned
  // dimensions whose valid tail is zero-padded by the kernel.
  uint32_t preferredVStep = cubeDk;
  if (!SolveVStep(dims.dv, ubSize, chunkSize, alignK, allowScoresStateOverlap, preferredVStep, vStepVal, tbufTotal,
                  outQueueMax, restBytes)) {
    return GRAPH_FAILED;
  }

  auto td = context->GetTilingData<ChunkGatedDeltaRuleTilingData>();
  if (td == nullptr) {
    return GRAPH_FAILED;
  }
  FillTilingData(td, dims, aivNum, ubSize, chunkSize, numChunks, padSize, scaleValueAttr, vStepVal, restBytes,
                 hasGamma);

  // One work item is one (batch, value-head, V-tile) tuple. V tiles write
  // disjoint output/state ranges, and exposing batch here also fills all cores
  // for small-head multi-batch inputs.
  uint32_t vTileCount = DivCeil(dims.dv, vStepVal);
  uint64_t totalWorkItems = static_cast<uint64_t>(dims.batch) * dims.hv * vTileCount;
  uint32_t workItems = (totalWorkItems > UINT32_MAX) ? UINT32_MAX : static_cast<uint32_t>(totalWorkItems);
  uint32_t blockDim = (workItems < aivNum) ? workItems : aivNum;
  if (blockDim == 0) {
    blockDim = kMinBlockDim;
  }
  context->SetBlockDim(blockDim);
  // Keep exactly three compiled kernels for the 64/96/128 padded Cube tiles.
  // Shapes above 128 use the generic fallback in the 96-wide kernel class.
  uint32_t tilingKey = 1;
  if (cubeDk == kSmallCubeDk) {
    tilingKey = 0;
  } else if (cubeDk == kLargeCubeDk) {
    tilingKey = 2;
  }
  context->SetTilingKey(tilingKey);

  LogTiling(dims, chunkSize, numChunks, scaleValueAttr, ubSize, vStepVal, blockDim, outQueueMax, tbufTotal, restBytes);

  size_t *ws = context->GetWorkspaceSizes(kWorkspaceCount);
  if (ws != nullptr) {
    uint64_t workspaceStrideV = CeilAlign(dims.dv, FP32_NUM_PER_BLOCK);
    uint64_t stateWorkspaceBytes =
      static_cast<uint64_t>(dims.batch) * dims.hv * dims.dk * workspaceStrideV * sizeof(float);
    uint64_t requiredWorkspace = stateWorkspaceBytes + static_cast<uint64_t>(blockDim) * kRawMatmulStageBytesPerCore;
    ws[0] = (requiredWorkspace > kWorkspaceBytes) ? requiredWorkspace : kWorkspaceBytes;
  }
  return GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(ChunkGatedDeltaRule).Tiling(ChunkGatedDeltaRuleTilingFunc, sizeof(ChunkGatedDeltaRuleTilingData));
