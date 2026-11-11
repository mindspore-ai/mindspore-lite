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
#include "fused_attn_out_transpose_quant_tiling.h"  // NOLINT(build/include_subdir)

using namespace ge;    // NOLINT(build/namespaces)
using namespace gert;  // NOLINT(build/namespaces)

namespace {
// Input indices of the FusedAttnOutTransposeQuant operator.
constexpr uint32_t kAttnInput = 0;
constexpr uint32_t kSmoothInput = 1;
// Shape layout: attn [..., M, D], leading dims flattened into H by tiling.
// D must be 128: the kernel's plain-copy fold gathers 128-element (256B) head
// runs so every copy and row base stays 32B aligned.
constexpr uint32_t kHeadDim = 128;
// The fold consumes the trailing (M, D) dims; attn is at least rank 2 and
// every dim before the trailing pair flattens into H.
constexpr size_t kAttnTrailingDims = 2;
// Rows are striped over cores at 8-row-group granularity so every core's
// token_scale DataCopy is a whole 32B block (fp32 x 8) at a 32B-aligned GM
// offset; the 32B-padded allocation absorbs the ragged last group.
constexpr uint32_t kScaleGroupRows = 8;
// Row-block execution: the kernel batches rowBlock rows into one block with
// a fixed ~4 event round-trips; bigger blocks amortize the sync (the op is
// sync-bound at small R). Candidates are multiples of 8 (whole groups, the
// kernel flushes 1-2 completed groups as one merged 64B copy) or divisors
// of 8 (sub-group blocks never straddle a group boundary).
constexpr uint32_t kRowBlockCandidates[] = {16, 8, 4, 2, 1};
constexpr uint32_t kMaxRowBlock = 16;
// Native absmax reduce (kernel v6) bounds: the second vcgmax stage needs
// 16-multiple maxima counts (n <= 2048 or n % 256 == 0) and the final vcmax
// bit mask spans at most 128 lanes (n <= 32768). n = h * kHeadDim means
// h <= 16 or h even; the real attention head counts (1/2/8/32) all pass.
constexpr uint32_t kReduceOneStageCols = 2048;
constexpr uint32_t kReduceAlignCols = 256;
constexpr uint32_t kReduceMaxCols = 32768;
// UB budget. Per row: fold fp16 + int8 y + one fp16 scratch that doubles as
// the absmax reduce staging and the quant staging (5B/col/row), plus (when
// smooth) act fp16 and the rows-tiled fp16 smooth table (2+2B/col/row).
// Fixed: the 16-slot fp32 scale stage and the fp16 vcmax dst (2/row,
// [value,index] pair).
constexpr uint32_t kUbBudgetBytes = 200 * 1024;  // ceiling; real budget = min(platform UB, this)
constexpr uint32_t kMinBlockDim = 1;
// Element byte sizes for the UB footprint mirror below (host-side stand-ins
// for the kernel's sizeof(half)/sizeof(int8_t)/sizeof(float)).
constexpr uint32_t kBytesPerHalf = 2;
constexpr uint32_t kBytesPerInt8 = 1;
constexpr uint32_t kBytesPerFloat = 4;
// The fp16 vcmax dst packs a [value, index] pair per row.
constexpr uint32_t kVcmaxPairSlots = 2;

uint32_t CeilDiv(uint32_t val, uint32_t d) { return (val + d - 1) / d; }

bool GetPlatformInfo(TilingContext *context, uint32_t &aivNum, uint32_t &ubBudget) {
  auto platformInfo = context->GetPlatformInfo();
  if (platformInfo == nullptr) {
    return false;
  }
  platform_ascendc::PlatformAscendC platform(platformInfo);
  aivNum = platform.GetCoreNumAiv();
  if (aivNum == 0) {
    aivNum = platform.GetCoreNum();
  }
  uint64_t ubSize = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  // Never budget more than the platform reports (310P UB can be 192KB); keep
  // the tuned 200KB ceiling when the platform is larger or the query fails.
  ubBudget = (ubSize == 0 || ubSize > kUbBudgetBytes) ? kUbBudgetBytes : static_cast<uint32_t>(ubSize);
  return aivNum > 0;
}

bool HasSmoothScale(TilingContext *context) {
  // Same optional-input detection as op B (ChunkGatedDeltaRule gamma pattern).
  return context->GetOptionalInputDesc(kSmoothInput) != nullptr &&
         context->GetOptionalInputTensor(kSmoothInput) != nullptr &&
         context->GetOptionalInputShape(kSmoothInput) != nullptr;
}

// attn [..., M, D]: flatten the leading dims into h; m = tokens and
// d = head dim (must be kHeadDim, see the constants above).
bool ParseAttnShape(TilingContext *context, uint32_t &h, uint32_t &m, uint32_t &d) {
  auto attnShape = context->GetInputShape(kAttnInput);
  if (attnShape == nullptr) {
    return false;
  }
  auto storage = attnShape->GetStorageShape();
  size_t dimNum = storage.GetDimNum();
  if (dimNum < kAttnTrailingDims) {
    return false;
  }
  h = 1;
  for (size_t i = 0; i + kAttnTrailingDims < dimNum; ++i) {
    h *= static_cast<uint32_t>(storage.GetDim(i));
  }
  m = static_cast<uint32_t>(storage.GetDim(dimNum - kAttnTrailingDims));
  d = static_cast<uint32_t>(storage.GetDim(dimNum - 1));
  return h != 0 && m != 0 && d == kHeadDim;
}

// The optional smooth_scale must be a plain [n] vector.
bool SmoothShapeValid(TilingContext *context, uint32_t n) {
  auto smoothShape = context->GetOptionalInputShape(kSmoothInput);
  return smoothShape != nullptr && smoothShape->GetStorageShape().GetDimNum() == 1 &&
         smoothShape->GetStorageShape().GetDim(0) == static_cast<int64_t>(n);
}

// Exact mirror of the kernel's InitBuffer footprint for one (R) candidate.
uint32_t BlockBytes(uint32_t n, bool hasSmooth, uint32_t rowBlock) {
  return rowBlock * n * (kBytesPerHalf + kBytesPerInt8 + kBytesPerHalf)  // fold + int8 y + reduce/quant fp16
         + (hasSmooth ? rowBlock * n * kBytesPerHalf : 0)                // act fp16
         + (hasSmooth ? rowBlock * n * kBytesPerHalf : 0)                // smooth fp16 table, rows-tiled
         + kMaxRowBlock * kBytesPerFloat                                 // fp32 scale stage
         + kMaxRowBlock * kVcmaxPairSlots * kBytesPerHalf;               // fp16 vcmax dst ([value,index] pair per row)
}

uint32_t FusedAttnOutTransposeQuantTilingFunc(TilingContext *context) {
  uint32_t h = 0;
  uint32_t m = 0;
  uint32_t d = 0;
  if (!ParseAttnShape(context, h, m, d)) {
    return GRAPH_FAILED;
  }
  uint32_t aivNum = 0;
  uint32_t ubBudget = kUbBudgetBytes;
  if (!GetPlatformInfo(context, aivNum, ubBudget)) {
    return GRAPH_FAILED;
  }
  uint32_t n = h * d;

  bool hasSmooth = HasSmoothScale(context);
  if (hasSmooth && !SmoothShapeValid(context, n)) {
    return GRAPH_FAILED;
  }

  // Native absmax reduce bounds (kernel v6, see the constants above).
  if (n > kReduceMaxCols || (n > kReduceOneStageCols && n % kReduceAlignCols != 0)) {
    return GRAPH_FAILED;
  }

  auto td = context->GetTilingData<FusedAttnOutTransposeQuantTilingData>();
  if (td == nullptr) {
    return GRAPH_FAILED;
  }

  // Largest rowBlock that fits the UB budget (block sync count ~ 1/R).
  uint32_t bestRowBlock = 0;
  for (uint32_t r = 0; r < sizeof(kRowBlockCandidates) / sizeof(kRowBlockCandidates[0]); ++r) {
    uint32_t rowBlock = kRowBlockCandidates[r];
    if (BlockBytes(n, hasSmooth, rowBlock) <= ubBudget) {
      bestRowBlock = rowBlock;
      break;
    }
  }
  if (bestRowBlock == 0) {
    // Even rowBlock=1 overflows UB.
    return GRAPH_FAILED;
  }

  // Row groups (8 rows) striped over cores; never launch more cores than
  // there are groups. The kernel derives its row range as
  // [gStart*8, gEnd*8) with g* = blockIdx*G/coreNum.
  uint32_t groups = CeilDiv(m, kScaleGroupRows);
  uint32_t blockDim = (groups < aivNum) ? groups : aivNum;
  if (blockDim == 0) {
    blockDim = kMinBlockDim;
  }
  td->coreNum = blockDim;
  td->m = m;
  td->h = h;
  td->d = d;
  td->n = n;
  td->rowBlock = bestRowBlock;
  td->hasSmooth = hasSmooth ? 1 : 0;
  // Fold load form: 2D DataCopyParams gather (ONE descriptor per row, H
  // blocks of 256B) when the src gap fits the uint16 32B-unit stride field
  // ((m-1)*d*2/32 <= 65535 -> m <= 8192 at d=128); above that the kernel
  // falls back to per-head plain copies. msprof @1536x8 put the scalar pipe
  // at 0.86 (descriptor-issue-bound) with the fold load as R*H of the
  // issues -- the 2D form cuts them by the head count (8x at H=8).
  constexpr uint32_t kDataCopyStrideMax = 65535;
  uint32_t foldLen32 = d * sizeof(uint16_t) / 32;  // fp16 block length, 32B units
  uint32_t foldSrcGap32 = (m - 1) * foldLen32;
  td->fold2d = (foldSrcGap32 <= kDataCopyStrideMax) ? 1 : 0;

  context->SetBlockDim(blockDim);
  context->SetTilingKey(0);

  size_t *ws = context->GetWorkspaceSizes(1);
  if (ws != nullptr) {
    // No device workspace: attn streams straight from the FA output.
    ws[0] = 0;
  }
  return GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(FusedAttnOutTransposeQuant)
  .Tiling(FusedAttnOutTransposeQuantTilingFunc, sizeof(FusedAttnOutTransposeQuantTilingData));
