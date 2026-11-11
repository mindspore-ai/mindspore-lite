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

#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "fused_swi_glu_act_quant_nz_tiling.h"  // NOLINT(build/include_subdir)

using namespace ge;    // NOLINT(build/namespaces)
using namespace gert;  // NOLINT(build/namespaces)

namespace {
// Input indices of the FusedSwiGLUActQuantNZ operator.
constexpr uint32_t kGateInput = 0;
constexpr uint32_t kUpInput = 1;
constexpr uint32_t kSmoothInput = 2;
// Rows are striped over cores at 8-row-group granularity so every core's
// token_scale DataCopy is a whole 32B block (fp32 x 8) at a 32B-aligned GM
// offset; the 32B-padded allocation absorbs the ragged last group.
constexpr uint32_t kScaleGroupRows = 8;
// Row blocks never straddle an 8-row scale group: R in {8, 4, 2, 1}.
constexpr uint32_t kMaxRowBlock = 8;
// Per-row absmax tree landing slots are 32B strided (fp16 x 16).
constexpr uint32_t kSlotStrideF16 = 16;
// dav_m200 has no DataCopyPad: N must keep every per-chunk copy 32B aligned
// for fp16 in and int8 out, hence a multiple of 32.
constexpr uint32_t kAlignCols = 32;
constexpr uint32_t kMinCols = 32;
constexpr uint32_t kMinColChunk = 32;
// DataCopyParams blockCount/strides are uint16 (strides in 32B blocks): the
// NZ per-row gather/scatter uses blockCount n/16 (fp16) and n/32 (int8), and
// the shared segment stride m1t*16-1 (fp16 and int8 fractals are both 512B).
constexpr uint32_t kMaxCopyParam = 65535;
// NZ fractal geometry (fp16): the fractal is 16 rows x 16 cols (C0 =
// 32B/sizeof(half) = 16); the int8 C0 is 32. kNz*Offset are the positions
// of the NZ storage suffix [n1, m1t, 16, 16] counted from the last dim.
constexpr uint32_t kNzFractalRows = 16;
constexpr uint32_t kNzHalfBlockCols = 16;
constexpr uint32_t kNzInt8BlockCols = 32;
constexpr size_t kNzN0Offset = 1;
constexpr size_t kNzM0Offset = 2;
constexpr size_t kNzM1tOffset = 3;
constexpr size_t kNzN1Offset = 4;
// The NZ branch only supports 2D origins [M, N] (see DetectLayout).
constexpr size_t kNzOriginRank = 2;
constexpr uint32_t kUbBudgetBytes = 200 * 1024;
constexpr uint32_t kMinBlockDim = 1;
// Element byte sizes for the TileBytes footprint mirror below (host-side
// stand-ins for the kernel's sizeof(half)/sizeof(float)/sizeof(int8_t)).
constexpr uint32_t kBytesPerHalf = 2;
constexpr uint32_t kBytesPerFloat = 4;
constexpr uint32_t kBytesPerInt8 = 1;
// The gate/up branch pair shares each chunk tile.
constexpr uint32_t kGateUpBranches = 2;
// Ping-pong gate/up tile pairs when colChunk < n.
constexpr uint32_t kDoubleBufferPairs = 2;
// The colChunk search halves the candidate each step (largest first).
constexpr uint32_t kColChunkHalving = 2;
// The preferred search requires R >= 2 before falling back to R=1.
constexpr uint32_t kMinPreferredRowBlock = 2;

uint32_t CeilDiv(uint32_t val, uint32_t d) { return (val + d - 1) / d; }
uint32_t AlignDown32(uint32_t v) { return v / kAlignCols * kAlignCols; }

// v4b row-block UB footprint (bytes) for a (rowBlock, colChunk) choice:
//   gate/up fp16 tiles    gateUpPairs * R*cc*2 each (MTE2 in, V reads; a
//                         second pair when colChunk < n -- the kernel's
//                         ping-pong double buffer)
//   act rows fp16         R*n*2                (V writes, row-resident)
//   y rows int8           R*n                  (V writes, MTE3 out)
//   maxRow slots          R*kSlotStrideF16*2   (per-row fold-to-1 landing)
//   scale staging         kScaleGroupRows*4
// colChunk < n (per-row chain, workF32 = cc, tmpH = max(n, cc)):
//   work fp32 scratch     cc*4                 (Cast-up + in-place Exp)
//   tmpH fp16             max(n, cc)*2         (chain staging, then absmax
//                         tree scratch, then pass-3 staging)
//   smooth table          n*2 (optional, resident fp16)
// colChunk == n (v4b fused chain, the block's R rows at once):
//   work fp32 scratch     R*cc*4               (fused Cast-up/Exp beat set)
//   tmpH fp16             R*cc*2               (fused chain staging; also
//                         covers the tree scratch n and pass-3 cc)
//   smooth tables         n*2 + R*n*2 (optional: resident fp16 row plus the
//                         R-replicated smooth2x_ the fused final Mul reads)
uint32_t TileBytes(uint32_t n, bool hasSmooth, uint32_t rowBlock, uint32_t colChunk) {
  uint32_t gateUpPairs = (colChunk < n) ? kDoubleBufferPairs : 1;
  uint32_t bytes = gateUpPairs * rowBlock * colChunk * kGateUpBranches * kBytesPerHalf;
  bytes += rowBlock * n * kBytesPerHalf;
  bytes += rowBlock * n * kBytesPerInt8;
  bytes += rowBlock * kSlotStrideF16 * kBytesPerHalf;
  bytes += kScaleGroupRows * kBytesPerFloat;
  if (colChunk == n) {
    bytes += rowBlock * colChunk * kBytesPerFloat;
    bytes += rowBlock * colChunk * kBytesPerHalf;
    if (hasSmooth) {
      bytes += n * kBytesPerHalf + rowBlock * n * kBytesPerHalf;
    }
  } else {
    bytes += colChunk * kBytesPerFloat;
    uint32_t staging = (n > colChunk) ? n : colChunk;
    bytes += staging * kBytesPerHalf;
    if (hasSmooth) {
      bytes += n * kBytesPerHalf;
    }
  }
  return bytes;
}

bool GetPlatformInfo(TilingContext *context, uint32_t &aivNum) {
  auto platformInfo = context->GetPlatformInfo();
  if (platformInfo == nullptr) {
    return false;
  }
  platform_ascendc::PlatformAscendC platform(platformInfo);
  aivNum = platform.GetCoreNumAiv();
  if (aivNum == 0) {
    aivNum = platform.GetCoreNum();
  }
  return aivNum > 0;
}

bool HasSmoothScale(TilingContext *context) {
  // Same optional-input detection as ChunkGatedDeltaRule's gamma input.
  return context->GetOptionalInputDesc(kSmoothInput) != nullptr &&
         context->GetOptionalInputTensor(kSmoothInput) != nullptr &&
         context->GetOptionalInputShape(kSmoothInput) != nullptr;
}

// [M..., N]: flatten leading dims into m, n = inter columns. The logical N
// comes from the ORIGIN shape; the STORAGE shape then disambiguates the
// layout (gate has no gamma-style [N] vector to cross-check, so the origin
// last dim plays that role). FRACTAL_NZ of a logical [M, N] fp16 tensor is
// [N/16, m1t, 16, 16] (column fractals outermost, 16x16 fractal row-major
// [m0, n0]; CANN include pto/common/pto_tile.hpp) -- n1*16 == N -- while
// ND storage equals the origin shape and ends in dim == N, which is >= 32
// and can never end in the 16, 16 NZ signature. m always comes from the
// ORIGIN shape: NZ pads rows up to m1t*16, and token_scale is a logical
// [M] ND output that must never be written past the logical end.
// Derives nU / isNz / m1t / m; returns false on any malformed shape.
bool DetectLayout(TilingContext *context, uint64_t &nU, uint32_t &isNz, uint32_t &m1t, uint32_t &m) {
  auto gateShape = context->GetInputShape(kGateInput);
  auto storage = gateShape->GetStorageShape();
  auto origin = gateShape->GetOriginShape();
  size_t dimNum = storage.GetDimNum();
  size_t originDimNum = origin.GetDimNum();
  if (dimNum == 0 || originDimNum == 0) {
    return false;
  }
  nU = static_cast<uint64_t>(origin.GetDim(originDimNum - 1));
  if (dimNum >= kNzN1Offset && storage.GetDim(dimNum - kNzN0Offset) == kNzHalfBlockCols &&
      storage.GetDim(dimNum - kNzM0Offset) == kNzFractalRows &&
      static_cast<uint64_t>(storage.GetDim(dimNum - kNzN1Offset)) * kNzHalfBlockCols == nU) {
    isNz = 1;
    m1t = static_cast<uint32_t>(storage.GetDim(dimNum - kNzM1tOffset));
    if (m1t == 0 || m1t * kNzFractalRows - 1 > kMaxCopyParam) {
      return false;
    }
    // NZ addressing (NzRowBase) maps a flattened logical row inside a single
    // prefix, which is only correct for 2D origins [M, N] (P = 1); with 3D+
    // origins the flattened row would cross per-prefix pad boundaries and
    // misaddress. Every site edge in the W8A8 graph (and the bench, through
    // GE-inserted TransData) is 2D, so guard rather than generalize. Pad
    // rows (m1t*16 > M) are never processed; storage must cover all M rows.
    if (originDimNum != kNzOriginRank) {
      return false;
    }
    uint64_t rows = static_cast<uint64_t>(origin.GetDim(0));
    if (rows == 0 || rows > 0xFFFFFFFFULL || rows > static_cast<uint64_t>(m1t) * kNzFractalRows) {
      return false;
    }
    m = static_cast<uint32_t>(rows);
  } else {
    // ND row-major: storage == origin, last dim is N.
    if (static_cast<uint64_t>(storage.GetDim(dimNum - 1)) != nU) {
      return false;
    }
    uint64_t rows = 1;
    for (size_t d = 0; d + 1 < dimNum; ++d) {
      rows *= static_cast<uint64_t>(storage.GetDim(d));
    }
    if (rows == 0 || rows > 0xFFFFFFFFULL) {
      return false;
    }
    m = static_cast<uint32_t>(rows);
  }
  return m != 0;
}

// Column contract: N must be a multiple of 32 (32B-aligned copies) and the
// NZ per-row gather/scatter block counts (n/16 fp16, n/32 int8) must fit
// the uint16 DataCopyParams fields.
bool ColumnsValid(uint32_t n) {
  return n >= kMinCols && n % kAlignCols == 0 && n / kNzHalfBlockCols <= kMaxCopyParam &&
         n / kNzInt8BlockCols <= kMaxCopyParam;
}

// Graph contract: up mirrors gate (full storage-shape equality, layout
// included).
bool UpMirrorsGate(TilingContext *context) {
  const auto &gateStorage = context->GetInputShape(kGateInput)->GetStorageShape();
  auto upShape = context->GetInputShape(kUpInput);
  if (upShape == nullptr) {
    return false;
  }
  auto upStorage = upShape->GetStorageShape();
  if (upStorage.GetDimNum() != gateStorage.GetDimNum()) {
    return false;
  }
  for (size_t d = 0; d < gateStorage.GetDimNum(); ++d) {
    if (upStorage.GetDim(d) != gateStorage.GetDim(d)) {
      return false;
    }
  }
  return true;
}

// The optional smooth is an [N] vector.
bool SmoothShapeValid(TilingContext *context, uint32_t n) {
  auto smoothShape = context->GetOptionalInputShape(kSmoothInput);
  return smoothShape != nullptr && smoothShape->GetStorageShape().GetDim(0) == static_cast<int64_t>(n);
}

// Largest row block r (kMaxRowBlock down to minR, powers of two) whose
// tile fits the UB budget, or 0 when none fits.
uint32_t LargestFittingRowBlock(uint32_t n, bool hasSmooth, uint32_t colChunk, uint32_t minR) {
  for (uint32_t r = kMaxRowBlock; r >= minR; r >>= 1) {
    if (TileBytes(n, hasSmooth, r, colChunk) <= kUbBudgetBytes) {
      return r;
    }
  }
  return 0;
}

// (rowBlock, colChunk) search per the comment at the call site: prefer the
// fewest chunks (largest colChunk) at R >= 2, taking the largest R that
// fits at each colChunk; fall back to R=1 only when no R >= 2 tile fits at
// any colChunk. Returns false when nothing fits at all.
bool SelectTile(uint32_t n, bool hasSmooth, uint32_t &rowBlock, uint32_t &colChunk) {
  rowBlock = 0;
  colChunk = 0;
  for (uint32_t minR = kMinPreferredRowBlock; minR >= 1 && rowBlock == 0; --minR) {
    for (uint32_t cc = n; cc >= kMinColChunk; cc = AlignDown32(cc / kColChunkHalving)) {
      uint32_t r = LargestFittingRowBlock(n, hasSmooth, cc, minR);
      if (r != 0) {
        rowBlock = r;
        colChunk = cc;
        break;
      }
    }
  }
  return rowBlock != 0;
}

uint32_t FusedSwiGLUActQuantNZTilingFunc(TilingContext *context) {
  auto gateShape = context->GetInputShape(kGateInput);
  if (gateShape == nullptr) {
    return GRAPH_FAILED;
  }
  uint32_t aivNum = 0;
  if (!GetPlatformInfo(context, aivNum)) {
    return GRAPH_FAILED;
  }

  uint64_t nU = 0;
  uint32_t isNz = 0;
  uint32_t m1t = 0;
  uint32_t m = 0;
  if (!DetectLayout(context, nU, isNz, m1t, m)) {
    return GRAPH_FAILED;
  }
  if (nU > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return GRAPH_FAILED;
  }
  uint32_t n = static_cast<uint32_t>(nU);
  if (!ColumnsValid(n)) {
    return GRAPH_FAILED;
  }
  if (!UpMirrorsGate(context)) {
    return GRAPH_FAILED;
  }

  bool hasSmooth = HasSmoothScale(context);
  if (hasSmooth && !SmoothShapeValid(context, n)) {
    return GRAPH_FAILED;
  }

  // A full row (fp16) must stay resident in UB: the per-token absmax needs
  // the whole row before any y element can be emitted. The selection below
  // enforces this implicitly (it fails when even (R=1, cc=32) overflows).
  auto td = context->GetTilingData<FusedSwiGLUActQuantNZTilingData>();
  if (td == nullptr) {
    return GRAPH_FAILED;
  }
  // Row groups (8 rows) are striped over cores; never launch more cores than
  // there are groups. The kernel derives its group range as
  // [blockIdx*G/coreNum, (blockIdx+1)*G/coreNum).
  uint32_t groups = CeilDiv(m, kScaleGroupRows);
  uint32_t blockDim = (groups < aivNum) ? groups : aivNum;
  if (blockDim == 0) {
    blockDim = kMinBlockDim;
  }
  td->coreNum = blockDim;
  td->m = m;
  td->n = n;
  td->hasSmooth = hasSmooth ? 1 : 0;
  // (rowBlock, colChunk) search: prefer the FEWEST chunks (largest colChunk)
  // at R >= 2 -- the activation chain issues 9 (smooth) / 8 (plain) vector
  // calls per (row, chunk) (per BLOCK on single-chunk shapes, where v4b
  // fuses the block's rows into one wide beat set), so chunk count sets the
  // chain issue cost, while R amortizes the block-level events (MTE2_V /
  // V_MTE2 pairs scale with chunks, the single V_S + batched MTE3 scale
  // with blocks). At each colChunk take the largest R that fits. Fall back
  // to R=1 only when no R >= 2 tile fits at any colChunk (very wide n).
  uint32_t rowBlock = 0;
  uint32_t colChunk = 0;
  if (!SelectTile(n, hasSmooth, rowBlock, colChunk)) {
    return GRAPH_FAILED;
  }
  td->colChunk = colChunk;
  td->rowBlock = rowBlock;
  td->isNz = isNz;
  td->m1t = m1t;

  context->SetBlockDim(blockDim);
  context->SetTilingKey(0);

  size_t *ws = context->GetWorkspaceSizes(1);
  if (ws != nullptr) {
    // No device workspace: rows stream straight from the QBMV3 outputs.
    ws[0] = 0;
  }
  return GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(FusedSwiGLUActQuantNZ)
  .Tiling(FusedSwiGLUActQuantNZTilingFunc, sizeof(FusedSwiGLUActQuantNZTilingData));
