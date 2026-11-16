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
#include "fused_qk_norm_rope_bsh_tiling.h"  // NOLINT(build/include_subdir)

using namespace ge;    // NOLINT(build/namespaces)
using namespace gert;  // NOLINT(build/namespaces)

namespace {
// Input indices of FusedQKNormRopeBSH (see fused_qk_norm_rope_bsh.cpp).
constexpr uint32_t kQInput = 0;
constexpr uint32_t kKInput = 1;
constexpr uint32_t kGammaQInput = 2;
constexpr uint32_t kGammaKInput = 3;
constexpr uint32_t kCosInput = 4;
constexpr uint32_t kSinInput = 5;

// Head dim must be a power of two in [32, 128]: >=32 keeps every per-row copy
// and Brcb source 32B aligned (dav_m200 has no DataCopyPad); <=128 because the
// q|k concat tile needs G >= 2 group tiles (>= 2*span fp32 squares) resident
// in UB alongside the fp16 tiles, and G=2, d=256 already exceeds the 200KB
// budget at R=1 (207KB). Deploy shape d=128.
constexpr uint32_t kMinD = 32;
constexpr uint32_t kMaxD = 128;
// UB budget for all per-core buffers (same conservative cap as every other
// op in this repo; the only empirically proven envelope on this chip).
constexpr uint32_t kUbBudgetBytes = 200 * 1024;
// Flat vector ops (Cast/Mul/Add plain count form) run over the whole concat
// tile [rows*G*span] in one instruction each. Counter-mode masks are int32,
// but the largest count verified on this chip is 8192 (fused_qk_norm_rope
// deploy); keep 16384 as the absolute ceiling -- shapes that would exceed it
// fall back to a smaller row block.
constexpr uint32_t kMaxFlatCount = 16384;
// Row-block candidates: larger R amortizes the per-block rotary tables,
// interleave calls and chain phases over more rows.
constexpr uint32_t kRowBlockCandidates[] = {8, 4, 2, 1};
constexpr uint32_t kGroupHeads = 16;
constexpr uint32_t kMinBlockDim = 1;
constexpr float kDefaultEpsilon = 1e-6;
// UB byte-cost terms for the fit check, mirroring the kernel's InitBuffer
// sizes (coefficients are bytes per unit of the dimension in the name).
constexpr uint32_t kTilesBytesPerGroupPerDim = 224;       // 3 fp16 + 2 fp32 concat tiles per (group, d)
constexpr uint32_t kRotTablesBytesPerDim = 64;            // per-row fp32 expanded cos/sin tables
constexpr uint32_t kCosSinBytesPerDim = 2;                // per-row fp16 cos/sin value bytes
constexpr uint32_t kRopeTmpBytesPerDim = 96;              // three fp32 rope temps per row
constexpr uint32_t kGammaStageBytesPerDim = 4;            // fp16 q/k gamma staging bytes
constexpr uint32_t kGammaExpandedBytesPerDim = 64;        // fp16 expanded q/k gamma tables
constexpr uint32_t kGammaConcatBytesPerGroupPerDim = 64;  // concat fp32 gamma table per (group, d)
constexpr uint32_t kUbSlackBytes = 512;                   // TBuf 32B-alignment slack
constexpr uint32_t kRopeHalves = 2;                       // NeoX rotate-half split (cos/sin hold half a head)

uint32_t CeilDiv(uint32_t val, uint32_t div) { return (val + div - 1) / div; }

bool IsPow2(uint32_t v) { return v != 0 && (v & (v - 1)) == 0; }

// Fully qualified: with both `using namespace ge` and `using namespace gert`
// open, an unqualified `Shape` is ambiguous (ge::Shape vs gert::Shape).
int64_t Numel(const gert::Shape *shape) {
  if (shape == nullptr) {
    return -1;
  }
  int64_t total = 1;
  for (size_t i = 0; i < shape->GetDimNum(); ++i) {
    int64_t dim = shape->GetDim(i);
    if (dim < 0) {
      return -1;
    }
    total *= dim;
  }
  return total;
}

bool GetPlatformInfo(const TilingContext *context, uint32_t &aivNum) {
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

void ParseEpsilon(const TilingContext *context, float &epsilon) {
  epsilon = kDefaultEpsilon;
  auto attrs = context->GetAttrs();
  if (attrs == nullptr || attrs->GetAttrNum() == 0) {
    return;
  }
  const float *epsPtr = attrs->GetAttrPointer<float>(0);
  if (epsPtr != nullptr) {
    epsilon = *epsPtr;
  }
}

// Exact mirror of the kernel's InitBuffer footprint, in bytes.
// span = 16*d (one head group tile); G = ceil(Hq/16) + ceil(Hkv/16) groups
// per row, concatenated slot-major at slot*span.
// Per row: three fp16 tiles nat/ilv/out [G*span] (6*G*span bytes) + fp32
// expanded rotary tables cosE/sinE (64*d) + fp16 cos/sin rows (2*d).
// Fixed per core: three rope fp32 temps (96*d), fp16 gamma staging for q/k
// (4*d), fp16 expanded gamma tables for q and k (64*d) + concat fp32 gamma
// table [G*span] (64*G*d), and TBuf 32B alignment slack.
uint32_t PerRowBytes(uint32_t d, uint32_t g) {
  return kTilesBytesPerGroupPerDim * g * d + kRotTablesBytesPerDim * d + kCosSinBytesPerDim * d;
}
uint32_t FixedBytes(uint32_t d, uint32_t g) {
  return kRopeTmpBytesPerDim * d + kGammaStageBytesPerDim * d + kGammaExpandedBytesPerDim * d +
         kGammaConcatBytesPerGroupPerDim * g * d + kUbSlackBytes;
}

// Input shape handles of FusedQKNormRopeBSH (see fused_qk_norm_rope_bsh.cpp).
struct OpInputs {
  const gert::StorageShape *q = nullptr;
  const gert::StorageShape *k = nullptr;
  const gert::StorageShape *gammaQ = nullptr;
  const gert::StorageShape *gammaK = nullptr;
  const gert::StorageShape *cos = nullptr;
  const gert::StorageShape *sin = nullptr;
};

// Shape values derived from the inputs and validated against each other.
struct ShapeInfo {
  uint32_t d = 0;    // head dim (from gamma_q)
  uint32_t m = 0;    // flattened leading dims of q
  uint32_t hq = 0;   // q head count
  uint32_t hkv = 0;  // kv head count
};

bool CollectInputs(const TilingContext *context, OpInputs &in) {
  in.q = context->GetInputShape(kQInput);
  in.k = context->GetInputShape(kKInput);
  in.gammaQ = context->GetInputShape(kGammaQInput);
  in.gammaK = context->GetInputShape(kGammaKInput);
  in.cos = context->GetInputShape(kCosInput);
  in.sin = context->GetInputShape(kSinInput);
  return in.q != nullptr && in.k != nullptr && in.gammaQ != nullptr && in.gammaK != nullptr && in.cos != nullptr &&
         in.sin != nullptr;
}

// D must come from gamma_q and agree with gamma_k.
bool ParseHeadDim(const OpInputs &in, ShapeInfo &si) {
  const auto &gammaQStorage = in.gammaQ->GetStorageShape();
  if (gammaQStorage.GetDimNum() != 1) {
    return false;
  }
  si.d = static_cast<uint32_t>(gammaQStorage.GetDim(0));
  if (si.d < kMinD || si.d > kMaxD || !IsPow2(si.d)) {
    return false;
  }
  const auto &gammaKStorage = in.gammaK->GetStorageShape();
  if (gammaKStorage.GetDimNum() != 1 || static_cast<uint32_t>(gammaKStorage.GetDim(0)) != si.d) {
    return false;
  }
  return true;
}

// q [M..., Hq*D]: flatten leading dims into m; D from gamma_q. Every dim must
// be a concrete positive value and both the last dim and the flattened
// leading product must fit their uint32 tiling fields -- a negative (dynamic)
// dim or an overflowing product is refused here, never narrowed silently.
bool ParseQShape(const OpInputs &in, ShapeInfo &si) {
  const auto &qStorage = in.q->GetStorageShape();
  const size_t qDimNum = qStorage.GetDimNum();
  if (qDimNum == 0) {
    return false;
  }
  const int64_t qLast = qStorage.GetDim(qDimNum - 1);
  if (qLast <= 0 || qLast % si.d != 0 || static_cast<uint64_t>(qLast) > UINT32_MAX) {
    return false;
  }
  si.hq = static_cast<uint32_t>(qLast) / si.d;
  uint64_t m = 1;
  for (size_t i = 0; i + 1 < qDimNum; ++i) {
    const int64_t dim = qStorage.GetDim(i);
    if (dim <= 0) {
      return false;
    }
    m *= static_cast<uint64_t>(dim);
    if (m > UINT32_MAX) {
      return false;
    }
  }
  si.m = static_cast<uint32_t>(m);
  return si.m != 0 && si.hq != 0;
}

// k [M..., Hkv*D]; cos/sin [M..., D/2]. Cross-check flattened sizes so a
// mismatched M fails here instead of reading out of bounds in the kernel.
bool ParseKvShape(const OpInputs &in, ShapeInfo &si) {
  const auto &kStorage = in.k->GetStorageShape();
  if (kStorage.GetDimNum() == 0) {
    return false;
  }
  const int64_t kLast = kStorage.GetDim(kStorage.GetDimNum() - 1);
  if (kLast <= 0 || kLast % si.d != 0 || static_cast<uint64_t>(kLast) > UINT32_MAX) {
    return false;
  }
  si.hkv = static_cast<uint32_t>(kLast) / si.d;
  const int64_t m = static_cast<int64_t>(si.m);
  const int64_t ropeNumel = m * (si.d / kRopeHalves);
  if (si.hkv == 0) {
    return false;
  }
  return Numel(&kStorage) == m * si.hkv * si.d && Numel(&(in.cos->GetStorageShape())) == ropeNumel &&
         Numel(&(in.sin->GetStorageShape())) == ropeNumel;
}

// Largest row block that fits the UB budget and keeps the flat concat-tile
// vector ops inside the verified counter-mode envelope.
uint32_t PickRowBlock(uint32_t d, uint32_t groups) {
  const uint32_t span = kGroupHeads * d;
  const uint32_t perRow = PerRowBytes(d, groups);
  const uint32_t fixed = FixedBytes(d, groups);
  for (size_t i = 0; i < sizeof(kRowBlockCandidates) / sizeof(kRowBlockCandidates[0]); ++i) {
    const uint32_t cand = kRowBlockCandidates[i];
    if (cand * perRow + fixed <= kUbBudgetBytes && cand * groups * span <= kMaxFlatCount) {
      return cand;
    }
  }
  return 0;
}

uint32_t FusedQKNormRopeBSHTilingFunc(TilingContext *context) {
  OpInputs in;
  if (!CollectInputs(context, in)) {
    return GRAPH_FAILED;
  }
  uint32_t aivNum = 0;
  if (!GetPlatformInfo(context, aivNum)) {
    return GRAPH_FAILED;
  }
  ShapeInfo si;
  if (!ParseHeadDim(in, si) || !ParseQShape(in, si) || !ParseKvShape(in, si)) {
    return GRAPH_FAILED;
  }

  float epsilon = kDefaultEpsilon;
  ParseEpsilon(context, epsilon);

  auto td = context->GetTilingData<FusedQKNormRopeBSHTilingData>();
  if (td == nullptr) {
    return GRAPH_FAILED;
  }

  const uint32_t groups = CeilDiv(si.hq, kGroupHeads) + CeilDiv(si.hkv, kGroupHeads);
  const uint32_t rowBlock = PickRowBlock(si.d, groups);
  if (rowBlock == 0) {
    return GRAPH_FAILED;
  }

  // Row blocks striped over cores: block range [blockIdx*B/coreNum, ...).
  const uint32_t blocks = CeilDiv(si.m, rowBlock);
  uint32_t blockDim = (blocks < aivNum) ? blocks : aivNum;
  if (blockDim == 0) {
    blockDim = kMinBlockDim;
  }
  td->coreNum = blockDim;
  td->m = si.m;
  td->hq = si.hq;
  td->hkv = si.hkv;
  td->d = si.d;
  td->rowBlock = rowBlock;
  td->epsilon = epsilon;
  td->invD = 1.0f / static_cast<float>(si.d);
  td->reserved0 = 0;
  td->reserved1 = 0;

  context->SetBlockDim(blockDim);
  context->SetTilingKey(0);

  size_t *ws = context->GetWorkspaceSizes(1);
  if (ws != nullptr) {
    // No device workspace: all staging lives in UB.
    ws[0] = 0;
  }
  return GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(FusedQKNormRopeBSH).Tiling(FusedQKNormRopeBSHTilingFunc, sizeof(FusedQKNormRopeBSHTilingData));
