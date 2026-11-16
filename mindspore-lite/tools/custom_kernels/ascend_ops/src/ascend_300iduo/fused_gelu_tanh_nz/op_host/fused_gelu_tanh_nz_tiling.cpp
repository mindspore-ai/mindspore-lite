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
#include "fused_gelu_tanh_nz_tiling.h"  // NOLINT(build/include_subdir)

using namespace ge;    // NOLINT(build/namespaces)
using namespace gert;  // NOLINT(build/namespaces)

namespace {
// Input index of the FusedGeluTanhNZ operator.
constexpr uint32_t kXInput = 0;
// The kernel walks the input storage buffer as one flat elementwise stream:
// for a pure elementwise map, FRACTAL_NZ storage ([L, nBlk, mBlk, 16, 16],
// dense row-major over 16x16 fractals) and ND storage ([d0, ..., dn], dense
// row-major) are both contiguous [totalElems] buffers, and x and y share the
// layout, so flat positions correspond one-to-one. totalElems is therefore
// the product of ALL storage dims, read from GetStorageShape() -- the exact
// size of the buffer the kernel walks, under either format (NZ padding, if
// the origin dims are not 16-aligned, is covered too: gelu on padding cells
// is never read downstream).
constexpr uint32_t kFractalElems = 256;  // 16x16 fp16 NZ fractal, 512B
// Chunk = 48 fractals: 2 x-buffers fp16 (2*2B) + 2 y-buffers fp16 (2*2B) +
// tmp fp16 (2B) + tmp fp32 (4B) + 1 hoisted const tensor fp16 (2B) =
// 16B per element = the full 192KB dav_m200 UB budget (v1/v2's fp32
// work/tmp pair already ran 16B/elem; v3b/v4c ran 14B = 172KB, v4h spends
// 2B of that on the constant tensor). Chunk size is unchanged since v2 so
// every bench A/B changes only the compute chain (the rejected v4j needed
// 20B/elem, forcing chunk 9728 -- it benched +2% WORSE: the extra const
// reads and chunk boundaries ate the scalar-op savings; see README v4).
// x and y ping-pong so
// MTE2/MTE3 overlap the vector chain (pipe partition, see the kernel
// source). Fixed cost per chunk (no full-row residency, unlike an absmax
// kernel), so no shape-driven halving loop is needed.
constexpr uint32_t kMaxChunkElems = 12288;
constexpr uint32_t kChunkBytesPerElem = 16;
// dav_m200 (Ascend310P) exposes 192KB of per-core UB, the true hard cap for
// the per-chunk InitBuffers. The compile-time assert below pins the chunk
// size against that platform cap, and the tiling function additionally
// re-checks the footprint against the UB size the runtime platform actually
// reports, so a platform with less usable UB fails loudly here instead of
// letting TPipe::InitBuffer hand the kernel invalid tensors.
constexpr uint32_t kUbBudgetBytes = 192 * 1024;
static_assert(kMaxChunkElems * kChunkBytesPerElem <= kUbBudgetBytes,
              "per-chunk UB footprint must fit the 192KB platform budget");
constexpr uint32_t kMinBlockDim = 1;

uint64_t CeilDiv(uint64_t val, uint64_t d) { return (val + d - 1) / d; }

bool GetPlatformInfo(const TilingContext *context, uint32_t &aivNum, uint64_t &ubBytes) {
  auto platformInfo = context->GetPlatformInfo();
  if (platformInfo == nullptr) {
    return false;
  }
  platform_ascendc::PlatformAscendC platform(platformInfo);
  aivNum = platform.GetCoreNumAiv();
  if (aivNum == 0) {
    aivNum = platform.GetCoreNum();
  }
  ubBytes = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
  return aivNum > 0 && ubBytes > 0;
}

uint32_t FusedGeluTanhNZTilingFunc(TilingContext *context) {
  auto xShape = context->GetInputShape(kXInput);
  if (xShape == nullptr) {
    return GRAPH_FAILED;
  }
  uint32_t aivNum = 0;
  uint64_t ubBytes = 0;
  if (!GetPlatformInfo(context, aivNum, ubBytes)) {
    return GRAPH_FAILED;
  }

  // Flatten every storage dim into totalElems (see the layout note above).
  // Storage dims are concrete positive int64s here; accumulate in uint64 and
  // reject a product that would overflow before it can wrap.
  const auto &storage = xShape->GetStorageShape();
  size_t dimNum = storage.GetDimNum();
  if (dimNum == 0) {
    return GRAPH_FAILED;
  }
  uint64_t total = 1;
  for (size_t d = 0; d < dimNum; ++d) {
    const int64_t dim = storage.GetDim(d);
    if (dim <= 0 || static_cast<uint64_t>(dim) > UINT64_MAX / total) {
      return GRAPH_FAILED;
    }
    total *= static_cast<uint64_t>(dim);
  }
  // 256-element granularity: NZ storage is inherently whole-fractal, and it
  // keeps every ND chunk boundary (and thus every DataCopy) 32B aligned.
  if (total % kFractalElems != 0) {
    return GRAPH_FAILED;
  }

  // Chunk sizing: clamp to the tensor so small shapes run as one chunk;
  // kMaxChunkElems is a multiple of 256 and totalElems was checked to be
  // one, so the clamp preserves the chunk alignment contract.
  const uint32_t chunkElems = (total < kMaxChunkElems) ? static_cast<uint32_t>(total) : kMaxChunkElems;
  // The chunk's UB footprint must fit what this platform really exposes.
  if (static_cast<uint64_t>(chunkElems) * kChunkBytesPerElem > ubBytes) {
    return GRAPH_FAILED;
  }
  const uint64_t totalChunks = CeilDiv(total, chunkElems);

  auto td = context->GetTilingData<FusedGeluTanhNZTilingData>();
  if (td == nullptr) {
    return GRAPH_FAILED;
  }
  // Chunks are striped over cores; never launch more cores than there are
  // chunks. The kernel derives its chunk range as
  // [blockIdx*totalChunks/coreNum, (blockIdx+1)*totalChunks/coreNum).
  uint32_t blockDim = (totalChunks < aivNum) ? static_cast<uint32_t>(totalChunks) : aivNum;
  if (blockDim == 0) {
    blockDim = kMinBlockDim;
  }
  td->coreNum = blockDim;
  td->totalElems = total;
  td->chunkElems = chunkElems;
  td->reserved0 = 0;
  td->reserved1 = 0;
  td->reserved2 = 0;

  context->SetBlockDim(blockDim);
  context->SetTilingKey(0);

  size_t *ws = context->GetWorkspaceSizes(1);
  if (ws != nullptr) {
    // No device workspace: elements stream straight from x to y.
    ws[0] = 0;
  }
  return GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(FusedGeluTanhNZ).Tiling(FusedGeluTanhNZTilingFunc, sizeof(FusedGeluTanhNZTilingData));
