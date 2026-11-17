/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
// Host/device shared parameters of the hand-written Cube channel (POD, packed to 8 bytes so that
// both sides agree byte for byte).
// Note: the structs must live in the global namespace (required by the CANN tiling macros) and the
// host and device copies must stay identical. CANN's REGISTER_TILING_DEFAULT / GET_TILING_DATA
// autogen emits the tiling key and references the struct from global scope; namespacing it makes
// those symbols <ns>::-qualified and they fail to resolve, aborting the device-kernel compile.
// Only the size/alignment constants below are namespaced.
#ifndef INNER_BATCH_MATMUL_TILING_DATA_H_
#define INNER_BATCH_MATMUL_TILING_DATA_H_

#include <cstdint>

namespace InnerBatchMatMul {
// Upper bound of the broadcast batch rank. The kernel walks the arrays below with this many
// entries (batchDimNum is always this value), so the host shape rules must agree -- the host
// tiling static_asserts IBMM_MAX_BATCH_DIM == kIbmmMaxBatchDim.
constexpr uint32_t kIbmmMaxBatchDim = 6;
// The 8-byte packing CANN assumes for the host->device tiling byte stream; #pragma pack and
// alignas must use the same value.
constexpr uint32_t kTilingPackAlign = 8;
}  // namespace InnerBatchMatMul

#pragma pack(push, 8)
struct alignas(InnerBatchMatMul::kTilingPackAlign) InnerBatchMatMulParams {
  uint32_t m;            // output rows M (second-to-last dim of x1)
  uint32_t n;            // output columns N (last dim of x2)
  uint32_t k;            // reduction dim K
  uint32_t batchCount;   // total batch count after broadcasting
  uint32_t coreNum;      // available AI Core count
  uint32_t usedCoreNum;  // cores actually used (= even split of min(totalTiles, coreNum))
  uint32_t baseM;        // tile size along M (= rows of one L0C accumulator)
  uint32_t baseN;        // tile size along N (= columns of one L0C accumulator)
  uint32_t tileMul;      // 1 = single tile; 2 = 2x2 output blocking (4 L0C accumulators, A/B reused 2x)
  uint32_t mTiles;       // tile count along M
  uint32_t nTiles;       // tile count along N
  uint32_t totalTiles;   // batchCount * mTiles * nTiles
  uint32_t tilePerCore;  // tiles per core (contiguous column-major assignment)
  uint32_t batchDimNum;  // number of broadcast batch dims (always 6, right aligned)
  uint32_t outBatchDim[InnerBatchMatMul::kIbmmMaxBatchDim];    // size of every broadcast batch dim
  uint64_t x1BatchStride[InnerBatchMatMul::kIbmmMaxBatchDim];  // element stride of each x1 batch dim (0 when broadcast)
  uint64_t x2BatchStride[InnerBatchMatMul::kIbmmMaxBatchDim];  // element stride of each x2 batch dim (0 when broadcast)
  uint64_t cBatchStride[InnerBatchMatMul::kIbmmMaxBatchDim];   // element stride of each output batch dim
  uint32_t strideAssign;  // 1 = cross-core stride assignment (cores share one B column when B is not resident)
  uint32_t grpSched;      // 1 = A-stationary phased schedule (each core sweeps all ni of one mi before moving on)
  uint32_t biasMode;      // 0 = no bias; 1 = row bias ([batch..., 1, N]); 2 = full bias ([batch..., M, N])
  uint64_t
    biasBatchStride[InnerBatchMatMul::kIbmmMaxBatchDim];  // element stride of each bias batch dim (0 when broadcast)
  uint32_t smallM;  // 1 = small-M path (M <= 64): baseM = M, single L0 buffer, per-chunk prefetch
};

struct alignas(InnerBatchMatMul::kTilingPackAlign) InnerBatchMatMulTilingData {
  InnerBatchMatMulParams cube;  // Cube channel parameters
  uint32_t useCube;             // reserved (always 1)
};
#pragma pack(pop)

#endif  // INNER_BATCH_MATMUL_TILING_DATA_H_
