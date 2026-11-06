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
 * ============================================================================
 */
// Hand-written Cube implementation of InnerBatchMatMul
// (Ascend 310P / dav_m200, fp16, FRACTAL_NZ inputs, no transpose).
//
// Input contract (declared by the op def; GE inserts the ND->NZ conversion):
//   x1 (A): [K/16][M/16][16(m)][16(k)]     x2 (B): [N/16][K/16][16(k)][16(n)]
//   output y: ND [batch..., M, N], batch dims broadcast right-aligned (numpy rules).
//
// Platform semantics (verified on the target SoC):
//   - LoadData: srcStride is a stride (neighbouring fractals are 1), dstGap is a gap;
//     startIndex != 0 is unreliable, so tensor-index offsets express the start;
//   - L0A fractal order is [m][k], L0B is [k][n]; with m1/n1 == 1 or k == 16 the order is
//     indistinguishable, so validate with m1 > 1 and k > 16;
//   - Mmad: k is an element count (multiple of 16); cmatrixInitVal=1 clears the first block,
//     0 accumulates;
//   - L0C fractal order is [n][m]; L0C -> UB must use the Enhanced overload
//     (BLOCK_MODE_MATRIX) with blockCount=n1 and blockLen=m1.
//
// Schedule and pipeline (tileMul=1, single tile):
//   - Tiles are assigned column-major (m varies fastest), so one core keeps working on the
//     same n column and the whole-K B block stays resident in L1 across m tiles;
//   - Per K chunk: MTE2 prefetches A/B -> MTE1 loads the chunk into L0 -> one Mmad; L0A/L0B
//     use two ping-pong buffers (the GemmExecNmPingPong pattern) so LoadL0 overlaps the
//     previous Mmad;
//   - C write-back (L0C->UB->Cast->NZ2ND->GM) is deferred to the start of the next tile so
//     it overlaps the next tile's compute.
//
// 2x2 output blocking for large K (tileMul=2):
//   - One "super tile" covers (2*baseM) x (2*baseN); L0C holds two row-quadrant accumulators
//     (baseM x 2*baseN each, 256KB in total; every accumulator is split into two
//     baseM x baseN halves in the [n][m] order used by the write-back);
//   - Every K chunk loads A(2*baseM x kL1) and B(kL1 x 2*baseN) once and reuses each of them
//     for two accumulators, halving the A/B re-reads from GM (MTE2-bound for large K; after
//     the halving the kernel is close to the Cube bound);
//   - Cost: the four accumulators occupy the whole 256KB L0C, so there is no L0C ping-pong
//     and the write-back is serialized at the end of the tile. It is only enabled for
//     "large K + per-core tile count does not grow", where the write-back share is small.
//   - Full-size bias (biasMode=2) uses a rank-16 decomposition: the tile's bias is prefetched
//     to L1 and an extra "identity I(128x128) x bias" Mmad writes it directly into L0C in fp32,
//     so the write-back does no bias vector work at all.
//
#ifndef INNER_BATCH_MATMUL_KERNEL_H_
#define INNER_BATCH_MATMUL_KERNEL_H_

#include "kernel_operator.h"                 // NOLINT(build/include_subdir)
#include "inner_batch_matmul_tiling_data.h"  // NOLINT(build/include_subdir)

using namespace AscendC;  // NOLINT(build/namespaces)

// NOTE: only the kernel ENTRY (in inner_batch_mat_mul.cpp) and the tiling macros must stay in the
// global namespace (see inner_batch_matmul_tiling_data.h); the kernel class, its helpers and the
// constants live in the InnerBatchMatMul namespace. The definitions are split over several headers
// so that no single header grows past the file-size gate; all of them share this namespace.
namespace InnerBatchMatMul {

// GM pointers of the op, grouped so that Init keeps a short parameter list.
struct GmAddrs {
  GM_ADDR x1;
  GM_ADDR x2;
  GM_ADDR bias;
  GM_ADDR y;
};

// Per K-chunk scalars of the 2x2 loop (computed once per chunk, shared by the load/Mmad steps).
struct Tile2Chunk {
  uint32_t ci = 0;
  uint32_t kf = 0;
  uint32_t kLenL1 = 0;
  bool isFull = false;
};

// Hoisted per-tile state and descriptors of the 2x2 path (constant inside a tile).
struct Tile2Tile {
  uint32_t m1 = 0;
  uint32_t n1 = 0;
  uint32_t m1sub = 0;
  uint32_t n1sub = 0;
  uint32_t kfFull = 0;
  uint32_t nChunk = 0;
  uint32_t nChunkFull = 0;
  uint32_t qm = 0;
  uint32_t qn = 0;
  uint32_t cq1 = 0;
  uint32_t aq1 = 0;
  uint32_t mLo = 0;
  uint32_t nLo = 0;
  uint32_t nCols = 0;
  uint64_t aBase0 = 0;
  uint64_t bBase0 = 0;
  uint64_t aStepElems = 0;
  uint64_t bStepElems = 0;
  uint64_t offA = 0;
  uint64_t offB = 0;
  uint64_t offC = 0;
  DataCopyParams paFull;
  DataCopyParams pbFull;
  LoadData2DParams ldaFull;
  LoadData2DParams ldbFull;
  MmadParams mpC;
  MmadParams mpE;
};

// Geometry of one write-back quadrant of the 2x2 path.
struct QuadDesc {
  uint32_t l0cOff = 0;
  uint32_t m1 = 0;
  uint32_t n1 = 0;
  uint32_t mRowOff = 0;
  uint32_t nColOff = 0;
  uint64_t cOff = 0;
};

class InnerBatchMatMulKernel {
 public:
  using Params = InnerBatchMatMulParams;

  __aicore__ inline InnerBatchMatMulKernel() = default;

  __aicore__ inline void Init(GmAddrs addrs, const Params *p, TPipe *pipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void ProcessTile(uint32_t tileIdx, uint32_t nextTileIdx, bool hasNext);
  __aicore__ inline void ProcessTileSmall(uint32_t tileIdx);
  __aicore__ inline void ProcessTile1(uint32_t tileIdx);
  __aicore__ inline void ProcessTile2(uint32_t tileIdx, uint32_t nextTileIdx, bool hasNext);
  __aicore__ inline void Tile2Prepare(const Tile2Tile &t, uint64_t biasOff, uint32_t batchIdx, uint32_t ni);
  __aicore__ inline void Tile2PrefetchChunk(const Tile2Chunk &ck, const Tile2Tile &t);
  __aicore__ inline void Tile2LoadL0(const Tile2Chunk &ck, uint32_t p, const Tile2Tile &t);
  __aicore__ inline void Tile2Mmad(const Tile2Chunk &ck, LocalTensor<float> &c1, Tile2Tile &t);
  __aicore__ inline void Tile2InitMmadParams(Tile2Tile &t);
  __aicore__ inline void Tile2FirstChunk(const Tile2Tile &t);
  __aicore__ inline void Tile2PrefetchNextBias(uint32_t nextTileIdx, bool hasNext);
  __aicore__ inline void Tile2WriteBack(LocalTensor<float> &c1, const Tile2Tile &t);
  __aicore__ inline void FlushPending();
  __aicore__ inline void InitSmallMBuffers(const Params *p);
  __aicore__ inline void InitMainBuffers(const Params *p, uint64_t bFullBytes);
  __aicore__ inline void InitTile2Buffers(const Params *p);
  __aicore__ inline void InitTile1Buffers(const Params *p);
  __aicore__ inline void AllocFullBiasStaging(const Params *p);
  __aicore__ inline void RunGrpSched(uint32_t blockIdx);
  __aicore__ inline void RunStrideSchedule(uint32_t blockIdx);
  __aicore__ inline bool RunContiguousSchedule(uint32_t blockIdx);
  __aicore__ inline void ComputeOffsets(uint32_t batchIdx, uint64_t &offA, uint64_t &offB, uint64_t &offC) const;
  __aicore__ inline uint64_t ComputeBiasOffset(uint32_t batchIdx) const;
  __aicore__ inline void LoadChunkA(uint32_t ci, uint64_t offA, uint32_t mLo, uint32_t m1);
  __aicore__ inline void LoadChunkB(uint32_t ci, uint64_t offB, uint32_t nLo, uint32_t nCols);
  __aicore__ inline void LoadBFull(uint64_t offB, uint32_t nLo, uint32_t nCols);
  __aicore__ inline void LoadL0(uint32_t ci, uint32_t buf, uint32_t m1, uint32_t n1, uint32_t kf);
  __aicore__ inline void LoadL0Small(uint32_t ci, uint32_t m1, uint32_t n1, uint32_t kf);
  __aicore__ inline void LoadL02(uint32_t ci, uint32_t buf, uint32_t m1, uint32_t n1, uint32_t kf);
  __aicore__ inline void PrefetchBiasL1(uint32_t slot, uint32_t m1, uint32_t n1, uint64_t biasOffTile);
  __aicore__ inline void BuildBiasAux(uint64_t biasOff, uint32_t nLo, uint32_t n1);
  __aicore__ inline void IssueBiasMmad(LocalTensor<float> &c1, uint32_t m1, uint32_t n1, uint32_t qm);
  __aicore__ inline void BuildBiasIdentL1();
  __aicore__ inline void IssueBiasMmadFull(LocalTensor<float> &c1, uint32_t m1, uint32_t n1, uint32_t qm, uint32_t qn);
  __aicore__ inline void WriteBackQuad(const QuadDesc &q);
  __aicore__ inline void Nz2NdByVec(LocalTensor<half> dst, LocalTensor<half> src, uint32_t m1, uint32_t n1);
  __aicore__ inline LocalTensor<half> L0A(uint32_t buf);
  __aicore__ inline LocalTensor<half> L0B(uint32_t buf);
  __aicore__ inline LocalTensor<half> L1A(uint32_t buf);
  __aicore__ inline LocalTensor<half> L1B(uint32_t buf);
  template <HardEvent e>
  __aicore__ inline void Sync();

  GlobalTensor<half> aGm_;
  GlobalTensor<half> bGm_;
  GlobalTensor<half> cGm_;
  GlobalTensor<half> biasGm_;
  const Params *p_ = nullptr;
  TPipe *pipe_ = nullptr;

  static constexpr uint32_t kFractalDim = 16;                           // fractal edge length
  static constexpr uint32_t kFractalElems = kFractalDim * kFractalDim;  // elements per fractal
  static constexpr uint64_t kBFullLimit = 768ULL << 10;                 // max bytes for a whole-K B in L1
  // Conservative upper bound of k for a single Mmad on the small-M path (the generic path uses
  // kL1 = 64/128); lowering it to 128 degrades to a smaller chunk with the same chunk count.
  static constexpr uint32_t kSmallMaxKL1 = 512;
  static constexpr uint32_t kBlockBytes = 32;        // one 32-byte block, the unit of a DataCopy blockLen
  static constexpr uint32_t kDummyL1Bytes = 32;      // placeholder allocation of the unused second L1 slot
  static constexpr uint32_t kUbBytes = 256U * 1024;  // total UB capacity
  static constexpr uint32_t kBiasModeRow = 1;        // biasMode: row broadcast [batch..., 1, N]
  static constexpr uint32_t kBiasModeFull = 2;       // biasMode: full size [batch..., M, N]
  uint32_t kL1_ = 256;    // K span loaded into L1 at once (default; recomputed in Init from baseN/tileMul)
  uint32_t tileMul_ = 1;  // 1 = single tile; 2 = 2x2 output blocking
  uint32_t tileM_ = 128;  // super-tile rows (= tileMul * baseM)
  uint32_t tileN_ = 128;  // super-tile columns (= tileMul * baseN)

  TBuf<TPosition::A1> a1Buf_;
  TBuf<TPosition::A1> a1Buf2_;
  TBuf<TPosition::A1> b1AuxA_;   // synthetic A' for row bias (all ones, tileM x 16)
  TBuf<TPosition::A1> b1AuxB_;   // synthetic B' for row bias (row 0 = bias, rest 0, tileN x 16)
  TBuf<TPosition::B1> b1Bias_;   // full-size bias of a whole tile (GM->L1 prefetch; B' source of the bias Mmad on 2x2)
  TBuf<TPosition::A1> b1Ident_;  // A' of the full-bias rank-16 decomposition (128x128 block-diagonal identity, 2x2)
  TBuf<TPosition::B1> b1Buf_;    // resident B (or the single chunk buffer when not resident)
  TBuf<TPosition::B1> b1Buf2_;   // second chunk buffer when B is not resident
  TBuf<TPosition::A2> a2Buf_;
  TBuf<TPosition::B2> b2Buf_;
  TBuf<TPosition::A2> a2Buf2_;
  TBuf<TPosition::B2> b2Buf2_;
  TBuf<TPosition::CO1> c1Buf_;
  TBuf<TPosition::VECCALC> ubFp32_;
  TBuf<TPosition::VECCALC> ubFp16_;
  TBuf<TPosition::VECCALC> ubNd_;        // NZ->ND flattening output (2x2 only; 1x1 reuses ubFp32)
  TBuf<TPosition::VECCALC> ubBiasHalf_;  // bias staging (fp16, NZ, grouped; 2x2 full bias does not use it)
  TBuf<TPosition::VECCALC> ubBiasF32_;   // bias staging (fp32, for the fp32 add)
  TBuf<TPosition::VECCALC> ubIdent_;     // UB staging of the identity A' (built once, then copied to b1Ident_)
  TBuf<TPosition::VECCALC> ubAux_;       // UB staging of the synthetic A'/B' ((tileM + tileN) * 16 half)

  bool pending_ = false;  // C of the previous tile not yet written back (overlaps the next tile, tileMul=1 only)
  uint32_t cFlip_ = 0;    // L0C half selection, flipped once per tile
                          // (with strideAssign=1 the tileIdx step is the even core count, so tileIdx&1 is constant)
  uint32_t pnHalf_ = 0, pnM1_ = 0, pnN1_ = 0, pnMlo_ = 0, pnNlo_ = 0;
  uint64_t pnOffC_ = 0;
  uint64_t pnBiasOff_ = 0;  // bias batch offset of the pending write-back tile
  event_t evM_[2];          // M->MTE1: L0 ping-pong tokens (GemmExecNmPingPong pattern)
  bool smallM_ = false;     // small-M path (M <= 64): single L0 buffer + per-chunk prefetch
  event_t evIn_[2]{};       // MTE2->MTE1: per-slot "L1 data ready" (replaces the full barrier)
  event_t evL0_;            // M->MTE1: release token of the single L0 buffer (small-M path)
  bool bResident_ = false;
  uint32_t bKey_ = 0xFFFFFFFFU;        // currently resident (batch, ni)
  uint32_t biasAuxKey_ = 0xFFFFFFFFU;  // cache key of the synthetic bias blocks (batch, ni)
  uint32_t kfAll_ = 1;                 // K/16
  uint32_t bGrp_ = 1;                  // n-fractals staged per bias copy/add
  uint32_t curBiasSlot_ = 0;           // L1 slot of the full-size bias (0/1)
  bool hasPrefetch_ = false;           // this tile's bias was prefetched by the previous tile
};

__aicore__ inline LocalTensor<half> InnerBatchMatMulKernel::L0A(uint32_t buf) {
  return (buf & 1U) != 0U ? a2Buf2_.Get<half>() : a2Buf_.Get<half>();
}

__aicore__ inline LocalTensor<half> InnerBatchMatMulKernel::L0B(uint32_t buf) {
  return (buf & 1U) != 0U ? b2Buf2_.Get<half>() : b2Buf_.Get<half>();
}

template <HardEvent e>
__aicore__ inline void InnerBatchMatMulKernel::Sync() {
  const event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(e));
  SetFlag<e>(ev);
  WaitFlag<e>(ev);
}

__aicore__ inline LocalTensor<half> InnerBatchMatMulKernel::L1A(uint32_t buf) {
  return (buf & 1U) != 0U ? a1Buf2_.Get<half>() : a1Buf_.Get<half>();
}

__aicore__ inline LocalTensor<half> InnerBatchMatMulKernel::L1B(uint32_t buf) {
  return (buf & 1U) != 0U ? b1Buf2_.Get<half>() : b1Buf_.Get<half>();
}

__aicore__ inline void InnerBatchMatMulKernel::Init(GmAddrs addrs, const Params *p, TPipe *pipe) {
  p_ = p;
  pipe_ = pipe;
  if (p == nullptr || p->usedCoreNum == 0 || p->batchCount == 0) {
    return;
  }
  aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(addrs.x1), 1ULL << 31);
  bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(addrs.x2), 1ULL << 31);
  cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(addrs.y), 1ULL << 31);
  biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(addrs.bias), 1ULL << 31);

  smallM_ = (p->smallM != 0U);
  tileMul_ = (p->tileMul > 1U) ? 2U : 1U;
  if (smallM_) {
    tileMul_ = 1U;
  }
  tileM_ = p->baseM * tileMul_;
  tileN_ = p->baseN * tileMul_;
  const uint64_t bFullBytes = static_cast<uint64_t>(p->k) * tileN_ * sizeof(half);
  bResident_ = bFullBytes <= kBFullLimit;
  kfAll_ = (p->k + kFractalDim - 1) / kFractalDim;
  if (smallM_) {
    InitSmallMBuffers(p);
    return;
  }
  InitMainBuffers(p, bFullBytes);
  if (tileMul_ > 1U) {
    InitTile2Buffers(p);
  } else {
    InitTile1Buffers(p);
  }
}

// Resource layout of the small-M path (M <= 64): L0A/L0B use a single buffer each and kL1 is derived
// from the 64KB capacity. B is not resident because mTiles is always 1 for these shapes.
__aicore__ inline void InnerBatchMatMulKernel::InitSmallMBuffers(const Params *p) {
  //   L0B: baseN*kL1*2 <= 64KB (baseN=256 -> 128, 128 -> 256, 64 -> 512, 32 -> 1024);
  //   L0A: m1*16*kL1*2 <= 64KB (M=64 shrinks kL1 further);
  // The small-M Mmad is very short (16 x baseN x kL1), so the serial "fill -> Mmad" cost of a
  // single buffer is far below the barrier/LoadData/Mmad count saved by fewer chunks.
  bResident_ = false;
  uint32_t kl = (64U << 10) / (tileN_ * static_cast<uint32_t>(sizeof(half)));
  const uint32_t klA = (64U << 10) / (tileM_ * static_cast<uint32_t>(sizeof(half)));
  if (kl > klA) {
    kl = klA;
  }
  if (kl > p->k) {
    kl = p->k;
  }
  // Conservative k bound of a single Mmad (see kSmallMaxKL1).
  if (kl > kSmallMaxKL1) {
    kl = kSmallMaxKL1;
  }
  kl = kl / kFractalDim * kFractalDim;
  kL1_ = (kl == 0U) ? kFractalDim : kl;
  const uint32_t aElemsS = tileM_ * kL1_;
  const uint32_t bElemsS = kL1_ * tileN_;
  pipe_->InitBuffer(a1Buf_, aElemsS * sizeof(half));
  pipe_->InitBuffer(a1Buf2_, aElemsS * sizeof(half));
  pipe_->InitBuffer(b1Buf_, bElemsS * sizeof(half));
  pipe_->InitBuffer(b1Buf2_, bElemsS * sizeof(half));
  pipe_->InitBuffer(a2Buf_, aElemsS * sizeof(half));
  pipe_->InitBuffer(b2Buf_, bElemsS * sizeof(half));
  pipe_->InitBuffer(c1Buf_, 2U * p->baseM * p->baseN * sizeof(float));  // two halves: write-back overlaps next tile
  pipe_->InitBuffer(ubFp32_, p->baseM * p->baseN * sizeof(float));
  pipe_->InitBuffer(ubFp16_, p->baseM * p->baseN * sizeof(half));
  if (p->biasMode == kBiasModeFull) {
    AllocFullBiasStaging(p);
  } else if (p->biasMode == kBiasModeRow) {
    const uint32_t n1 = p->baseN / 16;
    const uint32_t m1 = p->baseM / 16;
    pipe_->InitBuffer(ubBiasHalf_, n1 * kFractalElems * sizeof(half));
    pipe_->InitBuffer(ubBiasF32_, (256U + m1 * 256U) * sizeof(float));
    pipe_->InitBuffer(b1AuxA_, tileM_ * kFractalDim * sizeof(half));
    pipe_->InitBuffer(b1AuxB_, tileN_ * kFractalDim * sizeof(half));
    pipe_->InitBuffer(ubAux_, (tileM_ + tileN_) * kFractalDim * sizeof(half));
  }
  evIn_[0] = static_cast<event_t>(pipe_->AllocEventID<HardEvent::MTE2_MTE1>());
  evIn_[1] = static_cast<event_t>(pipe_->AllocEventID<HardEvent::MTE2_MTE1>());
  evL0_ = static_cast<event_t>(pipe_->AllocEventID<HardEvent::M_MTE1>());
}

// Buffers shared by the tileMul=1/2 paths: L1/L0 ping-pong slots and the L0 ping-pong tokens.
__aicore__ inline void InnerBatchMatMulKernel::InitMainBuffers(const Params *p, uint64_t bFullBytes) {
  // Per-slot MTE2->MTE1 tokens of the 2x2 main loop: each L1 buffer gets a "data has landed"
  // event, which replaces the per-chunk Sync<MTE2_MTE1> full barrier (independent of the L0
  // ping-pong tokens evM_). The 1x1 paths do not use these two IDs.
  evIn_[0] = static_cast<event_t>(pipe_->AllocEventID<HardEvent::MTE2_MTE1>());
  evIn_[1] = static_cast<event_t>(pipe_->AllocEventID<HardEvent::MTE2_MTE1>());
  // L0B capacity: tileN * kL1 <= 32768 elements (64KB); L0A: tileM * kL1 <= 32768 elements (64KB).
  // Single tile: kL1 = 128 for baseN <= 128, otherwise 64; 2x2: tileM = tileN = 256 -> kL1 = 64
  // so that both ping-pong pairs still fit.
  kL1_ = (tileMul_ > 1U || p->baseN > 128U) ? 64U : 128U;
  if (kL1_ > p->k) {
    kL1_ = p->k;
  }
  const uint32_t aElems = tileM_ * kL1_;
  pipe_->InitBuffer(a1Buf_, aElems * sizeof(half));
  pipe_->InitBuffer(a1Buf2_, aElems * sizeof(half));
  if (bResident_) {
    pipe_->InitBuffer(b1Buf_, static_cast<uint32_t>(bFullBytes));
    pipe_->InitBuffer(b1Buf2_, kDummyL1Bytes);
  } else {
    pipe_->InitBuffer(b1Buf_, kL1_ * tileN_ * sizeof(half));
    pipe_->InitBuffer(b1Buf2_, kL1_ * tileN_ * sizeof(half));
  }
  pipe_->InitBuffer(a2Buf_, aElems * sizeof(half));
  pipe_->InitBuffer(b2Buf_, kL1_ * tileN_ * sizeof(half));
  pipe_->InitBuffer(a2Buf2_, aElems * sizeof(half));
  pipe_->InitBuffer(b2Buf2_, kL1_ * tileN_ * sizeof(half));

  evM_[0] = static_cast<event_t>(pipe_->AllocEventID<HardEvent::M_MTE1>());
  evM_[1] = static_cast<event_t>(pipe_->AllocEventID<HardEvent::M_MTE1>());
}

// 2x2 path: four accumulators fill the whole L0C; the write-back walks the quadrants serially.
__aicore__ inline void InnerBatchMatMulKernel::InitTile2Buffers(const Params *p) {
  pipe_->InitBuffer(c1Buf_, tileM_ * tileN_ * sizeof(float));
  pipe_->InitBuffer(ubFp32_, p->baseM * p->baseN * sizeof(float));
  pipe_->InitBuffer(ubFp16_, p->baseM * p->baseN * sizeof(half));
  pipe_->InitBuffer(ubNd_, p->baseM * p->baseN * sizeof(half));
  if (p->biasMode == kBiasModeFull) {
    // Full-size bias uses the rank-16 decomposition: the whole-tile bias is prefetched to L1
    // as the B' source, then an extra "identity x bias" Mmad writes bias into L0C in exact fp32,
    // so the write-back needs no L1->UB copy + Cast + Add. Precondition: with tileMul=2 the
    // tiling guarantees baseM = baseN = 128, i.e. the identity is 8x8 fractals = 32KB, exactly
    // one L0A slot.
    pipe_->InitBuffer(b1Bias_, 4U * p->baseM * tileN_ * sizeof(half));  // 2 slots x whole tile
    const uint32_t idDim = p->baseM / kFractalDim;
    pipe_->InitBuffer(b1Ident_, idDim * idDim * kFractalElems * sizeof(half));
    pipe_->InitBuffer(ubIdent_, idDim * idDim * kFractalElems * sizeof(half));
    BuildBiasIdentL1();
  } else if (p->biasMode == kBiasModeRow) {
    const uint32_t n1 = p->baseN / 16;
    pipe_->InitBuffer(ubBiasHalf_, n1 * kFractalElems * sizeof(half));
    pipe_->InitBuffer(ubBiasF32_, kFractalElems * sizeof(float));
    // Row bias uses one extra K=16 Mmad: A' = all ones, row 0 of B' = bias.
    pipe_->InitBuffer(b1AuxA_, tileM_ * kFractalDim * sizeof(half));
    pipe_->InitBuffer(b1AuxB_, tileN_ * kFractalDim * sizeof(half));
    pipe_->InitBuffer(ubAux_, (tileM_ + tileN_) * kFractalDim * sizeof(half));
  }
}

// tileMul=1 path: two L0C halves so that the write-back overlaps the next tile, plus the bias staging.
__aicore__ inline void InnerBatchMatMulKernel::InitTile1Buffers(const Params *p) {
  pipe_->InitBuffer(c1Buf_, 2U * p->baseM * p->baseN * sizeof(float));
  pipe_->InitBuffer(ubFp32_, p->baseM * p->baseN * sizeof(float));
  pipe_->InitBuffer(ubFp16_, p->baseM * p->baseN * sizeof(half));
  if (p->biasMode == kBiasModeFull) {
    // Full-size bias: whole-tile fp16 staging + fp32 scratch of one n-fractal.
    AllocFullBiasStaging(p);
  } else if (p->biasMode == kBiasModeRow) {
    // Row bias: n1*16 half per tile + (256 + m1*256) fp32 (pattern + replication).
    const uint32_t n1 = p->baseN / 16;
    const uint32_t m1 = p->baseM / 16;
    pipe_->InitBuffer(ubBiasHalf_, n1 * kFractalElems * sizeof(half));  // whole segment (with padding)
    pipe_->InitBuffer(ubBiasF32_, (256U + m1 * 256U) * sizeof(float));
  }
}

// Bias staging shared by the full-size bias paths: whole-tile fp16 staging + fp32 scratch of one
// n-fractal, sized to the remaining UB budget.
__aicore__ inline void InnerBatchMatMulKernel::AllocFullBiasStaging(const Params *p) {
  const uint32_t perFrac = (p->baseM / 16) * 256;
  const uint32_t n1 = p->baseN / 16;
  // UB budget: ubFp32 + ubFp16 are always live; ubBiasHalf and ubBiasF32 are sized below.
  const uint32_t usedBytes = p->baseM * p->baseN * (sizeof(float) + sizeof(half));
  const uint32_t scratchBytes = perFrac * sizeof(float) + 8192U;
  const uint32_t freeBytes = (usedBytes + scratchBytes < kUbBytes) ? (kUbBytes - usedBytes - scratchBytes) : 0U;
  uint32_t grp = (perFrac > 0) ? (freeBytes / (perFrac * sizeof(half))) : 1U;
  bGrp_ = grp < 1U ? 1U : (grp > n1 ? n1 : grp);
  pipe_->InitBuffer(ubBiasHalf_, bGrp_ * perFrac * sizeof(half));
  pipe_->InitBuffer(ubBiasF32_, perFrac * sizeof(float));
}

// Bias offset of the pending tile (full-size / row bias, right-aligned batch broadcast).
__aicore__ inline uint64_t InnerBatchMatMulKernel::ComputeBiasOffset(uint32_t batchIdx) const {
  uint64_t off = 0;
  uint32_t remain = batchIdx;
  for (int32_t j = static_cast<int32_t>(p_->batchDimNum) - 1; j >= 0; --j) {
    const uint32_t dim = p_->outBatchDim[j];
    if (dim == 0) {
      return off;
    }
    const uint32_t idx = remain % dim;
    remain /= dim;
    off += static_cast<uint64_t>(idx) * p_->biasBatchStride[j];
  }
  return off;
}

__aicore__ inline void InnerBatchMatMulKernel::ComputeOffsets(uint32_t batchIdx, uint64_t &offA, uint64_t &offB,
                                                              uint64_t &offC) const {
  offA = 0;
  offB = 0;
  offC = 0;
  uint32_t remain = batchIdx;
  for (int32_t j = static_cast<int32_t>(p_->batchDimNum) - 1; j >= 0; --j) {
    const uint32_t dim = p_->outBatchDim[j];
    if (dim == 0) {
      return;
    }
    const uint32_t idx = remain % dim;
    remain /= dim;
    offA += static_cast<uint64_t>(idx) * p_->x1BatchStride[j];
    offB += static_cast<uint64_t>(idx) * p_->x2BatchStride[j];
    offC += static_cast<uint64_t>(idx) * p_->cBatchStride[j];
  }
}

}  // namespace InnerBatchMatMul

// Definitions split out of this header (file-size gate); all of them reopen namespace
// InnerBatchMatMul and only need the class declaration above.
#include "inner_batch_matmul_kernel_load.h"   // NOLINT(build/include_subdir)
#include "inner_batch_matmul_kernel_bias.h"   // NOLINT(build/include_subdir)
#include "inner_batch_matmul_kernel_tile2.h"  // NOLINT(build/include_subdir)
#include "inner_batch_matmul_kernel_tile1.h"  // NOLINT(build/include_subdir)

#endif  // INNER_BATCH_MATMUL_KERNEL_H_
