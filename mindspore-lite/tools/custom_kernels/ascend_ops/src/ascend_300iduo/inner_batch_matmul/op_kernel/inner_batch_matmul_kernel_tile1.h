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
// Small-M / 1x1 paths, the deferred write-back and the core schedules of the InnerBatchMatMul
// kernel.
// Included from inner_batch_matmul_kernel.h after the class declaration (not standalone).
#ifndef INNER_BATCH_MATMUL_KERNEL_TILE1_H_
#define INNER_BATCH_MATMUL_KERNEL_TILE1_H_

namespace InnerBatchMatMul {

// Small-M (M <= 64) main loop: baseM = M (mTiles = 1), single L0 buffer, per-chunk B prefetch
// (not resident). The differences from ProcessTile1, item by item:
//   1. LoadBFull is not used: the whole-column GM->L1 prologue is split into per-chunk
//      LoadChunkB calls that really pipeline with compute;
//   2. A goes through the fused LoadL0Small descriptor (at m1 == 1 the per-chunk A drops from kf
//      LoadData calls to 1);
//   3. the evIn_[2] tokens replace the Sync<MTE2_MTE1> full barrier (one barrier less per
//      chunk); only Sync<MTE1_M> (L0 filled) and Sync<MTE1_MTE2> (L1 slot released) remain;
//   4. the single L0 buffer is released through evL0_ (same semantics as the L0 ping-pong
//      tokens, but there is only one buffer).
__aicore__ inline void InnerBatchMatMulKernel::ProcessTileSmall(uint32_t tileIdx) {
  const uint32_t mTiles = p_->mTiles;  // the host guarantees mTiles == 1 for the small-M shapes
  const uint32_t nTiles = p_->nTiles;
  const uint32_t mi = tileIdx % mTiles;
  const uint32_t ni = (tileIdx / mTiles) % nTiles;
  const uint32_t batchIdx = tileIdx / (mTiles * nTiles);
  const uint32_t mLo = mi * p_->baseM;
  const uint32_t nLo = ni * p_->baseN;
  const uint32_t mRows = (p_->m - mLo) < p_->baseM ? (p_->m - mLo) : p_->baseM;
  const uint32_t nCols = (p_->n - nLo) < p_->baseN ? (p_->n - nLo) : p_->baseN;

  uint64_t offA = 0;
  uint64_t offB = 0;
  uint64_t offC = 0;
  ComputeOffsets(batchIdx, offA, offB, offC);

  // The L0C half must flip per tile: the small-M path forces strideAssign=1 and tileIdx then
  // advances by usedCoreNum (even), so tileIdx&1 would pick the same half and the write-back
  // would collide with the next tile's Mmad.
  const uint32_t cHalf = cFlip_;
  cFlip_ ^= 1U;
  LocalTensor<float> c1 = c1Buf_.Get<float>()[(cHalf * p_->baseM * p_->baseN)];
  const uint32_t m1 = (mRows + kFractalDim - 1) / kFractalDim;
  const uint32_t n1 = (nCols + kFractalDim - 1) / kFractalDim;
  const uint32_t nChunk = (p_->k + kL1_ - 1) / kL1_;

  FlushPending();  // write back the previous tile's C in parallel with this tile's compute

  // The row bias is added by the fp32 vector add in FlushPending (same as ProcessTile1); no L0C
  // preloading here, otherwise it would be accumulated twice.

  SetFlag<HardEvent::M_MTE1>(evL0_);  // seed the release token of the single L0 buffer
  LoadChunkA(0, offA, mLo, m1);
  LoadChunkB(0, offB, nLo, nCols);
  SetFlag<HardEvent::MTE2_MTE1>(evIn_[0]);
  for (uint32_t ci = 0; ci < nChunk; ++ci) {
    const uint32_t s = ci & 1U;
    const uint32_t k0 = ci * kL1_;
    const uint32_t kLenL1 = (p_->k - k0) < kL1_ ? (p_->k - k0) : kL1_;
    const uint32_t kf = (kLenL1 + kFractalDim - 1) / kFractalDim;

    // Prefetch chunk ci+1 into the other L1 slot: the previous chunk's LoadL0 was already
    // completed by the Sync<MTE1_MTE2> at the end of the previous iteration.
    if (ci + 1 < nChunk) {
      LoadChunkA(ci + 1, offA, mLo, m1);
      LoadChunkB(ci + 1, offB, nLo, nCols);
      SetFlag<HardEvent::MTE2_MTE1>(evIn_[1 - s]);
    }
    WaitFlag<HardEvent::MTE2_MTE1>(evIn_[s]);  // L1[s] is ready
    WaitFlag<HardEvent::M_MTE1>(evL0_);        // L0 was released by the previous Mmad
    LoadL0Small(ci, m1, n1, kf);
    Sync<HardEvent::MTE1_M>();
    MmadParams mp;
    mp.m = static_cast<uint16_t>(mRows);
    mp.n = static_cast<uint16_t>(nCols);
    mp.k = static_cast<uint16_t>(kLenL1);
    mp.cmatrixInitVal = (ci == 0);  // no bias / post-add bias: clear the first block; bias never preloads L0C
    mp.isBias = false;
    mp.unitFlag = 0;
    Mmad(c1, L0A(0), L0B(0), mp);
    SetFlag<HardEvent::M_MTE1>(evL0_);  // release L0
    Sync<HardEvent::MTE1_MTE2>();       // MTE1 finished reading L1[s] -> ci+2 may overwrite
  }
  WaitFlag<HardEvent::M_MTE1>(evL0_);  // the last Mmad finished (before write-back / bias Mmad)

  // Record the pending write-back (performed at the start of the next tile, which overlaps).
  pending_ = true;
  pnHalf_ = cHalf;
  pnM1_ = m1;
  pnN1_ = n1;
  pnMlo_ = mLo;
  pnNlo_ = nLo;
  pnOffC_ = offC;
  // Both row bias (biasMode=1) and full bias (2) must carry the batch offset of this batch in the
  // deferred write-back, otherwise every batch uses batch0's bias when the batch count is > 1.
  pnBiasOff_ = (p_->biasMode != 0) ? ComputeBiasOffset(batchIdx) : 0;
}

__aicore__ inline void InnerBatchMatMulKernel::ProcessTile1(uint32_t tileIdx) {
  const uint32_t mTiles = p_->mTiles;
  const uint32_t nTiles = p_->nTiles;
  // Column-major: one core keeps processing all m tiles of the same n column (the premise of
  // resident B).
  const uint32_t mi = tileIdx % mTiles;
  const uint32_t ni = (tileIdx / mTiles) % nTiles;
  const uint32_t batchIdx = tileIdx / (mTiles * nTiles);
  const uint32_t mLo = mi * p_->baseM;
  const uint32_t nLo = ni * p_->baseN;
  const uint32_t mRows = (p_->m - mLo) < p_->baseM ? (p_->m - mLo) : p_->baseM;
  const uint32_t nCols = (p_->n - nLo) < p_->baseN ? (p_->n - nLo) : p_->baseN;

  uint64_t offA = 0;
  uint64_t offB = 0;
  uint64_t offC = 0;
  ComputeOffsets(batchIdx, offA, offB, offC);

  // The L0C half flips per tile (same as cFlip_ on the small-M path). With strideAssign=1
  // tileIdx advances by the even usedCoreNum, so tileIdx&1 would stay on one half and the
  // deferred write-back would degrade to serial.
  const uint32_t cHalf = cFlip_;
  cFlip_ ^= 1U;
  LocalTensor<float> c1 = c1Buf_.Get<float>()[(cHalf * p_->baseM * p_->baseN)];
  const uint32_t m1 = (mRows + kFractalDim - 1) / kFractalDim;
  const uint32_t n1 = (nCols + kFractalDim - 1) / kFractalDim;
  const uint32_t nChunk = (p_->k + kL1_ - 1) / kL1_;

  FlushPending();  // write back the previous tile's C in parallel with this tile's compute

  // Resident B: reload when the column changes.
  const uint32_t bKey = batchIdx * nTiles + ni;
  if (bResident_ && bKey != bKey_) {
    LoadBFull(offB, nLo, nCols);
    bKey_ = bKey;
  }

  SetFlag<HardEvent::M_MTE1>(evM_[0]);  // L0 ping-pong token seed
  SetFlag<HardEvent::M_MTE1>(evM_[1]);
  LoadChunkA(0, offA, mLo, m1);
  if (!bResident_) {
    LoadChunkB(0, offB, nLo, nCols);
  }
  for (uint32_t ci = 0; ci < nChunk; ++ci) {
    const uint32_t k0 = ci * kL1_;
    const uint32_t kLenL1 = (p_->k - k0) < kL1_ ? (p_->k - k0) : kL1_;
    const uint32_t kf = (kLenL1 + 15) / 16;

    Sync<HardEvent::MTE2_MTE1>();
    if (ci + 1 < nChunk) {
      LoadChunkA(ci + 1, offA, mLo, m1);
      if (!bResident_) {
        LoadChunkB(ci + 1, offB, nLo, nCols);
      }
    }
    const uint32_t p = ci & 1U;
    WaitFlag<HardEvent::M_MTE1>(evM_[p]);  // L0[p] was released by the Mmad two steps ago
    LoadL0(ci, p, m1, n1, kf);
    Sync<HardEvent::MTE1_M>();
    MmadParams mp;
    mp.m = static_cast<uint16_t>(mRows);
    mp.n = static_cast<uint16_t>(nCols);
    mp.k = static_cast<uint16_t>(kLenL1);
    mp.cmatrixInitVal = (ci == 0);
    mp.isBias = false;
    Mmad(c1, L0A(p), L0B(p), mp);
    SetFlag<HardEvent::M_MTE1>(evM_[p]);
    Sync<HardEvent::MTE1_MTE2>();  // explicit: L1[buf] order between MTE1 and MTE2
  }
  WaitFlag<HardEvent::M_MTE1>(evM_[0]);
  WaitFlag<HardEvent::M_MTE1>(evM_[1]);

  // Record the pending write-back (performed at the start of the next tile, which overlaps).
  pending_ = true;
  pnHalf_ = cHalf;
  pnM1_ = m1;
  pnN1_ = n1;
  pnMlo_ = mLo;
  pnNlo_ = nLo;
  pnOffC_ = offC;
  // Both row bias (biasMode=1) and full bias (2) must carry the batch offset of this batch in the
  // deferred write-back, otherwise every batch uses batch0's bias when the batch count is > 1.
  pnBiasOff_ = (p_->biasMode != 0) ? ComputeBiasOffset(batchIdx) : 0;
}

__aicore__ inline void InnerBatchMatMulKernel::ProcessTile(uint32_t tileIdx, uint32_t nextTileIdx, bool hasNext) {
  if (smallM_) {
    ProcessTileSmall(tileIdx);
  } else if (tileMul_ > 1U) {
    ProcessTile2(tileIdx, nextTileIdx, hasNext);
  } else {
    ProcessTile1(tileIdx);
  }
}

__aicore__ inline void InnerBatchMatMulKernel::FlushPending() {
  if (!pending_) {
    return;
  }
  pending_ = false;
  Sync<HardEvent::MTE3_V>();  // the previous GM write finished, ubNd (= ub32) may be rewritten by V
  Sync<HardEvent::M_V>();     // the Mmad into L0C[half] finished
  // ub32 is reused as ubNd for the GM write, and the next writer of ub32 is the MTE2 L0C->UB copy
  // below, so the MTE3 -> MTE2 order must be enforced explicitly (MTE3_V alone does not cover it).
  Sync<HardEvent::MTE3_MTE2>();
  LocalTensor<float> ub32 = ubFp32_.Get<float>();
  LocalTensor<half> ub16 = ubFp16_.Get<half>();
  LocalTensor<float> c1 = c1Buf_.Get<float>()[(pnHalf_ * p_->baseM * p_->baseN)];
  DataCopyParams cp;
  cp.blockCount = static_cast<uint16_t>(pnN1_);
  cp.blockLen = static_cast<uint16_t>(pnM1_);
  cp.srcStride = 0;
  cp.dstStride = 0;
  DataCopyEnhancedParams ep;
  ep.blockMode = BlockMode::BLOCK_MODE_MATRIX;
  DataCopy(ub32, c1, cp, ep);
  if (p_->biasMode == kBiasModeFull) {
    // The full-size bias (NZ, fractal order [n][m] with [m][n] inside) matches the UB layout of
    // C. Stage it from GM into UB n-fractal by n-fractal (contiguous inside a fractal) and do a
    // single Cast/Add pass after one synchronization.
    const uint32_t m1All = p_->m / kFractalDim;
    const uint32_t perFrac = pnM1_ * kFractalElems;
    LocalTensor<half> bHalf = ubBiasHalf_.Get<half>();
    LocalTensor<float> bF32 = ubBiasF32_.Get<float>();
    const uint32_t nStage = bGrp_;  // n-fractals covered by one staging round
    for (uint32_t j0 = 0; j0 < pnN1_; j0 += nStage) {
      const uint32_t g = (pnN1_ - j0) < nStage ? (pnN1_ - j0) : nStage;
      for (uint32_t t = 0; t < g; ++t) {
        const uint32_t j = j0 + t;
        const uint64_t off = pnBiasOff_ + (static_cast<uint64_t>(pnNlo_) / kFractalDim + j) * m1All * kFractalElems +
                             (static_cast<uint64_t>(pnMlo_) / kFractalDim) * kFractalElems;
        DataCopy(bHalf[t * perFrac], biasGm_[off], perFrac);  // the m1 fractals of one n-fractal are contiguous
      }
      Sync<HardEvent::MTE2_V>();
      for (uint32_t t = 0; t < g; ++t) {
        const uint32_t idx = (j0 + t) * perFrac;
        Cast(bF32, bHalf[t * perFrac], RoundMode::CAST_NONE, perFrac);
        Add(ub32[idx], ub32[idx], bF32, perFrac);
      }
      Sync<HardEvent::V_MTE2>();
    }
  } else {
    Sync<HardEvent::MTE2_V>();
  }
  if (p_->biasMode == kBiasModeRow) {
    // Row bias ([B...,1,N] stored as NZ [N/16][1][16][16]; the first row of each fractal holds
    // the 16 values of those columns). Move all 16-value rows into UB at once, then use one
    // vector broadcast Add per fractal (src1RepStride = 0).
    const uint32_t n1 = pnN1_;
    const uint32_t m1 = pnM1_;
    const uint32_t perFrac = m1 * kFractalElems;
    LocalTensor<half> bH = ubBiasHalf_.Get<half>();
    LocalTensor<float> b16 = ubBiasF32_.Get<float>();
    // Read n1 fractals contiguously (512B each, only the first 32B row is valid) -> one 4KB copy.
    const uint64_t offBase = pnBiasOff_ + (static_cast<uint64_t>(pnNlo_) / kFractalDim) * kFractalElems;
    DataCopy(bH, biasGm_[offBase], n1 * kFractalElems);
    Sync<HardEvent::MTE2_V>();                 // the row-bias GM->UB copy is issued here, so it needs its own wait
    uint64_t maskBits[2] = {0xFFFFULL, 0ULL};  // 16 fp32 (one row) per repeat
    BinaryRepeatParams rp;
    rp.dstBlkStride = 1;
    rp.src0BlkStride = 1;
    rp.src1BlkStride = 1;
    rp.dstRepStride = 2;  // one row = 16 fp32 = 2 32B blocks
    rp.src0RepStride = 2;
    rp.src1RepStride = 0;  // source repeats -> row broadcast
    for (uint32_t j = 0; j < n1; ++j) {
      Cast(b16, bH[j * kFractalElems], RoundMode::CAST_NONE, kFractalDim);
      // repeatTimes is uint8_t while a tile can hold up to 256 rows (base_m = 256), so the row
      // broadcast is emitted in chunks of at most 255 repeats (each repeat is one 16-fp32 row).
      const uint32_t rows = m1 * kFractalDim;
      for (uint32_t r0 = 0; r0 < rows; r0 += 255U) {
        const uint32_t g = (rows - r0) < 255U ? (rows - r0) : 255U;
        Add<float, true>(ub32[j * perFrac + r0 * kFractalDim], ub32[j * perFrac + r0 * kFractalDim], b16, maskBits,
                         static_cast<uint8_t>(g), rp);
      }
    }
    Sync<HardEvent::V_MTE2>();
  }
  Cast(ub16, ub32, RoundMode::CAST_NONE, pnM1_ * pnN1_ * kFractalElems);

  // NZ([n][m][16][16]) -> ND([m][n]): vector flattening (see Nz2NdByVec), then one 2D GM write.
  LocalTensor<half> ubNd = ub32.ReinterpretCast<half>();
  Nz2NdByVec(ubNd, ub16, pnM1_, pnN1_);
  Sync<HardEvent::V_MTE3>();

  DataCopyParams gout;
  gout.blockCount = static_cast<uint16_t>(pnM1_ * kFractalDim);
  gout.blockLen = static_cast<uint16_t>(pnN1_);
  gout.srcStride = 0;
  gout.dstStride = static_cast<uint16_t>((p_->n - pnN1_ * kFractalDim) / kFractalDim);
  DataCopy(cGm_[pnOffC_ + static_cast<uint64_t>(pnMlo_) * p_->n + pnNlo_], ubNd, gout);
}

// A-stationary phased schedule: core c finishes all ni of mi = c first, then moves to
// mi = c + usedCoreNum. Every core pins exactly one A tile, so the same-wave A resident set is
// usedCoreNum*A_tile (kept within L2) and the A DRAM traffic drops from mTiles*nTiles*A_tile to
// mTiles*A_tile. The batch dim is the outermost loop.
__aicore__ inline void InnerBatchMatMulKernel::RunGrpSched(uint32_t blockIdx) {
  const uint32_t phases = p_->mTiles / p_->usedCoreNum;  // the host gate guarantees divisibility
  const uint32_t nTiles = p_->nTiles;
  const uint32_t tilesPerBatch = p_->mTiles * nTiles;
  // Every batch runs the whole (phase, ni) sequence; the last term of t matches the ProcessTile
  // decoding batchIdx = t/(mTiles*nTiles). With more than one batch every core still pins one A
  // tile inside each batch.
  for (uint32_t b = 0; b < p_->batchCount; ++b) {
    for (uint32_t ph = 0; ph < phases; ++ph) {
      const uint32_t mi = ph * p_->usedCoreNum + blockIdx;
      for (uint32_t ni = 0; ni < nTiles; ++ni) {
        const uint32_t t = b * tilesPerBatch + ni * p_->mTiles + mi;
        const bool more = (ni + 1U < nTiles) || (ph + 1U < phases) || (b + 1U < p_->batchCount);
        uint32_t nx = 0;
        if (more) {
          // Next (batch, phase, ni): advance ni first inside a batch, then the phase, then the batch.
          uint32_t nb = b;
          uint32_t nph = ph;
          uint32_t nni = ni + 1U;
          if (nni >= nTiles) {
            nni = 0U;
            nph = ph + 1U;
            if (nph >= phases) {
              nph = 0U;
              nb = b + 1U;
            }
          }
          nx = nb * tilesPerBatch + nni * p_->mTiles + (nph * p_->usedCoreNum + blockIdx);
        }
        ProcessTile(t, nx, more);
      }
    }
  }
}

// Cross-core stride: at step s the usedCoreNum cores process neighbouring m tiles of the same n
// column, so the B segment is shared by all cores.
__aicore__ inline void InnerBatchMatMulKernel::RunStrideSchedule(uint32_t blockIdx) {
  for (uint32_t t = blockIdx; t < p_->totalTiles; t += p_->usedCoreNum) {
    const uint32_t tn = t + p_->usedCoreNum;
    ProcessTile(t, (tn < p_->totalTiles) ? tn : 0U, tn < p_->totalTiles);
  }
}

// Contiguous assignment: one core owns tilePerCore consecutive tiles. Returns false when this core
// has no tile at all.
__aicore__ inline bool InnerBatchMatMulKernel::RunContiguousSchedule(uint32_t blockIdx) {
  const uint32_t start = blockIdx * p_->tilePerCore;
  if (start >= p_->totalTiles) {
    return false;
  }
  uint32_t end = start + p_->tilePerCore;
  if (end > p_->totalTiles) {
    end = p_->totalTiles;
  }
  for (uint32_t t = start; t < end; ++t) {
    ProcessTile(t, (t + 1U < end) ? (t + 1U) : 0U, t + 1U < end);
  }
  return true;
}

__aicore__ inline void InnerBatchMatMulKernel::Process() {
  if (p_ == nullptr || p_->usedCoreNum == 0) {
    return;
  }
  const uint32_t blockIdx = GetBlockIdx();
  if (blockIdx >= p_->usedCoreNum) {
    return;
  }
  if (p_->strideAssign != 0U) {
    if (p_->grpSched != 0U) {
      RunGrpSched(blockIdx);
    } else {
      RunStrideSchedule(blockIdx);
    }
    FlushPending();  // the last tile's C must be written back
    return;
  }
  if (!RunContiguousSchedule(blockIdx)) {
    return;
  }
  FlushPending();  // the last tile
}

}  // namespace InnerBatchMatMul

#endif  // INNER_BATCH_MATMUL_KERNEL_TILE1_H_
