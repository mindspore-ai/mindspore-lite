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
// 2x2 output-blocking main loop of the InnerBatchMatMul kernel.
// Included from inner_batch_matmul_kernel.h after the class declaration (not standalone).
#ifndef INNER_BATCH_MATMUL_KERNEL_TILE2_H_
#define INNER_BATCH_MATMUL_KERNEL_TILE2_H_

namespace InnerBatchMatMul {

// 2x2 main loop: one super tile = 4 accumulators (baseM x baseN), A and B are each reused twice.
__aicore__ inline void InnerBatchMatMulKernel::ProcessTile2(uint32_t tileIdx, uint32_t nextTileIdx, bool hasNext) {
  const uint32_t mTiles = p_->mTiles;
  const uint32_t nTiles = p_->nTiles;
  const uint32_t mi = tileIdx % mTiles;
  const uint32_t ni = (tileIdx / mTiles) % nTiles;
  const uint32_t batchIdx = tileIdx / (mTiles * nTiles);
  Tile2Tile t;
  t.mLo = mi * tileM_;
  t.nLo = ni * tileN_;
  const uint32_t mRows = (p_->m - t.mLo) < tileM_ ? (p_->m - t.mLo) : tileM_;
  t.nCols = (p_->n - t.nLo) < tileN_ ? (p_->n - t.nLo) : tileN_;
  t.m1 = (mRows + kFractalDim - 1) / kFractalDim;
  t.n1 = (t.nCols + kFractalDim - 1) / kFractalDim;
  t.m1sub = p_->baseM / kFractalDim;
  t.n1sub = p_->baseN / kFractalDim;
  t.qm = (t.m1 + t.m1sub - 1) / t.m1sub;
  t.qn = (t.n1 + t.n1sub - 1) / t.n1sub;
  t.nChunk = (p_->k + kL1_ - 1) / kL1_;

  ComputeOffsets(batchIdx, t.offA, t.offB, t.offC);
  const uint64_t biasOff = (p_->biasMode != 0) ? ComputeBiasOffset(batchIdx) : 0;
  LocalTensor<float> c1 = c1Buf_.Get<float>();

  Tile2Prepare(t, biasOff, batchIdx, ni);

  // ---- Hoisted tile-level state (descriptors/constants/tensors are built once; full chunks take
  // the fast path and tail chunks fall back to the generic helpers). ----
  const uint32_t kAll = p_->k;
  const uint32_t kL1 = kL1_;
  const uint32_t m1All = p_->m / kFractalDim;  // global M/16 (stride of the GM->L1 copies)
  // The L1->L0 source stride must be the real m-fractal count m1 of this tile (GM->L1 writes a
  // compact [k][m1] block); using tileM_/16 would read A from the wrong k-fractal for an M tail
  // tile (m1 < tileM_/16).
  t.kfFull = kL1 / kFractalDim;  // k-fractals of a full chunk
  t.nChunkFull = kAll / kL1;     // number of full chunks
  t.aBase0 = t.offA + (static_cast<uint64_t>(t.mLo) / kFractalDim) * kFractalElems;
  t.bBase0 = t.offB + (static_cast<uint64_t>(t.nLo) / kFractalDim) * (kAll / kFractalDim) * kFractalElems;
  t.aStepElems = static_cast<uint64_t>(kL1) * p_->m;        // A elements per chunk
  t.bStepElems = static_cast<uint64_t>(kL1) * kFractalDim;  // B elements per chunk
  t.paFull.blockCount = static_cast<uint16_t>(t.kfFull);    // A: full chunk GM->L1
  t.paFull.blockLen = static_cast<uint16_t>(t.m1 * kFractalDim);
  t.paFull.srcStride = static_cast<uint16_t>((m1All - t.m1) * kFractalDim);
  t.paFull.dstStride = 0;
  t.pbFull.blockCount = static_cast<uint16_t>(t.nCols / kFractalDim);  // B: full chunk GM->L1
  t.pbFull.blockLen = static_cast<uint16_t>(t.kfFull * kFractalDim);
  t.pbFull.srcStride = static_cast<uint16_t>((kAll / kFractalDim - t.kfFull) * kFractalDim);
  t.pbFull.dstStride = 0;
  t.ldaFull.startIndex = 0;  // A: L1->L0A
  t.ldaFull.repeatTimes = static_cast<uint8_t>(t.m1);
  t.ldaFull.srcStride = 1;
  t.ldaFull.dstGap = static_cast<uint16_t>(t.kfFull - 1);
  t.ldaFull.ifTranspose = false;
  t.ldbFull.startIndex = 0;  // B: L1->L0B (transposed load)
  t.ldbFull.repeatTimes = static_cast<uint8_t>(t.n1);
  t.ldbFull.srcStride = static_cast<uint16_t>(bResident_ ? kfAll_ : t.kfFull);
  t.ldbFull.dstGap = 0;
  t.ldbFull.ifTranspose = true;
  Tile2InitMmadParams(t);

  Tile2FirstChunk(t);

  for (uint32_t ci = 0; ci < t.nChunk; ++ci) {
    Tile2Chunk ck;
    ck.ci = ci;
    ck.isFull = (ci < t.nChunkFull);
    const uint32_t k0 = ci * kL1;
    ck.kLenL1 = (kAll - k0) < kL1 ? (kAll - k0) : kL1;
    ck.kf = (ck.kLenL1 + kFractalDim - 1) / kFractalDim;

    Tile2PrefetchChunk(ck, t);
    const uint32_t p = ci & 1U;
    WaitFlag<HardEvent::MTE2_MTE1>(evIn_[p]);  // L1[p] is ready (this slot only, no full barrier)
    WaitFlag<HardEvent::M_MTE1>(evM_[p]);      // L0[p] was released by the Mmad two steps ago
    Tile2LoadL0(ck, p, t);
    Sync<HardEvent::MTE1_M>();
    Tile2Mmad(ck, c1, t);
    SetFlag<HardEvent::M_MTE1>(evM_[p]);
    Sync<HardEvent::MTE1_MTE2>();  // explicit: L1[buf] order between MTE1 and MTE2
  }
  WaitFlag<HardEvent::M_MTE1>(evM_[0]);
  WaitFlag<HardEvent::M_MTE1>(evM_[1]);

  Tile2PrefetchNextBias(nextTileIdx, hasNext);
  if (p_->biasMode == kBiasModeRow) {  // one extra K=16 Mmad accumulates the row bias into L0C
    IssueBiasMmad(c1, t.m1, t.n1, t.qm);
  }
  Tile2WriteBack(c1, t);
}

// 2x2 tile prologue: full-size bias prefetch, resident-B reload, row-bias aux build, the full-bias
// "I x bias" L0C initialisation and the L0 ping-pong token seed.
__aicore__ inline void InnerBatchMatMulKernel::Tile2Prepare(const Tile2Tile &t, uint64_t biasOff, uint32_t batchIdx,
                                                            uint32_t ni) {
  LocalTensor<float> c1 = c1Buf_.Get<float>();
  // Prefetch the whole-tile full-size bias into L1 (the B' source of the bias Mmad); GM->L1 is
  // much faster than GM->UB on this SoC.
  if (p_->biasMode == kBiasModeFull &&
      !hasPrefetch_) {  // normally prefetched by the previous tile; only the first tile reads here
    const uint32_t m1All = p_->m / kFractalDim;
    const uint64_t offT = biasOff + (static_cast<uint64_t>(t.nLo) / kFractalDim) * m1All * kFractalElems +
                          (static_cast<uint64_t>(t.mLo) / kFractalDim) * kFractalElems;
    PrefetchBiasL1(curBiasSlot_, t.m1, t.n1, offT);  // one read for the whole tile (8KB burst)
  }

  // Resident B: reload when the column changes.
  const uint32_t bKey = batchIdx * p_->nTiles + ni;
  if (bResident_ && bKey != bKey_) {
    LoadBFull(t.offB, t.nLo, t.nCols);
    bKey_ = bKey;
  }

  if (p_->biasMode == kBiasModeRow) {
    if (biasAuxKey_ != bKey) {
      BuildBiasAux(biasOff, t.nLo, t.n1);
      biasAuxKey_ = bKey;
    }
  }

  if (p_->biasMode == kBiasModeFull) {
    // Wait for the bias prefetch (GM->L1), then write bias straight into L0C with the
    // "I x bias" Mmad (first block cleared, see IssueBiasMmadFull). This must finish before the
    // L0 tokens are seeded and needs an explicit Sync<M_MTE1>: otherwise the first LoadL02
    // would overwrite the L0A/L0B the bias Mmad is reading.
    Sync<HardEvent::MTE2_MTE1>();
    IssueBiasMmadFull(c1, t.m1, t.n1, t.qm, t.qn);
    hasPrefetch_ = false;
  }
  SetFlag<HardEvent::M_MTE1>(evM_[0]);  // L0 ping-pong token seed
  SetFlag<HardEvent::M_MTE1>(evM_[1]);
}

// Issues the prefetch of the next K chunk into the other L1 slot. The slot (ci+1)&1 currently holds
// chunk ci-1, whose MTE1 reads finished before the Sync<MTE1_MTE2> at the end of the previous
// iteration, so overwriting is safe.
__aicore__ inline void InnerBatchMatMulKernel::Tile2PrefetchChunk(const Tile2Chunk &ck, const Tile2Tile &t) {
  if (ck.ci + 1 >= t.nChunk) {
    return;
  }
  const uint32_t nx = ck.ci + 1;
  if (nx < t.nChunkFull) {  // the next chunk is also full -> use the hoisted descriptors
    const uint64_t aOff = t.aBase0 + static_cast<uint64_t>(nx) * t.aStepElems;
    const uint64_t bOff = t.bBase0 + static_cast<uint64_t>(nx) * t.bStepElems;
    DataCopy(L1A(nx), aGm_[aOff], t.paFull);
    if (!bResident_) {
      DataCopy(L1B(nx), bGm_[bOff], t.pbFull);
    }
  } else {
    LoadChunkA(nx, t.offA, t.mLo, t.m1);
    if (!bResident_) {
      LoadChunkB(nx, t.offB, t.nLo, t.nCols);
    }
  }
  SetFlag<HardEvent::MTE2_MTE1>(evIn_[nx & 1U]);
}

// Hoisted Mmad parameters of a full chunk and the L0C/L0A offsets of row quadrant 1.
__aicore__ inline void InnerBatchMatMulKernel::Tile2InitMmadParams(Tile2Tile &t) {
  t.mpC.m = static_cast<uint16_t>((t.m1 < t.m1sub ? t.m1 : t.m1sub) * kFractalDim);
  t.mpC.n = static_cast<uint16_t>(t.n1 * kFractalDim);
  t.mpC.k = static_cast<uint16_t>(kL1_);
  t.mpC.isBias = false;
  t.mpE.m = static_cast<uint16_t>((t.m1 - t.m1sub) * kFractalDim);
  t.mpE.n = static_cast<uint16_t>(t.n1 * kFractalDim);
  t.mpE.k = static_cast<uint16_t>(kL1_);
  t.mpE.isBias = false;
  t.cq1 = tileM_ * p_->baseN;
  t.aq1 = t.m1sub * t.kfFull * kFractalElems;
}

// Issues the load of the first K chunk and seeds the "first chunk has landed" token (paired with
// the per-slot WaitFlag in the loop). The resident-B LoadBFull was also issued before this point and
// uses the same MTE2 queue, so waiting on evIn_[0] already guarantees the slot is readable.
__aicore__ inline void InnerBatchMatMulKernel::Tile2FirstChunk(const Tile2Tile &t) {
  if (t.nChunkFull > 0U) {  // ci = 0 is always full when nChunk > 1; nChunk == 1 takes the generic tail path
    DataCopy(L1A(0), aGm_[t.aBase0], t.paFull);
    if (!bResident_) {
      DataCopy(L1B(0), bGm_[t.bBase0], t.pbFull);
    }
  } else {
    LoadChunkA(0, t.offA, t.mLo, t.m1);
    if (!bResident_) {
      LoadChunkB(0, t.offB, t.nLo, t.nCols);
    }
  }
  SetFlag<HardEvent::MTE2_MTE1>(evIn_[0]);
}

// Loads the current K chunk from L1 into L0: the hoisted descriptors for full chunks, the generic
// LoadL02 helper for the tail chunk.
__aicore__ inline void InnerBatchMatMulKernel::Tile2LoadL0(const Tile2Chunk &ck, uint32_t p, const Tile2Tile &t) {
  if (!ck.isFull) {
    LoadL02(ck.ci, p, t.m1, t.n1, ck.kf);
    return;
  }
  LocalTensor<half> a2 = L0A(p);
  LocalTensor<half> b2 = L0B(p);
  LocalTensor<half> aL1 = L1A(p);
  if (bResident_) {
    LocalTensor<half> bL1 = b1Buf_.Get<half>();
    for (uint32_t kk = 0; kk < t.kfFull; ++kk) {
      LoadData(a2[kk * kFractalElems], aL1[kk * t.m1 * kFractalElems], t.ldaFull);
    }
    for (uint32_t kk = 0; kk < t.kfFull; ++kk) {
      LoadData(b2[kk * t.n1 * kFractalElems], bL1[(ck.ci * t.kfFull + kk) * kFractalElems], t.ldbFull);
    }
  } else {
    LocalTensor<half> bL1 = L1B(p);
    for (uint32_t kk = 0; kk < t.kfFull; ++kk) {
      LoadData(a2[kk * kFractalElems], aL1[kk * t.m1 * kFractalElems], t.ldaFull);
    }
    for (uint32_t kk = 0; kk < t.kfFull; ++kk) {
      LoadData(b2[kk * t.n1 * kFractalElems], bL1[kk * kFractalElems], t.ldbFull);
    }
  }
}

// Accumulates the current K chunk into the four L0C accumulators (one or two Mmad calls per chunk).
__aicore__ inline void InnerBatchMatMulKernel::Tile2Mmad(const Tile2Chunk &ck, LocalTensor<float> &c1, Tile2Tile &t) {
  LocalTensor<half> a2 = L0A(ck.ci);
  LocalTensor<half> b2 = L0B(ck.ci);
  if (ck.isFull) {  // full chunk: hoisted Mmad parameters (only init/unitFlag change per chunk)
    t.mpC.cmatrixInitVal = (ck.ci == 0 && p_->biasMode != kBiasModeFull);
    t.mpC.unitFlag = (ck.ci + 1U < t.nChunk) ? 1U : 0U;
    Mmad(c1[0], a2, b2, t.mpC);
    if (t.m1 > t.m1sub) {
      t.mpE.cmatrixInitVal = t.mpC.cmatrixInitVal;
      t.mpE.unitFlag = t.mpC.unitFlag;
      Mmad(c1[t.cq1], a2[t.aq1], b2, t.mpE);
    }
    return;
  }
  for (uint32_t qi = 0; qi < t.qm; ++qi) {
    const uint32_t mq = (qi == 0) ? ((t.m1 < t.m1sub) ? t.m1 : t.m1sub) : (t.m1 - t.m1sub);
    MmadParams mp;
    mp.m = static_cast<uint16_t>(mq * kFractalDim);
    mp.n = static_cast<uint16_t>(t.n1 * kFractalDim);
    mp.k = static_cast<uint16_t>(ck.kLenL1);
    mp.cmatrixInitVal = (ck.ci == 0 && p_->biasMode != kBiasModeFull);
    mp.isBias = false;
    mp.unitFlag = (ck.ci + 1U < t.nChunk) ? 1U : 0U;
    Mmad(c1[qi * tileM_ * p_->baseN], a2[qi * t.m1sub * ck.kf * kFractalElems], b2, mp);
  }
}

// Full-size bias: prefetch the next tile's bias into the other slot so that it overlaps this tile's
// write-back and the next tile's K loop.
__aicore__ inline void InnerBatchMatMulKernel::Tile2PrefetchNextBias(uint32_t nextTileIdx, bool hasNext) {
  if (p_->biasMode != kBiasModeFull || !hasNext) {
    return;
  }
  const uint32_t mTiles = p_->mTiles;
  const uint32_t nTiles = p_->nTiles;
  const uint32_t nmi = nextTileIdx % mTiles;
  const uint32_t nni = (nextTileIdx / mTiles) % nTiles;
  const uint32_t nBatch = nextTileIdx / (mTiles * nTiles);
  const uint32_t nmLo = nmi * tileM_;
  const uint32_t nnLo = nni * tileN_;
  const uint32_t nmRows = (p_->m - nmLo) < tileM_ ? (p_->m - nmLo) : tileM_;
  const uint32_t nnCols = (p_->n - nnLo) < tileN_ ? (p_->n - nnLo) : tileN_;
  const uint32_t nm1 = (nmRows + kFractalDim - 1) / kFractalDim;
  const uint32_t nn1 = (nnCols + kFractalDim - 1) / kFractalDim;
  const uint32_t m1All = p_->m / kFractalDim;
  const uint64_t nOff = ComputeBiasOffset(nBatch) +
                        (static_cast<uint64_t>(nnLo) / kFractalDim) * m1All * kFractalElems +
                        (static_cast<uint64_t>(nmLo) / kFractalDim) * kFractalElems;
  const uint32_t nSlot = curBiasSlot_ ^ 1U;
  PrefetchBiasL1(nSlot, nm1, nn1, nOff);
  curBiasSlot_ = nSlot;
  hasPrefetch_ = true;
}

// Write-back of the four quadrants (L0C->UB->Cast->NZ2ND->GM). There is no L0C ping-pong, so the
// write-back completes at the end of the tile.
__aicore__ inline void InnerBatchMatMulKernel::Tile2WriteBack(LocalTensor<float> &c1, const Tile2Tile &t) {
  Sync<HardEvent::M_V>();
  const uint32_t nFrac0 = (t.n1 < t.n1sub) ? t.n1 : t.n1sub;
  const uint32_t nFrac1 = (t.n1 > t.n1sub) ? (t.n1 - t.n1sub) : 0U;
  for (uint32_t qi = 0; qi < t.qm; ++qi) {
    const uint32_t mq = (qi == 0) ? ((t.m1 < t.m1sub) ? t.m1 : t.m1sub) : (t.m1 - t.m1sub);
    for (uint32_t qj = 0; qj < t.qn; ++qj) {
      QuadDesc q;
      q.m1 = mq;
      q.n1 = (qj == 0) ? nFrac0 : nFrac1;
      // In the [n][m] L0C layout every n-fractal only spans mq m-fractals, so the column-quadrant
      // offset is qj*n1sub*mq rather than a fixed baseM*baseN: for an M tail tile (mq < baseM/16)
      // the second column quadrant would otherwise read from the wrong L0C position.
      q.l0cOff = qi * tileM_ * p_->baseN + qj * t.n1sub * mq * kFractalElems;
      q.mRowOff = t.mLo + qi * p_->baseM;  // global row offset
      q.nColOff = t.nLo + qj * p_->baseN;  // global column offset
      q.cOff = t.offC;
      WriteBackQuad(q);
    }
  }
}

}  // namespace InnerBatchMatMul

#endif  // INNER_BATCH_MATMUL_KERNEL_TILE2_H_
