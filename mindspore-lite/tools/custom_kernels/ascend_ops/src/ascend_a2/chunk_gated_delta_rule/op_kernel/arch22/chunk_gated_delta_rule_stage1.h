/** * copy from https://gitcode.com/cann/ops-transformer/tree/master/attention/chunk_gated_delta_rule
 *
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

/*!
 * \file chunk_gated_delta_rule_stage1.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_STAGE1_H
#define CHUNK_GATED_DELTA_RULE_STAGE1_H

#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_utils.h"
#include "../chunk_gated_delta_rule_tiling_data.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;

template <typename lowType>
using aT1 = MatmulType<TPosition::GM, CubeFormat::ND, lowType>;
template <typename lowType>
using bT1 = MatmulType<TPosition::GM, CubeFormat::ND, lowType>;
template <typename lowType>
using cT1 = MatmulType<TPosition::GM, CubeFormat::ND, lowType>;
template <typename lowType>
using StageOneMT = matmul::MatmulImpl<aT1<lowType>, bT1<lowType>, cT1<lowType>>;

constexpr uint64_t UB_REST_BYTES = 140 * 1024;  // 140KB
constexpr uint64_t INVERSE_SHAPE = 32;          // diagonal block side length
constexpr uint64_t INVERSE_COUNT = 5;           // workspace required for matrix inverse
constexpr uint32_t ALIGN_SIZE = 16;
constexpr uint32_t MAX_PARALLEL_NUM = 6;
constexpr uint32_t HALF_TWO = 2;

// Matmul shape parameters struct
struct MatmulShapeParams {
  uint64_t m;   // original M dimension
  uint64_t n;   // original N dimension
  uint64_t k;   // original K dimension
  uint64_t sm;  // single-compute M dimension
  uint64_t sn;  // single-compute N dimension
  uint64_t sk;  // single-compute K dimension
};

template <typename lowType>
struct GDRStageOneInitParams {
  // input
  GlobalTensor<lowType> query;  // (T, Nk, Dk)
  GlobalTensor<lowType> key;    // (T, Nk, Dk)
  GlobalTensor<lowType> value;  // (T, Nv, Dv)
  GlobalTensor<lowType> beta;   // (T, Nv)
  GlobalTensor<float> g;        // (T, Nv)
  // output
  GlobalTensor<float> gCumExp;      // (Nv, cg_len)
  GlobalTensor<lowType> kCumdecay;  // (Nv, cg_len, Dk)
  GlobalTensor<lowType> vInner;     // (Nv, cg_len, Dv)
  GlobalTensor<lowType> qPrime;     // (Nv, cg_len, Dk)
  GlobalTensor<lowType> kG;         // (Nv, cg_len, Dk)
  GlobalTensor<lowType> qK;         // (Nv, cg_len, C)
  // other
  GM_ADDR ws;
  GlobalTensor<float> stageOneMask;  // (Nv, cg_len, C)
  ChunkGroup cg;
  bool gOptional;
};

template <typename lowType>
class Stage1 {
 public:
  __aicore__ inline Stage1(StageOneMT<lowType> &mm) : mm(mm) {}
  __aicore__ inline void SetGlobalTensors(const GDRStageOneInitParams<lowType> &initParams) {
    queryGm_ = initParams.query;
    keyGm_ = initParams.key;
    valueGm_ = initParams.value;
    betaGm_ = initParams.beta;

    outGCumExpGm_ = initParams.gCumExp;
    outKCumdecayGm_ = initParams.kCumdecay;
    outVInnerGm_ = initParams.vInner;
    outQPrimeGm_ = initParams.qPrime;
    outKgBaseGm_ = initParams.kG;
    outQkGm_ = initParams.qK;
    stageOneMask_ = initParams.stageOneMask;

    if (gOptional_) {
      gGm_ = initParams.g;
    }
    // ckOffset_
    uint64_t workSpaceOffset = 0;
    uint64_t curWS = coreIdx_ * paraNum_ * ckOffset_ * sizeof(lowType);
    gBKWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.ws + workSpaceOffset + curWS));
    workSpaceOffset += coreNum_ * paraNum_ * ckOffset_ * sizeof(lowType);

    queryConGm_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.ws + workSpaceOffset + curWS));
    workSpaceOffset += coreNum_ * paraNum_ * ckOffset_ * sizeof(lowType);

    keyConGm_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.ws + workSpaceOffset + curWS));
    workSpaceOffset += coreNum_ * paraNum_ * ckOffset_ * sizeof(lowType);

    // ccOffset_
    curWS = coreIdx_ * paraNum_ * ccOffset_ * sizeof(lowType);
    kkWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.ws + workSpaceOffset + curWS));
    workSpaceOffset += coreNum_ * paraNum_ * ccOffset_ * sizeof(lowType);

    attnWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.ws + workSpaceOffset + curWS));
    workSpaceOffset += coreNum_ * paraNum_ * ccOffset_ * sizeof(lowType);

    // cvOffset_
    curWS = coreIdx_ * paraNum_ * cvOffset_ * sizeof(lowType);
    vBetaWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ lowType *>(initParams.ws + workSpaceOffset + curWS));
  }

  __aicore__ inline void InitLocalBuffers() {
    maxLen_ = AscendC::Std::max(AscendC::Std::max(dvAligned_ / HALF_TWO, dkAligned_ / HALF_TWO), chunkSize_);
    pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, chunkSize_ * maxLen_ * sizeof(float));
    pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, chunkSize_ * maxLen_ * sizeof(float));
    if (gOptional_) {
      pipe_->InitBuffer(gOutQueue_, BUFFER_NUM_ONE, chunkSize_ * sizeof(float));
    }

    pipe_->InitBuffer(tmpBuff_, UB_REST_BYTES);
    uint32_t buffOffset = 0;
    betaUbBfloat16_ = tmpBuff_.GetWithOffset<lowType>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
    buffOffset += halfChunkSize_ * sizeof(lowType);

    gCumUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_), buffOffset);
    buffOffset += chunkSize_ * sizeof(float);

    gBUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
    buffOffset += halfChunkSize_ * sizeof(float);

    gEndBroadUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
    buffOffset += halfChunkSize_ * sizeof(float);

    betaUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_ * paraNum_), buffOffset);
    buffOffset += halfChunkSize_ * sizeof(float) * paraNum_;

    uint64_t tmpBufferLen = chunkSize_ * maxLen_ * paraNum_;
    gBroadUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
    gammaUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
    valueUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
    qUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
    buffOffset += tmpBufferLen * sizeof(float);

    tmpBufferLen = chunkSize_ * maxLen_;
    gTransBroadUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
    attnUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
    gCumExpBroadUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
    buffOffset += tmpBufferLen * sizeof(float);

    tmpBufferLen = halfChunkSize_ * dkAligned_;
    qUbFloatCon_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
    kUbFloatCon_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
    buffOffset += tmpBufferLen * sizeof(float);

    inputGatherBuffer_ = tmpBuff_.GetWithOffset<uint32_t>(static_cast<uint32_t>(chunkSize_), buffOffset);
    buffOffset += chunkSize_ * sizeof(uint32_t);

    gCumSumUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * paraNum_), buffOffset);
    buffOffset += chunkSize_ * sizeof(float) * paraNum_;

    gCumExpUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * paraNum_), buffOffset);
    buffOffset += chunkSize_ * sizeof(float) * paraNum_;

    tmpBufferLen = halfChunkSize_ * halfChunkSize_ * INVERSE_COUNT;
    inverseBuffer_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
    qPrimeUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
    vBetaUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
    gBKUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
    kgUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
    kkLocal_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
    buffOffset += tmpBufferLen * sizeof(float);

    inverseGatherBuffer_ = tmpBuff_.GetWithOffset<uint32_t>(static_cast<uint32_t>(INVERSE_SHAPE), buffOffset);
    buffOffset += INVERSE_SHAPE * sizeof(uint32_t);

    inverseRes_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * halfChunkSize_), buffOffset);
  }

  __aicore__ inline void InitGatherBuffer() {
    for (uint32_t i = 0; i < chunkSize_; ++i) {
      inputGatherBuffer_.SetValue(i, i * BLOCK_SIZE);
    }
    for (uint32_t i = 0; i < INVERSE_SHAPE; ++i) {
      inverseGatherBuffer_.SetValue<uint32_t>(i, (i * chunkSize_) * sizeof(float));
    }
    int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventID);
    WaitFlag<HardEvent::S_V>(eventID);
  }

  __aicore__ inline void Init(const GDRStageOneInitParams<lowType> &initParams, TPipe *pipe,
                              const ChunkGatedDeltaRuleTilingData *tilingData) {
    pipe_ = pipe;
    tiling_ = tilingData;
    nk_ = tiling_->nk;
    nv_ = tiling_->nv;
    chunkSize_ = tiling_->chunkSize;
    dk_ = tiling_->dk;
    dv_ = tiling_->dv;
    paraNum_ = tiling_->stageOneParaNum;
    dkAligned_ = (dk_ + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
    dvAligned_ = (dv_ + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
    scale_ = tiling_->scale;
    coreNum_ = tiling_->aiCoreNum;
    cg_ = initParams.cg;
    gOptional_ = initParams.gOptional;
    vRowStride_ = nv_ * dv_;
    numChunk_ = (cg_.length + chunkSize_ - 1) / chunkSize_;
    subBlockIdx_ = GetSubBlockIdx();
    halfChunkSize_ = chunkSize_ / TASK_RATIO;
    subOffset_ = subBlockIdx_ * halfChunkSize_;
    coreIdx_ = GetBlockIdx();
    ccOffset_ = chunkSize_ * chunkSize_;
    ckOffset_ = chunkSize_ * dk_;
    cvOffset_ = chunkSize_ * dv_;
    if ASCEND_IS_AIV {
      coreIdx_ /= TASK_RATIO;
      InitLocalBuffers();
      InitGatherBuffer();
    }
    SetGlobalTensors(initParams);
  }

  __aicore__ inline void Process() {
    uint32_t totalChunk = nv_ * numChunk_;
    uint32_t tailChunkNum = totalChunk / coreNum_;   // chunks per tail core
    uint32_t formerChunkNum = tailChunkNum + 1;      // chunks per former core
    uint32_t formerCoreNum = totalChunk % coreNum_;  // number of former cores
    uint32_t start;
    uint32_t end;
    if (coreIdx_ < formerCoreNum) {
      start = coreIdx_ * formerChunkNum;
      end = start + formerChunkNum;
    } else {
      start = formerCoreNum * formerChunkNum + (coreIdx_ - formerCoreNum) * tailChunkNum;
      end = start + tailChunkNum;
    }

    for (uint32_t taskId = start; taskId < end; taskId += paraNum_) {
      uint32_t curParaNum = paraNum_ < end - taskId ? paraNum_ : end - taskId;
      // Obtain valid length of each chunk
      for (uint32_t i = 0; i < curParaNum; ++i) {
        uint32_t curTaskId = taskId + i;
        uint64_t curNId = curTaskId % nv_;
        uint64_t curCgId = curTaskId / nv_;
        SetChunkOffset(i, curNId, curCgId);
      }
      ProcessParaChunk(curParaNum);
    }
  }

 private:
  // ----------------------------------------------------------
  // SetChunkOffset
  //   curNId  : head index (Nv dimension)
  //   curCgId : chunk index within CG (0 ~ CG_CHUNKS-1)
  // ----------------------------------------------------------
  __aicore__ inline void SetChunkOffset(uint64_t id, uint64_t curNId, uint64_t curCgId) {
    validLenBatch_[id] = chunkSize_;
    // tail chunk handling
    if (curCgId == numChunk_ - 1 && cg_.length % chunkSize_ != 0) {
      validLenBatch_[id] = cg_.length % chunkSize_;
    }
    if (validLenBatch_[id] < halfChunkSize_) {
      subValidLenBatch_[id] = (subBlockIdx_ == 0) ? validLenBatch_[id] : 0;
    } else {
      subValidLenBatch_[id] = (subBlockIdx_ == 0) ? halfChunkSize_ : validLenBatch_[id] - halfChunkSize_;
    }
    // offset
    uint64_t cgLenPad = (cg_.length + chunkSize_ - 1) / chunkSize_ * chunkSize_;
    chunkRowBase_[id] = curNId * cgLenPad + curCgId * chunkSize_;

    chunkStartRowBatch_[id] = cg_.startPos + curCgId * chunkSize_;
    nIdBatch_[id] = curNId;
    bgOffsetBatch_[id] = chunkStartRowBatch_[id] * nv_ + curNId;
  }

  __aicore__ inline void ProcessParaChunk(int32_t curParaNum) {
    if ASCEND_IS_AIC {
      ParaChunkAIC(curParaNum);
    }
    if ASCEND_IS_AIV {
      ParaChunkAIV(curParaNum);
    }
  }

  __aicore__ inline void ParaChunkAIC(int32_t curParaNum) {
    AscendC::CrossCoreWaitFlag(0x9);  // sync0
    // key @ key.transpose(-1,-2)
    for (uint32_t i = 0; i < curParaNum; ++i) {
      AICProcess(keyConGm_[i * ckOffset_], keyConGm_[i * ckOffset_], kkWsGm_[i * ccOffset_],
                 {chunkSize_, chunkSize_, dk_, chunkSize_, chunkSize_, dk_}, true);
    }
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);  // sync1

    // query @ key.transpose(-1,-2)   stage1 out
    for (uint32_t i = 0; i < curParaNum; ++i) {
      AICProcess(queryConGm_[i * ckOffset_], keyConGm_[i * ckOffset_], outQkGm_[chunkRowBase_[i] * chunkSize_],
                 {validLenBatch_[i], validLenBatch_[i], dk_, validLenBatch_[i], validLenBatch_[i], dk_}, true);
    }
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0xA);  // sync5
    AscendC::CrossCoreWaitFlag(0x7);                // sync2

    // Inverse of the lower-left matrix
    for (uint32_t i = 0; i < curParaNum; ++i) {
      AttnInverseMMCompute(i * ccOffset_);
    }
    AscendC::CrossCoreWaitFlag(0x6);  // sync3

    // attn @ k_cumdecay
    for (uint32_t i = 0; i < curParaNum; ++i) {
      AICProcess(attnWsGm_[i * ccOffset_], gBKWsGm_[i * ckOffset_], outKCumdecayGm_[chunkRowBase_[i] * dk_],
                 {chunkSize_, dk_, chunkSize_, chunkSize_, dk_, chunkSize_});
    }
    AscendC::CrossCoreWaitFlag(0x5);  // sync4

    // attn @ v_beta    stage1 out
    for (uint32_t i = 0; i < curParaNum; ++i) {
      AICProcess(attnWsGm_[i * ccOffset_], vBetaWsGm_[i * cvOffset_], outVInnerGm_[chunkRowBase_[i] * dv_],
                 {chunkSize_, dv_, chunkSize_, chunkSize_, dv_, chunkSize_});
    }
  }

  __aicore__ inline void ParaChunkAIV(int32_t curParaNum) {
    // Obtain contiguous QK
    for (uint32_t i = 0; i < curParaNum; ++i) {
      uint64_t subRow = chunkStartRowBatch_[i] + subOffset_;
      uint64_t qk_base = subRow * nk_ * dk_ + nIdBatch_[i] * nk_ / nv_ * dk_;
      uint64_t wsOffset_ = i * ckOffset_ + subOffset_ * dk_;
      outKgGm_ = outKgBaseGm_[chunkRowBase_[i] * dk_];
      QKPreProcess(queryGm_[qk_base], queryConGm_[wsOffset_], outKgGm_, subValidLenBatch_[i]);
      QKPreProcess(keyGm_[qk_base], keyConGm_[wsOffset_], outKgGm_, subValidLenBatch_[i], true);
    }
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9);  // sync0
    if (gOptional_) {
      for (uint32_t i = 0; i < curParaNum; ++i) {
        // g_cum_exp = g.cumsum(dim=-1).exp()
        GCumExpCompute(gGm_[bgOffsetBatch_[i]], outGCumExpGm_[chunkRowBase_[i]], gCumSumUbFloat_[i * chunkSize_],
                       gCumExpUbFloat_[i * chunkSize_], validLenBatch_[i]);
        // attn_1 = (g_cum_exp[:None] / g_cum_exp[None,:]) * mask
        uint64_t gUbOffset = i * chunkSize_ * maxLen_;
        GammaCompute(gBroadUbFloat_[gUbOffset], gammaUbFloat_[gUbOffset], gCumSumUbFloat_[i * chunkSize_]);
      }
    }

    for (uint32_t i = 0; i < curParaNum; ++i) {
      uint64_t betaUbOffset = i * halfChunkSize_;
      BetaCopyInWithStride(betaGm_[bgOffsetBatch_[i]], betaUbFloat_[betaUbOffset], subValidLenBatch_[i]);
    }
    AscendC::CrossCoreWaitFlag(0x8);  // sync1

    for (uint32_t i = 0; i < curParaNum; ++i) {
      // attn_1 = kkt * attn_1
      KKBetaCompute(kkWsGm_[i * ccOffset_], betaUbFloat_[i * halfChunkSize_]);
      // Inverse of the attn_1 diagonal block; diagonal block shape is INVERSE_SHAPE=32
      InverseCompute(attnWsGm_[i * ccOffset_], gammaUbFloat_[i * chunkSize_ * maxLen_]);
    }
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x7);  // sync2

    for (uint32_t i = 0; i < curParaNum; ++i) {
      // kg = key * (g_cum_exp[-1, None] / g_cum_exp)[..., None]
      // k_cumdecay = -1.0 * k * beta * g_cum_exp
      outKgGm_ = outKgBaseGm_[chunkRowBase_[i] * dk_];
      uint64_t wsOffset_ = i * ckOffset_ + subOffset_ * dk_;
      GBKCompute(gBKWsGm_[i * ckOffset_], outKgGm_, betaUbFloat_[i * halfChunkSize_], gCumSumUbFloat_[i * chunkSize_],
                 gCumExpUbFloat_[i * chunkSize_], keyConGm_[wsOffset_]);
    }
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x6);  // sync3

    for (uint32_t i = 0; i < curParaNum; ++i) {
      // v_beta = value * beta.unsqueeze(-1)  # (C, Dv)
      uint64_t betaUbOffset = i * halfChunkSize_;
      uint64_t vOffset = chunkStartRowBatch_[i] * vRowStride_ + nIdBatch_[i] * dv_;
      uint64_t valueUbOffset = i * chunkSize_ * maxLen_;
      VBetaCompute(valueGm_[vOffset], vBetaWsGm_[i * cvOffset_], betaUbFloat_[betaUbOffset],
                   valueUbFloat_[valueUbOffset], subValidLenBatch_[i]);
    }
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);  // sync4

    for (uint32_t i = 0; i < curParaNum; ++i) {
      // q_prime = query * scale_ * g_cum_exp[:, None]  # (C, Dk)
      uint64_t wsOffset_ = i * ckOffset_ + subOffset_ * dk_;
      QPrimeCompute(outQPrimeGm_[chunkRowBase_[i] * dk_], gCumSumUbFloat_[i * chunkSize_],
                    gCumExpUbFloat_[i * chunkSize_], queryConGm_[wsOffset_]);
    }
    AscendC::CrossCoreWaitFlag(0xA);  // sync5
  }
  __aicore__ inline void QKPreProcess(const GlobalTensor<lowType> &srcGm, const GlobalTensor<lowType> &dstGm,
                                      const GlobalTensor<lowType> &outKgGm, uint32_t subValidRows,
                                      bool kgFlag = false) {
    // copyOut
    auto tmpTensor = outQueue_.AllocTensor<lowType>();
    if (subValidRows > 0) {
      // copyIn
      DataCopyInBf16WithStride(subValidRows, dk_, srcGm, nk_ * dk_);
      LocalTensor<lowType> bf16Tensor = inQueue_.DeQue<lowType>();
      DataCopy(tmpTensor, bf16Tensor, subValidRows * dkAligned_);
      inQueue_.FreeTensor(bf16Tensor);
    }

    if (subValidRows < halfChunkSize_) {
      Duplicate(tmpTensor[subValidRows * dkAligned_], static_cast<lowType>(0.0f),
                (halfChunkSize_ - subValidRows) * dkAligned_);
      PipeBarrier<PIPE_V>();
    }
    outQueue_.EnQue(tmpTensor);
    tmpTensor = outQueue_.DeQue<lowType>();

    uint32_t srcStride = (dkAligned_ - dk_) * sizeof(lowType) / BLOCK_SIZE;
    DataCopyExtParams outParams{static_cast<uint16_t>(halfChunkSize_), static_cast<uint32_t>(dk_ * sizeof(lowType)),
                                srcStride, 0, 0};
    DataCopyPad(dstGm, tmpTensor, outParams);
    if (!gOptional_ && kgFlag) {
      DataCopyPad(outKgGm[subOffset_ * dk_], tmpTensor, outParams);
    }
    outQueue_.FreeTensor(tmpTensor);
  }

  __aicore__ inline void GCumExpCompute(const GlobalTensor<float> src, const GlobalTensor<float> dst,
                                        LocalTensor<float> gCumSumUbFloat, LocalTensor<float> gCumExpUbFloat,
                                        uint32_t validLen) {
    // Copy g
    GCopyInWithStride(src, validLen);
    // CumSum compute
    uint32_t outer = 1;
    uint32_t inner = chunkSize_;
    CumSumInfo cumSumInfo{outer, inner};
    CumSum<float>(gCumSumUbFloat, gCumUbFloat_, gCumUbFloat_, cumSumInfo);
    PipeBarrier<PIPE_V>();
    // Exp compute
    Exp<float, 0, true>(gCumExpUbFloat, gCumSumUbFloat, chunkSize_);
    PipeBarrier<PIPE_V>();

    if (subBlockIdx_ == 0) {
      auto tmpOut = gOutQueue_.AllocTensor<float>();
      DataCopy(tmpOut, gCumSumUbFloat, chunkSize_);
      gOutQueue_.EnQue(tmpOut);
      tmpOut = gOutQueue_.DeQue<float>();
      DataCopyExtParams params{static_cast<uint16_t>(1), static_cast<uint32_t>(validLen * sizeof(float)), 0, 0, 0};
      DataCopyPad(dst, tmpOut, params);
      gOutQueue_.FreeTensor(tmpOut);
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void GammaCompute(const LocalTensor<float> gBroadUbFloat, LocalTensor<float> gammaUbFloat,
                                      LocalTensor<float> gCumSumUbFloat) {
    // BroadCast
    uint32_t divShape[2] = {chunkSize_, chunkSize_};
    uint32_t gShape[2] = {chunkSize_, 1};
    uint32_t gTransShape[2] = {1, chunkSize_};
    Broadcast<float, BROADCAST_AXIS, 1>(gBroadUbFloat, gCumSumUbFloat, divShape, gShape);
    Broadcast<float, BROADCAST_AXIS, 0>(gTransBroadUbFloat_, gCumSumUbFloat, divShape, gTransShape);
    PipeBarrier<PIPE_V>();
    // div
    Sub(gammaUbFloat, gBroadUbFloat, gTransBroadUbFloat_, ccOffset_);
    PipeBarrier<PIPE_V>();
    // mask  1
    DataCopyInFp32(ccOffset_, stageOneMask_[GetBlockIdx() * ccOffset_]);
    auto maskLocal = inQueue_.DeQue<float>();
    Mul(gammaUbFloat, gammaUbFloat, maskLocal, ccOffset_);
    PipeBarrier<PIPE_V>();
    // exp
    Exp<float, 0, true>(gammaUbFloat, gammaUbFloat, ccOffset_);
    PipeBarrier<PIPE_V>();
    // mask  2
    Mul(gammaUbFloat, gammaUbFloat, maskLocal, ccOffset_);
    PipeBarrier<PIPE_V>();

    inQueue_.FreeTensor(maskLocal);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void KKBetaCompute(const GlobalTensor<lowType> src, LocalTensor<float> betaUbFloat) {
    // copy value
    uint32_t kkLength = chunkSize_ * halfChunkSize_;
    uint64_t kkBeginOffset = subOffset_ * chunkSize_;
    DataCopyInBf16(kkLength, src[kkBeginOffset]);
    auto bf16Local = inQueue_.DeQue<lowType>();
    Cast(kkLocal_, bf16Local, RoundMode::CAST_NONE, kkLength);
    PipeBarrier<PIPE_V>();
    inQueue_.FreeTensor(bf16Local);

    uint32_t betaShape[2] = {halfChunkSize_, 1};
    uint32_t kkShape[2] = {halfChunkSize_, chunkSize_};
    Broadcast<float, BROADCAST_AXIS, 1>(attnUbFloat_, betaUbFloat, kkShape, betaShape);
    PipeBarrier<PIPE_V>();
    Mul(attnUbFloat_, kkLocal_, attnUbFloat_, chunkSize_ * halfChunkSize_);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void InverseCompute(const GlobalTensor<lowType> dst, LocalTensor<float> gammaUbFloat) {
    uint64_t curVecLen = chunkSize_ * halfChunkSize_;
    if (gOptional_) {
      Mul(attnUbFloat_, attnUbFloat_, gammaUbFloat[subOffset_ * chunkSize_], curVecLen);
    } else {
      DataCopyInFp32(curVecLen, stageOneMask_[GetBlockIdx() * ccOffset_ + subOffset_ * chunkSize_]);
      auto maskLocal = inQueue_.DeQue<float>();
      Mul(attnUbFloat_, attnUbFloat_, maskLocal, curVecLen);
      inQueue_.FreeTensor(maskLocal);
    }
    PipeBarrier<PIPE_V>();

    Muls(inverseRes_, attnUbFloat_, static_cast<float>(-1.0), curVecLen);
    PipeBarrier<PIPE_V>();
    int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventID);
    WaitFlag<HardEvent::V_S>(eventID);

    InverseAIV(subOffset_, INVERSE_SHAPE);
    auto tmp = outQueue_.AllocTensor<lowType>();
    Cast(tmp, inverseRes_, RoundMode::CAST_RINT, curVecLen);
    outQueue_.EnQue(tmp);
    DataCopyOutBf16(halfChunkSize_, chunkSize_, chunkSize_, dst[subOffset_ * chunkSize_]);
  }

  __aicore__ inline void InverseAIV(uint64_t offset, uint32_t inverseVecLen) {
    PipeBarrier<PIPE_V>();
    uint64_t inverseBufferOffset = 0;
    auto row = inverseBuffer_[inverseBufferOffset];
    inverseBufferOffset += inverseVecLen * inverseVecLen;
    auto col = inverseBuffer_[inverseBufferOffset];
    inverseBufferOffset += inverseVecLen * inverseVecLen + inverseVecLen;
    auto yLocal = inverseBuffer_[inverseBufferOffset];
    inverseBufferOffset += inverseVecLen * inverseVecLen;
    auto ei = inverseBuffer_[inverseBufferOffset];

    Duplicate(ei, static_cast<float>(0.0), inverseVecLen);
    Duplicate(yLocal, static_cast<float>(0.0), inverseVecLen * inverseVecLen);  // zero out yLocal
    inverseRes_.SetValue(offset, static_cast<float>(1.0));

    uint32_t srcShape[2] = {1, inverseVecLen};
    int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventID);
    WaitFlag<HardEvent::S_V>(eventID);
    for (int i = 1; i < inverseVecLen; ++i) {
      uint32_t curI = i - 1;
      uint32_t validRows = inverseVecLen - i;
      Gather(col, attnUbFloat_[offset + i * chunkSize_ + curI], inverseGatherBuffer_, (uint32_t)0, validRows);
      PipeBarrier<PIPE_V>();
      uint32_t dstShape[2] = {validRows, inverseVecLen};
      uint32_t colSrcShape[2] = {validRows, 1};
      Broadcast<float, BROADCAST_AXIS, 1>(col[inverseVecLen], col, dstShape, colSrcShape);
      Broadcast<float, BROADCAST_AXIS, 0>(row, inverseRes_[offset + curI * chunkSize_], dstShape, srcShape);
      PipeBarrier<PIPE_V>();
      MulAddDst(yLocal[i * inverseVecLen], col[inverseVecLen], row, inverseVecLen * validRows);
      PipeBarrier<PIPE_V>();
      int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::V_S));
      SetFlag<HardEvent::V_S>(eventID);
      WaitFlag<HardEvent::V_S>(eventID);
      ei.SetValue(i - 1, static_cast<float>(0.0));
      ei.SetValue(i, static_cast<float>(1.0));
      eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
      SetFlag<HardEvent::S_V>(eventID);
      WaitFlag<HardEvent::S_V>(eventID);
      // xi = (I - SUM) / Lii = I - SUM
      Sub(inverseRes_[offset + i * chunkSize_], ei, yLocal[i * inverseVecLen], inverseVecLen);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline void GBKCompute(const GlobalTensor<lowType> gBKWsGm, const GlobalTensor<lowType> outKgGm,
                                    LocalTensor<float> betaUbFloat, LocalTensor<float> gCumSumUbFloat,
                                    LocalTensor<float> gCumExpUbFloat, const GlobalTensor<lowType> keyContinousGm) {
    if (gOptional_) {
      // tmp = -1.0 * beta * exp(g_cum)
      Mul(gBUbFloat_, betaUbFloat, gCumExpUbFloat[subOffset_], halfChunkSize_);
      PipeBarrier<PIPE_V>();
      Muls(gBUbFloat_, gBUbFloat_, static_cast<float>(-1), halfChunkSize_);
    } else {
      Muls(gBUbFloat_, betaUbFloat, static_cast<float>(-1), halfChunkSize_);
    }
    PipeBarrier<PIPE_V>();
    // k_cumdecay = k * tmp =  -1.0 * k * beta * g_cum_exp
    uint32_t betaShape[2] = {halfChunkSize_, 1};
    uint32_t kShape[2] = {halfChunkSize_, dkAligned_};
    Broadcast<float, BROADCAST_AXIS, 1>(gBKUbFloat_, gBUbFloat_, kShape, betaShape);
    PipeBarrier<PIPE_V>();
    // data copy in kUbFloatCon_
    DataCopyInBf16WithStride(halfChunkSize_, dk_, keyContinousGm, dk_);
    auto bf16Key = inQueue_.DeQue<lowType>();
    Cast(kUbFloatCon_, bf16Key, RoundMode::CAST_NONE, halfChunkSize_ * dkAligned_);
    PipeBarrier<PIPE_V>();
    inQueue_.FreeTensor(bf16Key);
    // compute
    Mul(gBKUbFloat_, gBKUbFloat_, kUbFloatCon_, halfChunkSize_ * dkAligned_);
    PipeBarrier<PIPE_V>();
    gBKLocal_ = outQueue_.AllocTensor<lowType>();
    Cast(gBKLocal_, gBKUbFloat_, RoundMode::CAST_RINT, halfChunkSize_ * dkAligned_);
    outQueue_.EnQue<lowType>(gBKLocal_);
    uint64_t gBKBeginOffset = subOffset_ * dk_;
    DataCopyOutBf16(halfChunkSize_, dk_, dkAligned_, gBKWsGm[gBKBeginOffset]);
    PipeBarrier<PIPE_V>();
    if (gOptional_) {
      // kg = k * (g_cum_exp[-1, None] / g_cum_exp)[..., None]
      uint32_t gEndShape[2] = {1, 1};
      uint32_t gBroadShape[2] = {halfChunkSize_, 1};
      Broadcast<float, BROADCAST_AXIS, 0>(gEndBroadUbFloat_, gCumSumUbFloat[chunkSize_ - 1], gBroadShape, gEndShape);
      PipeBarrier<PIPE_V>();
      Sub(gEndBroadUbFloat_, gEndBroadUbFloat_, gCumSumUbFloat[subOffset_], halfChunkSize_);
      PipeBarrier<PIPE_V>();
      Exp<float, 0, true>(gEndBroadUbFloat_, gEndBroadUbFloat_, halfChunkSize_);
      PipeBarrier<PIPE_V>();
      Broadcast<float, BROADCAST_AXIS, 1>(kgUbFloat_, gEndBroadUbFloat_, kShape, gBroadShape);
      PipeBarrier<PIPE_V>();
      Mul(kgUbFloat_, kgUbFloat_, kUbFloatCon_, halfChunkSize_ * dkAligned_);
      PipeBarrier<PIPE_V>();
      uint64_t kgBeginOffset = subOffset_ * dk_;
      kgLocal_ = outQueue_.AllocTensor<lowType>();
      Cast(kgLocal_, kgUbFloat_, RoundMode::CAST_RINT, halfChunkSize_ * dkAligned_);
      outQueue_.EnQue<lowType>(kgLocal_);
      DataCopyOutBf16(halfChunkSize_, dk_, dkAligned_, outKgGm[kgBeginOffset]);  // stage1 out
    }
  }

  __aicore__ inline void VBetaCompute(const GlobalTensor<lowType> valueGm, const GlobalTensor<lowType> vBetaWsGm,
                                      LocalTensor<float> betaUbFloat, LocalTensor<float> valueUbFloat,
                                      uint32_t subValidRows) {
    uint64_t vBeginOffset = subOffset_ * vRowStride_;
    DataCopyInBf16WithStride(subValidRows, dv_, valueGm[vBeginOffset], vRowStride_);
    valueLocal_ = inQueue_.DeQue<lowType>();
    Cast(valueUbFloat, valueLocal_, AscendC::RoundMode::CAST_NONE, subValidRows * dvAligned_);
    PipeBarrier<PIPE_V>();
    inQueue_.FreeTensor(valueLocal_);
    if (subValidRows < halfChunkSize_) {
      Duplicate(valueUbFloat[subValidRows * dvAligned_], static_cast<float>(0.0f),
                (halfChunkSize_ - subValidRows) * dvAligned_);
      PipeBarrier<PIPE_V>();
    }
    uint32_t betaShape[2] = {halfChunkSize_, 1};
    uint32_t vShape[2] = {halfChunkSize_, dvAligned_};
    Broadcast<float, BROADCAST_AXIS, 1>(vBetaUbFloat_, betaUbFloat, vShape, betaShape);
    PipeBarrier<PIPE_V>();
    Mul(vBetaUbFloat_, valueUbFloat, vBetaUbFloat_, halfChunkSize_ * dvAligned_);
    PipeBarrier<PIPE_V>();
    vBetaLocal_ = outQueue_.AllocTensor<lowType>();
    Cast(vBetaLocal_, vBetaUbFloat_, RoundMode::CAST_RINT, halfChunkSize_ * dvAligned_);
    outQueue_.EnQue<lowType>(vBetaLocal_);
    DataCopyOutBf16(halfChunkSize_, dv_, dvAligned_, vBetaWsGm[subOffset_ * dv_]);
  }

  __aicore__ inline void QPrimeCompute(const GlobalTensor<lowType> outQPrimeGm, LocalTensor<float> gCumSumUbFloat,
                                       LocalTensor<float> gCumExpUbFloat,
                                       const GlobalTensor<lowType> queryContinousGm) {
    // data copy in qUbFloatCon_
    DataCopyInBf16WithStride(halfChunkSize_, dk_, queryContinousGm, dk_);
    auto bf16Query = inQueue_.DeQue<lowType>();
    Cast(qUbFloatCon_, bf16Query, RoundMode::CAST_NONE, halfChunkSize_ * dkAligned_);
    PipeBarrier<PIPE_V>();
    inQueue_.FreeTensor(bf16Query);
    // query * scale
    if (gOptional_) {
      Muls(qUbFloat_, qUbFloatCon_, scale_, halfChunkSize_ * dkAligned_);
      PipeBarrier<PIPE_V>();
      uint32_t gCumExpShape[2] = {halfChunkSize_, 1};
      uint32_t qShape[2] = {halfChunkSize_, dkAligned_};
      Broadcast<float, BROADCAST_AXIS, 1>(gCumExpBroadUbFloat_, gCumExpUbFloat[subOffset_], qShape, gCumExpShape);
      PipeBarrier<PIPE_V>();
      // query * scale * exp(g_cum)[:, None]       # (C, Dk)
      Mul(qPrimeUbFloat_, qUbFloat_, gCumExpBroadUbFloat_, halfChunkSize_ * dkAligned_);
    } else {
      Muls(qPrimeUbFloat_, qUbFloatCon_, scale_, halfChunkSize_ * dkAligned_);
    }
    PipeBarrier<PIPE_V>();
    uint64_t qgBeginOffset = subOffset_ * dk_;
    qPrimeLocal_ = outQueue_.AllocTensor<lowType>();
    Cast(qPrimeLocal_, qPrimeUbFloat_, RoundMode::CAST_RINT, halfChunkSize_ * dkAligned_);
    outQueue_.EnQue<lowType>(qPrimeLocal_);
    DataCopyOutBf16(halfChunkSize_, dk_, dkAligned_, outQPrimeGm[qgBeginOffset]);  // stage1 out
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void DataCopyInFp32(uint64_t len, GlobalTensor<float> y) {
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams kkParams{static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
    fp32InLocal_ = inQueue_.AllocTensor<float>();
    DataCopyPad(fp32InLocal_, y, kkParams, padParams);
    inQueue_.EnQue<float>(fp32InLocal_);
  }

  __aicore__ inline void DataCopyInBf16(uint64_t len, GlobalTensor<lowType> y) {
    DataCopyPadExtParams<lowType> padParams;
    DataCopyExtParams kkParams{static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(lowType)), 0, 0, 0};
    bf16InLocal_ = inQueue_.AllocTensor<lowType>();
    DataCopyPad(bf16InLocal_, y, kkParams, padParams);
    inQueue_.EnQue<lowType>(bf16InLocal_);
  }

  __aicore__ inline void BetaCopyInWithStride(const GlobalTensor<lowType> src, LocalTensor<float> betaUbFloat,
                                              uint32_t subValidRows) {
    uint64_t betaBeginOffset = subOffset_ * nv_;
    DataCopyInBf16WithStride(subValidRows, 1, src[betaBeginOffset], nv_);
    betaLocal_ = inQueue_.DeQue<lowType>();
    if (subValidRows < halfChunkSize_) {
      Duplicate(betaUbBfloat16_, lowType(0.0f), halfChunkSize_);
      PipeBarrier<PIPE_V>();
    }
    Gather(betaUbBfloat16_, betaLocal_, inputGatherBuffer_, static_cast<uint32_t>(0), subValidRows);
    PipeBarrier<PIPE_V>();

    Cast(betaUbFloat, betaUbBfloat16_, AscendC::RoundMode::CAST_NONE, halfChunkSize_);
    PipeBarrier<PIPE_V>();
    inQueue_.FreeTensor(betaLocal_);
  }

  __aicore__ inline void GCopyInWithStride(const GlobalTensor<float> src, uint32_t validLen) {
    DataCopyInFp32WithStride(validLen, 1, src, nv_);
    gLocal_ = inQueue_.DeQue<float>();
    if (validLen < chunkSize_) {
      Duplicate(gCumUbFloat_, 0.0f, chunkSize_);
      PipeBarrier<PIPE_V>();
    }
    Gather(gCumUbFloat_, gLocal_, inputGatherBuffer_, static_cast<uint32_t>(0), validLen);
    PipeBarrier<PIPE_V>();

    inQueue_.FreeTensor(gLocal_);
  }

  __aicore__ inline void DataCopyInFp32WithStride(uint64_t rows,  // number of rows to copy
                                                  uint64_t cols,  // number of elements per row
                                                  const GlobalTensor<float> src,
                                                  uint64_t srcRowStride,  // GM row stride
                                                  uint32_t dstRowStride = 0) {
    DataCopyPadExtParams<float> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<float>(0)};
    uint32_t srcGap = (srcRowStride - cols) * sizeof(float);
    DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(float)),
                             static_cast<uint32_t>(srcGap), static_cast<uint32_t>(dstRowStride), 0};
    fp32InLocal_ = inQueue_.AllocTensor<float>();
    DataCopyPad(fp32InLocal_, src, params, padParams);
    inQueue_.EnQue<float>(fp32InLocal_);
  }

  __aicore__ inline void DataCopyInBf16WithStride(uint64_t rows,  // number of rows to copy
                                                  uint64_t cols,  // number of elements per row
                                                  GlobalTensor<lowType> src, uint64_t srcRowStride) {
    DataCopyPadExtParams<lowType> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                               static_cast<float>(0)};
    uint32_t srcGap = (srcRowStride - cols) * sizeof(lowType);
    DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(lowType)),
                             static_cast<uint32_t>(srcGap), 0, 0};
    bf16InLocal_ = inQueue_.AllocTensor<lowType>();
    DataCopyPad(bf16InLocal_, src, params, padParams);
    inQueue_.EnQue<lowType>(bf16InLocal_);
  }

  __aicore__ inline void DataCopyOutBf16(uint32_t rows, uint32_t cols, uint32_t colsAligned, GlobalTensor<lowType> y) {
    auto tmpLocal = outQueue_.DeQue<lowType>();
    uint32_t srcStride = (colsAligned - cols) * sizeof(lowType) / BLOCK_SIZE;
    DataCopyExtParams yGMParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(lowType)),
                                static_cast<uint32_t>(srcStride), 0, 0};
    DataCopyPad(y, tmpLocal, yGMParams);
    outQueue_.FreeTensor(tmpLocal);
  }

  __aicore__ inline void AttnInverseMMCompute(uint64_t offset) {
    uint64_t leftDown = offset + chunkSize_ * INVERSE_SHAPE;
    uint64_t rightDown = leftDown + INVERSE_SHAPE;
    // bottom-left of right matrix @ top-left of right matrix -> bottom-left of right matrix
    InverseAICProcess(attnWsGm_[leftDown], attnWsGm_[offset], attnWsGm_[leftDown]);
    int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE2));
    SetFlag<HardEvent::FIX_MTE2>(eventID);
    WaitFlag<HardEvent::FIX_MTE2>(eventID);
    // bottom-right of right matrix @ bottom-left of right matrix -> bottom-left of right matrix
    InverseAICProcess(attnWsGm_[rightDown], attnWsGm_[leftDown], attnWsGm_[leftDown]);
    eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE2));
    SetFlag<HardEvent::FIX_MTE2>(eventID);
    WaitFlag<HardEvent::FIX_MTE2>(eventID);
  }

  __aicore__ inline void AICProcess(GlobalTensor<lowType> x, GlobalTensor<lowType> y, GlobalTensor<lowType> z,
                                    const MatmulShapeParams &shape, bool transB = false) {
    mm.SetOrgShape(shape.m, shape.n, shape.k);
    mm.SetSingleShape(shape.sm, shape.sn, shape.sk);
    mm.SetTensorA(x);
    mm.SetTensorB(y, transB);
    mm.IterateAll(z);
    mm.End();
  }

  __aicore__ inline void InverseAICProcess(GlobalTensor<lowType> x, GlobalTensor<lowType> y, GlobalTensor<lowType> z) {
    mm.SetOrgShape(chunkSize_, chunkSize_, chunkSize_);
    mm.SetSingleShape(INVERSE_SHAPE, INVERSE_SHAPE, INVERSE_SHAPE);
    mm.SetTensorA(x);
    mm.SetTensorB(y);
    mm.IterateAll(z);
    mm.End();
  }

  TPipe *pipe_;
  StageOneMT<lowType> &mm;
  const ChunkGatedDeltaRuleTilingData *tiling_;
  ChunkGroup cg_;
  uint32_t nk_;
  uint32_t nv_;
  uint32_t dk_;
  uint32_t dv_;
  uint32_t dkAligned_;
  uint32_t dvAligned_;
  uint32_t numChunk_;
  uint64_t vRowStride_;
  uint32_t halfChunkSize_;
  uint32_t subBlockIdx_;
  uint32_t subOffset_;
  uint32_t coreIdx_;
  uint32_t chunkSize_;
  uint32_t maxLen_;
  uint32_t coreNum_;
  float scale_;
  bool gOptional_;
  uint32_t paraNum_;
  uint64_t ccOffset_;
  uint64_t ckOffset_;
  uint64_t cvOffset_;
  uint32_t validLenBatch_[MAX_PARALLEL_NUM];
  uint32_t subValidLenBatch_[MAX_PARALLEL_NUM];
  uint32_t chunkRowBase_[MAX_PARALLEL_NUM];
  uint64_t chunkStartRowBatch_[MAX_PARALLEL_NUM];
  uint64_t nIdBatch_[MAX_PARALLEL_NUM];
  uint64_t bgOffsetBatch_[MAX_PARALLEL_NUM];

  // chunk GM pointers
  GlobalTensor<lowType> queryGm_;
  GlobalTensor<lowType> keyGm_;
  GlobalTensor<lowType> valueGm_;
  GlobalTensor<lowType> betaGm_;
  GlobalTensor<float> gGm_;
  GlobalTensor<float> outGCumExpGm_;
  GlobalTensor<lowType> outKCumdecayGm_;
  GlobalTensor<lowType> outVInnerGm_;
  GlobalTensor<lowType> outQPrimeGm_;
  GlobalTensor<lowType> outKgBaseGm_;
  GlobalTensor<lowType> outKgGm_;
  GlobalTensor<lowType> outQkGm_;
  GlobalTensor<lowType> vBetaWsGm_;
  GlobalTensor<lowType> kkWsGm_;
  GlobalTensor<lowType> attnWsGm_;
  GlobalTensor<lowType> gBKWsGm_;
  GlobalTensor<lowType> queryConGm_;
  GlobalTensor<lowType> keyConGm_;
  GlobalTensor<float> stageOneMask_;

  // UB queues
  TQue<QuePosition::VECIN, 1> inQueue_;
  TQue<QuePosition::VECOUT, 1> outQueue_;
  TQue<QuePosition::VECOUT, 1> gOutQueue_;

  TBuf<TPosition::VECCALC> tmpBuff_;

  // UB tensors
  LocalTensor<lowType> betaUbBfloat16_;
  LocalTensor<float> betaUbFloat_;
  LocalTensor<float> valueUbFloat_;
  LocalTensor<float> attnUbFloat_;
  LocalTensor<float> inverseBuffer_;
  LocalTensor<float> inverseRes_;
  LocalTensor<float> gCumUbFloat_;
  LocalTensor<float> gCumSumUbFloat_;
  LocalTensor<float> gCumExpUbFloat_;
  LocalTensor<float> gCumExpBroadUbFloat_;
  LocalTensor<float> gBUbFloat_;
  LocalTensor<float> qUbFloat_;
  LocalTensor<float> gBroadUbFloat_;
  LocalTensor<float> gTransBroadUbFloat_;
  LocalTensor<float> gEndBroadUbFloat_;
  LocalTensor<float> gammaUbFloat_;
  LocalTensor<uint32_t> inverseGatherBuffer_;
  LocalTensor<float> qUbFloatCon_;
  LocalTensor<float> kUbFloatCon_;
  LocalTensor<float> gBKUbFloat_;
  LocalTensor<float> kgUbFloat_;
  LocalTensor<float> vBetaUbFloat_;
  LocalTensor<float> qPrimeUbFloat_;
  LocalTensor<uint32_t> inputGatherBuffer_;

  LocalTensor<lowType> betaLocal_;
  LocalTensor<lowType> valueLocal_;
  LocalTensor<lowType> qPrimeLocal_;
  LocalTensor<lowType> vBetaLocal_;
  LocalTensor<float> kkLocal_;
  LocalTensor<float> gLocal_;
  LocalTensor<lowType> gBKLocal_;
  LocalTensor<lowType> kgLocal_;

  LocalTensor<lowType> bf16InLocal_;
  LocalTensor<float> fp32InLocal_;
};
}  // namespace ChunkGatedDeltaRule
#endif  // CHUNK_GATED_DELTA_RULE_STAGE1_H
