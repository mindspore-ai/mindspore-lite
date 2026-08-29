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
#include "ms_rms_norm_tiling_data.h"  // NOLINT(build/include_subdir)

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr float DEFAULT_EPSILON = 1.0e-6F;
constexpr uint64_t FP16_DMA_ALIGNMENT_ELEMENTS = 16U;
constexpr uint32_t FALLBACK_AIV_CORE_COUNT = 1U;
constexpr uint32_t MAX_WHOLE_ROW_ELEMENTS = 8U * 1024U;
constexpr uint32_t FP16_VECTOR_ELEMENTS = 128U;

struct ShapeInfo {
  uint64_t rows;
  uint32_t hidden;
  bool hasGamma;
};

bool ValidateShapes(const gert::Shape *xShape, const gert::Shape *gammaShape, ShapeInfo *shapeInfo = nullptr) {
  if (xShape == nullptr || xShape->GetDimNum() == 0) {
    printf("[MsRmsNorm] validation: x must have rank >= 1\n");
    return false;
  }

  const size_t rank = xShape->GetDimNum();
  const int64_t hidden = xShape->GetDim(rank - 1);
  if (hidden <= 0 || static_cast<uint64_t>(hidden) > std::numeric_limits<uint32_t>::max()) {
    printf("[MsRmsNorm] validation: the last x dimension must fit uint32\n");
    return false;
  }
  if (static_cast<uint64_t>(hidden) % FP16_DMA_ALIGNMENT_ELEMENTS != 0) {
    printf("[MsRmsNorm] validation: K must be a multiple of 16 FP16 elements, got %" PRId64 "\n",
           static_cast<int64_t>(hidden));
    return false;
  }
  if (static_cast<uint64_t>(hidden) > MAX_WHOLE_ROW_ELEMENTS) {
    printf("[MsRmsNorm] validation: K must be <= %u, got %" PRId64 "\n", MAX_WHOLE_ROW_ELEMENTS,
           static_cast<int64_t>(hidden));
    return false;
  }

  uint64_t rows = 1;
  for (size_t i = 0; i + 1 < rank; ++i) {
    const int64_t dim = xShape->GetDim(i);
    if (dim <= 0 || rows > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
      printf("[MsRmsNorm] validation: x has an empty or overflowing leading dimension\n");
      return false;
    }
    rows *= static_cast<uint64_t>(dim);
  }
  if (rows > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(hidden)) {
    printf("[MsRmsNorm] validation: x element count overflows uint64\n");
    return false;
  }

  const bool hasGamma = gammaShape != nullptr;
  if (hasGamma && (gammaShape->GetDimNum() != 1 || gammaShape->GetDim(0) != hidden)) {
    printf("[MsRmsNorm] validation: optional gamma must be 1D [K]\n");
    return false;
  }

  if (shapeInfo != nullptr) {
    *shapeInfo = ShapeInfo{rows, static_cast<uint32_t>(hidden), hasGamma};
  }
  return true;
}

float GetEpsilon(const gert::TilingContext *context) {
  float epsilon = DEFAULT_EPSILON;
  const auto *attrs = context->GetAttrs();
  if (attrs != nullptr && attrs->GetAttrNum() > 0) {
    const float *value = attrs->GetFloat(0);
    if (value != nullptr) {
      epsilon = *value;
    }
  }
  return epsilon;
}

void SplitWithTail(uint64_t length, uint32_t split, uint32_t &loops, uint32_t &tail) {
  loops = static_cast<uint32_t>((length - 1U) / split);
  tail = static_cast<uint32_t>(length - static_cast<uint64_t>(loops) * split);
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto *xStorage = context->GetInputShape(0);
  const auto *gammaStorage = context->GetInputShape(1);
  if (xStorage == nullptr) {
    return ge::GRAPH_FAILED;
  }

  ShapeInfo shape{};
  const gert::Shape *gammaShape = gammaStorage == nullptr ? nullptr : &gammaStorage->GetStorageShape();
  if (!ValidateShapes(&xStorage->GetStorageShape(), gammaShape, &shape)) {
    return ge::GRAPH_FAILED;
  }

  const float epsilon = GetEpsilon(context);
  if (!std::isfinite(epsilon) || epsilon < 0.0F) {
    printf("[MsRmsNorm] validation: epsilon must be finite and non-negative\n");
    return ge::GRAPH_FAILED;
  }

  uint32_t physicalVectorCores = FALLBACK_AIV_CORE_COUNT;
  auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  if (platform != nullptr && platform->GetCoreNumAiv() != 0U) {
    physicalVectorCores = platform->GetCoreNumAiv();
  }
  const uint64_t targetBlocks = std::min<uint64_t>(shape.rows, physicalVectorCores);
  const uint64_t blockM = (shape.rows + targetBlocks - 1U) / targetBlocks;
  const uint32_t splitM = static_cast<uint32_t>((shape.rows + blockM - 1U) / blockM);

  const uint32_t splitK = shape.hidden;
  uint32_t loopK = 0;
  uint32_t tailK = 0;
  SplitWithTail(shape.hidden, splitK, loopK, tailK);

  const uint32_t reduceSplitK = std::min(shape.hidden, FP16_VECTOR_ELEMENTS);
  uint32_t reduceLoopK = 0;
  uint32_t reduceTailK = 0;
  SplitWithTail(shape.hidden, reduceSplitK, reduceLoopK, reduceTailK);

  TilingData4RmsNorm tiling;
  tiling.set_originM(shape.rows);
  tiling.set_originK(shape.hidden);
  tiling.set_epsilon(epsilon);
  tiling.set_reciprocalOfHLength(1.0F / static_cast<float>(shape.hidden));
  tiling.set_hasGamma(shape.hasGamma ? 1U : 0U);
  tiling.tilingDataGm2Ub.set_blockM(blockM);
  tiling.tilingDataGm2Ub.set_splitM(splitM);
  tiling.tilingDataGm2Ub.set_splitK(splitK);
  tiling.tilingDataGm2Ub.set_loopK(loopK);
  tiling.tilingDataGm2Ub.set_tailK(tailK);
  tiling.tilingDataLargeReduce.set_reduceSplitK(reduceSplitK);
  tiling.tilingDataLargeReduce.set_reduceLoopK(reduceLoopK);
  tiling.tilingDataLargeReduce.set_reduceTailK(reduceTailK);

  auto *rawTiling = context->GetRawTilingData();
  size_t *workspace = context->GetWorkspaceSizes(1);
  if (rawTiling == nullptr || workspace == nullptr) {
    return ge::GRAPH_FAILED;
  }
  context->SetBlockDim(splitM);
  tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
  rawTiling->SetDataSize(tiling.GetDataSize());
  workspace[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
  if (context == nullptr) {
    return GRAPH_FAILED;
  }
  const gert::Shape *xShape = context->GetInputShape(0);
  const gert::Shape *gammaShape = context->GetInputShape(1);
  gert::Shape *yShape = context->GetOutputShape(0);
  if (yShape == nullptr || !optiling::ValidateShapes(xShape, gammaShape)) {
    return GRAPH_FAILED;
  }
  *yShape = *xShape;
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
  if (context == nullptr || context->GetInputDataType(0) != ge::DT_FLOAT16) {
    printf("[MsRmsNorm] validation: x must be FP16\n");
    return GRAPH_FAILED;
  }
  const ge::DataType gammaType = context->GetInputDataType(1);
  if (gammaType != ge::DT_UNDEFINED && gammaType != ge::DT_FLOAT16) {
    printf("[MsRmsNorm] validation: optional gamma must be FP16\n");
    return GRAPH_FAILED;
  }
  context->SetOutputDataType(0, ge::DT_FLOAT16);
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class MsRmsNorm : public OpDef {
 public:
  explicit MsRmsNorm(const char *name) : OpDef(name) {
    this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    this->Input("w").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    this->Attr("epsilon").AttrType(OPTIONAL).Float(1.0e-6F);

    this->SetInferShape(ge::InferShape);
    this->SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("kirin9020");
  }
};

OP_ADD(MsRmsNorm);
}  // namespace ops
