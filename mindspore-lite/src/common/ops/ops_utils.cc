/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <memory>
#include <utility>
#include "src/common/ops/ops_utils.h"
#include "mindapi/base/shared_ptr.h"
#ifdef PRIMITIVE_WRITEABLE
#include "ops/primitive_c.h"
#include "mindspore/ops/infer/conv3d.h"
#include "mindspore/ops/op_def/op_name.h"

namespace mindspore {
namespace lite {
namespace ops {
std::unique_ptr<schema::PrimitiveT> MSOp2SchemaOp(const mindspore::ops::Custom *op) {
  auto schema_op = std::make_unique<schema::CustomT>();
  if (schema_op == nullptr) {
    return nullptr;
  }
  if (op->GetAttr("type") != nullptr) {
    schema_op->type = op->get_type();
  }
  if (op->GetAttr("attr") != nullptr) {
    auto attr_map = op->get_attr();
    for (const auto &attr_item : attr_map) {
      auto attr = std::make_unique<schema::AttributeT>();
      if (attr == nullptr) {
        return nullptr;
      }
      attr->name = attr_item.first;
      attr->data = attr_item.second;
      schema_op->attr.emplace_back(std::move(attr));
    }
  }
  auto prim = std::make_unique<schema::PrimitiveT>();
  if (prim == nullptr) {
    return nullptr;
  }
  prim->value.value = schema_op.release();
  prim->value.type = schema::PrimitiveType_Custom;
  return prim;
}

std::unique_ptr<schema::PrimitiveT> MSOp2SchemaOp(const mindspore::ops::Conv3D *op) {
  auto schema_op = std::make_unique<schema::CustomT>();
  if (schema_op == nullptr) {
    return nullptr;
  }
  schema_op->type = "Conv3D";

  auto prim = std::make_unique<schema::PrimitiveT>();
  if (prim == nullptr) {
    return nullptr;
  }
  prim->value.value = schema_op.release();
  prim->value.type = schema::PrimitiveType_Custom;
  return prim;
}

std::unique_ptr<schema::PrimitiveT> MSOp2SchemaOp(const mindspore::ops::GridSampler2D *op) {
  auto schema_op = std::make_unique<schema::CustomT>();
  if (schema_op == nullptr) {
    return nullptr;
  }
  schema_op->type = "GridSampler";
  if (op->GetAttr("interpolation_mode") != nullptr) {
    auto attr = std::make_unique<schema::AttributeT>();
    if (attr == nullptr) {
      return nullptr;
    }
    attr->name = "interpolation_mode";
    auto interpolation_mode = op->get_interpolation_mode();
    std::vector<uint8_t> container(sizeof(int64_t));
    if (memcpy_s(container.data(), container.size(), &interpolation_mode, sizeof(int64_t)) != EOK) {
      MS_LOG(ERROR) << "GridSampler2D: DeepCopy interpolation_mode failed.";
      return nullptr;
    }
    attr->data = container;
    schema_op->attr.emplace_back(std::move(attr));
  }
  if (op->GetAttr("padding_mode") != nullptr) {
    auto attr = std::make_unique<schema::AttributeT>();
    if (attr == nullptr) {
      return nullptr;
    }
    attr->name = "padding_mode";
    auto padding_mode = op->get_padding_mode();
    std::vector<uint8_t> container(sizeof(int64_t));
    if (memcpy_s(container.data(), container.size(), &padding_mode, sizeof(int64_t)) != EOK) {
      MS_LOG(ERROR) << "GridSampler2D: DeepCopy padding_mode failed.";
      return nullptr;
    }
    attr->data = container;
    schema_op->attr.emplace_back(std::move(attr));
  }
  if (op->GetAttr("align_corners") != nullptr) {
    auto attr = std::make_unique<schema::AttributeT>();
    if (attr == nullptr) {
      return nullptr;
    }
    attr->name = "align_corners";
    auto align_corners = op->get_align_corners();
    std::vector<uint8_t> container(1);
    container[0] = align_corners;
    attr->data = container;
    schema_op->attr.emplace_back(std::move(attr));
  }

  auto prim = std::make_unique<schema::PrimitiveT>();
  if (prim == nullptr) {
    return nullptr;
  }
  prim->value.value = schema_op.release();
  prim->value.type = schema::PrimitiveType_Custom;
  return prim;
}
}  // namespace ops

template <typename T>
std::unique_ptr<schema::PrimitiveT> PrimitiveCreator(const PrimitivePtr &primitive) {
  auto ms_primc = api::MakeShared<T>(primitive);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

REG_MINDSPORE_OPERATOR(Abs)
REG_MINDSPORE_OPERATOR(Activation)
REG_MINDSPORE_OPERATOR(ActivationGrad)
REG_MINDSPORE_OPERATOR(Adam)
REG_MINDSPORE_OPERATOR(AddFusion)
REG_MINDSPORE_OPERATOR(AdderFusion)
REG_MINDSPORE_OPERATOR(AddGrad)
REG_MINDSPORE_OPERATOR(AddN)
REG_MINDSPORE_OPERATOR(All)
REG_MINDSPORE_OPERATOR(ApplyMomentum)
REG_MINDSPORE_OPERATOR(ArgMaxFusion)
REG_MINDSPORE_OPERATOR(ArgMinFusion)
REG_MINDSPORE_OPERATOR(Assert)
REG_MINDSPORE_OPERATOR(Assign)
REG_MINDSPORE_OPERATOR(AssignAdd)
REG_MINDSPORE_OPERATOR(AudioSpectrogram)
REG_MINDSPORE_OPERATOR(AvgPoolFusion)
REG_MINDSPORE_OPERATOR(AvgPoolGrad)
REG_MINDSPORE_OPERATOR(BatchNorm)
REG_MINDSPORE_OPERATOR(BatchNormGrad)
REG_MINDSPORE_OPERATOR(BatchToSpace)
REG_MINDSPORE_OPERATOR(BatchToSpaceND)
REG_MINDSPORE_OPERATOR(BiasAdd)
REG_MINDSPORE_OPERATOR(BinaryCrossEntropy)
REG_MINDSPORE_OPERATOR(BinaryCrossEntropyGrad)
REG_MINDSPORE_OPERATOR(BiasAddGrad)
REG_MINDSPORE_OPERATOR(BroadcastTo)
REG_MINDSPORE_OPERATOR(Cast)
REG_MINDSPORE_OPERATOR(Ceil)
REG_MINDSPORE_OPERATOR(Clip)
REG_MINDSPORE_OPERATOR(Concat)
REG_MINDSPORE_OPERATOR(Attention)
REG_MINDSPORE_OPERATOR(Conv2DBackpropFilterFusion)
REG_MINDSPORE_OPERATOR(Conv2DBackpropInputFusion)
REG_MINDSPORE_OPERATOR(Conv2DFusion)
REG_MINDSPORE_OPERATOR(Conv2dTransposeFusion)
REG_MINDSPORE_OPERATOR(Cos)
REG_MINDSPORE_OPERATOR(ConstantOfShape)
REG_MINDSPORE_OPERATOR(Crop)
REG_MINDSPORE_OPERATOR(CustomExtractFeatures)
REG_MINDSPORE_OPERATOR(CustomNormalize)
REG_MINDSPORE_OPERATOR(CustomPredict)
REG_MINDSPORE_OPERATOR(DeConv2DGradFilter)
REG_MINDSPORE_OPERATOR(Depend)
REG_MINDSPORE_OPERATOR(DepthToSpace)
REG_MINDSPORE_OPERATOR(DetectionPostProcess)
REG_MINDSPORE_OPERATOR(DivFusion)
REG_MINDSPORE_OPERATOR(DivGrad)
REG_MINDSPORE_OPERATOR(Dropout)
REG_MINDSPORE_OPERATOR(DropoutGrad)
REG_MINDSPORE_OPERATOR(Elu)
REG_MINDSPORE_OPERATOR(Eltwise)
REG_MINDSPORE_OPERATOR(Equal)
REG_MINDSPORE_OPERATOR(EmbeddingLookupFusion)
REG_MINDSPORE_OPERATOR(ExpFusion)
REG_MINDSPORE_OPERATOR(ExpandDims)
REG_MINDSPORE_OPERATOR(FakeQuantWithMinMaxVars)
REG_MINDSPORE_OPERATOR(FakeQuantWithMinMaxVarsPerChannel)
REG_MINDSPORE_OPERATOR(FftReal)
REG_MINDSPORE_OPERATOR(FftImag)
REG_MINDSPORE_OPERATOR(Flatten)
REG_MINDSPORE_OPERATOR(FlattenGrad)
REG_MINDSPORE_OPERATOR(Floor)
REG_MINDSPORE_OPERATOR(FloorDiv)
REG_MINDSPORE_OPERATOR(FloorMod)
REG_MINDSPORE_OPERATOR(Fill)
REG_MINDSPORE_OPERATOR(FillV2)
REG_MINDSPORE_OPERATOR(FullConnection)
REG_MINDSPORE_OPERATOR(FusedBatchNorm)
REG_MINDSPORE_OPERATOR(Gather)
REG_MINDSPORE_OPERATOR(GatherNd)
REG_MINDSPORE_OPERATOR(Greater)
REG_MINDSPORE_OPERATOR(GreaterEqual)
REG_MINDSPORE_OPERATOR(HashtableLookup)
REG_MINDSPORE_OPERATOR(InstanceNorm)
REG_MINDSPORE_OPERATOR(LayerNormFusion)
REG_MINDSPORE_OPERATOR(LeakyRelu)
REG_MINDSPORE_OPERATOR(Less)
REG_MINDSPORE_OPERATOR(LessEqual)
REG_MINDSPORE_OPERATOR(Log)
REG_MINDSPORE_OPERATOR(LogGrad)
REG_MINDSPORE_OPERATOR(LogicalAnd)
REG_MINDSPORE_OPERATOR(LogicalNot)
REG_MINDSPORE_OPERATOR(LogicalOr)
REG_MINDSPORE_OPERATOR(LpNormalization)
REG_MINDSPORE_OPERATOR(LRN)
REG_MINDSPORE_OPERATOR(LshProjection)
REG_MINDSPORE_OPERATOR(LSTM)
REG_MINDSPORE_OPERATOR(L2NormalizeFusion)
REG_MINDSPORE_OPERATOR(MatMulFusion)
REG_MINDSPORE_OPERATOR(Maximum)
REG_MINDSPORE_OPERATOR(MaximumGrad)
REG_MINDSPORE_OPERATOR(MaxPoolFusion)
REG_MINDSPORE_OPERATOR(MaxPoolGrad)
REG_MINDSPORE_OPERATOR(SwitchLayer)
REG_MINDSPORE_OPERATOR(Mfcc)
REG_MINDSPORE_OPERATOR(Minimum)
REG_MINDSPORE_OPERATOR(MinimumGrad)
REG_MINDSPORE_OPERATOR(Mod)
REG_MINDSPORE_OPERATOR(MulFusion)
REG_MINDSPORE_OPERATOR(MulGrad)
REG_MINDSPORE_OPERATOR(Neg)
REG_MINDSPORE_OPERATOR(NegGrad)
REG_MINDSPORE_OPERATOR(NotEqual)
REG_MINDSPORE_OPERATOR(NonMaxSuppression)
REG_MINDSPORE_OPERATOR(OneHot)
REG_MINDSPORE_OPERATOR(OnesLike)
REG_MINDSPORE_OPERATOR(PadFusion)
REG_MINDSPORE_OPERATOR(PartialFusion)
REG_MINDSPORE_OPERATOR(PowerGrad)
REG_MINDSPORE_OPERATOR(PowFusion)
REG_MINDSPORE_OPERATOR(PriorBox)
REG_MINDSPORE_OPERATOR(PReLUFusion)
REG_MINDSPORE_OPERATOR(QuantDTypeCast)
REG_MINDSPORE_OPERATOR(Rank)
REG_MINDSPORE_OPERATOR(Range)
REG_MINDSPORE_OPERATOR(Reciprocal)
REG_MINDSPORE_OPERATOR(RealDiv)
REG_MINDSPORE_OPERATOR(ReduceFusion)
REG_MINDSPORE_OPERATOR(Reshape)
REG_MINDSPORE_OPERATOR(Resize)
REG_MINDSPORE_OPERATOR(ReverseSequence)
REG_MINDSPORE_OPERATOR(ReverseV2)
REG_MINDSPORE_OPERATOR(Rfft)
REG_MINDSPORE_OPERATOR(ROIPooling)
REG_MINDSPORE_OPERATOR(Round)
REG_MINDSPORE_OPERATOR(Rsqrt)
REG_MINDSPORE_OPERATOR(ScaleFusion)
REG_MINDSPORE_OPERATOR(ScatterNd)
REG_MINDSPORE_OPERATOR(SGD)
REG_MINDSPORE_OPERATOR(Shape)
REG_MINDSPORE_OPERATOR(SigmoidCrossEntropyWithLogits)
REG_MINDSPORE_OPERATOR(SigmoidCrossEntropyWithLogitsGrad)
REG_MINDSPORE_OPERATOR(Sin)
REG_MINDSPORE_OPERATOR(SkipGram)
REG_MINDSPORE_OPERATOR(SliceFusion)
REG_MINDSPORE_OPERATOR(SmoothL1Loss)
REG_MINDSPORE_OPERATOR(SmoothL1LossGrad)
REG_MINDSPORE_OPERATOR(Softmax)
REG_MINDSPORE_OPERATOR(SoftmaxCrossEntropyWithLogits)
REG_MINDSPORE_OPERATOR(SpaceToBatch)
REG_MINDSPORE_OPERATOR(SpaceToBatchND)
REG_MINDSPORE_OPERATOR(SpaceToDepth)
REG_MINDSPORE_OPERATOR(SparseSoftmaxCrossEntropyWithLogits)
REG_MINDSPORE_OPERATOR(SparseToDense)
REG_MINDSPORE_OPERATOR(Split)
REG_MINDSPORE_OPERATOR(Sqrt)
REG_MINDSPORE_OPERATOR(Squeeze)
REG_MINDSPORE_OPERATOR(Square)
REG_MINDSPORE_OPERATOR(SquaredDifference)
REG_MINDSPORE_OPERATOR(Stack)
REG_MINDSPORE_OPERATOR(StridedSlice)
REG_MINDSPORE_OPERATOR(SubFusion)
REG_MINDSPORE_OPERATOR(SubGrad)
REG_MINDSPORE_OPERATOR(Switch)
REG_MINDSPORE_OPERATOR(TensorListFromTensor)
REG_MINDSPORE_OPERATOR(TensorListGetItem)
REG_MINDSPORE_OPERATOR(TensorListReserve)
REG_MINDSPORE_OPERATOR(TensorListSetItem)
REG_MINDSPORE_OPERATOR(TensorListStack)
REG_MINDSPORE_OPERATOR(TileFusion)
REG_MINDSPORE_OPERATOR(TopKFusion)
REG_MINDSPORE_OPERATOR(Transpose)
REG_MINDSPORE_OPERATOR(Unique)
REG_MINDSPORE_OPERATOR(UnsortedSegmentSum)
REG_MINDSPORE_OPERATOR(Unsqueeze)
REG_MINDSPORE_OPERATOR(Unstack)
REG_MINDSPORE_OPERATOR(LSTMGrad)
REG_MINDSPORE_OPERATOR(Where)
REG_MINDSPORE_OPERATOR(ZerosLike)
REG_MINDSPORE_OPERATOR(Select)
REG_MINDSPORE_OPERATOR(ScatterNdUpdate)
REG_MINDSPORE_OPERATOR(GRU)
REG_MINDSPORE_OPERATOR(NonZero)
REG_MINDSPORE_OPERATOR(InvertPermutation)
REG_MINDSPORE_OPERATOR(Size)
REG_MINDSPORE_OPERATOR(RandomStandardNormal)
REG_MINDSPORE_OPERATOR(CropAndResize)
REG_MINDSPORE_OPERATOR(Erf)
REG_MINDSPORE_OPERATOR(StridedSliceGrad)
REG_MINDSPORE_OPERATOR(IsFinite)
REG_MINDSPORE_OPERATOR(LinSpace)
REG_MINDSPORE_OPERATOR(UniformReal)
REG_MINDSPORE_OPERATOR(AbsGrad)
REG_MINDSPORE_OPERATOR(RsqrtGrad)
REG_MINDSPORE_OPERATOR(SqrtGrad)
REG_MINDSPORE_OPERATOR(LayerNormGrad)
REG_MINDSPORE_OPERATOR(ResizeGrad)
REG_MINDSPORE_OPERATOR(Splice)
REG_MINDSPORE_OPERATOR(LogSoftmax)
REG_MINDSPORE_OPERATOR(Call)
REG_MINDSPORE_OPERATOR(Custom)
REG_MINDSPORE_OPERATOR(CumSum)
REG_MINDSPORE_OPERATOR(SplitWithOverlap)
REG_MINDSPORE_OPERATOR(RaggedRange)
REG_MINDSPORE_OPERATOR(GLU)
REG_MINDSPORE_OPERATOR(TensorArray)
REG_MINDSPORE_OPERATOR(TensorArrayRead)
REG_MINDSPORE_OPERATOR(TensorArrayWrite)
REG_MINDSPORE_OPERATOR(Affine)
REG_MINDSPORE_OPERATOR(AllGather)
REG_MINDSPORE_OPERATOR(ReduceScatter)
REG_MINDSPORE_OPERATOR(DynamicQuant)
REG_MINDSPORE_OPERATOR(LSTMGradData)
REG_MINDSPORE_OPERATOR(LSTMGradWeight)
REG_MINDSPORE_OPERATOR(RandomNormal)
REG_MINDSPORE_OPERATOR(NLLLoss)
REG_MINDSPORE_OPERATOR(NLLLossGrad)
REG_MINDSPORE_OPERATOR(FormatTranspose)
REG_MINDSPORE_OPERATOR(GatherD)
REG_MINDSPORE_OPERATOR(GroupNormFusion)
REG_MINDSPORE_OPERATOR(Log1p)
REG_MINDSPORE_OPERATOR(TensorScatterAdd)
REG_MINDSPORE_OPERATOR(ScatterElements)
REG_MINDSPORE_OPERATOR(Triu)
REG_MINDSPORE_OPERATOR(Tril)
REG_MINDSPORE_OPERATOR(SparseFillEmptyRows)
REG_MINDSPORE_OPERATOR(SparseReshape)
REG_MINDSPORE_OPERATOR(SparseSegmentSum)
REG_MINDSPORE_OPERATOR(AdamWeightDecay)
REG_MINDSPORE_OPERATOR(Conv3D)
REG_MINDSPORE_OPERATOR(GridSampler2D)
}  // namespace lite
}  // namespace mindspore

#endif
