/* Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved. */

#include "register/register.h"

namespace domi {
REGISTER_CUSTOM_OP("MsRmsNorm")
  .FrameworkType(ONNX)
  .OriginOpType("MsRmsNorm")
  .ParseParamsByOperatorFn(AutoMappingByOpFn);
}  // namespace domi
