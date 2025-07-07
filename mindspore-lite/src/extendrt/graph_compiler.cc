/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "extendrt/graph_compiler.h"

namespace mindspore {
namespace infer {
ExcutionPlan GraphCompiler::Compile(FuncGraphPtr func_graph) { return {}; }
GraphId GraphCompiler::CompileSegment(const GraphSegmentPtr segment) { return -1; }
CompileResult GraphCompiler::LinkSegment() { return CompileResult(); }
ExcutionPlan GraphCompiler::Schedule(const CompileResult &compile_result) {
  return scheduler_.Schedule(compile_result);
}
}  // namespace infer
}  // namespace mindspore
