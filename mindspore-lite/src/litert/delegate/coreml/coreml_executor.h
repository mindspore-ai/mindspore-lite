/**
* Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_COREML_EXECUTOR_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_COREML_EXECUTOR_H_

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <string>
#include <vector>
#include "include/api/types.h"

constexpr int kNum12 = 12;

API_AVAILABLE(ios(kNum12))
@interface MSFeatureProvider : NSObject <MLFeatureProvider> {
  const std::vector<mindspore::MSTensor>* _ms_tensors;
  NSSet* _featureNames;
}

- (instancetype)initWithMSTensor:(const std::vector<mindspore::MSTensor>*)inputs
                 coreMLVersion:(int)coreMLVersion;
- (MLFeatureValue *)featureValueForName:(NSString *)featureName;
@property(nonatomic, readonly) NSSet<NSString *> *featureNames;
@end

API_AVAILABLE(ios(kNum12))
@interface CoreMLExecutor : NSObject

- (bool)run:(const std::vector<mindspore::MSTensor>&)inputs
                  outputs:(const std::vector<mindspore::MSTensor>&)outputs;

- (bool)load:(NSURL*)compileUrl;

@property MLModel* model;
@property(nonatomic, readonly) int coreMLVersion;
@end
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_COREML_EXECUTOR_H_
