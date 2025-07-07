/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#include "src/extendrt/delegate/factory.h"
#include "src/extendrt/delegate/type.h"

namespace mindspore {
using mindspore::ExtendDelegate;

template <typename T>
DelegateRegistry<T> &DelegateRegistry<T>::GetInstance() {
  static DelegateRegistry<T> instance;
  return instance;
}

template <typename T>
void DelegateRegistry<T>::RegDelegate(const mindspore::DeviceType &device_type, const std::string &provider,
                                      DelegateCreator<T> *creator) {
  auto it = creator_map_.find(device_type);
  if (it == creator_map_.end()) {
    HashMap<std::string, DelegateCreator<T> *> map;
    map[provider] = creator;
    creator_map_[device_type] = map;
    return;
  }
  it->second[provider] = creator;
}

template <typename T>
void DelegateRegistry<T>::UnRegDelegate(const mindspore::DeviceType &device_type, const std::string &provider) {
  auto it = creator_map_.find(device_type);
  if (it != creator_map_.end()) {
    creator_map_.erase(it);
  }
}

template <typename T>
T DelegateRegistry<T>::GetDelegate(const mindspore::DeviceType &device_type, const std::string &provider,
                                   const std::shared_ptr<Context> &ctx, const ConfigInfos &config_infos) {
  //  find common delegate
  auto it = creator_map_.find(device_type);
  if (it == creator_map_.end()) {
    MS_LOG(ERROR) << "Find device type " << device_type << " failed.";
    return nullptr;
  }
  auto creator_it = it->second.find(provider);
  if (creator_it == it->second.end()) {
    MS_LOG(ERROR) << "Find provider " << provider << " failed.";
    return nullptr;
  }
  return (*(creator_it->second))(ctx, config_infos);
}

template class DelegateRegistry<ExtendDelegate *>;
template class DelegateRegistry<std::shared_ptr<LiteGraphExecutor>>;

}  // namespace mindspore
