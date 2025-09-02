#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "=======Preparing Dataset======="
[ -d "dataset" ] && echo "dataset was already created" && exit 0
PLACES_DATA_PATH=$1
if [ ! -d ${PLACES_DATA_PATH}/val_256/ ]; then
  echo "The path" ${PLACES_DATA_PATH} "does not contain Places validation dataset. Please read the README file!" && exit 1
fi
class_id=0
sp="/-\|"
classes=("4" "98" "6" "7" "10" "15" "17" "70" "26" "30")
echo -n 'Prep class '
for class in "${classes[@]}"; do
  mkdir -p dataset/$class_id
  f=0
  i=1
  echo -n $(($class_id+1)) ' '
  cat scripts/places365_val.txt | grep -w ${class} | awk '{print $1}' | while read line
  do 
    printf "\b${sp:i++%${#sp}:1}"
    convert -colorspace RGB -gravity center -crop '224x224+0+0' ${PLACES_DATA_PATH}/val_256/$line dataset/$class_id/$f.bmp; 
    f=$(($f+1)); 
  done
  printf "\b"
  class_id=$(($class_id+1))
done
echo ' '
