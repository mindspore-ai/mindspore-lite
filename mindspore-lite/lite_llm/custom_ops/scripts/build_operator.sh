#!/usr/bin/env bash
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 OPERATOR_DIR INSTALL_MODE [TARGET]" >&2
    exit 2
fi

operator_dir=$(cd -- "$1" && pwd)
install_mode=$2
target=${3:-package}
build_dir="${operator_dir}/build_out"
install_dir="${build_dir}/installFile"
tools_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

case "${install_mode}" in
    INSTALL) ;;
    *)
        echo "Unsupported install mode: ${install_mode}" >&2
        exit 2
        ;;
esac

if [[ -z "${DDK_PATH:-}" ]]; then
    echo "DDK_PATH is not set; source ddk_env/tools/tools_ascendc/set_ascendc_env.sh first." >&2
    exit 2
fi

mkdir -p "${build_dir}"
rm -rf "${build_dir:?}/"*

# Optional SoC override supplied by the caller (defaults to the preset value).
soc_args=()
if [[ -n "${ASCEND_COMPUTE_UNIT:-}" ]]; then
    soc_args+=("-DASCEND_COMPUTE_UNIT=${ASCEND_COMPUTE_UNIT}")
fi

cmake --preset default -S "${operator_dir}" -DHOST=true "${soc_args[@]}"
cmake --build "${build_dir}" --target "${target}" -j "${BUILD_JOBS:-16}"

# Configure a non-HOST tree so the DDK deployment rules are selected. NUL
# delimiters preserve cache values containing spaces.
mkdir -p "${install_dir}"
mapfile -d '' -t cmake_options < <(
    python3 "${tools_dir}/cmake_preset_args.py" \
        "${operator_dir}/CMakePresets.json" "${operator_dir}"
)
cmake -S "${operator_dir}" -B "${install_dir}" \
    "${cmake_options[@]}" "-D${install_mode}=true" "${soc_args[@]}"
cmake --build "${install_dir}" -j "${BUILD_JOBS:-16}"
cmake --install "${install_dir}"

echo "Build and install succeeded: ${operator_dir}"
