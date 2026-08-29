#!/bin/bash
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

# Standalone build for the lite_llm module (independent of the main
# mindspore-lite build): host x86 for development/tests, OHOS cross-compile
# for the Kirin NPU NNRT backend.

set -e

TOP_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${TOP_DIR}/build"
OUTPUT_DIR="${TOP_DIR}/output"
WHEEL_DIR="${BUILD_DIR}/wheel"
VERSION="$(cat "${TOP_DIR}/export/version.txt")"
PACKAGE_NAME="mindspore-lite-llm-linux-x64-${VERSION}"

# Defaults
THREAD_NUM=8
BUILD_TYPE="Release"
VERBOSE_FLAG=""
INC_BUILD="off"
BACKEND="host"
TESTS="off"

usage()
{
  echo "Usage:"
  echo "bash build.sh [-d] [-r] [-b NAME] [-t VAL] [-v] [-i] [-j[n]] [-h]"
  echo ""
  echo "Options:"
  echo "    -d       Debug mode"
  echo "    -r       Release mode (default)"
  echo "    -b NAME    Backend to build (default: host). Currently supported:"
  echo "                 nnrt  — Kirin NPU, OHOS cross-compile (requires OHOS_NDK)"
  echo "                 host  — x64 host build (dev/test, no device backend)"
  echo "                 Future backends (qnn/metal/...) get their own value here."
  echo "    -t VAL     Build unit tests: on|ut to build, off to skip (default: off)"
  echo "    -v       Display build commands"
  echo "    -i       Enable incremental build (do not clean build directory)"
  echo "    -j[n]    Set the number of build threads (default: -j8)"
  echo "    -h       Print usage"
  echo ""
  echo "Environment variables:"
  echo "    OHOS_NDK                 native root (<SDK>/native, NDK), required with -b nnrt"
  echo "    MSLITE_LLM_ENABLE_NNRT    on/off (default: off on host, on with -b nnrt)"
  echo "    MSLITE_LLM_WHL_REPO       extra index URL for wheel build deps (optional)"
  echo ""
  echo "Packaging (after a successful build):"
  echo "    host:  builds mslite-llm-{version}.whl into output/tool/"
  echo "    nnrt:  assembles output/${PACKAGE_NAME}.tar.gz with lib/ include/ bin/ + tool/ + ascendc_ops/*.run"
  echo ""
  echo "Run tests:"
  echo "    ctest --test-dir build --output-on-failure"
  echo "    MSLITE_LLM_ST_DEVICE=1 pytest tests/st --gguf=/path/model.gguf  # ST (model-based e2e, see tests/st/)"
}

# Parse arguments
while getopts 'db:rt:vij:h' opt
do
  case "${opt}" in
    d)
      BUILD_TYPE="Debug" ;;
    r)
      BUILD_TYPE="Release" ;;
    b)
      BACKEND="${OPTARG}" ;;
    t)
      case "${OPTARG}" in
        on|ut) TESTS="on" ;;
        off)   TESTS="off" ;;
        *)
          echo "Error: invalid value ${OPTARG} for -t (expected on|ut|off)" >&2
          usage
          exit 1
          ;;
      esac
      ;;
    v)
      VERBOSE_FLAG="-v" ;;
    i)
      INC_BUILD="on" ;;
    j)
      THREAD_NUM="${OPTARG}"
      if ! [[ "${THREAD_NUM}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: -j requires a positive integer, got '${THREAD_NUM}'"
        usage
        exit 1
      fi
      ;;
    h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option -${opt}"
      usage
      exit 1
      ;;
  esac
done

# ---------- Backend/test switches ----------
# A backend implies its target platform: nnrt = Kirin NPU, so the build
# cross-compiles for OHOS; host keeps the local x64 build for dev/testing.
# Only nnrt is wired up today; future backends (qnn/metal/...) plug in here
# with their own toolchain/default switches.
case "${BACKEND}" in
  host|nnrt) ;;
  *)
    echo "Error: unknown backend '${BACKEND}' (supported: host, nnrt)" >&2
    usage
    exit 1
    ;;
esac

# NNRT (Kirin NPU) is the only backend; default off for host development where
# the fake-backend unit tests run, on for the nnrt cross-compile.  Unit tests
# are enabled explicitly via -t (default off, matching the main repo build.sh).
if [[ "${BACKEND}" == "nnrt" ]]; then
  NNRT="${MSLITE_LLM_ENABLE_NNRT:-on}"
else
  NNRT="${MSLITE_LLM_ENABLE_NNRT:-off}"
fi

echo "---------------- lite_llm: build start ----------------"

echo "---------------- lite_llm: build start ----------------"
echo "  BUILD_TYPE : ${BUILD_TYPE}"
echo "  BACKEND    : ${BACKEND}"
echo "  NNRT       : ${NNRT}"
echo "  TESTS      : ${TESTS}"
echo "  THREAD_NUM : ${THREAD_NUM}"
echo "-------------------------------------------------------"

if [[ "${INC_BUILD}" == "off" ]]; then
  rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}" || exit 1
cd "${BUILD_DIR}" || exit 1

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DMSLITE_LLM_ENABLE_NNRT="${NNRT}"
  -DMSLITE_LLM_BUILD_TESTS="${TESTS}"
)

# Mirror the main repo's third-party download switch (the lite_llm build is
# a standalone cmake project; the main repo maps MSLITE_ENABLE_GITEE_MIRROR
# to ENABLE_GITEE in its own CMakeLists, which never runs here).
if [[ "${MSLITE_ENABLE_GITEE_MIRROR:-off}" == "on" ]]; then
  CMAKE_ARGS+=(-DENABLE_GITEE=on)
fi

if [[ "${BACKEND}" == "nnrt" ]]; then
  # OHOS_NDK / toolchain existence are validated at configure time in the
  # top-level CMakeLists.txt (MSLITE_LLM_ENABLE_NNRT=on derives the HarmonyOS
  # cross-compile and fails early with a clear message); here we just forward
  # the toolchain to CMake.
  CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE="${OHOS_NDK}/build/cmake/ohos.toolchain.cmake")
  # The provisioned third-party cache (gtest etc.) is keyed without the
  # toolchain, so a cross build would reuse the host-ABI artifacts.  Give the
  # cross build its own cache directory to force toolchain-matched builds.
  export MSLIBS_CACHE_PATH="${BUILD_DIR}/.mslib-cross"
fi

cmake "${CMAKE_ARGS[@]}" ..
cmake --build . -j"${THREAD_NUM}" ${VERBOSE_FLAG}

# ---------- Packaging ----------
# build_wheel(): the wheel reads its version from export/version.txt, the
# single source of truth (also used by the CMake engine build and the release
# archive name) — no copy needed.
build_wheel()
{
  rm -rf "${WHEEL_DIR}"
  mkdir -p "${WHEEL_DIR}" "${OUTPUT_DIR}/tool"
  # --no-build-isolation: reuse the local setuptools instead of downloading a
  # build environment; deps are declared in pyproject.toml for pip installs.
  python3 -m pip wheel "${TOP_DIR}/export" --no-deps --no-build-isolation \
    -w "${WHEEL_DIR}" >/dev/null
  cp -f "${WHEEL_DIR}"/*.whl "${OUTPUT_DIR}/tool/"
}

# Assemble the deployable archive without CPack: the C++ artifacts come from
# the install() rules (lib/ include/ bin/ via `cmake --install`), then the
# script collects the Python wheel and the AscendC operator .run packages
# into the same staging tree and tar-s it — three artifact kinds CPack cannot
# compose in one pass.
assemble_archive()
{
  local staging="${BUILD_DIR}/staging"
  rm -rf "${staging}"
  mkdir -p "${staging}" "${OUTPUT_DIR}"

  # C++ artifacts: engine .so/.a + headers + mslite-chat (install() rules).
  cmake --install "${BUILD_DIR}" --prefix "${staging}" >/dev/null

  # Python export wheel (built by build_wheel into output/tool).
  if compgen -G "${OUTPUT_DIR}/tool/"*.whl > /dev/null; then
    mkdir -p "${staging}/tool"
    cp -f "${OUTPUT_DIR}"/tool/*.whl "${staging}/tool/"
  fi

  # AscendC operator .run packages (CI-produced under custom_ops/output).
  if compgen -G "${TOP_DIR}/custom_ops/output/"*.run > /dev/null; then
    mkdir -p "${staging}/ascendc_ops"
    cp -f "${TOP_DIR}"/custom_ops/output/*.run "${staging}/ascendc_ops/"
  fi

  tar -C "${staging}" -czf "${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz" .
}

build_wheel
# Both host and device-backend builds produce the full deployable archive
# (runtime .so + headers + export wheel + CI .run packages); -b nnrt
# cross-compiles the .so for the Kirin NPU target while host produces the x64
# build.
assemble_archive

echo "---------------- lite_llm: build end   ----------------"
echo "  OUTPUT     : ${OUTPUT_DIR}"
