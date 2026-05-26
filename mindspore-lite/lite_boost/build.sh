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

set -e

TOP_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${TOP_DIR}/../.." && pwd)"
BUILD_DIR="${TOP_DIR}/build"
PYTHON_DIR="${TOP_DIR}/python"
DIST_DIR="${PYTHON_DIR}/dist"
OUTPUT_DIR="${ROOT_DIR}/output"
LITE_BOOST_OUTPUT_DIR="${TOP_DIR}/output"

# Defaults
THREAD_NUM=8
BUILD_TYPE="Release"
VERBOSE_FLAG=""
INC_BUILD="off"

usage()
{
  echo "Usage:"
  echo "bash build.sh [-d] [-r] [-v] [-i] [-j[n]] [-h]"
  echo ""
  echo "Options:"
  echo "    -d       Debug mode"
  echo "    -r       Release mode (default)"
  echo "    -v       Display build commands"
  echo "    -i       Enable incremental build (do not clean build directory)"
  echo "    -j[n]    Set the number of build threads (default: -j8)"
  echo "    -h       Print usage"
  echo ""
  echo "Environment variables:"
  echo "    ENABLE_GLIBCXX    Set ON to use CXX11 ABI=1, OFF to use ABI=0 (default: ON)"
}

# Parse arguments
while getopts 'drivj:h' opt
do
  case "${opt}" in
    d)
      BUILD_TYPE="Debug" ;;
    r)
      BUILD_TYPE="Release" ;;
    v)
      VERBOSE_FLAG="VERBOSE=1" ;;
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

# ---------- Environment validation ----------
check_env()
{
  local has_error=0

  # CANN
  if [[ -z "${ASCEND_HOME_PATH}" && -z "${ASCEND_TOOLKIT_HOME}" ]]; then
    echo "Error: CANN environment not found. Please set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME."
    has_error=1
  elif [[ -n "${ASCEND_HOME_PATH}" && ! -d "${ASCEND_HOME_PATH}" ]]; then
    echo "Error: ASCEND_HOME_PATH='${ASCEND_HOME_PATH}' is not a valid directory."
    has_error=1
  elif [[ -z "${ASCEND_HOME_PATH}" && -n "${ASCEND_TOOLKIT_HOME}" && ! -d "${ASCEND_TOOLKIT_HOME}" ]]; then
    echo "Error: ASCEND_TOOLKIT_HOME='${ASCEND_TOOLKIT_HOME}' is not a valid directory."
    has_error=1
  fi

  if [[ ${has_error} -eq 1 ]]; then
    exit 1
  fi
}

check_env

echo "---------------- Lite Boost: build start ----------------"
echo "  BUILD_TYPE : ${BUILD_TYPE}"
echo "  THREAD_NUM : ${THREAD_NUM}"
echo "  VERBOSE    : ${VERBOSE_FLAG:-off}"
echo "  INC_BUILD  : ${INC_BUILD}"
echo "---------------------------------------------------------"

if [[ "${INC_BUILD}" == "off" ]]; then
  rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}" || exit 1
cd "${BUILD_DIR}" || exit 1

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE="${BUILD_TYPE}")

if [[ -n "${ASCEND_PATH:-}" ]]; then
  CMAKE_ARGS+=(-DASCEND_PATH="${ASCEND_PATH}")
fi
if [[ -n "${PYTORCH_INSTALL_PATH:-}" ]]; then
  CMAKE_ARGS+=(-DPYTORCH_INSTALL_PATH="${PYTORCH_INSTALL_PATH}")
fi
if [[ -n "${PYTORCH_NPU_INSTALL_PATH:-}" ]]; then
  CMAKE_ARGS+=(-DPYTORCH_NPU_INSTALL_PATH="${PYTORCH_NPU_INSTALL_PATH}")
fi
if [[ -n "${Python3_EXECUTABLE:-}" ]]; then
  CMAKE_ARGS+=(-DPython3_EXECUTABLE="${Python3_EXECUTABLE}")
fi
if [[ -n "${CXX_STANDARD:-}" ]]; then
  CMAKE_ARGS+=(-DCXX_STANDARD="${CXX_STANDARD}")
fi
if [[ -n "${ENABLE_GLIBCXX:-}" ]]; then
  CMAKE_ARGS+=(-DENABLE_GLIBCXX="${ENABLE_GLIBCXX}")
fi

cmake "${CMAKE_ARGS[@]}" ..
make -j"${THREAD_NUM}" ${VERBOSE_FLAG}

cd "${PYTHON_DIR}"
rm -rf "${DIST_DIR}"
python setup.py bdist_wheel "${TOP_DIR}"

mkdir -p "${OUTPUT_DIR}" || exit 1
mkdir -p "${LITE_BOOST_OUTPUT_DIR}" || exit 1
cp -f "${DIST_DIR}"/*.whl "${OUTPUT_DIR}/" || exit 1
cp -f "${DIST_DIR}"/*.whl "${LITE_BOOST_OUTPUT_DIR}/" || exit 1

echo "---------------- Lite Boost: build end   ----------------"
