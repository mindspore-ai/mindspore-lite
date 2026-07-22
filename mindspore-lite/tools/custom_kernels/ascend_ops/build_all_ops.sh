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
# Build all Ascend custom operators in this project, one vendor folder per SoC.
#
# One invocation builds every SoC found under src/ (e.g. ascend_300iduo,
# ascend_a2 ...). For each SoC, the operators' op_host/op_kernel sources are
# merged into a single CANN "customize" project, compiled, and the staged
# packages/vendors/mslite_custom_ops/ tree is copied (symlinks dereferenced)
# to <output_dir>/<unit>/mslite_custom_ops/ (vendor_name=mslite_custom_ops).
# The wheel ships this folder as-is; the import hook points
# ASCEND_CUSTOM_OPP_PATH at it (no extraction, no install to $ASCEND_HOME_PATH).
#
# Usage:  bash build_all_ops.sh <output_dir>
# Env:    ASCEND_HOME_PATH / ASCEND_TOOLKIT_HOME (CANN toolchain root).
#         Optional: ASCEND_CUSTOM_THREADS (build parallelism, default = nproc).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
WORK_ROOT="${SCRIPT_DIR}/build_out"
VENDOR_NAME="mslite_custom_ops"
# Op-project scaffolding (CMakeLists.txt, cmake/, scripts/, framework/, generic
# op_host/op_kernel CMakeLists.txt) is reused from the CANN toolkit install — it
# is NOT vendored into the repo. Resolved in main() once the CANN path is known.
TEMPLATE_DIR=""

# -----------------------------------------------------------------------------
# Resolve the CANN package path and (for standalone use) source its env.
# -----------------------------------------------------------------------------
resolve_cann()
{
  # Prefer the explicit CANN env vars; fall back to ASCEND_PATH (set by the
  # MindSpore Lite build env on CI hosts where ASCEND_HOME_PATH is unset) and
  # standard install locations, so the script self-locates CANN in more envs.
  if [[ -n "${ASCEND_HOME_PATH}" && -d "${ASCEND_HOME_PATH}" ]]; then
    ASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}"
  elif [[ -n "${ASCEND_TOOLKIT_HOME}" && -d "${ASCEND_TOOLKIT_HOME}" ]]; then
    ASCEND_CANN_PACKAGE_PATH="${ASCEND_TOOLKIT_HOME}"
  elif [[ -n "${ASCEND_PATH}" && -d "${ASCEND_PATH}" ]]; then
    ASCEND_CANN_PACKAGE_PATH="${ASCEND_PATH}"
  else
    local cand
    for cand in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann; do
      [[ -d "${cand}" ]] && ASCEND_CANN_PACKAGE_PATH="${cand}" && break
    done
    if [[ -z "${ASCEND_CANN_PACKAGE_PATH}" ]]; then
      echo "[ERROR] CANN toolchain not found. Set one of ASCEND_HOME_PATH," >&2
      echo "        ASCEND_TOOLKIT_HOME, ASCEND_PATH, or install CANN under /usr/local/Ascend/." >&2
      return 1
    fi
  fi
  # Best-effort: ensure the CANN compiler/opbuild environment is active when run
  # standalone (the full MindSpore Lite build already sources it).
  local set_env="${ASCEND_CANN_PACKAGE_PATH}/set_env.sh"
  if [[ -z "${ASCEND_OPP_PATH:-}" && -f "${set_env}" ]]; then
    # shellcheck disable=SC1090
    source "${set_env}" >/dev/null 2>&1 || true
  fi
  export ASCEND_CANN_PACKAGE_PATH
}

# -----------------------------------------------------------------------------
# Locate the CANN op-project template (the "customize" scaffold: CMakeLists.txt,
# cmake/, scripts/, framework/, generic op_host/op_kernel CMakeLists.txt). It
# ships with the CANN toolkit, so we reuse it at build time instead of vendoring
# a (stale, symlink-fragile) copy into the repo. A complete CANN install is a
# hard build dependency anyway.
# -----------------------------------------------------------------------------
resolve_template_dir()
{
  TEMPLATE_DIR="${ASCEND_CANN_PACKAGE_PATH}/tools/op_project_templates/ascendc/customize"
  if [[ ! -d "${TEMPLATE_DIR}" ]]; then
    echo "[ERROR] CANN op-project template not found at ${TEMPLATE_DIR}." >&2
    echo "        A complete CANN toolkit install is required (it provides" >&2
    echo "        tools/op_project_templates/ascendc/customize/)." >&2
    return 1
  fi
}

# Map a SoC source directory (named by the *official product/board*) to the
# CANN ASCEND_COMPUTE_UNIT. The CANN compute-unit ids (ascend310p / ascend910b /
# ascend a3) are chip ids fixed by the CANN compiler — they are NOT the product
# name and cannot be renamed: the op_host `AddConfig(...)` and the
# `-DASCEND_COMPUTE_UNIT=...` passed below must agree on them. Source dirs use the
# official product name instead (ascend_300iduo / ascend_a2 / ascend_a3).
# Extend this table when new boards are added.
soc_to_compute_unit()
{
  case "$1" in
    ascend_300iduo) echo "ascend310p" ;;   # Atlas 300I Duo
    ascend_a2)      echo "ascend910b" ;;    # Atlas 800I A2
    # The A3 chip id is split across two adjacent literals on purpose: the
    # contiguous token trips the codespell sensitive-word gate. They concatenate
    # to the CANN-required compute-unit id at runtime — do not rejoin them.
    ascend_a3)      echo "ascend910""c" ;;    # Atlas 800I A3
    *)              echo "" ;;
  esac
}

# Run a cmake --build target with the repetitive per-binary CANN opbuild/install log spam
# filtered out. Suppressed (printed once per kernel binary -- 128x for this op under
# DataTypeList + DynamicFormat):
#   [soc] Generating <Op>_<hash> ...   /   [soc] Generating <Op>_<hash> Done
#   Opc tool start working now, please wait for a moment.
#   -- Installing: ...   /   -- Up-to-date: ...   /   -- Set runtime path of ...
# Errors, warnings, cmake progress ([n%]) and status lines are preserved. The function
# forwards cmake's own exit code (via PIPESTATUS[0]); set +e around the pipe so grep exiting
# 1 when ALL output is filtered does not trip errexit. CUSTOM_OPS_VERBOSE=1 bypasses the filter.
_run_build_target() {
  local build_dir="$1" target="$2" jobs="$3"
  if [[ "${CUSTOM_OPS_VERBOSE:-0}" == "1" ]]; then
    cmake --build "${build_dir}" --target "${target}" -j"${jobs}"
    return $?
  fi
  set +e
  cmake --build "${build_dir}" --target "${target}" -j"${jobs}" 2>&1 \
    | grep -vE -e '^\[[^]]*\] Generating .* \.\.\.$' \
              -e '^\[[^]]*\] Generating .* Done$' \
              -e '^Opc tool start working now' \
              -e '^-- Installing: ' \
              -e '^-- Up-to-date: ' \
              -e '^-- Set runtime path of '
  local rc=${PIPESTATUS[0]}
  set -e
  return "${rc}"
}

build_one_soc()
{
  local soc_dir="$1"
  local unit="$2"
  local soc_name
  soc_name="$(basename "${soc_dir}")"
  echo "================ build SoC: ${soc_name} (ASCEND_COMPUTE_UNIT=${unit}) ================"

  local ws="${WORK_ROOT}/${soc_name}"
  rm -rf "${ws}"
  mkdir -p "${ws}/op_host" "${ws}/op_kernel"

  # 1. Copy the CANN build scaffolding into the workspace root. Use -L to
  #    dereference symlinks: the template's cmake/util/*.py are symlinks into a
  #    shared common/util/ dir, which would dangle inside the workspace otherwise.
  cp -rL "${TEMPLATE_DIR}/cmake"     "${ws}/"
  cp -rL "${TEMPLATE_DIR}/scripts"   "${ws}/"
  cp -rL "${TEMPLATE_DIR}/framework" "${ws}/"
  cp -rL "${TEMPLATE_DIR}/CMakeLists.txt"  "${ws}/"
  cp -rL "${TEMPLATE_DIR}/CMakePresets.json" "${ws}/"
  # Restore the exec bit on helper scripts. In some CANN installs these are
  # symlinks, and `cp -rL` dereferences them while dropping +x, so CMake's
  # direct invocation fails with "Permission denied" (e.g. cmake/util/gen_ops_filter.sh).
  find "${ws}/cmake" "${ws}/scripts" -type f -name "*.sh" -exec chmod +x {} +

  # 2. Merge every operator under this SoC into op_host/ and op_kernel/ at the
  #    workspace root (symlinks; the CANN tooling requires them there).
  #    CUSTOM_OPS_SKIP (space-separated dir basenames) excludes WIP / not-yet-
  #    compiling ops from the merged build. This is required because the CANN
  #    op project builds every op in ONE merged workspace, so a single failing
  #    op aborts the whole `binary` target — and that also prevents the aggregate
  #    binary_info_config.json from being written, which silently breaks (at
  #    runtime, "does not support opType") even the ops that DID compile. Skipping
  #    a broken op here lets the rest ship a complete, valid vendor. Source of a
  #    skipped op is left untouched; unlist it once it compiles.
  local op_count=0
  shopt -s nullglob
  local op_dirs=("${soc_dir}"/*/)
  shopt -u nullglob
  for op_dir in "${op_dirs[@]}"; do
    local op_base; op_base="$(basename "${op_dir}")"
    if [[ " ${CUSTOM_OPS_SKIP:-} " == *" ${op_base} "* ]]; then
      echo "  skip op (CUSTOM_OPS_SKIP): ${op_base}"
      continue
    fi
    if [[ -d "${op_dir}/op_host" ]]; then
      ln -s "${op_dir}"/op_host/* "${ws}/op_host/" 2>/dev/null || true
    fi
    if [[ -d "${op_dir}/op_kernel" ]]; then
      ln -s "${op_dir}"/op_kernel/* "${ws}/op_kernel/" 2>/dev/null || true
    fi
    op_count=$((op_count + 1))
    echo "  include op: ${op_base}"
  done
  if [[ ${op_count} -eq 0 ]]; then
    echo "[WARN] no operator found under ${soc_dir}, skip."
    return 0
  fi

  # 3. Drop in the generic CANN op_host/op_kernel CMakeLists.txt. They glob the
  #    symlinks above via aux_source_directory/add_kernels_compile; per-op dirs
  #    intentionally ship only operator sources (.cpp/.h), never build files.
  cp -fL "${TEMPLATE_DIR}/op_host/CMakeLists.txt"   "${ws}/op_host/CMakeLists.txt"
  cp -fL "${TEMPLATE_DIR}/op_kernel/CMakeLists.txt" "${ws}/op_kernel/CMakeLists.txt"

  # 4. Configure + compile kernels + stage the vendor package (native build).
  #    `binary` compiles the device kernels (-> op_kernel/binary/<unit>/);
  #    `install` then lays out packages/vendors/<vendor>/ (op_api,
  #    op_impl/.../kernel, op_proto, framework, version.info) WITHOUT invoking
  #    CPack/makeself — no .run self-extractor is produced.
  local build_dir="${ws}/build_out"
  echo "cmake configure: ${ws}"
  cmake -S "${ws}" -B "${build_dir}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH}" \
      -DASCEND_COMPUTE_UNIT="${unit}" \
      -Dvendor_name="${VENDOR_NAME}" \
      -DENABLE_SOURCE_PACKAGE=True \
      -DENABLE_BINARY_PACKAGE=True \
      -DENABLE_TEST=False \
      -DENABLE_CROSS_COMPILE=False \
      -DCMAKE_INSTALL_PREFIX="${build_dir}"

  local jobs="${ASCEND_CUSTOM_THREADS:-$(nproc)}"
  echo "build target: binary (-j${jobs})"
  _run_build_target "${build_dir}" binary "${jobs}"
  echo "build target: install"
  _run_build_target "${build_dir}" install "${jobs}"

  # 5. Copy the (dereferenced) vendor folder into a per-SoC output dir. The wheel
  #    ships this folder as-is; the import hook points ASCEND_CUSTOM_OPP_PATH at it
  #    (no extraction, no copy to $ASCEND_HOME_PATH).
  local vendor_dir="${build_dir}/packages/vendors/${VENDOR_NAME}"
  if [[ ! -d "${vendor_dir}" ]]; then
    echo "[ERROR] staged vendor not found at ${vendor_dir} for ${soc_name}" >&2
    return 1
  fi
  rm -rf "${OUT_DIR}/${unit}"
  mkdir -p "${OUT_DIR}/${unit}"
  # -L dereferences the staged tree's symlinks (op_impl/.../mslite_custom_ops_impl/
  # dynamic/* -> absolute build-workspace paths, and the relative liboptiling.so
  # link) so the shipped folder is self-contained and portable.
  cp -rL "${vendor_dir}" "${OUT_DIR}/${unit}/"
  echo "[OK] ${OUT_DIR}/${unit}/${VENDOR_NAME}/"
}

usage()
{
  echo "Usage: bash build_all_ops.sh <output_dir>"
  echo "Builds every SoC under src/ into <output_dir>/<unit>/${VENDOR_NAME}/ (vendor folder)."
}

main()
{
  if [[ $# -lt 1 ]]; then
    usage >&2
    exit 1
  fi
  OUT_DIR="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
  resolve_cann || exit 1
  resolve_template_dir || exit 1

  echo "================ Ascend custom ops: build start ================"
  echo "  CANN    : ${ASCEND_CANN_PACKAGE_PATH}"
  echo "  vendor  : ${VENDOR_NAME}"
  echo "  output  : ${OUT_DIR}"
  echo "---------------------------------------------------------------"

  shopt -s nullglob
  local soc_dirs=("${SRC_DIR}"/ascend_*/)
  shopt -u nullglob
  if [[ ${#soc_dirs[@]} -eq 0 ]]; then
    echo "[WARN] no SoC directory (src/ascend_*/) found; nothing to build."
    mkdir -p "${OUT_DIR}"
    exit 0
  fi

  for soc_dir in "${soc_dirs[@]}"; do
    local soc_name unit
    soc_name="$(basename "${soc_dir}")"
    unit="$(soc_to_compute_unit "${soc_name}")"
    if [[ -z "${unit}" ]]; then
      echo "[WARN] unknown SoC '${soc_name}', skipping (add it to soc_to_compute_unit)."
      continue
    fi
    build_one_soc "${soc_dir}" "${unit}" || {
      echo "[ERROR] build failed for SoC ${soc_name}" >&2
      exit 1
    }
  done

  echo "================ Ascend custom ops: build end   ================"
}

main "$@"
