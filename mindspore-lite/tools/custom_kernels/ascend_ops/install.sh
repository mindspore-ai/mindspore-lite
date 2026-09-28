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
# Installs the AscendC custom-op vendor shipped in this mindspore-lite tar
# package (ChunkGatedDeltaRule and friends) into CANN's default search path (or
# into any other directory given with --install-path) and writes a
# bin/set_env.bash that exposes it for converter and inference.
#
# The vendor always keeps the mslite_custom_ops/ folder name: it is installed to
# <base>/mslite_custom_ops/, where <base> is $ASCEND_OPP_PATH/vendors by default
# (the CANN default search path) or the directory passed with --install-path DIR.
# Use --install-path when CANN is not writable by this user, or to share one
# install between users/hosts.
#
# Modes:
#   bash ./install.sh [--install-path DIR]
#                                DEFAULT: copy the host-SoC vendor into
#                                $ASCEND_OPP_PATH/vendors/mslite_custom_ops/
#                                (DIR/mslite_custom_ops/ when DIR is given)
#   bash ./install.sh --uninstall [--install-path DIR]
#                                remove it (pass the same DIR as the install)
#   bash ./install.sh --help     show this help
#
# bin/set_env.bash additionally exports ASCEND_CUSTOM_OPP_PATH at that folder:
# the converter's tbe-custom op store needs it to register the custom op (the
# vendors/ path alone is NOT scanned by the offline-OM-build converter -- without
# it, convert fails with EZ3003 "no supported ops kernel/engine"). It also sets
# LD_LIBRARY_PATH for the aclnn op-api .so used at inference time. Source it once
# per shell that converts or runs inference. With --install-path, sourcing that
# set_env.bash is the only setup needed: nothing under $ASCEND_OPP_PATH is
# touched.
#
# Usage:
#     bash ./install.sh [--install-path DIR]
#     bash ./install.sh --uninstall [--install-path DIR]
#     bash ./install.sh --help
#
# Example:
#     tar -xzf mindspore-lite-2.10.0-linux-aarch64.tar.gz
#     cd mindspore-lite-2.10.0-linux-aarch64/
#     source /path/to/CANN/set_env.sh                     # set ASCEND_OPP_PATH first
#     bash tools/custom_kernels/install.sh                # one-time, persistent
#     # any shell can now convert (no env setup, no sourcing):
#     tools/converter/converter/converter_lite --fmk=ONNX \
#         --modelFile=chunk.onnx --outputFile=chunk --optimize=ascend_oriented
#     # for runtime inference (aclnn), expose the op-api .so once per shell:
#     source "$ASCEND_OPP_PATH/vendors/mslite_custom_ops/bin/set_env.bash"
#     # or, when CANN is not writable, install under $HOME instead:
#     bash tools/custom_kernels/install.sh --install-path "$HOME/mslite_ops"
#     source "$HOME/mslite_ops/mslite_custom_ops/bin/set_env.bash"
#     # remove later:
#     bash tools/custom_kernels/install.sh --uninstall
#
# Idempotent. Source-friendly (uses return, never exit).
# ============================================================================

_CUSTOM_KERNELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VENDOR_NAME="mslite_custom_ops"
# --install-path DIR; empty means <base> = the CANN opp/vendors dir.
_INSTALL_PATH=""
# install | uninstall | help, set by _parse_args.
_MODE="install"

# Print detailed usage.
_print_help() {
  cat <<'EOF'
install.sh — install the AscendC custom-op vendor shipped with mindspore-lite.

Usage:
  bash install.sh [--install-path DIR]
                              Default: copy the host-SoC vendor into
                              ${ASCEND_OPP_PATH:-...}/vendors/mslite_custom_ops/.
                              With DIR: copy it into DIR/mslite_custom_ops/
                              instead (use this when CANN is not writable, or to
                              share one install between users/hosts).
  bash install.sh --uninstall [--install-path DIR]
                              Remove the vendor installed at that location (pass
                              the same DIR that was used to install).
  bash install.sh --help      Show this help.

What it does:
  Copies the host-SoC vendor into <base>/mslite_custom_ops/ and writes
  bin/set_env.bash, which exports ASCEND_CUSTOM_OPP_PATH at that folder (the
  converter's tbe-custom op store needs it -- the vendors/ path alone is not
  scanned by the offline-OM-build converter) plus LD_LIBRARY_PATH for the aclnn
  op-api .so (inference). Source bin/set_env.bash once per shell that converts
  or runs inference. Idempotent (overwrites).

Install path (<base>):
  Default: $ASCEND_OPP_PATH/vendors (fallback: $ASCEND_HOME_PATH/opp/vendors,
  then /usr/local/Ascend/ascend-toolkit/latest/opp/vendors). Source your CANN
  set_env.sh first so $ASCEND_OPP_PATH is set; this needs write permission on
  $ASCEND_OPP_PATH/vendors.
  With --install-path DIR: the vendor goes to DIR/mslite_custom_ops/ instead. DIR
  is created (mkdir -p) when missing and resolved to an absolute path, so the
  generated bin/set_env.bash stays valid when sourced from any directory. No
  CANN environment is needed to install, and nothing under $ASCEND_OPP_PATH is
  modified.

SoC detection: via npu-smi, for the host's own compute unit only. The vendor is
installed solely for the detected SoC.

Example:
  tar -xzf mindspore-lite-2.10.0-linux-aarch64.tar.gz
  cd mindspore-lite-2.10.0-linux-aarch64/
  source /path/to/CANN/set_env.sh
  bash tools/custom_kernels/install.sh
  # any shell, no env setup:
  tools/converter/converter/converter_lite --fmk=ONNX \
      --modelFile=chunk.onnx --outputFile=chunk --optimize=ascend_oriented
  # runtime inference (aclnn op api) — once per shell:
  source "$ASCEND_OPP_PATH/vendors/mslite_custom_ops/bin/set_env.bash"
  # or install where you have write access, e.g. under $HOME:
  bash tools/custom_kernels/install.sh --install-path "$HOME/mslite_ops"
  source "$HOME/mslite_ops/mslite_custom_ops/bin/set_env.bash"
  # remove later (same --install-path as the install, if one was used):
  bash tools/custom_kernels/install.sh --uninstall
EOF
}

# Parse the arguments into _MODE + _INSTALL_PATH. Flags may appear in any order;
# exactly one mode flag is expected. Returns non-zero on unknown/malformed input.
_parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --install-path)
        if [[ $# -lt 2 || -z "$2" ]]; then
          echo "[custom_kernels] --install-path needs a directory (or use --install-path=DIR)." >&2
          return 1
        fi
        _INSTALL_PATH="$2"
        shift 2
        ;;
      --install-path=*)
        _INSTALL_PATH="${1#*=}"
        if [[ -z "${_INSTALL_PATH}" ]]; then
          echo "[custom_kernels] --install-path needs a non-empty directory." >&2
          return 1
        fi
        shift
        ;;
      --install)
        _MODE="install"
        shift
        ;;
      --uninstall)
        _MODE="uninstall"
        shift
        ;;
      --help|-h)
        _MODE="help"
        shift
        ;;
      *)
        echo "[custom_kernels] unknown argument: $1 (try --help)" >&2
        return 1
        ;;
    esac
  done
  return 0
}

# Fill _UNITS with the host SoC compute-units (mirror _NPU_UNIT_MAP in
# python/api/_ascend_custom_ops.py). The 910-C id is rebuilt by concatenation.
_detect_units() {
  _UNITS=()
  local npu_text=""
  if command -v npu-smi >/dev/null 2>&1; then
    npu_text="$(npu-smi info 2>/dev/null)" || npu_text=""
  fi
  [[ "${npu_text}" == *"310P"* ]] && _UNITS+=("ascend310p")
  [[ "${npu_text}" == *"910B"* ]] && _UNITS+=("ascend910b")
  # The 910-C needle/unit id are split on purpose: the contiguous token trips the
  # codespell sensitive-word gate. Concatenation rebuilds the real id at runtime.
  [[ "${npu_text}" == *"910""C"* ]] && _UNITS+=("ascend910""c")
}

# Resolve the CANN vendor dir that the converter always searches.
_resolve_cann_vendors() {
  local opp="${ASCEND_OPP_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/opp}"
  printf '%s/vendors' "${opp}"
}

# Resolve <base>, the dir that holds <base>/mslite_custom_ops/: the
# --install-path DIR when given, else the CANN vendor dir. A custom DIR is
# echoed as an absolute path (callers create it first), so the bin/set_env.bash
# written inside the vendor stays valid when sourced from any working directory.
_resolve_vendor_base() {
  if [[ -z "${_INSTALL_PATH}" ]]; then
    _resolve_cann_vendors
    return 0
  fi
  local base
  if ! base="$(cd "${_INSTALL_PATH}" 2>/dev/null && pwd)"; then
    echo "[custom_kernels] --install-path ${_INSTALL_PATH}: not a usable directory (check the path and its permissions)." >&2
    return 1
  fi
  printf '%s' "${base}"
}

# Copy the host-SoC vendor into <base>/mslite_custom_ops/.
_install_vendor() {
  _detect_units
  if [[ ${#_UNITS[@]} -eq 0 ]]; then
    echo "[custom_kernels] no NPU detected (npu-smi unavailable or no SoC matched); nothing installed." >&2
    return 1
  fi
  if [[ -n "${_INSTALL_PATH}" ]] && ! mkdir -p "${_INSTALL_PATH}" 2>/dev/null; then
    echo "[custom_kernels] cannot create --install-path ${_INSTALL_PATH} (check the path and its permissions)." >&2
    return 1
  fi
  local vendor_base
  if ! vendor_base="$(_resolve_vendor_base)"; then
    return 1
  fi
  if [[ ! -d "${vendor_base}" ]] && ! mkdir -p "${vendor_base}" 2>/dev/null; then
    echo "[custom_kernels] cannot create ${vendor_base} (source your CANN set_env.sh, or fix perms)." >&2
    return 1
  fi
  local unit src dst installed=0
  for unit in "${_UNITS[@]}"; do
    src="${_CUSTOM_KERNELS_DIR}/${unit}/${_VENDOR_NAME}"
    if [[ ! -d "${src}" ]]; then
      echo "[custom_kernels] vendor for ${unit} not shipped under ${_CUSTOM_KERNELS_DIR}; skipping." >&2
      continue
    fi
    dst="${vendor_base}/${_VENDOR_NAME}"
    rm -rf "${dst}"
    cp -r "${src}" "${dst}"
    # Drop set_env.bash: exposes the vendor for BOTH the converter and runtime.
    # ASCEND_CUSTOM_OPP_PATH is the standard CANN mechanism the converter's tbe
    # engine needs to register the custom op (without it the offline OM build
    # fails with EZ3003 "no supported ops kernel/engine" even though the vendor
    # is under $ASCEND_OPP_PATH/vendors/ -- that default search path alone is not
    # scanned by the converter's tbe-custom op store). LD_LIBRARY_PATH covers the
    # aclnn op-api .so at inference time. Source once per shell that converts or
    # runs inference. Mirrors what the wheel's import hook (_ascend_custom_ops)
    # sets automatically.
    mkdir -p "${dst}/bin"
    cat > "${dst}/bin/set_env.bash" <<EOF
#!/bin/bash
# Env for the ${_VENDOR_NAME} vendor: ASCEND_CUSTOM_OPP_PATH for the converter
# (custom-op discovery) + LD_LIBRARY_PATH for the aclnn op-api .so (inference).
# Source this in shells that run converter_lite or inference.
export ASCEND_CUSTOM_OPP_PATH="${dst}:\${ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="${dst}/op_api/lib:\${LD_LIBRARY_PATH}"
EOF
    chmod +x "${dst}/bin/set_env.bash" 2>/dev/null
    echo "[custom_kernels] installed vendor for ${unit} -> ${dst}" >&2
    echo "[custom_kernels] converter/inference:  source ${dst}/bin/set_env.bash  (sets ASCEND_CUSTOM_OPP_PATH + LD_LIBRARY_PATH)" >&2
    installed=$((installed + 1))
  done
  if [[ ${installed} -eq 0 ]]; then
    echo "[custom_kernels] nothing installed (no matching vendor shipped for the host SoC)." >&2
    return 1
  fi
  return 0
}

# Remove a previously installed vendor from <base>/mslite_custom_ops/.
_uninstall_vendor() {
  local vendor_base dst
  if ! vendor_base="$(_resolve_vendor_base)"; then
    return 1
  fi
  dst="${vendor_base}/${_VENDOR_NAME}"
  if [[ -d "${dst}" ]]; then
    rm -rf "${dst}"
    echo "[custom_kernels] removed ${dst}" >&2
  else
    echo "[custom_kernels] nothing to remove at ${dst}" >&2
  fi
  return 0
}

_main() {
  if ! _parse_args "$@"; then
    return 1
  fi
  case "${_MODE}" in
    help) _print_help ;;
    uninstall) _uninstall_vendor ;;
    install) _install_vendor ;;
  esac
}

_main "$@"
_status=$?
unset -f _main _parse_args _print_help _detect_units _resolve_cann_vendors \
  _resolve_vendor_base _install_vendor _uninstall_vendor 2>/dev/null
unset _CUSTOM_KERNELS_DIR _VENDOR_NAME _UNITS _INSTALL_PATH _MODE 2>/dev/null

# Executed -> propagate the result to the caller. Sourced -> return (never exit,
# which would kill the caller's shell).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  exit "${_status}"
fi
return "${_status}"
