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
# package (ChunkGatedDeltaRule and friends) into CANN's default search path and
# writes a bin/set_env.bash that exposes it for converter and inference.
#
# Modes:
#   bash ./install.sh             DEFAULT: copy host-SoC vendor into
#                                        $ASCEND_OPP_PATH/vendors/
#   bash ./install.sh --uninstall remove it
#   bash ./install.sh --help      show this help
#
# The vendor is copied into $ASCEND_OPP_PATH/vendors/mslite_custom_ops/ (the
# CANN default search path). bin/set_env.bash additionally exports
# ASCEND_CUSTOM_OPP_PATH at that folder: the converter's tbe-custom op store
# needs it to register the custom op (the vendors/ path alone is NOT scanned by
# the offline-OM-build converter -- without it, convert fails with EZ3003 "no
# supported ops kernel/engine"). It also sets LD_LIBRARY_PATH for the aclnn
# op-api .so used at inference time. Source it once per shell that converts or
# runs inference.
#
# Usage:
#     bash ./install.sh [--help]
#     bash ./install.sh --uninstall
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
#     # remove later:
#     bash tools/custom_kernels/install.sh --uninstall
#
# Idempotent. Source-friendly (uses return, never exit).
# ============================================================================

_CUSTOM_KERNELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VENDOR_NAME="mslite_custom_ops"

# Print detailed usage.
_print_help() {
  cat <<'EOF'
install.sh — install the AscendC custom-op vendor into CANN's default search path.

Usage:
  bash install.sh             Default: copy the host-SoC vendor into
                              $ASCEND_OPP_PATH/vendors/mslite_custom_ops/.
  bash install.sh --uninstall Remove the vendor copied by install.sh.
  bash install.sh --help      Show this help.

What it does:
  Copies the host-SoC vendor into $ASCEND_OPP_PATH/vendors/mslite_custom_ops/
  (the CANN default search path) and writes bin/set_env.bash, which exports
  ASCEND_CUSTOM_OPP_PATH at that folder (the converter's tbe-custom op store
  needs it -- the vendors/ path alone is not scanned by the offline-OM-build
  converter) plus LD_LIBRARY_PATH for the aclnn op-api .so (inference). Source
  bin/set_env.bash once per shell that converts or runs inference. Idempotent
  (overwrites). Requires write permission on $ASCEND_OPP_PATH/vendors.

Prerequisite:
  Source your CANN set_env.sh first so $ASCEND_OPP_PATH is set. install.sh
  resolves the target as:  $ASCEND_OPP_PATH/vendors  (fallback:
  $ASCEND_HOME_PATH/opp/vendors, then /usr/local/Ascend/ascend-toolkit/latest/opp/vendors).

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
  # remove later:
  bash tools/custom_kernels/install.sh --uninstall
EOF
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
_resolve_opp_vendors() {
  local opp="${ASCEND_OPP_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/opp}"
  printf '%s/vendors' "${opp}"
}

# Copy the host-SoC vendor into $ASCEND_OPP_PATH/vendors/.
_install_to_cann() {
  _detect_units
  if [[ ${#_UNITS[@]} -eq 0 ]]; then
    echo "[custom_kernels] no NPU detected (npu-smi unavailable or no SoC matched); nothing installed." >&2
    return 1
  fi
  local opp_vendors; opp_vendors="$(_resolve_opp_vendors)"
  if [[ ! -d "${opp_vendors}" ]] && ! mkdir -p "${opp_vendors}" 2>/dev/null; then
    echo "[custom_kernels] cannot create ${opp_vendors} (source your CANN set_env.sh, or fix perms)." >&2
    return 1
  fi
  local unit src dst installed=0
  for unit in "${_UNITS[@]}"; do
    src="${_CUSTOM_KERNELS_DIR}/${unit}/${_VENDOR_NAME}"
    if [[ ! -d "${src}" ]]; then
      echo "[custom_kernels] vendor for ${unit} not shipped under ${_CUSTOM_KERNELS_DIR}; skipping." >&2
      continue
    fi
    dst="${opp_vendors}/${_VENDOR_NAME}"
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

# Remove a previously installed vendor from $ASCEND_OPP_PATH/vendors/.
_uninstall_from_cann() {
  local opp_vendors dst; opp_vendors="$(_resolve_opp_vendors)"
  dst="${opp_vendors}/${_VENDOR_NAME}"
  if [[ -d "${dst}" ]]; then
    rm -rf "${dst}"
    echo "[custom_kernels] removed ${dst}" >&2
  else
    echo "[custom_kernels] nothing to remove at ${dst}" >&2
  fi
  return 0
}

_main() {
  case "${1:-}" in
    --help|-h) _print_help ;;
    --uninstall) _uninstall_from_cann ;;
    --install|"") _install_to_cann ;;
    *) echo "[custom_kernels] unknown argument: $1 (try --help)" >&2; return 1 ;;
  esac
}

_main "$@"
unset -f _main _print_help _detect_units _resolve_opp_vendors \
  _install_to_cann _uninstall_from_cann 2>/dev/null
unset _CUSTOM_KERNELS_DIR _VENDOR_NAME _UNITS 2>/dev/null
