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

# ============================================================================
# Qwen3-8B common-prefix inference launcher (1P / 2P).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "${ASCEND_OPP_PATH:-}" ]; then
  echo "Error: ASCEND_OPP_PATH is unset; source the Ascend/CANN environment first." >&2
  exit 1
fi

# Runtime plumbing, not inference behavior. User-facing behavior is controlled
# exclusively by the command-line arguments forwarded below.
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_OPP_PATH}/vendors/mslite_custom_ops"
export HCCL_NPU_SOCKET_PORT_RANGE="21500-21600"
export PYTHONUNBUFFERED=1

if [ "$#" -eq 0 ]; then
  echo "Usage: bash infer.sh --device-ids IDS [options]"
  echo "Model directory: 1P=qwen3_8b_onnx, 2P=qwen3_8b_tp2_onnx, 4P=qwen3_8b_tp4_onnx"
  echo "Defaults: --model-id ./Qwen3-8B"
  echo "          --common-config-dir configs (1P) or configs/tpN (TP)"
  echo "          --common-prefix-text '你好，'"
  echo "          --suffix-prompt '请用一句话介绍一下你自己'"
  echo "          --max-new-tokens 64"
  exit 1
fi

DEVICE_IDS=""
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --device-id|--device-ids)
      if ((i + 1 >= ${#ARGS[@]})); then
        echo "Error: ${ARGS[$i]} requires a value." >&2
        exit 1
      fi
      DEVICE_IDS="${ARGS[$((i + 1))]}"
      ;;
    --device-id=*|--device-ids=*)
      DEVICE_IDS="${ARGS[$i]#*=}"
      ;;
  esac
done

if [ -z "$DEVICE_IDS" ]; then
  echo "Error: --device-id or --device-ids is required." >&2
  exit 1
fi

DEVICE_COUNT="$(awk -F, '{print NF}' <<< "$DEVICE_IDS")"
case "$DEVICE_COUNT" in
  1) COMMON_MODEL_DIR="./qwen3_8b_onnx" ;;
  2) COMMON_MODEL_DIR="./qwen3_8b_tp2_onnx" ;;
  *)
    echo "Error: device count must be 1, 2; got $DEVICE_COUNT." >&2
    exit 1
    ;;
esac

echo "=== ${DEVICE_COUNT}P model_dir=$COMMON_MODEL_DIR ==="
exec python3 infer_qwen3_8b_mslite_tp.py "$@" \
  --common-model-dir "$COMMON_MODEL_DIR"
