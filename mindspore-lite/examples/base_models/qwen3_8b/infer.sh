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
# Qwen3-8B inference launcher - dispatches single-chip / TP=2 / TP=4 by device
# count. The Python entry (infer_qwen3_8b_mslite.py) auto-resolves model paths
# and generates the HCCL rank-table for TP, so this wrapper just forwards args.
#
# Usage:
#   bash infer.sh 2            -> Single-chip (device 2, zero-copy decode)
#   bash infer.sh 2,3          -> TP=2 (devices 2,3, single-card dual-chip HCCS)
#   bash infer.sh 2,3,4,5      -> TP=4 (devices 2,3,4,5, dual-card quad-chip)
#
# Assumes models already exported + converted (see export_and_convert.sh)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export ASCEND_CUSTOM_OPP_PATH=${ASCEND_OPP_PATH}/vendors/mslite_custom_ops
export HCCL_NPU_SOCKET_PORT_RANGE=21500-21600   # TP>=2 needs an HCCL port range
export PYTHONUNBUFFERED=1

# ---- Parse device IDs from $1 ----
if [ -z "${1:-}" ]; then
  echo "Error: missing argument. Please specify device IDs."
  echo ""
  echo "Usage:"
  echo "  bash infer.sh <device_id_list>"
  echo ""
  echo "Examples:"
  echo "  bash infer.sh 2          # Single-chip (device 2)"
  echo "  bash infer.sh 2,3        # TP=2 (devices 2,3)"
  echo "  bash infer.sh 2,3,4,5    # TP=4 (devices 2,3,4,5)"
  echo ""
  echo "Format: comma-separated device IDs (no spaces). Count determines parallelism (1=single, 2=TP2, 4=TP4)."
  exit 1
fi
DEVICE_IDS="$1"

# ---- Common config ----
PROMPT="Hello, please introduce yourself in one sentence."
MAX_TOKENS=64
WARMUP=3
MODEL_ID="${MODEL_ID:-./Qwen3-8B}"

echo "=== devices=$DEVICE_IDS ==="
python3 infer_qwen3_8b_mslite.py \
  --device-ids "$DEVICE_IDS" \
  --model-id "$MODEL_ID" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_TOKENS" \
  --warmup "$WARMUP"
