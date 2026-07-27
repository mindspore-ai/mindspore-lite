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
# Qwen2.5-7B inference launcher — auto-selects single-chip / TP=2 / TP=4
# based on the number of device IDs passed.
#
# Usage:
#   bash infer.sh 2          → 单卡（device 2，single-chip zero-copy）
#   bash infer.sh 2,3        → TP=2（devices 2,3，单卡双芯 HCCS）
#   bash infer.sh 2,3,4,5    → TP=4（devices 2,3,4,5，两卡四芯）
#   bash infer.sh             → 默认 0,1（TP=2）
#
# Assumes models already exported + converted (see export_and_convert.sh)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /home/yf/env.sh
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_OPP_PATH}/vendors/mslite_custom_ops
export HCCL_NPU_SOCKET_PORT_RANGE=20500-20600
export PYTHONUNBUFFERED=1

# ---- Parse device IDs from $1 ----
if [ -z "${1:-}" ]; then
  echo "错误：缺少参数。请指定要使用的设备 ID。"
  echo ""
  echo "用法："
  echo "  bash infer.sh <设备ID列表>"
  echo ""
  echo "示例："
  echo "  bash infer.sh 2          # 单卡（device 2）"
  echo "  bash infer.sh 2,3        # 双卡 TP=2（devices 2,3）"
  echo "  bash infer.sh 2,3,4,5    # 四卡 TP=4（devices 2,3,4,5）"
  echo ""
  echo "参数格式：设备 ID 用逗号分隔（无空格），数量决定并行度（1=单卡, 2=TP2, 4=TP4）。"
  exit 1
fi
DEVICE_IDS="$1"
IFS=',' read -ra DEVS <<< "$DEVICE_IDS"
TP_SIZE=${#DEVS[@]}

# ---- Common config ----
PROMPT="你好，请用一句话介绍一下你自己"
MAX_TOKENS=64
WARMUP=3
MODEL_ID="./Qwen2.5-7B-Instruct"

echo "=== TP_SIZE=$TP_SIZE  devices=$DEVICE_IDS ==="

# ============================================================================
# Single-chip path (TP_SIZE == 1): no HCCL, zero-copy decode
# ============================================================================
if [ "$TP_SIZE" -eq 1 ]; then
  DEV="${DEVS[0]}"
  PREFILL_MODEL="./qwen2_5_7b_onnx/prefill/qwen2_5_7b_llm_prefill_rank0_graph.mindir"
  DECODE_MODEL="./qwen2_5_7b_onnx/decode/qwen2_5_7b_llm_decode_rank0_graph.mindir"

  echo "=== single-chip (device $DEV, zero-copy) ==="
  python3 infer_qwen2_5_7b_mslite.py \
    --device-ids "$DEV" \
    --prefill-model "$PREFILL_MODEL" \
    --decode-model  "$DECODE_MODEL" \
    --model-id "$MODEL_ID" \
    --prompt "$PROMPT" \
    --max-new-tokens "$MAX_TOKENS"
  exit 0
fi

# ============================================================================
# TP path (TP_SIZE >= 2): HCCL, multi-process
# ============================================================================

# Auto-select model dir based on TP_SIZE.
# 2p/4p base export is static seq=64 (online GE dynamic prefill fails with
# aicore error), matches the default chat prompt (~35 tokens → bucket 64).
# 4p decode is full export (DECODE_CHUNK_SIZE=999); both prefill & decode in base.
# For multi-seq 2p/4p perf sweep: SEQ_LIST=32,64,128 bash export_and_convert.sh 2p|4p → _seqN dirs.
if [ "$TP_SIZE" -eq 4 ]; then
  PF_DIR="./qwen2_5_7b_tp4_onnx"
  DC_DIR="./qwen2_5_7b_tp4_onnx"
else
  PF_DIR="./qwen2_5_7b_tp_onnx"
  DC_DIR="./qwen2_5_7b_tp_onnx"
fi

# Build comma-separated rank paths from TP_SIZE
PF_RANKS=""
DC_RANKS=""
for R in $(seq 0 $((TP_SIZE - 1))); do
  [ -n "$PF_RANKS" ] && PF_RANKS="$PF_RANKS,"
  [ -n "$DC_RANKS" ] && DC_RANKS="$DC_RANKS,"
  PF_RANKS="$PF_RANKS$PF_DIR/prefill/qwen2_5_7b_llm_prefill_rank${R}_graph.mindir"
  DC_RANKS="$DC_RANKS$DC_DIR/decode/qwen2_5_7b_llm_decode_rank${R}_graph.mindir"
done

# Generate rank_table + config (from actual device IDs)
RUN_DIR="$SCRIPT_DIR/tp_run"
mkdir -p "$RUN_DIR"
DEVICES_JSON=""
for IDX in "${!DEVS[@]}"; do
  DEV="${DEVS[$IDX]}"
  [ -n "$DEVICES_JSON" ] && DEVICES_JSON="$DEVICES_JSON,"
  DEVICES_JSON="$DEVICES_JSON{\"device_id\":\"$DEV\",\"rank_id\":\"$IDX\"}"
done
cat > "$RUN_DIR/rank_table.json" <<EOF
{"version":"1.0","server_count":"1","server_list":[
  {"server_id":"127.0.0.1","device":[$DEVICES_JSON],
   "host_nic_ip":"reserve"}],"status":"completed"}
EOF
cat > "$RUN_DIR/config_file.ini" <<EOF
[ascend_context]
rank_table_file=$RUN_DIR/rank_table.json
plugin_custom_ops=All
EOF

echo "=== TP=$TP_SIZE inference (devices $DEVICE_IDS, prefill=$PF_DIR decode=$DC_DIR) ==="
python3 infer_qwen2_5_7b_mslite.py \
  --device-ids "$DEVICE_IDS" \
  --prefill-ranks "$PF_RANKS" \
  --decode-ranks  "$DC_RANKS" \
  --model-id "$MODEL_ID" \
  --config-file "$RUN_DIR/config_file.ini" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_TOKENS" \
  --warmup "$WARMUP" \
  --tp-size "$TP_SIZE"
