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
# count. All paths use the GE provider (provider=ge, optimize=none convert).
#
# Shape plan:
#   * 1p (1 device)   -> dynamic single-graph bucketing via configs/ge_*.cfg
#                        (KV dim3 = 8 heads); see infer_qwen3_8b_mslite_1p.py for the
#                        benchmark loop (infer.sh 1p does a single inference).
#   * 2p (2 devices)  -> dynamicDims bucketing via configs/tp2/ge_*.cfg
#                        (KV dim3 = 4 heads per rank), HCCL multi-process.
#   * 4p (4 devices)  -> code path exists (configs/tp4/, KV dim3 = 2) but NOT
#                        validated on this box (known 4p HCCL precision issue
#                        on 300I Duo PCIe topology).
# The class infer_qwen3_8b_mslite_tp.py auto-picks the cfg dir from --device-ids
# count (1 -> configs/, 2 -> configs/tp2, 4 -> configs/tp4).
#
# Usage:
#   bash infer.sh 0            -> Single-chip (device 0)
#   bash infer.sh 0,1          -> TP=2 (devices 0,1, single-card dual-chip HCCS)
#   bash infer.sh 0,1,2,3      -> TP=4 (devices 0-3)
#
# Assumes models already exported + converted (see export_and_convert.sh)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export ASCEND_CUSTOM_OPP_PATH=${ASCEND_OPP_PATH}/vendors/mslite_custom_ops
export HCCL_NPU_SOCKET_PORT_RANGE=21500-21600   # TP>=2 needs an HCCL port range
export PYTHONUNBUFFERED=1

# ---- Parse device IDs ($1) and mode ($2) ----
if [ -z "${1:-}" ]; then
  echo "Error: missing argument. Please specify device IDs."
  echo ""
  echo "Usage:"
  echo "  bash infer.sh <device_id_list> [mode]"
  echo ""
  echo "Modes:"
  echo "  infer  (default)  精度/功能验证：单 prompt -> 命中一档 -> 输出 + 核心点 + 显存峰值"
  echo "  perf              性能验证：六档扫描 (500/1000/1600/2000/2800/3072 tokens),"
  echo "                    逐档 prefill/decode 计时 + 显存峰值 + 核心点 6/6 汇总表"
  echo "                    (1p 走 infer_qwen3_8b_mslite_1p.py 动态单图; 2p/4p 走 --perf-sweep 单进程多档)"
  echo "  prof [buckets]    prof 采集：用 msprof 包装, 逐档 3 次 warmup + 1 次采集"
  echo "                    可选第三参数指定档位 (逗号分隔, 默认全 6 档: 512,1024,1664,2048,2816,3072)"
  echo ""
  echo "Examples:"
  echo "  bash infer.sh 0            # Single-chip (device 0), 单档精度验证"
  echo "  bash infer.sh 0 perf       # 1p 六档性能扫描 (动态单图 + KV 切片)"
  echo "  bash infer.sh 0,1          # TP=2 (devices 0,1, cfg auto: configs/tp2), 单档精度验证"
  echo "  bash infer.sh 0,1 perf     # TP=2 六档性能扫描 (单进程多档, worker 复用)"
  echo "  bash infer.sh 0,1,2,3      # TP=4 (devices 0-3, cfg auto: configs/tp4, unvalidated)"
  echo "  bash infer.sh 0 prof            # 1p 全6档 prof 采集"
  echo "  bash infer.sh 0 prof 1024       # 1p 只采集 1024 档 (prefill seq=1024 + decode kv=1536)"
  echo "  bash infer.sh 0 prof 512,2048   # 1p 采集 512 和 2048 两档"
  echo "  bash infer.sh 0,1 prof 1024     # 2p 采集 1024 档"
  echo ""
  echo "Format: comma-separated device IDs (no spaces). Count determines parallelism (1=single, 2=TP2, 4=TP4)."
  echo ""
  echo "GE dynamicDims cfg dirs (6-bucket seq/KV):"
  echo "  * 1p cfg directory  = configs/           -> KV dim3=8 heads"
  echo "  * 2p cfg directory  = configs/tp2/       -> KV dim3=4 heads per rank"
  echo "  * 4p cfg directory  = configs/tp4/       -> KV dim3=2 heads per rank"
  echo "  Override the auto cfg_dir with:  python3 infer_qwen3_8b_mslite_tp.py ... --bucket-cfg-dir <dir>"
  exit 1
fi
DEVICE_IDS="$1"
MODE="${2:-infer}"
if [ "$MODE" != "infer" ] && [ "$MODE" != "perf" ] && [ "$MODE" != "prof" ]; then
  echo "Error: unknown mode '$MODE' (expected 'infer', 'perf', or 'prof')"; exit 1
fi

# ---- Common config ----
PROMPT="Hello, please introduce yourself in one sentence."
MAX_TOKENS=64
WARMUP=3
MODEL_ID="${MODEL_ID:-./Qwen3-8B}"

if [ "$MODE" = "infer" ]; then
  # ================= 精度/功能验证：单 prompt -> 命中一档 =================
  _DEV_COUNT="$(echo "$DEVICE_IDS" | awk -F, '{print NF}')"

  if [ "$_DEV_COUNT" -eq 1 ]; then
    # 1p: 动态单图功能验证 (infer_qwen3_8b_mslite_1p.py --single-prompt)
    echo "=== [infer] 1p 动态单图功能验证  device=$DEVICE_IDS ==="
    python3 infer_qwen3_8b_mslite_1p.py \
      --device-id "$DEVICE_IDS" \
      --single-prompt "$PROMPT" \
      --max-new-tokens "$MAX_TOKENS"
  else
    echo "=== [infer] devices=$DEVICE_IDS ==="
    python3 infer_qwen3_8b_mslite_tp.py \
      --device-ids "$DEVICE_IDS" \
      --model-id "$MODEL_ID" \
      --prompt "$PROMPT" \
      --max-new-tokens "$MAX_TOKENS" \
      --warmup "$WARMUP"
  fi

  exit 0
fi

# ================= prof 采集：msprof 包装, 逐档 3 warmup + 1 采集 =================
if [ "$MODE" = "prof" ]; then
  # Check msprof availability
  if ! command -v msprof &>/dev/null; then
    echo "Error: msprof not found in PATH. Please source CANN env (set_env.sh) or install Ascend profiling tools."
    exit 1
  fi

  # Parse bucket list (optional 3rd arg; default all 6)
  PROF_BUCKETS_ARG="${3:-}"
  if [ -z "$PROF_BUCKETS_ARG" ]; then
    PROF_BUCKETS_ARG="512,1024,1664,2048,2816,3072"
  fi
  IFS=',' read -ra PROF_BUCKET_LIST <<< "$PROF_BUCKETS_ARG"

  _DEV_COUNT="$(echo "$DEVICE_IDS" | awk -F, '{print NF}')"
  PROF_MAX_TOKENS="${MAX_NEW_TOKENS:-16}"
  PROF_DIR="prof_data"
  mkdir -p "$PROF_DIR"

  echo "=== [prof] msprof 采集  devices=$DEVICE_IDS  buckets=${PROF_BUCKET_LIST[*]}  prefill/decode 分阶段 ==="

  for seq in "${PROF_BUCKET_LIST[@]}"; do
    BUCKET_DIR="$PROF_DIR/bucket_${seq}"
    echo ""
    echo "############################################################"
    echo "### prof bucket: prefill_seq=$seq  (decode kv_len=$((seq + 512)))"
    echo "############################################################"

    if [ "$_DEV_COUNT" -eq 1 ]; then
      # 1p: prefill 和 decode 分别采集, 各自 build + 3 warmup + 1 次
      msprof --output="$BUCKET_DIR/prefill" \
        --application="python3 infer_qwen3_8b_mslite_1p.py \
          --device-id $DEVICE_IDS \
          --buckets $seq \
          --max-new-tokens $PROF_MAX_TOKENS \
          --prof-phase prefill" 2>&1

      msprof --output="$BUCKET_DIR/decode" \
        --application="python3 infer_qwen3_8b_mslite_1p.py \
          --device-id $DEVICE_IDS \
          --buckets $seq \
          --max-new-tokens $PROF_MAX_TOKENS \
          --prof-phase decode" 2>&1
    else
      # 2p: infer_qwen3_8b_mslite_tp.py single-bucket, --warmup 3 = 3 warmup + 1 timed
      msprof --output="$BUCKET_DIR" \
        --application="python3 infer_qwen3_8b_mslite_tp.py \
          --device-ids $DEVICE_IDS \
          --model-id $MODEL_ID \
          --prompt-tokens $seq \
          --max-new-tokens $PROF_MAX_TOKENS \
          --warmup 3" 2>&1
    fi
  done

  echo ""
  echo "============================================================"
  echo "=== prof 采集完成 ==="
  echo "============================================================"
  for seq in "${PROF_BUCKET_LIST[@]}"; do
    if [ "$_DEV_COUNT" -eq 1 ]; then
      echo "  bucket $seq -> $PROF_DIR/bucket_${seq}/prefill/  +  $PROF_DIR/bucket_${seq}/decode/"
    else
      echo "  bucket $seq -> $PROF_DIR/bucket_${seq}/"
    fi
  done
  exit 0
fi

# ================= 性能验证：六档扫描（prefill + decode）=================
# 逐档强制命中一个 prefill 档位，每档计时 + 显存峰值 + 核心点断言。
# 六档 prefill_seq -> kv_len(=seq+512)：512->1024 1024->1536 1664->2176
#   2048->2560 2816->3328 3072->3584。核心点：prefill KV 只 pad 到该档 kv_len（非最大 3584）。
#
# 按并行度分流：
#   * 1p (1 device)  -> infer_qwen3_8b_mslite_1p.py（动态单图 + KV 切片, ge.dynamicDims 6 档）
#   * 2p (2 devices) -> infer_qwen3_8b_mslite_tp.py --prompt-tokens（HCCL 六档扫描, 原样保留）
#   * 4p (4 devices) -> 同 2p 逻辑（未验证）
PERF_MAX_TOKENS="${MAX_NEW_TOKENS:-16}"   # 分档性能只需少量 decode 步

_DEV_COUNT="$(echo "$DEVICE_IDS" | awk -F, '{print NF}')"

if [ "$_DEV_COUNT" -eq 1 ]; then
  # ============ 1p：动态单图分档（infer_qwen3_8b_mslite_1p.py）============
  PERF_OUT="${PERF_OUT:-_dynamic_bucket_results.json}"
  echo "=== [perf] 1p 动态单图六档性能验证  device=$DEVICE_IDS  max_new_tokens=$PERF_MAX_TOKENS  repeats=${PERF_REPEATS:-3} ==="
  echo "  (单 prefill mindir + 单 decode mindir, ge.dynamicDims, 一次编译多档推理, KV 切片)"
  python3 infer_qwen3_8b_mslite_1p.py \
    --device-id "$DEVICE_IDS" \
    --max-new-tokens "$PERF_MAX_TOKENS" \
    --buckets all \
    --repeats "${PERF_REPEATS:-3}" \
    --out "$PERF_OUT"
  exit 0
fi

# ============ 2p / 4p：单进程多档性能扫描（--perf-sweep, worker 复用）============
OUT_DIR="bench_tp2_bucket_results"
mkdir -p "$OUT_DIR"
PERF_JSON="$OUT_DIR/perf_sweep.json"
RUN_LOG="$OUT_DIR/perf_sweep.log"

echo "=== [perf] 单进程多档性能验证  devices=$DEVICE_IDS  max_new_tokens=$PERF_MAX_TOKENS  repeats=${PERF_REPEATS:-3} ==="
echo "  (单 build, 6 档循环, worker 复用, 替代 6 次独立进程)"

set +e
python3 infer_qwen3_8b_mslite_tp.py \
  --device-ids "$DEVICE_IDS" \
  --model-id "$MODEL_ID" \
  --perf-sweep \
  --max-new-tokens "$PERF_MAX_TOKENS" \
  --repeats "${PERF_REPEATS:-3}" \
  --json-out "$PERF_JSON" 2>&1 | tee "$RUN_LOG"
rc=${PIPESTATUS[0]}
set -e

if [ "$rc" -ne 0 ] || [ ! -s "$PERF_JSON" ]; then
  echo "  [FAIL] rc=$rc, 见 $RUN_LOG"
  exit 1
fi

kv_ok_count=$(grep -c "核心点 OK" "$RUN_LOG" || true)
echo "  核心点 OK: ${kv_ok_count}/6 档"

echo ""
echo "============================================================"
echo "=== TP=2 六档分档性能汇总 ==="
echo "============================================================"
python3 - "$PERF_JSON" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
rows = data.get("buckets", [])
hdr = f"{'ntok':>6} {'seq':>6} {'kv_len':>7} {'prefill_ms':>11} {'decode_ms':>10}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['prompt_tokens']:>6} {str(r['prefill_seq']):>6} {str(r['kv_len']):>7} "
          f"{r['prefill_ms']:>11} {r['avg_decode_ms']:>10}")
print("")
ok = all(r.get("kv_len") == (r.get("prefill_seq") or 0) + 512 for r in rows)
print(f"核心点校验 (kv_len == prefill_seq + 512, 非最大 3584): "
      f"{'PASS' if ok else 'FAIL'} ({len(rows)}/6 档)")
PY
