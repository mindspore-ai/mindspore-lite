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
# fix_ddk_tmp_linker.sh — 修复 DDK 二进制的 /tmp ELF 解释器在多用户机器上失效的问题。
#
# 背景（本仓库实测，Ubuntu 22.04 x86_64 共享机）：
#   DDK 的部分二进制（opbuild / omg / ascendc_tiling_tool / ir_model_compile /
#   npu_kernel_launch）把 ELF 解释器硬编码为 /tmp/ld-linux-x86-64-<glibc>.so.2。
#   set_ascendc_env.sh 会在 /tmp 建该符号链接指向系统 linker。
#
#   但当 /tmp 是 sticky（drwxrwxrwt）且该符号链接被**另一个用户**创建时，
#   内核 fs.protected_symlinks 行为会拒绝为 ELF 解释器跟随它 → 执行报
#   "Permission denied"（rc=126），opbuild 静默失败 → build_out/autogen/
#   aic-<soc>-ops-info.ini 不生成 → cmake add_bin_compile_target 报
#   FileNotFoundError。
#
#   由于无法删除他人拥有的 /tmp 符号链接，本脚本把这批二进制的 ELF 解释器
#   直接改写为系统 linker 的绝对路径（glibc 版本一致，兼容），绕开 /tmp。
#
# 用法：
#   bash scripts/fix_ddk_tmp_linker.sh [DDK_ROOT]
#   默认 DDK_ROOT = $DDK_PATH 的推导根，或 /home/zhugd/tool/hisi_ddk/ddk_2
#
# 幂等：已指向绝对路径的二进制会跳过；可重复运行。
set -euo pipefail

DDK_ROOT="${1:-${HIAI_DDK:-/home/zhugd/tool/hisi_ddk/ddk_2}}"
# 系统 linker（Ubuntu 22.04 x86_64）；glibc 与 DDK bundled 版本一致（2.35）
SYS_LD="${SYS_LD:-/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2}"

if ! command -v patchelf >/dev/null 2>&1; then
  echo "[FATAL] 需要 patchelf；请先 apt install patchelf" >&2
  exit 2
fi
if [ ! -e "$SYS_LD" ]; then
  echo "[FATAL] 系统 linker 不存在: $SYS_LD（按发行版调整 SYS_LD）" >&2
  exit 2
fi

# DDK 里把解释器硬编码到 /tmp 的二进制（实测 5 个）
BINARIES=(
  "tools/tools_ascendc/package/opbuild"
  "tools/tools_ascendc/package/ascendc_tiling_tool"
  "tools/tools_ascendc/package/ir_model_compile"
  "tools/tools_ascendc/package/npu_kernel_launch"
  "tools/tools_omg/master/omg"
)

patched=0; skipped=0; missing=0
for rel in "${BINARIES[@]}"; do
  bin="$DDK_ROOT/$rel"
  if [ ! -f "$bin" ]; then
    echo "[skip] 不存在: $rel"
    missing=$((missing+1)); continue
  fi
  cur="$(patchelf --print-interpreter "$bin" 2>/dev/null || echo '?')"
  if [ "$cur" = "$SYS_LD" ]; then
    echo "[ok]   已是绝对路径: $rel"
    skipped=$((skipped+1)); continue
  fi
  case "$cur" in
    /tmp/*)
      cp "$bin" "$bin.bak-tmplinker"           # 备份一次
      patchelf --set-interpreter "$SYS_LD" "$bin"
      echo "[fix]  $rel  ($cur -> $SYS_LD)"
      patched=$((patched+1));;
    *)
      echo "[skip] 解释器非 /tmp（$cur）: $rel"
      skipped=$((skipped+1));;
  esac
done

echo "----"
echo "patched=$patched skipped=$skipped missing=$missing"
echo "如需还原：把每个 <bin>.bak-tmplinker 覆盖回原文件即可。"
