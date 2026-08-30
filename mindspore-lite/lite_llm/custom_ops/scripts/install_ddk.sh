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
# install_ddk.sh — 标准 DDK 安装脚本：安装完成即可用，无需再跑任何修复/准备步骤。
#
# 背景（多用户共享机实测）：
#   DDK 的部分二进制（tools/tools_omg/*/omg 及 tools/tools_ascendc/package/ 下的
#   opbuild / ascendc_tiling_tool / ir_model_compile / npu_kernel_launch）把 ELF
#   解释器硬编码为 /tmp/ld-linux-x86-64-<glibc>.so.2，依赖 set_ascendc_env.sh
#   运行时在 /tmp 建软链。当 /tmp 为 sticky 且该软链被其他用户创建时，内核
#   fs.protected_symlinks 拒绝跟随 → "Permission denied"，且普通用户无法删除
#   他人软链。此外 env 脚本 / omg wrapper 运行时还要 chmod、mkdir 等写操作，
#   共享只读安装会失败。
#
#   本脚本在安装时一次性完成：ELF 解释器改写为系统 loader 绝对路径（patchelf，
#   备份为 <bin>.bak-tmplinker）、补齐执行位、预建运行时目录、并自检 omg 可执行。
#   此后所有用户只需 source set_ascendc_env.sh，不再产生任何共享状态写入。
#
# 用法：
#   bash scripts/install_ddk.sh <SRC> <DEST>   # SRC: DDK 目录或 tar 包(.tar.gz/.tgz/.tar)
#                                              # 安装到 DEST；tar 包解包后须直接含 tools/ ddk/
#   bash scripts/install_ddk.sh <DEST>          # 仅对已安装的 DEST 做安装后准备（幂等，可反复跑）
#   --force                                     # DEST 已存在且非空时强制重新复制
#
# 前提：
#   - 系统 glibc >= DDK 要求版本（自动检测；ddk_0820 要求 2.35，即 Ubuntu 22.04+）
#   - patchelf 已安装（仅当二进制仍含 /tmp 解释器时需要；apt install patchelf）
#   - 安装到共享目录（如 /opt/hisi_ddk）需要 root
set -euo pipefail

usage() {
  cat <<'EOF'
标准 DDK 安装脚本：安装完成即可用（ELF 解释器改系统 loader、执行位、运行时目录、自检）。

用法：
  bash scripts/install_ddk.sh <SRC> <DEST>   # SRC: DDK 目录或 tar 包；安装到 DEST
  bash scripts/install_ddk.sh <DEST>          # 对已安装的 DEST 做安装后准备（幂等，可反复跑）
  --force                                     # DEST 已存在且非空时强制重新复制
EOF
  exit "${1:-0}"
}

log() { printf '[install_ddk] %s\n' "$*"; }
die() { printf '[install_ddk] [ERROR] %s\n' "$*" >&2; exit 1; }

detect_sys_ld() {
  for p in /lib64/ld-linux-x86-64.so.2 \
           /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 \
           /usr/lib64/ld-linux-x86-64.so.2 \
           /usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2; do
    [ -e "$p" ] && { SYS_LD="$p"; return 0; }
  done
  return 1
}

sys_glibc() {
  # 优先 getconf（单行输出，无 head 提前退出的 SIGPIPE 隐患）；ldd 回退时用 || true 兜底
  local v
  v="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}' || true)"
  [ -n "$v" ] || v="$(ldd --version 2>/dev/null | head -1 | awk '{print $NF}' || true)"
  printf '%s\n' "$v"
}
ver_ge() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" = "$1" ]; }

# DDK 要求的 glibc 版本：glibc_version.txt 与 /tmp 解释器名中出现版本的最大值
ddk_glibc_req() {
  local root="$1" f versions=""
  versions="$(find "$root/tools/tools_omg" -name glibc_version.txt -exec cat {} \; 2>/dev/null || true)"
  for f in "$root/tools/tools_ascendc/package/opbuild" \
           "$root/tools/tools_ascendc/package/ascendc_tiling_tool" \
           "$root/tools/tools_ascendc/package/ir_model_compile" \
           "$root/tools/tools_ascendc/package/npu_kernel_launch" \
           "$root"/tools/tools_omg/*/omg; do
    [ -f "$f" ] || continue
    versions="$versions
$(readelf -l "$f" 2>/dev/null | sed -n 's/.*interpreter: \/tmp\/ld-linux-x86-64-\([0-9.]*\)\.so\.2.*/\1/p' || true)"
  done
  printf '%s\n' "$versions" | grep -E '^[0-9.]+$' | sort -V | tail -1 || true
}

check_glibc() {
  local sys req
  sys="$(sys_glibc)"
  req="$(ddk_glibc_req "$DEST")"
  [ -n "$sys" ] || die "无法检测系统 glibc 版本"
  [ -n "$req" ] || { log "未检测到 glibc 版本要求（无 /tmp 解释器），跳过"; return 0; }
  if ver_ge "$sys" "$req"; then
    log "glibc 检查通过：系统 $sys >= DDK 要求 $req"
  else
    die "系统 glibc ($sys) 低于 DDK 要求 ($req)：需 Ubuntu 22.04+；旧 glibc 上的 /tmp loader 单用户流程不在本脚本支持范围"
  fi
}

# 把 /tmp 解释器改写为系统 loader 绝对路径（多用户冲突根治；备份一次，幂等）
patch_interpreters() {
  local f cur n=0 bins
  bins="$(find "$DEST/tools/tools_ascendc/package" "$DEST/tools/tools_omg" \
          -maxdepth 3 -type f \( -name omg -o -name opbuild -o -name ascendc_tiling_tool \
          -o -name ir_model_compile -o -name npu_kernel_launch \) 2>/dev/null || true)"
  for f in $bins; do
    [ -f "$f" ] || continue
    cur="$(readelf -l "$f" 2>/dev/null | awk '/interpreter/{print $NF}' | tr -d ']' || true)"
    case "$cur" in
      /tmp/*)
        command -v patchelf >/dev/null 2>&1 || die "需要 patchelf（apt install patchelf）"
        [ -f "$f.bak-tmplinker" ] || cp "$f" "$f.bak-tmplinker"
        patchelf --set-interpreter "$SYS_LD" "$f"
        log "解释器 $cur -> $SYS_LD : ${f#"$DEST"/}"
        n=$((n + 1));;
      "$SYS_LD") ;;
      *) [ -n "$cur" ] && log "跳过（解释器=$cur）: ${f#"$DEST"/}";;
    esac
  done
  log "ELF 解释器处理完成（改动 $n 个）"
}

prepare_perms() {
  find "$DEST/tools/tools_omg" -maxdepth 2 -type f -name omg -exec chmod +x {} + 2>/dev/null || true
  for t in opbuild ascendc_tiling_tool ir_model_compile npu_kernel_launch msopgen opc ascendebug; do
    [ -f "$DEST/tools/tools_ascendc/package/$t" ] && chmod +x "$DEST/tools/tools_ascendc/package/$t"
  done
  # DDK bundled loader（旧 glibc 分支会 chmod，预先补执行位无副作用）
  find "$DEST/tools/tools_omg" -path '*/x86_64-pc-linux-gnu-*/ld-linux-x86-64.so.2' -exec chmod +x {} + 2>/dev/null || true
  [ -d "$DEST/ddk/ccec_compiler/bin" ] && chmod -R +x "$DEST/ddk/ccec_compiler/bin"
  log "执行位已补齐"
}

prepare_dirs() {
  local p
  if [ -d "$DEST/tools/platform" ]; then
    for p in "$DEST"/tools/platform/*/; do
      [ -d "$p" ] || continue
      mkdir -p "$p/simulator/model/conf" "$p/simulator/camodel_log"
    done
  fi
  mkdir -p "$DEST/ddk/npu_simulator/model/conf" "$DEST/ddk/npu_simulator/camodel_log" 2>/dev/null || true
  log "运行时目录已预建"
}

verify_install() {
  local out rc
  log "自检：运行 omg --help"
  if [ -f "$DEST/tools/tools_omg/omg" ]; then
    set +e
    out="$(bash "$DEST/tools/tools_omg/omg" --help 2>&1)"
    rc=$?
    set -e
  elif [ -f "$DEST/tools/tools_omg/master/omg" ]; then
    set +e
    out="$(env LD_LIBRARY_PATH="$DEST/tools/tools_omg/master/lib64" "$DEST/tools/tools_omg/master/omg" --help 2>&1)"
    rc=$?
    set -e
  else
    die "找不到 omg 工具（$DEST/tools/tools_omg/ 下无 omg）"
  fi
  if [ "$rc" -eq 0 ] && case "$out" in *--model*) true;; *) false;; esac; then
    log "自检通过：omg 可执行"
  else
    printf '%s\n' "$out" | tail -20 >&2
    die "omg 自检失败（rc=$rc）"
  fi
}

main() {
  local SRC="" DEST="" FORCE=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --force) FORCE=1; shift;;
      --help | -h) usage 0;;
      *) if [ -z "$SRC" ]; then SRC="$1"; else DEST="$1"; fi; shift;;
    esac
  done
  [ -n "$DEST" ] || { DEST="$SRC"; SRC=""; }
  [ -n "$DEST" ] || usage 1

  DEST="$(realpath -m "$DEST")"
  if [ -n "$SRC" ]; then
    SRC="$(realpath -m "$SRC")"
    [ -e "$SRC" ] || die "源不存在: $SRC"
    if [ -d "$SRC" ]; then
      { [ -d "$SRC/tools" ] && [ -d "$SRC/ddk" ]; } || die "源不是 DDK 目录（缺 tools/ 或 ddk/）: $SRC"
    else
      case "$SRC" in
        *.tar.gz | *.tgz | *.tar) ;;
        *) die "源必须是 DDK 目录或 tar 包: $SRC";;
      esac
    fi
    if [ -e "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ]; then
      if [ "$FORCE" -eq 1 ]; then
        log "DEST 已存在，--force 强制重新复制"
      else
        log "DEST 已存在且非空，跳过复制，仅做安装后准备（幂等）；如需重装请用 --force"
        SRC=""
      fi
    fi
  fi

  if [ -n "$SRC" ] && [ "$SRC" != "$DEST" ]; then
    [ -w "$(dirname "$DEST")" ] || die "目标父目录不可写: $(dirname "$DEST")（共享安装需要 root）"
    mkdir -p "$DEST"
    if [ -d "$SRC" ]; then
      log "复制 $SRC -> $DEST"
      cp -a "$SRC/." "$DEST/"
    else
      log "解包 $SRC -> $DEST"
      tar -xf "$SRC" -C "$DEST"
    fi
  fi

  { [ -d "$DEST/tools" ] && [ -d "$DEST/ddk" ]; } || die "目标不是完整 DDK（缺 tools/ 或 ddk/）: $DEST"
  [ -w "$DEST" ] || die "目标不可写: $DEST（共享安装需要 root，或改用个人目录）"

  detect_sys_ld || die "未找到系统 loader（$SYS_LD 候选均不存在）"
  check_glibc
  patch_interpreters
  prepare_perms
  prepare_dirs
  verify_install

  cat <<EOF

安装完成，DDK 已就绪：$DEST
使用：
  source $DEST/tools/tools_ascendc/set_ascendc_env.sh
之后 omg / opbuild 等工具可直接使用（不再依赖 /tmp loader，多用户互不冲突）。
还原 ELF 解释器：把各 <bin>.bak-tmplinker 覆盖回原文件。
EOF
}

main "$@"
