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

# Shared bash utilities for scripts/pre_commit/. Sourced, not executed.
# Idempotent: PRE_COMMIT_COMMON_LOADED guards against re-sourcing so the
# readonly color constants are not redefined.
#
# This library deliberately does NOT enable set -u/pipefail. Callers own
# their own shell flags; forcing them here would change the exit semantics
# of every script that sources this file (notably githooks/pre-push, which
# runs 1000+ lines without set -u).
#
# Color constants (CYAN/MAGENTA/WHITE/etc.) are consumed by scripts that
# source this file; shellcheck analyzes each file in isolation and would
# flag every exported symbol as unused, so the whole file disables SC2034.

# shellcheck disable=SC2034
[[ -n "${PRE_COMMIT_COMMON_LOADED:-}" ]] && return 0
PRE_COMMIT_COMMON_LOADED=1

# Resolve this file's directory via BASH_SOURCE so the library works
# regardless of the caller's CWD (git invokes hooks from repo root; install
# scripts may be run from anywhere). readlink -f resolves symlinks, which
# matters if a user symlinks githooks/* into .git/hooks.
_PRE_COMMIT_LIB_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit; pwd)"
# shellcheck source=../versions.sh
source "${_PRE_COMMIT_LIB_DIR}/../versions.sh"

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly MAGENTA='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly WHITE='\033[1;37m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Logging. Pure printers: never call exit. Use die() when a caller wants
# log_error + exit 1 in one step. log_debug is intentionally NOT gated on
# a DEBUG flag to preserve the existing pre-push verbosity contract.
log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*" >&2; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_step()    { echo -e "${BLUE}${BOLD}[STEP]${NC} $*"; }
log_header()  { echo -e "${MAGENTA}${BOLD}=== $* ===${NC}"; }
log_debug()   { echo -e "${WHITE}[DEBUG]${NC} $*"; }

die() { log_error "$*"; exit 1; }

command_exists() { command -v "$1" >/dev/null 2>&1; }

detect_architecture() {
    case "$(uname -m)" in
        arm64|aarch64) echo "arm64" ;;
        x86_64)        echo "x86_64" ;;
        *)             uname -m ;;
    esac
}

detect_os() {
    case "$(uname -s)" in
        Darwin) echo "macos" ;;
        Linux)
            if [[ -f /etc/debian_version ]] || [[ -f /etc/lsb-release ]]; then echo "debian"
            elif [[ -f /etc/redhat-release ]]; then echo "redhat"
            else echo "linux"
            fi ;;
        MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
        *) uname -s ;;
    esac
}

is_arm64_mac()    { [[ "$(detect_os)" == "macos" ]] && [[ "$(detect_architecture)" == "arm64" ]]; }
is_debian_based() { command_exists apt-get && { [[ -f /etc/debian_version ]] || [[ -f /etc/lsb-release ]]; }; }
