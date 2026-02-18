#!/bin/bash
# libhdr10plus ビルドスクリプト
#
# 使い方:
#   ./build_libhdr10plus.sh [インストール先ディレクトリ]
#
# 前提条件:
#   sudo apt install libssl-dev
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
#   . $HOME/.cargo/env
#   cargo install cargo-c

set -e

HDR10PLUS_VER=2.1.4
HDR10PLUS_SRC=hdr10plus_tool

# スクリプトのあるディレクトリ（ソースをダウンロードする場所）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# インストール先（引数で指定、省略時はスクリプトディレクトリ/build_libhdr10plus）
INSTALL_DIR="${1:-${SCRIPT_DIR}/build_libhdr10plus}"
INSTALL_DIR="$(mkdir -p "$INSTALL_DIR" && cd "$INSTALL_DIR" && pwd)"

echo "libhdr10plus build script"
echo "  Source dir: ${SCRIPT_DIR}"
echo "  Install dir: ${INSTALL_DIR}"

# ソースのダウンロード（スクリプトディレクトリで行う）
cd "${SCRIPT_DIR}"
if [ ! -e "${HDR10PLUS_SRC}" ]; then
    echo "Downloading hdr10plus_tool ${HDR10PLUS_VER}..."
    wget -q -O "${HDR10PLUS_SRC}.tar.gz" "https://github.com/quietvoid/hdr10plus_tool/archive/refs/tags/${HDR10PLUS_VER}.tar.gz"
    tar xf "${HDR10PLUS_SRC}.tar.gz"
    rm "${HDR10PLUS_SRC}.tar.gz"
    mv "${HDR10PLUS_SRC}-${HDR10PLUS_VER}" "${HDR10PLUS_SRC}"
fi

# ビルド & インストール
echo "Building libhdr10plus..."
cd "${SCRIPT_DIR}/${HDR10PLUS_SRC}/hdr10plus"
cargo cinstall --release --prefix="$INSTALL_DIR"

# 静的リンクのため .so を削除して .a だけ残す
echo "Removing shared libraries (keeping static only)..."
find "${INSTALL_DIR}" -name "libhdr10plus-rs.so*" -delete 2>/dev/null || true

echo "libhdr10plus installed to: ${INSTALL_DIR}"
