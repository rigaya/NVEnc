#!/bin/bash
# libdovi ビルドスクリプト
#
# 使い方:
#   ./build_libdovi.sh [インストール先ディレクトリ]
#
# 前提条件:
#   sudo apt install libssl-dev
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
#   . $HOME/.cargo/env
#   cargo install cargo-c

set -e

DOVI_VER=3.3.1
DOVI_SRC=dovi_tool

# スクリプトのあるディレクトリ（ソースをダウンロードする場所）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# インストール先（引数で指定、省略時はスクリプトディレクトリ/build_libdovi）
INSTALL_DIR="${1:-${SCRIPT_DIR}/build_libdovi}"
INSTALL_DIR="$(mkdir -p "$INSTALL_DIR" && cd "$INSTALL_DIR" && pwd)"

echo "libdovi build script"
echo "  Source dir: ${SCRIPT_DIR}"
echo "  Install dir: ${INSTALL_DIR}"

# ソースのダウンロード（スクリプトディレクトリで行う）
cd "${SCRIPT_DIR}"
if [ ! -e "${DOVI_SRC}" ]; then
    echo "Downloading dovi_tool ${DOVI_VER}..."
    wget -q -O "${DOVI_SRC}.tar.gz" "https://github.com/quietvoid/dovi_tool/archive/refs/tags/${DOVI_VER}.tar.gz"
    tar xf "${DOVI_SRC}.tar.gz"
    rm "${DOVI_SRC}.tar.gz"
    mv "${DOVI_SRC}-${DOVI_VER}" "${DOVI_SRC}"
fi

# ビルド & インストール
echo "Building libdovi..."
cd "${SCRIPT_DIR}/${DOVI_SRC}/dolby_vision"
cargo cinstall --release --prefix="$INSTALL_DIR"

# 静的リンクのため .so を削除して .a だけ残す
echo "Removing shared libraries (keeping static only)..."
find "${INSTALL_DIR}" -name "libdovi.so*" -delete 2>/dev/null || true

echo "libdovi installed to: ${INSTALL_DIR}"
