#!/bin/bash
# embed_resource.sh - バイナリリソースを.oファイルに変換
#
# Usage: embed_resource.sh <input_file> <output_file> <symbol_name>
#
# objcopyは入力ファイルパスからシンボル名を生成するため、
# out-of-treeビルドではパスが異なりシンボル名が一致しない問題がある。
# このスクリプトはシンボル名付きファイル名で処理し、正しいシンボルを生成する。

set -e

INPUT="$1"
OUTPUT="$2"
SYMBOL_BASE="$3"

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ] || [ -z "$SYMBOL_BASE" ]; then
    echo "Usage: $0 <input_file> <output_file> <symbol_name>" >&2
    exit 1
fi

# 出力を絶対パスに変換
OUTPUT=$(realpath -m "$OUTPUT")

# 一時ディレクトリを作成
WORKDIR=$(mktemp -d)

# クリーンアップ関数
cleanup() {
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

# シンボル名をそのままファイル名として使用
# objcopyはファイル名からシンボル名を生成するため、これで正しいシンボルになる
TMPFILE="${WORKDIR}/${SYMBOL_BASE}"
cp "$INPUT" "$TMPFILE"

case "$(uname -m)" in
    aarch64|arm64)
        ELF_ARCH="elf64-littleaarch64"
        BIN_ARCH="aarch64"
        ;;
    x86_64|amd64)
        ELF_ARCH="elf64-x86-64"
        BIN_ARCH="i386:x86-64"
        ;;
    *)
        echo "Unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

# カレントディレクトリを一時ディレクトリに変更して相対パスで処理
cd "$WORKDIR"
objcopy -I binary -O "$ELF_ARCH" -B "$BIN_ARCH" "$SYMBOL_BASE" "$OUTPUT"
objcopy --add-section .note.GNU-stack=/dev/null --set-section-flags .note.GNU-stack=contents,readonly "$OUTPUT"
