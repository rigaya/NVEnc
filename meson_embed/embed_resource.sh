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

# カレントディレクトリを一時ディレクトリに変更して相対パスで処理
cd "$WORKDIR"
objcopy -I binary -O elf64-x86-64 -B i386:x86-64 "$SYMBOL_BASE" "$OUTPUT"
