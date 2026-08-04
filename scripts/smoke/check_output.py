#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NVEnc 終了バッチ用: 出力ファイル / ログ / aup2 設定のスモークチェック。"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def _safe_print(file, *args: object) -> None:
    text = " ".join(str(a) for a in args)
    try:
        print(text, file=file)
    except UnicodeEncodeError:
        enc = getattr(file, "encoding", None) or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"), file=file)


def eprint(*args: object) -> None:
    _safe_print(sys.stderr, *args)


def pprint(*args: object) -> None:
    _safe_print(sys.stdout, *args)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_capture(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return p.returncode, p.stdout or "", p.stderr or ""


def parse_aup2_plugin_config(aup2: Path, name_contains: list[str]) -> dict[str, Any] | None:
    text = aup2.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r"^\[plugin\.\d+\]$", line, re.I):
            name = ""
            config_raw = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("["):
                s = lines[j].strip()
                if s.startswith("plugin.name="):
                    name = s.split("=", 1)[1]
                elif s.startswith("config="):
                    config_raw = s.split("=", 1)[1]
                j += 1
            if any(key in name for key in name_contains) and config_raw:
                try:
                    return json.loads(config_raw)
                except json.JSONDecodeError as ex:
                    raise RuntimeError(f"aup2 plugin config JSON 解析失敗: {aup2} ({name}): {ex}") from ex
            i = j
            continue
        i += 1
    return None


def extract_expected_cmd(conf: dict[str, Any]) -> str:
    enc = conf.get("enc") or {}
    video = conf.get("video") or {}
    parts = []
    cmd = (enc.get("cmd") or "").strip()
    cmdex = (video.get("cmdex") or "").strip()
    if cmd:
        parts.append(cmd)
    if cmdex:
        parts.append(cmdex)
    return " ".join(parts).strip()


def tokenize_cli(s: str) -> list[str]:
    # 簡易トークナイズ（引用符対応）
    return [t for t in re.findall(r'"[^"]*"|\'[^\']*\'|\S+', s) if t]


def option_pairs(tokens: list[str]) -> list[tuple[str, str | None]]:
    pairs: list[tuple[str, str | None]] = []
    i = 0
    while i < len(tokens):
        t = tokens[i].strip('"')
        if t.startswith("-"):
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                pairs.append((t, tokens[i + 1].strip('"')))
                i += 2
            else:
                pairs.append((t, None))
                i += 1
        else:
            i += 1
    return pairs


def extract_log_nvenc_options(log_text: str) -> str | None:
    lines = log_text.splitlines()
    for idx, line in enumerate(lines):
        if "NVEnc options" in line:
            buf: list[str] = []
            for fol in lines[idx + 1 :]:
                s = fol.rstrip()
                if not s:
                    if buf:
                        break
                    continue
                # 次セクションっぽい行で打ち切り
                if re.match(r"^(AAC|raw |NVEnc \[|NVEncC|L-SMASH|CPU|encoded |converting |Aviutl)", s):
                    break
                if s.startswith("---"):
                    break
                buf.append(s.strip())
            return " ".join(buf).strip() if buf else None
    return None


def log_expects_audio(log_text: str) -> bool:
    # 例: audio: 0:00:14.047 2ch 48.0kHz 674274 samples
    m = re.search(
        r"audio:\s*[0-9:.]+(?:\s+\d+ch)?(?:\s+[\d.]+kHz)?\s+(\d+)\s+samples",
        log_text,
        re.I,
    )
    if m:
        return int(m.group(1)) > 0
    # audio: none / 音声なし などは対象外
    if re.search(r"audio:\s*(none|なし|disabled)", log_text, re.I):
        return False
    return False


def log_has_error(log_text: str) -> list[str]:
    hits = []
    for line in log_text.splitlines():
        if re.search(r"\[error\]|auo \[error\]", line, re.I):
            hits.append(line.strip())
    return hits


def probe_ffprobe(ffprobe: Path, media: Path) -> dict[str, Any]:
    cmd = [
        str(ffprobe),
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,codec_name,width,height,sample_rate,channels:format=duration,nb_streams",
        "-of",
        "json",
        str(media),
    ]
    code, out, err = run_capture(cmd)
    if code != 0:
        raise RuntimeError(f"ffprobe 失敗 (exit={code}): {err.strip() or out.strip()}")
    return json.loads(out)


def probe_mediainfo(mediainfo: Path, media: Path) -> dict[str, Any]:
    cmd = [str(mediainfo), "--Output=JSON", str(media)]
    code, out, err = run_capture(cmd)
    if code != 0:
        raise RuntimeError(f"MediaInfo 失敗 (exit={code}): {err.strip() or out.strip()}")
    return json.loads(out)


def mediainfo_counts(mi: dict[str, Any]) -> tuple[int, int]:
    media = mi.get("media") or {}
    tracks = media.get("track") or []
    v = sum(1 for t in tracks if t.get("@type") == "Video")
    a = sum(1 for t in tracks if t.get("@type") == "Audio")
    # General の VideoCount/AudioCount もフォールバック
    for t in tracks:
        if t.get("@type") == "General":
            try:
                v = max(v, int(t.get("VideoCount") or 0))
                a = max(a, int(t.get("AudioCount") or 0))
            except ValueError:
                pass
    return v, a


def ffprobe_counts(fp: dict[str, Any]) -> tuple[int, int, float | None]:
    streams = fp.get("streams") or []
    v = sum(1 for s in streams if s.get("codec_type") == "video")
    a = sum(1 for s in streams if s.get("codec_type") == "audio")
    dur = None
    try:
        dur = float((fp.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        pass
    return v, a, dur


def compare_cli(expected: str, actual: str) -> list[str]:
    """aup2 の enc.cmd(+cmdex) のオプションがログ CLI に含まれるか。"""
    problems: list[str] = []
    if not expected:
        return problems
    if not actual:
        problems.append("ログから NVEnc options を抽出できませんでした")
        return problems

    exp_pairs = option_pairs(tokenize_cli(expected))
    act_tokens = [t.strip('"') for t in tokenize_cli(actual)]
    act_text = " ".join(act_tokens)

    for opt, val in exp_pairs:
        if opt not in act_tokens and opt not in act_text:
            problems.append(f"ログCLIに無いオプション: {opt}" + (f" {val}" if val is not None else ""))
            continue
        if val is None:
            continue
        # --crf 22 のように隣接、または文字列として含まれること
        ok = False
        for i, t in enumerate(act_tokens):
            if t == opt and i + 1 < len(act_tokens) and act_tokens[i + 1] == val:
                ok = True
                break
            if t.startswith(opt + "=") and t.split("=", 1)[1] == val:
                ok = True
                break
        if not ok and f"{opt} {val}" not in act_text:
            problems.append(f"ログCLIの値が不一致: {opt} {val}")
    return problems


def write_result(path: Path, ok: bool, lines: list[str]) -> None:
    body = ["PASS" if ok else "FAIL", *lines, ""]
    path.write_text("\n".join(body), encoding="utf-8-sig")


def main() -> int:
    ap = argparse.ArgumentParser(description="NVEnc smoke check (after-bat)")
    ap.add_argument("--savpath", required=True, help="出力ファイル (%{savpath})")
    ap.add_argument("--logpath", default="", help="ログファイル (%{logpath})")
    ap.add_argument("--config", required=True, help="smoke_config.json")
    ap.add_argument("--aup2", default="", help="aup2 パス (未指定時は config)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    savpath = Path(args.savpath)
    logpath = Path(args.logpath) if args.logpath else savpath.with_name(savpath.stem + "_log.txt")
    aup2 = Path(args.aup2 or cfg.get("aup2") or "")
    mediainfo = Path(cfg["mediainfo"])
    ffprobe = Path(cfg["ffprobe"])
    name_keys = cfg.get("plugin_name_contains") or ["NVEnc"]
    require_video = bool(cfg.get("require_video", True))
    require_audio_if_log = bool(cfg.get("require_audio_if_log_has_audio", True))
    min_duration = float(cfg.get("min_duration_sec", 0.1))
    result_name = cfg.get("result_filename") or "smoke_result.txt"
    result_path = savpath.with_name(result_name) if savpath.parent.exists() else Path(result_name)

    reports: list[str] = []
    errors: list[str] = []

    def ok(msg: str) -> None:
        reports.append(f"[OK] {msg}")
        pprint(f"[OK] {msg}")

    def ng(msg: str) -> None:
        errors.append(msg)
        reports.append(f"[NG] {msg}")
        eprint(f"[NG] {msg}")

    pprint("=== NVEnc smoke check ===")
    pprint(f"savpath : {savpath}")
    pprint(f"logpath : {logpath}")
    pprint(f"aup2    : {aup2}")

    # --- 出力ファイル存在 ---
    if not savpath.is_file():
        ng(f"出力ファイルがありません: {savpath}")
        write_result(result_path, False, reports)
        return 1
    ok(f"出力ファイルあり ({savpath.stat().st_size} bytes)")

    # --- ツール存在 ---
    if not ffprobe.is_file():
        ng(f"ffprobe がありません: {ffprobe}")
    if not mediainfo.is_file():
        ng(f"MediaInfo がありません: {mediainfo}")
    if errors:
        write_result(result_path, False, reports)
        return 1

    # --- ffprobe ---
    try:
        fp = probe_ffprobe(ffprobe, savpath)
        fv, fa, fdur = ffprobe_counts(fp)
        ok(f"ffprobe: video={fv}, audio={fa}, duration={fdur}")
        if require_video and fv < 1:
            ng("ffprobe: 映像ストリームがありません")
        if fdur is not None and fdur < min_duration:
            ng(f"ffprobe: duration が短すぎます ({fdur})")
    except Exception as ex:  # noqa: BLE001
        ng(str(ex))
        fv, fa, fdur = 0, 0, None

    # --- MediaInfo ---
    try:
        mi = probe_mediainfo(mediainfo, savpath)
        mv, ma = mediainfo_counts(mi)
        ok(f"MediaInfo: video={mv}, audio={ma}")
        if require_video and mv < 1:
            ng("MediaInfo: 映像トラックがありません")
        # 相互確認
        if fv != mv:
            ng(f"映像トラック数不一致 ffprobe={fv} MediaInfo={mv}")
        if fa != ma:
            ng(f"音声トラック数不一致 ffprobe={fa} MediaInfo={ma}")
    except Exception as ex:  # noqa: BLE001
        ng(str(ex))
        ma = 0

    # --- ログ ---
    log_text = ""
    if not logpath.is_file():
        ng(f"ログファイルがありません (ログ自動保存: 出力先と同じ を想定): {logpath}")
    else:
        log_text = logpath.read_text(encoding="utf-8-sig", errors="replace")
        ok(f"ログファイルあり ({len(log_text)} chars)")

        err_lines = log_has_error(log_text)
        if err_lines:
            for line in err_lines[:10]:
                ng(f"ログに error: {line}")
        else:
            ok("ログに [error] なし")

        expects_audio = log_expects_audio(log_text)
        if require_audio_if_log and expects_audio:
            if fa < 1 or ma < 1:
                ng("ログ上は音声ありだが、出力に音声トラックがありません (AviUtl2 OUTPUT_INFO 系を疑う)")
            else:
                ok("ログの音声あり / 出力に音声トラックあり")
        elif expects_audio is False:
            ok("ログ上は音声なし/不明のため音声必須チェックをスキップ")

        log_cli = extract_log_nvenc_options(log_text)
        if log_cli:
            ok(f"ログ CLI 抽出: {log_cli[:120]}{'...' if len(log_cli) > 120 else ''}")
        else:
            ng("ログから NVEnc options を抽出できませんでした")

        # --- aup2 CLI 照合 ---
        if aup2 and aup2.is_file():
            try:
                conf = parse_aup2_plugin_config(aup2, name_keys)
                if conf is None:
                    ng(f"aup2 から NVEnc 設定を見つけられません: {aup2}")
                else:
                    expected = extract_expected_cmd(conf)
                    ok(f"aup2 enc.cmd(+cmdex): {expected}")
                    if log_cli:
                        cli_problems = compare_cli(expected, log_cli)
                        for p in cli_problems:
                            ng(p)
                        if not cli_problems:
                            ok("aup2 の CLI オプションはログ CLI に含まれています")
            except Exception as ex:  # noqa: BLE001
                ng(str(ex))
        else:
            ng(f"aup2 がありません: {aup2}")

    passed = len(errors) == 0
    write_result(result_path, passed, reports)
    pprint(f"result -> {result_path}")
    pprint("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as ex:  # noqa: BLE001
        eprint(f"[FATAL] {ex}")
        sys.exit(2)
