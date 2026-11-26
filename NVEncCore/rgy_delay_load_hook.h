#pragma once

#ifdef _WIN32
#include <windows.h>
#include <delayimp.h>
#include <cstdio>
#include "rgy_tchar.h"
#include "rgy_version.h"
#if ENABLE_PROBE_BUILD
#include "rgy_probe_cli.h"
#endif

// 遅延ロードの例外を処理する前に、読み込めなかった依存先を記録する。
static FARPROC WINAPI rgy_delay_load_failure_hook(unsigned notification, PDelayLoadInfo info) {
    const char *operation = notification == dliFailLoadLib ? "LoadLibrary" : "GetProcAddress";
    std::fprintf(stderr, "Delay load failed (%s, error %lu): %s, ", operation, info->dwLastError, info->szDll);
    if (info->dlp.fImportByName) {
        std::fprintf(stderr, "function %s\n", info->dlp.szProcName);
    } else {
        std::fprintf(stderr, "ordinal %lu\n", info->dlp.dwOrdinal);
    }
    std::fflush(stderr);
#if ENABLE_PROBE_BUILD
    try {
        if (!rgy_probe::active()) {
            return nullptr;
        }
        // 異常終了の前に、標準エラーへ向け直した標準出力のログをフラッシュする。
        std::fflush(stdout);
        auto symbol = info->dlp.fImportByName ? rgy_probe::Json(info->dlp.szProcName)
            : rgy_probe::Json("#" + std::to_string(info->dlp.dwOrdinal));
        rgy_probe::finish_delay_load(notification == dliFailLoadLib, info->dwLastError, info->szDll, std::move(symbol));
    } catch (...) {
        // 診断結果を出せなくても、遅延ロード本来の例外処理に委ねる。
    }
#endif
    return nullptr;
}

// 静的ライブラリ内で未参照のまま省かれないよう、実行ファイル側で定義する。
// __pfnDliFailureHook2を定義するため、実行ファイルの1つの翻訳単位（XXEncC.cpp）からだけインクルードする。
// 複数の翻訳単位からインクルードすると多重定義になる。
extern "C" const PfnDliHook __pfnDliFailureHook2 = rgy_delay_load_failure_hook;
#endif
