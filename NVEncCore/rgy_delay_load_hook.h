#pragma once

#ifdef _WIN32
#include <windows.h>
#include <delayimp.h>
#include <cstdio>

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
    return nullptr;
}

// 静的ライブラリ内で未参照のまま省かれないよう、実行ファイル側で定義する。
// __pfnDliFailureHook2を定義するため、実行ファイルの1つの翻訳単位（XXEncC.cpp）からだけインクルードする。
// 複数の翻訳単位からインクルードすると多重定義になる。
extern "C" const PfnDliHook __pfnDliFailureHook2 = rgy_delay_load_failure_hook;
#endif
