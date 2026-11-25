#pragma once

#include <cstdio>
#include "rgy_tchar.h"

// 診断用の縮小構成では、通常起動から省略された機能に到達しないようにする。
inline bool rgy_probe_cli_args(int& argc, TCHAR **argv) {
    if (argc == 2 && (_tcscmp(argv[1], _T("--version")) == 0 || _tcscmp(argv[1], _T("--help")) == 0)) {
        return true;
    }

    bool probe = false;
    int next = 1;
    for (int i = 1; i < argc; ++i) {
        if (_tcscmp(argv[i], _T("--probe-report")) == 0) {
            // 診断結果の出力を実装するまでは、この印だけを除いて既存の引数解析に渡す。
            probe = true;
            continue;
        }
        if (_tcscmp(argv[i], _T("--probe-devices")) == 0 || _tcscmp(argv[i], _T("--probe-inventory")) == 0) {
            probe = true;
        }
        argv[next++] = argv[i];
    }
    argc = next;
    argv[argc] = nullptr;
    if (probe) {
        return true;
    }

    _ftprintf(stderr, _T("This binary is for GPU diagnostics only.\n"));
    return false;
}
