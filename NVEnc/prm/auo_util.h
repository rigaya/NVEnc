// -----------------------------------------------------------------------------------------
// x264guiEx/x265guiEx/svtAV1guiEx/ffmpegOut/QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2010-2022 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// --------------------------------------------------------------------------------------------

#ifndef _AUO_UTIL_H_
#define _AUO_UTIL_H_

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#if (_MSC_VER >= 1800)
#include <VersionHelpers.h>
#endif
#include <string.h>
#include <vector>
#include <string>
#include <cstdarg>
#include <stddef.h>
#include <stdio.h>
#include <algorithm>
#include <intrin.h>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")

#include "rgy_util.h"
#include "rgy_codepage.h"
#include "rgy_env.h"
#include "rgy_filesystem.h"
#include "auo.h"
#include "auo_version.h"

//関数マクロ
#define clamp(x, low, high) (((x) <= (high)) ? (((x) >= (low)) ? (x) : (low)) : (high))
#define foreach(it,a) \
    for (auto (it)=(a).begin();(it)!=(a).end();(it)++)

//基本的な関数
static inline int div_round(int i, int div) { // 四捨五入
    int ret = i / div;
    if (div <= (i % div) * 2) {
        ret++;
    }
    return ret;
};
static inline int64_t div_round(int64_t i, int64_t div) { // 四捨五入
    int64_t ret = i / div;
    if (div <= (i % div) * 2) {
        ret++;
    }
    return ret;
};

static inline BOOL check_process_exitcode(PROCESS_INFORMATION *pi) {
    DWORD exit_code;
    if (!GetExitCodeProcess(pi->hProcess, &exit_code))
        return TRUE;
    return exit_code != 0;
}

//最後に"\"なしで戻る
static inline void get_exe_name(char *exe_name, size_t nSize) {
    strcpy_s(exe_name, nSize, PathGetFilename(getExePathA()).c_str());
}
static inline void get_aviutl_dir(char *aviutl_dir, size_t nSize) {
    PathGetDirectory(aviutl_dir, nSize, getExePathA().c_str());
}
static inline void get_aviutl_dir(WCHAR *aviutl_dir, size_t nSize) {
    PathGetDirectory(aviutl_dir, nSize, getExePathW().c_str());
}
static inline void get_auo_path(char *auo_path, size_t nSize) {
    strcpy_s(auo_path, nSize, getModulePathA(GetModuleHandleA(AUO_NAME)).c_str());
}
static inline void get_auo_path(WCHAR *auo_path, size_t nSize) {
    wcscpy_s(auo_path, nSize, getModulePathW(GetModuleHandleW(AUO_NAME_W)).c_str());
}
static inline void get_auo_dir(char *auo_dir, size_t nSize) {
    PathGetDirectory(auo_dir, nSize, getModulePathA(GetModuleHandleA(AUO_NAME)).c_str());
}
static inline void get_auo_dir(wchar_t *auo_dir, size_t nSize) {
    PathGetDirectory(auo_dir, nSize, getModulePathW(GetModuleHandleW(AUO_NAME_W)).c_str());
}
static bool is_aviutl2() {
    return AVIUTL_TARGET_VER == 2;
}

//cmd中のtarget_argを抜き出し削除する
//del_valueが+1ならその後の値を削除する、-1ならその前の値を削除する
//値を削除できたらTRUEを返す
BOOL del_arg(TCHAR *cmd, TCHAR *target_arg, int del_arg_delta);

static DWORD cpu_core_count() {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors;
}

#endif //_AUO_UTIL_H_
