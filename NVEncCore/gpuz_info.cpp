// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2011-2016 rigaya
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

#if (defined(_WIN32) || defined(_WIN64))
#include <Windows.h>
#include "gpuz_info.h"

//大文字小文字を無視して、1文字検索
static inline const WCHAR *wcsichr(const WCHAR *str, int c) {
    c = tolower(c);
    for (; *str; str++)
        if (c == tolower(*str))
            return str;
    return NULL;
}

//大文字小文字を無視して、文字列を検索
static inline const WCHAR *wcsistr(const WCHAR *str, const WCHAR *substr) {
    size_t len = 0;
    if (substr && (len = wcslen(substr)) != NULL)
        for (; (str = wcsichr(str, substr[0])) != NULL; str++)
            if (_wcsnicmp(str, substr, len) == NULL)
                return str;
    return NULL;
}

int get_gpuz_info(GPUZ_SH_MEM *data) {
    HANDLE memmap = OpenFileMapping(FILE_MAP_READ, FALSE, SHMEM_NAME);
    if (NULL == memmap) {
        return 1;
    }

    GPUZ_SH_MEM *ptr = (GPUZ_SH_MEM *)MapViewOfFile(memmap, FILE_MAP_READ, 0, 0, 0);
    if (ptr == nullptr) {
        CloseHandle(memmap);
        return 1;
    }
    memcpy(data, ptr, sizeof(data[0]));
    UnmapViewOfFile(ptr);
    CloseHandle(memmap);
    return 0;
}

double gpu_core_clock(GPUZ_SH_MEM *data) {
    for (int i = 0; i < MAX_RECORDS; i++) {
        if (   wcsistr(data->sensors[i].name, L"Core")
            && wcsistr(data->sensors[i].name, L"Clock")) {
            return data->sensors[i].value;
        }
    }
    return 0.0;
}

double gpu_load(GPUZ_SH_MEM *data) {
    for (int i = 0; i < MAX_RECORDS; i++) {
        if (wcsistr(data->sensors[i].name, L"GPU Load")) {
            return data->sensors[i].value;
        }
    }
    return 0.0;
}

double video_engine_load(GPUZ_SH_MEM *data, bool *pbVideoEngineUsage) {
    for (int i = 0; i < MAX_RECORDS; i++) {
        if (wcsistr(data->sensors[i].name, L"Video Engine Load")) {
            if (pbVideoEngineUsage) *pbVideoEngineUsage = true;
            return data->sensors[i].value;
        }
    }
    if (pbVideoEngineUsage) *pbVideoEngineUsage = false;
    return 0.0;
}

#endif //#if (defined(_WIN32) || defined(_WIN64))
