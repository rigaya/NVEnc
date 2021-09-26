// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
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

#pragma once
#ifndef __RGY_TCHAR_H__
#define __RGY_TCHAR_H__

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <tchar.h>
#else
#include <cstddef>
#include <cstring>
#include <cstdio>

typedef char TCHAR;
#define _T(x) x
#define _tmain main
#define _tcslen strlen
#define _ftprintf fprintf
#define _stscanf_s sscanf
#define _stscanf sscanf
#define _tcscmp strcmp
#define _tcsnccmp strncmp
#define _tcsicmp strcasecmp
#define _tcschr strchr
#define _tcsrchr strrchr
#define _tcsstr strstr
#define _tcscat_s strcat_s
#define _tcstol strtol
#define _tcsdup strdup
#define _tfopen fopen
#define _tfopen_s fopen_s
#define _stprintf_s sprintf_s
#define _vsctprintf _vscprintf
#define _vstprintf_s _vsprintf_s
#define _tcstok_s strtok_s
#define _tcserror strerror
#define _fgetts fgets
#define _tcscpy strcpy
#define _tcsncpy strncpy
#define _tremove remove
#define _trename rename
#define _istalpha isalpha
#define _tcsftime strftime

#define _SH_DENYRW      0x10    // deny read/write mode
#define _SH_DENYWR      0x20    // deny write mode
#define _SH_DENYRD      0x30    // deny read mode
#define _SH_DENYNO      0x40    // deny none mode
#define _SH_SECURE      0x80    // secure mode

static inline FILE *_tfsopen(const TCHAR *filename, const TCHAR *mode, int shflag) {
    return fopen(filename, mode);
}

static inline char *_tcscpy_s(TCHAR *dst, const TCHAR *src) {
    return strcpy(dst, src);
}

static inline char *_tcscpy_s(TCHAR *dst, size_t size, const TCHAR *src) {
    return strcpy(dst, src);
}
#endif //#if defined(_WIN32) || defined(_WIN64)

#include <string>

typedef std::basic_string<TCHAR> tstring;

#endif // __RGY_TCHAR_H__
