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

#include "rgy_version.h"
#include "rgy_rev.h"
#include "rgy_osdep.h"
#include "rgy_util.h"

#define SSTRING(str) STRING(str)
#define STRING(str) #str

const TCHAR *get_encoder_version() {
    return
        _T(ENCODER_NAME) _T(" (")
        BUILD_ARCH_STR _T(") ") VER_STR_FILEVERSION_TCHAR _T(" (r") ENCODER_REV _T(") by rigaya, ")  _T(__DATE__) _T(" ") _T(__TIME__)
#if defined(_MSC_VER)
        _T(" (VC ") _T(SSTRING(_MSC_VER))
#elif defined(__clang__)
        _T(" (clang ") _T(SSTRING(__clang_major__)) _T(".") _T(SSTRING(__clang_minor__)) _T(".") _T(SSTRING(__clang_patchlevel__))
#elif defined(__GNUC__)
        _T(" (gcc ") _T(SSTRING(__GNUC__)) _T(".") _T(SSTRING(__GNUC_MINOR__)) _T(".") _T(SSTRING(__GNUC_PATCHLEVEL__))
#else
        _T(" (unknown")
#endif
        _T("/")
#ifdef _WIN32
        _T("Win")
#elif  __linux
        _T("Linux")
#else
        _T("unknown")
#endif
        _T(")");
}
