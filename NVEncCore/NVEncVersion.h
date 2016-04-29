// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2014-2016 rigaya
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
// ------------------------------------------------------------------------------------------

#pragma once

#define VER_FILEVERSION              0,2,6,0
#define VER_STR_FILEVERSION          "2.06"
#define VER_STR_FILEVERSION_TCHAR _T("2.06")

#ifdef _M_IX86
#define BUILD_ARCH_STR _T("x86")
#else
#define BUILD_ARCH_STR _T("x64")
#endif

#define ENABLE_AVCODEC_OUT_THREAD 1
#define ENABLE_AVCODEC_AUDPROCESS_THREAD 1

#ifdef NVENC_AUO
#define FOR_AUO    1
#define RAW_READER 0
#define AVI_READER 0
#define AVS_READER 0
#define VPY_READER 0
#define ENABLE_AVCUVID_READER 0
#else
#define FOR_AUO    0
#define RAW_READER 1
#define AVI_READER 0
#define AVS_READER 1
#define VPY_READER 1
#define ENABLE_AVCUVID_READER 1
#endif

