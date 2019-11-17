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
// ------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_CONFIG_H__
#define __RGY_CONFIG_H__

#define VER_FILEVERSION              0,4,56,0
#define VER_STR_FILEVERSION          "4.56"
#define VER_STR_FILEVERSION_TCHAR _T("4.56")

#ifdef _M_IX86
#define BUILD_ARCH_STR _T("x86")
#else
#define BUILD_ARCH_STR _T("x64")
#endif

static const int HW_TIMEBASE = 100000;

#if _UNICODE
const wchar_t *get_encoder_version();
#else
const char *get_encoder_version();
#endif

#define ENCODER_QSV    0
#define ENCODER_NVENC  1
#define ENCODER_VCEENC 0

#define ENABLE_OPENCL 1

#define CUVID_DISABLE_CROP 1

#define ENABLE_AVCODEC_OUT_THREAD 1
#define ENABLE_AVCODEC_AUDPROCESS_THREAD 1
#define ENABLE_CPP_REGEX 1
#define ENABLE_DTL 1

#define ENABLE_NVTX 0

#ifdef _M_IX86
#define ENABLE_NVML 0
#define ENABLE_NVRTC 0
#else
#define ENABLE_NVML 1
#define ENABLE_NVRTC 1
#endif

#define GPU_VENDOR "NVIDIA"

#define NV_DRIVER_VER_MIN 418810

#ifdef NVENC_AUO
#define ENCODER_NAME  "NVEnc"
#define AUO_NAME      "NVEnc.auo"
#define FOR_AUO                   1
#define ENABLE_RAW_READER         0
#define ENABLE_AVI_READER         0
#define ENABLE_AVISYNTH_READER    0
#define ENABLE_VAPOURSYNTH_READER 0
#define ENABLE_AVSW_READER        0
#define ENABLE_SM_READER          0
#else
#define ENCODER_NAME "NVEncC"
#define DECODER_NAME "cuvid"
#define FOR_AUO                   0
#define ENABLE_RAW_READER         1
#define ENABLE_AVI_READER         1
#define ENABLE_AVISYNTH_READER    1
#define ENABLE_VAPOURSYNTH_READER 1
#define ENABLE_AVSW_READER        1
#define ENABLE_SM_READER          1
#endif

#endif //__RGY_CONFIG_H__
