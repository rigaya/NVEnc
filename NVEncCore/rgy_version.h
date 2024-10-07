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

#define VER_FILEVERSION             0,7,70,0
#define VER_STR_FILEVERSION          "7.70"
#define VER_STR_FILEVERSION_TCHAR _T("7.70")

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
#define ENCODER_MPP    0
#define CLFILTERS_AUF  0

#define CUVID_DISABLE_CROP 1

#define AV1_TIMESTAMP_OVERRIDE 1

#define ENABLE_AVCODEC_OUT_THREAD 1
#define ENABLE_AVCODEC_AUDPROCESS_THREAD 1

#define ENAVLE_LIBAV_DOVI_PARSER 0

#define ENABLE_NVENC_SDK_TUNE 0

#define ENABLE_NVTX 0

#define GPU_VENDOR "NVIDIA"

#define ENABLE_DOVI_METADATA_OPTIONS 1
#define ENABLE_KEYFRAME_INSERT 1

#if defined(_WIN32) || defined(_WIN64)
#define NV_DRIVER_VER_MIN 418081
#define ENABLE_OPENCL 1
#define ENABLE_CPP_REGEX 1
#define ENABLE_DTL 1
#define ENABLE_PERF_COUNTER 1
#define AV_CHANNEL_LAYOUT_STRUCT_AVAIL 1
#define AV_FRAME_DURATION_AVAIL 1
#define AVCODEC_PAR_CODED_SIDE_DATA_AVAIL 1
#define ENABLE_LIBASS_SUBBURN 1
#define ENABLE_D3D11 1
#define ENABLE_D3D11_DEVINFO_WMI 1
#define ENABLE_LIBPLACEBO 1
#define ENABLE_LIBDOVI 1

#ifndef ENABLE_NVOFFRUC_HEADER
#define ENABLE_NVOFFRUC_HEADER 0
#endif

#ifdef _M_IX86
#define ENABLE_NVML 0
#define ENABLE_NVRTC 0
#define ENABLE_VMAF 0
#define ENABLE_NVVFX 0
#define ENABLE_NVOFFRUC 0
#define ENABLE_NVSDKNGX 0
#else
#define ENABLE_NVML 1
#define ENABLE_NVRTC 1
#define ENABLE_VMAF 1
#define ENABLE_NVVFX 1
#define ENABLE_NVOFFRUC 1
#define ENABLE_NVSDKNGX 1
#endif

#define ENABLE_VPP_SMOOTH_QP_FRAME 0
#define ENABLE_AVOID_IDLE_CLOCK 0

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
#define ENABLE_LIBAVDEVICE        0
#define ENABLE_CAPTION2ASS        0
#define ENABLE_AUTO_PICSTRUCT     0
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
#define ENABLE_LIBAVDEVICE        1
#define ENABLE_CAPTION2ASS        0
#define ENABLE_AUTO_PICSTRUCT     1
#endif

#else //#if defined(WIN32) || defined(WIN64)
#define NV_DRIVER_VER_MIN 418030
#include "rgy_config.h"
#define ENCODER_NAME               "NVEnc"
#define DECODER_NAME               "cuvid"
#define FOR_AUO                    0
#define ENABLE_RAW_READER          1
#define ENABLE_NVML                1
#define ENABLE_NVRTC               1
#define ENABLE_NVVFX               0
#define ENABLE_CAPTION2ASS         0
#define ENABLE_VPP_SMOOTH_QP_FRAME 0
#define ENABLE_VMAF                0
#define ENABLE_NVOFFRUC            0
#define ENABLE_D3D11               0
#define ENABLE_D3D11_DEVINFO_WMI   0
#define ENABLE_NVSDKNGX            0
#define ENABLE_LIBPLACEBO          0
#endif // #if defined(WIN32) || defined(WIN64)

#endif //__RGY_CONFIG_H__
