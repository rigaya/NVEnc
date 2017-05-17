// -----------------------------------------------------------------------------------------
// QSVEnc by rigaya
// -----------------------------------------------------------------------------------------
//
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
#ifndef __RGY_ERR_H__
#define __RGY_ERR_H__

#define RGY_ERR_QSV 0
#define RGY_ERR_AMF 0
#define RGY_ERR_NV  1

#include "rgy_tchar.h"
#if RGY_ERR_QSV
#include "mfxDefs.h"
#endif
#if RGY_ERR_AMF
#endif
#if RGY_ERR_NV
#include "nvEncodeAPI.h"
#endif

#include <algorithm>

enum RGY_ERR {
    RGY_ERR_NONE                        = 0,
    RGY_ERR_UNKNOWN                     = -1,
    RGY_ERR_NULL_PTR                    = -2,
    RGY_ERR_UNSUPPORTED                 = -3,
    RGY_ERR_MEMORY_ALLOC                = -4,
    RGY_ERR_NOT_ENOUGH_BUFFER           = -5,
    RGY_ERR_INVALID_HANDLE              = -6,
    RGY_ERR_LOCK_MEMORY                 = -7,
    RGY_ERR_NOT_INITIALIZED             = -8,
    RGY_ERR_NOT_FOUND                   = -9,
    RGY_ERR_MORE_DATA                   = -10,
    RGY_ERR_MORE_SURFACE                = -11,
    RGY_ERR_ABORTED                     = -12,
    RGY_ERR_DEVICE_LOST                 = -13,
    RGY_ERR_INCOMPATIBLE_VIDEO_PARAM    = -14,
    RGY_ERR_INVALID_VIDEO_PARAM         = -15,
    RGY_ERR_UNDEFINED_BEHAVIOR          = -16,
    RGY_ERR_DEVICE_FAILED               = -17,
    RGY_ERR_MORE_BITSTREAM              = -18,
    RGY_ERR_INCOMPATIBLE_AUDIO_PARAM    = -19,
    RGY_ERR_INVALID_AUDIO_PARAM         = -20,
    RGY_ERR_GPU_HANG                    = -21,
    RGY_ERR_REALLOC_SURFACE             = -22,
    RGY_ERR_ACCESS_DENIED               = -23,
    RGY_ERR_INVALID_PARAM               = -24,
    RGY_ERR_OUT_OF_RANGE                = -25,
    RGY_ERR_ALREADY_INITIALIZED         = -26,
    RGY_ERR_INVALID_FORMAT              = -27,
    RGY_ERR_WRONG_STATE                 = -28,
    RGY_ERR_FILE_OPEN                   = -29,
    RGY_ERR_INPUT_FULL                  = -30,
    RGY_ERR_INVALID_CODEC               = -31,
    RGY_ERR_INVALID_DATA_TYPE           = -32,
    RGY_ERR_INVALID_RESOLUTION          = -33,
    RGY_ERR_NO_DEVICE                   = -34,
    RGY_ERR_INVALID_DEVICE              = -35,
    RGY_ERR_INVALID_CALL                = -36,
    RGY_ERR_INVALID_VERSION             = -37,
    RGY_ERR_MAP_FAILED                  = -38,

    RGY_WRN_IN_EXECUTION                = 1,
    RGY_WRN_DEVICE_BUSY                 = 2,
    RGY_WRN_VIDEO_PARAM_CHANGED         = 3,
    RGY_WRN_PARTIAL_ACCELERATION        = 4,
    RGY_WRN_INCOMPATIBLE_VIDEO_PARAM    = 5,
    RGY_WRN_VALUE_NOT_CHANGED           = 6,
    RGY_WRN_OUT_OF_RANGE                = 7,
    RGY_WRN_FILTER_SKIPPED              = 10,
    RGY_WRN_INCOMPATIBLE_AUDIO_PARAM    = 11,

    RGY_PRINT_OPTION_DONE = 20,
    RGY_PRINT_OPTION_ERR  = 21,

    RGY_ERR_INVALID_COLOR_FORMAT = -100,

    RGY_ERR_MORE_DATA_SUBMIT_TASK       = -10000,
};

#if RGY_ERR_QSV
mfxStatus err_to_mfx(RGY_ERR err);
RGY_ERR err_to_rgy(mfxStatus err);
#endif //#if RGY_ERR_QSV

#if RGY_ERR_AMF
AMF_ERR err_to_amf(RGYErr err);
RGYErr err_to_rgy(AMF_ERR err);
#endif //#if RGY_ERR_AMF


#if RGY_ERR_NV
NVENCSTATUS err_to_nv(RGY_ERR err);
RGY_ERR err_to_rgy(NVENCSTATUS err);
#endif //#if RGY_ERR_NV

const TCHAR *get_err_mes(RGY_ERR sts);

#endif //__RGY_ERR_H__
