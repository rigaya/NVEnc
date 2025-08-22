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

#ifndef _AUO_VIDEO_H_
#define _AUO_VIDEO_H_

#include "auo.h"
#include "convert.h"
#include "auo_conf.h"
#include "auo_system.h"
#include "rgy_tchar.h"

typedef struct {
    DWORD FOURCC;   //FOURCC
    DWORD size;  //1ピクセルあたりバイト数
} COLORFORMAT_DATA;

enum {
    CF_YUY2 = 0,
    CF_YC48 = 1,
    CF_RGB  = 2,
    CF_LW48 = 3,
    CF_PA64 = 4,
    CF_HF64 = 5,
};
static const TCHAR * const CF_NAME[] = { _T("YUY2"), _T("YC48"), _T("RGB"), _T("LW48"), _T("PA64"), _T("HF64") };

#ifndef MAKEFOURCC
#define MAKEFOURCC(ch0, ch1, ch2, ch3)                              \
                ((DWORD)(BYTE)(ch0) | ((DWORD)(BYTE)(ch1) << 8) |   \
                ((DWORD)(BYTE)(ch2) << 16) | ((DWORD)(BYTE)(ch3) << 24 ))
#endif

static const COLORFORMAT_DATA COLORFORMATS[] = {
    { MAKEFOURCC('Y', 'U', 'Y', '2'), 2 }, //YUY2
    { MAKEFOURCC('Y', 'C', '4', '8'), 6 }, //YC48
    { NULL,                           3 }, //RGB
    { MAKEFOURCC('L', 'W', '4', '8'), 6 }, //LW48
    { MAKEFOURCC('P', 'A', '6', '4'), 8 }, //PA64
    { MAKEFOURCC('H', 'F', '6', '4'), 8 }  //HF64
};

static const int DROP_FRAME_FLAG = INT_MAX;

BOOL check_videnc_mp4_output(const TCHAR *exe_path, const TCHAR *temp_filename);

BOOL setup_afsvideo(const OUTPUT_INFO *oip, const SYSTEM_DATA *sys_dat, CONF_GUIEX *conf, PRM_ENC *pe);
void close_afsvideo(PRM_ENC *pe);

AUO_RESULT video_output(CONF_GUIEX *conf, const OUTPUT_INFO *oip, PRM_ENC *pe, const SYSTEM_DATA *sys_dat);

#endif //_AUO_VIDEO_H_