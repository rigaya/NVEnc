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

#ifndef _AUO_H_
#define _AUO_H_

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

const int   MAX_PATH_LEN          = 1024; //NTFSでは32768文字らしいが...いらんやろ
const int   MAX_APPENDIX_LEN      = 63; //適当

const int   MAX_CMD_LEN           = 8192; //コマンドラインの最大長はよくわからん

const DWORD AUDIO_BUFFER_DEFAULT  = 48000;
const DWORD AUDIO_BUFFER_MAX      = AUDIO_BUFFER_DEFAULT * 30;

enum {
    VIDEO_OUTPUT_DISABLED = -2,
    VIDEO_OUTPUT_RAW      = -1,
    VIDEO_OUTPUT_MP4      = 0,
    VIDEO_OUTPUT_MKV      = 1,
    VIDEO_OUTPUT_MPEG2    = 3,
};

static const char *const OUTPUT_FILE_EXT[]        = {  ".mp4",     ".mkv",     ".264" };
static const char *const OUTPUT_FILE_EXT_FILTER[] = { "*.mp4",    "*.mkv",    "*.264"    };
static const char *const OUTPUT_FILE_EXT_DESC[]   = { "mp4 file", "mkv file", "raw file" };

enum {
    MUXER_DISABLED = VIDEO_OUTPUT_DISABLED,
    MUXER_MP4      = VIDEO_OUTPUT_MP4,
    MUXER_MKV      = VIDEO_OUTPUT_MKV,
    MUXER_TC2MP4   = VIDEO_OUTPUT_MP4 + 2,
    MUXER_MPG      = VIDEO_OUTPUT_MPEG2,
    MUXER_MP4_RAW  = VIDEO_OUTPUT_MP4 + 4,
    MUXER_INTERNAL,
};

enum {
    AUO_RESULT_SUCCESS    = 0x0000,
    AUO_RESULT_ERROR      = 0x0001,
    AUO_RESULT_ABORT      = 0x0002,
    AUO_RESULT_WARNING    = 0x0004,
};
typedef DWORD AUO_RESULT;

typedef struct {
    WCHAR *text;
    DWORD value;
} PRIORITY_CLASS;

const DWORD AVIUTLSYNC_PRIORITY_CLASS = 0;

const PRIORITY_CLASS priority_table[] = {
    {L"AviutlSync",       AVIUTLSYNC_PRIORITY_CLASS   },
    {L"higher",           HIGH_PRIORITY_CLASS         },
    {L"high",             ABOVE_NORMAL_PRIORITY_CLASS },
    {L"normal",           NORMAL_PRIORITY_CLASS       },
    {L"low",              BELOW_NORMAL_PRIORITY_CLASS },
    {L"lower",            IDLE_PRIORITY_CLASS         },
    {L"",                 NORMAL_PRIORITY_CLASS       },
    {L"realtime(非推奨)", REALTIME_PRIORITY_CLASS     },
    {NULL,                0                           }
};

typedef struct {
    char   name[256]; //フォント名(family name)
    double size;      //フォントサイズ
    int    style;     //フォントスタイル
} AUO_FONT_INFO;

void write_log_auo_line_fmt(int log_type_index, const char *format, ... );
void write_log_auo_enc_time(const char *mes, DWORD time);

#endif //_AUO_H_
