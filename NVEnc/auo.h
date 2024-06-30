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

#ifndef _AUO_H_
#define _AUO_H_

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <string>
#include "auo_version.h"

const int   MAX_PATH_LEN          = 1024; //NTFSでは32768文字らしいが...いらんやろ
const int   MAX_APPENDIX_LEN      = 63; //適当

const int   MAX_CMD_LEN           = 16 * 1024; //コマンドラインの最大長はよくわからん

const DWORD AUDIO_BUFFER_DEFAULT  = 48000;
const DWORD AUDIO_BUFFER_MAX      = AUDIO_BUFFER_DEFAULT * 30;

enum {
    VIDEO_OUTPUT_UNKNOWN  = -3,
    VIDEO_OUTPUT_DISABLED = -2,
    VIDEO_OUTPUT_RAW      = -1,
    VIDEO_OUTPUT_MP4      = 0,
    VIDEO_OUTPUT_MKV      = 1,
    VIDEO_OUTPUT_MPEG2    = 3,
};

#if ENCODER_QSV
static const wchar_t *ENCODER_NAME_W = L"QSV";
static const char    *ENOCDER_RAW_EXT = ".264";
static const char    *ENCODER_APP_NAME = "QSVEnc";
static const wchar_t *ENCODER_APP_NAME_W = L"QSVEnc";
static const char    *ENCODER_REPLACE_MACRO = "%{qsvenccpath}";
#elif ENCODER_NVENC
static const wchar_t *ENCODER_NAME_W = L"NVENC";
static const char    *ENOCDER_RAW_EXT = ".264";
static const char    *ENCODER_APP_NAME = "NVEnc";
static const wchar_t *ENCODER_APP_NAME_W = L"NVEnc";
static const char    *ENCODER_REPLACE_MACRO = "%{nvenccpath}";
#elif ENCODER_VCEENC
static const wchar_t *ENCODER_NAME_W = L"VCE";
static const char    *ENOCDER_RAW_EXT = ".264";
static const char    *ENCODER_APP_NAME = "VCEEnc";
static const wchar_t *ENCODER_APP_NAME_W = L"VCEEnc";
static const char    *ENCODER_REPLACE_MACRO = "%{vceenccpath}";
#else
static_assert(false);
#endif

static const char *const OUTPUT_FILE_EXT[]        = {  ".mp4",     ".mkv",     ".264"    };
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

typedef struct AUO_FONT_INFO {
    char   name[256]; //フォント名(family name)
    double size;      //フォントサイズ
    int    style;     //フォントスタイル
} AUO_FONT_INFO;

void write_log_line_fmt(int log_type_index, const wchar_t *format, ...);
void write_log_auo_line_fmt(int log_type_index, const wchar_t *format, ...);
void write_log_auo_enc_time(const wchar_t *mes, DWORD time);

int load_lng(const char *lang);
const char *get_auo_version_info();
std::string get_last_out_stg_appendix();

bool checkIfModuleLoaded(const wchar_t *moduleName);

#endif //_AUO_H_
