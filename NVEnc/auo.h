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
#include "rgy_tchar.h"

#if _M_X64
#define AVIUTL_TARGET_VER 2
typedef wchar_t aviutlchar;
#include "output2.h"
#define OUTPUT_INFO_FLAG_VIDEO (OUTPUT_INFO::FLAG_VIDEO)
#define OUTPUT_INFO_FLAG_AUDIO (OUTPUT_INFO::FLAG_AUDIO)
#define func_get_video_ex func_get_video
#else
#define AVIUTL_TARGET_VER 1
typedef char aviutlchar;
#include "output.h"
#endif

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
};

#if ENCODER_X264
static const char *ENCODER_NAME   = "x264";
static const wchar_t *ENCODER_NAME_W = L"x264";
static const TCHAR *ENOCDER_RAW_EXT = _T(".264");
static const char *ENCODER_APP_NAME = ENCODER_NAME;
static const wchar_t *ENCODER_APP_NAME_W = ENCODER_NAME_W;
static const TCHAR *ENCODER_REPLACE_MACRO = _T("%{x264path}");
static const char *const OUTPUT_FILE_EXT[]        = {  ".mp4",     ".mkv",     ".264"    };
static const char *const OUTPUT_FILE_EXT_FILTER[] = { "*.mp4",    "*.mkv",    "*.264"    };
static const char *const OUTPUT_FILE_EXT_DESC[]   = { "mp4 file", "mkv file", "raw file" };
#elif ENCODER_X265
static const char *ENCODER_NAME   = "x265";
static const wchar_t *ENCODER_NAME_W   = L"x265";
static const TCHAR *ENOCDER_RAW_EXT = _T(".265");
static const char *ENCODER_APP_NAME = ENCODER_NAME;
static const wchar_t *ENCODER_APP_NAME_W = ENCODER_NAME_W;
static const TCHAR *ENCODER_REPLACE_MACRO = _T("%{x265path}");
static const char *const OUTPUT_FILE_EXT[]        = {  ".mp4",     ".mkv",     ".265"    };
static const char *const OUTPUT_FILE_EXT_FILTER[] = { "*.mp4",    "*.mkv",    "*.265"    };
static const char *const OUTPUT_FILE_EXT_DESC[]   = { "mp4 file", "mkv file", "raw file" };
#elif ENCODER_SVTAV1
static const char *ENCODER_NAME   = "svt-av1";
static const wchar_t *ENCODER_NAME_W   = L"svt-av1";
static const TCHAR *ENOCDER_RAW_EXT = _T(".av1");
static const char *ENCODER_APP_NAME = "SvtAv1EncApp";
static const wchar_t *ENCODER_APP_NAME_W = L"SvtAv1EncApp";
static const TCHAR *ENCODER_REPLACE_MACRO = _T("%{svtav1path}");
static const char *const OUTPUT_FILE_EXT[]        = {  ".mp4",     ".mkv",     ".av1"    };
static const char *const OUTPUT_FILE_EXT_FILTER[] = { "*.mp4",    "*.mkv",    "*.av1"    };
static const char *const OUTPUT_FILE_EXT_DESC[]   = { "mp4 file", "mkv file", "raw file" };
#elif ENCODER_FFMPEG
static const char    *ENCODER_NAME   =  "ffmpeg";
static const wchar_t *ENCODER_NAME_W = L"ffmpeg";
static const TCHAR    *ENOCDER_RAW_EXT = _T(".264");
static const char    *ENCODER_APP_NAME = "ffmpeg";
static const wchar_t *ENCODER_APP_NAME_W = L"ffmpeg";
static const TCHAR    *ENCODER_REPLACE_MACRO = _T("%{ffmpegpath}");
#elif ENCODER_QSV
static const wchar_t *ENCODER_NAME_W = L"QSV";
static const TCHAR    *ENOCDER_RAW_EXT = _T(".264");
static const char    *ENCODER_APP_NAME = "QSVEnc";
static const wchar_t *ENCODER_APP_NAME_W = L"QSVEnc";
static const TCHAR    *ENCODER_REPLACE_MACRO = _T("%{qsvenccpath}");
static const char *const OUTPUT_FILE_EXT[]        = {  ".mp4",     ".mkv",     ".264"    };
static const char *const OUTPUT_FILE_EXT_FILTER[] = { "*.mp4",    "*.mkv",    "*.264"    };
static const char *const OUTPUT_FILE_EXT_DESC[]   = { "mp4 file", "mkv file", "raw file" };
#elif ENCODER_NVENC
static const wchar_t *ENCODER_NAME_W = L"NVENC";
static const TCHAR    *ENOCDER_RAW_EXT = _T(".264");
static const char    *ENCODER_APP_NAME = "NVEnc";
static const wchar_t *ENCODER_APP_NAME_W = L"NVEnc";
static const TCHAR    *ENCODER_REPLACE_MACRO = _T("%{nvenccpath}");
static const char *const OUTPUT_FILE_EXT[]        = {  ".mp4",     ".mkv",     ".264"    };
static const char *const OUTPUT_FILE_EXT_FILTER[] = { "*.mp4",    "*.mkv",    "*.264"    };
static const char *const OUTPUT_FILE_EXT_DESC[]   = { "mp4 file", "mkv file", "raw file" };
#elif ENCODER_VCEENC
static const wchar_t *ENCODER_NAME_W = L"VCE";
static const TCHAR    *ENOCDER_RAW_EXT = _T(".264");
static const char    *ENCODER_APP_NAME = "VCEEnc";
static const wchar_t *ENCODER_APP_NAME_W = L"VCEEnc";
static const TCHAR    *ENCODER_REPLACE_MACRO = _T("%{vceenccpath}");
static const char *const OUTPUT_FILE_EXT[]        = {  ".mp4",     ".mkv",     ".264"    };
static const char *const OUTPUT_FILE_EXT_FILTER[] = { "*.mp4",    "*.mkv",    "*.264"    };
static const char *const OUTPUT_FILE_EXT_DESC[]   = { "mp4 file", "mkv file", "raw file" };
#else
static_assert(false);
#endif

enum {
    MUXER_DISABLED = VIDEO_OUTPUT_DISABLED,
    MUXER_MP4      = VIDEO_OUTPUT_MP4,
    MUXER_MKV      = VIDEO_OUTPUT_MKV,
    MUXER_TC2MP4   = VIDEO_OUTPUT_MP4 + 2,
    MUXER_MP4_RAW,
    MUXER_INTERNAL,
    MUXER_MAX_COUNT,
};

enum {
    AUO_RESULT_SUCCESS    = 0x0000,
    AUO_RESULT_ERROR      = 0x0001,
    AUO_RESULT_ABORT      = 0x0002,
    AUO_RESULT_WARNING    = 0x0004,
};
typedef DWORD AUO_RESULT;

typedef struct AUO_FONT_INFO {
    wchar_t name[256]; //フォント名(family name)
    double size;      //フォントサイズ
    int    style;     //フォントスタイル
} AUO_FONT_INFO;

void write_log_line_fmt(int log_type_index, const wchar_t *format, ...);
void write_log_auo_line_fmt(int log_type_index, const wchar_t *format, ...);
void write_log_auo_enc_time(const wchar_t *mes, DWORD time);

int load_lng(const TCHAR *lang);
const aviutlchar *get_auo_version_info();
std::wstring get_last_out_stg_appendix();

bool checkIfModuleLoaded(const wchar_t *moduleName);

#endif //_AUO_H_
