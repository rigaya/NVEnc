// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
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
#ifndef __RGY_AVUTIL_H__
#define __RGY_AVUTIL_H__

#include "rgy_version.h"

#if ENABLE_AVSW_READER
#include <algorithm>

#pragma warning (push)
#pragma warning (disable: 4244)
#pragma warning (disable: 4819)
extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/opt.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
}
#pragma comment (lib, "avcodec.lib")
#pragma comment (lib, "avformat.lib")
#pragma comment (lib, "avutil.lib")
#pragma comment (lib, "swresample.lib")
#pragma comment (lib, "avfilter.lib")
#pragma warning (pop)

#include "rgy_util.h"

#if _DEBUG
#define RGY_AV_LOG_LEVEL AV_LOG_WARNING
#else
#define RGY_AV_LOG_LEVEL AV_LOG_ERROR
#endif

typedef struct CodecMap {
    AVCodecID avcodec_id;   //avcodecのコーデックID
    RGY_CODEC rgy_codec; //QSVのfourcc
} CodecMap;

//QSVでデコード可能なコーデックのリスト
static const CodecMap HW_DECODE_LIST[] = {
    { AV_CODEC_ID_H264,       RGY_CODEC_H264 },
    { AV_CODEC_ID_HEVC,       RGY_CODEC_HEVC },
    { AV_CODEC_ID_MPEG2VIDEO, RGY_CODEC_MPEG2 },
    { AV_CODEC_ID_VP8,        RGY_CODEC_VP8 },
    { AV_CODEC_ID_VP9,        RGY_CODEC_VP9 },
#if ENCODER_QSV
    { AV_CODEC_ID_VC1,        RGY_CODEC_VC1 },
#endif
#if ENCODER_NVENC
    { AV_CODEC_ID_MPEG1VIDEO, RGY_CODEC_MPEG1 },
#endif
    //{ AV_CODEC_ID_WMV3,       RGY_CODEC_VC1   },
    //{ AV_CODEC_ID_MPEG4,      RGY_CODEC_MPEG4   },
};

static const AVPixelFormat HW_DECODE_PIXFMT_LIST[] ={
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_YUVJ420P,
    AV_PIX_FMT_NV12,
#if ENCODER_QSV
    AV_PIX_FMT_YUV420P10LE,
#endif
};

static const TCHAR *RGY_AVCODEC_AUTO = _T("auto");
static const TCHAR *RGY_AVCODEC_COPY = _T("copy");

static const int AVQSV_DEFAULT_AUDIO_BITRATE = 192;

static inline bool av_isvalid_q(AVRational q) {
    return q.den * q.num != 0;
}

static inline bool avcodecIsCopy(const TCHAR *codec) {
    return codec == nullptr || 0 == _tcsicmp(codec, RGY_AVCODEC_COPY);
}
static inline bool avcodecIsAuto(const TCHAR *codec) {
    return codec != nullptr && 0 == _tcsicmp(codec, RGY_AVCODEC_AUTO);
}

//AV_LOG_TRACE    56 - RGY_LOG_TRACE -3
//AV_LOG_DEBUG    48 - RGY_LOG_DEBUG -2
//AV_LOG_VERBOSE  40 - RGY_LOG_MORE  -1
//AV_LOG_INFO     32 - RGY_LOG_INFO   0
//AV_LOG_WARNING  24 - RGY_LOG_WARN   1
//AV_LOG_ERROR    16 - RGY_LOG_ERROR  2
static inline int log_level_av2rgy(int level) {
    return clamp((AV_LOG_INFO / 8) - (level / 8), RGY_LOG_TRACE, RGY_LOG_ERROR);
}

static inline int log_level_rgy2av(int level) {
    return clamp(AV_LOG_INFO - level * 8, AV_LOG_QUIET, AV_LOG_TRACE);
}

//"<mes> for codec"型のエラーメッセージを作成する  
static tstring errorMesForCodec(const TCHAR *mes, AVCodecID targetCodec) {
    return mes + tstring(_T(" for ")) + char_to_tstring(avcodec_get_name(targetCodec)) + tstring(_T(".\n"));
};

static const AVRational HW_NATIVE_TIMEBASE = { 1, HW_TIMEBASE };
static const TCHAR *AVCODEC_DLL_NAME[] = {
    _T("avcodec-57.dll"), _T("avformat-57.dll"), _T("avutil-55.dll"), _T("avfilter-6.dll"), _T("swresample-2.dll")
};

enum RGYAVCodecType : uint32_t {
    RGY_AVCODEC_DEC = 0x01,
    RGY_AVCODEC_ENC = 0x02,
};

enum RGYAVFormatType : uint32_t {
    RGY_AVFORMAT_DEMUX = 0x01,
    RGY_AVFORMAT_MUX   = 0x02,
};

//NV_ENC_PIC_STRUCTから、AVFieldOrderを返す
AVFieldOrder picstrcut_rgy_to_avfieldorder(RGY_PICSTRUCT picstruct);

//avcodecのエラーを表示
tstring qsv_av_err2str(int ret);

//コーデックの種類を表示
tstring get_media_type_string(AVCodecID codecId);

//必要なavcodecのdllがそろっているかを確認
bool check_avcodec_dll();

//avcodecのdllが存在しない場合のエラーメッセージ
tstring error_mes_avcodec_dll_not_found();

//avcodecのライセンスがLGPLであるかどうかを確認
bool checkAvcodecLicense();

//avqsvでサポートされている動画コーデックを表示
tstring getHWDecSupportedCodecList();

//利用可能な音声エンコーダ/デコーダを表示
tstring getAVCodecs(RGYAVCodecType flag);

//利用可能なフォーマットを表示
tstring getAVFormats(RGYAVFormatType flag);

//利用可能なフィルターを表示
tstring getAVFilters();

//チャンネルレイアウトを表示
std::string getChannelLayoutChar(int channels, uint64_t channel_layout);
tstring getChannelLayoutString(int channels, uint64_t channel_layout);

//利用可能なプロトコル情報のリストを取得
vector<std::string> getAVProtocolList(int bOutput);

//利用可能なプロトコル情報を取得
tstring getAVProtocols();

//protocolを使用
bool usingAVProtocols(std::string filename, int bOutput);

//avformatのネットワークを初期化する
bool avformatNetworkInit();

//avformatのネットワークを閉じる
void avformatNetworkDeinit();

//バージョン情報の取得
tstring getAVVersions();

MAP_PAIR_0_1_PROTO(csp, avpixfmt, AVPixelFormat, rgy, RGY_CSP);

#endif //ENABLE_AVSW_READER

#endif //__RGY_AVUTIL_H__
