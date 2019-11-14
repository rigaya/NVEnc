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
// --------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_PRM_H__
#define __RGY_PRM_H__

#include "rgy_def.h"
#include "rgy_caption.h"
#include "rgy_simd.h"
#include "rgy_hdr10plus.h"

static const int BITSTREAM_BUFFER_SIZE =  4 * 1024 * 1024;
static const int OUTPUT_BUF_SIZE       = 16 * 1024 * 1024;

static const int RGY_DEFAULT_PERF_MONITOR_INTERVAL = 500;
static const int DEFAULT_IGNORE_DECODE_ERROR = 10;

struct AudioSelect {
    int      trackID;         //選択した音声トラックのリスト 1,2,...(1から連番で指定)
    tstring  decCodecPrm;     //音声エンコードのデコーダのパラメータ
    tstring  encCodec;        //音声エンコードのコーデック
    tstring  encCodecPrm;     //音声エンコードのコーデックのパラメータ
    tstring  encCodecProfile; //音声エンコードのコーデックのプロファイル
    int      encBitrate;      //音声エンコードに選択した音声トラックのビットレート
    int      encSamplingRate;         //サンプリング周波数
    tstring  extractFilename;      //抽出する音声のファイル名のリスト
    tstring  extractFormat;        //抽出する音声ファイルのフォーマット
    tstring  filter;               //音声フィルタ
    uint64_t streamChannelSelect[MAX_SPLIT_CHANNELS]; //入力音声の使用するチャンネル
    uint64_t streamChannelOut[MAX_SPLIT_CHANNELS];    //出力音声のチャンネル

    AudioSelect();
    ~AudioSelect() {};
};

struct AudioSource {
    tstring filename;
    std::map<int, AudioSelect> select;

    AudioSource();
    ~AudioSource() {};
};

struct SubtitleSelect {
    int trackID;
    tstring encCodec;
    tstring encCodecPrm;
    tstring decCodecPrm;
    bool asdata;

    SubtitleSelect();
    ~SubtitleSelect() {};
};

struct SubSource {
    tstring filename;
    std::map<int, SubtitleSelect> select;

    SubSource();
    ~SubSource() {};
};

struct DataSelect {
    int trackID;

    DataSelect();
    ~DataSelect() {};
};

struct RGYParamCommon {
    tstring inputFilename;        //入力ファイル名
    tstring outputFilename;       //出力ファイル名
    tstring muxOutputFormat;   //出力フォーマット

    std::string maxCll;
    std::string masterDisplay;
    tstring dynamicHdr10plusJson;
    std::string videoCodecTag;
    float seekSec;               //指定された秒数分先頭を飛ばす
    int nSubtitleSelectCount;
    SubtitleSelect **ppSubtitleSelectList;
    std::vector<SubSource> subSource;
    std::vector<AudioSource> audioSource;
    int nAudioSelectCount; //pAudioSelectの数
    AudioSelect **ppAudioSelectList;
    int        nDataSelectCount;
    DataSelect **ppDataSelectList;
    int audioResampler;
    int demuxAnalyzeSec;
    int AVMuxTarget;                       //RGY_MUX_xxx
    int videoTrack;
    int videoStreamId;
    int nTrimCount;
    sTrim *pTrimList;
    bool copyChapter;
    bool keyOnChapter;
    C2AFormat caption2ass;
    int audioIgnoreDecodeError;
    muxOptList *muxOpt;
    bool disableMp4Opt;
    tstring chapterFile;
    tstring keyFile;
    TCHAR *AVInputFormat;
    RGYAVSync AVSyncMode;     //avsyncの方法 (NV_AVSYNC_xxx)


    int outputBufSizeMB;         //出力バッファサイズ

    RGYParamCommon();
    ~RGYParamCommon();
};

struct RGYParamControl {
    int threadCsp;
    int simdCsp;
    tstring logfile;              //ログ出力先
    int loglevel;                 //ログ出力レベル
    tstring logFramePosList;     //framePosList出力先
    TCHAR *logMuxVidTsFile;
    int threadOutput;
    int threadAudio;
    int threadInput;
    int procSpeedLimit;      //処理速度制限 (0で制限なし)
    int64_t perfMonitorSelect;
    int64_t perfMonitorSelectMatplot;
    int     perfMonitorInterval;

    RGYParamControl();
    ~RGYParamControl();
};

bool trim_active(const sTrimParam *pTrim);
std::pair<bool, int> frame_inside_range(int frame, const std::vector<sTrim> &trimList);
bool rearrange_trim_list(int frame, int offset, std::vector<sTrim> &trimList);

const CX_DESC list_simd[] = {
    { _T("auto"),     -1  },
    { _T("none"),     NONE },
    { _T("sse2"),     SSE2 },
    { _T("sse3"),     SSE3|SSE2 },
    { _T("ssse3"),    SSSE3|SSE3|SSE2 },
    { _T("sse41"),    SSE41|SSSE3|SSE3|SSE2 },
    { _T("avx"),      AVX|SSE42|SSE41|SSSE3|SSE3|SSE2 },
    { _T("avx2"),     AVX2|AVX|SSE42|SSE41|SSSE3|SSE3|SSE2 },
    { NULL, 0 }
};

template <uint32_t size>
static bool bSplitChannelsEnabled(uint64_t(&streamChannels)[size]) {
    bool bEnabled = false;
    for (uint32_t i = 0; i < size; i++) {
        bEnabled |= streamChannels[i] != 0;
    }
    return bEnabled;
}

template <uint32_t size>
static void setSplitChannelAuto(uint64_t(&streamChannels)[size]) {
    for (uint32_t i = 0; i < size; i++) {
        streamChannels[i] = ((uint64_t)1) << i;
    }
}

template <uint32_t size>
static bool isSplitChannelAuto(uint64_t(&streamChannels)[size]) {
    bool isAuto = true;
    for (uint32_t i = 0; isAuto && i < size; i++) {
        isAuto &= (streamChannels[i] == (((uint64_t)1) << i));
    }
    return isAuto;
}

unique_ptr<RGYHDR10Plus> initDynamicHDR10Plus(const tstring &dynamicHdr10plusJson, shared_ptr<RGYLog> log);

#endif //__RGY_PRM_H__
