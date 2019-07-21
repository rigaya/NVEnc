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

#include "rgy_avutil.h"
#include "rgy_caption.h"

static const int BITSTREAM_BUFFER_SIZE =  4 * 1024 * 1024;
static const int OUTPUT_BUF_SIZE       = 16 * 1024 * 1024;

static const int RGY_DEFAULT_PERF_MONITOR_INTERVAL = 500;
static const int DEFAULT_IGNORE_DECODE_ERROR = 10;

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
    int nAudioSourceCount;
    TCHAR **ppAudioSourceList;
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


#endif //__RGY_PRM_H__
