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

#include "rgy_util.h"
#include "rgy_avutil.h"
#include "rgy_prm.h"

RGYParamCommon::RGYParamCommon() :
    inputFilename(),
    outputFilename(),
    muxOutputFormat(),
    maxCll(),
    masterDisplay(),
    dynamicHdr10plusJson(),
    videoCodecTag(),
    seekSec(0.0f),               //指定された秒数分先頭を飛ばす
    nSubtitleSelectCount(0),
    ppSubtitleSelectList(nullptr),
    nAudioSourceCount(0),
    ppAudioSourceList(nullptr),
    nAudioSelectCount(0), //pAudioSelectの数
    ppAudioSelectList(nullptr),
    nDataSelectCount(0),
    ppDataSelectList(nullptr),
    audioResampler(RGY_RESAMPLER_SWR),
    demuxAnalyzeSec(0),
    AVMuxTarget(RGY_MUX_NONE),                       //RGY_MUX_xxx
    videoTrack(0),
    videoStreamId(0),
    nTrimCount(0),
    pTrimList(nullptr),
    copyChapter(false),
    keyOnChapter(false),
    caption2ass(FORMAT_INVALID),
    audioIgnoreDecodeError(DEFAULT_IGNORE_DECODE_ERROR),
    muxOpt(nullptr),
    chapterFile(),
    AVInputFormat(nullptr),
    AVSyncMode(RGY_AVSYNC_ASSUME_CFR),     //avsyncの方法 (RGY_AVSYNC_xxx)
    outputBufSizeMB(OUTPUT_BUF_SIZE) {

}

RGYParamCommon::~RGYParamCommon() {};

RGYParamControl::RGYParamControl() :
    logfile(),              //ログ出力先
    loglevel(RGY_LOG_INFO),                 //ログ出力レベル
    logFramePosList(),     //framePosList出力先
    logMuxVidTsFile(nullptr),
    threadOutput(RGY_OUTPUT_THREAD_AUTO),
    threadAudio(RGY_INPUT_THREAD_AUTO),
    threadInput(RGY_AUDIO_THREAD_AUTO),
    procSpeedLimit(0),      //処理速度制限 (0で制限なし)
    perfMonitorSelect(0),
    perfMonitorSelectMatplot(0),
    perfMonitorInterval(RGY_DEFAULT_PERF_MONITOR_INTERVAL),
    threadCsp(0),
    simdCsp(-1) {

}
RGYParamControl::~RGYParamControl() {};