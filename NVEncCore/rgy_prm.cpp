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

#include <set>
#include <iostream>
#include <iomanip>
#include "rgy_util.h"
#include "rgy_avutil.h"
#include "rgy_prm.h"
#include "rgy_err.h"
#include "rgy_perf_monitor.h"

AudioSelect::AudioSelect() :
    trackID(0),
    decCodecPrm(),
    encCodec(),
    encCodecPrm(),
    encCodecProfile(),
    encBitrate(0),
    encSamplingRate(0),
    extractFilename(),
    extractFormat(),
    filter(),
    streamChannelSelect(),
    streamChannelOut() {
    memset(streamChannelSelect, 0, sizeof(streamChannelSelect));
    memset(streamChannelOut, 0, sizeof(streamChannelOut));
}

AudioSource::AudioSource() :
    filename(),
    select() {

}

SubtitleSelect::SubtitleSelect() :
    trackID(0),
    encCodec(),
    encCodecPrm(),
    decCodecPrm(),
    asdata(false) {

}

DataSelect::DataSelect() :
    trackID(0) {

}

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
    audioSource(),
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
    outputBufSizeMB(8) {

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


bool trim_active(const sTrimParam *pTrim) {
    if (pTrim == nullptr) {
        return false;
    }
    if (pTrim->list.size() == 0) {
        return false;
    }
    if (pTrim->list[0].start == 0 && pTrim->list[0].fin == TRIM_MAX) {
        return false;
    }
    return true;
}

//block index (空白がtrimで削除された領域)
//       #0       #0         #1         #1       #2    #2
//   |        |----------|         |----------|     |------
std::pair<bool, int> frame_inside_range(int frame, const std::vector<sTrim> &trimList) {
    int index = 0;
    if (trimList.size() == 0) {
        return std::make_pair(true, index);
    }
    if (frame < 0) {
        return std::make_pair(false, index);
    }
    for (; index < (int)trimList.size(); index++) {
        if (frame < trimList[index].start) {
            return std::make_pair(false, index);
        }
        if (frame <= trimList[index].fin) {
            return std::make_pair(true, index);
        }
    }
    return std::make_pair(false, index);
}

bool rearrange_trim_list(int frame, int offset, std::vector<sTrim> &trimList) {
    if (trimList.size() == 0)
        return true;
    if (frame < 0)
        return false;
    for (uint32_t i = 0; i < trimList.size(); i++) {
        if (trimList[i].start >= frame) {
            trimList[i].start = clamp(trimList[i].start + offset, 0, TRIM_MAX);
        }
        if (trimList[i].fin && trimList[i].fin >= frame) {
            trimList[i].fin = (int)clamp((int64_t)trimList[i].fin + offset, 0, (int64_t)TRIM_MAX);
        }
    }
    return false;
}

#if !FOR_AUO
unique_ptr<RGYHDR10Plus> initDynamicHDR10Plus(const tstring &dynamicHdr10plusJson, shared_ptr<RGYLog> log) {
    unique_ptr<RGYHDR10Plus> hdr10plus;
    if (!PathFileExists(dynamicHdr10plusJson.c_str())) {
        log->write(RGY_LOG_ERROR, _T("Cannot find the file specified : %s.\n"), dynamicHdr10plusJson.c_str());
    } else {
        hdr10plus = std::make_unique<RGYHDR10Plus>();
        auto ret = hdr10plus->init(dynamicHdr10plusJson);
        if (ret == RGY_ERR_NOT_FOUND) {
            log->write(RGY_LOG_ERROR, _T("Cannot find \"%s\" required for --dhdr10-info.\n"), RGYHDR10Plus::HDR10PLUS_GEN_EXE_NAME);
            hdr10plus.reset();
        } else if (ret != RGY_ERR_NONE) {
            log->write(RGY_LOG_ERROR, _T("Failed to initialize hdr10plus reader: %s.\n"), get_err_mes((RGY_ERR)ret));
            hdr10plus.reset();
        }
        log->write(RGY_LOG_DEBUG, _T("initialized hdr10plus reader: %s\n"), dynamicHdr10plusJson.c_str());
    }
    return hdr10plus;
}
#endif
