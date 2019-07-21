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



static int getAudioTrackIdx(const RGYParamCommon *common, int iTrack) {
    for (int i = 0; i < common->nAudioSelectCount; i++) {
        if (iTrack == common->ppAudioSelectList[i]->trackID) {
            return i;
        }
    }
    return -1;
}

static int getFreeAudioTrack(const RGYParamCommon *common) {
    for (int iTrack = 1;; iTrack++) {
        if (0 > getAudioTrackIdx(common, iTrack)) {
            return iTrack;
        }
    }
#ifndef _MSC_VER
    return -1;
#endif //_MSC_VER
}

static int getSubTrackIdx(const RGYParamCommon *common, int iTrack) {
    for (int i = 0; i < common->nSubtitleSelectCount; i++) {
        if (iTrack == common->ppSubtitleSelectList[i]->trackID) {
            return i;
        }
    }
    return -1;
}

static int getDataTrackIdx(const RGYParamCommon *common, int iTrack) {
    for (int i = 0; i < common->nDataSelectCount; i++) {
        if (iTrack == common->ppDataSelectList[i]->trackID) {
            return i;
        }
    }
    return -1;
}

int parse_one_input_option(const TCHAR *option_name, const TCHAR *strInput[], int &i, int nArgNum, VideoInfo *input, sArgsData *argData, ParseCmdError &err) {
    if (IS_OPTION("fps")) {
        i++;
        int a[2] ={ 0 };
        if (2 == _stscanf_s(strInput[i], _T("%d/%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d:%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d,%d"), &a[0], &a[1])) {
            input->fpsN = a[0];
            input->fpsD = a[1];
        } else {
            double d;
            if (1 == _stscanf_s(strInput[i], _T("%lf"), &d)) {
                int rate = (int)(d * 1001.0 + 0.5);
                if (rate % 1000 == 0) {
                    input->fpsN = rate;
                    input->fpsD = 1001;
                } else {
                    input->fpsD = 100000;
                    input->fpsN = (int)(d * input->fpsD + 0.5);
                    rgy_reduce(input->fpsN, input->fpsD);
                }
            } else {
                CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
            }
        }
        return 0;
    }
    if (IS_OPTION("input-res")) {
        i++;
        int a[2] ={ 0 };
        if (2 == _stscanf_s(strInput[i], _T("%dx%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d:%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d,%d"), &a[0], &a[1])) {
            input->srcWidth  = a[0];
            input->srcHeight = a[1];
        } else {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("output-res")) {
        i++;
        int a[2] ={ 0 };
        if (2 == _stscanf_s(strInput[i], _T("%dx%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d:%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d,%d"), &a[0], &a[1])) {
            input->dstWidth  = a[0];
            input->dstHeight = a[1];
        } else {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("crop")) {
        i++;
        sInputCrop a ={ 0 };
        if (4 == _stscanf_s(strInput[i], _T("%d,%d,%d,%d"), &a.c[0], &a.c[1], &a.c[2], &a.c[3])
            || 4 == _stscanf_s(strInput[i], _T("%d:%d:%d:%d"), &a.c[0], &a.c[1], &a.c[2], &a.c[3])) {
            memcpy(&input->crop, &a, sizeof(a));
        } else {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("raw")) {
        input->type = RGY_INPUT_FMT_RAW;
        return 0;
    }
    if (IS_OPTION("y4m")) {
        input->type = RGY_INPUT_FMT_Y4M;
#if ENABLE_AVI_READER
        return 0;
    }
    if (IS_OPTION("avi")) {
        input->type = RGY_INPUT_FMT_AVI;
#endif
#if ENABLE_AVISYNTH_READER
        return 0;
    }
    if (IS_OPTION("avs")) {
        input->type = RGY_INPUT_FMT_AVS;
#endif
#if ENABLE_VAPOURSYNTH_READER
        return 0;
    }
    if (IS_OPTION("vpy")) {
        input->type = RGY_INPUT_FMT_VPY;
        return 0;
    }
    if (IS_OPTION("vpy-mt")) {
        input->type = RGY_INPUT_FMT_VPY_MT;
#endif
#if ENABLE_AVSW_READER
        return 0;
    }
    if (IS_OPTION("avcuvid")
        || IS_OPTION("avhw")) {
        input->type = RGY_INPUT_FMT_AVHW;
#endif
        return 0;
    }
    if (IS_OPTION("avsw")) {
        input->type = RGY_INPUT_FMT_AVSW;
        return 0;
    }
    if (IS_OPTION("tff")) {
        input->picstruct = RGY_PICSTRUCT_FRAME_TFF;
        return 0;
    }
    if (IS_OPTION("bff")) {
        input->picstruct = RGY_PICSTRUCT_FRAME_BFF;
        return 0;
    }
    if (IS_OPTION("interlace") || IS_OPTION("interlaced")) {
        i++;
        int value = 0;
        if (get_list_value(list_interlaced, strInput[i], &value)) {
            input->picstruct = (RGY_PICSTRUCT)value;
        } else {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    return -10;
}

int parse_one_common_option(const TCHAR *option_name, const TCHAR *strInput[], int &i, int nArgNum, RGYParamCommon *common, sArgsData *argData, ParseCmdError &err) {

    if (IS_OPTION("input")) {
        i++;
        common->inputFilename = strInput[i];
        return 0;
    }
    if (IS_OPTION("output")) {
        i++;
        common->outputFilename = strInput[i];
        return 0;
    }

    if (   IS_OPTION("input-analyze")
        || IS_OPTION("avcuvid-analyze")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        } else if (value < 0) {
            CMD_PARSE_SET_ERR(strInput[0], _T("input-analyze requires non-negative value."), option_name, strInput[i]);
            return 1;
        } else {
            common->demuxAnalyzeSec = (int)((std::min)(value, USHRT_MAX));
        }
        return 0;
    }
    if (IS_OPTION("video-track")) {
        i++;
        int v = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v)) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (v == 0) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        common->videoTrack = v;
        return 0;
    }
    if (IS_OPTION("video-streamid")) {
        i++;
        int v = 0;
        if (1 != _stscanf_s(strInput[i], _T("%i"), &v)) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        common->videoStreamId = v;
        return 0;
    }
    if (IS_OPTION("video-tag")) {
        i++;
        common->videoCodecTag = tchar_to_string(strInput[i]);
        return 0;
    }
    if (IS_OPTION("trim")) {
        i++;
        auto trim_str_list = split(strInput[i], _T(","));
        std::vector<sTrim> trim_list;
        for (auto trim_str : trim_str_list) {
            sTrim trim;
            if (2 != _stscanf_s(trim_str.c_str(), _T("%d:%d"), &trim.start, &trim.fin) || (trim.fin > 0 && trim.fin < trim.start)) {
                CMD_PARSE_SET_ERR(strInput[0], _T("Invalid Value"), option_name, trim_str.c_str());
                return 1;
            }
            if (trim.fin == 0) {
                trim.fin = TRIM_MAX;
            } else if (trim.fin < 0) {
                trim.fin = trim.start - trim.fin - 1;
            }
            trim_list.push_back(trim);
        }
        if (trim_list.size()) {
            std::sort(trim_list.begin(), trim_list.end(), [](const sTrim& trimA, const sTrim& trimB) { return trimA.start < trimB.start; });
            for (int j = (int)trim_list.size() - 2; j >= 0; j--) {
                if (trim_list[j].fin > trim_list[j+1].start) {
                    trim_list[j].fin = trim_list[j+1].fin;
                    trim_list.erase(trim_list.begin() + j+1);
                }
            }
            common->nTrimCount = (int)trim_list.size();
            common->pTrimList = (sTrim *)malloc(sizeof(common->pTrimList[0]) * trim_list.size());
            memcpy(common->pTrimList, &trim_list[0], sizeof(common->pTrimList[0]) * trim_list.size());
        }
        return 0;
    }
    if (IS_OPTION("seek")) {
        i++;
        int ret = 0;
        int hh = 0, mm = 0;
        float sec = 0.0f;
        if (   3 != (ret = _stscanf_s(strInput[i], _T("%d:%d:%f"),    &hh, &mm, &sec))
            && 2 != (ret = _stscanf_s(strInput[i],    _T("%d:%f"),         &mm, &sec))
            && 1 != (ret = _stscanf_s(strInput[i],       _T("%f"),              &sec))) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (ret <= 2) {
            hh = 0;
        }
        if (ret <= 1) {
            mm = 0;
        }
        if (hh < 0 || mm < 0 || sec < 0) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        if (hh > 0 && mm >= 60) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        mm += hh * 60;
        if (mm > 0 && sec >= 60.0f) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        common->seekSec = sec + mm * 60;
        return 0;
    }
    if (IS_OPTION("audio-source")) {
        i++;
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        size_t audioSourceLen = _tcslen(strInput[i]) + 1;
        TCHAR *pAudioSource = (TCHAR *)malloc(sizeof(strInput[i][0]) * audioSourceLen);
        memcpy(pAudioSource, strInput[i], sizeof(strInput[i][0]) * audioSourceLen);
        common->ppAudioSourceList = (TCHAR **)realloc(common->ppAudioSourceList, sizeof(common->ppAudioSourceList[0]) * (common->nAudioSourceCount + 1));
        common->ppAudioSourceList[common->nAudioSourceCount] = pAudioSource;
        common->nAudioSourceCount++;
        return 0;
    }
    if (IS_OPTION("audio-file")) {
        i++;
        const TCHAR *ptr = strInput[i];
        AudioSelect *pAudioSelect = nullptr;
        int audioIdx = -1;
        int trackId = 0;
        if (_tcschr(ptr, '?') == nullptr || 1 != _stscanf(ptr, _T("%d?"), &trackId)) {
            //トラック番号を適当に発番する (カウントは1から)
            trackId = argData->nParsedAudioFile+1;
            audioIdx = getAudioTrackIdx(common, trackId);
            if (audioIdx < 0 || common->ppAudioSelectList[audioIdx]->extractFilename.length() > 0) {
                trackId = getFreeAudioTrack(common);
                pAudioSelect = new AudioSelect();
                pAudioSelect->trackID = trackId;
            } else {
                pAudioSelect = common->ppAudioSelectList[audioIdx];
            }
        } else if (i <= 0) {
            //トラック番号は1から連番で指定
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid track number"), option_name, strInput[i]);
            return 1;
        } else {
            audioIdx = getAudioTrackIdx(common, trackId);
            if (audioIdx < 0) {
                pAudioSelect = new AudioSelect();
                pAudioSelect->trackID = trackId;
            } else {
                pAudioSelect = common->ppAudioSelectList[audioIdx];
            }
            ptr = _tcschr(ptr, '?') + 1;
        }
        assert(pAudioSelect != nullptr);
        const TCHAR *qtr = _tcschr(ptr, ':');
        if (qtr != NULL && !(ptr + 1 == qtr && qtr[1] == _T('\\'))) {
            pAudioSelect->extractFormat = ptr;
            ptr = qtr + 1;
        }
        size_t filename_len = _tcslen(ptr);
        //ファイル名が""でくくられてたら取り除く
        if (ptr[0] == _T('\"') && ptr[filename_len-1] == _T('\"')) {
            filename_len -= 2;
            ptr++;
        }
        //ファイル名が重複していないかを確認する
        for (int j = 0; j < common->nAudioSelectCount; j++) {
            if (common->ppAudioSelectList[j]->extractFilename.length() > 0
                && 0 == _tcsicmp(common->ppAudioSelectList[j]->extractFilename.c_str(), ptr)) {
                CMD_PARSE_SET_ERR(strInput[0], _T("Same output file name is used more than twice"), option_name, nullptr);
                return 1;
            }
        }

        if (audioIdx < 0) {
            audioIdx = common->nAudioSelectCount;
            //新たに要素を追加
            common->ppAudioSelectList = (AudioSelect **)realloc(common->ppAudioSelectList, sizeof(common->ppAudioSelectList[0]) * (common->nAudioSelectCount + 1));
            common->ppAudioSelectList[common->nAudioSelectCount] = pAudioSelect;
            common->nAudioSelectCount++;
        }
        common->ppAudioSelectList[audioIdx]->extractFilename = ptr;
        argData->nParsedAudioFile++;
        return 0;
    }
    if (IS_OPTION("format") || IS_OPTION("input-output")) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            common->muxOutputFormat = strInput[i];
            if (0 != _tcsicmp(strInput[i], _T("raw"))) {
                common->AVMuxTarget |= RGY_MUX_VIDEO;
            }
        }
        return 0;
    }
    if (IS_OPTION("input-format")) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            common->AVInputFormat = _tcsdup(strInput[i]);
        } else {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
#if ENABLE_AVSW_READER
    auto set_audio_prm = [&](std::function<void(AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr)> func_set) {
        const TCHAR *ptr = nullptr;
        const TCHAR *ptrDelim = nullptr;
        int trackId = 0;
        if (i+1 < nArgNum) {
            if (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0')) {
                i++;
                ptrDelim = _tcschr(strInput[i], _T('?'));
                ptr = (ptrDelim == nullptr) ? strInput[i] : ptrDelim+1;
            }
            if (ptrDelim != nullptr) {
                tstring temp = tstring(strInput[i]).substr(0, ptrDelim - strInput[i]);
                trackId = std::stoi(temp);
            }
        }
        AudioSelect *pAudioSelect = nullptr;
        int audioIdx = getAudioTrackIdx(common, trackId);
        if (audioIdx < 0) {
            pAudioSelect = new AudioSelect();
            if (trackId != 0) {
                //もし、trackID=0以外の指定であれば、
                //これまでalltrackに指定されたパラメータを探して引き継ぐ
                AudioSelect *pAudioSelectAll = nullptr;
                for (int itrack = 0; itrack < common->nAudioSelectCount; itrack++) {
                    if (common->ppAudioSelectList[itrack]->trackID == 0) {
                        pAudioSelectAll = common->ppAudioSelectList[itrack];
                    }
                }
                if (pAudioSelectAll) {
                    *pAudioSelect = *pAudioSelectAll;
                }
            }
            pAudioSelect->trackID = trackId;
        } else {
            pAudioSelect = common->ppAudioSelectList[audioIdx];
        }
        func_set(pAudioSelect, trackId, ptr);
        if (trackId == 0) {
            for (int itrack = 0; itrack < common->nAudioSelectCount; itrack++) {
                func_set(common->ppAudioSelectList[itrack], trackId, ptr);
            }
        }

        if (audioIdx < 0) {
            audioIdx = common->nAudioSelectCount;
            //新たに要素を追加
            common->ppAudioSelectList = (AudioSelect **)realloc(common->ppAudioSelectList, sizeof(common->ppAudioSelectList[0]) * (common->nAudioSelectCount + 1));
            common->ppAudioSelectList[common->nAudioSelectCount] = pAudioSelect;
            common->nAudioSelectCount++;
        }
        return 0;
    };
    auto set_sub_prm = [&](std::function<void(SubtitleSelect *pSubSelect, int trackId, const TCHAR *prmstr)> func_set) {
        const TCHAR *ptr = nullptr;
        const TCHAR *ptrDelim = nullptr;
        int trackId = 0;
        if (i+1 < nArgNum) {
            if (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0')) {
                i++;
                ptrDelim = _tcschr(strInput[i], _T('?'));
                ptr = (ptrDelim == nullptr) ? strInput[i] : ptrDelim+1;
            }
            if (ptrDelim != nullptr) {
                tstring temp = tstring(strInput[i]).substr(0, ptrDelim - strInput[i]);
                trackId = std::stoi(temp);
            }
        }
        SubtitleSelect *pSubSelect = nullptr;
        int subIdx = getSubTrackIdx(common, trackId);
        if (subIdx < 0) {
            pSubSelect = new SubtitleSelect();
            if (trackId != 0) {
                //もし、trackID=0以外の指定であれば、
                //これまでalltrackに指定されたパラメータを探して引き継ぐ
                SubtitleSelect *pSubSelectAll = nullptr;
                for (int itrack = 0; itrack < common->nSubtitleSelectCount; itrack++) {
                    if (common->ppSubtitleSelectList[itrack]->trackID == 0) {
                        pSubSelectAll = common->ppSubtitleSelectList[itrack];
                    }
                }
                if (pSubSelectAll) {
                    *pSubSelect = *pSubSelectAll;
                }
            }
            pSubSelect->trackID = trackId;
        } else {
            pSubSelect = common->ppSubtitleSelectList[subIdx];
        }
        func_set(pSubSelect, trackId, ptr);
        if (trackId == 0) {
            for (int itrack = 0; itrack < common->nSubtitleSelectCount; itrack++) {
                func_set(common->ppSubtitleSelectList[itrack], trackId, ptr);
            }
        }

        if (subIdx < 0) {
            subIdx = common->nSubtitleSelectCount;
            //新たに要素を追加
            common->ppSubtitleSelectList = (SubtitleSelect **)realloc(common->ppSubtitleSelectList, sizeof(common->ppSubtitleSelectList[0]) * (common->nSubtitleSelectCount + 1));
            common->ppSubtitleSelectList[common->nSubtitleSelectCount] = pSubSelect;
            common->nSubtitleSelectCount++;
        }
        return 0;
    };
    if (IS_OPTION("audio-copy") || IS_OPTION("copy-audio")) {
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        std::set<int> trackSet; //重複しないよう、setを使う
        if (i+1 < nArgNum && (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0'))) {
            i++;
            auto trackListStr = split(strInput[i], _T(","));
            for (auto str : trackListStr) {
                int iTrack = 0;
                if (1 != _stscanf(str.c_str(), _T("%d"), &iTrack) || iTrack < 1) {
                    CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                    return 1;
                } else {
                    trackSet.insert(iTrack);
                }
            }
        } else {
            trackSet.insert(0);
        }

        for (auto it = trackSet.begin(); it != trackSet.end(); it++) {
            int trackId = *it;
            AudioSelect *pAudioSelect = nullptr;
            int audioIdx = getAudioTrackIdx(common, trackId);
            if (audioIdx < 0) {
                pAudioSelect = new AudioSelect();
                pAudioSelect->trackID = trackId;
            } else {
                pAudioSelect = common->ppAudioSelectList[audioIdx];
            }
            pAudioSelect->encCodec = RGY_AVCODEC_COPY;

            if (audioIdx < 0) {
                audioIdx = common->nAudioSelectCount;
                //新たに要素を追加
                common->ppAudioSelectList = (AudioSelect **)realloc(common->ppAudioSelectList, sizeof(common->ppAudioSelectList[0]) * (common->nAudioSelectCount + 1));
                common->ppAudioSelectList[common->nAudioSelectCount] = pAudioSelect;
                common->nAudioSelectCount++;
            }
            argData->nParsedAudioCopy++;
        }
        return 0;
    }
    if (IS_OPTION("audio-codec")) {
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
            if (trackId != 0 || pAudioSelect->encCodec.length() == 0) {
                if (prmstr == nullptr) {
                    pAudioSelect->encCodec = RGY_AVCODEC_AUTO;
                } else {
                    tstring prm = prmstr;
                    auto delimEnc = prm.find(_T(":"));
                    auto delimDec = prm.find(_T("#"));
                    pAudioSelect->encCodec = prm.substr(0, std::min(delimEnc, delimDec));
                    if (delimEnc != tstring::npos) {
                        pAudioSelect->encCodecPrm = prm.substr(delimEnc + 1, (delimEnc < delimDec) ? delimDec - delimEnc - 1 : tstring::npos);
                    }
                    if (delimDec != tstring::npos) {
                        pAudioSelect->decCodecPrm = prm.substr(delimDec + 1, (delimDec < delimEnc) ? delimEnc - delimDec - 1 : tstring::npos);
                    }
                }
            }
        });
        if (ret) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return ret;
        }
        return 0;
    }
    if (IS_OPTION("audio-profile")) {
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
            if (trackId != 0 || pAudioSelect->encCodecProfile.length() == 0) {
                pAudioSelect->encCodecProfile = prmstr;
            }
        });
        if (ret) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return ret;
        }
        return 0;
    }
    if (IS_OPTION("audio-bitrate")) {
        try {
            auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || pAudioSelect->encBitrate == 0) {
                    pAudioSelect->encBitrate = std::stoi(prmstr);
                }
            });
            return ret;
        } catch (...) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
    }
    if (IS_OPTION("audio-ignore-decode-error")) {
        i++;
        uint32_t value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        common->audioIgnoreDecodeError = value;
        return 0;
    }
    //互換性のため残す
    if (IS_OPTION("audio-ignore-notrack-error")) {
        return 0;
    }
    if (IS_OPTION("audio-samplerate")) {
        try {
            auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || pAudioSelect->encSamplingRate == 0) {
                    pAudioSelect->encSamplingRate = std::stoi(prmstr);
                }
            });
            return ret;
        } catch (...) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
    }
    if (IS_OPTION("audio-resampler")) {
        i++;
        int v = 0;
        if (PARSE_ERROR_FLAG != (v = get_value_from_chr(list_resampler, strInput[i]))) {
            common->audioResampler = v;
        } else if (1 == _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_resampler) - 1) {
            common->audioResampler = v;
        } else {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("audio-stream")) {
        //ここで、av_get_channel_layout()を使うため、チェックする必要がある
        if (!check_avcodec_dll()) {
            _ftprintf(stderr, _T("%s\n--audio-stream could not be used.\n"), error_mes_avcodec_dll_not_found().c_str());
            return 1;
        }

        try {
            auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || (pAudioSelect->streamChannelSelect[0] == 0 && pAudioSelect->streamChannelOut[0] == 0)) {
                    auto streamSelectList = split(tchar_to_string(prmstr), ",");
                    if (streamSelectList.size() > _countof(pAudioSelect->streamChannelSelect)) {
                        return 1;
                    }
                    static const char *DELIM = ":";
                    for (uint32_t j = 0; j < streamSelectList.size(); j++) {
                        auto selectPtr = streamSelectList[j].c_str();
                        auto selectDelimPos = strstr(selectPtr, DELIM);
                        if (selectDelimPos == nullptr) {
                            auto channelLayout = av_get_channel_layout(selectPtr);
                            pAudioSelect->streamChannelSelect[j] = channelLayout;
                            pAudioSelect->streamChannelOut[j]    = RGY_CHANNEL_AUTO; //自動
                        } else if (selectPtr == selectDelimPos) {
                            pAudioSelect->streamChannelSelect[j] = RGY_CHANNEL_AUTO;
                            pAudioSelect->streamChannelOut[j]    = av_get_channel_layout(selectDelimPos + strlen(DELIM));
                        } else {
                            pAudioSelect->streamChannelSelect[j] = av_get_channel_layout(streamSelectList[j].substr(0, selectDelimPos - selectPtr).c_str());
                            pAudioSelect->streamChannelOut[j]    = av_get_channel_layout(selectDelimPos + strlen(DELIM));
                        }
                    }
                }
                return 0;
            });
            if (ret) {
                CMD_PARSE_SET_ERR(strInput[0], _T("Too much streams splitted"), option_name, strInput[i]);
                return ret;
            }
            return ret;
        } catch (...) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
    }
    if (IS_OPTION("audio-filter")) {
        try {
            auto ret = set_audio_prm([](AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || pAudioSelect->filter.length() == 0) {
                    pAudioSelect->filter = prmstr;
                }
            });
            return ret;
        } catch (...) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
    }
#endif //#if ENABLE_AVCODEC_QSV_READER
    if (IS_OPTION("chapter-copy") || IS_OPTION("copy-chapter")) {
        common->copyChapter = TRUE;
        return 0;
    }
    if (IS_OPTION("chapter")) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            common->chapterFile = strInput[i];
        } else {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i+1]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("key-on-chapter")) {
        common->keyOnChapter = true;
        return 0;
    }
    if (IS_OPTION("keyfile")) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            common->keyFile = strInput[i];
        } else {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i+1]);
            return 1;
        }
        return 0;
    }
#if ENABLE_AVSW_READER
    if (IS_OPTION("sub-copy") || IS_OPTION("copy-sub")) {
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_SUBTITLE);
        std::map<int, SubtitleSelect> trackSet; //重複しないように
        if (i+1 < nArgNum && (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0'))) {
            i++;
            auto trackListStr = split(strInput[i], _T(","));
            for (auto str : trackListStr) {
                int iTrack = 0;
                if (1 != _stscanf(str.c_str(), _T("%d"), &iTrack) || iTrack < 1) {
                    if (str == _T("asdata")) {
                        trackSet[iTrack].trackID = iTrack;
                        trackSet[iTrack].encCodec = RGY_AVCODEC_COPY;
                        trackSet[iTrack].asdata = true;
                    } else {
                        CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return 1;
                    }
                } else {
                    trackSet[iTrack].trackID = iTrack;
                    trackSet[iTrack].encCodec = RGY_AVCODEC_COPY;
                    auto options = str.find(_T('?'));
                    if (str.substr(options+1) == _T("asdata")) {
                        trackSet[iTrack].asdata = true;
                    }
                }
            }
        } else {
            trackSet[0].trackID = 0;
            trackSet[0].encCodec = RGY_AVCODEC_COPY;
        }

        for (auto it = trackSet.begin(); it != trackSet.end(); it++) {
            int trackId = it->first;
            SubtitleSelect *pSubtitleSelect = nullptr;
            int subIdx = getSubTrackIdx(common, trackId);
            if (subIdx < 0) {
                pSubtitleSelect = new SubtitleSelect();
            } else {
                pSubtitleSelect = common->ppSubtitleSelectList[subIdx];
            }
            pSubtitleSelect[0] = it->second;

            if (subIdx < 0) {
                subIdx = common->nSubtitleSelectCount;
                //新たに要素を追加
                common->ppSubtitleSelectList = (SubtitleSelect **)realloc(common->ppSubtitleSelectList, sizeof(common->ppSubtitleSelectList[0]) * (common->nSubtitleSelectCount + 1));
                common->ppSubtitleSelectList[common->nSubtitleSelectCount] = pSubtitleSelect;
                common->nSubtitleSelectCount++;
            }
        }
        return 0;
    }
    if (IS_OPTION("sub-codec")) {
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        auto ret = set_sub_prm([](SubtitleSelect *pSubSelect, int trackId, const TCHAR *prmstr) {
            if (trackId != 0 || pSubSelect->encCodec.length() == 0) {
                if (prmstr == nullptr) {
                    pSubSelect->encCodec = RGY_AVCODEC_AUTO;
                } else {
                    tstring prm = prmstr;
                    auto delimEnc = prm.find(_T(":"));
                    auto delimDec = prm.find(_T("#"));
                    pSubSelect->encCodec = prm.substr(0, std::min(delimEnc, delimDec));
                    if (delimEnc != tstring::npos) {
                        pSubSelect->encCodecPrm = prm.substr(delimEnc + 1, (delimEnc < delimDec) ? delimDec - delimEnc - 1 : tstring::npos);
                    }
                    if (delimDec != tstring::npos) {
                        pSubSelect->decCodecPrm = prm.substr(delimDec + 1, (delimDec < delimEnc) ? delimEnc - delimDec - 1 : tstring::npos);
                    }
                }
            }
        });
        if (ret) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return ret;
        }
        return 0;
    }
    if (IS_OPTION("caption2ass")) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            C2AFormat format = FORMAT_INVALID;
            if (PARSE_ERROR_FLAG != (format = (C2AFormat)get_value_from_chr(list_caption2ass, strInput[i]))) {
                common->caption2ass = format;
            } else {
                CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return 1;
            }
        } else {
            common->caption2ass = FORMAT_SRT;
        }
        return 0;
    }
    if (IS_OPTION("no-caption2ass")) {
        common->caption2ass = FORMAT_INVALID;
        return 0;
    }
    if (IS_OPTION("data-copy")) {
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_SUBTITLE);
        std::map<int, DataSelect> trackSet; //重複しないように
        if (i+1 < nArgNum && (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0'))) {
            i++;
            auto trackListStr = split(strInput[i], _T(","));
            for (auto str : trackListStr) {
                int iTrack = 0;
                if (1 != _stscanf(str.c_str(), _T("%d"), &iTrack) || iTrack < 1) {
                    CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                    return 1;
                } else {
                    trackSet[iTrack].trackID = iTrack;
                }
            }
        } else {
            trackSet[0].trackID = 0;
        }

        for (auto it = trackSet.begin(); it != trackSet.end(); it++) {
            int trackId = it->first;
            DataSelect *pDataSelect = nullptr;
            int dataIdx = getDataTrackIdx(common, trackId);
            if (dataIdx < 0) {
                pDataSelect = new DataSelect();
            } else {
                pDataSelect = common->ppDataSelectList[dataIdx];
            }
            pDataSelect[0] = it->second;

            if (dataIdx < 0) {
                dataIdx = common->nDataSelectCount;
                //新たに要素を追加
                common->ppDataSelectList = (DataSelect **)realloc(common->ppDataSelectList, sizeof(common->ppDataSelectList[0]) * (common->nDataSelectCount + 1));
                common->ppDataSelectList[common->nDataSelectCount] = pDataSelect;
                common->nDataSelectCount++;
            }
        }
        return 0;
    }
#endif //#if ENABLE_AVSW_READER
    if (IS_OPTION("output-buf")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < 0) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        common->outputBufSizeMB = (std::min)(value, RGY_OUTPUT_BUF_MB_MAX);
        return 0;
    }
    if (IS_OPTION("avsync")) {
        int value = 0;
        i++;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_avsync, strInput[i]))) {
            common->AVSyncMode = (RGYAVSync)value;
        } else {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("mux-option")) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            auto ptr = _tcschr(strInput[i], ':');
            if (ptr == nullptr) {
                CMD_PARSE_SET_ERR(strInput[0], _T("invalid value"), option_name, nullptr);
                return 1;
            } else {
                if (common->muxOpt == nullptr) {
                    common->muxOpt = new muxOptList();
                }
                common->muxOpt->push_back(std::make_pair<tstring, tstring>(tstring(strInput[i]).substr(0, ptr - strInput[i]), tstring(ptr+1)));
            }
        } else {
            CMD_PARSE_SET_ERR(strInput[0], _T("invalid option"), option_name, nullptr);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("max-cll")) {
        i++;
        common->maxCll = tchar_to_string(strInput[i]);
        return 0;
    }
    if (IS_OPTION("master-display")) {
        i++;
        common->masterDisplay = tchar_to_string(strInput[i]);
        return 0;
    }
    if (IS_OPTION("dhdr10-info")) {
        i++;
        common->dynamicHdr10plusJson = strInput[i];
        return 0;
    }
    return -10;
}

int parse_one_ctrl_option(const TCHAR *option_name, const TCHAR *strInput[], int &i, int nArgNum, RGYParamControl *ctrl, sArgsData *argData, ParseCmdError &err) {
    if (IS_OPTION("log")) {
        i++;
        ctrl->logfile = strInput[i];
        return 0;
    }
    if (IS_OPTION("log-level")) {
        i++;
        int value = 0;
        if (get_list_value(list_log_level, strInput[i], &value)) {
            ctrl->loglevel = value;
        } else {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("log-framelist")) {
        i++;
        ctrl->logFramePosList = strInput[i];
        return 0;
    }
    if (IS_OPTION("log-mux-ts")) {
        i++;
        ctrl->logMuxVidTsFile = _tcsdup(strInput[i]);
        return 0;
    }
    if (IS_OPTION("max-procfps")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        if (value < 0) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return -0;
        }
        ctrl->procSpeedLimit = (std::min)(value, INT_MAX);
        return 0;
    }
    if (IS_OPTION("input-thread") || IS_OPTION("thread-input")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < -1 || value >= 2) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        ctrl->threadInput = (int8_t)value;
        return 0;
    }
    if (IS_OPTION("no-output-thread")) {
        ctrl->threadOutput = 0;
        return 0;
    }
    if (IS_OPTION("output-thread") || IS_OPTION("thread-output")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < -1 || value >= 2) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        ctrl->threadOutput = value;
        return 0;
    }
    if (IS_OPTION("audio-thread") || IS_OPTION("thread-audio")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < -1 || value >= 3) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        ctrl->threadAudio = value;
        return 0;
    }
    if (IS_OPTION("thread-csp")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < -1 || value >= 2) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        ctrl->threadCsp = value;
        return 0;
    }
    if (IS_OPTION("simd-csp")) {
        i++;
        int value = 0;
        if (get_list_value(list_simd, strInput[i], &value)) {
            ctrl->simdCsp = value;
        } else {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("perf-monitor")) {
        if (strInput[i+1][0] == _T('-') || _tcslen(strInput[i+1]) == 0) {
            ctrl->perfMonitorSelect = (int)PERF_MONITOR_ALL;
        } else {
            i++;
            auto items = split(strInput[i], _T(","));
            for (const auto& item : items) {
                int value = 0;
                if (PARSE_ERROR_FLAG == (value = get_value_from_chr(list_pref_monitor, item.c_str()))) {
                    CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, item.c_str());
                    return 1;
                }
                ctrl->perfMonitorSelect |= value;
            }
        }
        return 0;
    }
    if (IS_OPTION("perf-monitor-interval")) {
        i++;
        int v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v)) {
            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        ctrl->perfMonitorInterval = std::max(50, v);
        return 0;
    }
    return -10;
}

#define OPT_FLOAT(str, opt, prec) if ((param->opt) != (defaultPrm->opt)) cmd << _T(" ") << (str) << _T(" ") << std::setprecision(prec) << (param->opt);
#define OPT_NUM(str, opt) if ((param->opt) != (defaultPrm->opt)) cmd << _T(" ") << (str) << _T(" ") << (int)(param->opt);
#define OPT_LST(str, opt, list) if ((param->opt) != (defaultPrm->opt)) cmd << _T(" ") << (str) << _T(" ") << get_chr_from_value(list, (param->opt));
#define OPT_BOOL(str_true, str_false, opt) if ((param->opt) != (defaultPrm->opt)) cmd << _T(" ") << ((param->opt) ? (str_true) : (str_false));

#define OPT_TCHAR(str, opt) if ((param->opt) && _tcslen(param->opt)) cmd << _T(" ") << str << _T(" ") << (param->opt);
#define OPT_TSTR(str, opt) if (param->opt.length() > 0) cmd << _T(" ") << str << _T(" ") << param->opt.c_str();
#define OPT_CHAR(str, opt) if ((param->opt) && _tcslen(param->opt)) cmd << _T(" ") << str << _T(" ") << char_to_tstring(param->opt);
#define OPT_STR(str, opt) if (param->opt.length() > 0) cmd << _T(" ") << str << _T(" ") << char_to_tstring(param->opt).c_str();
#define OPT_CHAR_PATH(str, opt) if ((param->opt) && _tcslen(param->opt)) cmd << _T(" ") << str << _T(" \"") << (param->opt) << _T("\"");
#define OPT_STR_PATH(str, opt) if (param->opt.length() > 0) cmd << _T(" ") << str << _T(" \"") << (param->opt.c_str()) << _T("\"");

tstring gen_cmd(const VideoInfo *param, const VideoInfo *defaultPrm, bool save_disabled_prm) {
    std::basic_stringstream<TCHAR> cmd;
    switch (param->type) {
    case RGY_INPUT_FMT_RAW:    cmd << _T(" --raw"); break;
    case RGY_INPUT_FMT_Y4M:    cmd << _T(" --y4m"); break;
    case RGY_INPUT_FMT_AVI:    cmd << _T(" --avi"); break;
    case RGY_INPUT_FMT_AVS:    cmd << _T(" --avs"); break;
    case RGY_INPUT_FMT_VPY:    cmd << _T(" --vpy"); break;
    case RGY_INPUT_FMT_VPY_MT: cmd << _T(" --vpy-mt"); break;
    case RGY_INPUT_FMT_AVHW:   cmd << _T(" --avhw"); break;
    case RGY_INPUT_FMT_AVSW:   cmd << _T(" --avsw"); break;
    default: break;
    }
    if (save_disabled_prm || param->picstruct != RGY_PICSTRUCT_FRAME) {
        OPT_LST(_T("--interlace"), picstruct, list_interlaced);
    }
    if (cropEnabled(param->crop)) {
        cmd << _T(" --crop ") << param->crop.e.left << _T(",") << param->crop.e.up
            << _T(",") << param->crop.e.right << _T(",") << param->crop.e.bottom;
    }
    if (param->fpsN * param->fpsD > 0) {
        cmd << _T(" --fps ") << param->fpsN << _T("/") << param->fpsD;
    }
    if (param->srcWidth * param->srcHeight > 0) {
        cmd << _T(" --input-res ") << param->srcWidth << _T("x") << param->srcHeight;
    }
    if (param->dstWidth * param->dstHeight > 0) {
        cmd << _T(" --output-res ") << param->dstWidth << _T("x") << param->dstHeight;
    }
    return cmd.str();
}

tstring gen_cmd(const RGYParamCommon *param, const RGYParamCommon *defaultPrm, bool save_disabled_prm) {
    std::basic_stringstream<TCHAR> cmd;

    OPT_STR_PATH(_T("-i"), inputFilename);
    OPT_STR_PATH(_T("-o"), outputFilename);

    std::basic_stringstream<TCHAR> tmp;
#if ENABLE_AVSW_READER
    OPT_NUM(_T("--input-analyze"), demuxAnalyzeSec);
    if (param->nTrimCount > 0) {
        cmd << _T(" --trim ");
        for (int i = 0; i < param->nTrimCount; i++) {
            if (i > 0) cmd << _T(",");
            cmd << param->pTrimList[i].start << _T(":") << param->pTrimList[i].fin;
        }
    }
    OPT_FLOAT(_T("--seek"), seekSec, 2);
    OPT_TCHAR(_T("--input-format"), AVInputFormat);
    OPT_TSTR(_T("--output-format"), muxOutputFormat);
    OPT_STR(_T("--video-tag"), videoCodecTag);
    OPT_NUM(_T("--video-track"), videoTrack);
    OPT_NUM(_T("--video-streamid"), videoStreamId);
    if (param->muxOpt) {
        for (uint32_t i = 0; i < param->muxOpt->size(); i++) {
            cmd << _T(" -m ") << param->muxOpt->at(i).first << _T(":") << param->muxOpt->at(i).second;
        }
    }
    tmp.str(tstring());
    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec == RGY_AVCODEC_COPY) {
            if (pAudioSelect->trackID == 0) {
                tmp << _T(","); // --audio-copy のみの指定 (トラックIDを省略)
            } else {
                tmp << _T(",") << pAudioSelect->trackID;
            }
        }
    }
    if (!tmp.str().empty()) {
        cmd << _T(" --audio-copy ") << tmp.str().substr(1);
    }
    tmp.str(tstring());

    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY) {
            cmd << _T(" --audio-codec ") << pAudioSelect->trackID;
            if (pAudioSelect->encCodec != RGY_AVCODEC_AUTO) {
                cmd << _T("?") << pAudioSelect->encCodec;
            }
            if (pAudioSelect->encCodecPrm.length() > 0) {
                cmd << _T(":") << pAudioSelect->encCodecPrm;
            }
        }
    }

    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->encCodecProfile.length() > 0) {
            cmd << _T(" --audio-profile ") << pAudioSelect->trackID << _T("?") << pAudioSelect->encCodecProfile;
        }
    }

    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->encBitrate > 0) {
            cmd << _T(" --audio-bitrate ") << pAudioSelect->trackID << _T("?") << pAudioSelect->encBitrate;
        }
    }

    for (int i = 0; i < param->nAudioSelectCount; i++) {
        tmp.str(tstring());
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        for (int j = 0; j < MAX_SPLIT_CHANNELS; j++) {
            if (pAudioSelect->streamChannelSelect[j] == 0) {
                break;
            }
            if (j > 0) cmd << _T(",");
            if (pAudioSelect->streamChannelSelect[j] != RGY_CHANNEL_AUTO) {
                char buf[256];
                av_get_channel_layout_string(buf, _countof(buf), 0, pAudioSelect->streamChannelOut[j]);
                cmd << char_to_tstring(buf);
            }
            if (pAudioSelect->streamChannelOut[j] != RGY_CHANNEL_AUTO) {
                cmd << _T(":");
                char buf[256];
                av_get_channel_layout_string(buf, _countof(buf), 0, pAudioSelect->streamChannelOut[j]);
                cmd << char_to_tstring(buf);
            }
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --audio-stream ") << pAudioSelect->trackID << _T("?") << tmp.str();
        }
    }
    tmp.str(tstring());

    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->encSamplingRate > 0) {
            cmd << _T(" --audio-samplerate ") << pAudioSelect->trackID << _T("?") << pAudioSelect->encSamplingRate;
        }
    }
    OPT_LST(_T("--audio-resampler"), audioResampler, list_resampler);

    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->filter.length() > 0) {
            cmd << _T(" --audio-filter ") << pAudioSelect->trackID << _T("?") << pAudioSelect->filter;
        }
    }
    for (int i = 0; i < param->nAudioSelectCount; i++) {
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        if (pAudioSelect->extractFilename.length() > 0) {
            cmd << _T(" --audio-file ") << pAudioSelect->trackID << _T("?");
            if (pAudioSelect->extractFormat.length() > 0) {
                cmd << pAudioSelect->extractFormat << _T(":");
            }
            cmd << _T("\"") << pAudioSelect->extractFilename << _T("\"");
        }
    }
    for (int i = 0; i < param->nAudioSourceCount; i++) {
        cmd << _T(" --audio-source ") << _T("\"") << param->ppAudioSourceList[i] << _T("\"");
    }
    OPT_NUM(_T("--audio-ignore-decode-error"), audioIgnoreDecodeError);
    if (param->muxOpt) {
        for (uint32_t i = 0; i < param->muxOpt->size(); i++) {
            cmd << _T(" -m ") << (*param->muxOpt)[i].first << _T(":") << (*param->muxOpt)[i].second;
        }
    }

    tmp.str(tstring());
    for (int i = 0; i < param->nSubtitleSelectCount; i++) {
        tmp << _T(",") << param->ppSubtitleSelectList[i]->trackID;
        if (param->ppSubtitleSelectList[i]->asdata) {
            tmp << _T("?asdata");
        }
    }
    if (!tmp.str().empty()) {
        cmd << _T(" --sub-copy ") << tmp.str().substr(1);
    }
    tmp.str(tstring());
    OPT_LST(_T("--caption2ass"), caption2ass, list_caption2ass);

    tmp.str(tstring());
    for (int i = 0; i < param->nDataSelectCount; i++) {
        tmp << _T(",") << param->ppDataSelectList[i]->trackID;
    }
    if (!tmp.str().empty()) {
        cmd << _T(" --data-copy ") << tmp.str().substr(1);
    }
    tmp.str(tstring());

    OPT_STR_PATH(_T("--chapter"), chapterFile);
    OPT_BOOL(_T("--chapter-copy"), _T(""), copyChapter);
    //OPT_BOOL(_T("--chapter-no-trim"), _T(""), chapterNoTrim);
    OPT_BOOL(_T("--key-on-chapter"), _T(""), keyOnChapter);
    OPT_STR_PATH(_T("--keyfile"), keyFile);
    OPT_LST(_T("--avsync"), AVSyncMode, list_avsync);
#endif //#if ENABLE_AVSW_READER

    OPT_STR(_T("--max-cll"), maxCll);
    OPT_STR(_T("--master-display"), masterDisplay);
    OPT_TSTR(_T("--dhdr10-info"), dynamicHdr10plusJson);

    OPT_NUM(_T("--output-buf"), outputBufSizeMB);
    return cmd.str();
}

tstring gen_cmd(const RGYParamControl *param, const RGYParamControl *defaultPrm, bool save_disabled_prm) {
    std::basic_stringstream<TCHAR> cmd;
    OPT_NUM(_T("--thread-output"), threadOutput);
    OPT_NUM(_T("--thread-input"), threadInput);
    OPT_NUM(_T("--thread-audio"), threadAudio);
    OPT_NUM(_T("--thread-csp"), threadCsp);
    OPT_LST(_T("--simd-csp"), simdCsp, list_simd);
    OPT_NUM(_T("--max-procfps"), procSpeedLimit);
    OPT_STR_PATH(_T("--log"), logfile);
    OPT_LST(_T("--log-level"), loglevel, list_log_level);
    OPT_STR_PATH(_T("--log-framelist"), logFramePosList);
    OPT_CHAR_PATH(_T("--log-mux-ts"), logMuxVidTsFile);
    if (param->perfMonitorSelect != defaultPrm->perfMonitorSelect) {
        auto select = (int)param->perfMonitorSelect;
        std::basic_stringstream<TCHAR> tmp;
        tmp.str(tstring());
        for (int i = 0; list_pref_monitor[i].desc; i++) {
            auto check = list_pref_monitor[i].value;
            if ((select & check) == check) {
                tmp << _T(",") << list_pref_monitor[i].desc;
                select &= (~check);
            }
        }
        if (tmp.str().empty()) {
            cmd << _T(" --perf-monitor");
        } else {
            cmd << _T(" --perf-monitor ") << tmp.str().substr(1);
        }
    }
    OPT_NUM(_T("--perf-monitor-interval"), perfMonitorInterval);
    return cmd.str();
}

unique_ptr<RGYHDR10Plus> initDynamicHDR10Plus(const tstring &dynamicHdr10plusJson, shared_ptr<RGYLog> log) {
    unique_ptr<RGYHDR10Plus> hdr10plus;
    if (dynamicHdr10plusJson.length() > 0) {
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
    }
    return hdr10plus;
}
