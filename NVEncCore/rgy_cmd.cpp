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
#include "rgy_cmd.h"
#include "rgy_perf_monitor.h"

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
#if ENABLE_SM_READER
        return 0;
    }
    if (IS_OPTION("sm")) {
        input->type = RGY_INPUT_FMT_SM;
#endif
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
        || IS_OPTION("avqsv")
        || IS_OPTION("avvce")
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

    if (IS_OPTION("input") || IS_OPTION("input-file")) {
        i++;
        common->inputFilename = strInput[i];
        return 0;
    }
    if (IS_OPTION("output") || IS_OPTION("output-file")) {
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
#if ENABLE_AVSW_READER && !FOR_AUO
    if (IS_OPTION("audio-source")) {
        i++;
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        AudioSource src;
        const TCHAR *ptr = strInput[i];
        const TCHAR *qtr = _tcsrchr(ptr, _T(':'));
        if (qtr == nullptr) {
            src.filename = strInput[i];
            src.select[0].encCodec = RGY_AVCODEC_COPY;
            common->audioSource.push_back(src);
            return 0;
        }
        src.filename = tstring(strInput[i]).substr(0, qtr - ptr);
        auto channel_select_list = split(qtr+1, _T(":"));
        for (auto channel : channel_select_list) {
            int trackId = 0;
            auto channel_id_split = channel.find(_T('?'));
            if (channel_id_split != std::string::npos) {
                try {
                    trackId = std::stoi(channel.substr(0, channel_id_split));
                } catch (...) {
                    CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                    return -1;
                }
                channel = channel.substr(channel_id_split+1);
            }
            AudioSelect &chSel = src.select[trackId];
            chSel.trackID = trackId;
            for (const auto& param : split(channel, _T(";"))) {
                auto pos = param.find_first_of(_T("="));
                if (pos != std::string::npos) {
                    auto param_arg = param.substr(0, pos);
                    auto param_val = param.substr(pos+1);
                    if (param_arg == _T("codec")) {
                        chSel.encCodec = param_val;
                    } else if (param_arg == _T("bitrate")) {
                        try {
                            chSel.encBitrate = std::stoi(param_val);
                        } catch (...) {
                            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                            return -1;
                        }
                    } else if (param_arg == _T("samplerate")) {
                        try {
                            chSel.encSamplingRate = std::stoi(param_val);
                        } catch (...) {
                            CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                            return -1;
                        }
                    } else if (param_arg == _T("profile")) {
                        chSel.encCodecProfile = param_val;
                    } else if (param_arg == _T("filter")) {
                        chSel.filter = param_val;
                    } else if (param_arg == _T("enc_prm")) {
                        chSel.encCodecPrm = param_val;
                    } else {
                        CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    if (chSel.encCodec.length() == 0) {
                        chSel.encCodec = RGY_AVCODEC_AUTO;
                    }
                    continue;
                } else {
                    if (param == _T("copy")) {
                        chSel.encCodec = RGY_AVCODEC_COPY;
                    } else {
                        CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                }
            }
            if (chSel.encCodec.length() == 0) {
                chSel.encCodec = RGY_AVCODEC_COPY;
            }
        }
        common->audioSource.push_back(src);
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
    if (IS_OPTION("format") || IS_OPTION("output-format")) {
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
    auto set_audio_prm = [&](std::function<void(AudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr)> func_set) {
        const TCHAR *ptr = nullptr;
        const TCHAR *ptrDelim = nullptr;
        int trackId = 0;
        if (i+1 < nArgNum) {
            int test_val = 0;
            if ((strInput[i+1][0] != _T('-') || (_stscanf_s(strInput[i+1], _T("%d"), &test_val) == 1 && test_val < 0)) && strInput[i+1][0] != _T('\0')) {
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
    if (IS_OPTION("audio-delay")) {
        try {
            auto ret = set_audio_prm([](AudioSelect* pAudioSelect, int trackId, const TCHAR* prmstr) {
                if (trackId != 0 || pAudioSelect->addDelayMs == 0) {
                    pAudioSelect->addDelayMs = std::stoi(prmstr);
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
#if ENABLE_AVSW_READER && !FOR_AUO
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
    if (IS_OPTION("sub-source")) {
        i++;
        common->AVMuxTarget |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        SubSource src;
        const TCHAR *ptr = strInput[i];
        const TCHAR *qtr = _tcsrchr(ptr, _T(':'));
        if (qtr == nullptr) {
            src.filename = strInput[i];
            src.select[0].encCodec = RGY_AVCODEC_COPY;
            common->subSource.push_back(src);
            return 0;
        }
        src.filename = tstring(strInput[i]).substr(0, qtr - ptr);
        auto channel_select_list = split(qtr+1, _T(":"));
        for (auto channel : channel_select_list) {
            int trackId = 0;
            auto channel_id_split = channel.find(_T('?'));
            if (channel_id_split != std::string::npos) {
                try {
                    trackId = std::stoi(channel.substr(0, channel_id_split));
                } catch (...) {
                    CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                    return -1;
                }
                channel = channel.substr(channel_id_split+1);
            }
            SubtitleSelect &chSel = src.select[trackId];
            chSel.trackID = trackId;
            for (const auto &param : split(channel, _T(";"))) {
                auto pos = param.find_first_of(_T("="));
                if (pos != std::string::npos) {
                    auto param_arg = param.substr(0, pos);
                    auto param_val = param.substr(pos+1);
                    if (param_arg == _T("codec")) {
                        chSel.encCodec = param_val;
                    } else if (param_arg == _T("enc_prm")) {
                        chSel.encCodecPrm = param_val;
                    } else {
                        CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    if (chSel.encCodec.length() == 0) {
                        chSel.encCodec = RGY_AVCODEC_AUTO;
                    }
                    continue;
                } else {
                    if (param == _T("copy")) {
                        chSel.encCodec = RGY_AVCODEC_COPY;
                    } else {
                        CMD_PARSE_SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                }
            }
            if (chSel.encCodec.length() == 0) {
                chSel.encCodec = RGY_AVCODEC_COPY;
            }
        }
        common->subSource.push_back(src);
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
    if (IS_OPTION("no-mp4opt")) {
        common->disableMp4Opt = true;
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
#if !FOR_AUO
    for (int i = 0; i < param->nAudioSelectCount; i++) {
        tmp.str(tstring());
        const AudioSelect *pAudioSelect = param->ppAudioSelectList[i];
        for (int j = 0; j < MAX_SPLIT_CHANNELS; j++) {
            if (pAudioSelect->streamChannelSelect[j] == 0) {
                break;
            }
            if (j > 0) tmp << _T(",");
            if (pAudioSelect->streamChannelSelect[j] != RGY_CHANNEL_AUTO) {
                char buf[256];
                av_get_channel_layout_string(buf, _countof(buf), 0, pAudioSelect->streamChannelOut[j]);
                tmp << char_to_tstring(buf);
            }
            if (pAudioSelect->streamChannelOut[j] != RGY_CHANNEL_AUTO) {
                tmp << _T(":");
                char buf[256];
                av_get_channel_layout_string(buf, _countof(buf), 0, pAudioSelect->streamChannelOut[j]);
                tmp << char_to_tstring(buf);
            }
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --audio-stream ") << pAudioSelect->trackID << _T("?") << tmp.str();
        }
    }
#endif
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
        if (pAudioSelect->encCodec != RGY_AVCODEC_COPY
            && pAudioSelect->addDelayMs > 0) {
            cmd << _T(" --audio-delay ") << pAudioSelect->trackID << _T("?") << pAudioSelect->addDelayMs;
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
    for (const auto &src : param->audioSource) {
        if (src.filename.length() > 0) {
            cmd << _T(" --audio-source ") << _T("\"") << src.filename << _T("\"");
            for (const auto &channel : src.select) {
                cmd << _T(":");
                if (channel.first > 0) {
                    cmd << channel.first << _T("?");
                }
                const auto &sel = channel.second;
                if (sel.encCodec.length() == 0) {
                    ; //何もしない
                } else if (sel.encCodec == RGY_AVCODEC_COPY) {
                    cmd << _T("copy");
                } else {
                    tmp.str(tstring());
                    cmd << _T(";codec=") << sel.encCodec;
                    if (sel.encBitrate > 0) {
                        cmd << _T(";bitrate=") << sel.encBitrate;
                    }
                    if (sel.encCodecPrm.length() > 0) {
                        cmd << _T(";prm=") << sel.encCodecPrm;
                    }
                    if (sel.encCodecProfile.length() > 0) {
                        cmd << _T(";profile=") << sel.encCodecProfile;
                    }
                    if (sel.encSamplingRate > 0) {
                        cmd << _T(";samplerate=") << sel.encSamplingRate;
                    }
                    if (sel.filter.length() > 0) {
                        cmd << _T(";filter=") << _T("\"") << sel.filter << _T("\"");
                    }
                }
                if (!tmp.str().empty()) {
                    cmd << tmp.str().substr(1);
                }
            }
        }
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
    for (const auto &src : param->subSource) {
        if (src.filename.length() > 0) {
            cmd << _T(" --sub-source ") << _T("\"") << src.filename << _T("\"");
            for (const auto &channel : src.select) {
                cmd << _T(":");
                if (channel.first > 0) {
                    cmd << channel.first << _T("?");
                }
                const auto &sel = channel.second;
                if (sel.encCodec.length() == 0) {
                    ; //何もしない
                } else if (sel.encCodec == RGY_AVCODEC_COPY) {
                    cmd << _T("copy");
                } else {
                    tmp.str(tstring());
                    cmd << _T(";codec=") << sel.encCodec;
                    if (sel.encCodecPrm.length() > 0) {
                        cmd << _T(";prm=") << sel.encCodecPrm;
                    }
                }
                if (!tmp.str().empty()) {
                    cmd << tmp.str().substr(1);
                }
            }
        }
    }
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
#endif //#if ENABLE_AVSW_READER
    OPT_BOOL(_T("--no-mp4opt"), _T(""), disableMp4Opt);
    OPT_LST(_T("--avsync"), AVSyncMode, list_avsync);

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

tstring gen_cmd_help_input() {
    tstring str =
        _T("\n")
        _T("-i,--input <filename>           set input filename\n")
        _T("-o,--output <filename>          set output filename\n")
        _T("\n")
        _T(" Input formats (auto detected from extension of not set)\n")
        _T("   --raw                        set input as raw format\n")
        _T("   --y4m                        set input as y4m format\n")
#if ENABLE_AVI_READER
        _T("   --avi                        set input as avi format\n")
#endif
#if ENABLE_AVISYNTH_READER
        _T("   --avs                        set input as avs format\n")
#endif
#if ENABLE_VAPOURSYNTH_READER
        _T("   --vpy                        set input as vpy format\n")
        _T("   --vpy-mt                     set input as vpy(mt) format\n")
#endif
#if ENABLE_AVSW_READER
        _T("   --avhw                       use libavformat + hw decode for input\n")
        _T("   --avsw                       set input to use avcodec + sw decoder\n")
#endif
        _T("   --input-res <int>x<int>        set input resolution\n")
        _T("   --crop <int>,<int>,<int>,<int> crop pixels from left,top,right,bottom\n")
        _T("                                    left crop is unavailable with avhw reader\n")
        _T("   --output-res <int>x<int>     set output resolution\n")
        _T("   --fps <int>/<int> or <float> set framerate\n")
        _T("   --interlace <string>         set input as interlaced\n")
        _T("                                  tff, bff\n");
    return str;
}

tstring gen_cmd_help_common() {
#if ENABLE_AVSW_READER
    tstring str = strsprintf(_T("")
        _T("   --input-analyze <int>       set time (sec) which reader analyze input file.\n")
        _T("                                 default: 5 (seconds).\n")
        _T("                                 could be only used with avhw/avsw reader.\n")
        _T("                                 use if reader fails to detect audio stream.\n")
        _T("   --video-track <int>          set video track to encode in track id\n")
        _T("                                 1 (default)  highest resolution video track\n")
        _T("                                 2            next high resolution video track\n")
        _T("                                   ... \n")
        _T("                                 -1           lowest resolution video track\n")
        _T("                                 -2           next low resolution video track\n")
        _T("                                   ... \n")
        _T("   --video-streamid <int>       set video track to encode in stream id\n")
        _T("   --audio-source <string>      input extra audio file.\n")
        _T("   --audio-file [<int>?][<string>:]<string>\n")
        _T("                                extract audio into file.\n")
        _T("                                 could be only used with avhw/avsw reader.\n")
        _T("                                 below are optional,\n")
        _T("                                  in [<int>?], specify track number to extract.\n")
        _T("                                  in [<string>?], specify output format.\n")
        _T("   --trim <int>:<int>[,<int>:<int>]...\n")
        _T("                                trim video for the frame range specified.\n")
        _T("                                 frame range should not overwrap each other.\n")
        _T("   --seek [<int>:][<int>:]<int>[.<int>] (hh:mm:ss.ms)\n")
        _T("                                skip video for the time specified,\n")
        _T("                                 seek will be inaccurate but fast.\n")
        _T("   --input-format <string>      set input format of input file.\n")
        _T("                                 this requires use of avhw/avsw reader.\n")
        _T("-f,--output-format <string>     set output format of output file.\n")
        _T("                                 if format is not specified, output format will\n")
        _T("                                 be guessed from output file extension.\n")
        _T("                                 set \"raw\" for H.264/ES output.\n")
        _T("   --audio-copy [<int>[,...]]   mux audio with video during output.\n")
        _T("                                 could be only used with\n")
        _T("                                 avhw/avsw reader and avcodec muxer.\n")
        _T("                                 by default copies all audio tracks.\n")
        _T("                                 \"--audio-copy 1,2\" will extract\n")
        _T("                                 audio track #1 and #2.\n")
        _T("   --audio-codec [<int>?]<string>\n")
        _T("                                encode audio to specified format.\n")
        _T("                                  in [<int>?], specify track number to encode.\n")
        _T("   --audio-profile [<int>?]<string>\n")
        _T("                                specify audio profile.\n")
        _T("                                  in [<int>?], specify track number to apply.\n")
        _T("   --audio-bitrate [<int>?]<int>\n")
        _T("                                set encode bitrate for audio (kbps).\n")
        _T("                                  in [<int>?], specify track number of audio.\n")
        _T("   --audio-ignore-decode-error <int>  (default: %d)\n")
        _T("                                set numbers of continuous packets of audio decode\n")
        _T("                                 error to ignore, replaced by silence.\n")
        _T("   --audio-samplerate [<int>?]<int>\n")
        _T("                                set sampling rate for audio (Hz).\n")
        _T("                                  in [<int>?], specify track number of audio.\n")
        _T("   --audio-resampler <string>   set audio resampler.\n")
        _T("                                  swr (swresampler: default), soxr (libsoxr)\n")
        _T("   --audio-delay [<int>?]<int>  set audio delay (ms).\n")
        _T("   --audio-stream [<int>?][<string1>][:<string2>][,[<string1>][:<string2>]][..\n")
        _T("       set audio streams in channels.\n")
        _T("         in [<int>?], specify track number to split.\n")
        _T("         in <string1>, set input channels to use from source stream.\n")
        _T("           if unset, all input channels will be used.\n")
        _T("         in <string2>, set output channels to mix.\n")
        _T("           if unset, all input channels will be copied without mixing.\n")
        _T("       example1: --audio-stream FL,FR\n")
        _T("         splitting dual mono audio to each stream.\n")
        _T("       example2: --audio-stream :stereo\n")
        _T("         mixing input channels to stereo.\n")
        _T("       example3: --audio-stream 5.1,5.1:stereo\n")
        _T("         keeping 5.1ch audio and also adding downmixed stereo stream.\n")
        _T("       usable symbols\n")
        _T("         mono       = FC\n")
        _T("         stereo     = FL + FR\n")
        _T("         2.1        = FL + FR + LFE\n")
        _T("         3.0        = FL + FR + FC\n")
        _T("         3.0(back)  = FL + FR + BC\n")
        _T("         3.1        = FL + FR + FC + LFE\n")
        _T("         4.0        = FL + FR + FC + BC\n")
        _T("         quad       = FL + FR + BL + BR\n")
        _T("         quad(side) = FL + FR + SL + SR\n")
        _T("         5.0        = FL + FR + FC + SL + SR\n")
        _T("         5.1        = FL + FR + FC + LFE + SL + SR\n")
        _T("         6.0        = FL + FR + FC + BC + SL + SR\n")
        _T("         6.0(front) = FL + FR + FLC + FRC + SL + SR\n")
        _T("         hexagonal  = FL + FR + FC + BL + BR + BC\n")
        _T("         6.1        = FL + FR + FC + LFE + BC + SL + SR\n")
        _T("         6.1(front) = FL + FR + LFE + FLC + FRC + SL + SR\n")
        _T("         7.0        = FL + FR + FC + BL + BR + SL + SR\n")
        _T("         7.0(front) = FL + FR + FC + FLC + FRC + SL + SR\n")
        _T("         7.1        = FL + FR + FC + LFE + BL + BR + SL + SR\n")
        _T("         7.1(wide)  = FL + FR + FC + LFE + FLC + FRC + SL + SR\n")
        _T("   --audio-filter [<int>?]<string>\n")
        _T("                                set audio filter.\n")
        _T("                                  in [<int>?], specify track number of audio.\n")
        _T("   --chapter-copy               copy chapter to output file.\n")
        _T("   --chapter <string>           set chapter from file specified.\n")
        _T("   --key-on-chapter             set key frame on chapter.\n")
        _T("   --sub-source <string>        input extra subtitle file.\n")
        _T("   --sub-copy [<int>[,...]]     copy subtitle to output file.\n")
        _T("                                 these could be only used with\n")
        _T("                                 avhw/avsw reader and avcodec muxer.\n")
        _T("                                 below are optional,\n")
        _T("                                  in [<int>?], specify track number to copy.\n")
        _T("   --caption2ass [<string>]     enable caption2ass during encode.\n")
        _T("                                  !! This feature requires Caption.dll !!\n")
        _T("                                 supported formats ... srt (default), ass\n")
        _T("   --data-copy [<int>[,...]]    copy data stream to output file.\n")
        _T("\n")
        _T("   --avsync <string>            method for AV sync (default: cfr)\n")
        _T("                                 cfr      ... assume cfr\n")
        _T("                                 forcecfr ... check timestamp and force cfr\n")
        _T("                                 vfr      ... honor source timestamp and enable vfr output.\n")
        _T("                                              only available for avsw/avhw reader,\n")
        _T("                                              and could not be used with --trim.\n")
        _T("-m,--mux-option <string1>:<string2>\n")
        _T("                                set muxer option name and value.\n")
        _T("                                 these could be only used with\n")
        _T("                                 avhw/avsw reader and avcodec muxer.\n"),
        DEFAULT_IGNORE_DECODE_ERROR);
#else
    tstring str;
#endif
    return str;
}

tstring gen_cmd_help_ctrl() {
    tstring str = strsprintf(_T("\n")
        _T("   --log <string>               set log file name\n")
        _T("   --log-level <string>         set log level\n")
        _T("                                  debug, info(default), warn, error\n")
        _T("   --log-framelist <string>     output frame info of avhw reader to path\n"));

    str += strsprintf(_T("")
        _T("   --max-procfps <int>         limit encoding speed for lower utilization.\n")
        _T("                                 default:0 (no limit)\n"));
#if ENABLE_AVCODEC_OUT_THREAD
    str += strsprintf(_T("")
        _T("   --output-thread <int>        set output thread num\n")
        _T("                                 -1: auto (= default)\n")
        _T("                                  0: disable (slow, but less memory usage)\n")
        _T("                                  1: use one thread\n")
#if 0
        _T("   --audio-thread <int>         set audio thread num, available only with output thread\n")
        _T("                                 -1: auto (= default)\n")
        _T("                                  0: disable (slow, but less memory usage)\n")
        _T("                                  1: use one thread\n")
        _T("                                  2: use two thread\n")
#endif //#if ENABLE_AVCODEC_AUDPROCESS_THREAD
    );
#endif //#if ENABLE_AVCODEC_OUT_THREAD
    str += strsprintf(_T("\n")
        _T("   --perf-monitor [<string>][,<string>]...\n")
        _T("       check performance info of encoder and output to log file\n")
        _T("       select counter from below, default = all\n")
        _T("                                 \n")
        _T("     counters for perf-monitor\n")
        _T("                                 all          ... monitor all info\n")
        _T("                                 cpu_total    ... cpu total usage (%%)\n")
        _T("                                 cpu_kernel   ... cpu kernel usage (%%)\n")
#if defined(_WIN32) || defined(_WIN64)
        _T("                                 cpu_main     ... cpu main thread usage (%%)\n")
        _T("                                 cpu_enc      ... cpu encode thread usage (%%)\n")
        _T("                                 cpu_in       ... cpu input thread usage (%%)\n")
        _T("                                 cpu_out      ... cpu output thread usage (%%)\n")
        _T("                                 cpu_aud_proc ... cpu aud proc thread usage (%%)\n")
        _T("                                 cpu_aud_enc  ... cpu aud enc thread usage (%%)\n")
#endif //#if defined(_WIN32) || defined(_WIN64)
        _T("                                 cpu          ... monitor all cpu info\n")
#if defined(_WIN32) || defined(_WIN64)
        _T("                                 gpu_load    ... gpu usage (%%)\n")
        _T("                                 gpu_clock   ... gpu avg clock\n")
        _T("                                 vee_load    ... gpu video encoder usage (%%)\n")
        _T("                                 ved_load    ... gpu video decoder usage (%%)\n")
        _T("                                 gpu         ... monitor all gpu info\n")
#endif //#if defined(_WIN32) || defined(_WIN64)
        _T("                                 queue       ... queue usage\n")
        _T("                                 mem_private ... private memory (MB)\n")
        _T("                                 mem_virtual ... virtual memory (MB)\n")
        _T("                                 mem         ... monitor all memory info\n")
        _T("                                 io_read     ... io read  (MB/s)\n")
        _T("                                 io_write    ... io write (MB/s)\n")
        _T("                                 io          ... monitor all io info\n")
        _T("                                 fps         ... encode speed (fps)\n")
        _T("                                 fps_avg     ... encode avg. speed (fps)\n")
        _T("                                 bitrate     ... encode bitrate (kbps)\n")
        _T("                                 bitrate_avg ... encode avg. bitrate (kbps)\n")
        _T("                                 frame_out   ... written_frames\n")
        _T("                                 \n")
        _T("   --perf-monitor-interval <int> set perf monitor check interval (millisec)\n")
        _T("                                 default 500, must be 50 or more\n"));
    return str;
}