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

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <set>
#include <sstream>
#include <iomanip>
#include <Windows.h>
#include <shellapi.h>
#include "rgy_version.h"
#include "rgy_perf_monitor.h"
#include "rgy_caption.h"
#include "NVEncParam.h"
#include "NVEncCmd.h"
#include "NVEncFilterAfs.h"
#include "rgy_avutil.h"

tstring GetNVEncVersion() {
    static const TCHAR *const ENABLED_INFO[] = { _T("disabled"), _T("enabled") };
    tstring version;
    version += get_encoder_version();
    version += _T("\n");
    version += strsprintf(_T("  [NVENC API v%d.%d, CUDA %d.%d]\n"),
        NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION,
        CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    version += _T(" reader: raw");
    if (ENABLE_AVI_READER) version += _T(", avi");
    if (ENABLE_AVISYNTH_READER) version += _T(", avs");
    if (ENABLE_VAPOURSYNTH_READER) version += _T(", vpy");
#if ENABLE_AVSW_READER
    version += strsprintf(_T(", avhw [%s]"), getHWDecSupportedCodecList().c_str());
#endif //#if ENABLE_AVSW_READER
    version += _T("\n");
    return version;
}

const TCHAR *cmd_short_opt_to_long(TCHAR short_opt) {
    const TCHAR *option_name = nullptr;
    switch (short_opt) {
    case _T('b'):
        option_name = _T("bframes");
        break;
    case _T('c'):
        option_name = _T("codec");
        break;
    case _T('d'):
        option_name = _T("device");
        break;
    case _T('u'):
        option_name = _T("quality");
        break;
    case _T('f'):
        option_name = _T("output-format");
        break;
    case _T('i'):
        option_name = _T("input");
        break;
    case _T('o'):
        option_name = _T("output");
        break;
    case _T('m'):
        option_name = _T("mux-option");
        break;
    case _T('v'):
        option_name = _T("version");
        break;
    case _T('h'):
    case _T('?'):
        option_name = _T("help");
        break;
    default:
        break;
    }
    return option_name;
}

#define IS_OPTION(x) (0 == _tcscmp(option_name, _T(x)))

static int getAudioTrackIdx(const InEncodeVideoParam* pParams, int iTrack) {
    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        if (iTrack == pParams->ppAudioSelectList[i]->nAudioSelect) {
            return i;
        }
    }
    return -1;
}

static int getFreeAudioTrack(const InEncodeVideoParam* pParams) {
    for (int iTrack = 1;; iTrack++) {
        if (0 > getAudioTrackIdx(pParams, iTrack)) {
            return iTrack;
        }
    }
#ifndef _MSC_VER
    return -1;
#endif //_MSC_VER
}

bool get_list_value(const CX_DESC * list, const TCHAR *chr, int *value) {
    for (int i = 0; list[i].desc; i++) {
        if (0 == _tcsicmp(list[i].desc, chr)) {
            *value = list[i].value;
            return true;
        }
    }
    return false;
};
bool get_list_guid_value(const guid_desc * list, const TCHAR *chr, int *value) {
    for (int i = 0; list[i].desc; i++) {
        if (0 == _tcsicmp(list[i].desc, chr)) {
            *value = list[i].value;
            return true;
        }
    }
    return false;
};

struct sArgsData {
    tstring cachedlevel, cachedprofile;
    uint32_t nParsedAudioFile = 0;
    uint32_t nParsedAudioEncode = 0;
    uint32_t nParsedAudioCopy = 0;
    uint32_t nParsedAudioBitrate = 0;
    uint32_t nParsedAudioSamplerate = 0;
    uint32_t nParsedAudioSplit = 0;
    uint32_t nParsedAudioFilter = 0;
    uint32_t nTmpInputBuf = 0;
    int nBframes = -1;
};

int parse_one_option(const TCHAR *option_name, const TCHAR* strInput[], int& i, int nArgNum, InEncodeVideoParam *pParams, NV_ENC_CODEC_CONFIG *codecPrm, sArgsData *argData, ParseCmdError& err) {
#define SET_ERR(app_name, errmes, opt_name, err_val) \
    err.strAppName = (app_name) ? app_name : _T(""); \
    err.strErrorMessage = (errmes) ? errmes : _T(""); \
    err.strOptionName = (opt_name) ? opt_name : _T(""); \
    err.strErrorValue = (err_val) ? err_val : _T("");

    if (IS_OPTION("device")) {
        int deviceid = -1;
        if (i + 1 < nArgNum) {
            i++;
            int value = 0;
            if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
                deviceid = value;
            }
        }
        if (deviceid < 0) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        pParams->deviceID = deviceid;
        return 0;
    }
    if (IS_OPTION("preset")) {
        i++;
        int value = get_value_from_name(strInput[i], list_nvenc_preset_names);
        if (value >= 0) {
            pParams->preset = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("input")) {
        i++;
        pParams->inputFilename = strInput[i];
        return 0;
    }
    if (IS_OPTION("output")) {
        i++;
        pParams->outputFilename = strInput[i];
        return 0;
    }
    if (IS_OPTION("fps")) {
        i++;
        int a[2] = { 0 };
        if (   2 == _stscanf_s(strInput[i], _T("%d/%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d:%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d,%d"), &a[0], &a[1])) {
            pParams->input.fpsN = a[0];
            pParams->input.fpsD = a[1];
        } else {
            double d;
            if (1 == _stscanf_s(strInput[i], _T("%lf"), &d)) {
                int rate = (int)(d * 1001.0 + 0.5);
                if (rate % 1000 == 0) {
                    pParams->input.fpsN = rate;
                    pParams->input.fpsD = 1001;
                } else {
                    pParams->input.fpsD = 100000;
                    pParams->input.fpsN = (int)(d * pParams->input.fpsD + 0.5);
                    rgy_reduce(pParams->input.fpsN, pParams->input.fpsD);
                }
            } else  {
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
            }
        }
        return 0;
    }
    if (IS_OPTION("input-res")) {
        i++;
        int a[2] = { 0 };
        if (   2 == _stscanf_s(strInput[i], _T("%dx%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d:%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d,%d"), &a[0], &a[1])) {
            pParams->input.srcWidth  = a[0];
            pParams->input.srcHeight = a[1];
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("output-res")) {
        i++;
        int a[2] = { 0 };
        if (   2 == _stscanf_s(strInput[i], _T("%dx%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d:%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d,%d"), &a[0], &a[1])) {
            pParams->input.dstWidth  = a[0];
            pParams->input.dstHeight = a[1];
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("crop")) {
        i++;
        sInputCrop a = { 0 };
        if (   4 == _stscanf_s(strInput[i], _T("%d,%d,%d,%d"), &a.c[0], &a.c[1], &a.c[2], &a.c[3])
            || 4 == _stscanf_s(strInput[i], _T("%d:%d:%d:%d"), &a.c[0], &a.c[1], &a.c[2], &a.c[3])) {
            memcpy(&pParams->input.crop, &a, sizeof(a));
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("codec")) {
        i++;
        int value = 0;
        if (get_list_value(list_nvenc_codecs_for_opt, strInput[i], &value)) {
            pParams->codec = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("raw")) {
        pParams->input.type = RGY_INPUT_FMT_RAW;
        return 0;
    }
    if (IS_OPTION("y4m")) {
        pParams->input.type = RGY_INPUT_FMT_Y4M;
#if ENABLE_AVI_READER
        return 0;
    }
    if (IS_OPTION("avi")) {
        pParams->input.type = RGY_INPUT_FMT_AVI;
#endif
#if ENABLE_AVISYNTH_READER
        return 0;
    }
    if (IS_OPTION("avs")) {
        pParams->input.type = RGY_INPUT_FMT_AVS;
#endif
#if ENABLE_VAPOURSYNTH_READER
        return 0;
    }
    if (IS_OPTION("vpy")) {
        pParams->input.type = RGY_INPUT_FMT_VPY;
        return 0;
    }
    if (IS_OPTION("vpy-mt")) {
        pParams->input.type = RGY_INPUT_FMT_VPY_MT;
#endif
#if ENABLE_AVSW_READER
        return 0;
    }
    if (IS_OPTION("avcuvid")
        || IS_OPTION("avhw")) {
        pParams->input.type = RGY_INPUT_FMT_AVHW;
        if (strInput[i+1][0] != _T('-') && strInput[i+1][0] != _T('\0')) {
            i++;
            int value = 0;
            if (get_list_value(list_cuvid_mode, strInput[i], &value)) {
                pParams->nHWDecType = value;
            } else {
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
            }
        }
#endif
        return 0;
    }
    if (IS_OPTION("avsw")) {
        pParams->input.type = RGY_INPUT_FMT_AVSW;
        return 0;
    }
    if (   IS_OPTION("input-analyze")
        || IS_OPTION("avcuvid-analyze")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        } else if (value < 0) {
            SET_ERR(strInput[0], _T("input-analyze requires non-negative value."), option_name, strInput[i]);
            return 1;
        } else {
            pParams->nAVDemuxAnalyzeSec = (int)((std::min)(value, USHRT_MAX));
        }
        return 0;
    }
    if (IS_OPTION("video-track")) {
        i++;
        int v = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (v == 0) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nVideoTrack = v;
        return 0;
    }
    if (IS_OPTION("video-streamid")) {
        i++;
        int v = 0;
        if (1 != _stscanf_s(strInput[i], _T("%i"), &v)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nVideoStreamId = v;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("trim"))) {
        i++;
        auto trim_str_list = split(strInput[i], _T(","));
        std::vector<sTrim> trim_list;
        for (auto trim_str : trim_str_list) {
            sTrim trim;
            if (2 != _stscanf_s(trim_str.c_str(), _T("%d:%d"), &trim.start, &trim.fin) || (trim.fin > 0 && trim.fin < trim.start)) {
                SET_ERR(strInput[0], _T("Invalid Value"), option_name, trim_str.c_str());
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
            pParams->nTrimCount = (int)trim_list.size();
            pParams->pTrimList = (sTrim *)malloc(sizeof(pParams->pTrimList[0]) * trim_list.size());
            memcpy(pParams->pTrimList, &trim_list[0], sizeof(pParams->pTrimList[0]) * trim_list.size());
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("seek"))) {
        i++;
        int ret = 0;
        int hh = 0, mm = 0;
        float sec = 0.0f;
        if (   3 != (ret = _stscanf_s(strInput[i], _T("%d:%d:%f"),    &hh, &mm, &sec))
            && 2 != (ret = _stscanf_s(strInput[i],    _T("%d:%f"),         &mm, &sec))
            && 1 != (ret = _stscanf_s(strInput[i],       _T("%f"),              &sec))) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (ret <= 2) {
            hh = 0;
        }
        if (ret <= 1) {
            mm = 0;
        }
        if (hh < 0 || mm < 0 || sec < 0) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        if (hh > 0 && mm >= 60) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        mm += hh * 60;
        if (mm > 0 && sec >= 60.0f) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        pParams->fSeekSec = sec + mm * 60;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-source"))) {
        i++;
        pParams->nAVMux |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        size_t audioSourceLen = _tcslen(strInput[i]) + 1;
        TCHAR *pAudioSource = (TCHAR *)malloc(sizeof(strInput[i][0]) * audioSourceLen);
        memcpy(pAudioSource, strInput[i], sizeof(strInput[i][0]) * audioSourceLen);
        pParams->ppAudioSourceList = (TCHAR **)realloc(pParams->ppAudioSourceList, sizeof(pParams->ppAudioSourceList[0]) * (pParams->nAudioSourceCount + 1));
        pParams->ppAudioSourceList[pParams->nAudioSourceCount] = pAudioSource;
        pParams->nAudioSourceCount++;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-file"))) {
        i++;
        const TCHAR *ptr = strInput[i];
        sAudioSelect *pAudioSelect = nullptr;
        int audioIdx = -1;
        int trackId = 0;
        if (_tcschr(ptr, '?') == nullptr || 1 != _stscanf(ptr, _T("%d?"), &trackId)) {
            //トラック番号を適当に発番する (カウントは1から)
            trackId = argData->nParsedAudioFile+1;
            audioIdx = getAudioTrackIdx(pParams, trackId);
            if (audioIdx < 0 || pParams->ppAudioSelectList[audioIdx]->pAudioExtractFilename != nullptr) {
                trackId = getFreeAudioTrack(pParams);
                pAudioSelect = (sAudioSelect *)calloc(1, sizeof(pAudioSelect[0]));
                pAudioSelect->nAudioSelect = trackId;
            } else {
                pAudioSelect = pParams->ppAudioSelectList[audioIdx];
            }
        } else if (i <= 0) {
            //トラック番号は1から連番で指定
            SET_ERR(strInput[0], _T("Invalid track number"), option_name, strInput[i]);
            return 1;
        } else {
            audioIdx = getAudioTrackIdx(pParams, trackId);
            if (audioIdx < 0) {
                pAudioSelect = (sAudioSelect *)calloc(1, sizeof(pAudioSelect[0]));
                pAudioSelect->nAudioSelect = trackId;
            } else {
                pAudioSelect = pParams->ppAudioSelectList[audioIdx];
            }
            ptr = _tcschr(ptr, '?') + 1;
        }
        assert(pAudioSelect != nullptr);
        const TCHAR *qtr = _tcschr(ptr, ':');
        if (qtr != NULL && !(ptr + 1 == qtr && qtr[1] == _T('\\'))) {
            pAudioSelect->pAudioExtractFormat = _tcsdup(ptr);
            ptr = qtr + 1;
        }
        size_t filename_len = _tcslen(ptr);
        //ファイル名が""でくくられてたら取り除く
        if (ptr[0] == _T('\"') && ptr[filename_len-1] == _T('\"')) {
            filename_len -= 2;
            ptr++;
        }
        //ファイル名が重複していないかを確認する
        for (int j = 0; j < pParams->nAudioSelectCount; j++) {
            if (pParams->ppAudioSelectList[j]->pAudioExtractFilename != nullptr
                && 0 == _tcsicmp(pParams->ppAudioSelectList[j]->pAudioExtractFilename, ptr)) {
                SET_ERR(strInput[0], _T("Same output file name is used more than twice"), option_name, nullptr);
                return 1;
            }
        }

        if (audioIdx < 0) {
            audioIdx = pParams->nAudioSelectCount;
            //新たに要素を追加
            pParams->ppAudioSelectList = (sAudioSelect **)realloc(pParams->ppAudioSelectList, sizeof(pParams->ppAudioSelectList[0]) * (pParams->nAudioSelectCount + 1));
            pParams->ppAudioSelectList[pParams->nAudioSelectCount] = pAudioSelect;
            pParams->nAudioSelectCount++;
        }
        pParams->ppAudioSelectList[audioIdx]->pAudioExtractFilename = _tcsdup(ptr);
        argData->nParsedAudioFile++;
        return 0;
    }
    if (   0 == _tcscmp(option_name, _T("format"))
        || 0 == _tcscmp(option_name, _T("output-format"))) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            pParams->sAVMuxOutputFormat = strInput[i];
            if (0 != _tcsicmp(strInput[i], _T("raw"))) {
                pParams->nAVMux |= RGY_MUX_VIDEO;
            }
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("input-format"))) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            pParams->pAVInputFormat = _tcsdup(strInput[i]);
        } else {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
#if ENABLE_AVSW_READER
    auto set_audio_prm = [&](std::function<void(sAudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr)> func_set) {
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
        sAudioSelect *pAudioSelect = nullptr;
        int audioIdx = getAudioTrackIdx(pParams, trackId);
        if (audioIdx < 0) {
            pAudioSelect = (sAudioSelect *)calloc(1, sizeof(pAudioSelect[0]));
            if (trackId != 0) {
                //もし、trackID=0以外の指定であれば、
                //これまでalltrackに指定されたパラメータを探して引き継ぐ
                sAudioSelect *pAudioSelectAll = nullptr;
                for (int itrack = 0; itrack < pParams->nAudioSelectCount; itrack++) {
                    if (pParams->ppAudioSelectList[itrack]->nAudioSelect == 0) {
                        pAudioSelectAll = pParams->ppAudioSelectList[itrack];
                    }
                }
                if (pAudioSelectAll) {
                    *pAudioSelect = *pAudioSelectAll;
                }
            }
            pAudioSelect->nAudioSelect = trackId;
        } else {
            pAudioSelect = pParams->ppAudioSelectList[audioIdx];
        }
        func_set(pAudioSelect, trackId, ptr);
        if (trackId == 0) {
            for (int itrack = 0; itrack < pParams->nAudioSelectCount; itrack++) {
                func_set(pParams->ppAudioSelectList[itrack], trackId, ptr);
            }
        }

        if (audioIdx < 0) {
            audioIdx = pParams->nAudioSelectCount;
            //新たに要素を追加
            pParams->ppAudioSelectList = (sAudioSelect **)realloc(pParams->ppAudioSelectList, sizeof(pParams->ppAudioSelectList[0]) * (pParams->nAudioSelectCount + 1));
            pParams->ppAudioSelectList[pParams->nAudioSelectCount] = pAudioSelect;
            pParams->nAudioSelectCount++;
        }
        return 0;
    };
    if (   0 == _tcscmp(option_name, _T("audio-copy"))
        || 0 == _tcscmp(option_name, _T("copy-audio"))) {
        pParams->nAVMux |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        std::set<int> trackSet; //重複しないよう、setを使う
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            auto trackListStr = split(strInput[i], _T(","));
            for (auto str : trackListStr) {
                int iTrack = 0;
                if (1 != _stscanf(str.c_str(), _T("%d"), &iTrack) || iTrack < 1) {
                    SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            sAudioSelect *pAudioSelect = nullptr;
            int audioIdx = getAudioTrackIdx(pParams, trackId);
            if (audioIdx < 0) {
                pAudioSelect = (sAudioSelect *)calloc(1, sizeof(pAudioSelect[0]));
                pAudioSelect->nAudioSelect = trackId;
            } else {
                pAudioSelect = pParams->ppAudioSelectList[audioIdx];
            }
            pAudioSelect->pAVAudioEncodeCodec = _tcsdup(RGY_AVCODEC_COPY);

            if (audioIdx < 0) {
                audioIdx = pParams->nAudioSelectCount;
                //新たに要素を追加
                pParams->ppAudioSelectList = (sAudioSelect **)realloc(pParams->ppAudioSelectList, sizeof(pParams->ppAudioSelectList[0]) * (pParams->nAudioSelectCount + 1));
                pParams->ppAudioSelectList[pParams->nAudioSelectCount] = pAudioSelect;
                pParams->nAudioSelectCount++;
            }
            argData->nParsedAudioCopy++;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-codec"))) {
        pParams->nAVMux |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        auto ret = set_audio_prm([](sAudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
            if (trackId != 0 || pAudioSelect->pAVAudioEncodeCodec == nullptr) {
                if (prmstr == nullptr) {
                    pAudioSelect->pAVAudioEncodeCodec = _tcsdup(RGY_AVCODEC_AUTO);
                } else {
                    pAudioSelect->pAVAudioEncodeCodec = _tcsdup(prmstr);
                    auto delim = _tcschr(pAudioSelect->pAVAudioEncodeCodec, _T(':'));
                    if (delim != nullptr) {
                        pAudioSelect->pAVAudioEncodeCodecPrm = _tcsdup(delim+1);
                        delim[0] = _T('\0');
                    } else {
                        pAudioSelect->pAVAudioEncodeCodecPrm = nullptr;
                    }
                }
            }
        });
        if (ret) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return ret;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-profile"))) {
        pParams->nAVMux |= (RGY_MUX_VIDEO | RGY_MUX_AUDIO);
        auto ret = set_audio_prm([](sAudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
            if (trackId != 0 || pAudioSelect->pAVAudioEncodeCodecProfile == nullptr) {
                pAudioSelect->pAVAudioEncodeCodecProfile = _tcsdup(prmstr);
            }
        });
        if (ret) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return ret;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-bitrate"))) {
        try {
            auto ret = set_audio_prm([](sAudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || pAudioSelect->nAVAudioEncodeBitrate == 0) {
                    pAudioSelect->nAVAudioEncodeBitrate = std::stoi(prmstr);
                }
            });
            return ret;
        } catch (...) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
    }
    if (0 == _tcscmp(option_name, _T("audio-ignore-decode-error"))) {
        i++;
        uint32_t value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nAudioIgnoreDecodeError = value;
        return 0;
    }
    //互換性のため残す
    if (0 == _tcscmp(option_name, _T("audio-ignore-notrack-error"))) {
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-samplerate"))) {
        try {
            auto ret = set_audio_prm([](sAudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || pAudioSelect->nAudioSamplingRate == 0) {
                    pAudioSelect->nAudioSamplingRate = std::stoi(prmstr);
                }
            });
            return ret;
        } catch (...) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
    }
    if (0 == _tcscmp(option_name, _T("audio-resampler"))) {
        i++;
        int v = 0;
        if (PARSE_ERROR_FLAG != (v = get_value_from_chr(list_resampler, strInput[i]))) {
            pParams->nAudioResampler = v;
        } else if (1 == _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_resampler) - 1) {
            pParams->nAudioResampler = v;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-stream"))) {
        //ここで、av_get_channel_layout()を使うため、チェックする必要がある
        if (!check_avcodec_dll()) {
            _ftprintf(stderr, _T("%s\n--audio-stream could not be used.\n"), error_mes_avcodec_dll_not_found().c_str());
            return 1;
        }

        try {
            auto ret = set_audio_prm([](sAudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || (pAudioSelect->pnStreamChannelSelect[0] == 0 && pAudioSelect->pnStreamChannelOut[0] == 0)) {
                    auto streamSelectList = split(tchar_to_string(prmstr), ",");
                    if (streamSelectList.size() > _countof(pAudioSelect->pnStreamChannelSelect)) {
                        return 1;
                    }
                    static const char *DELIM = ":";
                    for (uint32_t j = 0; j < streamSelectList.size(); j++) {
                        auto selectPtr = streamSelectList[j].c_str();
                        auto selectDelimPos = strstr(selectPtr, DELIM);
                        if (selectDelimPos == nullptr) {
                            auto channelLayout = av_get_channel_layout(selectPtr);
                            pAudioSelect->pnStreamChannelSelect[j] = channelLayout;
                            pAudioSelect->pnStreamChannelOut[j]    = RGY_CHANNEL_AUTO; //自動
                        } else if (selectPtr == selectDelimPos) {
                            pAudioSelect->pnStreamChannelSelect[j] = RGY_CHANNEL_AUTO;
                            pAudioSelect->pnStreamChannelOut[j]    = av_get_channel_layout(selectDelimPos + strlen(DELIM));
                        } else {
                            pAudioSelect->pnStreamChannelSelect[j] = av_get_channel_layout(streamSelectList[j].substr(0, selectDelimPos - selectPtr).c_str());
                            pAudioSelect->pnStreamChannelOut[j]    = av_get_channel_layout(selectDelimPos + strlen(DELIM));
                        }
                    }
                }
                return 0;
            });
            if (ret) {
                SET_ERR(strInput[0], _T("Too much streams splitted"), option_name, strInput[i]);
                return ret;
            }
            return ret;
        } catch (...) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
    }
    if (0 == _tcscmp(option_name, _T("audio-filter"))) {
        try {
            auto ret = set_audio_prm([](sAudioSelect *pAudioSelect, int trackId, const TCHAR *prmstr) {
                if (trackId != 0 || (pAudioSelect->pAudioFilter == nullptr)) {
                    if (pAudioSelect->pAudioFilter) {
                        free(pAudioSelect->pAudioFilter);
                    }
                    pAudioSelect->pAudioFilter = _tcsdup(prmstr);
                }
            });
            return ret;
        } catch (...) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
    }
#endif //#if ENABLE_AVCODEC_QSV_READER
    if (   0 == _tcscmp(option_name, _T("chapter-copy"))
        || 0 == _tcscmp(option_name, _T("copy-chapter"))) {
        pParams->bCopyChapter = TRUE;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("chapter"))) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            pParams->sChapterFile = strInput[i];
        } else {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i+1]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("key-on-chapter")) {
        pParams->keyOnChapter = true;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("keyfile"))) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            pParams->keyFile = strInput[i];
        } else {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i+1]);
            return 1;
        }
        return 0;
    }
    if (   0 == _tcscmp(option_name, _T("sub-copy"))
        || 0 == _tcscmp(option_name, _T("copy-sub"))) {
        pParams->nAVMux |= (RGY_MUX_VIDEO | RGY_MUX_SUBTITLE);
        std::set<int> trackSet; //重複しないよう、setを使う
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            auto trackListStr = split(strInput[i], _T(","));
            for (auto str : trackListStr) {
                int iTrack = 0;
                if (1 != _stscanf(str.c_str(), _T("%d"), &iTrack) || iTrack < 1) {
                    SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                    return 1;
                } else {
                    trackSet.insert(iTrack);
                }
            }
        } else {
            trackSet.insert(0);
        }
        for (int iTrack = 0; iTrack < pParams->nSubtitleSelectCount; iTrack++) {
            trackSet.insert(pParams->pSubtitleSelect[iTrack]);
        }
        if (pParams->pSubtitleSelect) {
            free(pParams->pSubtitleSelect);
        }

        pParams->pSubtitleSelect = (int *)malloc(sizeof(pParams->pSubtitleSelect[0]) * trackSet.size());
        pParams->nSubtitleSelectCount = (int)trackSet.size();
        int iTrack = 0;
        for (auto it = trackSet.begin(); it != trackSet.end(); it++, iTrack++) {
            pParams->pSubtitleSelect[iTrack] = *it;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("caption2ass"))) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            C2AFormat format = FORMAT_INVALID;
            if (PARSE_ERROR_FLAG != (format = (C2AFormat)get_value_from_chr(list_caption2ass, strInput[i]))) {
                pParams->caption2ass = format;
            } else {
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return 1;
            }
        } else {
            pParams->caption2ass = FORMAT_SRT;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-caption2ass"))) {
        pParams->caption2ass = FORMAT_INVALID;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("avsync"))) {
        int value = 0;
        i++;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_avsync, strInput[i]))) {
            pParams->nAVSyncMode = (RGYAVSync)value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("mux-option"))) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            auto ptr = _tcschr(strInput[i], ':');
            if (ptr == nullptr) {
                SET_ERR(strInput[0], _T("invalid value"), option_name, nullptr);
                return 1;
            } else {
                if (pParams->pMuxOpt == nullptr) {
                    pParams->pMuxOpt = new muxOptList();
                }
                pParams->pMuxOpt->push_back(std::make_pair<tstring, tstring>(tstring(strInput[i]).substr(0, ptr - strInput[i]), tstring(ptr+1)));
            }
        } else {
            SET_ERR(strInput[0], _T("invalid option"), option_name, nullptr);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("cqp")) {
        i++;
        int a[3] = { 0 };
        if (   3 == _stscanf_s(strInput[i], _T("%d:%d:%d"), &a[0], &a[1], &a[2])
            || 3 == _stscanf_s(strInput[i], _T("%d/%d/%d"), &a[0], &a[1], &a[2])
            || 3 == _stscanf_s(strInput[i], _T("%d.%d.%d"), &a[0], &a[1], &a[2])
            || 3 == _stscanf_s(strInput[i], _T("%d,%d,%d"), &a[0], &a[1], &a[2])) {
            pParams->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
            pParams->encConfig.rcParams.constQP.qpIntra  = a[0];
            pParams->encConfig.rcParams.constQP.qpInterP = a[1];
            pParams->encConfig.rcParams.constQP.qpInterB = a[2];
            return 0;
        }
        if (   2 == _stscanf_s(strInput[i], _T("%d:%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d/%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d.%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d,%d"), &a[0], &a[1])) {
            pParams->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
            pParams->encConfig.rcParams.constQP.qpIntra  = a[0];
            pParams->encConfig.rcParams.constQP.qpInterP = a[1];
            pParams->encConfig.rcParams.constQP.qpInterB = a[1];
            return 0;
        }
        if (1 == _stscanf_s(strInput[i], _T("%d"), &a[0])) {
            pParams->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
            pParams->encConfig.rcParams.constQP.qpIntra  = a[0];
            pParams->encConfig.rcParams.constQP.qpInterP = a[0];
            pParams->encConfig.rcParams.constQP.qpInterB = a[0];
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("vbr")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
            pParams->encConfig.rcParams.averageBitRate = value * 1000;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("vbrhq") || IS_OPTION("vbr2")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR_HQ;
            pParams->encConfig.rcParams.averageBitRate = value * 1000;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("cbr")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
            pParams->encConfig.rcParams.averageBitRate = value * 1000;
            pParams->encConfig.rcParams.maxBitRate = value * 1000;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("cbrhq")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR_HQ;
            pParams->encConfig.rcParams.averageBitRate = value * 1000;
            pParams->encConfig.rcParams.maxBitRate = value * 1000;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("vbr-quality")) {
        i++;
        double value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%lf"), &value)) {
            value = (std::max)(0.0, value);
            int value_int = (int)value;
            pParams->encConfig.rcParams.targetQuality = (uint8_t)clamp(value_int, 0, 51);
            pParams->encConfig.rcParams.targetQualityLSB = (uint8_t)clamp((int)((value - value_int) * 256.0), 0, 255);
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("qp-init") || IS_OPTION("qp-max") || IS_OPTION("qp-min")) {
        i++;
        int a[4] = { 0 };
        if (   4 == _stscanf_s(strInput[i], _T("%d;%d:%d:%d"), &a[3], &a[0], &a[1], &a[2])
            || 4 == _stscanf_s(strInput[i], _T("%d;%d/%d/%d"), &a[3], &a[0], &a[1], &a[2])
            || 4 == _stscanf_s(strInput[i], _T("%d;%d.%d.%d"), &a[3], &a[0], &a[1], &a[2])
            || 4 == _stscanf_s(strInput[i], _T("%d;%d,%d,%d"), &a[3], &a[0], &a[1], &a[2])) {
            a[3] = a[3] ? 1 : 0;
        } else if (
               3 == _stscanf_s(strInput[i], _T("%d:%d:%d"), &a[0], &a[1], &a[2])
            || 3 == _stscanf_s(strInput[i], _T("%d/%d/%d"), &a[0], &a[1], &a[2])
            || 3 == _stscanf_s(strInput[i], _T("%d.%d.%d"), &a[0], &a[1], &a[2])
            || 3 == _stscanf_s(strInput[i], _T("%d,%d,%d"), &a[0], &a[1], &a[2])) {
            a[3] = 1;
        } else if (
               3 == _stscanf_s(strInput[i], _T("%d;%d:%d"), &a[3], &a[0], &a[1])
            || 3 == _stscanf_s(strInput[i], _T("%d;%d/%d"), &a[3], &a[0], &a[1])
            || 3 == _stscanf_s(strInput[i], _T("%d;%d.%d"), &a[3], &a[0], &a[1])
            || 3 == _stscanf_s(strInput[i], _T("%d;%d,%d"), &a[3], &a[0], &a[1])) {
            a[3] = a[3] ? 1 : 0;
            a[2] = a[1];
        } else if (
               2 == _stscanf_s(strInput[i], _T("%d:%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d/%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d.%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d,%d"), &a[0], &a[1])) {
            a[3] = 1;
            a[2] = a[1];
        } else if (2 == _stscanf_s(strInput[i], _T("%d;%d"), &a[3], &a[0])) {
            a[3] = a[3] ? 1 : 0;
            a[1] = a[0];
            a[2] = a[0];
        } else if (1 == _stscanf_s(strInput[i], _T("%d"), &a[0])) {
            a[3] = 1;
            a[1] = a[0];
            a[2] = a[0];
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        NV_ENC_QP *ptrQP = nullptr;
        if (IS_OPTION("qp-init")) {
            pParams->encConfig.rcParams.enableInitialRCQP = a[3];
            ptrQP = &pParams->encConfig.rcParams.initialRCQP;
        } else if (IS_OPTION("qp-max")) {
            pParams->encConfig.rcParams.enableMaxQP = a[3];
            ptrQP = &pParams->encConfig.rcParams.maxQP;
        } else if (IS_OPTION("qp-min")) {
            pParams->encConfig.rcParams.enableMinQP = a[3];
            ptrQP = &pParams->encConfig.rcParams.minQP;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        ptrQP->qpIntra  = a[0];
        ptrQP->qpInterP = a[1];
        ptrQP->qpInterB = a[2];
        return 0;
    }
    if (IS_OPTION("gop-len")) {
        i++;
        int value = 0;
        if (0 == _tcsnccmp(strInput[i], _T("auto"), _tcslen(_T("auto")))) {
            pParams->encConfig.gopLength = 0;
        } else if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->encConfig.gopLength = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("strict-gop")) {
        pParams->encConfig.rcParams.strictGOPTarget = 1;
        return 0;
    }
    if (IS_OPTION("bframes")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            argData->nBframes = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("bref-mode")) {
        i++;
        int value = 0;
        if (get_list_value(list_bref_mode, strInput[i], &value)) {
            codecPrm[NV_ENC_H264].h264Config.useBFramesAsRef = (NV_ENC_BFRAME_REF_MODE)value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("max-bitrate") || IS_OPTION("maxbitrate")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->encConfig.rcParams.maxBitRate = value * 1000;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("lookahead")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->encConfig.rcParams.enableLookahead = value > 0;
            pParams->encConfig.rcParams.lookaheadDepth = (uint16_t)clamp(value, 0, 32);
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("no-i-adapt")) {
        pParams->encConfig.rcParams.disableIadapt = 1;
        return 0;
    }
    if (IS_OPTION("no-b-adapt")) {
        pParams->encConfig.rcParams.disableBadapt = 1;
        return 0;
    }
    if (IS_OPTION("vbv-bufsize")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->encConfig.rcParams.vbvBufferSize = value * 1000;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("aq")) {
        pParams->encConfig.rcParams.enableAQ = 1;
        return 0;
    }
    if (IS_OPTION("aq-temporal")) {
        pParams->encConfig.rcParams.enableTemporalAQ = 1;
        return 0;
    }
    if (IS_OPTION("aq-strength")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->encConfig.rcParams.aqStrength = clamp(value, 0, 15);
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("disable-aq")
        || IS_OPTION("no-aq")) {
        pParams->encConfig.rcParams.enableAQ = 0;
        return 0;
    }
    if (IS_OPTION("direct")) {
        i++;
        int value = 0;
        if (get_list_value(list_bdirect, strInput[i], &value)) {
            codecPrm[NV_ENC_H264].h264Config.bdirectMode = (NV_ENC_H264_BDIRECT_MODE)value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("adapt-transform")) {
        codecPrm[NV_ENC_H264].h264Config.adaptiveTransformMode = NV_ENC_H264_ADAPTIVE_TRANSFORM_ENABLE;
        return 0;
    }
    if (IS_OPTION("no-adapt-transform")) {
        codecPrm[NV_ENC_H264].h264Config.adaptiveTransformMode = NV_ENC_H264_ADAPTIVE_TRANSFORM_DISABLE;
        return 0;
    }
    if (IS_OPTION("ref")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            codecPrm[NV_ENC_H264].h264Config.maxNumRefFrames = value;
            codecPrm[NV_ENC_HEVC].hevcConfig.maxNumRefFramesInDPB = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("weightp")) {
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            pParams->nWeightP = 1;
            return 0;
        }
        i++;
        if (0 == _tcscmp(strInput[i], _T("force"))) {
            pParams->nWeightP = 2;
        }
        return 0;
    }
    if (IS_OPTION("mv-precision")) {
        i++;
        int value = 0;
        if (get_list_value(list_mv_presicion, strInput[i], &value)) {
            pParams->encConfig.mvPrecision = (NV_ENC_MV_PRECISION)value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("vpp-deinterlace")) {
        i++;
        int value = 0;
        if (get_list_value(list_deinterlace, strInput[i], &value)) {
            pParams->vpp.deinterlace = (cudaVideoDeinterlaceMode)value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        if (pParams->vpp.deinterlace != cudaVideoDeinterlaceMode_Weave
            && pParams->input.picstruct & RGY_PICSTRUCT_INTERLACED) {
            pParams->input.picstruct = RGY_PICSTRUCT_FRAME_TFF;
        }
        return 0;
    }
    if (IS_OPTION("vpp-resize")) {
        i++;
        int value = 0;
        if (get_list_value(list_nppi_resize, strInput[i], &value)) {
            pParams->vpp.resizeInterp = (NppiInterpolationMode)value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("vpp-gauss")) {
        i++;
        int value = 0;
        if (get_list_value(list_nppi_gauss, strInput[i], &value)) {
            pParams->vpp.gaussMaskSize = (NppiMaskSize)value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("vpp-unsharp")) {
        pParams->vpp.unsharp.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            pParams->vpp.unsharp.radius = FILTER_DEFAULT_UNSHARP_RADIUS;
            return 0;
        }
        i++;
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                std::transform(param_arg.begin(), param_arg.end(), param_arg.begin(), tolower);
                if (param_arg == _T("enable")) {
                    if (param_val == _T("true")) {
                        pParams->vpp.unsharp.enable = true;
                    } else if (param_val == _T("false")) {
                        pParams->vpp.unsharp.enable = false;
                    } else {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("radius")) {
                    try {
                        pParams->vpp.unsharp.radius = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("weight")) {
                    try {
                        pParams->vpp.unsharp.weight = std::stof(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("threshold")) {
                    try {
                        pParams->vpp.unsharp.threshold = std::stof(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-edgelevel")) {
        pParams->vpp.edgelevel.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                std::transform(param_arg.begin(), param_arg.end(), param_arg.begin(), tolower);
                if (param_arg == _T("enable")) {
                    if (param_val == _T("true")) {
                        pParams->vpp.edgelevel.enable = true;
                    } else if (param_val == _T("false")) {
                        pParams->vpp.edgelevel.enable = false;
                    } else {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("strength")) {
                    try {
                        pParams->vpp.edgelevel.strength = std::stof(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("threshold")) {
                    try {
                        pParams->vpp.edgelevel.threshold = std::stof(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("black")) {
                    try {
                        pParams->vpp.edgelevel.black = std::stof(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("white")) {
                    try {
                        pParams->vpp.edgelevel.white = std::stof(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-delogo-select")) {
        i++;
        pParams->vpp.delogo.logoSelect = strInput[i];
        return 0;
    }
    if (IS_OPTION("vpp-delogo-add")) {
        pParams->vpp.delogo.mode = DELOGO_MODE_ADD;
        return 0;
    }
    if (IS_OPTION("vpp-delogo-pos")) {
        i++;
        int posOffsetX, posOffsetY;
        if (   2 != _stscanf_s(strInput[i], _T("%dx%d"), &posOffsetX, &posOffsetY)
            && 2 != _stscanf_s(strInput[i], _T("%d,%d"), &posOffsetX, &posOffsetY)
            && 2 != _stscanf_s(strInput[i], _T("%d/%d"), &posOffsetX, &posOffsetY)
            && 2 != _stscanf_s(strInput[i], _T("%d:%d"), &posOffsetX, &posOffsetY)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        pParams->vpp.delogo.posX = posOffsetX;
        pParams->vpp.delogo.posY = posOffsetY;
        return 0;
    }
    if (IS_OPTION("vpp-delogo-depth")) {
        i++;
        int depth;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &depth)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        pParams->vpp.delogo.depth = depth;
        return 0;
    }
    if (IS_OPTION("vpp-delogo-y")) {
        i++;
        int value;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        pParams->vpp.delogo.Y = value;
        return 0;
    }
    if (IS_OPTION("vpp-delogo-cb")) {
        i++;
        int value;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        pParams->vpp.delogo.Cb = value;
        return 0;
    }
    if (IS_OPTION("vpp-delogo-cr")) {
        i++;
        int value;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        pParams->vpp.delogo.Cr = value;
        return 0;
    }

    if (IS_OPTION("vpp-delogo")) {
        pParams->vpp.delogo.enable = true;

        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;
        vector<tstring> param_list;
        bool flag_comma = false;
        const TCHAR *pstr = strInput[i];
        const TCHAR *qstr = strInput[i];
        for (; *pstr; pstr++) {
            if (*pstr == _T('\"')) {
                flag_comma ^= true;
            }
            if (!flag_comma && *pstr == _T(',')) {
                param_list.push_back(tstring(qstr, pstr - qstr));
                qstr = pstr+1;
            }
        }
        param_list.push_back(tstring(qstr, pstr - qstr));

        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                std::transform(param_arg.begin(), param_arg.end(), param_arg.begin(), tolower);
                if (param_arg == _T("enable")) {
                    if (param_val == _T("true")) {
                        pParams->vpp.delogo.enable = true;
                    } else if (param_val == _T("false")) {
                        pParams->vpp.delogo.enable = false;
                    } else {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("file")) {
                    try {
                        pParams->vpp.delogo.logoFilePath = param_val;
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("select")) {
                    try {
                        pParams->vpp.delogo.logoSelect = param_val;
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("add")) {
                    if (param_val == _T("true")) {
                        pParams->vpp.delogo.mode = DELOGO_MODE_ADD;
                    } else if (param_val == _T("false")) {
                        pParams->vpp.delogo.mode = DELOGO_MODE_REMOVE;
                    } else {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("pos")) {
                    int posOffsetX, posOffsetY;
                    if (   2 != _stscanf_s(param_val.c_str(), _T("%dx%d"), &posOffsetX, &posOffsetY)
                        && 2 != _stscanf_s(param_val.c_str(), _T("%d,%d"), &posOffsetX, &posOffsetY)
                        && 2 != _stscanf_s(param_val.c_str(), _T("%d/%d"), &posOffsetX, &posOffsetY)
                        && 2 != _stscanf_s(param_val.c_str(), _T("%d:%d"), &posOffsetX, &posOffsetY)) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    pParams->vpp.delogo.posX = posOffsetX;
                    pParams->vpp.delogo.posY = posOffsetY;
                    continue;
                }
                if (param_arg == _T("depth")) {
                    try {
                        pParams->vpp.delogo.depth = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("y")) {
                    try {
                        pParams->vpp.delogo.Y = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("cb")) {
                    try {
                        pParams->vpp.delogo.Cb = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("cr")) {
                    try {
                        pParams->vpp.delogo.Cr = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("auto_nr")) {
                    if (param_val == _T("true") || param_val == _T("on")) {
                        pParams->vpp.delogo.autoNR = true;
                    } else if (param_val == _T("false") || param_val == _T("off")) {
                        pParams->vpp.delogo.autoNR = false;
                    } else {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("auto_fade")) {
                    if (param_val == _T("true") || param_val == _T("on")) {
                        pParams->vpp.delogo.autoFade = true;
                    } else if (param_val == _T("false") || param_val == _T("off")) {
                        pParams->vpp.delogo.autoFade = false;
                    } else {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("nr_area")) {
                    try {
                        pParams->vpp.delogo.NRArea = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("nr_value")) {
                    try {
                        pParams->vpp.delogo.NRValue = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("log")) {
                    if (param_val == _T("true") || param_val == _T("on")) {
                        pParams->vpp.delogo.log = true;
                    } else if (param_val == _T("false") || param_val == _T("off")) {
                        pParams->vpp.delogo.log = false;
                    } else {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
            } else {
                pParams->vpp.delogo.logoFilePath = param;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-knn")) {
        pParams->vpp.knn.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            pParams->vpp.knn.radius = FILTER_DEFAULT_KNN_RADIUS;
            return 0;
        }
        i++;
        int radius = FILTER_DEFAULT_KNN_RADIUS;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &radius)) {
            for (const auto& param : split(strInput[i], _T(","))) {
                auto pos = param.find_first_of(_T("="));
                if (pos != std::string::npos) {
                    auto param_arg = param.substr(0, pos);
                    auto param_val = param.substr(pos+1);
                    std::transform(param_arg.begin(), param_arg.end(), param_arg.begin(), tolower);
                    if (param_arg == _T("enable")) {
                        if (param_val == _T("true")) {
                            pParams->vpp.knn.enable = true;
                        } else if (param_val == _T("false")) {
                            pParams->vpp.knn.enable = false;
                        } else {
                            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                            return -1;
                        }
                        continue;
                    }
                    if (param_arg == _T("radius")) {
                        try {
                            pParams->vpp.knn.radius = std::stoi(param_val);
                        } catch (...) {
                            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                            return -1;
                        }
                        continue;
                    }
                    if (param_arg == _T("strength")) {
                        try {
                            pParams->vpp.knn.strength = std::stof(param_val);
                        } catch (...) {
                            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                            return -1;
                        }
                        continue;
                    }
                    if (param_arg == _T("lerp")) {
                        try {
                            pParams->vpp.knn.lerpC = std::stof(param_val);
                        } catch (...) {
                            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                            return -1;
                        }
                        continue;
                    }
                    if (param_arg == _T("th_weight")) {
                        try {
                            pParams->vpp.knn.weight_threshold = std::stof(param_val);
                        } catch (...) {
                            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                            return -1;
                        }
                        continue;
                    }
                    if (param_arg == _T("th_lerp")) {
                        try {
                            pParams->vpp.knn.lerp_threshold = std::stof(param_val);
                        } catch (...) {
                            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                            return -1;
                        }
                        continue;
                    }
                    SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                    return -1;
                }
            }
        } else {
            pParams->vpp.knn.radius = radius;
        }
        return 0;
    }
    if (IS_OPTION("vpp-pmd")) {
        pParams->vpp.pmd.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                std::transform(param_arg.begin(), param_arg.end(), param_arg.begin(), tolower);
                if (param_arg == _T("enable")) {
                    if (param_val == _T("true")) {
                        pParams->vpp.pmd.enable = true;
                    } else if (param_val == _T("false")) {
                        pParams->vpp.pmd.enable = false;
                    } else {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("apply_count")) {
                    try {
                        pParams->vpp.pmd.applyCount = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("strength")) {
                    try {
                        pParams->vpp.pmd.strength = std::stof(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("threshold")) {
                    try {
                        pParams->vpp.pmd.threshold = std::stof(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("useexp")) {
                    try {
                        pParams->vpp.pmd.useExp = std::stoi(param_val) != 0;
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
            }
        }
        return 0;
    }

    if (IS_OPTION("vpp-deband")) {
        pParams->vpp.deband.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                std::transform(param_arg.begin(), param_arg.end(), param_arg.begin(), tolower);
                if (param_arg == _T("enable")) {
                    if (param_val == _T("true")) {
                        pParams->vpp.deband.enable = true;
                    } else if (param_val == _T("false")) {
                        pParams->vpp.deband.enable = false;
                    } else {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("range")) {
                    try {
                        pParams->vpp.deband.range = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("thre")) {
                    try {
                        pParams->vpp.deband.threY = std::stoi(param_val);
                        pParams->vpp.deband.threCb = pParams->vpp.deband.threY;
                        pParams->vpp.deband.threCr = pParams->vpp.deband.threY;
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_y")) {
                    try {
                        pParams->vpp.deband.threY = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_cb")) {
                    try {
                        pParams->vpp.deband.threCb = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_cr")) {
                    try {
                        pParams->vpp.deband.threCr = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("dither")) {
                    try {
                        pParams->vpp.deband.ditherY = std::stoi(param_val);
                        pParams->vpp.deband.ditherC = pParams->vpp.deband.ditherY;
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("dither_y")) {
                    try {
                        pParams->vpp.deband.ditherY = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("dither_c")) {
                    try {
                        pParams->vpp.deband.ditherC = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("sample")) {
                    try {
                        pParams->vpp.deband.sample = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("seed")) {
                    try {
                        pParams->vpp.deband.seed = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("blurfirst")) {
                    pParams->vpp.deband.blurFirst = (param_val == _T("true")) || (param_val == _T("on"));
                    continue;
                }
                if (param_arg == _T("rand_each_frame")) {
                    pParams->vpp.deband.randEachFrame = (param_val == _T("true")) || (param_val == _T("on"));
                    continue;
                }
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
            } else {
                if (param == _T("blurfirst")) {
                    pParams->vpp.deband.blurFirst = true;
                    continue;
                }
                if (param == _T("rand_each_frame")) {
                    pParams->vpp.deband.randEachFrame = true;
                    continue;
                }
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-afs")) {
        pParams->vpp.afs.enable = true;

        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;
        vector<tstring> param_list;
        bool flag_comma = false;
        const TCHAR *pstr = strInput[i];
        const TCHAR *qstr = strInput[i];
        for (; *pstr; pstr++) {
            if (*pstr == _T('\"')) {
                flag_comma ^= true;
            }
            if (!flag_comma && *pstr == _T(',')) {
                param_list.push_back(tstring(qstr, pstr - qstr));
                qstr = pstr+1;
            }
        }
        param_list.push_back(tstring(qstr, pstr - qstr));
        for (const auto& param : param_list) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                std::transform(param_arg.begin(), param_arg.end(), param_arg.begin(), tolower);
                if (param_arg == _T("ini")) {
                    if (NVEncFilterAfs::read_afs_inifile(&pParams->vpp.afs, param_val.c_str())) {
                        SET_ERR(strInput[0], _T("ini file does not exist."), option_name, strInput[i]);
                        return -1;
                    }
                }
                if (param_arg == _T("preset")) {
                    try {
                        int value = 0;
                        if (get_list_value(list_afs_preset, param_val.c_str(), &value)) {
                            NVEncFilterAfs::set_preset(&pParams->vpp.afs, value);
                        } else {
                            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                            return -1;
                        }
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
            }
        }
        for (const auto& param : param_list) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                std::transform(param_arg.begin(), param_arg.end(), param_arg.begin(), tolower);
                if (param_arg == _T("enable")) {
                    if (param_val == _T("true")) {
                        pParams->vpp.afs.enable = true;
                    } else if (param_val == _T("false")) {
                        pParams->vpp.afs.enable = false;
                    } else {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("top")) {
                    try {
                        pParams->vpp.afs.clip.top = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("bottom")) {
                    try {
                        pParams->vpp.afs.clip.bottom = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("left")) {
                    try {
                        pParams->vpp.afs.clip.left = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("right")) {
                    try {
                        pParams->vpp.afs.clip.right = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("method_switch")) {
                    try {
                        pParams->vpp.afs.method_switch = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("coeff_shift")) {
                    try {
                        pParams->vpp.afs.coeff_shift = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_shift")) {
                    try {
                        pParams->vpp.afs.thre_shift = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_deint")) {
                    try {
                        pParams->vpp.afs.thre_deint = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_motion_y")) {
                    try {
                        pParams->vpp.afs.thre_Ymotion = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_motion_c")) {
                    try {
                        pParams->vpp.afs.thre_Cmotion = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("level")) {
                    try {
                        pParams->vpp.afs.analyze = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("shift")) {
                    pParams->vpp.afs.shift = (param_val == _T("true")) || (param_val == _T("on"));
                    continue;
                }
                if (param_arg == _T("drop")) {
                    pParams->vpp.afs.drop = (param_val == _T("true")) || (param_val == _T("on"));
                    continue;
                }
                if (param_arg == _T("smooth")) {
                    pParams->vpp.afs.smooth = (param_val == _T("true")) || (param_val == _T("on"));
                    continue;
                }
                if (param_arg == _T("24fps")) {
                    pParams->vpp.afs.force24 = (param_val == _T("true")) || (param_val == _T("on"));
                    continue;
                }
                if (param_arg == _T("tune")) {
                    pParams->vpp.afs.tune = (param_val == _T("true")) || (param_val == _T("on"));
                    continue;
                }
                if (param_arg == _T("rff")) {
                    pParams->vpp.afs.rff = (param_val == _T("true")) || (param_val == _T("on"));
                    continue;
                }
                if (param_arg == _T("timecode")) {
                    pParams->vpp.afs.timecode = (param_val == _T("true")) || (param_val == _T("on"));
                    continue;
                }
                if (param_arg == _T("log")) {
                    pParams->vpp.afs.log = (param_val == _T("true")) || (param_val == _T("on"));
                    continue;
                }
                if (param_arg == _T("ini")) {
                    continue;
                }
                if (param_arg == _T("preset")) {
                    continue;
                }
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
            } else {
                if (param == _T("shift")) {
                    pParams->vpp.afs.shift = true;
                    continue;
                }
                if (param == _T("drop")) {
                    pParams->vpp.afs.drop = true;
                    continue;
                }
                if (param == _T("smooth")) {
                    pParams->vpp.afs.smooth = true;
                    continue;
                }
                if (param == _T("24fps")) {
                    pParams->vpp.afs.force24 = true;
                    continue;
                }
                if (param == _T("tune")) {
                    pParams->vpp.afs.tune = true;
                    continue;
                }
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-rff")) {
        pParams->vpp.rff = true;
        return 0;
    }

    if (IS_OPTION("vpp-tweak")) {
        pParams->vpp.tweak.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                std::transform(param_arg.begin(), param_arg.end(), param_arg.begin(), tolower);
                if (param_arg == _T("enable")) {
                    if (param_val == _T("true")) {
                        pParams->vpp.tweak.enable = true;
                    } else if (param_val == _T("false")) {
                        pParams->vpp.tweak.enable = false;
                    } else {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("brightness")) {
                    try {
                        pParams->vpp.tweak.brightness = std::stof(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("contrast")) {
                    try {
                        pParams->vpp.tweak.contrast = std::stof(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("gamma")) {
                    try {
                        pParams->vpp.tweak.gamma = std::stof(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("saturation")) {
                    try {
                        pParams->vpp.tweak.saturation = std::stof(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("hue")) {
                    try {
                        pParams->vpp.tweak.hue = std::stof(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
            } else {
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
            }
        }
        return 0;
    }

    if (IS_OPTION("vpp-pad")) {
        pParams->vpp.pad.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                std::transform(param_arg.begin(), param_arg.end(), param_arg.begin(), tolower);
                if (param_arg == _T("enable")) {
                    if (param_val == _T("true")) {
                        pParams->vpp.pad.enable = true;
                    } else if (param_val == _T("false")) {
                        pParams->vpp.pad.enable = false;
                    } else {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("r")) {
                    try {
                        pParams->vpp.pad.right = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("l")) {
                    try {
                        pParams->vpp.pad.left = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("t")) {
                    try {
                        pParams->vpp.pad.top = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("b")) {
                    try {
                        pParams->vpp.pad.bottom = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
            } else {
                int val[4] = { 0 };
                if (   4 == _stscanf_s(strInput[i], _T("%d,%d,%d,%d"), &val[0], &val[1], &val[2], &val[3])
                    || 4 == _stscanf_s(strInput[i], _T("%d:%d:%d:%d"), &val[0], &val[1], &val[2], &val[3])) {
                    pParams->vpp.pad.left   = val[0];
                    pParams->vpp.pad.top    = val[1];
                    pParams->vpp.pad.right  = val[2];
                    pParams->vpp.pad.bottom = val[3];
                    continue;
                }
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
            }
        }
        return 0;
    }

    if (IS_OPTION("vpp-select-every")) {
        pParams->vpp.selectevery.enable = true;
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;
        for (const auto& param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                std::transform(param_arg.begin(), param_arg.end(), param_arg.begin(), tolower);
                if (param_arg == _T("enable")) {
                    if (param_val == _T("true")) {
                        pParams->vpp.selectevery.enable = true;
                    } else if (param_val == _T("false")) {
                        pParams->vpp.selectevery.enable = false;
                    } else {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("offset")) {
                    try {
                        pParams->vpp.selectevery.offset = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("step")) {
                    try {
                        pParams->vpp.selectevery.step = std::stoi(param_val);
                    } catch (...) {
                        SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
            } else {
                try {
                    pParams->vpp.selectevery.step = std::stoi(strInput[i]);
                } catch (...) {
                    SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                    return -1;
                }
                continue;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-perf-monitor")) {
        pParams->vpp.bCheckPerformance = true;
        return 0;
    }
    if (IS_OPTION("no-vpp-perf-monitor")) {
        pParams->vpp.bCheckPerformance = false;
        return 0;
    }
    if (IS_OPTION("tff")) {
        pParams->input.picstruct = RGY_PICSTRUCT_FRAME_TFF;
        return 0;
    }
    if (IS_OPTION("bff")) {
        pParams->input.picstruct = RGY_PICSTRUCT_FRAME_BFF;
        return 0;
    }
    if (IS_OPTION("interlace") || IS_OPTION("interlaced")) {
        i++;
        int value = 0;
        if (get_list_value(list_interlaced, strInput[i], &value)) {
            pParams->input.picstruct = (RGY_PICSTRUCT)value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("cavlc")) {
        codecPrm[NV_ENC_H264].h264Config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
        return 0;
    }
    if (IS_OPTION("cabac")) {
        codecPrm[NV_ENC_H264].h264Config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_CABAC;
        return 0;
    }
    if (IS_OPTION("bluray")) {
        pParams->bluray = TRUE;
        return 0;
    }
    if (IS_OPTION("lossless")) {
        pParams->lossless = TRUE;
        return 0;
    }
    if (IS_OPTION("no-deblock")) {
        codecPrm[NV_ENC_H264].h264Config.disableDeblockingFilterIDC = 1;
        return 0;
    }
    if (IS_OPTION("slices:h264")) {
        i++;
        try {
            int value = std::stoi(strInput[i]);
            codecPrm[NV_ENC_H264].h264Config.sliceMode = 3;
            codecPrm[NV_ENC_H264].h264Config.sliceModeData = value;
        } catch (...) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("slices:hevc")) {
        i++;
        try {
            int value = std::stoi(strInput[i]);
            codecPrm[NV_ENC_HEVC].hevcConfig.sliceMode = 3;
            codecPrm[NV_ENC_HEVC].hevcConfig.sliceModeData = value;
        } catch (...) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("slices")) {
        i++;
        try {
            int value = std::stoi(strInput[i]);
            codecPrm[NV_ENC_H264].h264Config.sliceMode = 3;
            codecPrm[NV_ENC_HEVC].hevcConfig.sliceMode = 3;
            codecPrm[NV_ENC_H264].h264Config.sliceModeData = value;
            codecPrm[NV_ENC_HEVC].hevcConfig.sliceModeData = value;
        } catch (...) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("deblock")) {
        codecPrm[NV_ENC_H264].h264Config.disableDeblockingFilterIDC = 0;
        return 0;
    }
    if (IS_OPTION("aud:h264")) {
        codecPrm[NV_ENC_H264].h264Config.outputAUD = 1;
        return 0;
    }
    if (IS_OPTION("aud:hevc")) {
        codecPrm[NV_ENC_HEVC].hevcConfig.outputAUD = 1;
        return 0;
    }
    if (IS_OPTION("aud")) {
        codecPrm[NV_ENC_H264].h264Config.outputAUD = 1;
        codecPrm[NV_ENC_HEVC].hevcConfig.outputAUD = 1;
        return 0;
    }
    if (IS_OPTION("pic-struct:h264")) {
        codecPrm[NV_ENC_H264].h264Config.outputPictureTimingSEI = 1;
        return 0;
    }
    if (IS_OPTION("pic-struct:hevc")) {
        codecPrm[NV_ENC_HEVC].hevcConfig.outputPictureTimingSEI = 1;
        return 0;
    }
    if (IS_OPTION("pic-struct")) {
        codecPrm[NV_ENC_H264].h264Config.outputPictureTimingSEI = 1;
        codecPrm[NV_ENC_HEVC].hevcConfig.outputPictureTimingSEI = 1;
        return 0;
    }
    if (IS_OPTION("fullrange:h264")) {
        codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.videoFullRangeFlag = 1;
        return 0;
    }
    if (IS_OPTION("fullrange:hevc")) {
        codecPrm[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.videoFullRangeFlag = 1;
        return 0;
    }
    if (IS_OPTION("fullrange")) {
        codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.videoFullRangeFlag = 1;
        codecPrm[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.videoFullRangeFlag = 1;
        return 0;
    }
    if (IS_OPTION("videoformat") || IS_OPTION("videoformat:h264") || IS_OPTION("videoformat:hevc")) {
        const bool for_h264 = IS_OPTION("videoformat") || IS_OPTION("videoformat:h264");
        const bool for_hevc = IS_OPTION("videoformat") || IS_OPTION("videoformat:hevc");
        i++;
        int value = 0;
        if (get_list_value(list_videoformat, strInput[i], &value)) {
            if (for_h264) codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.videoFormat = value;
            if (for_hevc) codecPrm[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.videoFormat = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("colormatrix") || IS_OPTION("colormatrix:h264") || IS_OPTION("colormatrix:hevc")) {
        const bool for_h264 = IS_OPTION("colormatrix") || IS_OPTION("colormatrix:h264");
        const bool for_hevc = IS_OPTION("colormatrix") || IS_OPTION("colormatrix:hevc");
        i++;
        int value = 0;
        if (get_list_value(list_colormatrix, strInput[i], &value)) {
            if (for_h264) codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.colourMatrix = value;
            if (for_hevc) codecPrm[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.colourMatrix = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("colorprim") || IS_OPTION("colorprim:h264") || IS_OPTION("colorprim:hevc")) {
        const bool for_h264 = IS_OPTION("colorprim") || IS_OPTION("colorprim:h264");
        const bool for_hevc = IS_OPTION("colorprim") || IS_OPTION("colorprim:hevc");
        i++;
        int value = 0;
        if (get_list_value(list_colorprim, strInput[i], &value)) {
            if (for_h264) codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.colourPrimaries = value;
            if (for_hevc) codecPrm[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.colourPrimaries = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("transfer") || IS_OPTION("transfer:h264") || IS_OPTION("transfer:hevc")) {
        const bool for_h264 = IS_OPTION("transfer") || IS_OPTION("transfer:h264");
        const bool for_hevc = IS_OPTION("transfer") || IS_OPTION("transfer:hevc");
        i++;
        int value = 0;
        if (get_list_value(list_transfer, strInput[i], &value)) {
            if (for_h264) codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.transferCharacteristics = value;
            if (for_hevc) codecPrm[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.transferCharacteristics = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("level") || IS_OPTION("level:h264") || IS_OPTION("level:hevc")) {
        const bool for_h264 = IS_OPTION("level") || IS_OPTION("level:h264");
        const bool for_hevc = IS_OPTION("level") || IS_OPTION("level:hevc");
        i++;
        auto getLevel = [](const CX_DESC *desc, const TCHAR *argvstr, int *levelValue) {
            int value = 0;
            bool bParsed = false;
            if (desc != nullptr) {
                if (PARSE_ERROR_FLAG != (value = get_value_from_chr(desc, argvstr))) {
                    *levelValue = value;
                    bParsed = true;
                } else {
                    double val_float = 0.0;
                    if (1 == _stscanf_s(argvstr, _T("%lf"), &val_float)) {
                        value = (int)(val_float * 10 + 0.5);
                        if (value == desc[get_cx_index(desc, value)].value) {
                            *levelValue = value;
                            bParsed = true;
                        } else {
                            value = (int)(val_float + 0.5);
                            if (value == desc[get_cx_index(desc, value)].value) {
                                *levelValue = value;
                                bParsed = true;
                            }
                        }
                    }
                }
            }
            return bParsed;
        };
        bool flag = false;
        int value = 0;
        if (for_h264 && getLevel(list_avc_level, strInput[i], &value)) {
            codecPrm[NV_ENC_H264].h264Config.level = value;
            flag = true;
        }
        if (for_hevc && getLevel(list_hevc_level, strInput[i], &value)) {
            codecPrm[NV_ENC_HEVC].hevcConfig.level = value;
            flag = true;
        }
        if (!flag) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("profile") || IS_OPTION("profile:h264") || IS_OPTION("profile:hevc")) {
        const bool for_h264 = IS_OPTION("profile") || IS_OPTION("profile:h264");
        const bool for_hevc = IS_OPTION("profile") || IS_OPTION("profile:hevc");
        i++;
        bool flag = false;
        if (for_h264) {
            GUID zero = { 0 };
            GUID result_guid = get_guid_from_name(strInput[i], h264_profile_names);
            if (0 != memcmp(&result_guid, &zero, sizeof(result_guid))) {
                pParams->encConfig.profileGUID = result_guid;
                pParams->yuv444 = memcmp(&pParams->encConfig.profileGUID, &NV_ENC_H264_PROFILE_HIGH_444_GUID, sizeof(result_guid)) == 0;
                flag = true;
            }
        }
        if (for_hevc) {
            int result = get_value_from_name(strInput[i], h265_profile_names);
            if (-1 != result) {
                //下位16bitを使用する
                uint16_t *ptr = (uint16_t *)&codecPrm[NV_ENC_HEVC].hevcConfig.tier;
                ptr[0] = (uint16_t)result;
                if (result == NV_ENC_PROFILE_HEVC_MAIN444) {
                    pParams->yuv444 = TRUE;
                }
                if (result == NV_ENC_PROFILE_HEVC_MAIN10) {
                    codecPrm[NV_ENC_HEVC].hevcConfig.pixelBitDepthMinus8 = 2;
                    pParams->yuv444 = FALSE;
                } else if (result == NV_ENC_PROFILE_HEVC_MAIN) {
                    codecPrm[NV_ENC_HEVC].hevcConfig.pixelBitDepthMinus8 = 0;
                    pParams->yuv444 = FALSE;
                }
                flag = true;
            }
        }
        if (!flag) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("chromaloc") || IS_OPTION("chromaloc:h264") || IS_OPTION("chromaloc:hevc")) {
        const bool for_h264 = IS_OPTION("chromaloc") || IS_OPTION("chromaloc:h264");
        const bool for_hevc = IS_OPTION("chromaloc") || IS_OPTION("chromaloc:hevc");
        i++;
        int value = 0;
        if (get_list_value(list_chromaloc, strInput[i], &value)) {
            if (for_h264) {
                codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.chromaSampleLocationFlag = value != 0;
                codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.chromaSampleLocationTop = value;
                codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.chromaSampleLocationBot = value;
            }
            if (for_hevc) {
                codecPrm[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.chromaSampleLocationFlag = value != 0;
                codecPrm[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.chromaSampleLocationTop = value;
                codecPrm[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.chromaSampleLocationBot = value;
            }
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("tier")) {
        i++;
        int value = 0;
        if (get_list_value(h265_tier_names, strInput[i], &value)) {
            //上位16bitを使用する
            uint16_t *ptr = (uint16_t *)&codecPrm[NV_ENC_HEVC].hevcConfig.tier;
            ptr[1] = (uint16_t)value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("max-cll")) {
        i++;
        pParams->sMaxCll = tchar_to_string(strInput[i]);
        return 0;
    }
    if (IS_OPTION("master-display")) {
        i++;
        pParams->sMasterDisplay = tchar_to_string(strInput[i]);
        return 0;
    }
    if (IS_OPTION("output-depth")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            codecPrm[NV_ENC_HEVC].hevcConfig.pixelBitDepthMinus8 = clamp(value - 8, 0, 4);
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("sar") || IS_OPTION("par") || IS_OPTION("dar")) {
        i++;
        int a[2] = { 0 };
        if (   2 == _stscanf_s(strInput[i], _T("%d:%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d/%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d.%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d,%d"), &a[0], &a[1])) {
            if (IS_OPTION("dar")) {
                a[0] = -a[0];
                a[1] = -a[1];
            }
            pParams->par[0] = a[0];
            pParams->par[1] = a[1];
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("cu-max")) {
        i++;
        int value = 0;
        if (get_list_value(list_hevc_cu_size, strInput[i], &value)) {
            codecPrm[NV_ENC_HEVC].hevcConfig.maxCUSize = (NV_ENC_HEVC_CUSIZE)value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("cu-min")) {
        i++;
        int value = 0;
        if (get_list_value(list_hevc_cu_size, strInput[i], &value)) {
            codecPrm[NV_ENC_HEVC].hevcConfig.minCUSize = (NV_ENC_HEVC_CUSIZE)value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("cuda-schedule")) {
        i++;
        int value = 0;
        if (get_list_value(list_cuda_schedule, strInput[i], &value)) {
            pParams->nCudaSchedule = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("max-procfps")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        if (value < 0) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return -0;
        }
        pParams->nProcSpeedLimit = (std::min)(value, INT_MAX);
        if (get_list_value(list_cuda_schedule, _T("sync"), &value)) {
            pParams->nCudaSchedule = value;
        }
        return 0;
    }
    if (IS_OPTION("log")) {
        i++;
        pParams->logfile = strInput[i];
        return 0;
    }
    if (IS_OPTION("log-level")) {
        i++;
        int value = 0;
        if (get_list_value(list_log_level, strInput[i], &value)) {
            pParams->loglevel = value;
        } else {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("log-framelist")) {
        i++;
        pParams->sFramePosListLog = strInput[i];
        return 0;
    }
    if (IS_OPTION("log-mux-ts")) {
        i++;
        pParams->pMuxVidTsLogFile = _tcsdup(strInput[i]);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("output-buf"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < 0) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nOutputBufSizeMB = (std::min)(value, RGY_OUTPUT_BUF_MB_MAX);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("input-thread"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < -1 || value >= 2) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nInputThread = (int8_t)value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("no-output-thread"))) {
        pParams->nOutputThread = 0;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("output-thread"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < -1 || value >= 2) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nOutputThread = value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-thread"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < -1 || value >= 3) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nAudioThread = value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("max-procfps"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < 0) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nProcSpeedLimit = (uint16_t)(std::min)(value, (int)UINT16_MAX);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("perf-monitor"))) {
        if (strInput[i+1][0] == _T('-') || _tcslen(strInput[i+1]) == 0) {
            pParams->nPerfMonitorSelect = (int)PERF_MONITOR_ALL;
        } else {
            i++;
            auto items = split(strInput[i], _T(","));
            for (const auto& item : items) {
                int value = 0;
                if (PARSE_ERROR_FLAG == (value = get_value_from_chr(list_pref_monitor, item.c_str()))) {
                    SET_ERR(strInput[0], _T("Unknown value"), option_name, item.c_str());
                    return 1;
                }
                pParams->nPerfMonitorSelect |= value;
            }
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("perf-monitor-interval"))) {
        i++;
        int v;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &v)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nPerfMonitorInterval = std::max(50, v);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("session-retry"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            SET_ERR(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < 0) {
            SET_ERR(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        pParams->sessionRetry = value;
        return 0;
    }
    tstring mes = _T("Unknown option: --");
    mes += option_name;
    SET_ERR(strInput[0], (TCHAR *)mes.c_str(), NULL, strInput[i]);
    return -1;
}
#undef IS_OPTION

int parse_cmd(InEncodeVideoParam *pParams, NV_ENC_CODEC_CONFIG *codecPrm, int nArgNum, const TCHAR **strInput, ParseCmdError& err, bool ignore_parse_err) {
    sArgsData argsData;

    for (int i = 1; i < nArgNum; i++) {
        if (strInput[i] == nullptr) {
            return -1;
        }
        const TCHAR *option_name = nullptr;
        if (strInput[i][0] == _T('-')) {
            if (strInput[i][1] == _T('-')) {
                option_name = &strInput[i][2];
            } else if (strInput[i][2] == _T('\0')) {
                if (nullptr == (option_name = cmd_short_opt_to_long(strInput[i][1]))) {
                    SET_ERR(strInput[0], strsprintf(_T("Unknown options: \"%s\""), strInput[i]).c_str(), nullptr, nullptr);
                    return -1;
                }
            } else {
                if (ignore_parse_err) continue;
                SET_ERR(strInput[0], strsprintf(_T("Invalid options: \"%s\""), strInput[i]).c_str(), nullptr, nullptr);
                return -1;
            }
        }

        if (nullptr == option_name) {
            if (ignore_parse_err) continue;
            SET_ERR(strInput[0], strsprintf(_T("Unknown option: \"%s\""), strInput[i]).c_str(), nullptr, nullptr);
            return -1;
        }
        auto sts = parse_one_option(option_name, strInput, i, nArgNum, pParams, codecPrm, &argsData, err);
        if (!ignore_parse_err && sts != 0) {
            return sts;
        }
    }

    //Bフレームの設定
    if (argsData.nBframes < 0) {
        //特に指定されていない場合はデフォルト値を反映する
        switch (pParams->codec) {
        case NV_ENC_H264:
            argsData.nBframes = DEFAULT_B_FRAMES_H264;
            break;
        case NV_ENC_HEVC:
            argsData.nBframes = DEFAULT_B_FRAMES_HEVC;
            break;
        default:
            SET_ERR(strInput[0], _T("Unknown Output codec.\n"), nullptr, nullptr);
            return -1;
            break;
        }
    }
    pParams->encConfig.frameIntervalP = argsData.nBframes + 1;

    return 0;
}

int parse_cmd(InEncodeVideoParam *pParams, NV_ENC_CODEC_CONFIG *codecPrm, const char *cmda, ParseCmdError& err, bool ignore_parse_err) {
    if (cmda == nullptr) {
        return 0;
    }
    std::wstring cmd = char_to_wstring(cmda);
    int argc = 0;
    auto argvw = CommandLineToArgvW(cmd.c_str(), &argc);
    if (argc <= 1) {
        return 0;
    }
    vector<tstring> argv_tstring;
    for (int i = 0; i < argc; i++) {
        argv_tstring.push_back(wstring_to_tstring(argvw[i]));
    }
    LocalFree(argvw);

    vector<TCHAR *> argv_tchar;
    for (int i = 0; i < argc; i++) {
        argv_tchar.push_back((TCHAR *)argv_tstring[i].data());
    }
    argv_tchar.push_back(_T(""));
    const TCHAR **strInput = (const TCHAR **)argv_tchar.data();
    int ret = parse_cmd(pParams, codecPrm, argc, strInput, err, ignore_parse_err);
    return ret;
}

#pragma warning (push)
#pragma warning (disable: 4127)
tstring gen_cmd(const InEncodeVideoParam *pParams, const NV_ENC_CODEC_CONFIG codecPrmArg[2], bool save_disabled_prm) {
    std::basic_stringstream<TCHAR> cmd;
    InEncodeVideoParam encPrmDefault;
    NV_ENC_CODEC_CONFIG codecPrmDefault[2];
    codecPrmDefault[NV_ENC_H264] = DefaultParamH264();
    codecPrmDefault[NV_ENC_HEVC] = DefaultParamHEVC();

    NV_ENC_CODEC_CONFIG codecPrm[2];
    if (codecPrmArg != nullptr) {
        memcpy(codecPrm, codecPrmArg, sizeof(codecPrm));
    } else {
        memcpy(codecPrm, codecPrmDefault, sizeof(codecPrm));
        codecPrm[pParams->codec] = pParams->encConfig.encodeCodecConfig;
    }

#define OPT_FLOAT(str, opt, prec) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << (str) << _T(" ") << std::setprecision(prec) << (pParams->opt);
#define OPT_NUM(str, opt) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << (str) << _T(" ") << (int)(pParams->opt);
#define OPT_NUM_HEVC(str, codec, opt) if ((codecPrm[NV_ENC_HEVC].hevcConfig.opt) != (codecPrmDefault[NV_ENC_HEVC].hevcConfig.opt)) cmd << _T(" ") << (str) << ((save_disabled_prm) ? codec : _T("")) << _T(" ") << (int)(codecPrm[NV_ENC_HEVC].hevcConfig.opt);
#define OPT_NUM_H264(str, codec, opt) if ((codecPrm[NV_ENC_H264].h264Config.opt) != (codecPrmDefault[NV_ENC_H264].h264Config.opt)) cmd << _T(" ") << (str) << ((save_disabled_prm) ? codec : _T("")) << _T(" ") << (int)(codecPrm[NV_ENC_H264].h264Config.opt);
#define OPT_GUID(str, opt, list) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << (str) << _T(" ") << get_name_from_guid((pParams->opt), list);
#define OPT_GUID_HEVC(str, codec, opt, list) if ((codecPrm[NV_ENC_HEVC].hevcConfig.opt) != (codecPrmDefault[NV_ENC_HEVC].hevcConfig.opt)) cmd << _T(" ") << (str) << ((save_disabled_prm) ? codec : _T("")) << _T(" ") << get_name_from_value((codecPrm[NV_ENC_HEVC].hevcConfig.opt), list);
#define OPT_LST(str, opt, list) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << (str) << _T(" ") << get_chr_from_value(list, (pParams->opt));
#define OPT_LST_HEVC(str, codec, opt, list) if ((codecPrm[NV_ENC_HEVC].hevcConfig.opt) != (codecPrmDefault[NV_ENC_HEVC].hevcConfig.opt)) cmd << _T(" ") << (str) << ((save_disabled_prm) ? codec : _T("")) << _T(" ") << get_chr_from_value(list, (codecPrm[NV_ENC_HEVC].hevcConfig.opt));
#define OPT_LST_H264(str, codec, opt, list) if ((codecPrm[NV_ENC_H264].h264Config.opt) != (codecPrmDefault[NV_ENC_H264].h264Config.opt)) cmd << _T(" ") << (str) << ((save_disabled_prm) ? codec : _T("")) << _T(" ") << get_chr_from_value(list, (codecPrm[NV_ENC_H264].h264Config.opt));
#define OPT_QP(str, qp, enable, force) { \
    if ((force) || (enable) \
    || (pParams->qp.qpIntra) != (encPrmDefault.qp.qpIntra) \
    || (pParams->qp.qpInterP) != (encPrmDefault.qp.qpInterP) \
    || (pParams->qp.qpInterB) != (encPrmDefault.qp.qpInterB)) { \
        if (enable) { \
            cmd << _T(" ") << (str) << _T(" "); \
        } else { \
            cmd << _T(" ") << (str) << _T(" 0;"); \
        } \
        if ((pParams->qp.qpIntra) == (pParams->qp.qpInterP) && (pParams->qp.qpIntra) == (pParams->qp.qpInterB)) { \
            cmd << (int)(pParams->qp.qpIntra); \
        } else { \
            cmd << (int)(pParams->qp.qpIntra) << _T(":") << (int)(pParams->qp.qpInterP) << _T(":") << (int)(pParams->qp.qpInterB); \
        } \
    } \
}
#define OPT_BOOL(str_true, str_false, opt) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << ((pParams->opt) ? (str_true) : (str_false));
#define OPT_BOOL_HEVC(str_true, str_false, codec, opt) \
    if ((codecPrm[NV_ENC_HEVC].hevcConfig.opt) != (codecPrmDefault[NV_ENC_HEVC].hevcConfig.opt)) { \
        cmd << _T(" "); \
        if ((codecPrm[NV_ENC_HEVC].hevcConfig.opt)) { \
            if (_tcslen(str_true)) { cmd << (str_true) << ((save_disabled_prm) ? (codec) : _T("")); } \
        } else { \
            if (_tcslen(str_false)) { cmd << (str_false) << ((save_disabled_prm) ? (codec) : _T("")); } \
        } \
    }
#define OPT_BOOL_H264(str_true, str_false, codec, opt) \
    if ((codecPrm[NV_ENC_H264].h264Config.opt) != (codecPrmDefault[NV_ENC_H264].h264Config.opt)) { \
        cmd << _T(" "); \
        if ((codecPrm[NV_ENC_H264].h264Config.opt)) { \
            if (_tcslen(str_true)) { cmd << (str_true) << ((save_disabled_prm) ? (codec) : _T("")); }\
        } else { \
            if (_tcslen(str_false)) { cmd << (str_false) << ((save_disabled_prm) ? (codec) : _T("")); }\
        } \
    }
#define OPT_CHAR(str, opt) if ((pParams->opt) && _tcslen(pParams->opt)) cmd << _T(" ") << str << _T(" ") << (pParams->opt);
#define OPT_STR(str, opt) if (pParams->opt.length() > 0) cmd << _T(" ") << str << _T(" ") << (pParams->opt.c_str());
#define OPT_CHAR_PATH(str, opt) if ((pParams->opt) && _tcslen(pParams->opt)) cmd << _T(" ") << str << _T(" \"") << (pParams->opt) << _T("\"");
#define OPT_STR_PATH(str, opt) if (pParams->opt.length() > 0) cmd << _T(" ") << str << _T(" \"") << (pParams->opt.c_str()) << _T("\"");

    OPT_NUM(_T("-d"), deviceID);
    cmd << _T(" -c ") << get_chr_from_value(list_nvenc_codecs_for_opt, pParams->codec);
    OPT_STR_PATH(_T("-i"), inputFilename);
    OPT_STR_PATH(_T("-o"), outputFilename);
    switch (pParams->input.type) {
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
    if (save_disabled_prm || pParams->input.picstruct != RGY_PICSTRUCT_FRAME) {
        OPT_LST(_T("--interlace"), input.picstruct, list_interlaced);
    }
    if (cropEnabled(pParams->input.crop)) {
        cmd << _T(" --crop ") << pParams->input.crop.e.left << _T(",") << pParams->input.crop.e.up
            << _T(",") << pParams->input.crop.e.right << _T(",") << pParams->input.crop.e.bottom;
    }
    if (pParams->input.fpsN * pParams->input.fpsD > 0) {
        cmd << _T(" --fps ") << pParams->input.fpsN << _T("/") << pParams->input.fpsD;
    }
    if (pParams->input.srcWidth * pParams->input.srcHeight > 0) {
        cmd << _T(" --input-res ") << pParams->input.srcWidth << _T("x") << pParams->input.srcHeight;
    }
    if (pParams->input.dstWidth * pParams->input.dstHeight > 0) {
        cmd << _T(" --output-res ") << pParams->input.dstWidth << _T("x") << pParams->input.dstHeight;
    }
    if (save_disabled_prm) {
        switch (pParams->encConfig.rcParams.rateControlMode) {
        case NV_ENC_PARAMS_RC_CBR:
        case NV_ENC_PARAMS_RC_CBR_HQ:
        case NV_ENC_PARAMS_RC_VBR:
        case NV_ENC_PARAMS_RC_VBR_HQ: {
            OPT_QP(_T("--cqp"), encConfig.rcParams.constQP, true, true);
        } break;
        case NV_ENC_PARAMS_RC_CONSTQP:
        default: {
            cmd << _T(" --vbr ") << pParams->encConfig.rcParams.averageBitRate / 1000;
        } break;
        }
    }
    switch (pParams->encConfig.rcParams.rateControlMode) {
    case NV_ENC_PARAMS_RC_CBR: {
        cmd << _T(" --cbr ") << pParams->encConfig.rcParams.averageBitRate / 1000;
    } break;
    case NV_ENC_PARAMS_RC_CBR_HQ: {
        cmd << _T(" --cbrhq ") << pParams->encConfig.rcParams.averageBitRate / 1000;
    } break;
    case NV_ENC_PARAMS_RC_VBR: {
        cmd << _T(" --vbr ") << pParams->encConfig.rcParams.averageBitRate / 1000;
    } break;
    case NV_ENC_PARAMS_RC_VBR_HQ: {
        cmd << _T(" --vbrhq ") << pParams->encConfig.rcParams.averageBitRate / 1000;
    } break;
    case NV_ENC_PARAMS_RC_CONSTQP:
    default: {
        OPT_QP(_T("--cqp"), encConfig.rcParams.constQP, true, true);
    } break;
    }
    if (pParams->encConfig.rcParams.rateControlMode != NV_ENC_PARAMS_RC_CONSTQP || save_disabled_prm) {
        OPT_NUM(_T("--vbv-bufsize"), encConfig.rcParams.vbvBufferSize / 1000);
        if ((pParams->encConfig.rcParams.targetQuality) != (encPrmDefault.encConfig.rcParams.targetQuality)
            || (pParams->encConfig.rcParams.targetQualityLSB) != (encPrmDefault.encConfig.rcParams.targetQualityLSB)) {
            float val = pParams->encConfig.rcParams.targetQuality + pParams->encConfig.rcParams.targetQualityLSB / 256.0f;
            cmd << _T(" --vbr-quality ") << std::fixed << std::setprecision(2) << val;
        }
        OPT_NUM(_T("--max-bitrate"), encConfig.rcParams.maxBitRate / 1000);
    }
    if (pParams->encConfig.rcParams.enableInitialRCQP || save_disabled_prm) {
        OPT_QP(_T("--qp-init"), encConfig.rcParams.initialRCQP, pParams->encConfig.rcParams.enableInitialRCQP, false);
    }
    if (pParams->encConfig.rcParams.enableMinQP || save_disabled_prm) {
        OPT_QP(_T("--qp-min"), encConfig.rcParams.minQP, pParams->encConfig.rcParams.enableMinQP, false);
    }
    if (pParams->encConfig.rcParams.enableMaxQP || save_disabled_prm) {
        OPT_QP(_T("--qp-max"), encConfig.rcParams.maxQP, pParams->encConfig.rcParams.enableMaxQP, false);
    }
    if (pParams->encConfig.rcParams.enableLookahead || save_disabled_prm) {
        OPT_NUM(_T("--lookahead"), encConfig.rcParams.lookaheadDepth);
    }
    OPT_BOOL(_T("--no-i-adapt"), _T(""), encConfig.rcParams.disableIadapt);
    OPT_BOOL(_T("--no-b-adapt"), _T(""), encConfig.rcParams.disableBadapt);
    OPT_BOOL(_T("--strict-gop"), _T(""), encConfig.rcParams.strictGOPTarget);
    if (pParams->encConfig.gopLength == 0) {
        cmd << _T(" --gop-len auto");
    } else {
        OPT_NUM(_T("--gop-len"), encConfig.gopLength);
    }
    OPT_NUM(_T("-b"), encConfig.frameIntervalP-1);
    OPT_BOOL(_T("--weightp"), _T(""), nWeightP);
    OPT_BOOL(_T("--aq"), _T("--no-aq"), encConfig.rcParams.enableAQ);
    OPT_BOOL(_T("--aq-temporal"), _T(""), encConfig.rcParams.enableTemporalAQ);
    OPT_NUM(_T("--aq-strength"), encConfig.rcParams.aqStrength);
    OPT_LST(_T("--mv-precision"), encConfig.mvPrecision, list_mv_presicion);
    if (pParams->par[0] > 0 && pParams->par[1] > 0) {
        cmd << _T(" --sar ") << pParams->par[0] << _T(":") << pParams->par[1];
    } else if (pParams->par[0] < 0 && pParams->par[1] < 0) {
        cmd << _T(" --dar ") << -1 * pParams->par[0] << _T(":") << -1 * pParams->par[1];
    }
    OPT_BOOL(_T("--lossless"), _T(""), lossless);

    if (pParams->codec == NV_ENC_HEVC || save_disabled_prm) {
        OPT_LST_HEVC(_T("--level"), _T(":hevc"), level, list_hevc_level);
        OPT_GUID_HEVC(_T("--profile"), _T(":hevc"), tier & 0xffff, h265_profile_names);
        OPT_LST_HEVC(_T("--tier"), _T(":hevc"), tier >> 16, h265_tier_names);
        OPT_NUM_HEVC(_T("--ref"), _T(""), maxNumRefFramesInDPB);
        if (codecPrm[NV_ENC_HEVC].hevcConfig.pixelBitDepthMinus8 != codecPrmDefault[NV_ENC_HEVC].hevcConfig.pixelBitDepthMinus8) {
            cmd << _T(" --output-depth ") << codecPrm[NV_ENC_HEVC].hevcConfig.pixelBitDepthMinus8 + 8;
        }
        OPT_NUM_HEVC(_T("--slices"), _T(":hevc"), sliceModeData);
        OPT_BOOL_HEVC(_T("--aud"), _T(""), _T(":hevc"), outputAUD);
        OPT_BOOL_HEVC(_T("--pic-struct"), _T(""), _T(":hevc"), outputPictureTimingSEI);
        OPT_BOOL_HEVC(_T("--fullrange"), _T(""), _T(":hevc"), hevcVUIParameters.videoFullRangeFlag);
        OPT_LST_HEVC(_T("--videoformat"), _T(":hevc"), hevcVUIParameters.videoFormat, list_videoformat);
        OPT_LST_HEVC(_T("--colormatrix"), _T(":hevc"), hevcVUIParameters.colourMatrix, list_colormatrix);
        OPT_LST_HEVC(_T("--colorprim"), _T(":hevc"), hevcVUIParameters.colourPrimaries, list_colorprim);
        OPT_LST_HEVC(_T("--chromaloc"), _T(":hevc"), hevcVUIParameters.chromaSampleLocationTop, list_chromaloc);
        OPT_LST_HEVC(_T("--transfer"), _T(":hevc"), hevcVUIParameters.transferCharacteristics, list_transfer);
        OPT_STR(_T("--max-cll"), sMaxCll);
        OPT_STR(_T("--master-display"), sMasterDisplay);
        OPT_LST_HEVC(_T("--cu-max"), _T(""), maxCUSize, list_hevc_cu_size);
        OPT_LST_HEVC(_T("--cu-min"), _T(""), minCUSize, list_hevc_cu_size);
    }
    if (pParams->codec == NV_ENC_H264 || save_disabled_prm) {
        OPT_LST_H264(_T("--level"), _T(":h264"), level, list_avc_level);
        OPT_GUID(_T("--profile"), encConfig.profileGUID, h264_profile_names);
        OPT_NUM_H264(_T("--ref"), _T(""), maxNumRefFrames);
        OPT_LST_H264(_T("--bref-mode"), _T(""), useBFramesAsRef, list_bref_mode);
        OPT_LST_H264(_T("--direct"), _T(""), bdirectMode, list_bdirect);
        OPT_LST_H264(_T("--adapt-transform"), _T(""), adaptiveTransformMode, list_adapt_transform);
        OPT_NUM_H264(_T("--slices"), _T(":h264"), sliceModeData);
        OPT_BOOL_H264(_T("--aud"), _T(""), _T(":h264"), outputAUD);
        OPT_BOOL_H264(_T("--pic-struct"), _T(""), _T(":h264"), outputPictureTimingSEI);
        OPT_BOOL_H264(_T("--fullrange"), _T(""), _T(":h264"), h264VUIParameters.videoFullRangeFlag);
        OPT_LST_H264(_T("--videoformat"), _T(":h264"), h264VUIParameters.videoFormat, list_videoformat);
        OPT_LST_H264(_T("--colormatrix"), _T(":h264"), h264VUIParameters.colourMatrix, list_colormatrix);
        OPT_LST_H264(_T("--colorprim"), _T(":h264"), h264VUIParameters.colourPrimaries, list_colorprim);
        OPT_LST_H264(_T("--chromaloc"), _T(":h264"), h264VUIParameters.chromaSampleLocationTop, list_chromaloc);
        OPT_LST_H264(_T("--transfer"), _T(":h264"), h264VUIParameters.transferCharacteristics, list_transfer);
        if ((codecPrm[NV_ENC_H264].h264Config.entropyCodingMode) != (codecPrmDefault[NV_ENC_H264].h264Config.entropyCodingMode)) {
            cmd << _T(" --") << get_chr_from_value(list_entropy_coding, codecPrm[NV_ENC_H264].h264Config.entropyCodingMode);
        }
        OPT_BOOL(_T("--bluray"), _T(""), bluray);
        OPT_BOOL_H264(_T("--no-deblock"), _T("--deblock"), _T(""), disableDeblockingFilterIDC);
    }

    std::basic_stringstream<TCHAR> tmp;
#if ENABLE_AVSW_READER
    OPT_NUM(_T("--input-analyze"), nAVDemuxAnalyzeSec);
    if (pParams->nTrimCount > 0) {
        cmd << _T(" --trim ");
        for (int i = 0; i < pParams->nTrimCount; i++) {
            if (i > 0) cmd << _T(",");
            cmd << pParams->pTrimList[i].start << _T(":") << pParams->pTrimList[i].fin;
        }
    }
    OPT_FLOAT(_T("--seek"), fSeekSec, 2);
    OPT_CHAR(_T("--input-format"), pAVInputFormat);
    OPT_STR(_T("--output-format"), sAVMuxOutputFormat);
    OPT_NUM(_T("--video-track"), nVideoTrack);
    OPT_NUM(_T("--video-streamid"), nVideoStreamId);
    if (pParams->pMuxOpt) {
        for (uint32_t i = 0; i < pParams->pMuxOpt->size(); i++) {
            cmd << _T(" -m ") << pParams->pMuxOpt->at(i).first << _T(":") << pParams->pMuxOpt->at(i).second;
        }
    }
    tmp.str(tstring());
    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        const sAudioSelect *pAudioSelect = pParams->ppAudioSelectList[i];
        if (_tcscmp(pAudioSelect->pAVAudioEncodeCodec, RGY_AVCODEC_COPY) == 0) {
            if (pAudioSelect->nAudioSelect == 0) {
                tmp << _T(","); // --audio-copy のみの指定 (トラックIDを省略)
            } else {
                tmp << _T(",") << pAudioSelect->nAudioSelect;
            }
        }
    }
    if (!tmp.str().empty()) {
        cmd << _T(" --audio-copy ") << tmp.str().substr(1);
    }
    tmp.str(tstring());

    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        const sAudioSelect *pAudioSelect = pParams->ppAudioSelectList[i];
        if (_tcscmp(pAudioSelect->pAVAudioEncodeCodec, RGY_AVCODEC_COPY) != 0) {
            cmd << _T(" --audio-codec ") << pAudioSelect->nAudioSelect;
            if (_tcscmp(pAudioSelect->pAVAudioEncodeCodec, RGY_AVCODEC_AUTO) != 0) {
                cmd << _T("?") << pAudioSelect->pAVAudioEncodeCodec;
            }
            if (pAudioSelect->pAVAudioEncodeCodecPrm) {
                cmd << _T(":") << pAudioSelect->pAVAudioEncodeCodecPrm;
            }
        }
    }

    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        const sAudioSelect *pAudioSelect = pParams->ppAudioSelectList[i];
        if (_tcscmp(pAudioSelect->pAVAudioEncodeCodec, RGY_AVCODEC_COPY) != 0
            && pAudioSelect->pAVAudioEncodeCodecProfile != nullptr) {
            cmd << _T(" --audio-profile ") << pAudioSelect->nAudioSelect << _T("?") << pAudioSelect->pAVAudioEncodeCodecProfile;
        }
    }

    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        const sAudioSelect *pAudioSelect = pParams->ppAudioSelectList[i];
        if (_tcscmp(pAudioSelect->pAVAudioEncodeCodec, RGY_AVCODEC_COPY) != 0) {
            cmd << _T(" --audio-bitrate ") << pAudioSelect->nAudioSelect << _T("?") << pAudioSelect->nAVAudioEncodeBitrate;
        }
    }

    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        tmp.str(tstring());
        const sAudioSelect *pAudioSelect = pParams->ppAudioSelectList[i];
        for (int j = 0; j < MAX_SPLIT_CHANNELS; j++) {
            if (pAudioSelect->pnStreamChannelSelect[j] == 0) {
                break;
            }
            if (j > 0) cmd << _T(",");
            if (pAudioSelect->pnStreamChannelSelect[j] != RGY_CHANNEL_AUTO) {
                char buf[256];
                av_get_channel_layout_string(buf, _countof(buf), 0, pAudioSelect->pnStreamChannelOut[j]);
                cmd << char_to_tstring(buf);
            }
            if (pAudioSelect->pnStreamChannelOut[j] != RGY_CHANNEL_AUTO) {
                cmd << _T(":");
                char buf[256];
                av_get_channel_layout_string(buf, _countof(buf), 0, pAudioSelect->pnStreamChannelOut[j]);
                cmd << char_to_tstring(buf);
            }
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --audio-stream ") << pAudioSelect->nAudioSelect << _T("?") << tmp.str();
        }
    }
    tmp.str(tstring());

    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        const sAudioSelect *pAudioSelect = pParams->ppAudioSelectList[i];
        if (_tcscmp(pAudioSelect->pAVAudioEncodeCodec, RGY_AVCODEC_COPY) != 0) {
            cmd << _T(" --audio-samplerate ") << pAudioSelect->nAudioSelect << _T("?") << pAudioSelect->nAudioSamplingRate;
        }
    }
    OPT_LST(_T("--audio-resampler"), nAudioResampler, list_resampler);

    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        const sAudioSelect *pAudioSelect = pParams->ppAudioSelectList[i];
        if (_tcscmp(pAudioSelect->pAVAudioEncodeCodec, RGY_AVCODEC_COPY) != 0) {
            cmd << _T(" --audio-filter ") << pAudioSelect->nAudioSelect << _T("?") << pAudioSelect->pAudioFilter;
        }
    }
    for (int i = 0; i < pParams->nAudioSelectCount; i++) {
        const sAudioSelect *pAudioSelect = pParams->ppAudioSelectList[i];
        if (pAudioSelect->pAudioExtractFilename) {
            cmd << _T(" --audio-file ") << pAudioSelect->nAudioSelect << _T("?");
            if (pAudioSelect->pAudioExtractFormat) {
                cmd << pAudioSelect->pAudioExtractFormat << _T(":");
            }
            cmd << _T("\"") << pAudioSelect->pAudioExtractFilename << _T("\"");
        }
    }
    for (int i = 0; i < pParams->nAudioSourceCount; i++) {
        cmd << _T(" --audio-source ") << _T("\"") << pParams->ppAudioSourceList[i] << _T("\"");
    }
    OPT_NUM(_T("--audio-ignore-decode-error"), nAudioIgnoreDecodeError);
    if (pParams->pMuxOpt) {
        for (uint32_t i = 0; i < pParams->pMuxOpt->size(); i++) {
            cmd << _T(" -m ") << (*pParams->pMuxOpt)[i].first << _T(":") << (*pParams->pMuxOpt)[i].second;
        }
    }

    tmp.str(tstring());
    for (int i = 0; i < pParams->nSubtitleSelectCount; i++) {
        tmp << _T(",") << pParams->pSubtitleSelect[i];
    }
    if (!tmp.str().empty()) {
        cmd << _T(" --sub-copy ") << tmp.str().substr(1);
    }
    tmp.str(tstring());
    OPT_LST(_T("--caption2ass"), caption2ass, list_caption2ass);
    OPT_STR_PATH(_T("--chapter"), sChapterFile);
    OPT_BOOL(_T("--chapter-copy"), _T(""), bCopyChapter);
    //OPT_BOOL(_T("--chapter-no-trim"), _T(""), bChapterNoTrim);
    OPT_BOOL(_T("--key-on-chapter"), _T(""), keyOnChapter);
    OPT_STR_PATH(_T("--keyfile"), keyFile);
    OPT_LST(_T("--avsync"), nAVSyncMode, list_avsync);
#endif //#if ENABLE_AVSW_READER

    OPT_LST(_T("--vpp-deinterlace"), vpp.deinterlace, list_deinterlace);
    OPT_BOOL(_T("--vpp-rff"), _T(""), vpp.rff);
    OPT_LST(_T("--vpp-resize"), vpp.resizeInterp, list_nppi_resize);

#define ADD_FLOAT(str, opt, prec) if ((pParams->opt) != (encPrmDefault.opt)) tmp << _T(",") << (str) << _T("=") << std::setprecision(prec) << (pParams->opt);
#define ADD_NUM(str, opt) if ((pParams->opt) != (encPrmDefault.opt)) tmp << _T(",") << (str) << _T("=") << (pParams->opt);
#define ADD_LST(str, opt, list) if ((pParams->opt) != (encPrmDefault.opt)) tmp << _T(",") << (str) << _T("=") << get_chr_from_value(list, (pParams->opt));
#define ADD_BOOL(str, opt) if ((pParams->opt) != (encPrmDefault.opt)) tmp << _T(",") << (str) << _T("=") << ((pParams->opt) ? (_T("true")) : (_T("false")));
#define ADD_CHAR(str, opt) if ((pParams->opt) && _tcslen(pParams->opt)) tmp << _T(",") << (str) << _T("=") << (pParams->opt);
#define ADD_PATH(str, opt) if ((pParams->opt) && _tcslen(pParams->opt)) tmp << _T(",") << (str) << _T("=\"") << (pParams->opt) << _T("\"");
#define ADD_STR(str, opt) if (pParams->opt.length() > 0) tmp << _T(",") << (str) << _T("=") << (pParams->opt.c_str());

    if (pParams->vpp.afs != encPrmDefault.vpp.afs) {
        tmp.str(tstring());
        if (!pParams->vpp.afs.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (pParams->vpp.afs.enable || save_disabled_prm) {
            ADD_NUM(_T("top"), vpp.afs.clip.top);
            ADD_NUM(_T("bottom"), vpp.afs.clip.bottom);
            ADD_NUM(_T("left"), vpp.afs.clip.left);
            ADD_NUM(_T("right"), vpp.afs.clip.right);
            ADD_NUM(_T("method_switch"), vpp.afs.method_switch);
            ADD_NUM(_T("coeff_shift"), vpp.afs.coeff_shift);
            ADD_NUM(_T("thre_shift"), vpp.afs.thre_shift);
            ADD_NUM(_T("thre_deint"), vpp.afs.thre_deint);
            ADD_NUM(_T("thre_motion_y"), vpp.afs.thre_Ymotion);
            ADD_NUM(_T("thre_motion_c"), vpp.afs.thre_Cmotion);
            ADD_NUM(_T("level"), vpp.afs.analyze);
            ADD_BOOL(_T("shift"), vpp.afs.shift);
            ADD_BOOL(_T("drop"), vpp.afs.drop);
            ADD_BOOL(_T("smooth"), vpp.afs.smooth);
            ADD_BOOL(_T("24fps"), vpp.afs.force24);
            ADD_BOOL(_T("tune"), vpp.afs.tune);
            ADD_BOOL(_T("rff"), vpp.afs.rff);
            ADD_BOOL(_T("timecode"), vpp.afs.timecode);
            ADD_BOOL(_T("log"), vpp.afs.log);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-afs ") << tmp.str().substr(1);
        } else if (pParams->vpp.afs.enable) {
            cmd << _T(" --vpp-afs");
        }
    }
    if (pParams->vpp.knn != encPrmDefault.vpp.knn) {
        tmp.str(tstring());
        if (!pParams->vpp.knn.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (pParams->vpp.knn.enable || save_disabled_prm) {
            ADD_NUM(_T("radius"), vpp.knn.radius);
            ADD_FLOAT(_T("strength"), vpp.knn.strength, 3);
            ADD_FLOAT(_T("lerp"), vpp.knn.lerpC, 3);
            ADD_FLOAT(_T("th_weight"), vpp.knn.weight_threshold, 3);
            ADD_FLOAT(_T("th_lerp"), vpp.knn.lerp_threshold, 3);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-knn ") << tmp.str().substr(1);
        } else if (pParams->vpp.knn.enable) {
            cmd << _T(" --vpp-knn");
        }
    }
    if (pParams->vpp.pmd != encPrmDefault.vpp.pmd) {
        tmp.str(tstring());
        if (!pParams->vpp.pmd.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (pParams->vpp.pmd.enable || save_disabled_prm) {
            ADD_NUM(_T("apply_count"), vpp.pmd.applyCount);
            ADD_FLOAT(_T("strength"), vpp.pmd.strength, 3);
            ADD_FLOAT(_T("threshold"), vpp.pmd.threshold, 3);
            ADD_NUM(_T("useexp"), vpp.pmd.useExp);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-pmd ") << tmp.str().substr(1);
        } else if (pParams->vpp.pmd.enable) {
            cmd << _T(" --vpp-pmd");
        }
    }
    OPT_LST(_T("--vpp-gauss"), vpp.gaussMaskSize, list_nppi_gauss);
    if (pParams->vpp.unsharp != encPrmDefault.vpp.unsharp) {
        tmp.str(tstring());
        if (!pParams->vpp.unsharp.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (pParams->vpp.unsharp.enable || save_disabled_prm) {
            ADD_NUM(_T("radius"), vpp.unsharp.radius);
            ADD_FLOAT(_T("weight"), vpp.unsharp.weight, 3);
            ADD_FLOAT(_T("threshold"), vpp.unsharp.threshold, 3);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-unsharp ") << tmp.str().substr(1);
        } else if (pParams->vpp.unsharp.enable) {
            cmd << _T(" --vpp-unsharp");
        }
    }
    if (pParams->vpp.edgelevel != encPrmDefault.vpp.edgelevel) {
        tmp.str(tstring());
        if (!pParams->vpp.edgelevel.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (pParams->vpp.edgelevel.enable || save_disabled_prm) {
            ADD_FLOAT(_T("strength"), vpp.edgelevel.strength, 3);
            ADD_FLOAT(_T("threshold"), vpp.edgelevel.threshold, 3);
            ADD_FLOAT(_T("black"), vpp.edgelevel.black, 3);
            ADD_FLOAT(_T("white"), vpp.edgelevel.white, 3);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-edgelevel ") << tmp.str().substr(1);
        } else if (pParams->vpp.edgelevel.enable) {
            cmd << _T(" --vpp-edgelevel");
        }
    }
    if (pParams->vpp.tweak != encPrmDefault.vpp.tweak) {
        tmp.str(tstring());
        if (!pParams->vpp.tweak.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (pParams->vpp.tweak.enable || save_disabled_prm) {
            ADD_FLOAT(_T("brightness"), vpp.tweak.brightness, 3);
            ADD_FLOAT(_T("contrast"), vpp.tweak.contrast, 3);
            ADD_FLOAT(_T("gamma"), vpp.tweak.gamma, 3);
            ADD_FLOAT(_T("saturation"), vpp.tweak.saturation, 3);
            ADD_FLOAT(_T("hue"), vpp.tweak.hue, 3);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-tweak ") << tmp.str().substr(1);
        } else if (pParams->vpp.tweak.enable) {
            cmd << _T(" --vpp-tweak");
        }
    }
    if (pParams->vpp.pad != encPrmDefault.vpp.pad) {
        tmp.str(tstring());
        if (!pParams->vpp.pad.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (pParams->vpp.pad.enable || save_disabled_prm) {
            ADD_NUM(_T("r"), vpp.pad.right);
            ADD_NUM(_T("l"), vpp.pad.left);
            ADD_NUM(_T("t"), vpp.pad.top);
            ADD_NUM(_T("b"), vpp.pad.bottom);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-pad ") << tmp.str().substr(1);
        } else if (pParams->vpp.pad.enable) {
            cmd << _T(" --vpp-pad");
        }
    }
    if (pParams->vpp.deband != encPrmDefault.vpp.deband) {
        tmp.str(tstring());
        if (!pParams->vpp.deband.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (pParams->vpp.deband.enable || save_disabled_prm) {
            ADD_NUM(_T("range"), vpp.deband.range);
            if (pParams->vpp.deband.threY == pParams->vpp.deband.threCb
                && pParams->vpp.deband.threY == pParams->vpp.deband.threCr) {
                ADD_NUM(_T("thre"), vpp.deband.threY);
            } else {
                ADD_NUM(_T("thre_y"), vpp.deband.threY);
                ADD_NUM(_T("thre_cb"), vpp.deband.threCb);
                ADD_NUM(_T("thre_cr"), vpp.deband.threCr);
            }
            if (pParams->vpp.deband.ditherY == pParams->vpp.deband.ditherC) {
                ADD_NUM(_T("dither"), vpp.deband.ditherY);
            } else {
                ADD_NUM(_T("dither_y"), vpp.deband.ditherY);
                ADD_NUM(_T("dither_c"), vpp.deband.ditherC);
            }
            ADD_NUM(_T("sample"), vpp.deband.sample);
            ADD_BOOL(_T("blurfirst"), vpp.deband.blurFirst);
            ADD_BOOL(_T("rand_each_frame"), vpp.deband.randEachFrame);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-deband ") << tmp.str().substr(1);
        } else if (pParams->vpp.deband.enable) {
            cmd << _T(" --vpp-deband");
        }
    }
    if (pParams->vpp.delogo != encPrmDefault.vpp.delogo) {
        tmp.str(tstring());
        if (!pParams->vpp.delogo.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (pParams->vpp.delogo.enable || save_disabled_prm) {
            ADD_PATH(_T("file"), vpp.delogo.logoFilePath.c_str());
            ADD_PATH(_T("select"), vpp.delogo.logoSelect.c_str());
            if (pParams->vpp.delogo.posX != encPrmDefault.vpp.delogo.posX
                || pParams->vpp.delogo.posY != encPrmDefault.vpp.delogo.posY) {
                tmp << _T(",pos=") << pParams->vpp.delogo.posX << _T("x") << pParams->vpp.delogo.posY;
            }
            ADD_NUM(_T("depth"), vpp.delogo.depth);
            ADD_NUM(_T("y"),  vpp.delogo.Y);
            ADD_NUM(_T("cb"), vpp.delogo.Cb);
            ADD_NUM(_T("cr"), vpp.delogo.Cr);
            ADD_BOOL(_T("add"), vpp.delogo.mode);
            ADD_BOOL(_T("auto_fade"), vpp.delogo.autoFade);
            ADD_BOOL(_T("auto_nr"), vpp.delogo.autoNR);
            ADD_NUM(_T("nr_area"), vpp.delogo.NRArea);
            ADD_NUM(_T("nr_value"), vpp.delogo.NRValue);
            ADD_BOOL(_T("log"), vpp.delogo.log);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-delogo ") << tmp.str().substr(1);
        }
    }
    if (pParams->vpp.selectevery != encPrmDefault.vpp.selectevery) {
        tmp.str(tstring());
        if (!pParams->vpp.selectevery.enable && save_disabled_prm) {
            tmp << _T(",enable=false");
        }
        if (pParams->vpp.selectevery.enable || save_disabled_prm) {
            ADD_NUM(_T("step"), vpp.selectevery.step);
            ADD_NUM(_T("offset"), vpp.selectevery.offset);
        }
        if (!tmp.str().empty()) {
            cmd << _T(" --vpp-select-every ") << tmp.str().substr(1);
        }
    }
    OPT_BOOL(_T("--vpp-perf-monitor"), _T("--no-vpp-perf-monitor"), vpp.bCheckPerformance);

    OPT_LST(_T("--cuda-schedule"), nCudaSchedule, list_cuda_schedule);
    OPT_NUM(_T("--output-buf"), nOutputBufSizeMB);
    OPT_NUM(_T("--output-thread"), nOutputThread);
    OPT_NUM(_T("--input-thread"), nInputThread);
    OPT_NUM(_T("--audio-thread"), nAudioThread);
    OPT_NUM(_T("--max-procfps"), nProcSpeedLimit);
    OPT_STR_PATH(_T("--log"), logfile);
    OPT_LST(_T("--log-level"), loglevel, list_log_level);
    OPT_STR_PATH(_T("--log-framelist"), sFramePosListLog);
    OPT_CHAR_PATH(_T("--log-mux-ts"), pMuxVidTsLogFile);
    if (pParams->nPerfMonitorSelect != encPrmDefault.nPerfMonitorSelect) {
        auto select = (int)pParams->nPerfMonitorSelect;
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
    OPT_NUM(_T("--perf-monitor-interval"), nPerfMonitorInterval);
    OPT_NUM(_T("--session-retry"), sessionRetry);
    return cmd.str();
}
#pragma warning (pop)

#undef SET_ERR