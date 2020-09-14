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


#include "rgy_osdep.h"
#include "rgy_tchar.h"
#include <locale.h>
#include <signal.h>
#include <fcntl.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <set>
#include <cstdio>
#include "rgy_version.h"
#include "rgy_util.h"
#include "NVEncDevice.h"
#include "NVEncParam.h"
#include "NVEncUtil.h"
#include "NVEncFilterAfs.h"
#include "NVEncCmd.h"
#include "NVEncCore.h"

static void show_version() {
    _ftprintf(stdout, _T("%s"), GetNVEncVersion().c_str());
}

static void show_help() {
    _ftprintf(stdout, _T("%s\n"), encoder_help().c_str());
}

static void show_device_list() {
    if (!check_if_nvcuda_dll_available()) {
        _ftprintf(stdout, _T("CUDA not available.\n"));
        return;
    }

    InEncodeVideoParam encPrm;
    encPrm.deviceID = -1;
    encPrm.ctrl.loglevel = RGY_LOG_INFO;

    NVEncCore nvEnc;
    if (NV_ENC_SUCCESS == nvEnc.Initialize(&encPrm)
        && NV_ENC_SUCCESS == nvEnc.ShowDeviceList(&encPrm)) {
        return;
    }
}

static int show_hw(int deviceid) {
    show_version();

    InEncodeVideoParam encPrm;
    encPrm.deviceID = deviceid;
    encPrm.ctrl.loglevel = RGY_LOG_DEBUG;

    NVEncCore nvEnc;
    if (NV_ENC_SUCCESS == nvEnc.Initialize(&encPrm)
        && NV_ENC_SUCCESS == nvEnc.ShowCodecSupport(&encPrm)) {
        return 0;
    }
    return 1;
}

static void show_environment_info() {
    show_version();
    _ftprintf(stdout, _T("%s\n"), getEnviromentInfo(false).c_str());
}

static int show_nvenc_features(int deviceid) {
    show_version();
    _ftprintf(stdout, _T("\n%s\n"), getEnviromentInfo(false).c_str());

    InEncodeVideoParam encPrm;
    encPrm.deviceID = deviceid;
    encPrm.ctrl.loglevel = RGY_LOG_INFO;

    NVEncCore nvEnc;
    if (NV_ENC_SUCCESS == nvEnc.Initialize(&encPrm)
        && NV_ENC_SUCCESS == nvEnc.ShowNVEncFeatures(&encPrm)) {
        return 0;
    }
    return 1;
}

static void show_option_list() {
    show_version();

    std::vector<std::string> optList;
    for (const auto &optHelp : createOptionList()) {
        optList.push_back(optHelp.first);
    }
    std::sort(optList.begin(), optList.end());

    _ftprintf(stdout, _T("Option List:\n"));
    for (const auto &optHelp : optList) {
        _ftprintf(stdout, _T("--%s\n"), char_to_tstring(optHelp).c_str());
    }
}

int parse_print_options(const TCHAR *option_name, const TCHAR *arg1) {

#define IS_OPTION(x) (0 == _tcscmp(option_name, _T(x)))

    if (IS_OPTION("help")) {
        show_version();
        show_help();
        return 1;
    }
    if (IS_OPTION("version")) {
        show_version();
        return 1;
    }
    if (IS_OPTION("option-list")) {
        show_option_list();
        return 1;
    }
    if (IS_OPTION("check-device")) {
        show_device_list();
        return 1;
    }
    if (IS_OPTION("check-hw")) {
        int deviceid = 0;
        if (arg1 && arg1[0] != '-') {
            int value = 0;
            if (1 == _stscanf_s(arg1, _T("%d"), &value)) {
                deviceid = value;
            }
        }
        return show_hw(deviceid) == 0 ? 1 : -1;
    }
    if (IS_OPTION("check-environment")) {
        show_environment_info();
        return 1;
    }
    if (IS_OPTION("check-features")) {
        int deviceid = 0;
        if (arg1 && arg1[0] != '-') {
            int value = 0;
            if (1 == _stscanf_s(arg1, _T("%d"), &value)) {
                deviceid = value;
            }
        }
        return show_nvenc_features(deviceid) == 0 ? 1 : -1;
    }
#if ENABLE_AVSW_READER
    if (0 == _tcscmp(option_name, _T("check-avversion"))) {
        _ftprintf(stdout, _T("%s\n"), getAVVersions().c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-codecs"))) {
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC | RGY_AVCODEC_ENC)).c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-encoders"))) {
        _ftprintf(stdout, _T("%s\n"), getAVCodecs(RGY_AVCODEC_ENC).c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-decoders"))) {
        _ftprintf(stdout, _T("%s\n"), getAVCodecs(RGY_AVCODEC_DEC).c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-profiles"))) {
        auto list = getAudioPofileList(arg1);
        if (list.size() == 0) {
            _ftprintf(stdout, _T("Failed to find codec name \"%s\"\n"), arg1);
        } else {
            _ftprintf(stdout, _T("profile name for \"%s\"\n"), arg1);
            for (const auto& name : list) {
                _ftprintf(stdout, _T("  %s\n"), name.c_str());
            }
        }
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-protocols"))) {
        _ftprintf(stdout, _T("%s\n"), getAVProtocols().c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-formats"))) {
        _ftprintf(stdout, _T("%s\n"), getAVFormats((RGYAVFormatType)(RGY_AVFORMAT_DEMUX | RGY_AVFORMAT_MUX)).c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-filters"))) {
        _ftprintf(stdout, _T("%s\n"), getAVFilters().c_str());
        return 1;
    }
#endif //#if ENABLE_AVSW_READER
#undef IS_OPTION
    return 0;
}

#if defined(_WIN32) || defined(_WIN64)
bool check_locale_is_ja() {
    const WORD LangID_ja_JP = MAKELANGID(LANG_JAPANESE, SUBLANG_JAPANESE_JAPAN);
    return GetUserDefaultLangID() == LangID_ja_JP;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

//Ctrl + C ハンドラ
static bool g_signal_abort = false;
#pragma warning(push)
#pragma warning(disable:4100)
static void sigcatch(int sig) {
    g_signal_abort = true;
}
#pragma warning(pop)
static int set_signal_handler() {
    int ret = 0;
    if (SIG_ERR == signal(SIGINT, sigcatch)) {
        _ftprintf(stderr, _T("failed to set signal handler.\n"));
    }
    return ret;
}

int _tmain(int argc, TCHAR **argv) {
#if defined(_WIN32) || defined(_WIN64)
    if (check_locale_is_ja()) {
        _tsetlocale(LC_ALL, _T("Japanese"));
    }
#endif //#if defined(_WIN32) || defined(_WIN64)

    if (argc == 1) {
        show_version();
        show_help();
        return 1;
    }

    for (int iarg = 1; iarg < argc; iarg++) {
        const TCHAR *option_name = nullptr;
        if (argv[iarg][0] == _T('-')) {
            if (argv[iarg][1] == _T('\0')) {
                continue;
            } else if (argv[iarg][1] == _T('-')) {
                option_name = &argv[iarg][2];
            } else if (argv[iarg][2] == _T('\0')) {
                if (nullptr == (option_name = cmd_short_opt_to_long(argv[iarg][1]))) {
                    continue;
                }
            }
        }
        if (option_name != nullptr) {
            int ret = parse_print_options(option_name, (iarg+1 < argc) ? argv[iarg+1] : _T(""));
            if (ret != 0) {
                return ret == 1 ? 0 : 1;
            }
        }
    }

    InEncodeVideoParam encPrm;
    NV_ENC_CODEC_CONFIG codecPrm[2] = { 0 };
    codecPrm[NV_ENC_H264] = DefaultParamH264();
    codecPrm[NV_ENC_HEVC] = DefaultParamHEVC();

    vector<const TCHAR *> argvCopy(argv, argv + argc);
    argvCopy.push_back(_T(""));
    if (parse_cmd(&encPrm, codecPrm, argc, argvCopy.data())) {
        return 1;
    }
    //オプションチェック
    if (0 == encPrm.common.inputFilename.length()) {
        _ftprintf(stderr, _T("Input file is not specified.\n"));
        return -1;
    }
    if (0 == encPrm.common.outputFilename.length()) {
        _ftprintf(stderr, _T("Output file is not specified.\n"));
        return -1;
    }

#if defined(_WIN32) || defined(_WIN64)
    //set stdin to binary mode when using pipe input
    if (_tcscmp(encPrm.common.inputFilename.c_str(), _T("-")) == NULL) {
        if (_setmode(_fileno(stdin), _O_BINARY) == 1) {
            _ftprintf(stderr, _T("Error: failed to switch stdin to binary mode."));
            return 1;
        }
    }

    //set stdout to binary mode when using pipe output
    if (_tcscmp(encPrm.common.outputFilename.c_str(), _T("-")) == NULL) {
        if (_setmode(_fileno(stdout), _O_BINARY) == 1) {
            _ftprintf(stderr, _T("Error: failed to switch stdout to binary mode."));
            return 1;
        }
    }
#endif //#if defined(_WIN32) || defined(_WIN64)

    encPrm.encConfig.encodeCodecConfig = codecPrm[encPrm.codec];

    int ret = 1;

    NVEncCore nvEnc;
    if (   NV_ENC_SUCCESS == nvEnc.Initialize(&encPrm)
        && NV_ENC_SUCCESS == nvEnc.InitEncode(&encPrm)) {
        nvEnc.SetAbortFlagPointer(&g_signal_abort);
        set_signal_handler();
        nvEnc.PrintEncodingParamsInfo(RGY_LOG_INFO);
        ret = (NV_ENC_SUCCESS == nvEnc.Encode()) ? 0 : 1;
    }
    return ret;
}
