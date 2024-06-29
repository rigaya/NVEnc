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
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <vector>
#include <set>
#include <cstdio>
#include "rgy_version.h"
#include "rgy_util.h"
#include "rgy_filesystem.h"
#include "rgy_codepage.h"
#include "rgy_resource.h"
#include "rgy_env.h"
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

    const int deviceID = -1;
    const auto loglevel = RGY_LOG_INFO;
    const int cudaSchedule = 0;
    const bool skipHWDecodeCheck = false;

    NVEncCtrl nvEnc;
    if (NV_ENC_SUCCESS == nvEnc.Initialize(deviceID, loglevel)
        && NV_ENC_SUCCESS == nvEnc.ShowDeviceList(cudaSchedule, skipHWDecodeCheck)) {
        return;
    }
}

static int show_hw(int deviceid, const RGYParamLogLevel& loglevelPrint) {
    show_version();

    const int cudaSchedule = 0;
    const bool skipHWDecodeCheck = false;

    NVEncCtrl nvEnc;
    if (NV_ENC_SUCCESS == nvEnc.Initialize(deviceid, loglevelPrint.get(RGY_LOGT_APP))
        && NV_ENC_SUCCESS == nvEnc.ShowCodecSupport(cudaSchedule, skipHWDecodeCheck)) {
        return 0;
    }
    return 1;
}

static void show_environment_info() {
    show_version();
    _ftprintf(stdout, _T("%s\n"), getEnviromentInfo().c_str());
}

static int show_nvenc_features(int deviceid, const RGYParamLogLevel& loglevelPrint) {
    show_version();
    _ftprintf(stdout, _T("\n%s\n"), getEnviromentInfo().c_str());

    const int cudaSchedule = 0;
    const bool skipHWDecodeCheck = false;

    NVEncCtrl nvEnc;
    if (NV_ENC_SUCCESS == nvEnc.Initialize(deviceid, loglevelPrint.get(RGY_LOGT_APP))
        && NV_ENC_SUCCESS == nvEnc.ShowNVEncFeatures(cudaSchedule, skipHWDecodeCheck)) {
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

int parse_print_options(const TCHAR *option_name, const TCHAR *arg1, const RGYParamLogLevel& loglevelPrint) {

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
        return show_hw(deviceid, loglevelPrint) == 0 ? 1 : -1;
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
        return show_nvenc_features(deviceid, loglevelPrint) == 0 ? 1 : -1;
    }
#if ENABLE_AVSW_READER
    if (0 == _tcscmp(option_name, _T("check-avcodec-dll"))) {
        const auto ret = check_avcodec_dll();
        _ftprintf(stdout, _T("%s\n"), ret ? _T("yes") : _T("no"));
        if (!ret) {
            _ftprintf(stdout, _T("%s\n"), error_mes_avcodec_dll_not_found().c_str());
        }
        return ret ? 1 : -1;
    }
    if (0 == _tcscmp(option_name, _T("check-avversion"))) {
        _ftprintf(stdout, _T("%s\n"), getAVVersions().c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-codecs"))) {
        _ftprintf(stdout, _T("Video\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC), { AVMEDIA_TYPE_VIDEO }).c_str());
        _ftprintf(stdout, _T("\nAudio\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC | RGY_AVCODEC_ENC), { AVMEDIA_TYPE_AUDIO }).c_str());
        _ftprintf(stdout, _T("\nSbutitles\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC | RGY_AVCODEC_ENC), { AVMEDIA_TYPE_SUBTITLE }).c_str());
        _ftprintf(stdout, _T("\nData / Attachment\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC | RGY_AVCODEC_ENC), { AVMEDIA_TYPE_DATA, AVMEDIA_TYPE_ATTACHMENT }).c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-encoders"))) {
        _ftprintf(stdout, _T("Audio\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_ENC), { AVMEDIA_TYPE_AUDIO }).c_str());
        _ftprintf(stdout, _T("\nSbutitles\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_ENC), { AVMEDIA_TYPE_SUBTITLE }).c_str());
        _ftprintf(stdout, _T("\nData / Attachment\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_ENC), { AVMEDIA_TYPE_DATA, AVMEDIA_TYPE_ATTACHMENT }).c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-decoders"))) {
        _ftprintf(stdout, _T("Video\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC), { AVMEDIA_TYPE_VIDEO }).c_str());
        _ftprintf(stdout, _T("\nAudio\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC), { AVMEDIA_TYPE_AUDIO }).c_str());
        _ftprintf(stdout, _T("\nSbutitles\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC), { AVMEDIA_TYPE_SUBTITLE }).c_str());
        _ftprintf(stdout, _T("\nData / Attachment\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC), { AVMEDIA_TYPE_DATA, AVMEDIA_TYPE_ATTACHMENT }).c_str());
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
    if (0 == _tcscmp(option_name, _T("check-avdevices"))) {
        _ftprintf(stdout, _T("%s\n"), getAVDevices().c_str());
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

bool check_locale_is_chs() {
	const WORD LangID_zh_CN = MAKELANGID(LANG_CHINESE, SUBLANG_CHINESE_SIMPLIFIED);
	return GetUserDefaultLangID() == LangID_zh_CN;
}

static tstring getErrorFmtStr(uint32_t err) {
    TCHAR errmes[4097];
    FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, NULL, err, NULL, errmes, _countof(errmes), NULL);
    return errmes;
}

static int run_on_os_codepage() {
    auto exepath = getExePath();
    auto tmpexe = std::filesystem::path(PathRemoveExtensionS(exepath));
    tmpexe += strsprintf(_T("A_%x"), GetCurrentProcessId());
    tmpexe += std::filesystem::path(exepath).extension();
    std::filesystem::copy_file(exepath, tmpexe, std::filesystem::copy_options::overwrite_existing);

    SetLastError(0);
    HANDLE handle = BeginUpdateResourceW(tmpexe.wstring().c_str(), FALSE);
    if (handle == NULL) {
        auto lasterr = GetLastError();
        _ftprintf(stderr, _T("Failed to create temporary exe file: [%d] %s.\n"), lasterr, getErrorFmtStr(lasterr).c_str());
        return 1;
    }
    void *manifest = nullptr;
    int size = getEmbeddedResource(&manifest, _T("APP_OSCODEPAGE_MANIFEST"), _T("EXE_DATA"), NULL);
    if (size == 0) {
        _ftprintf(stderr, _T("Failed to load manifest for OS codepage mode.\n"));
        return 1;
    }
    SetLastError(0);
    if (!UpdateResourceW(handle, RT_MANIFEST, CREATEPROCESS_MANIFEST_RESOURCE_ID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), manifest, size)) {
        auto lasterr = GetLastError();
        _ftprintf(stderr, _T("Failed to update manifest for ansi mode: [%d] %s.\n"), lasterr, getErrorFmtStr(lasterr).c_str());
        return 1;
    }
    SetLastError(0);
    if (!EndUpdateResourceW(handle, FALSE)) {
        auto lasterr = GetLastError();
        _ftprintf(stderr, _T("Failed to finish update manifest for OS codepage mode: [%d] %s.\n"), lasterr, getErrorFmtStr(lasterr).c_str());
        return 1;
    }

    const auto commandline = str_replace(str_replace(GetCommandLineW(),
        std::filesystem::path(exepath).filename(), std::filesystem::path(tmpexe).filename()),
        CODEPAGE_CMDARG, CODEPAGE_CMDARG_APPLIED);

    int ret = 0;
    try {
        DWORD flags = 0; // CREATE_NO_WINDOW;

        HANDLE hStdIn, hStdOut, hStdErr;
        DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_INPUT_HANDLE),  GetCurrentProcess(), &hStdIn,  0, TRUE, DUPLICATE_SAME_ACCESS);
        DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_OUTPUT_HANDLE), GetCurrentProcess(), &hStdOut, 0, TRUE, DUPLICATE_SAME_ACCESS);
        DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_ERROR_HANDLE),  GetCurrentProcess(), &hStdErr, 0, TRUE, DUPLICATE_SAME_ACCESS);

        SECURITY_ATTRIBUTES sa;
        memset(&sa, 0, sizeof(SECURITY_ATTRIBUTES));
        sa.nLength = sizeof(sa);
        sa.lpSecurityDescriptor = NULL;
        sa.bInheritHandle = TRUE; //TRUEでハンドルを引き継ぐ

        STARTUPINFO si;
        memset(&si, 0, sizeof(STARTUPINFO));
        si.cb = sizeof(STARTUPINFO);
        //si.dwFlags |= STARTF_USESHOWWINDOW;
        si.dwFlags |= STARTF_USESTDHANDLES;
        //si.wShowWindow |= SW_SHOWMINNOACTIVE;
        si.hStdInput = hStdIn;
        si.hStdOutput = hStdOut;
        si.hStdError = hStdErr;

        PROCESS_INFORMATION pi;
        memset(&pi, 0, sizeof(PROCESS_INFORMATION));

        SetLastError(0);
        if (CreateProcess(nullptr, (LPWSTR)commandline.c_str(), &sa, nullptr, TRUE, flags, nullptr, nullptr, &si, &pi) == 0) {
            auto lasterr = GetLastError();
            _ftprintf(stderr, _T("Failed to run process in OS codepage mode: [%d] %s.\n"), lasterr, getErrorFmtStr(lasterr).c_str());
            ret = 1;
        } else {
            WaitForSingleObject(pi.hProcess, INFINITE);
            DWORD proc_ret = 0;
            if (GetExitCodeProcess(pi.hProcess, &proc_ret)) {
                ret = (int)proc_ret;
            }
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
        }
    } catch (...) {
        ret = 1;
    }
    std::filesystem::remove(tmpexe);
    return ret;
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
    _tsetlocale(LC_CTYPE, _T(".UTF8"));
#endif //#if defined(_WIN32) || defined(_WIN64)

    if (argc == 1) {
        show_version();
        show_help();
        return 1;
    }

#if defined(_WIN32) || defined(_WIN64)
    if (GetACP() == CODE_PAGE_UTF8) {
        bool switch_to_os_cp = false;
        for (int iarg = 1; iarg < argc; iarg++) {
            if (iarg + 1 < argc
                && _tcscmp(argv[iarg + 0], CODEPAGE_CMDARG) == 0) {
                if (_tcscmp(argv[iarg + 1], _T("os")) == 0) {
                    switch_to_os_cp = true;
                } else if (_tcscmp(argv[iarg + 1], _T("utf8")) == 0) {
                    switch_to_os_cp = false;
                } else {
                    _ftprintf(stderr, _T("Unknown option for %s.\n"), CODEPAGE_CMDARG);
                    return 1;
                }
            }
        }
        if (switch_to_os_cp) {
            return run_on_os_codepage();
        }
    }
#endif //#if defined(_WIN32) || defined(_WIN64)

    //log-levelの取得
    RGYParamLogLevel loglevelPrint(RGY_LOG_ERROR);
    for (int iarg = 1; iarg < argc - 1; iarg++) {
        if (tstring(argv[iarg]) == _T("--log-level")) {
            parse_log_level_param(argv[iarg], argv[iarg + 1], loglevelPrint);
            break;
        }
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
            int ret = parse_print_options(option_name, (iarg+1 < argc) ? argv[iarg+1] : _T(""), loglevelPrint);
            if (ret != 0) {
                return ret == 1 ? 0 : 1;
            }
        }
    }

    InEncodeVideoParam encPrm;
    NV_ENC_CODEC_CONFIG codecPrm[RGY_CODEC_NUM] = { 0 };
    codecPrm[RGY_CODEC_H264] = DefaultParamH264();
    codecPrm[RGY_CODEC_HEVC] = DefaultParamHEVC();
    codecPrm[RGY_CODEC_AV1]  = DefaultParamAV1();

    //optionファイルの読み取り
    std::vector<tstring> argvCnfFile;
    for (int iarg = 1; iarg < argc; iarg++) {
        const TCHAR *option_name = nullptr;
        if (argv[iarg][0] == _T('-')) {
            if (argv[iarg][1] == _T('\0')) {
                continue;
            } else if (argv[iarg][1] == _T('-')) {
                option_name = &argv[iarg][2];
            }
        }
        if (option_name != nullptr
            && tstring(option_name) == _T("option-file")) {
            if (iarg + 1 >= argc) {
                _ftprintf(stderr, _T("option file name is not specified.\n"));
                return -1;
            }
            tstring cnffile = argv[iarg + 1];
            vector_cat(argvCnfFile, cmd_from_config_file(argv[iarg + 1]));
        }
    }

    std::vector<const TCHAR *> argvCopy(argv, argv + argc);
    //optionファイルのパラメータを追加
    for (size_t i = 0; i < argvCnfFile.size(); i++) {
        if (argvCnfFile[i].length() > 0) {
            argvCopy.push_back(argvCnfFile[i].c_str());
        }
    }
    argvCopy.push_back(_T(""));

    if (parse_cmd(&encPrm, codecPrm, (int)argvCopy.size()-1, argvCopy.data())) {
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

    if (encPrm.common.inputFilename != _T("-")
        && encPrm.common.outputFilename != _T("-")
        && rgy_path_is_same(encPrm.common.inputFilename, encPrm.common.outputFilename)) {
        _ftprintf(stderr, _T("destination file is equal to source file!\n"));
        return 1;
    }

#if defined(_WIN32) || defined(_WIN64)
    //set stdin to binary mode when using pipe input
    if (_tcscmp(encPrm.common.inputFilename.c_str(), _T("-")) == NULL) {
        if (_setmode(_fileno(stdin), _O_BINARY) == 1) {
            _ftprintf(stderr, _T("Error: failed to switch stdin to binary mode.\n"));
            return 1;
        }
    }

    //set stdout to binary mode when using pipe output
    if (_tcscmp(encPrm.common.outputFilename.c_str(), _T("-")) == NULL) {
        if (_setmode(_fileno(stdout), _O_BINARY) == 1) {
            _ftprintf(stderr, _T("Error: failed to switch stdout to binary mode.\n"));
            return 1;
        }
    }
#endif //#if defined(_WIN32) || defined(_WIN64)

    encPrm.encConfig.encodeCodecConfig = codecPrm[encPrm.codec_rgy];

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
