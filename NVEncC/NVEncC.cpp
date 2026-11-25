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
#include "rgy_cmd_selftest.h"
#include "rgy_help_selftest_cases.h"
#include "NVEncHelpCodegenCases.h"
#include "rgy_preview_server.h"
#include "rgy_preview_history.h"
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
    if (RGY_ERR_NONE == nvEnc.Initialize(deviceID, ENABLE_VULKAN ? RGYParamInitVulkan::TargetVendor : RGYParamInitVulkan::Disable, loglevel)
        && RGY_ERR_NONE == nvEnc.ShowDeviceList(cudaSchedule, skipHWDecodeCheck)) {
        return;
    }
}

static int show_hw(int deviceid, const RGYParamLogLevel& loglevelPrint) {
    show_version();

    const int cudaSchedule = 0;
    const bool skipHWDecodeCheck = false;

    NVEncCtrl nvEnc;
    if (RGY_ERR_NONE == nvEnc.Initialize(deviceid, ENABLE_VULKAN ? RGYParamInitVulkan::TargetVendor : RGYParamInitVulkan::Disable, loglevelPrint.get(RGY_LOGT_APP))
        && RGY_ERR_NONE == nvEnc.ShowCodecSupport(cudaSchedule, skipHWDecodeCheck)) {
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
    if (RGY_ERR_NONE == nvEnc.Initialize(deviceid, ENABLE_VULKAN ? RGYParamInitVulkan::TargetVendor : RGYParamInitVulkan::Disable, loglevelPrint.get(RGY_LOGT_APP))
        && RGY_ERR_NONE == nvEnc.ShowNVEncFeatures(cudaSchedule, skipHWDecodeCheck)) {
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

static int show_nvenc_preset_tune_params(const InEncodeVideoParam& encPrm) {
    show_version();
    _ftprintf(stdout, _T("\n%s\n"), getEnviromentInfo().c_str());

    const int cudaSchedule = 0;
    const bool skipHWDecodeCheck = false;
    const RGYParamLogLevel loglevelPrint(encPrm.ctrl.loglevel);

    int profile = -1;
    if (encPrm.codec_rgy == RGY_CODEC_H264) {
        profile = encPrm.h264.profile;
    } else if (encPrm.codec_rgy == RGY_CODEC_HEVC) {
        profile = encPrm.hevc.profile;
    } else if (encPrm.codec_rgy == RGY_CODEC_AV1) {
        profile = encPrm.av1.profile;
    } else {
        _ftprintf(stdout, _T("Unknown codec.\n"));
        return 1;
    }

    int deviceID = 0;
    if (encPrm.deviceID >= 0) {
        deviceID = encPrm.deviceID;
    }
    
    NVEncCtrl nvEnc;
    if (RGY_ERR_NONE == nvEnc.Initialize(deviceID, ENABLE_VULKAN ? RGYParamInitVulkan::TargetVendor : RGYParamInitVulkan::Disable, loglevelPrint.get(RGY_LOGT_APP))
        && RGY_ERR_NONE == nvEnc.ShowNVEncPresetTuneParams(cudaSchedule, skipHWDecodeCheck, encPrm.codec_rgy, profile, encPrm.preset, encPrm.tuningInfo)) {
        return 0;
    }
    return 1;
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
    if (IS_OPTION("check-cmd-parse")) {
        // コマンドラインのパーサと生成の対応関係を自己診断する
        const tstring filter = (arg1[0] != _T('-')) ? arg1 : _T("");
        if (filter == _T("help-test")) return rgy_help_selftest_cases();
        if (filter == _T("help-codegen-test")) return rgy_help_codegen_cases();
        const bool helpMode = filter == _T("help") || filter.compare(0, 5, _T("help:")) == 0;
        RGYCmdSelfTest<InEncodeVideoParam> selftest(
            [](InEncodeVideoParam *prm, const std::vector<tstring>& args) {
                auto argv = rgy_cmd_selftest_argv(args);
                return parse_cmd(prm, (int)argv.size() - 1, argv.data());
            },
            // 無効なフィルタの状態も保存し、生成省略による偽の一致を防ぐ。
            [helpMode](const InEncodeVideoParam *prm) { return gen_cmd(prm, helpMode); });
        if (helpMode) return selftest.runHelpValidation(encoder_help(), filter == _T("help") ? tstring() : filter.substr(5));
        return selftest.run(filter);
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
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC | RGY_AVCODEC_ENC), { AVMEDIA_TYPE_VIDEO }).c_str());
        _ftprintf(stdout, _T("\nAudio\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC | RGY_AVCODEC_ENC), { AVMEDIA_TYPE_AUDIO }).c_str());
        _ftprintf(stdout, _T("\nSubtitles\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC | RGY_AVCODEC_ENC), { AVMEDIA_TYPE_SUBTITLE }).c_str());
        _ftprintf(stdout, _T("\nData / Attachment\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_DEC | RGY_AVCODEC_ENC), { AVMEDIA_TYPE_DATA, AVMEDIA_TYPE_ATTACHMENT }).c_str());
        return 1;
    }
    if (0 == _tcscmp(option_name, _T("check-encoders"))) {
        _ftprintf(stdout, _T("Video\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_ENC), { AVMEDIA_TYPE_VIDEO }).c_str());
        _ftprintf(stdout, _T("Audio\n"));
        _ftprintf(stdout, _T("%s\n"), getAVCodecs((RGYAVCodecType)(RGY_AVCODEC_ENC), { AVMEDIA_TYPE_AUDIO }).c_str());
        _ftprintf(stdout, _T("\nSubtitles\n"));
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
        _ftprintf(stdout, _T("\nSubtitles\n"));
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

// 通常起動と常駐要求で同じエンコード経路を使い、終了時のリソースも要求ごとに破棄する。
static RGY_ERR run_encode_params(InEncodeVideoParam& params, std::atomic<bool> *previewAbort = nullptr) {
    NVEncCore nvEnc;
    auto sts = nvEnc.Init(&params);
    if (sts != RGY_ERR_NONE) return sts;
    if (previewAbort) {
        nvEnc.SetPreviewAbortFlagPointer(previewAbort);
    } else {
        nvEnc.SetAbortFlagPointer(&g_signal_abort);
        set_signal_handler();
    }
    nvEnc.PrintEncodingParamsInfo(RGY_LOG_INFO);
    return nvEnc.Encode();
}

#include "rgy_delay_load_hook.h"
#if ENABLE_PROBE_BUILD
#include "rgy_probe_cli.h"
#endif

int _tmain(int argc, TCHAR **argv) {
#if ENABLE_PROBE_BUILD
    if (!rgy_probe_cli_args(argc, argv)) return 1;
#endif
#if defined(_WIN32) || defined(_WIN64)
    _tsetlocale(LC_CTYPE, _T(".UTF8"));
#endif //#if defined(_WIN32) || defined(_WIN64)

    bool previewServer = false;
    for (int iarg = 1; iarg < argc; ++iarg) {
        if (_tcscmp(argv[iarg], _T("--preview-server")) == 0) previewServer = true;
    }
    if (previewServer) {
        std::vector<tstring> defaults;
        for (int iarg = 1; iarg < argc; ++iarg) {
            if (_tcscmp(argv[iarg], _T("--preview-server")) == 0) continue;
            if (_tcscmp(argv[iarg], _T("--log-level")) != 0 || iarg + 1 >= argc) {
                _ftprintf(stderr, _T("--preview-server: Invalid startup arguments.\n"));
                return 1;
            }
            defaults.push_back(argv[iarg]);
            defaults.push_back(argv[++iarg]);
        }
        rgy_output_set_preview_server_mode(true);
        auto callback = [defaults](const std::vector<tstring>& args, std::atomic<bool> *abort, bool windowSeek, int64_t outputStartUs) -> RGY_ERR {
            RGYPreviewInputScope windowScope(windowSeek, abort, outputStartUs);
            auto mergedArgs = defaults;
            mergedArgs.insert(mergedArgs.end(), args.begin(), args.end());
            auto previewArgv = rgy_cmd_selftest_argv(mergedArgs);
            InEncodeVideoParam params;
            if (parse_cmd(&params, (int)previewArgv.size() - 1, previewArgv.data()) != 0) return RGY_ERR_INVALID_PARAM;
            // 出力開始時刻(outputStartUs)は入力の絶対時刻なので、出力の時刻が入力の絶対時刻になる場合だけ受け付ける。
            // 窓の読み込みと--timestamp-passthroughが必要で、--trimや割合指定の--seekとは組み合わせない。
            // --seekなし(先頭から読む)の要求は、先頭のフレームから表示に使うので、出力開始時刻を指定する対象にしない。
            if (outputStartUs != INT64_MIN && (!windowSeek || !params.common.timestampPassThrough
                || params.common.seekSec <= 0 || params.common.seekRatio > 0
                || params.common.nTrimCount > 0)) return RGY_ERR_INVALID_PARAM;
            const auto result = run_encode_params(params, abort);
            // 入力初期化中の取消も、一般の入力エラーではなくrunの取消として返す。
            return windowSeek && abort && abort->load() && result != RGY_ERR_NONE ? RGY_ERR_ABORTED : result;
        };
        auto probe = [defaults](const std::vector<tstring>& args, bool& history) -> RGY_ERR {
            history = true;
            auto mergedArgs = defaults;
            mergedArgs.insert(mergedArgs.end(), args.begin(), args.end());
            auto previewArgv = rgy_cmd_selftest_argv(mergedArgs);
            InEncodeVideoParam params;
            if (parse_cmd(&params, (int)previewArgv.size() - 1, previewArgv.data()) != 0) return RGY_ERR_INVALID_PARAM;
            if (rgy_preview_external_history(params)) return RGY_ERR_NONE;
            if (params.vppnv.deinterlace != cudaVideoDeinterlaceMode_Weave) return RGY_ERR_NONE;
            const auto filters = NVEncCore::CreateVppFilterOrder(&params, false, cropEnabled(params.input.crop), rgy_preview_resize_type(params.input, params.vpp));
            for (const auto type : filters) {
                if (rgy_preview_filter_depends_on_history(type, params.vpp)) return RGY_ERR_NONE;
            }
            history = false;
            return RGY_ERR_NONE;
        };
        return rgy_preview_server_main(callback, "NVEnc", VER_STR_FILEVERSION, probe, true, true);
    }

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

    if (parse_cmd(&encPrm, (int)argvCopy.size()-1, argvCopy.data())) {
        return 1;
    }

    for (int i = 1; i < (int)argvCopy.size()-1; i++) {
        if (tstring(argvCopy[i]) == _T("--check-preset-params")) {
            return show_nvenc_preset_tune_params(encPrm);
        }
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

    // Do not call rgy_path_is_same on stdin / Windows named pipes: equivalent()
    // would connect to a waiting pipe and consume the instance before open.
    if (encPrm.common.inputFilename != _T("-")
        && encPrm.common.outputFilename != _T("-")
        && !rgy_path_is_windows_named_pipe(encPrm.common.inputFilename)
        && !rgy_path_is_windows_named_pipe(encPrm.common.outputFilename)
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
    if (encPrm.ctrl.processMonitorDevUsageReset) {
        return processMonitorRGYDeviceResetEntry();
    }
    if (encPrm.ctrl.processMonitorDevUsage) {
        return processMonitorRGYDeviceUsage(encPrm.deviceID);
    }

    return run_encode_params(encPrm) == RGY_ERR_NONE ? 0 : 1;
}
