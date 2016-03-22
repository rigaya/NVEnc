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

#define WIN32_MEAN_AND_LEAN
#define NOMINMAX
#include <Windows.h>
#include <locale.h>
#include <tchar.h>
#include <locale.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cstdio>
#include "NVEncVersion.h"
#include "NVEncCore.h"
#include "NVEncFeature.h"
#include "NVEncParam.h"
#include "NVEncUtil.h"

bool check_locale_is_ja() {
    const WORD LangID_ja_JP = MAKELANGID(LANG_JAPANESE, SUBLANG_JAPANESE_JAPAN);
    return GetUserDefaultLangID() == LangID_ja_JP;
}

static void show_version() {
    static const TCHAR *const ENABLED_INFO[] = { _T("disabled"), _T("enabled") };
    _ftprintf(stdout, _T("NVEncC (%s) %s by rigaya [NVENC API v%d.%d], build %s %s\n"), BUILD_ARCH_STR, VER_STR_FILEVERSION_TCHAR, NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION, _T(__DATE__), _T(__TIME__));
    _ftprintf(stdout, _T("  avi reader: %s\n"), ENABLED_INFO[!!AVI_READER]);
    _ftprintf(stdout, _T("  avs reader: %s\n"), ENABLED_INFO[!!AVS_READER]);
    _ftprintf(stdout, _T("  vpy reader: %s\n"), ENABLED_INFO[!!VPY_READER]);
    _ftprintf(stdout, _T("  avcuvid reader: %s\n"), ENABLED_INFO[!!ENABLE_AVCUVID_READER]);
    _ftprintf(stdout, _T("\n"));
}


//適当に改行しながら表示する
static void print_list_options(FILE *fp, TCHAR *option_name, const CX_DESC *list, int default_index) {
    const TCHAR *indent_space = _T("                                  ");
    const int indent_len = (int)_tcslen(indent_space);
    const int max_len = 77;
    int print_len = _ftprintf(fp, _T("   %s "), option_name);
    while (print_len < indent_len)
         print_len += _ftprintf(stdout, _T(" "));
    for (int i = 0; list[i].desc; i++) {
        if (print_len + _tcslen(list[i].desc) + _tcslen(_T(", ")) >= max_len) {
            _ftprintf(fp, _T("\n%s"), indent_space);
            print_len = indent_len;
        } else {
            if (i)
                print_len += _ftprintf(fp, _T(", "));
        }
        print_len += _ftprintf(fp, _T("%s"), list[i].desc);
    }
    _ftprintf(fp, _T("\n%s default: %s\n"), indent_space, list[default_index].desc);
}

typedef struct ListData {
    const TCHAR *name;
    const CX_DESC *list;
    int default_index;
} ListData;

static void PrintMultipleListOptions(FILE *fp, const TCHAR *option_name, const TCHAR *option_desc, const vector<ListData>& listDatas) {
    const TCHAR *indent_space = _T("                                  ");
    const int indent_len = (int)_tcslen(indent_space);
    const int max_len = 79;
    int print_len = _ftprintf(fp, _T("   %s "), option_name);
    while (print_len < indent_len)
        print_len += _ftprintf(fp, _T(" "));
    _ftprintf(fp, _T("%s\n"), option_desc);
    const auto data_name_max_len = indent_len + 4 + std::accumulate(listDatas.begin(), listDatas.end(), 0,
        [](const int max_len, const ListData data) { return (std::max)(max_len, (int)_tcslen(data.name)); });

    for (const auto& data : listDatas) {
        print_len = _ftprintf(fp, _T("%s- %s: "), indent_space, data.name);
        while (print_len < data_name_max_len)
            print_len += _ftprintf(fp, _T(" "));
        for (int i = 0; data.list[i].desc; i++) {
            const int desc_len = (int)(_tcslen(data.list[i].desc) + _tcslen(_T(", ")) + ((i == data.default_index) ? _tcslen(_T("(default)")) : 0));
            if (print_len + desc_len >= max_len) {
                _ftprintf(fp, _T("\n%s"), indent_space);
                print_len = indent_len;
                while (print_len < data_name_max_len)
                    print_len += _ftprintf(fp, _T(" "));
            } else {
                if (i)
                    print_len += _ftprintf(fp, _T(", "));
            }
            print_len += _ftprintf(fp, _T("%s%s"), data.list[i].desc, (i == data.default_index) ? _T("(default)") : _T(""));
        }
        _ftprintf(fp, _T("\n"));
    }
}

static void show_help_ja() {
    show_version();

    _ftprintf(stdout, _T("使用方法: NVEncC.exe [オプション] -i <入力ファイル名> -o <出力ファイル名>\n"));
    _ftprintf(stdout, _T("\n")
        _T("入力は %s%sraw YUV, YUV4MPEG2(y4m)です。\n")
        _T("Raw入力時には, fps, input-res の指定も必要です。\n")
        _T("\n")
        _T("出力フォーマットは raw H.264/AVC または H.265/HEVC ESです。\n")
        _T("\n")
        _T("実行例:\n")
        _T("  NVEncC -i \"<avsfilename>\" -o \"<outfilename>\"\n")
        _T("  avs2pipemod -y4mp \"<avsfile>\" | NVEncC --y4m -i - -o \"<outfilename>\"\n")
        _T("\n")
        _T("オプション: \n")
        _T("-h,-? --help                      ヘルプの表示\n")
        _T("   --help-ja                      ヘルプの表示(日本語)\n")
        _T("   --help-en                      ヘルプの表示(英語)\n")
        _T("-v,--version                      バージョン情報の表示\n")
        _T("   --check-device                 利用可能なGPUのDeviceIdを表示\n")
        _T("   --check-hw [<int>]             指定したDeviceIdでNVEncが使用可能か確認\n")
        _T("                                    指定のない場合はDeviceId #0\n")
        _T("   --check-features [<int>]       指定したDeviceIdでNVEncの使用可能な機能を表示\n")
        _T("                                    指定のない場合はDeviceId #0\n")
        _T("   --check-environment            NVEncの認識している環境情報を表示\n")
        _T("\n")
        _T("-d,--device                       NVEncで使用するDeviceIdを指定(デフォルト:0)\n")
        _T("\n")
        _T("-i,--input <filename>             入力ファイル名の指定\n")
        _T("-o,--output <filename>            出力ファイル名の指定\n")
        _T("\n")
        _T(" 入力フォーマット (指定のない場合、拡張子から自動判定)\n")
        _T("   --raw                          rawとしてファイルを読み込み\n")
        _T("   --y4m                          y4mとしてファイルを読み込み\n")
#if AVI_READER
        _T("   --avi                          aviとしてファイルを読み込み\n")
#endif
#if AVS_READER
        _T("   --avs                          avsとしてファイルを読み込み\n")
#endif
#if VPY_READER
        _T("   --vpy                          vpyとしてファイルを読み込み\n")
        _T("   --vpy-mt                       vpy(mt)としてファイルを読み込み\n")
#endif
#if ENABLE_AVCUVID_READER
        _T("   --avcuvid                      libavformat + cuvidで読み込み\n")
#endif
        _T("\n"),
        (AVI_READER) ? _T("avi, ") : _T(""),
        (AVS_READER) ? _T("avs, ") : _T(""));

    _ftprintf(stdout, _T("")
        _T("   --input-res <int>x<int>        入力解像度\n")
        _T("   --crop <int>,<int>,<int>,<int> 左、上、右、下の切り落とし画素数\n")
        _T("                                    avcuivid reader使用時は左cropは無効\n")
        _T("   --output-res <int>x<int>       出力解像度\n")
        _T("-f,--fps <int>/<int> or <float>   フレームレートの指定\n")
        _T("\n")
        _T("-c,--codec <string>               出力コーデックの指定\n")
        _T("                                    h264 (or avc), h265 (or hevc)\n")
        _T("   --profile <string>             プロファイルの指定\n")
        _T("                                    H.264: baseline, main, high, high444\n")
        _T("                                    HEVC : main\n"));
    PrintMultipleListOptions(stdout, _T("--level <string>"), _T("コーデックレベルの指定"),
        { { _T("H.264"), list_avc_level,   0 },
          { _T("HEVC"),  list_hevc_level,  0 } 
    });
    _ftprintf(stdout, _T("")
        _T("   --level <string>               Levelの指定\n")
        _T("   --sar <int>:<int>              SAR(PAR, 画素比)の指定\n")
        _T("   --dar <int>:<int>              DAR(画面比)の指定\n")
        _T("\n")
        _T("   --cqp <int> or                 固定量子化量でエンコード\n")
        _T("         <int>:<int>:<int>          デフォルト: <I>:<P>:<B>=<%d>:<%d>:<%d>\n")
        _T("   --vbr <int>                    可変ビットレートのビットレート (kbps)\n")
        _T("   --vbr2 <int>                   可変ビットレート(2)のビットレート (kbps)\n")
        _T("   --cbr <int>                    固定ビットレートのビットレート (kbps)\n")
        _T("                                    デフォルト %d kbps\n")
        _T("\n")
        _T("   --maxbitrate <int>             最大ビットレート(kbps) / デフォルト: %d kbps\n")
        _T("   --qp-init <int> or             初期QPを設定\n")
        _T("             <int>:<int>:<int>      デフォルト: 自動\n")
        _T("   --qp-max <int> or              最大QPを設定\n")
        _T("            <int>:<int>:<int>       デフォルト: 指定なし\n")
        _T("   --qp-min <int> or              最小QPを設定\n")
        _T("             <int>:<int>:<int>      デフォルト: 指定なし\n")
        _T("   --gop-len <int>                GOPのフレーム数 / デフォルト: %d frames%s\n")
        _T("-b,--bframes <int>                連続Bフレーム数 / デフォルト %d フレーム\n")
        _T("   --ref <int>                    参照距離 / デフォルト %d フレーム\n")
        _T("   --aq                           適応的量子化(AQ)を有効にする\n")
        _T("   --mv-precision <string>        動きベクトル精度 / デフォルト: Q-pel\n")
        _T("                                    Q-pel    … 1/4画素精度 (高精度)\n")
        _T("                                    half-pel … 1/2画素精度\n")
        _T("                                    full-pel … 1  画素精度 (低精度)\n")
        _T("   --vbv-bufsize <int>            VBVバッファサイズ (kbit) / デフォルト 自動\n")
        _T("   --vpp-deinterlace <string>     インタレ解除を行う(avcuvid使用時のみ)\n")
        _T("                                    none(デフォルト), bob, adaptive\n")
        _T("   --fullrange                    fullrangeの指定\n"),
        DEFAUTL_QP_I, DEFAULT_QP_P, DEFAULT_QP_B,
        DEFAULT_AVG_BITRATE / 1000, DEFAULT_MAX_BITRATE / 1000,
        DEFAULT_GOP_LENGTH, (DEFAULT_GOP_LENGTH == 0) ? _T(" (自動)") : _T(""),
        DEFAULT_B_FRAMES, DEFAULT_REF_FRAMES);
        print_list_options(stdout, _T("--videoformat <string>"), list_videoformat, 0);
        print_list_options(stdout, _T("--colormatrix <string>"), list_colormatrix, 0);
        print_list_options(stdout, _T("--colorprim <string>"),   list_colorprim,   0);
        print_list_options(stdout, _T("--transfer <string>"),    list_transfer,    0);
    _ftprintf(stdout, _T("\n")
        _T("   --log <string>                 ログファイル名の指定\n")
        _T("   --log-level <string>           ログレベルの指定 / デフォルト: info\n")
        _T("                                    debug, info, warn, error\n"));

    _ftprintf(stdout, _T("\n")
        _T("H.264/AVC\n")
        _T("   --interlaced <string>          インタレ保持エンコ\n")
        _T("                                    tff, bff\n")
        _T("   --cabac                        CABACを使用する\n")
        _T("   --cavlc                        CAVLCを使用する\n")
        _T("   --bluray                       Bluray用出力を行う / デフォルト: オフ\n")
        _T("   --lossless                     ロスレス出力を行う / デフォルト: オフ\n")
        _T("   --deblock                      デブロックフィルタを有効にする\n")
        _T("   --no-deblock                   デブロックフィルタを無効にする\n"));

        _ftprintf(stdout, _T("\n")
            _T("H.265/HEVC\n")
            _T("   --cu-max <int>                 CUの最大サイズを指定する\n")
            _T("   --cu-min  <int>                CUの最小サイズを指定する\n")
            _T("                                    8, 16, 32 を指定可能"));
}

static void show_help_en() {
    show_version();

    _ftprintf(stdout, _T("Usage: NVEncC.exe [Options] -i <input file> -o <output file>\n"));
    _ftprintf(stdout, _T("\n")
        _T("Input can be %s%sraw YUV, YUV4MPEG2(y4m).\n")
        _T("When Input is in raw format, fps, input-res is required.\n")
        _T("\n")
        _T("Ouput format will be in raw H.264/AVC or H.265/HEVC ES.\n")
        _T("\n")
        _T("Example:\n")
        _T("  NVEncC -i \"<avsfilename>\" -o \"<outfilename>\"\n")
        _T("  avs2pipemod -y4mp \"<avsfile>\" | NVEncC --y4m -i - -o \"<outfilename>\"\n")
        _T("\n")
        _T("Options: \n")
        _T("-h,-? --help                      print help\n")
        _T("   --help-ja                      print help in Japanese\n")
        _T("   --help-en                      print help in English\n")
        _T("-v,--version                      print version info\n")
        _T("   --check-device                 show DeviceId for GPUs available on system")
        _T("   --check-hw [<int>]             check NVEnc codecs for specefied DeviceId\n")
        _T("                                    if unset, will check DeviceId #0\n")
        _T("   --check-features [<int>]       check for NVEnc Features for specefied DeviceId\n")
        _T("                                    if unset, will check DeviceId #0\n")
        _T("   --check-environment            check for Environment Info\n")
        _T("\n")
        _T("-d,--device <int>                 set DeviceId used in NVEnc (default:0)\n")
        _T("\n")
        _T("-i,--input <filename>             set input filename\n")
        _T("-o,--output <filename>            set output filename\n")
        _T("\n")
        _T(" Input formats (auto detected from extension of not set)\n")
        _T("   --raw                          set input as raw format\n")
        _T("   --y4m                          set input as y4m format\n")
#if AVI_READER
        _T("   --avi                          set input as avi format\n")
#endif
#if AVS_READER
        _T("   --avs                          set input as avs format\n")
#endif
#if VPY_READER
        _T("   --vpy                          set input as vpy format\n")
        _T("   --vpy-mt                       set input as vpy(mt) format\n")
#endif
#if ENABLE_AVCUVID_READER
        _T("   --avcuvid                      use libavformat + cuvid\n")
#endif
        _T("\n"),
        (AVI_READER) ? _T("avi, ") : _T(""),
        (AVS_READER) ? _T("avs, ") : _T(""));
    _ftprintf(stdout, _T("")
        _T("   --input-res <int>x<int>        set input resolution\n")
        _T("   --crop <int>,<int>,<int>,<int> crop pixels from left,top,right,bottom\n")
        _T("                                    left crop is unavailable with avcuivid reader\n")
        _T("   --output-res <int>x<int>       set output resolution\n")
        _T("-f,--fps <int>/<int> or <float>   set framerate\n")
        _T("\n")
        _T("-c,--codec <string>               set ouput codec\n")
        _T("                                    h264 (or avc), h265 (or hevc)\n")
        _T("   --profile <string>             set codec profile\n")
        _T("                                    H.264: baseline, main, high(default)\n")
        _T("                                    HEVC : main\n"));

    PrintMultipleListOptions(stdout, _T("--level <string>"), _T("set codec level"),
        { { _T("H.264"), list_avc_level,   0 },
          { _T("HEVC"),  list_hevc_level,  0 }
    });
    _ftprintf(stdout, _T("")
        _T("   --sar <int>:<int>              set SAR ratio\n")
        _T("   --dar <int>:<int>              set DAR ratio\n")
        _T("\n")
        _T("   --cqp <int> or                 encode in Constant QP mode\n")
        _T("         <int>:<int>:<int>          Default: <I>:<P>:<B>=<%d>:<%d>:<%d>\n")
        _T("   --vbr <int>                    set bitrate for VBR mode (kbps)\n")
        _T("   --vbr2 <int>                   set bitrate for VBR2 mode (kbps)\n")
        _T("   --cbr <int>                    set bitrate for CBR mode (kbps)\n")
        _T("                                    Default: %d kbps\n")
        _T("\n")
        _T("   --maxbitrate <int>             set Max Bitrate (kbps) / Default: %d kbps\n")
        _T("   --qp-init <int> or             set initial QP\n")
        _T("             <int>:<int>:<int>      Default: auto\n")
        _T("   --qp-max <int> or              set max QP\n")
        _T("            <int>:<int>:<int>       Default: unset\n")
        _T("   --qp-min <int> or              set min QP\n")
        _T("             <int>:<int>:<int>      Default: unset\n")
        _T("   --gop-len <int>                set GOP Length / Default: %d frames%s\n")
        _T("-b,--bframes <int>                set B frames / Default %d frames\n")
        _T("   --ref <int>                    set Ref frames / Default %d frames\n")
        _T("   --aq                           enable adaptive quantization\n")
        _T("   --mv-precision <string>        set MV Precision / Default: Q-pel\n")
        _T("                                    Q-pel    (High Quality)\n")
        _T("                                    half-pel\n")
        _T("                                    full-pel (Low Quality)\n")
        _T("   --vbv-bufsize <int>            set vbv buffer size (kbit) / Default: auto\n")
        _T("   --vpp-deinterlace <string>     set deinterlace mode / Default: none\n")
        _T("                                    none, bob, adaptive\n")
        _T("                                    available only with avcuvid reader\n")
        _T("   --fullrange                    set fullrange\n"),
        DEFAUTL_QP_I, DEFAULT_QP_P, DEFAULT_QP_B,
        DEFAULT_AVG_BITRATE / 1000, DEFAULT_MAX_BITRATE / 1000,
        DEFAULT_GOP_LENGTH, (DEFAULT_GOP_LENGTH == 0) ? _T(" (auto)") : _T(""),
        DEFAULT_B_FRAMES, DEFAULT_REF_FRAMES);
    print_list_options(stdout, _T("--videoformat <string>"), list_videoformat, 0);
    print_list_options(stdout, _T("--colormatrix <string>"), list_colormatrix, 0);
    print_list_options(stdout, _T("--colorprim <string>"),   list_colorprim,   0);
    print_list_options(stdout, _T("--transfer <string>"),    list_transfer,    0);
    _ftprintf(stdout, _T("\n")
        _T("   --log <string>                 set log file name\n")
        _T("   --log-level <string>           set log level\n")
        _T("                                    debug, info(default), warn, error\n"));
    _ftprintf(stdout, _T("\n")
        _T("H.264/AVC\n")
        _T("   --interlaced <string>          interlaced encoding\n")
        _T("                                    tff, bff\n")
        _T("   --cabac                        use CABAC\n")
        _T("   --cavlc                        use CAVLC (no CABAC)\n")
        _T("   --bluray                       for bluray / Default: off\n")
        _T("   --lossless                     for lossless / Default: off\n")
        _T("   --(no-)deblock                 enable(disable) deblock filter\n"));

    _ftprintf(stdout, _T("\n")
        _T("H.265/HEVC\n")
        _T("   --cu-max <int>                 set max CU size\n")
        _T("   --cu-min  <int>                set min CU size\n")
        _T("                                    8, 16, 32 are avaliable"));
}

static void show_help() {
    (check_locale_is_ja()) ? show_help_ja() : show_help_en();
}

static const TCHAR *short_opt_to_long(TCHAR short_opt) {
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
        option_name = _T("fps");
        break;
    case _T('i'):
        option_name = _T("input");
        break;
    case _T('o'):
        option_name = _T("output");
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

static void PrintHelp(const TCHAR *strAppName, const TCHAR *strErrorMessage, const TCHAR *strOptionName, const TCHAR *strErrorValue = nullptr) {
    if (strErrorMessage) {
        if (strOptionName) {
            if (strErrorValue) {
                _ftprintf(stderr, _T("Error: %s \"%s\" for \"--%s\"\n"), strErrorMessage, strErrorValue, strOptionName);
                if (0 == _tcsnccmp(strErrorValue, _T("--"), _tcslen(_T("--")))
                    || (strErrorValue[0] == _T('-') && strErrorValue[2] == _T('\0') && short_opt_to_long(strErrorValue[1]) != nullptr)) {
                    _ftprintf(stderr, _T("       \"--%s\" requires value.\n\n"), strOptionName);
                }
            } else {
                _ftprintf(stderr, _T("Error: %s for --%s\n\n"), strErrorMessage, strOptionName);
            }
        } else {
            _ftprintf(stderr, _T("Error: %s\n\n"), strErrorMessage);
        }
    } else {
        show_help();
    }
}

static void show_device_list() {
    if (!check_if_nvcuda_dll_available()) {
        _ftprintf(stdout, _T("CUDA not available.\n"));
        return;
    }

    NVEncoderGPUInfo gpuInfo;
    auto gpuList = gpuInfo.getGPUList();
    if (0 == gpuList.size()) {
        _ftprintf(stdout, _T("No GPU found suitable for NVEnc Encoding.\n"));
        return;
    }

    for (uint32_t i = 0; i < gpuList.size(); i++) {
        _ftprintf(stdout, _T("DeviceId #%d: %s\n"), gpuList[i].first, gpuList[i].second.c_str());
    }
}

static void show_hw(int deviceid) {
    show_version();
    
    NVEncFeature nvFeature;
    nvFeature.createCacheAsync(deviceid);
    auto nvEncCaps = nvFeature.GetCachedNVEncCapability();
    if (nvEncCaps.size()) {
        _ftprintf(stdout, _T("Avaliable Codec(s)\n"));
        for (auto codecNVEncCaps : nvEncCaps) {
            _ftprintf(stdout, _T("%s\n"), get_name_from_guid(codecNVEncCaps.codec, list_nvenc_codecs));
        }
    } else {
        _ftprintf(stdout, _T("No NVEnc support.\n"));
    }
}

static void show_environment_info() {
    _ftprintf(stderr, _T("%s\n"), getEnviromentInfo(false).c_str());
}

static void show_nvenc_features(int deviceid) {
    NVEncFeature nvFeature;
    if (nvFeature.createCacheAsync(deviceid)) {
        _ftprintf(stdout, _T("error on checking features.\n"));
        return;
    }
    auto nvEncCaps = nvFeature.GetCachedNVEncCapability();


    _ftprintf(stdout, _T("%s\n"), getEnviromentInfo(false).c_str());
    if (nvEncCaps.size() == 0) {
        _ftprintf(stdout, _T("No NVEnc support.\n"));
    } else {
        _ftprintf(stdout, _T("List of available features.\n"));
        for (auto codecNVEncCaps : nvEncCaps) {
            _ftprintf(stdout, _T("Codec: %s\n"), get_name_from_guid(codecNVEncCaps.codec, list_nvenc_codecs));
            size_t max_length = 0;
            std::for_each(codecNVEncCaps.caps.begin(), codecNVEncCaps.caps.end(), [&max_length](const NVEncCap& x) { max_length = (std::max)(max_length, _tcslen(x.name)); });
            for (auto cap : codecNVEncCaps.caps) {
                _ftprintf(stdout, _T("%s"), cap.name);
                for (size_t i = _tcslen(cap.name); i <= max_length; i++)
                    _ftprintf(stdout, _T(" "));
                _ftprintf(stdout, _T("%d\n"), cap.value);
            }
            _ftprintf(stdout, _T("\n"));
        }
    }
}

#define IS_OPTION(x) (0 == _tcscmp(option_name, _T(x)))

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
    uint32_t nTmpInputBuf = 0;
};

int parse_one_option(const TCHAR *option_name, const TCHAR* strInput[], int& i, int nArgNum, InEncodeVideoParam *pParams, NV_ENC_CODEC_CONFIG *codecPrm, sArgsData *argData) {
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        pParams->deviceID = deviceid;
        return 0;
    }
    if (IS_OPTION("input")) {
        i++;
        auto length = _tcslen(strInput[i]) + 1;
        pParams->input.filename = (TCHAR *)malloc(sizeof(pParams->input.filename[0]) * length);
        memcpy(pParams->input.filename, strInput[i], sizeof(pParams->input.filename[0]) * length);
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
            pParams->input.rate  = a[0];
            pParams->input.scale = a[1];
        } else {
            double d;
            if (1 == _stscanf_s(strInput[i], _T("%lf"), &d)) {
                int rate = (int)(d * 1001.0 + 0.5);
                if (rate % 1000 == 0) {
                    pParams->input.rate = rate;
                    pParams->input.scale = 1001;
                } else {
                    pParams->input.scale = 100000;
                    pParams->input.rate = (int)(d * pParams->input.scale + 0.5);
                    int gcd = nv_get_gcd(pParams->input.rate, pParams->input.scale);
                    pParams->input.scale /= gcd;
                    pParams->input.rate  /= gcd;
                }
            } else  {
                PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            pParams->input.width  = a[0];
            pParams->input.height = a[1];
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("raw")) {
        pParams->input.type = NV_ENC_INPUT_RAW;
        return 0;
    }
    if (IS_OPTION("y4m")) {
        pParams->input.type = NV_ENC_INPUT_Y4M;
#if AVI_READER
        return 0;
    }
    if (IS_OPTION("avi")) {
        pParams->input.type = NV_ENC_INPUT_AVI;
#endif
#if AVS_READER
        return 0;
    }
    if (IS_OPTION("avs")) {
        pParams->input.type = NV_ENC_INPUT_AVS;
#endif
#if VPY_READER
        return 0;
    }
    if (IS_OPTION("vpy")) {
        pParams->input.type = NV_ENC_INPUT_VPY;
        return 0;
    }
    if (IS_OPTION("vpy-mt")) {
        pParams->input.type = NV_ENC_INPUT_VPY_MT;
#endif
#if ENABLE_AVCUVID_READER
        return 0;
    }
    if (IS_OPTION("avcuvid")) {
        pParams->input.type = NV_ENC_INPUT_AVCUVID;
#endif
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
        if (1 == _stscanf_s(strInput[i], _T("%d"), &a[0])) {
            pParams->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
            pParams->encConfig.rcParams.constQP.qpIntra  = a[0];
            pParams->encConfig.rcParams.constQP.qpInterP = a[0];
            pParams->encConfig.rcParams.constQP.qpInterB = a[0];
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("vbr")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR_MINQP;
            pParams->encConfig.rcParams.averageBitRate = value * 1000;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("vbr2")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_2_PASS_VBR;
            pParams->encConfig.rcParams.averageBitRate = value * 1000;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("qp-init") || IS_OPTION("qp-max") || IS_OPTION("qp-min")) {
        NV_ENC_QP *ptrQP = nullptr;
        if (IS_OPTION("qp-init")) {
            pParams->encConfig.rcParams.enableInitialRCQP = 1;
            ptrQP = &pParams->encConfig.rcParams.initialRCQP;
            return 0;
        }
        if (IS_OPTION("qp-max")) {
            pParams->encConfig.rcParams.enableMaxQP = 1;
            ptrQP = &pParams->encConfig.rcParams.maxQP;
            return 0;
        }
        if (IS_OPTION("qp-min")) {
            pParams->encConfig.rcParams.enableMinQP = 1;
            ptrQP = &pParams->encConfig.rcParams.minQP;
        } else {
            return -1;
        }
        i++;
        int a[3] = { 0 };
        if (   3 == _stscanf_s(strInput[i], _T("%d:%d:%d"), &a[0], &a[1], &a[2])
            || 3 == _stscanf_s(strInput[i], _T("%d/%d/%d"), &a[0], &a[1], &a[2])
            || 3 == _stscanf_s(strInput[i], _T("%d.%d.%d"), &a[0], &a[1], &a[2])
            || 3 == _stscanf_s(strInput[i], _T("%d,%d,%d"), &a[0], &a[1], &a[2])) {
            ptrQP->qpIntra  = a[0];
            ptrQP->qpInterP = a[1];
            ptrQP->qpInterB = a[2];
            return 0;
        }
        if (1 == _stscanf_s(strInput[i], _T("%d"), &a[0])) {
            ptrQP->qpIntra  = a[0];
            ptrQP->qpInterP = a[0];
            ptrQP->qpInterB = a[0];
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("gop-len")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->encConfig.gopLength = value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("bframes")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->encConfig.frameIntervalP = value + 1;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("vbv-bufsize")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->encConfig.rcParams.vbvBufferSize = value * 1000;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("aq")) {
        pParams->encConfig.rcParams.enableAQ = 1;
        return 0;
    }
    if (IS_OPTION("disable-aq")) {
        pParams->encConfig.rcParams.enableAQ = 0;
        return 0;
    }
    if (IS_OPTION("ref")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            codecPrm[NV_ENC_H264].h264Config.maxNumRefFrames = value;
            codecPrm[NV_ENC_HEVC].hevcConfig.maxNumRefFramesInDPB = value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("mv-precision")) {
        i++;
        int value = 0;
        if (get_list_value(list_mv_presicion, strInput[i], &value)) {
            pParams->encConfig.mvPrecision = (NV_ENC_MV_PRECISION)value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("vbv-bufsize")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->encConfig.rcParams.vbvBufferSize = value * 1000;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
    }  else if (IS_OPTION("interlaced")) {
        i++;
        int value = 0;
        if (get_list_value(list_interlaced, strInput[i], &value)) {
            pParams->picStruct = (NV_ENC_PIC_STRUCT)value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
        pParams->yuv444 = TRUE;
        return 0;
    }
    if (IS_OPTION("no-deblock")) {
        codecPrm[NV_ENC_H264].h264Config.disableDeblockingFilterIDC = 1;
        return 0;
    }
    if (IS_OPTION("deblock")) {
        codecPrm[NV_ENC_H264].h264Config.disableDeblockingFilterIDC = 0;
        return 0;
    }
    if (IS_OPTION("fullrange")) {
        codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.videoFullRangeFlag = 1;
        codecPrm[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.videoFullRangeFlag = 1;
        return 0;
    }
    if (IS_OPTION("videoformat")) {
        i++;
        int value = 0;
        if (get_list_value(list_videoformat, strInput[i], &value)) {
            codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.videoFormat = value;
            codecPrm[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.videoFormat = value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("colormatrix")) {
        i++;
        int value = 0;
        if (get_list_value(list_colormatrix, strInput[i], &value)) {
            codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.colourMatrix = value;
            codecPrm[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.colourMatrix = value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("colorprim")) {
        i++;
        int value = 0;
        if (get_list_value(list_colorprim, strInput[i], &value)) {
            codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.colourPrimaries = value;
            codecPrm[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.colourPrimaries = value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("transfer")) {
        i++;
        int value = 0;
        if (get_list_value(list_transfer, strInput[i], &value)) {
            codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.transferCharacteristics = value;
            codecPrm[NV_ENC_HEVC].hevcConfig.hevcVUIParameters.transferCharacteristics = value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("level")) {
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
                        }
                    }
                }
            }
            return bParsed;
        };
        bool flag = false;
        int value = 0;
        if (getLevel(list_avc_level, strInput[i], &value)) {
            codecPrm[NV_ENC_H264].h264Config.level = value;
            flag = true;
        }
        if (getLevel(list_hevc_level, strInput[i], &value)) {
            codecPrm[NV_ENC_HEVC].hevcConfig.level = value;
            flag = true;
        }
        if (!flag) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("profile")) {
        i++;
        bool flag = false;
        GUID zero = { 0 };
        GUID result_guid = get_guid_from_name(strInput[i], h264_profile_names);
        if (0 != memcmp(&result_guid, &zero, sizeof(result_guid))) {
            pParams->encConfig.profileGUID = result_guid;
            flag = true;
        }
        int result = get_value_from_name(strInput[i], h265_profile_names);
        if (-1 != result) {
            codecPrm[NV_ENC_HEVC].hevcConfig.tier = result;
            flag = true;
        }
        if (!flag) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        if (0 == memcmp(&pParams->encConfig.profileGUID, &NV_ENC_H264_PROFILE_HIGH_444_GUID, sizeof(result_guid))) {
            pParams->yuv444 = TRUE;
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("cu-min")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            codecPrm[NV_ENC_HEVC].hevcConfig.minCUSize = (NV_ENC_HEVC_CUSIZE)value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("log-framelist"))) {
        i++;
        pParams->sFramePosListLog = strInput[i];
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("output-buf"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < 0) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return 1;
        }
        pParams->nOutputBufSizeMB = (std::min)(value, NV_OUTPUT_BUF_MB_MAX);
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("input-thread"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < -1 || value >= 2) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < -1 || value >= 2) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return 1;
        }
        pParams->nOutputThread = (int8_t)value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-thread"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < -1 || value >= 3) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return 1;
        }
        pParams->nAudioThread = (int8_t)value;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("max-procfps"))) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < 0) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return 1;
        }
        pParams->nProcSpeedLimit = (uint16_t)(std::min)(value, (int)UINT16_MAX);
        return 0;
    }
    _ftprintf(stderr, _T("Invalid option: %s.\n"), option_name);
    return -1;
}

int parse_cmd(InEncodeVideoParam *pParams, NV_ENC_CODEC_CONFIG *codecPrm, int nArgNum, const TCHAR **strInput) {

    if (nArgNum == 1) {
        show_help();
        return 1;
    }
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
                if (nullptr == (option_name = short_opt_to_long(strInput[i][1]))) {
                    PrintHelp(strInput[0], strsprintf(_T("Unknown options: \"%s\""), strInput[i]).c_str(), NULL);
                    return -1;
                }
            } else {
                PrintHelp(strInput[0], strsprintf(_T("Invalid options: \"%s\""), strInput[i]).c_str(), NULL);
                return -1;
            }
        }

        if (nullptr == option_name) {
            PrintHelp(strInput[0], strsprintf(_T("Unknown option: \"%s\""), strInput[i]).c_str(), NULL);
            return -1;
        }

        if (IS_OPTION("help")) {
            show_help();
            return 1;
        }
        if (IS_OPTION("help-ja")) {
            show_help_ja();
            return 1;
        }
        if (IS_OPTION("help-en")) {
            show_help_en();
            return 1;
        }
        if (IS_OPTION("version")) {
            show_version();
            return 1;
        }
        if (IS_OPTION("check-device")) {
            show_device_list();
            return 1;
        }
        if (IS_OPTION("check-hw")) {
            int deviceid = 0;
            if (i + 1 < nArgNum) {
                i++;
                int value = 0;
                if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
                    deviceid = value;
                }
            }
            show_hw(deviceid);
            return 1;
        }
        if (IS_OPTION("check-environment")) {
            show_environment_info();
            return 1;
        }
        if (IS_OPTION("check-features")) {
            int deviceid = 0;
            if (i + 1 < nArgNum) {
                i++;
                int value = 0;
                if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
                    deviceid = value;
                }
            }
            show_nvenc_features(deviceid);
            return 1;
        }
#if ENABLE_AVCUVID_READER
        if (0 == _tcscmp(option_name, _T("check-avversion"))) {
            _ftprintf(stdout, _T("%s\n"), getAVVersions().c_str());
            return 1;
        }
        if (0 == _tcscmp(option_name, _T("check-codecs"))) {
            _ftprintf(stdout, _T("%s\n"), getAVCodecs((AVQSVCodecType)(AVQSV_CODEC_DEC | AVQSV_CODEC_ENC)).c_str());
            return 1;
        }
        if (0 == _tcscmp(option_name, _T("check-encoders"))) {
            _ftprintf(stdout, _T("%s\n"), getAVCodecs(AVQSV_CODEC_ENC).c_str());
            return 1;
        }
        if (0 == _tcscmp(option_name, _T("check-decoders"))) {
            _ftprintf(stdout, _T("%s\n"), getAVCodecs(AVQSV_CODEC_DEC).c_str());
            return 1;
        }
        if (0 == _tcscmp(option_name, _T("check-protocols"))) {
            _ftprintf(stdout, _T("%s\n"), getAVProtocols().c_str());
            return 1;
        }
        if (0 == _tcscmp(option_name, _T("check-formats"))) {
            _ftprintf(stdout, _T("%s\n"), getAVFormats((AVQSVFormatType)(AVQSV_FORMAT_DEMUX | AVQSV_FORMAT_MUX)).c_str());
            return 1;
        }
#endif //#if ENABLE_AVCUVID_READER
        auto sts = parse_one_option(option_name, strInput, i, nArgNum, pParams, codecPrm, &argsData);
        if (sts != 0) {
            return sts;
        }
    }

#undef IS_OPTION
    //オプションチェック
    if (pParams->input.filename == nullptr || _tcslen(pParams->input.filename) == 0) {
        _ftprintf(stderr, _T("Input file is not specified.\n"));
        return -1;
    }
    if (0 == pParams->outputFilename.length()) {
        _ftprintf(stderr, _T("Output file is not specified.\n"));
        return -1;
    }

    return 0;
}

int _tmain(int argc, TCHAR **argv) {
    if (check_locale_is_ja())
        _tsetlocale(LC_ALL, _T("japanese"));

    InEncodeVideoParam encPrm = { 0 };
    NV_ENC_CODEC_CONFIG codecPrm[2] = { 0 };
    encPrm.encConfig = NVEncCore::DefaultParam();
    encPrm.inputBuffer = 16;
    encPrm.picStruct = NV_ENC_PIC_STRUCT_FRAME;
    encPrm.preset = 0;
    codecPrm[NV_ENC_H264] = NVEncCore::DefaultParamH264();
    codecPrm[NV_ENC_HEVC] = NVEncCore::DefaultParamHEVC();

    vector<const TCHAR *> argvCopy(argv, argv + argc);
    argvCopy.push_back(_T(""));
    if (parse_cmd(&encPrm, codecPrm, argc, argvCopy.data())) {
        return 1;
    }

    encPrm.encConfig.encodeCodecConfig = codecPrm[encPrm.codec];

    int ret = 1;

    NVEncCore nvEnc;
    if (   NV_ENC_SUCCESS == nvEnc.Initialize(&encPrm)
        && NV_ENC_SUCCESS == nvEnc.InitEncode(&encPrm)) {
        nvEnc.PrintEncodingParamsInfo(NV_LOG_INFO);
        ret = (NV_ENC_SUCCESS == nvEnc.Encode()) ? 0 : 1;
    }

    return ret;
}
