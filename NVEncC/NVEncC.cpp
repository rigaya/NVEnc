//  -----------------------------------------------------------------------------------------
//    NVEnc by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------

#include <Windows.h>
#include <locale.h>
#include <tchar.h>
#include <locale.h>
#include <algorithm>
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
#ifdef _M_IX86
    _ftprintf(stdout, _T("NVEncC (x86) %s by rigaya, build %s %s\n"), VER_STR_FILEVERSION_TCHAR, _T(__DATE__), _T(__TIME__));
#else
    _ftprintf(stdout, _T("NVEncC (x64) %s by rigaya, build %s %s\n"), VER_STR_FILEVERSION_TCHAR, _T(__DATE__), _T(__TIME__));
#endif
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
        _T("   --check-hw                     NVEncが使用可能かチェック\n")
        _T("   --check-features               NVEncの使用可能な機能を表示\n")
        _T("   --check-environment            NVEncの認識している環境情報を表示\n")
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
        _T("\n")
        _T("   --input-res <int>x<int>        入力解像度\n")
        _T("   --crop <int>,<int>,<int>,<int> 左、上、右、下の切り落とし画素数\n")
        _T("   --output-res <int>x<int>       出力解像度\n")
        _T("-f,--fps <int>/<int> or <float>   フレームレートの指定\n")
        _T("\n")
        _T("-c,--codec <string>               出力コーデックの指定\n")
        _T("                                    h264 (or avc), h265 (or hevc)\n")
        _T("   --profile <string>             プロファイルの指定\n")
        _T("                                    H.264: baseline, main, high(デフォルト)\n")
        _T("                                    HEVC : main\n")
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
        _T("   --max-bitrate <int>            最大ビットレート(kbps) / デフォルト: %d kbps\n")
        _T("   --gop-len <int>                GOPのフレーム数 / デフォルト: %d frames%s\n")
        _T("-b,--bframes <int>                連続Bフレーム数 / デフォルト %d フレーム\n")
        _T("   --ref <int>                    参照距離 / デフォルト %d フレーム\n")
        _T("   --aq                           適応的量子化(AQ)を有効にする\n")
        _T("   --mv-precision <string>        動きベクトル精度 / デフォルト: Q-pel\n")
        _T("                                    Q-pel    … 1/4画素精度 (高精度)\n")
        _T("                                    half-pel … 1/2画素精度\n")
        _T("                                    full-pel … 1  画素精度 (低精度)\n")
        _T("   --vbv-bufsize <int>            VBVバッファサイズ (kbit) / デフォルト 自動\n")
        _T("\n")
        _T("H.264/AVC\n")
        _T("   --interlaced <string>          インタレ保持エンコ\n")
        _T("                                    tff, bff\n")
        _T("   --cabac                        CABACを使用する\n")
        _T("   --cavlc                        CAVLCを使用する\n")
        _T("   --bluray                       Bluray用出力を行う / デフォルト: オフ\n")
        _T("   --lossless                     ロスレス出力を行う / デフォルト: オフ\n")
        _T("   --deblock                      デブロックフィルタを有効にする\n")
        _T("   --no-deblock                   デブロックフィルタを無効にする\n")
        _T("   --fullrange                    fullrangeの指定\n")
        _T("   --log <string>                 ログファイル名の指定\n")
        _T("   --log-level <string>           ログレベルの指定 / デフォルト: info\n")
        _T("                                    debug, info, warn, error\n"),
        (AVI_READER) ? _T("avi, ") : _T(""),
        (AVS_READER) ? _T("avs, ") : _T(""),
        DEFAUTL_QP_I, DEFAULT_QP_P, DEFAULT_QP_B,
        DEFAULT_AVG_BITRATE / 1000, DEFAULT_MAX_BITRATE / 1000,
        DEFAULT_GOP_LENGTH, (DEFAULT_GOP_LENGTH == 0) ? _T(" (自動)") : _T(""),
        DEFAULT_B_FRAMES, DEFAULT_REF_FRAMES);

        print_list_options(stdout, _T("--videoformat <string>"), list_videoformat, 0);
        print_list_options(stdout, _T("--colormatrix <string>"), list_colormatrix, 0);
        print_list_options(stdout, _T("--colorprim <string>"),   list_colorprim,   0);
        print_list_options(stdout, _T("--transfer <string>"),    list_transfer,    0);

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
        _T("   --check-hw                     Check for NVEnc\n")
        _T("   --check-features               Check for NVEnc Features\n")
        _T("   --check-environment            Check for Environment Info\n")
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
        _T("\n")
        _T("   --input-res <int>x<int>        set input resolution\n")
        _T("   --crop <int>,<int>,<int>,<int> crop pixels from left,top,right,bottom\n")
        _T("-f,--fps <int>/<int> or <float>   set framerate\n")
        _T("\n")
        _T("-c,--codec <string>               set ouput codec\n")
        _T("                                    h264 (or avc), h265 (or hevc)\n")
        _T("   --profile <string>             set codec profile\n")
        _T("                                    H.264: baseline, main, high(default)\n")
        _T("                                    HEVC : main\n")
        _T("   --level <string>               set codec level\n")
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
        _T("   --max-bitrate <int>            set Max Bitrate (kbps) / Default: %d kbps\n")
        _T("   --gop-len <int>                set GOP Length / Default: %d frames%s")
        _T("-b,--bframes <int>                set B frames / Default %d frames\n")
        _T("   --ref <int>                    set Ref frames / Default %d frames\n")
        _T("   --aq                           enable adaptive quantization\n")
        _T("   --mv-precision <string>        set MV Precision / Default: Q-pel\n")
        _T("                                    Q-pel    (High Quality)\n")
        _T("                                    half-pel\n")
        _T("                                    full-pel (Low Quality)n")
        _T("   --vbv-bufsize <int>            set vbv buffer size (kbit) / Default: auto\n")
        _T("\n")
        _T("H.264/AVC\n")
        _T("   --interlaced <string>          interlaced encoding\n")
        _T("                                    tff, bff\n")
        _T("   --cabac                        use CABAC\n")
        _T("   --cavlc                        use CAVLC (no CABAC)\n")
        _T("   --bluray                       for bluray / Default: off\n")
        _T("   --lossless                     for lossless / Default: off\n")
        _T("   --(no-)deblock                 enable(disable) deblock filter\n")
        _T("   --fullrange                    set fullrange\n")
        _T("   --log <string>                 set log file name\n")
        _T("   --log-level <string>           set log level\n")
        _T("                                    debug, info(default), warn, error\n"),
        (AVI_READER) ? _T("avi, ") : _T(""),
        (AVS_READER) ? _T("avs, ") : _T(""),
        DEFAUTL_QP_I, DEFAULT_QP_P, DEFAULT_QP_B,
        DEFAULT_AVG_BITRATE / 1000, DEFAULT_MAX_BITRATE / 1000,
        DEFAULT_GOP_LENGTH, (DEFAULT_GOP_LENGTH == 0) ? _T(" (auto)") : _T(""),
        DEFAULT_B_FRAMES, DEFAULT_REF_FRAMES);

        print_list_options(stdout, _T("--videoformat <string>"), list_videoformat, 0);
        print_list_options(stdout, _T("--colormatrix <string>"), list_colormatrix, 0);
        print_list_options(stdout, _T("--colorprim <string>"),   list_colorprim,   0);
        print_list_options(stdout, _T("--transfer <string>"),    list_transfer,    0);

        _ftprintf(stdout, _T("\n")
            _T("H.265/HEVC\n")
            _T("   --cu-max <int>                 set max CU size\n")
            _T("   --cu-min  <int>                set min CU size\n")
            _T("                                    8, 16, 32 are avaliable"));
}

static void show_help() {
    (check_locale_is_ja()) ? show_help_ja() : show_help_en();
}

static void show_hw() {
    show_version();
    
    NVEncFeature nvFeature;
    nvFeature.createCacheAsync(0);
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

static void show_nvenc_features() {
    NVEncFeature nvFeature;
    if (nvFeature.createCacheAsync(0)) {
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

static inline BOOL check_range(int value, int min, int max) {
    return (min <= value && value <= max);
}

int parse_cmd(InEncodeVideoParam *conf_set, NV_ENC_CODEC_CONFIG *codecPrm, int argc, TCHAR **argv) {

    auto invalid_option_value = [](const TCHAR *option_name, const TCHAR *option_value) {
        _ftprintf(stderr, _T("Invalid value. %s : %s\n"), option_name, option_value);
    };

    if (argc == 1) {
        show_help();
        return 1;
    }

    for (int i_arg = 1; i_arg < argc; i_arg++) {
        TCHAR *option_name = nullptr;
        if (argv[i_arg][0] == _T('-')) {
            switch (argv[i_arg][1]) {
                case _T('-'):
                    option_name = &argv[i_arg][2];
                    break;
                case _T('b'):
                    option_name = _T("bframes");
                    break;
                case _T('c'):
                    option_name = _T("codec");
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
                    _ftprintf(stderr, _T("Unknown Option : %s"), argv[i_arg]);
                    return -1;
            }
        }

        if (nullptr == option_name) {
            _ftprintf(stderr, _T("Unknown Option : %s"), argv[i_arg]);
            return -1;
        }

        auto get_list_value = [](const CX_DESC * list, const TCHAR *chr, int *value) {
            for (int i = 0; list[i].desc; i++) {
                if (0 == _tcsicmp(list[i].desc, chr)) {
                    *value = list[i].value;
                    return true;
                }
            }
            return false;
        };
        auto get_list_guid_value = [](const guid_desc * list, const TCHAR *chr, int *value) {
            for (int i = 0; list[i].desc; i++) {
                if (0 == _tcsicmp(list[i].desc, chr)) {
                    *value = list[i].value;
                    return true;
                }
            }
            return false;
        };

        
#define IS_OPTION(x) (0 == _tcscmp(option_name, _T(x)))
        if (IS_OPTION("help")) {
            show_help();
            return 1;
        } else if (IS_OPTION("help-ja")) {
            show_help_ja();
            return 1;
        } else if (IS_OPTION("help-en")) {
            show_help_en();
            return 1;
        } else if (IS_OPTION("version")) {
            show_version();
            return 1;
        } else if (IS_OPTION("check-hw")) {
            show_hw();
            return 1;
        } else if (IS_OPTION("check-environment")) {
            show_environment_info();
            return 1;
        } else if (IS_OPTION("check-features")) {
            show_nvenc_features();
            return 1;
        } else if (IS_OPTION("input")) {
            i_arg++;
            auto length = _tcslen(argv[i_arg]) + 1;
            conf_set->input.filename = (TCHAR *)malloc(sizeof(conf_set->input.filename[0]) * length);
            memcpy(conf_set->input.filename, argv[i_arg], sizeof(conf_set->input.filename[0]) * length);
        } else if (IS_OPTION("output")) {
            i_arg++;
            conf_set->outputFilename = argv[i_arg];
        } else if (IS_OPTION("fps")) {
            i_arg++;
            int a[2] = { 0 };
            if (   2 == _stscanf_s(argv[i_arg], _T("%d/%d"), &a[0], &a[1])
                || 2 == _stscanf_s(argv[i_arg], _T("%d:%d"), &a[0], &a[1])
                || 2 == _stscanf_s(argv[i_arg], _T("%d,%d"), &a[0], &a[1])) {
                conf_set->input.rate  = a[0];
                conf_set->input.scale = a[1];
            } else {
                double d;
                if (1 == _stscanf_s(argv[i_arg], _T("%lf"), &d)) {
                    int rate = (int)(d * 1001.0 + 0.5);
                    if (rate % 1000 == 0) {
                        conf_set->input.rate = rate;
                        conf_set->input.scale = 1001;
                    } else {
                        conf_set->input.scale = 100000;
                        conf_set->input.rate = (int)(d * conf_set->input.scale + 0.5);
                        int gcd = nv_get_gcd(conf_set->input.rate, conf_set->input.scale);
                        conf_set->input.scale /= gcd;
                        conf_set->input.rate  /= gcd;
                    }
                } else  {
                    invalid_option_value(option_name, argv[i_arg]);
                    return -1;
                }
            }
        } else if (IS_OPTION("input-res")) {
            i_arg++;
            int a[2] = { 0 };
            if (   2 == _stscanf_s(argv[i_arg], _T("%dx%d"), &a[0], &a[1])
                || 2 == _stscanf_s(argv[i_arg], _T("%d:%d"), &a[0], &a[1])
                || 2 == _stscanf_s(argv[i_arg], _T("%d,%d"), &a[0], &a[1])) {
                conf_set->input.width  = a[0];
                conf_set->input.height = a[1];
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("output-res")) {
            i_arg++;
            int a[2] = { 0 };
            if (   2 == _stscanf_s(argv[i_arg], _T("%dx%d"), &a[0], &a[1])
                || 2 == _stscanf_s(argv[i_arg], _T("%d:%d"), &a[0], &a[1])
                || 2 == _stscanf_s(argv[i_arg], _T("%d,%d"), &a[0], &a[1])) {
                conf_set->input.dstWidth  = a[0];
                conf_set->input.dstHeight = a[1];
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("crop")) {
            i_arg++;
            sInputCrop a = { 0 };
            if (   4 == _stscanf_s(argv[i_arg], _T("%d,%d,%d,%d"), &a.c[0], &a.c[1], &a.c[2], &a.c[3])
                || 4 == _stscanf_s(argv[i_arg], _T("%d:%d:%d:%d"), &a.c[0], &a.c[1], &a.c[2], &a.c[3])) {
                memcpy(&conf_set->input.crop, &a, sizeof(a));
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("codec")) {
            i_arg++;
            int value = 0;
            if (get_list_value(list_nvenc_codecs_for_opt, argv[i_arg], &value)) {
                conf_set->codec = value;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("raw")) {
            conf_set->input.type = NV_ENC_INPUT_RAW;
        } else if (IS_OPTION("y4m")) {
            conf_set->input.type = NV_ENC_INPUT_Y4M;
#if AVI_READER
        } else if (IS_OPTION("avi")) {
            conf_set->input.type = NV_ENC_INPUT_AVI;
#endif
#if AVS_READER
        } else if (IS_OPTION("avs")) {
            conf_set->input.type = NV_ENC_INPUT_AVS;
#endif
#if VPY_READER
        } else if (IS_OPTION("vpy")) {
            conf_set->input.type = NV_ENC_INPUT_VPY;
        } else if (IS_OPTION("vpy-mt")) {
            conf_set->input.type = NV_ENC_INPUT_VPY_MT;
#endif
#if ENABLE_AVCUVID_READER
        } else if (IS_OPTION("avcuvid")) {
            conf_set->input.type = NV_ENC_INPUT_AVCUVID;
#endif
        } else if (IS_OPTION("cqp")) {
            i_arg++;
            int a[3] = { 0 };
            if (   3 == _stscanf_s(argv[i_arg], _T("%d:%d:%d"), &a[0], &a[1], &a[2])
                || 3 == _stscanf_s(argv[i_arg], _T("%d/%d/%d"), &a[0], &a[1], &a[2])
                || 3 == _stscanf_s(argv[i_arg], _T("%d.%d.%d"), &a[0], &a[1], &a[2])
                || 3 == _stscanf_s(argv[i_arg], _T("%d,%d,%d"), &a[0], &a[1], &a[2])) {
                conf_set->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
                conf_set->encConfig.rcParams.constQP.qpIntra  = a[0];
                conf_set->encConfig.rcParams.constQP.qpInterP = a[1];
                conf_set->encConfig.rcParams.constQP.qpInterB = a[2];
            } else if (1 == _stscanf_s(argv[i_arg], _T("%d"), &a[0])) {
                conf_set->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
                conf_set->encConfig.rcParams.constQP.qpIntra  = a[0];
                conf_set->encConfig.rcParams.constQP.qpInterP = a[0];
                conf_set->encConfig.rcParams.constQP.qpInterB = a[0];
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("vbr")) {
            i_arg++;
            int value = 0;
            if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
                conf_set->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
                conf_set->encConfig.rcParams.averageBitRate = value * 1000;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("vbr2")) {
            i_arg++;
            int value = 0;
            if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
                conf_set->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_2_PASS_VBR;
                conf_set->encConfig.rcParams.averageBitRate = value * 1000;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("cbr")) {
            i_arg++;
            int value = 0;
            if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
                conf_set->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
                conf_set->encConfig.rcParams.averageBitRate = value * 1000;
                conf_set->encConfig.rcParams.maxBitRate = value * 1000;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("gop-len")) {
            i_arg++;
            int value = 0;
            if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
                conf_set->encConfig.gopLength = value;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("bframes")) {
            i_arg++;
            int value = 0;
            if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
                conf_set->encConfig.frameIntervalP = value + 1;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("max-bitrate")) {
            i_arg++;
            int value = 0;
            if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
                conf_set->encConfig.rcParams.maxBitRate = value * 1000;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("vbv-bufsize")) {
            i_arg++;
            int value = 0;
            if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
                conf_set->encConfig.rcParams.vbvBufferSize = value * 1000;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("aq")) {
            conf_set->encConfig.rcParams.enableAQ = 1;
        } else if (IS_OPTION("disable-aq")) {
            conf_set->encConfig.rcParams.enableAQ = 0;
        } else if (IS_OPTION("ref")) {
            i_arg++;
            int value = 0;
            if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
                codecPrm[NV_ENC_H264].h264Config.maxNumRefFrames = value;
                codecPrm[NV_ENC_HEVC].hevcConfig.maxNumRefFramesInDPB = value;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("mv-precision")) {
            i_arg++;
            int value = 0;
            if (get_list_value(list_mv_presicion, argv[i_arg], &value)) {
                conf_set->encConfig.mvPrecision = (NV_ENC_MV_PRECISION)value;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("vbv-bufsize")) {
            i_arg++;
            int value = 0;
            if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
                conf_set->encConfig.rcParams.vbvBufferSize = value * 1000;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("interlaced")) {
            i_arg++;
            int value = 0;
            if (get_list_value(list_interlaced, argv[i_arg], &value)) {
                conf_set->picStruct = (NV_ENC_PIC_STRUCT)value;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("cavlc")) {
            codecPrm[NV_ENC_H264].h264Config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
        } else if (IS_OPTION("cabac")) {
            codecPrm[NV_ENC_H264].h264Config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_CABAC;
        } else if (IS_OPTION("bluray")) {
            conf_set->bluray = TRUE;
        } else if (IS_OPTION("lossless")) {
            conf_set->lossless = TRUE;
            conf_set->yuv444 = TRUE;
        } else if (IS_OPTION("no-deblock")) {
            codecPrm[NV_ENC_H264].h264Config.disableDeblockingFilterIDC = 1;
        } else if (IS_OPTION("deblock")) {
            codecPrm[NV_ENC_H264].h264Config.disableDeblockingFilterIDC = 0;
        } else if (IS_OPTION("fullrange")) {
            codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.videoFullRangeFlag = 1;
        } else if (IS_OPTION("videoformat")) {
            i_arg++;
            int value = 0;
            if (get_list_value(list_videoformat, argv[i_arg], &value)) {
                codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.videoFormat = value;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("colormatrix")) {
            i_arg++;
            int value = 0;
            if (get_list_value(list_colormatrix, argv[i_arg], &value)) {
                codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.colourMatrix = value;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("colorprim")) {
            i_arg++;
            int value = 0;
            if (get_list_value(list_colorprim, argv[i_arg], &value)) {
                codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.colourPrimaries = value;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("transfer")) {
            i_arg++;
            int value = 0;
            if (get_list_value(list_transfer, argv[i_arg], &value)) {
                codecPrm[NV_ENC_H264].h264Config.h264VUIParameters.transferCharacteristics = value;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("level")) {
            i_arg++;
            bool flag = false;
            int value = 0;
            if (get_list_value(list_avc_level, argv[i_arg], &value)) {
                codecPrm[NV_ENC_H264].h264Config.level = value;
                flag = true;
            }
            if (get_list_value(list_hevc_level, argv[i_arg], &value)) {
                codecPrm[NV_ENC_HEVC].hevcConfig.level = value;
                flag = true;
            }
            if (!flag) {
                if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
                    codecPrm[NV_ENC_H264].h264Config.level = value;
                    codecPrm[NV_ENC_HEVC].hevcConfig.level = value;
                } else {
                    invalid_option_value(option_name, argv[i_arg]);
                    return -1;
                }
            }
        } else if (IS_OPTION("profile")) {
            i_arg++;
            bool flag = false;
            GUID zero = { 0 };
            GUID result_guid = get_guid_from_name(argv[i_arg], h264_profile_names);
            if (0 != memcmp(&result_guid, &zero, sizeof(result_guid))) {
                conf_set->encConfig.profileGUID = result_guid;
                flag = true;
            }
            int result = get_value_from_name(argv[i_arg], h265_profile_names);
            if (-1 != result) {
                codecPrm[NV_ENC_HEVC].hevcConfig.tier = result;
                flag = true;
            }
            if (!flag) {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
            if (0 == memcmp(&conf_set->encConfig.profileGUID, &NV_ENC_H264_PROFILE_HIGH_444_GUID, sizeof(result_guid))) {
                conf_set->yuv444 = TRUE;
            }
        } else if (IS_OPTION("sar") || IS_OPTION("par") || IS_OPTION("dar")) {
            i_arg++;
            int a[2] = { 0 };
            if (   2 == _stscanf_s(argv[i_arg], _T("%d:%d"), &a[0], &a[1])
                || 2 == _stscanf_s(argv[i_arg], _T("%d/%d"), &a[0], &a[1])
                || 2 == _stscanf_s(argv[i_arg], _T("%d.%d"), &a[0], &a[1])
                || 2 == _stscanf_s(argv[i_arg], _T("%d,%d"), &a[0], &a[1])) {
                if (IS_OPTION("dar")) {
                    a[0] = -a[0];
                    a[1] = -a[1];
                }
                conf_set->par[0] = a[0];
                conf_set->par[1] = a[1];
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("cu-max")) {
            i_arg++;
            int value = 0;
            if (get_list_value(list_hevc_cu_size, argv[i_arg], &value)) {
                codecPrm[NV_ENC_HEVC].hevcConfig.maxCUSize = (NV_ENC_HEVC_CUSIZE)value;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("cu-min")) {
            i_arg++;
            int value = 0;
            if (1 == _stscanf_s(argv[i_arg], _T("%d"), &value)) {
                codecPrm[NV_ENC_HEVC].hevcConfig.minCUSize = (NV_ENC_HEVC_CUSIZE)value;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        } else if (IS_OPTION("log")) {
            i_arg++;
            conf_set->logfile = argv[i_arg];
        } else if (IS_OPTION("log-level")) {
            i_arg++;
            int value = 0;
            if (get_list_value(list_log_level, argv[i_arg], &value)) {
                conf_set->loglevel = value;
            } else {
                invalid_option_value(option_name, argv[i_arg]);
                return -1;
            }
        }
    }

#undef IS_OPTION
    //オプションチェック
    if (conf_set->input.filename == nullptr || _tcslen(conf_set->input.filename) == 0) {
        _ftprintf(stderr, _T("Input file is not specified.\n"));
        return -1;
    }
    if (0 == conf_set->outputFilename.length()) {
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

    if (parse_cmd(&encPrm, codecPrm, argc, argv)) {
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
