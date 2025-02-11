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

#include <set>
#include <sstream>
#include <numeric>
#include <iomanip>
#include "rgy_version.h"
#include "NVEncParam.h"
#include "NVEncUtil.h"
#include "NVEncCmd.h"
#include "NVEncFilterAfs.h"
#include "rgy_osdep.h"
#include "rgy_perf_monitor.h"
#include "rgy_avutil.h"

tstring GetNVEncVersion() {
    static const TCHAR *const ENABLED_INFO[] = { _T("no"), _T("yes") };
    tstring version;
    version += get_encoder_version();
    version += _T("\n");
    version += strsprintf(_T("  [NVENC API v%d.%d, CUDA %d.%d]\n"),
        NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION,
        CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    version += _T(" reader: raw, y4m");
    if (ENABLE_AVI_READER) version += _T(", avi");
    if (ENABLE_AVISYNTH_READER) version += _T(", avs");
    if (ENABLE_VAPOURSYNTH_READER) version += _T(", vpy");
#if ENABLE_AVSW_READER
    version += _T(", avsw");
    version += strsprintf(_T(", avhw [%s]"), getHWDecSupportedCodecList().c_str());
#endif //#if ENABLE_AVSW_READER
    version += _T("\n");
    version += _T(" others\n");
    version += strsprintf(_T("  nvml       : %s\n"), ENABLED_INFO[ENABLE_NVML]);
    version += strsprintf(_T("  nvrtc      : %s\n"), ENABLED_INFO[ENABLE_NVRTC]);
    version += strsprintf(_T("  nvvfx      : %s\n"), ENABLED_INFO[ENABLE_NVVFX]);
    version += strsprintf(_T("  ngx        : %s\n"), ENABLED_INFO[ENABLE_NVSDKNGX]);
    version += strsprintf(_T("  nvof fruc  : %s\n"), ENABLED_INFO[ENABLE_NVOFFRUC]);
    version += strsprintf(_T("  libass     : %s\n"), ENABLED_INFO[ENABLE_LIBASS_SUBBURN]);
    version += strsprintf(_T("  libdovi    : %s\n"), ENABLED_INFO[ENABLE_LIBDOVI]);
    version += strsprintf(_T("  d3d11      : %s\n"), ENABLED_INFO[ENABLE_D3D11]);
    version += strsprintf(_T("  vulkan     : %s\n"), ENABLED_INFO[ENABLE_VULKAN]);
    version += strsprintf(_T("  libplacebo : %s\n"), ENABLED_INFO[ENABLE_LIBPLACEBO]);
    version += strsprintf(_T("  vmaf       : %s\n"), ENABLED_INFO[ENABLE_VMAF]);
    return version;
}

typedef struct ListData {
    const TCHAR *name;
    const CX_DESC *list;
    int default_index;
} ListData;

static tstring PrintMultipleListOptions(const TCHAR *option_name, const TCHAR *option_desc, const vector<ListData>& listDatas) {
    tstring str;
    const TCHAR *indent_space = _T("                                ");
    const int indent_len = (int)_tcslen(indent_space);
    const int max_len = 79;
    str += strsprintf(_T("   %s "), option_name);
    while ((int)str.length() < indent_len) {
        str += _T(" ");
    }
    str += strsprintf(_T("%s\n"), option_desc);
    const auto data_name_max_len = indent_len + 4 + std::accumulate(listDatas.begin(), listDatas.end(), 0,
        [](const int max_len, const ListData data) { return (std::max)(max_len, (int)_tcslen(data.name)); });

    for (const auto& data : listDatas) {
        tstring line = strsprintf(_T("%s- %s: "), indent_space, data.name);
        while ((int)line.length() < data_name_max_len) {
            line += strsprintf(_T(" "));
        }
        for (int i = 0; data.list[i].desc; i++) {
            if (i > 0 && data.list[i].value == data.list[i-1].value) {
                continue; //連続で同じ値を示す文字列があるときは、先頭のみ表示する
            }
            const int desc_len = (int)(_tcslen(data.list[i].desc) + _tcslen(_T(", ")) + ((i == data.default_index) ? _tcslen(_T("(default)")) : 0));
            if (line.length() + desc_len >= max_len) {
                str += line + _T("\n");
                line = indent_space;
                while ((int)line.length() < data_name_max_len) {
                    line += strsprintf(_T(" "));
                }
            } else {
                if (i) {
                    line += strsprintf(_T(", "));
                }
            }
            line += strsprintf(_T("%s%s"), data.list[i].desc, (i == data.default_index) ? _T("(default)") : _T(""));
        }
        str += line + _T("\n");
    }
    return str;
}

tstring encoder_help() {
    tstring str;
    str += strsprintf(_T("Usage: NVEncC.exe [Options] -i <input file> -o <output file>\n"));
    str += strsprintf(_T("\n")
        _T("Input can be %s%sraw YUV, YUV4MPEG2(y4m).\n")
        _T("When Input is in raw format, fps, input-res is required.\n")
        _T("\n")
        _T("Ouput format will be in raw H.264/AVC or H.265/HEVC ES.\n")
        _T("\n")
        _T("Example:\n")
        _T("  NVEncC -i \"<avsfilename>\" -o \"<outfilename>\"\n")
        _T("  avs2pipemod -y4mp \"<avsfile>\" | NVEncC --y4m -i - -o \"<outfilename>\"\n"),
        (ENABLE_AVI_READER) ? _T("avi, ") : _T(""),
        (ENABLE_AVISYNTH_READER) ? _T("avs, ") : _T(""));
    str += strsprintf(_T("\n")
        _T("Information Options: \n")
        _T("-h,-? --help                    print help\n")
        _T("-v,--version                    print version info\n")
        _T("   --check-device               show DeviceId for GPUs available on system\n")
        _T("   --check-hw [<int>]           check NVEnc codecs for specified DeviceId\n")
        _T("                                  if unset, will check DeviceId #0\n")
        _T("   --check-features [<int>]     check for NVEnc Features for specified DeviceId\n")
        _T("                                  if unset, will check DeviceId #0\n")
        _T("   --check-environment          check for Environment Info\n")
#if ENABLE_AVSW_READER
        _T("   --check-avversion            show dll version\n")
        _T("   --check-codecs               show codecs available\n")
        _T("   --check-encoders             show audio encoders available\n")
        _T("   --check-decoders             show audio decoders available\n")
        _T("   --check-profiles <string>    show profile names available for specified codec\n")
        _T("   --check-formats              show in/out formats available\n")
        _T("   --check-protocols            show in/out protocols available\n")
        _T("   --check-avdevices            show in/out avdvices available\n")
        _T("   --check-filters              show filters available\n")
        _T("   --option-list                show option list\n")
#endif
        _T("\n"));
    str += strsprintf(_T("\n")
        _T("Basic Encoding Options: \n")
        _T("-d,--device <int>               set DeviceId used in NVEnc (default:-1 as auto)\n")
        _T("                                  use --check-device to show device ids.\n"));
    str += gen_cmd_help_input();
    str += strsprintf(_T("")
        _T("\n")
        _T("-c,--codec <string>             set output codec\n")
        _T("                                  h264 (or avc), h265 (or hevc), av1, raw\n")
        _T("   --profile <string>           set codec profile\n")
        _T("                                  H.264: baseline, main, high(default), high444\n")
        _T("                                  HEVC : main, main10, main444\n")
        _T("                                  AV1  : main, high\n")
        _T("   --tier <string>              set codec tier\n")
        _T("                                  HEVC : main, high\n")
        _T("                                  AV1  : 0, 1\n")
        _T("   --lossless                   for lossless encoding / default: off\n"));

    str += PrintMultipleListOptions(_T("--level <string>"), _T("set codec level"),
        { { _T("H.264"), list_avc_level,   0 },
          { _T("HEVC"),  list_hevc_level,  0 },
          { _T("AV1"),   list_av1_level,   0 }
    });
    str += strsprintf(_T("")
        _T("   --output-depth <int>         set output bit depth ( 8(default), 10 )\n")
        _T("   --output-csp <string>        set output csp ( yuv420(default), yuv444 )\n")
        _T("   --sar <int>:<int>            set Sample  Aspect Ratio\n")
        _T("   --dar <int>:<int>            set Display Aspect Ratio\n")
        _T("\n")
        _T("   --qvbr <float>               set bitrate for QVBR mode (kbps)\n")
        _T("                                  same as \"--vbr 0 --vbr-quality <float>\"\n")
        _T("   --vbr <int>                  set bitrate for VBR mode (kbps)\n")
        _T("   --cbr <int>                  set bitrate for CBR mode (kbps)\n")
        _T("                                  default: %d kbps\n")
        _T("   --cqp <int> or               encode in Constant QP mode\n")
        _T("         <int>:<int>:<int>        default: <I>:<P>:<B>=<%d>:<%d>:<%d>\n")
        _T("\n")
        _T("-u,--preset <string>            set encoder preset\n")
        _T("                                  default, performance, quality\n")
        _T("\n")
        _T("   --vbr-quality <float>        target quality for VBR mode (0-51, 0=auto)\n")
        _T("   --multipass <string>         multipass mode for VBR, CBR mode\n")
        _T("                                  none, 2pass-quarter, 2pass-full\n")
        _T("   --max-bitrate <int>          set Max Bitrate (kbps)\n")
        _T("\n")
        _T("   --dynamic-rc <int>:<int>,<param1>=<value>[,<param2>=<value>][...]\n")
        _T("     change the rate control mode within the specified range of output frames\n")
        _T("    params\n")
        _T("      cqp=<int> or <int>:<int>:<int>\n")
        _T("      vbr=<int>\n")
        _T("      vbrhq=<int>\n")
        _T("      cbr=<int>\n")
        _T("      cbrhq=<int>\n")
        _T("      qvbr=<float>\n")
        _T("      max-bitrate=<int>\n")
        _T("      vbr-quality=<float>\n")
        _T("\n")
        _T("   --qp-init <int> or           set initial QP\n")
        _T("             <int>:<int>:<int>    default: auto\n")
        _T("   --qp-max <int> or            set max QP\n")
        _T("            <int>:<int>:<int>     default: unset\n")
        _T("   --qp-min <int> or            set min QP\n")
        _T("             <int>:<int>:<int>    default: unset\n")
        _T("   --chroma-qp-offset <int>     set chroma QP Offset\n")
        _T("   --gop-len <int>              set GOP Length / default: %d frames%s\n")
        _T("   --lookahead <int>            enable lookahead and set lookahead depth (1-32)\n")
        _T("                                  default: %d frames\n")
        _T("   --lookahead-level <int>      set lookahead level (0 - 3) [HEVC only]\n")
        _T("                                  default: 0\n")
#if ENABLE_NVENC_SDK_TUNE
        _T("   --tune <string>              set tuning info (default: auto)\n")
        _T("                                  undef, hq, uhq, lowlatency, ultralowlatency, lossless\n")
#endif
        _T("   --strict-gop                 avoid GOP len fluctuation\n")
        _T("   --no-i-adapt                 disable adapt. I frame insertion\n")
        _T("   --no-b-adapt                 disable adapt. B frame insertion\n")
        _T("                                  for lookahead mode only, default: off\n")
        _T("-b,--bframes <int>              set number of consecutive B frames\n")
        _T("                                  default: %d frames\n")
        _T("   --ref <int>                  set ref frames / default %d frames\n")
        _T("   --multiref-l0 <int>          set multiple ref frames (L0)\n")
        _T("   --multiref-l1 <int>          set multiple ref frames (L1)\n")
        _T("   --weightp                    enable weighted prediction for P frame\n")
        _T("   --nonrefp                    enable adapt. non-reference P frame insertion\n")
        _T("   --mv-precision <string>      set MV Precision / default: auto\n")
        _T("                                  auto,\n")
        _T("                                  Q-pel (High Quality),\n")
        _T("                                  half-pel,\n")
        _T("                                  full-pel (Low Quality, not recommended)\n")
        _T("   --slices <int>               number of slices, default 0 (auto)\n")
        _T("   --vbv-bufsize <int>          set vbv buffer size (kbit) / default: auto\n")
        _T("   --(no-)aq                    enable spatial adaptive quantization\n")
        _T("   --aq-temporal                enable temporal adaptive quantization\n")
        _T("   --aq-strength <int>          set aq strength (weak 1 - 15 strong)\n")
        _T("                                  default: 0 = auto\n")
        _T("   --bref-mode <string>         set B frame reference mode\n")
        _T("                                  - auto (default)\n")
        _T("                                  - disabled\n")
        _T("                                  - each\n")
        _T("                                  - middle\n")
        _T("   --tf-level <int>             set temporal filtering level (0 (default), 4)\n")
        _T("                                  requires bframes >= 4\n")
        _T("   --temporal-layers <int>      set number of temporal layers\n")
        _T("   --direct <string>            [H264] set B Direct mode\n")
        _T("                                  auto(default), none, spatial, temporal\n")
        _T("   --(no-)adapt-transform       [H264] set adaptive transform mode (default=auto)\n")
        _T("   --hierarchial-p              [H264] enable hierarchial P frames\n")
        _T("   --hierarchial-b              [H264] enable hierarchial B frames\n"),
        DEFAUTL_QP_I, DEFAULT_QP_P, DEFAULT_QP_B,
        DEFAULT_AVG_BITRATE / 1000,
        DEFAULT_GOP_LENGTH, (DEFAULT_GOP_LENGTH == 0) ? _T(" (auto)") : _T(""),
        DEFAULT_LOOKAHEAD,
        DEFAULT_B_FRAMES, DEFAULT_REF_FRAMES);

    str += strsprintf(_T("\n")
        _T("   --cabac                      [H264] use CABAC\n")
        _T("   --cavlc                      [H264] use CAVLC (no CABAC)\n")
        _T("   --bluray                     [H264] for bluray / default: off\n")
        _T("   --(no-)deblock               [H264] enable(disable) deblock filter\n"));

    str += strsprintf(_T("\n")
        _T("   --cu-max <int>               [HEVC] set max CU size\n")
        _T("   --cu-min  <int>              [HEVC] set min CU size\n")
        _T("                                  8, 16, 32 are avaliable\n")
        _T("    warning: it is not recommended to use --cu-max or --cu-min,\n")
        _T("             leaving it auto will enhance video quality.\n")
        _T("   --alpha-bitrate-ratio  <int> [HEVC] set ratio for alpha bitrate.\n")
        _T("                                       smaller value will use more bitrate for alpha.\n")
        _T("   --alpha-channel-mode <string> [HEVC] set alpha channel mode.\n")
        _T("                                  - straight (default)\n")
        _T("                                  - premultiplied\n"));

    str += strsprintf(_T("\n")
        _T("   --part-size-min <int>        [AV1] min size of luma coding block partition.\n")
        _T("   --part-size-max <int>        [AV1] max size of luma coding block partition.\n")
        _T("                                  0 (auto,default), 4, 8, 16, 32, 64\n")
        _T("   --tile-columns <int>         [AV1] number of tile columns (default:0=auto).\n")
        _T("   --tile-rows <int>            [AV1] number of tile rows (default:0=auto).\n")
        _T("                                  0 (auto,default), 1, 2, 4, 8, 16, 32, 64\n")
        _T("   --refs-forward <int>         [AV1] max number of forward reference frame.\n")
        _T("                                  0 (auto,default), 1, 2, 3, 4\n")
        _T("   --refs-backward <int>        [AV1] max number of L1 list reference frame.\n")
        _T("                                  0 (auto,default), 1, 2, 3\n"));

    str += strsprintf(_T("")
        _T("   --aud                        insert aud nal unit to ouput stream.\n")
        _T("   --repeat-headers             output VPS,SPS and PPS for every IDR frame.\n")
        _T("   --pic-struct                 insert pic-timing SEI with pic_struct.\n"));

    str += strsprintf(_T("")
        _T("   --split-enc <string>         set cuda schedule mode (default: sync).\n")
        _T("                                  - auto (default), auto_forced\n")
        _T("                                  - forced_2, forced_3, forced_4, disable\n"));

    str += _T("\n");
    str += gen_cmd_help_common();
    str += _T("\n");

    str += strsprintf(_T("\n")
        _T("   --vpp-deinterlace <string>   set deinterlace mode / default: none\n")
        _T("                                  none, bob, adaptive (normal)\n")
        _T("                                  available only with avhw reader\n"));
    str += print_list_options(_T("--vpp-gauss <int>"),         list_nppi_gauss,  0);
    if (ENABLE_NVVFX) {
        str += strsprintf(_T("\n")
            _T("   --vpp-nvvfx-denoise [<param1>=<value>][,<param2>=<value>][...]\n")
            _T("     enable nvvfx webcam denoise filter.\n")
            _T("    params\n")
            _T("      strength=<int>        0 ... conservative (default)\n")
            _T("                            1 ... aggressive \n"));
        str += strsprintf(_T("\n")
            _T("   --vpp-nvvfx-artifact-reduction [<param1>=<value>][,<param2>=<value>][...]\n")
            _T("     enable nvvfx artifact reduction filter.\n")
            _T("    params\n")
            _T("      mode=<int>            0 ... conservative (default)\n")
            _T("                            1 ... aggressive \n"));
        str += strsprintf(_T("\n")
            _T("   --vpp-nvvfx-model-dir <string> set directory which has nxxmfx models.\n"));
    }
    if (ENABLE_NVSDKNGX) {
        str += strsprintf(_T("\n")
            _T("   --vpp-ngx-truehdr [<param1>=<value>][,<param2>=<value>][...]\n")
            _T("     enable ngx truehdr filter.\n")
            _T("    params\n")
            _T("      contrast=<int>     the difference between lights and darks (default %d, 0 - 200).\n")
            _T("      saturation=<int>   color intensity (default %d, 0 - 200).\n")
            _T("      middlegray=<int>   average brightness (default %d, 10 - 100).\n")
            _T("      maxluminance=<int> peak brightness in nits (default %d, 400 - 2000).\n"),
            FILTER_DEFAULT_NGX_TRUEHDR_CONTRAST, FILTER_DEFAULT_NGX_TRUEHDR_SATURATION,
            FILTER_DEFAULT_NGX_TRUEHDR_MIDDLE_GRAY, FILTER_DEFAULT_NGX_TRUEHDR_MAX_LUMINANCE);
    }

    str += _T("\n");
    str += gen_cmd_help_vpp();
    str += _T("\n");
    str += strsprintf(_T("")
        _T("   --cuda-schedule <string>     set cuda schedule mode (default: sync).\n")
        _T("       auto  : let cuda driver to decide\n")
        _T("       spin  : CPU will spin when waiting GPU tasks,\n")
        _T("               will provide highest performance but with high CPU utilization.\n")
        _T("       yield : CPU will yield when waiting GPU tasks.\n")
        _T("       sync  : CPU will sleep when waiting GPU tasks, performance might\n")
        _T("                drop slightly, while CPU utilization will be lower,\n")
        _T("                especially on HW decode mode.\n"));
    str += _T("")
        _T("   --disable-nvml <int>        disable NVML GPU monitoring (default 0, 0-2)\n");
        _T("   --disable-dx11              disable DX11 initilization.\n");
        _T("   --disable-vulkan            disable Vulkan initilization.\n");
    str += gen_cmd_help_ctrl();
    return str;
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
        option_name = _T("preset");
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

bool get_list_guid_value(const guid_desc *list, const TCHAR *chr, int *value) {
    for (int i = 0; list[i].desc; i++) {
        if (0 == _tcsicmp(list[i].desc, chr)) {
            *value = list[i].value;
            return true;
        }
    }
    return false;
};

template <size_t size>
void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, const guid_desc(&list)[size]) {
    std::vector<CX_DESC> descList;
    for (size_t i = 0; i < size; i++) {
        CX_DESC x;
        x.desc = list[i].desc;
        x.value = 0;
        descList.push_back(x);
    }
    CX_DESC x;
    x.desc = nullptr;
    x.value = 0;
    descList.push_back(x);
    print_cmd_error_invalid_value(strOptionName, strErrorValue, descList.data());
}

void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, const std::vector<std::pair<RGY_CODEC, std::vector<guid_desc>>>& codec_list) {
    std::vector<std::pair<RGY_CODEC, const CX_DESC *>> cx_codec_list_ptr;
    std::vector<std::pair<RGY_CODEC, std::vector<CX_DESC>> > cx_codec_list_vec;
    for (const auto& codec : codec_list) {
        auto v = std::vector<CX_DESC>();
        for (const auto &guid : codec.second) {
            CX_DESC x;
            x.desc = guid.desc;
            x.value = 0;
            v.push_back(x);
        }
        CX_DESC x;
        x.desc = nullptr;
        x.value = 0;
        v.push_back(x);
        cx_codec_list_vec.push_back(std::make_pair(codec.first, v));
    }
    for (const auto &codec : cx_codec_list_vec) {
        cx_codec_list_ptr.push_back(std::make_pair(codec.first, codec.second.data()));
    }
    print_cmd_error_invalid_value(strOptionName, strErrorValue, cx_codec_list_ptr);
}

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
            print_cmd_error_invalid_value(option_name, strInput[i], _T("device id should be positive value."));
            return 1;
        }
        pParams->deviceID = deviceid;
        return 0;
    }
    if (IS_OPTION("preset")) {
        i++;
        int value = get_value_from_name(strInput[i], list_nvenc_preset_names_ver10);
        if (value >= 0) {
            pParams->preset = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_nvenc_preset_names_ver10);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("codec")) {
        i++;
        int value = 0;
        if (get_list_value(list_nvenc_codecs_for_opt, strInput[i], &value)) {
            pParams->codec_rgy = (RGY_CODEC)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_nvenc_codecs_for_opt);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("cqp")) {
        i++;
        int ret = pParams->rcParam.qp.parse(strInput[i]);
        if (ret != 0) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        pParams->rcParam.rc_mode = NV_ENC_PARAMS_RC_CONSTQP;
        return 0;
    }
    if (IS_OPTION("vbr")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->rcParam.rc_mode = NV_ENC_PARAMS_RC_VBR;
            pParams->rcParam.avg_bitrate = value * 1000;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("vbrhq") || IS_OPTION("vbr2")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->rcParam.rc_mode = NV_ENC_PARAMS_RC_VBR_HQ;
            pParams->rcParam.avg_bitrate = value * 1000;
            pParams->multipass = NV_ENC_TWO_PASS_FULL_RESOLUTION;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("cbr")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->rcParam.rc_mode = NV_ENC_PARAMS_RC_CBR;
            pParams->rcParam.avg_bitrate = value * 1000;
            pParams->rcParam.max_bitrate = value * 1000;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("cbrhq")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->rcParam.rc_mode = NV_ENC_PARAMS_RC_CBR_HQ;
            pParams->rcParam.avg_bitrate = value * 1000;
            pParams->rcParam.max_bitrate = value * 1000;
            pParams->multipass = NV_ENC_TWO_PASS_FULL_RESOLUTION;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("qvbr")) {
        i++;
        double value = 0;
        if (tstring(_T("auto")) == tstring(strInput[i])) {
            pParams->rcParam.targetQuality = -1;
            pParams->rcParam.targetQualityLSB = -1;
            pParams->rcParam.rc_mode = NV_ENC_PARAMS_RC_QVBR;
        } else if (1 == _stscanf_s(strInput[i], _T("%lf"), &value)) {
            value = (std::max)(0.0, value);
            int value_int = (int)value;
            pParams->rcParam.targetQuality = (uint8_t)clamp(value_int, 0, 51);
            pParams->rcParam.targetQualityLSB = (uint8_t)clamp((int)((value - value_int) * 256.0), 0, 255);

            pParams->rcParam.rc_mode = NV_ENC_PARAMS_RC_QVBR;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("vbr-quality")) {
        i++;
        double value = 0;
        if (tstring(_T("auto")) == tstring(strInput[i])) {
            pParams->rcParam.targetQuality = 0;
            pParams->rcParam.targetQualityLSB = 0;
        } else if (1 == _stscanf_s(strInput[i], _T("%lf"), &value)) {
            value = (std::max)(0.0, value);
            int value_int = (int)value;
            pParams->rcParam.targetQuality = (uint8_t)clamp(value_int, 0, 51);
            pParams->rcParam.targetQualityLSB = (uint8_t)clamp((int)((value - value_int) * 256.0), 0, 255);
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("multipass")) {
        i++;
        int value = 0;
        if (get_list_value(list_nvenc_multipass_mode, strInput[i], &value)) {
            pParams->multipass = (NV_ENC_MULTI_PASS)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_nvenc_multipass_mode);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("dynamic-rc")) {
        if (i+1 >= nArgNum || strInput[i+1][0] == _T('-')) {
            return 0;
        }
        i++;
        bool rc_mode_defined = false;
        auto paramList = std::vector<std::string>{ "start", "end", "cqp", "max-bitrate", "vbr-quality", "multipass" };
        for (int j = 0; list_nvenc_rc_method_en[j].desc; j++) {
            paramList.push_back(tolowercase(tchar_to_string(list_nvenc_rc_method_en[j].desc)));
        }
        NVEncRCParam rcPrm;
        for (const auto &param : split(strInput[i], _T(","))) {
            auto pos = param.find_first_of(_T("="));
            if (pos != std::string::npos) {
                auto param_arg = param.substr(0, pos);
                auto param_val = param.substr(pos+1);
                param_arg = tolowercase(param_arg);
                if (param_arg == _T("start")) {
                    try {
                        rcPrm.start = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("end")) {
                    try {
                        rcPrm.end = std::stoi(param_val);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("cqp")) {
                    int ret = rcPrm.qp.parse(param_val.c_str());
                    if (ret != 0) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    rcPrm.rc_mode = NV_ENC_PARAMS_RC_CONSTQP;
                    rc_mode_defined = true;
                    continue;
                }
                int temp = 0;
                if (get_list_value(list_nvenc_rc_method_en, touppercase(param_arg).c_str(), &temp)) {
                    try {
                        rcPrm.avg_bitrate = std::stoi(param_val) * 1000;
                        rcPrm.rc_mode = (NV_ENC_PARAMS_RC_MODE)temp;
                        if (temp == NV_ENC_PARAMS_RC_CBR_HQ || temp == NV_ENC_PARAMS_RC_VBR_HQ) {
                            pParams->multipass = NV_ENC_TWO_PASS_FULL_RESOLUTION;
                        }
                        rc_mode_defined = true;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("max-bitrate")) {
                    try {
                        rcPrm.max_bitrate = std::stoi(param_val) * 1000;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("qvbr")) {
                    try {
                        auto value = (std::max)(0.0, std::stod(param_val));
                        int value_int = (int)value;
                        rcPrm.targetQuality = (uint8_t)clamp(value_int, 0, 51);
                        rcPrm.targetQualityLSB = (uint8_t)clamp((int)((value - value_int) * 256.0), 0, 255);
                        rcPrm.avg_bitrate = 0;
                        rcPrm.rc_mode = NV_ENC_PARAMS_RC_VBR;
                        rc_mode_defined = true;
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("vbr-quality")) {
                    try {
                        auto value = (std::max)(0.0, std::stod(param_val));
                        int value_int = (int)value;
                        rcPrm.targetQuality = (uint8_t)clamp(value_int, 0, 51);
                        rcPrm.targetQualityLSB = (uint8_t)clamp((int)((value - value_int) * 256.0), 0, 255);
                    } catch (...) {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                if (param_arg == _T("multipass")) {
                    int value = 0;
                    if (get_list_value(list_nvenc_multipass_mode, param_val.c_str(), &value)) {
                        pParams->multipass = (NV_ENC_MULTI_PASS)value;
                    } else {
                        print_cmd_error_invalid_value(tstring(option_name) + _T(" ") + param_arg + _T("="), param_val);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param_arg, paramList);
                return 1;
            } else {
                pos = param.find_first_of(_T(":"));
                if (pos != std::string::npos) {
                    auto param_val0 = param.substr(0, pos);
                    auto param_val1 = param.substr(pos+1);
                    try {
                        rcPrm.start = std::stoi(param_val0);
                        rcPrm.end   = std::stoi(param_val1);
                    } catch (...) {
                        print_cmd_error_invalid_value(option_name, param);
                        return 1;
                    }
                    continue;
                }
                print_cmd_error_unknown_opt_param(option_name, param, paramList);
                return 1;
            }
        }
        if (!rc_mode_defined) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("rate control mode unspecified!"));
            return 1;
        }
        if (rcPrm.start < 0) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("start frame ID unspecified!"));
            return 1;
        }
        if (rcPrm.end > 0 && rcPrm.start > rcPrm.end) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("start frame ID must be smaller than end frame ID!"));
            return 1;
        }
        pParams->dynamicRC.push_back(rcPrm);
        return 0;
    }
    if (IS_OPTION("qp-init")) {
        i++;
        int ret = pParams->qpInit.parse(strInput[i]);
        if (ret != 0) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("qp-min")) {
        i++;
        int ret = pParams->qpMin.parse(strInput[i]);
        if (ret != 0) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("qp-max")) {
        i++;
        int ret = pParams->qpMax.parse(strInput[i]);
        if (ret != 0) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("chroma-qp-offset")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->chromaQPOffset = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("gop-len")) {
        i++;
        int value = 0;
        if (0 == _tcsnccmp(strInput[i], _T("auto"), _tcslen(_T("auto")))) {
            pParams->gopLength = 0;
        } else if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->gopLength = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("strict-gop")) {
        pParams->strictGOP = true;
        return 0;
    }
    if (IS_OPTION("bframes")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->bFrames = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("bref-mode")) {
        i++;
        int value = 0;
        if (get_list_value(list_bref_mode, strInput[i], &value)) {
            pParams->brefMode = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_bref_mode);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("max-bitrate") || IS_OPTION("maxbitrate")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->rcParam.max_bitrate = value * 1000;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("lookahead")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->enableLookahead = value > 0;
            pParams->lookahead = clamp(value, 0, 32);
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("lookahead-level")) {
        i++;
        int value = 0;
        if (get_list_value(list_lookahead_level, strInput[i], &value)) {
            pParams->lookaheadLevel = (NV_ENC_LOOKAHEAD_LEVEL)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_lookahead_level);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("tune") && ENABLE_NVENC_SDK_TUNE) {
        i++;
        int value = 0;
        if (get_list_value(list_tuning_info, strInput[i], &value)) {
            pParams->tuningInfo = (NV_ENC_TUNING_INFO)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_tuning_info);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("no-i-adapt")) {
        pParams->disableIadapt = true;
        return 0;
    }
    if (IS_OPTION("no-b-adapt")) {
        pParams->disableBadapt = true;
        return 0;
    }
    if (IS_OPTION("vbv-bufsize")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->vbvBufferSize = value * 1000;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("aq")) {
        pParams->enableAQ = true;
        return 0;
    }
    if (IS_OPTION("aq-temporal")) {
        pParams->enableAQTemporal = true;
        return 0;
    }
    if (IS_OPTION("aq-strength")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->aqStrength = clamp(value, 0, 15);
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("disable-aq")
        || IS_OPTION("no-aq")) {
        pParams->enableAQ = false;
        return 0;
    }
    if (IS_OPTION("no-aq-temporal")) {
        pParams->enableAQTemporal = false;
        return 0;
    }
    if (IS_OPTION("direct")) {
        i++;
        int value = 0;
        if (get_list_value(list_bdirect, strInput[i], &value)) {
            codecPrm[RGY_CODEC_H264].h264Config.bdirectMode = (NV_ENC_H264_BDIRECT_MODE)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_bdirect);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("adapt-transform")) {
        codecPrm[RGY_CODEC_H264].h264Config.adaptiveTransformMode = NV_ENC_H264_ADAPTIVE_TRANSFORM_ENABLE;
        return 0;
    }
    if (IS_OPTION("no-adapt-transform")) {
        codecPrm[RGY_CODEC_H264].h264Config.adaptiveTransformMode = NV_ENC_H264_ADAPTIVE_TRANSFORM_DISABLE;
        return 0;
    }
    if (IS_OPTION("hierarchial-p")) {
        codecPrm[RGY_CODEC_H264].h264Config.hierarchicalPFrames = 1;
        return 0;
    }
    if (IS_OPTION("hierarchial-b")) {
        codecPrm[RGY_CODEC_H264].h264Config.hierarchicalBFrames = 1;
        return 0;
    }
    if (IS_OPTION("tf-level")) {
        i++;
        int value = 0;
        if (get_list_value(list_temporal_filter_level, strInput[i], &value)) {
            pParams->temporalFilterLevel = (NV_ENC_TEMPORAL_FILTER_LEVEL)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_temporal_filter_level);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("ref")) {
        i++;
        try {
            int value = std::stoi(strInput[i]);
            codecPrm[RGY_CODEC_H264].h264Config.maxNumRefFrames = value;
            codecPrm[RGY_CODEC_HEVC].hevcConfig.maxNumRefFramesInDPB = value;
            codecPrm[RGY_CODEC_AV1 ].av1Config.maxNumRefFramesInDPB = value;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("multiref-l0")) {
        i++;
        int value = 0;
        if (get_list_value(list_num_refs, strInput[i], &value)) {
            codecPrm[RGY_CODEC_H264].h264Config.numRefL0 = (NV_ENC_NUM_REF_FRAMES)value;
            codecPrm[RGY_CODEC_HEVC].hevcConfig.numRefL0 = (NV_ENC_NUM_REF_FRAMES)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_num_refs);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("multiref-l1")) {
        i++;
        int value = 0;
        if (get_list_value(list_num_refs, strInput[i], &value)) {
            codecPrm[RGY_CODEC_H264].h264Config.numRefL1 = (NV_ENC_NUM_REF_FRAMES)value;
            codecPrm[RGY_CODEC_HEVC].hevcConfig.numRefL1 = (NV_ENC_NUM_REF_FRAMES)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_num_refs);
            return 1;
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
    if (IS_OPTION("nonrefp")) {
        pParams->nonrefP = 1;
        return 0;
    }
    if (IS_OPTION("mv-precision")) {
        i++;
        int value = 0;
        if (get_list_value(list_mv_presicion, strInput[i], &value)) {
            pParams->mvPrecision = (NV_ENC_MV_PRECISION)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_mv_presicion);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("cavlc")) {
        codecPrm[RGY_CODEC_H264].h264Config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC;
        return 0;
    }
    if (IS_OPTION("cabac")) {
        codecPrm[RGY_CODEC_H264].h264Config.entropyCodingMode = NV_ENC_H264_ENTROPY_CODING_MODE_CABAC;
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
    if (IS_OPTION("lossless-ignore-input-csp")) {
        pParams->losslessIgnoreInputCsp = TRUE;
        return 0;
    }
    if (IS_OPTION("no-deblock")) {
        codecPrm[RGY_CODEC_H264].h264Config.disableDeblockingFilterIDC = 1;
        return 0;
    }
    if (IS_OPTION("slices:h264")) {
        i++;
        try {
            int value = std::stoi(strInput[i]);
            codecPrm[RGY_CODEC_H264].h264Config.sliceMode = 3;
            codecPrm[RGY_CODEC_H264].h264Config.sliceModeData = value;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("slices:hevc")) {
        i++;
        try {
            int value = std::stoi(strInput[i]);
            codecPrm[RGY_CODEC_HEVC].hevcConfig.sliceMode = 3;
            codecPrm[RGY_CODEC_HEVC].hevcConfig.sliceModeData = value;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("slices")) {
        i++;
        try {
            int value = std::stoi(strInput[i]);
            codecPrm[RGY_CODEC_H264].h264Config.sliceMode = 3;
            codecPrm[RGY_CODEC_HEVC].hevcConfig.sliceMode = 3;
            codecPrm[RGY_CODEC_H264].h264Config.sliceModeData = value;
            codecPrm[RGY_CODEC_HEVC].hevcConfig.sliceModeData = value;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("tile-columns")) {
        i++;
        int value = 0;
        if (get_list_value(list_av1_tiles, strInput[i], &value)) {
            codecPrm[RGY_CODEC_AV1].av1Config.enableCustomTileConfig = 0;
            codecPrm[RGY_CODEC_AV1].av1Config.numTileColumns = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_av1_tiles);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("tile-rows")) {
        i++;
        int value = 0;
        if (get_list_value(list_av1_tiles, strInput[i], &value)) {
            codecPrm[RGY_CODEC_AV1].av1Config.enableCustomTileConfig = 0;
            codecPrm[RGY_CODEC_AV1].av1Config.numTileRows = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_av1_tiles);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("part-size-min")) {
        i++;
        int value = 0;
        if (get_list_value(list_part_size_av1, strInput[i], &value)) {
            codecPrm[RGY_CODEC_AV1].av1Config.minPartSize = (NV_ENC_AV1_PART_SIZE)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_part_size_av1);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("part-size-max")) {
        i++;
        int value = 0;
        if (get_list_value(list_part_size_av1, strInput[i], &value)) {
            codecPrm[RGY_CODEC_AV1].av1Config.maxPartSize = (NV_ENC_AV1_PART_SIZE)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_part_size_av1);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("temporal-layers") || IS_OPTION("max-temporal-layers")) {
        i++;
        try {
            int value = std::stoi(strInput[i]);
            pParams->temporalLayers = value;
        } catch (...) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("refs-forward")) {
        i++;
        int value = 0;
        if (get_list_value(list_av1_refs_forward, strInput[i], &value)) {
            codecPrm[RGY_CODEC_AV1].av1Config.numFwdRefs = (NV_ENC_NUM_REF_FRAMES)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_av1_refs_forward);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("refs-backward")) {
        i++;
        int value = 0;
        if (get_list_value(list_av1_refs_backward, strInput[i], &value)) {
            codecPrm[RGY_CODEC_AV1].av1Config.numBwdRefs = (NV_ENC_NUM_REF_FRAMES)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_av1_refs_forward);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("av1-out-annexb")) {
        codecPrm[RGY_CODEC_AV1].av1Config.outputAnnexBFormat = 1;
        return 0;
    }
    if (IS_OPTION("deblock")) {
        codecPrm[RGY_CODEC_H264].h264Config.disableDeblockingFilterIDC = 0;
        return 0;
    }
    if (IS_OPTION("repeat-headers:h264")) {
        codecPrm[RGY_CODEC_H264].h264Config.repeatSPSPPS = 1;
        return 0;
    }
    if (IS_OPTION("repeat-headers:hevc")) {
        codecPrm[RGY_CODEC_HEVC].hevcConfig.repeatSPSPPS = 1;
        return 0;
    }
    if (IS_OPTION("repeat-headers:av1")) {
        codecPrm[RGY_CODEC_AV1].av1Config.repeatSeqHdr = 1;
        return 0;
    }
    if (IS_OPTION("repeat-headers")) {
        codecPrm[RGY_CODEC_H264].h264Config.repeatSPSPPS = 1;
        codecPrm[RGY_CODEC_HEVC].hevcConfig.repeatSPSPPS = 1;
        codecPrm[RGY_CODEC_AV1].av1Config.repeatSeqHdr  = 1;
        return 0;
    }
    if (IS_OPTION("aud:h264")) {
        codecPrm[RGY_CODEC_H264].h264Config.outputAUD = 1;
        return 0;
    }
    if (IS_OPTION("aud:hevc")) {
        codecPrm[RGY_CODEC_HEVC].hevcConfig.outputAUD = 1;
        return 0;
    }
    if (IS_OPTION("aud")) {
        codecPrm[RGY_CODEC_H264].h264Config.outputAUD = 1;
        codecPrm[RGY_CODEC_HEVC].hevcConfig.outputAUD = 1;
        return 0;
    }
    if (IS_OPTION("pic-struct:h264")) {
        codecPrm[RGY_CODEC_H264].h264Config.outputPictureTimingSEI = 1;
        return 0;
    }
    if (IS_OPTION("pic-struct:hevc")) {
        codecPrm[RGY_CODEC_HEVC].hevcConfig.outputPictureTimingSEI = 1;
        return 0;
    }
    if (IS_OPTION("pic-struct")) {
        codecPrm[RGY_CODEC_H264].h264Config.outputPictureTimingSEI = 1;
        codecPrm[RGY_CODEC_HEVC].hevcConfig.outputPictureTimingSEI = 1;
        return 0;
    }
    if (IS_OPTION("level") || IS_OPTION("level:h264") || IS_OPTION("level:hevc") || IS_OPTION("level:av1")) {
        const bool for_h264 = IS_OPTION("level") || IS_OPTION("level:h264");
        const bool for_hevc = IS_OPTION("level") || IS_OPTION("level:hevc");
        const bool for_av1  = IS_OPTION("level") || IS_OPTION("level:av1");
        i++;
        bool flag = false;
        int value = 0;
        if (for_h264 && (value = get_value_from_chr(list_avc_level_parse, strInput[i])) != PARSE_ERROR_FLAG) {
            codecPrm[RGY_CODEC_H264].h264Config.level = value;
            flag = true;
        }
        if (for_hevc && (value = get_value_from_chr(list_hevc_level_parse, strInput[i])) != PARSE_ERROR_FLAG) {
            codecPrm[RGY_CODEC_HEVC].hevcConfig.level = value;
            flag = true;
        }
        if (for_av1 && (value = get_value_from_chr(list_av1_level_parse, strInput[i])) != PARSE_ERROR_FLAG) {
            codecPrm[RGY_CODEC_AV1].av1Config.level = value;
            flag = true;
        }
        if (!flag) {
            print_cmd_error_invalid_value(option_name, strInput[i], std::vector < std::pair<RGY_CODEC, const CX_DESC *>>{
                { RGY_CODEC_H264, list_avc_level },
                { RGY_CODEC_HEVC, list_hevc_level },
                { RGY_CODEC_AV1,  list_av1_level }
            });
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("profile") || IS_OPTION("profile:h264") || IS_OPTION("profile:hevc") || IS_OPTION("profile:av1")) {
        const bool for_h264 = IS_OPTION("profile") || IS_OPTION("profile:h264");
        const bool for_hevc = IS_OPTION("profile") || IS_OPTION("profile:hevc");
        const bool for_av1  = IS_OPTION("profile") || IS_OPTION("profile:av1");
        i++;
        bool flag = false;
        if (for_h264) {
            GUID zero = { 0 };
            GUID result_guid = get_guid_from_name(strInput[i], h264_profile_names);
            if (0 != memcmp(&result_guid, &zero, sizeof(result_guid))) {
                pParams->encConfig.profileGUID = result_guid;
                if (!FOR_AUO) pParams->yuv444 = memcmp(&pParams->encConfig.profileGUID, &NV_ENC_H264_PROFILE_HIGH_444_GUID, sizeof(result_guid)) == 0;
                flag = true;
            }
        }
        if (for_hevc) {
            int result = get_value_from_name(strInput[i], h265_profile_names);
            if (-1 != result) {
                //下位16bitを使用する
                uint16_t *ptr = (uint16_t *)&codecPrm[RGY_CODEC_HEVC].hevcConfig.tier;
                ptr[0] = (uint16_t)result;
                if (!FOR_AUO) {
                    if (result == NV_ENC_PROFILE_HEVC_MAIN444) {
                        pParams->yuv444 = TRUE;
                    }
                    if (result == NV_ENC_PROFILE_HEVC_MAIN10) {
                        pParams->outputDepth = 10;
                        pParams->yuv444 = FALSE;
                    } else if (result == NV_ENC_PROFILE_HEVC_MAIN) {
                        pParams->outputDepth = 8;
                        pParams->yuv444 = FALSE;
                    }
                }
                flag = true;
            }
        }
        if (for_av1) {
            int result = get_value_from_name(strInput[i], av1_profile_names);
            if (-1 != result) {
                //下位16bitを使用する
                uint16_t *ptr = (uint16_t *)&codecPrm[RGY_CODEC_AV1].av1Config.tier;
                ptr[0] = (uint16_t)result;
                flag = true;
            }
        }
        if (!flag) {
            print_cmd_error_invalid_value(option_name, strInput[i], std::vector<std::pair<RGY_CODEC, std::vector<guid_desc>>>{
                { RGY_CODEC_H264, make_vector(h264_profile_names) },
                { RGY_CODEC_HEVC, make_vector(h265_profile_names) },
                { RGY_CODEC_AV1,  make_vector(av1_profile_names) }
            });
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("tier") || IS_OPTION("tier:hevc") || IS_OPTION("tier:av1")) {
        const bool for_hevc = IS_OPTION("tier") || IS_OPTION("tier:hevc");
        const bool for_av1  = IS_OPTION("tier") || IS_OPTION("tier:av1");
        i++;
        int value = 0;
        if (for_hevc && get_list_value(h265_tier_names, strInput[i], &value)) {
            //上位16bitを使用する
            uint16_t *ptr = (uint16_t *)&codecPrm[RGY_CODEC_HEVC].hevcConfig.tier;
            ptr[1] = (uint16_t)value;
        } else if (for_av1 && get_list_value(av1_tier_names, strInput[i], &value)) {
            //上位16bitを使用する
            uint16_t *ptr = (uint16_t *)&codecPrm[RGY_CODEC_AV1].av1Config.tier;
            ptr[1] = (uint16_t)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], std::vector<std::pair<RGY_CODEC, const CX_DESC *>>{
                { RGY_CODEC_HEVC, h265_tier_names },
                { RGY_CODEC_AV1,  av1_tier_names }
            });
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("output-depth")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->outputDepth = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("output-csp")) {
        i++;
        int value = 0;
        if (get_list_value(list_output_csp, strInput[i], &value)) {
            const RGY_CSP csp = (RGY_CSP)value;
            pParams->yuv444 = csp == RGY_CSP_YUV444 || csp == RGY_CSP_RGB;
            pParams->alphaChannel = csp == RGY_CSP_YUVA420;
            pParams->rgb = csp == RGY_CSP_RGB;
            if (!FOR_AUO && pParams->yuv444) {
                //H264
                memcpy(&pParams->encConfig.profileGUID, &NV_ENC_H264_PROFILE_HIGH_444_GUID, sizeof(pParams->encConfig.profileGUID));
                //HEVC
                uint16_t *ptr = (uint16_t *)&codecPrm[RGY_CODEC_HEVC].hevcConfig.tier;
                ptr[0] = (uint16_t)NV_ENC_PROFILE_HEVC_MAIN444;
            }
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_output_csp);
            return 1;
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
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("cu-max")) {
        i++;
        int value = 0;
        if (get_list_value(list_hevc_cu_size, strInput[i], &value)) {
            codecPrm[RGY_CODEC_HEVC].hevcConfig.maxCUSize = (NV_ENC_HEVC_CUSIZE)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_hevc_cu_size);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("cu-min")) {
        i++;
        int value = 0;
        if (get_list_value(list_hevc_cu_size, strInput[i], &value)) {
            codecPrm[RGY_CODEC_HEVC].hevcConfig.minCUSize = (NV_ENC_HEVC_CUSIZE)value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_hevc_cu_size);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("alpha-bitrate-ratio")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->alphaBitrateRatio = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("alpha-channel-mode")) {
        i++;
        int value = 0;
        if (get_list_value(list_hevc_alpha_channel_mode, strInput[i], &value)) {
            pParams->alphaChannelMode = value;
        } else if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->alphaChannelMode = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("split-enc")) {
        i++;
        int value = 0;
        if (get_list_value(list_split_enc_mode, strInput[i], &value)) {
            pParams->splitEncMode = (decltype(pParams->splitEncMode))value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_split_enc_mode);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("cuda-schedule")) {
        i++;
        int value = 0;
        if (get_list_value(list_cuda_schedule, strInput[i], &value)) {
            pParams->cudaSchedule = value;
        } else {
            print_cmd_error_invalid_value(option_name, strInput[i], list_cuda_schedule);
            return 1;
        }
        return 0;
    }
    if (IS_OPTION("session-retry")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if (value < 0) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("session-retry should be specified in positive value."));
            return 1;
        }
        pParams->sessionRetry = value;
        return 0;
    }
    if (IS_OPTION("disable-nvml")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            print_cmd_error_invalid_value(option_name, strInput[i]);
            return 1;
        }
        if (value < 0) {
            print_cmd_error_invalid_value(option_name, strInput[i], _T("disable-nvml should be specified in positive value."));
            return 1;
        }
        pParams->disableNVML = value;
        return 0;
    }
    if (IS_OPTION("disable-dx11")) {
        pParams->disableDX11 = true;
        return 0;
    }
    if (IS_OPTION("disable-vulkan")) {
        pParams->disableVulkan = true;
        return 0;
    }

    auto ret = parse_one_input_option(option_name, strInput, i, nArgNum, &pParams->input, &pParams->inprm, argData);
    if (ret >= 0) return ret;

    ret = parse_one_common_option(option_name, strInput, i, nArgNum, &pParams->common, argData);
    if (ret >= 0) return ret;

    ret = parse_one_ctrl_option(option_name, strInput, i, nArgNum, &pParams->ctrl, argData);
    if (ret >= 0) return ret;

    ret = parse_one_vppnv_option(option_name, strInput, i, nArgNum, &pParams->vppnv, argData, pParams->vpp.resize_algo);
    if (ret >= 0) return ret;

    ret = parse_one_vpp_option(option_name, strInput, i, nArgNum, &pParams->vpp, argData);
    if (ret >= 0) return ret;

    print_cmd_error_unknown_opt(strInput[i]);
    return 1;
}

int parse_cmd(InEncodeVideoParam *pParams, NV_ENC_CODEC_CONFIG *codecPrm, int nArgNum, const TCHAR **strInput, bool ignore_parse_err) {
    sArgsData argsData;

    bool debug_cmd_parser = false;
    for (int i = 1; i < nArgNum; i++) {
        if (tstring(strInput[i]) == _T("--debug-cmd-parser")) {
            debug_cmd_parser = true;
            break;
        }
    }

    if (debug_cmd_parser) {
        for (int i = 1; i < nArgNum; i++) {
            _ftprintf(stderr, _T("arg[%3d]: %s\n"), i, strInput[i]);
        }
    }

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
                    print_cmd_error_invalid_value(tstring(), tstring(), strsprintf(_T("Unknown option: \"%s\""), strInput[i]));
                    return -1;
                }
            } else {
                if (ignore_parse_err) continue;
                print_cmd_error_invalid_value(tstring(), tstring(), strsprintf(_T("Invalid option: \"%s\""), strInput[i]));
                return -1;
            }
        }

        if (option_name == nullptr) {
            if (ignore_parse_err) continue;
            print_cmd_error_unknown_opt(strInput[i]);
            return -1;
        }
        if (debug_cmd_parser) {
            _ftprintf(stderr, _T("parsing %3d: %s: "), i, strInput[i]);
        }
        auto sts = parse_one_option(option_name, strInput, i, nArgNum, pParams, codecPrm, &argsData);
        if (debug_cmd_parser) {
            _ftprintf(stderr, _T("%s\n"), (sts == 0) ? _T("OK") : _T("ERR"));
        }
        if (!ignore_parse_err && sts != 0) {
            return sts;
        }
    }

    return 0;
}

#if defined(_WIN32) || defined(_WIN64)
int parse_cmd(InEncodeVideoParam *pParams, NV_ENC_CODEC_CONFIG *codecPrm, const char *cmda, bool ignore_parse_err) {
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
    if (wcslen(argvw[0]) != 0) {
        argv_tstring.push_back(_T("")); // 最初は実行ファイルのパスが入っているのを模擬するため、空文字列を入れておく
    }
    for (int i = 0; i < argc; i++) {
        argv_tstring.push_back(wstring_to_tstring(argvw[i]));
    }
    LocalFree(argvw);

    vector<TCHAR *> argv_tchar;
    for (int i = 0; i < argc; i++) {
        argv_tchar.push_back((TCHAR *)argv_tstring[i].data());
    }
    argv_tchar.push_back(_T("")); // 最後に空白を追加
    const TCHAR **strInput = (const TCHAR **)argv_tchar.data();
    int ret = parse_cmd(pParams, codecPrm, (int)argv_tchar.size() - 1 /*最後の空白の分*/, strInput, ignore_parse_err);
    return ret;
}
#endif //#if defined(_WIN32) || defined(_WIN64)

#pragma warning (push)
#pragma warning (disable: 4127)
tstring gen_cmd(const InEncodeVideoParam *pParams, const NV_ENC_CODEC_CONFIG codecPrmArg[2], bool save_disabled_prm) {
    std::basic_stringstream<TCHAR> cmd;
    InEncodeVideoParam encPrmDefault;
    NV_ENC_CODEC_CONFIG codecPrmDefault[RGY_CODEC_NUM];
    codecPrmDefault[RGY_CODEC_H264] = DefaultParamH264();
    codecPrmDefault[RGY_CODEC_HEVC] = DefaultParamHEVC();
    codecPrmDefault[RGY_CODEC_AV1]  = DefaultParamAV1();

    NV_ENC_CODEC_CONFIG codecPrm[RGY_CODEC_NUM];
    if (codecPrmArg != nullptr) {
        memcpy(codecPrm, codecPrmArg, sizeof(codecPrm));
    } else {
        memcpy(codecPrm, codecPrmDefault, sizeof(codecPrm));
        codecPrm[pParams->codec_rgy] = pParams->encConfig.encodeCodecConfig;
    }

#define OPT_FLOAT(str, opt, prec) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << (str) << _T(" ") << std::setprecision(prec) << (pParams->opt);
#define OPT_NUM(str, opt) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << (str) << _T(" ") << (int)(pParams->opt);
#define OPT_NUM_AV1(str, codec, opt)  if ((codecPrm[RGY_CODEC_AV1].av1Config.opt) != (codecPrmDefault[RGY_CODEC_AV1].av1Config.opt)) cmd << _T(" ") << (str) << ((save_disabled_prm) ? codec : _T("")) << _T(" ") << (int)(codecPrm[RGY_CODEC_AV1].av1Config.opt);
#define OPT_NUM_HEVC(str, codec, opt) if ((codecPrm[RGY_CODEC_HEVC].hevcConfig.opt) != (codecPrmDefault[RGY_CODEC_HEVC].hevcConfig.opt)) cmd << _T(" ") << (str) << ((save_disabled_prm) ? codec : _T("")) << _T(" ") << (int)(codecPrm[RGY_CODEC_HEVC].hevcConfig.opt);
#define OPT_NUM_H264(str, codec, opt) if ((codecPrm[RGY_CODEC_H264].h264Config.opt) != (codecPrmDefault[RGY_CODEC_H264].h264Config.opt)) cmd << _T(" ") << (str) << ((save_disabled_prm) ? codec : _T("")) << _T(" ") << (int)(codecPrm[RGY_CODEC_H264].h264Config.opt);
#define OPT_GUID(str, opt, list) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << (str) << _T(" ") << get_name_from_guid((pParams->opt), list);
#define OPT_GUID_AV1(str, codec, opt, list) if ((codecPrm[RGY_CODEC_AV1].av1Config.opt) != (codecPrmDefault[RGY_CODEC_AV1].av1Config.opt)) cmd << _T(" ") << (str) << ((save_disabled_prm) ? codec : _T("")) << _T(" ") << get_name_from_value((codecPrm[RGY_CODEC_AV1].av1Config.opt), list);
#define OPT_GUID_HEVC(str, codec, opt, list) if ((codecPrm[RGY_CODEC_HEVC].hevcConfig.opt) != (codecPrmDefault[RGY_CODEC_HEVC].hevcConfig.opt)) cmd << _T(" ") << (str) << ((save_disabled_prm) ? codec : _T("")) << _T(" ") << get_name_from_value((codecPrm[RGY_CODEC_HEVC].hevcConfig.opt), list);
#define OPT_LST(str, opt, list) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << (str) << _T(" ") << get_chr_from_value(list, (pParams->opt));
#define OPT_LST_AV1(str, codec, opt, list) if ((codecPrm[RGY_CODEC_AV1].av1Config.opt) != (codecPrmDefault[RGY_CODEC_AV1].av1Config.opt)) cmd << _T(" ") << (str) << ((save_disabled_prm) ? codec : _T("")) << _T(" ") << get_chr_from_value(list, (codecPrm[RGY_CODEC_AV1].av1Config.opt));
#define OPT_LST_HEVC(str, codec, opt, list) if ((codecPrm[RGY_CODEC_HEVC].hevcConfig.opt) != (codecPrmDefault[RGY_CODEC_HEVC].hevcConfig.opt)) cmd << _T(" ") << (str) << ((save_disabled_prm) ? codec : _T("")) << _T(" ") << get_chr_from_value(list, (codecPrm[RGY_CODEC_HEVC].hevcConfig.opt));
#define OPT_LST_H264(str, codec, opt, list) if ((codecPrm[RGY_CODEC_H264].h264Config.opt) != (codecPrmDefault[RGY_CODEC_H264].h264Config.opt)) cmd << _T(" ") << (str) << ((save_disabled_prm) ? codec : _T("")) << _T(" ") << get_chr_from_value(list, (codecPrm[RGY_CODEC_H264].h264Config.opt));
#define OPT_QP(str, qp, enable, force) { \
    if ((force) || (enable) \
    || (pParams->qp.qpI) != (encPrmDefault.qp.qpI) \
    || (pParams->qp.qpP) != (encPrmDefault.qp.qpP) \
    || (pParams->qp.qpB) != (encPrmDefault.qp.qpB)) { \
        if (enable) { \
            cmd << _T(" ") << (str) << _T(" "); \
        } else { \
            cmd << _T(" ") << (str) << _T(" 0;"); \
        } \
        if ((pParams->qp.qpI) == (pParams->qp.qpP) && (pParams->qp.qpI) == (pParams->qp.qpB)) { \
            cmd << (int)(pParams->qp.qpI); \
        } else if ((pParams->qp.qpP) == (pParams->qp.qpB)) { \
            cmd << (int)(pParams->qp.qpI) << _T(":") << (int)(pParams->qp.qpP); \
        } else { \
            cmd << (int)(pParams->qp.qpI) << _T(":") << (int)(pParams->qp.qpP) << _T(":") << (int)(pParams->qp.qpB); \
        } \
    } \
}
#define OPT_BOOL(str_true, str_false, opt) if ((pParams->opt) != (encPrmDefault.opt)) cmd << _T(" ") << ((pParams->opt) ? (str_true) : (str_false));
#define OPT_BOOL_AV1(str_true, str_false, codec, opt) \
    if ((codecPrm[RGY_CODEC_AV1].av1Config.opt) != (codecPrmDefault[RGY_CODEC_AV1].av1Config.opt)) { \
        cmd << _T(" "); \
        if ((codecPrm[RGY_CODEC_AV1].av1Config.opt)) { \
            if (_tcslen(str_true)) { cmd << (str_true) << ((save_disabled_prm) ? (codec) : _T("")); } \
        } else { \
            if (_tcslen(str_false)) { cmd << (str_false) << ((save_disabled_prm) ? (codec) : _T("")); } \
        } \
    }
#define OPT_BOOL_HEVC(str_true, str_false, codec, opt) \
    if ((codecPrm[RGY_CODEC_HEVC].hevcConfig.opt) != (codecPrmDefault[RGY_CODEC_HEVC].hevcConfig.opt)) { \
        cmd << _T(" "); \
        if ((codecPrm[RGY_CODEC_HEVC].hevcConfig.opt)) { \
            if (_tcslen(str_true)) { cmd << (str_true) << ((save_disabled_prm) ? (codec) : _T("")); } \
        } else { \
            if (_tcslen(str_false)) { cmd << (str_false) << ((save_disabled_prm) ? (codec) : _T("")); } \
        } \
    }
#define OPT_BOOL_H264(str_true, str_false, codec, opt) \
    if ((codecPrm[RGY_CODEC_H264].h264Config.opt) != (codecPrmDefault[RGY_CODEC_H264].h264Config.opt)) { \
        cmd << _T(" "); \
        if ((codecPrm[RGY_CODEC_H264].h264Config.opt)) { \
            if (_tcslen(str_true)) { cmd << (str_true) << ((save_disabled_prm) ? (codec) : _T("")); }\
        } else { \
            if (_tcslen(str_false)) { cmd << (str_false) << ((save_disabled_prm) ? (codec) : _T("")); }\
        } \
    }
#define OPT_TCHAR(str, opt) if ((pParams->opt) && _tcslen(pParams->opt)) cmd << _T(" ") << str << _T(" ") << (pParams->opt);
#define OPT_TSTR(str, opt) if (pParams->opt.length() > 0) cmd << _T(" ") << str << _T(" ") << pParams->opt.c_str();
#define OPT_CHAR(str, opt) if ((pParams->opt) && _tcslen(pParams->opt)) cmd << _T(" ") << str << _T(" ") << char_to_tstring(pParams->opt);
#define OPT_STR(str, opt) if (pParams->opt.length() > 0) cmd << _T(" ") << str << _T(" ") << char_to_tstring(pParams->opt).c_str();
#define OPT_CHAR_PATH(str, opt) if ((pParams->opt) && _tcslen(pParams->opt)) cmd << _T(" ") << str << _T(" \"") << (pParams->opt) << _T("\"");
#define OPT_STR_PATH(str, opt) if (pParams->opt.length() > 0) cmd << _T(" ") << str << _T(" \"") << (pParams->opt.c_str()) << _T("\"");

    OPT_NUM(_T("-d"), deviceID);
    cmd << _T(" -c ") << get_chr_from_value(list_nvenc_codecs_for_opt, pParams->codec_rgy);
    if ((pParams->preset) != (encPrmDefault.preset)) cmd << _T(" -u ") << get_name_from_value(pParams->preset, list_nvenc_preset_names_ver10);

    cmd << gen_cmd(&pParams->input, &encPrmDefault.input, &pParams->inprm, &encPrmDefault.inprm, save_disabled_prm);

    #pragma warning(push)
    #pragma warning(disable: 4063) //C4063: case '16' は '_NV_ENC_PARAMS_RC_MODE' の switch の値として正しくありません。
    if (save_disabled_prm) {
        switch (pParams->rcParam.rc_mode) {
        case NV_ENC_PARAMS_RC_CBR:
        case NV_ENC_PARAMS_RC_CBR_HQ:
        case NV_ENC_PARAMS_RC_VBR:
        case NV_ENC_PARAMS_RC_VBR_HQ: {
            OPT_QP(_T("--cqp"), rcParam.qp, true, true);
        } break;
        case NV_ENC_PARAMS_RC_CONSTQP:
        default: {
            cmd << _T(" --vbr ") << pParams->rcParam.avg_bitrate / 1000;
        } break;
        }
    }

    if (pParams->rcParam.rc_mode != NV_ENC_PARAMS_RC_QVBR
        && (pParams->rcParam.targetQuality > 0 || pParams->rcParam.targetQualityLSB > 0)) {
        const float val = pParams->rcParam.targetQuality + pParams->rcParam.targetQualityLSB / 256.0f;
        cmd << _T(" --vbr-quality ") << std::fixed << std::setprecision(2) << val;
    }

    switch (pParams->rcParam.rc_mode) {
    case NV_ENC_PARAMS_RC_CBR: {
        cmd << _T(" --cbr ") << pParams->rcParam.avg_bitrate / 1000;
    } break;
    case NV_ENC_PARAMS_RC_CBR_HQ: {
        cmd << _T(" --cbrhq ") << pParams->rcParam.avg_bitrate / 1000;
    } break;
    case NV_ENC_PARAMS_RC_VBR: {
        cmd << _T(" --vbr ") << pParams->rcParam.avg_bitrate / 1000;
    } break;
    case NV_ENC_PARAMS_RC_VBR_HQ: {
        cmd << _T(" --vbrhq ") << pParams->rcParam.avg_bitrate / 1000;
    } break;
    case NV_ENC_PARAMS_RC_QVBR: {
        if (pParams->rcParam.targetQuality > 0 || pParams->rcParam.targetQualityLSB > 0) {
            const float val = pParams->rcParam.targetQuality + pParams->rcParam.targetQualityLSB / 256.0f;
            cmd << _T(" --qvbr ") << std::fixed << std::setprecision(2) << val;
        } else {
            cmd << _T(" --qvbr auto");
        }
    } break;
    case NV_ENC_PARAMS_RC_CONSTQP:
    default: {
        OPT_QP(_T("--cqp"), rcParam.qp, true, true);
    } break;
    }
    #pragma warning(pop)

    OPT_LST(_T("--multipass"), multipass, list_nvenc_multipass_mode);
    if (pParams->rcParam.rc_mode != NV_ENC_PARAMS_RC_CONSTQP || save_disabled_prm) {
        OPT_NUM(_T("--vbv-bufsize"), vbvBufferSize / 1000);
        OPT_NUM(_T("--max-bitrate"), rcParam.max_bitrate / 1000);
    }
    if (pParams->qpInit.enable || save_disabled_prm) {
        OPT_QP(_T("--qp-init"), qpInit, pParams->qpInit.enable, false);
    }
    if (pParams->qpMin.enable || save_disabled_prm) {
        OPT_QP(_T("--qp-min"), qpMin, pParams->qpMin.enable, false);
    }
    if (pParams->qpMax.enable || save_disabled_prm) {
        OPT_QP(_T("--qp-max"), qpMax, pParams->qpMax.enable, false);
    }
    OPT_NUM(_T("--chroma-qp-offset"), chromaQPOffset);

    if (pParams->enableLookahead || save_disabled_prm) {
        OPT_NUM(_T("--lookahead"), lookahead);
    }
    OPT_LST(_T("--lookahead-level"), lookaheadLevel, list_lookahead_level);
    OPT_LST(_T("--tune"), tuningInfo, list_tuning_info);
    OPT_BOOL(_T("--no-i-adapt"), _T(""), disableIadapt);
    OPT_BOOL(_T("--no-b-adapt"), _T(""), disableBadapt);
    OPT_BOOL(_T("--strict-gop"), _T(""), strictGOP);
    if (pParams->gopLength == 0) {
        cmd << _T(" --gop-len auto");
    } else {
        OPT_NUM(_T("--gop-len"), gopLength);
    }
    OPT_NUM(_T("-b"), bFrames);
    OPT_NUM(_T("--output-depth"), outputDepth);
    OPT_LST(_T("--bref-mode"), brefMode, list_bref_mode);
    OPT_BOOL(_T("--weightp"), _T(""), nWeightP);
    OPT_BOOL(_T("--nonrefp"), _T(""), nonrefP);
    OPT_BOOL(_T("--aq"), _T("--no-aq"), enableAQ);
    OPT_BOOL(_T("--aq-temporal"), _T(""), enableAQTemporal);
    OPT_NUM(_T("--aq-strength"), aqStrength);
    OPT_LST(_T("--mv-precision"), mvPrecision, list_mv_presicion);
    if (pParams->par[0] > 0 && pParams->par[1] > 0) {
        cmd << _T(" --sar ") << pParams->par[0] << _T(":") << pParams->par[1];
    } else if (pParams->par[0] < 0 && pParams->par[1] < 0) {
        cmd << _T(" --dar ") << -1 * pParams->par[0] << _T(":") << -1 * pParams->par[1];
    }
    OPT_BOOL(_T("--lossless"), _T(""), lossless);
    OPT_BOOL(_T("--lossless-ignore-input-csp"), _T(""), losslessIgnoreInputCsp);

    if (pParams->rgb) {
        cmd << _T(" --output-csp ") << get_cx_desc(list_output_csp, (int)RGY_CSP_RGB);
    } else if (pParams->yuv444) {
        cmd << _T(" --output-csp ") << get_cx_desc(list_output_csp, (int)RGY_CSP_YUV444);
    } else if (pParams->alphaChannel) {
        cmd << _T(" --output-csp ") << get_cx_desc(list_output_csp, (int)RGY_CSP_YUVA420);
    }
    OPT_LST(_T("--tf-level"), temporalFilterLevel, list_temporal_filter_level);
    OPT_NUM(_T("--temporal-layers"), temporalLayers);

    if (pParams->codec_rgy == RGY_CODEC_AV1 || save_disabled_prm) {
        OPT_LST_AV1(_T("--level"), _T(":av1"), level, list_av1_level);
        OPT_GUID_AV1(_T("--profile"), _T(":av1"), tier & 0xffff, av1_profile_names);
        OPT_LST_AV1(_T("--tier"), _T(":av1"), tier >> 16, av1_tier_names);
        OPT_BOOL_AV1(_T("--repeat-headers"), _T(""), _T(":av1"), repeatSeqHdr);
        OPT_BOOL_AV1(_T("--av1-out-annexb"), _T(""), _T(""), outputAnnexBFormat);

        OPT_LST_AV1(_T("--tile-columns"),  _T(""), numTileColumns, list_av1_tiles);
        OPT_LST_AV1(_T("--tile-rows"),     _T(""), numTileRows,    list_av1_tiles);
        OPT_LST_AV1(_T("--part-size-min"), _T(""), minPartSize,    list_part_size_av1);
        OPT_LST_AV1(_T("--part-size-max"), _T(""), maxPartSize,    list_part_size_av1);
        OPT_LST_AV1(_T("--refs-forward"),  _T(""), numFwdRefs, list_av1_refs_forward);
        OPT_LST_AV1(_T("--refs-backward"), _T(""), numBwdRefs, list_av1_refs_backward);
    }
    if (pParams->codec_rgy == RGY_CODEC_HEVC || save_disabled_prm) {
        OPT_LST_HEVC(_T("--level"), _T(":hevc"), level, list_hevc_level);
        OPT_GUID_HEVC(_T("--profile"), _T(":hevc"), tier & 0xffff, h265_profile_names);
        OPT_LST_HEVC(_T("--tier"), _T(":hevc"), tier >> 16, h265_tier_names);
        OPT_NUM_HEVC(_T("--ref"), _T(""), maxNumRefFramesInDPB);
        OPT_NUM_HEVC(_T("--multiref-l0"), _T(""), numRefL0);
        OPT_NUM_HEVC(_T("--multiref-l1"), _T(""), numRefL1);
        OPT_NUM_HEVC(_T("--slices"), _T(":hevc"), sliceModeData);
        OPT_BOOL_HEVC(_T("--aud"), _T(""), _T(":hevc"), outputAUD);
        OPT_BOOL_HEVC(_T("--repeat-headers"), _T(""), _T(":hevc"), repeatSPSPPS);
        OPT_BOOL_HEVC(_T("--pic-struct"), _T(""), _T(":hevc"), outputPictureTimingSEI);
        OPT_LST_HEVC(_T("--cu-max"), _T(""), maxCUSize, list_hevc_cu_size);
        OPT_LST_HEVC(_T("--cu-min"), _T(""), minCUSize, list_hevc_cu_size);
        OPT_NUM(_T("--alpha-bitrate-ratio"), alphaBitrateRatio);
        OPT_LST(_T("--alpha-channel-mode"), alphaChannelMode, list_hevc_alpha_channel_mode);
    }
    if (pParams->codec_rgy == RGY_CODEC_H264 || save_disabled_prm) {
        OPT_LST_H264(_T("--level"), _T(":h264"), level, list_avc_level);
        OPT_GUID(_T("--profile"), encConfig.profileGUID, h264_profile_names);
        OPT_NUM_H264(_T("--ref"), _T(""), maxNumRefFrames);
        OPT_NUM_H264(_T("--multiref-l0"), _T(""), numRefL0);
        OPT_NUM_H264(_T("--multiref-l1"), _T(""), numRefL1);
        OPT_LST_H264(_T("--direct"), _T(""), bdirectMode, list_bdirect);
        OPT_LST_H264(_T("--adapt-transform"), _T(""), adaptiveTransformMode, list_adapt_transform);
        OPT_NUM_H264(_T("--slices"), _T(":h264"), sliceModeData);
        OPT_BOOL_H264(_T("--aud"), _T(""), _T(":h264"), outputAUD);
        OPT_BOOL_H264(_T("--repeat-headers"), _T(""), _T(":h264"), repeatSPSPPS);
        OPT_BOOL_H264(_T("--pic-struct"), _T(""), _T(":h264"), outputPictureTimingSEI);
        if ((codecPrm[RGY_CODEC_H264].h264Config.entropyCodingMode) != (codecPrmDefault[RGY_CODEC_H264].h264Config.entropyCodingMode)) {
            cmd << _T(" --") << get_chr_from_value(list_entropy_coding, codecPrm[RGY_CODEC_H264].h264Config.entropyCodingMode);
        }
        OPT_BOOL(_T("--bluray"), _T(""), bluray);
        OPT_BOOL_H264(_T("--no-deblock"), _T("--deblock"), _T(""), disableDeblockingFilterIDC);
    }

    OPT_LST(_T("--split-enc"), splitEncMode, list_split_enc_mode);

    cmd << gen_cmd(&pParams->common, &encPrmDefault.common, save_disabled_prm);

#define ADD_FLOAT(str, opt, prec) if ((pParams->opt) != (encPrmDefault.opt)) tmp << _T(",") << (str) << _T("=") << std::setprecision(prec) << (pParams->opt);
#define ADD_NUM(str, opt) if ((pParams->opt) != (encPrmDefault.opt)) tmp << _T(",") << (str) << _T("=") << (pParams->opt);
#define ADD_LST(str, opt, list) if ((pParams->opt) != (encPrmDefault.opt)) tmp << _T(",") << (str) << _T("=") << get_chr_from_value(list, (pParams->opt));
#define ADD_BOOL(str, opt) if ((pParams->opt) != (encPrmDefault.opt)) tmp << _T(",") << (str) << _T("=") << ((pParams->opt) ? (_T("true")) : (_T("false")));
#define ADD_CHAR(str, opt) if ((pParams->opt) && _tcslen(pParams->opt)) tmp << _T(",") << (str) << _T("=") << (pParams->opt);
#define ADD_PATH(str, opt) if ((pParams->opt) && _tcslen(pParams->opt)) tmp << _T(",") << (str) << _T("=\"") << (pParams->opt) << _T("\"");
#define ADD_STR(str, opt) if (pParams->opt.length() > 0) tmp << _T(",") << (str) << _T("=") << (pParams->opt.c_str());

    cmd << gen_cmd(&pParams->vppnv, &encPrmDefault.vppnv, pParams->vpp.resize_algo, save_disabled_prm);

    cmd << gen_cmd(&pParams->vpp, &encPrmDefault.vpp, save_disabled_prm);

    OPT_LST(_T("--cuda-schedule"), cudaSchedule, list_cuda_schedule);
    OPT_NUM(_T("--session-retry"), sessionRetry);
    OPT_NUM(_T("--disable-nvml"), disableNVML);
    OPT_BOOL(_T("--disable-dx11"), _T(""), disableDX11);
    OPT_BOOL(_T("--disable-vulkan"), _T(""), disableVulkan);

    cmd << gen_cmd(&pParams->ctrl, &encPrmDefault.ctrl, save_disabled_prm);

    return cmd.str();
}
#pragma warning (pop)

#undef CMD_PARSE_SET_ERR
