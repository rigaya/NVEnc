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
#include <Windows.h>
#include <locale.h>
#include <tchar.h>
#include <locale.h>
#include <signal.h>
#include <fcntl.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <set>
#include <cstdio>
#include "rgy_version.h"
#include "NVEncCore.h"
#include "NVEncFeature.h"
#include "NVEncParam.h"
#include "NVEncUtil.h"
#include "NVEncFilterAfs.h"
#include "rgy_util.h"

#if ENABLE_CPP_REGEX
#include <regex>
#endif //#if ENABLE_CPP_REGEX
#if ENABLE_DTL
#include <dtl/dtl.hpp>
#endif //#if ENABLE_DTL

bool check_locale_is_ja() {
    const WORD LangID_ja_JP = MAKELANGID(LANG_JAPANESE, SUBLANG_JAPANESE_JAPAN);
    return GetUserDefaultLangID() == LangID_ja_JP;
}

static tstring GetNVEncVersion() {
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
    if (ENABLE_AVSW_READER) version += strsprintf(_T(", avhw [%s]"), getHWDecSupportedCodecList().c_str());
    version += _T("\n");
    return version;
}

static void show_version() {
    _ftprintf(stdout, _T("%s"), GetNVEncVersion().c_str());
}

class CombinationGenerator {
public:
    CombinationGenerator(int i) : m_nCombination(i) {

    }
    void create(vector<int> used) {
        if ((int)used.size() == m_nCombination) {
            m_nCombinationList.push_back(used);
        }
        for (int i = 0; i < m_nCombination; i++) {
            if (std::find(used.begin(), used.end(), i) == used.end()) {
                vector<int> u = used;
                u.push_back(i);
                create(u);
            }
        }
    }
    vector<vector<int>> generate() {
        vector<int> used;
        create(used);
        return m_nCombinationList;
    };
    int m_nCombination;
    vector<vector<int>> m_nCombinationList;
};

//適当に改行しながら表示する
static tstring PrintListOptions(const TCHAR *option_name, const CX_DESC *list, int default_index) {
    const TCHAR *indent_space = _T("                                ");
    const int indent_len = (int)_tcslen(indent_space);
    const int max_len = 77;
    tstring str = strsprintf(_T("   %s "), option_name);
    while ((int)str.length() < indent_len)
        str += _T(" ");
    int line_len = (int)str.length();
    for (int i = 0; list[i].desc; i++) {
        if (line_len + _tcslen(list[i].desc) + _tcslen(_T(", ")) >= max_len) {
            str += strsprintf(_T("\n%s"), indent_space);
            line_len = indent_len;
        } else {
            if (i) {
                str += strsprintf(_T(", "));
                line_len += 2;
            }
        }
        str += strsprintf(_T("%s"), list[i].desc);
        line_len += (int)_tcslen(list[i].desc);
    }
    str += strsprintf(_T("\n%s default: %s\n"), indent_space, list[default_index].desc);
    return str;
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

static tstring help() {
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
        _T("   --check-formats              show in/out formats available\n")
        _T("   --check-protocols            show in/out protocols available\n")
        _T("   --check-filters              show filters available\n")
#endif
        _T("\n"));
    str += strsprintf(_T("\n")
        _T("Basic Encoding Options: \n")
        _T("-d,--device <int>               set DeviceId used in NVEnc (default:-1 as auto)\n")
        _T("                                  use --check-device to show device ids.\n")
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
        _T("   --avhw [<string>]           use libavformat + cuvid for input\n")
        _T("                                 this enables full hw transcode and resize.\n")
        _T("                                 avhw mode could be set as a  option\n")
        _T("                                  - native (default)\n")
        _T("                                  - cuda\n")
        _T("   --avsw                       set input to use avcodec + sw decoder\n")
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
        _T("   --audio-source <string>      input extra audio file\n")
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
        _T("   --sub-copy [<int>[,...]]     copy subtitle to output file.\n")
        _T("                                 these could be only used with\n")
        _T("                                 avhw/avsw reader and avcodec muxer.\n")
        _T("                                 below are optional,\n")
        _T("                                  in [<int>?], specify track number to copy.\n")
        _T("\n")
        _T("   --avsync <string>            method for AV sync (default: cfr)\n")
        _T("                                 cfr      ... assume cfr, no check but fast\n")
        _T("                                 forcecfr ... check timestamp and force cfr.\n")
        _T("-m,--mux-option <string1>:<string2>\n")
        _T("                                set muxer option name and value.\n")
        _T("                                 these could be only used with\n")
        _T("                                 avhw/avsw reader and avcodec muxer.\n"),
        DEFAULT_IGNORE_DECODE_ERROR);
#endif
    str += strsprintf(_T("")
        _T("   --input-res <int>x<int>        set input resolution\n")
        _T("   --crop <int>,<int>,<int>,<int> crop pixels from left,top,right,bottom\n")
        _T("                                    left crop is unavailable with avhw reader\n")
        _T("   --output-res <int>x<int>     set output resolution\n")
        _T("   --fps <int>/<int> or <float> set framerate\n")
        _T("\n")
        _T("-c,--codec <string>             set output codec\n")
        _T("                                  h264 (or avc), h265 (or hevc)\n")
        _T("   --profile <string>           set codec profile\n")
        _T("                                  H.264: baseline, main, high(default), high444\n")
        _T("                                  HEVC : main, main10, main444\n")
        _T("   --lossless                   for lossless (YUV444 only) / default: off\n"));

    str += PrintMultipleListOptions(_T("--level <string>"), _T("set codec level"),
        { { _T("H.264"), list_avc_level,   0 },
          { _T("HEVC"),  list_hevc_level,  0 }
    });
    str += strsprintf(_T("")
        _T("   --output-depth <int>         set output bit depth ( 8(default), 10 )\n")
        _T("   --sar <int>:<int>            set Sample  Aspect Ratio\n")
        _T("   --dar <int>:<int>            set Display Aspect Ratio\n")
        _T("\n")
        _T("   --cqp <int> or               encode in Constant QP mode\n")
        _T("         <int>:<int>:<int>        default: <I>:<P>:<B>=<%d>:<%d>:<%d>\n")
        _T("   --vbr <int>                  set bitrate for VBR mode (kbps)\n")
        _T("   --vbrhq <int>                set bitrate for VBR (High Quality) mode (kbps)\n")
        _T("   --cbr <int>                  set bitrate for CBR mode (kbps)\n")
        _T("   --cbrhq <int>                set bitrate for CBR (High Quality) mode (kbps)\n")
        _T("                                  default: %d kbps\n")
        _T("\n")
        _T("   --preset <string>            set encoder preset\n")
        _T("                                  default, performance, quality\n")
        _T("\n")
        _T("   --vbr-quality <float>        target quality for VBR mode (0-51, 0=auto)\n")
        _T("   --max-bitrate <int>          set Max Bitrate (kbps)\n")
        _T("   --qp-init <int> or           set initial QP\n")
        _T("             <int>:<int>:<int>    default: auto\n")
        _T("   --qp-max <int> or            set max QP\n")
        _T("            <int>:<int>:<int>     default: unset\n")
        _T("   --qp-min <int> or            set min QP\n")
        _T("             <int>:<int>:<int>    default: unset\n")
        _T("   --gop-len <int>              set GOP Length / default: %d frames%s\n")
        _T("   --lookahead <int>            enable lookahead and set lookahead depth (1-32)\n")
        _T("                                  default: %d frames\n")
        _T("   --strict-gop                 avoid GOP len fluctuation\n")
        _T("   --no-i-adapt                 disable adapt. I frame insertion\n")
        _T("   --no-b-adapt                 disable adapt. B frame insertion\n")
        _T("                                  for lookahead mode only, default: off\n")
        _T("-b,--bframes <int>              set number of consecutive B frames\n")
        _T("                                  default: H.264 - %d frames, HEVC - %d frames\n")
        _T("   --ref <int>                  set Ref frames / default %d frames\n")
        _T("   --weightp                    enable weighted prediction for P frame\n")
        _T("                                  FOR H.264 ONLY\n")
        _T("   --(no-)aq                    enable spatial adaptive quantization\n")
        _T("   --aq-temporal                enable temporal adaptive quantization\n")
        _T("                                  FOR H.264 ONLY\n")
        _T("   --aq-strength <int>          set aq strength (weak 1 - 15 strong)\n")
        _T("                                  FOR H.264 ONLY, default: 0 = auto\n")
        _T("   --direct <string>            set H.264 B Direct mode\n")
        _T("                                  auto(default), none, spatial, temporal\n")
        _T("   --(no-)adapt-transform       set H.264 adaptive transform mode (default=auto)\n")
        _T("   --mv-precision <string>      set MV Precision / default: auto\n")
        _T("                                  auto,\n")
        _T("                                  Q-pel (High Quality),\n")
        _T("                                  half-pel,\n")
        _T("                                  full-pel (Low Quality, not recommended)\n")
        _T("   --vbv-bufsize <int>          set vbv buffer size (kbit) / default: auto\n"),
        DEFAUTL_QP_I, DEFAULT_QP_P, DEFAULT_QP_B,
        DEFAULT_AVG_BITRATE / 1000,
        DEFAULT_GOP_LENGTH, (DEFAULT_GOP_LENGTH == 0) ? _T(" (auto)") : _T(""),
        DEFAULT_LOOKAHEAD,
        DEFAULT_B_FRAMES_H264, DEFAULT_B_FRAMES_HEVC, DEFAULT_REF_FRAMES);

    str += PrintListOptions(_T("--videoformat <string>"), list_videoformat, 0);
    str += PrintListOptions(_T("--colormatrix <string>"), list_colormatrix, 0);
    str += PrintListOptions(_T("--colorprim <string>"), list_colorprim, 0);
    str += PrintListOptions(_T("--transfer <string>"), list_transfer, 0);
    str += strsprintf(_T("")
        _T("   --fullrange                  set fullrange\n")
        _T("   --max-cll <int>,<int>        set MaxCLL and MaxFall in nits. e.g. \"1000,300\"\n")
        _T("   --master-display <string>    set Mastering display data.\n")
        _T("      e.g. \"G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)\"\n"));

    str += strsprintf(_T("\n")
        _T("H.264/AVC\n")
        _T("   --tff                        same as --interlace tff\n")
        _T("   --bff                        same as --interlace bff\n")
        _T("   --interlace <string>         set input as interlaced\n")
        _T("                                  tff, bff\n")
        _T("   --cabac                      use CABAC\n")
        _T("   --cavlc                      use CAVLC (no CABAC)\n")
        _T("   --bluray                     for bluray / default: off\n")
        _T("   --(no-)deblock               enable(disable) deblock filter\n"));

    str += strsprintf(_T("\n")
        _T("H.265/HEVC\n")
        _T("   --cu-max <int>               set max CU size\n")
        _T("   --cu-min  <int>              set min CU size\n")
        _T("                                  8, 16, 32 are avaliable\n")
        _T("    warning: it is not recommended to use --cu-max or --cu-min,\n")
        _T("             leaving it auto will enhance video quality.\n"));
    str += strsprintf(_T("\n")
        _T("   --vpp-deinterlace <string>   set deinterlace mode / default: none\n")
        _T("                                  none, bob, adaptive (normal)\n")
        _T("                                  available only with avhw reader\n"));
    str += PrintListOptions(_T("--vpp-resize <string>"),     list_nppi_resize, 0);
    str += PrintListOptions(_T("--vpp-gauss <int>"),         list_nppi_gauss,  0);
    str += strsprintf(_T("")
        _T("   --vpp-knn [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     enable denoise filter by K-nearest neighbor.\n")
        _T("    params\n")
        _T("      radius=<int>              radius of knn (default=%d)\n")
        _T("      strength=<float>          strength of knn (default=%.2f, 0.0-1.0)\n")
        _T("      lerp=<float>              balance of orig & blended pixel (default=%.2f)\n")
        _T("                                  lower value results strong denoise.\n")
        _T("      th_lerp=<float>           edge detect threshold (default=%.2f, 0.0-1.0)\n")
        _T("                                  higher value will preserve edge.\n"),
        FILTER_DEFAULT_KNN_RADIUS, FILTER_DEFAULT_KNN_STRENGTH, FILTER_DEFAULT_KNN_LERPC,
        FILTER_DEFAULT_KNN_LERPC_THRESHOLD);
    str += strsprintf(_T("\n")
        _T("   --vpp-pmd [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     enable denoise filter by pmd.\n")
        _T("    params\n")
        _T("      apply_count=<int>         count to apply pmd denoise (default=%d)\n")
        _T("      strength=<float>          strength of pmd (default=%.2f, 0.0-100.0)\n")
        _T("      threshold=<float>         threshold of pmd (default=%.2f, 0.0-255.0)\n")
        _T("                                  lower value will preserve edge.\n"),
        FILTER_DEFAULT_PMD_APPLY_COUNT, FILTER_DEFAULT_PMD_STRENGTH, FILTER_DEFAULT_PMD_THRESHOLD);
    str += strsprintf(_T("\n")
        _T("   --vpp-unsharp [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     enable unsharp filter.\n")
        _T("    params\n")
        _T("      radius=<int>              filter range for edge detection (default=%d, 1-9)\n")
        _T("      weight=<float>            strength of filter (default=%.2f, 0-10)\n")
        _T("      threshold=<float>         min brightness change to be sharpened (default=%.2f, 0-255)\n"),
        FILTER_DEFAULT_UNSHARP_RADIUS, FILTER_DEFAULT_UNSHARP_WEIGHT, FILTER_DEFAULT_UNSHARP_THRESHOLD);
    str += strsprintf(_T("\n")
        _T("   --vpp-edgelevel [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     edgelevel filter to enhance edge.\n")
        _T("    params\n")
        _T("      strength=<float>          strength (default=%d, -31 - 31)\n")
        _T("      threshold=<float>         threshold to ignore noise (default=%.1f, 0-255)\n")
        _T("      black=<float>             allow edge to be darker on edge enhancement\n")
        _T("                                  (default=%.1f, 0-31)\n")
        _T("      white=<float>             allow edge to be brighter on edge enhancement\n")
        _T("                                  (default=%.1f, 0-31)\n"),
        FILTER_DEFAULT_EDGELEVEL_STRENGTH, FILTER_DEFAULT_EDGELEVEL_THRESHOLD, FILTER_DEFAULT_EDGELEVEL_BLACK, FILTER_DEFAULT_EDGELEVEL_WHITE);
    str += strsprintf(_T("\n")
        _T("   --vpp-deband [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     enable deband filter.\n")
        _T("    params\n")
        _T("      range=<int>               range (default=%d, 0-127)\n")
        _T("      sample=<int>              sample (default=%d, 0-2)\n")
        _T("      thre=<int>                threshold for y, cb & cr\n")
        _T("      thre_y=<int>              threshold for y (default=%d, 0-31)\n")
        _T("      thre_cb=<int>             threshold for cb (default=%d, 0-31)\n")
        _T("      thre_cr=<int>             threshold for cr (default=%d, 0-31)\n")
        _T("      dither=<int>              strength of dither for y, cb & cr\n")
        _T("      dither_y=<int>            strength of dither for y (default=%d, 0-31)\n")
        _T("      dither_c=<int>            strength of dither for cb/cr (default=%d, 0-31)\n")
        _T("      seed=<int>                rand seed (default=%d)\n")
        _T("      blurfirst                 blurfirst (default=%s)\n")
        _T("      rand_each_frame           generate rand for each frame (default=%s)\n"),
        FILTER_DEFAULT_DEBAND_RANGE, FILTER_DEFAULT_DEBAND_MODE,
        FILTER_DEFAULT_DEBAND_THRE_Y, FILTER_DEFAULT_DEBAND_THRE_CB, FILTER_DEFAULT_DEBAND_THRE_CR,
        FILTER_DEFAULT_DEBAND_DITHER_Y, FILTER_DEFAULT_DEBAND_DITHER_C,
        FILTER_DEFAULT_DEBAND_SEED,
        FILTER_DEFAULT_DEBAND_BLUR_FIRST ? _T("on") : _T("off"),
        FILTER_DEFAULT_DEBAND_RAND_EACH_FRAME ? _T("on") : _T("off"));
    str += strsprintf(_T("")
        _T("   --vpp-afs [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     enable auto field shift deinterlacer\n")
        _T("    params\n")
        _T("      preset=<string>\n")
        _T("          default, triple, double, anime, cinema, min_afterimg,\n")
        _T("          24fps, 24fps_sd, 30fps\n")
        _T("      ini=<string>\n")
        _T("          read setting from ini file specified (output of afs.auf)\n")
        _T("\n")
        _T("      !! params from preset & ini will be overrided by user settings below !!\n")
        _T("\n")
        _T("                   Aviutlでのパラメータ名\n")
        _T("      top=<int>           (上)         clip range to scan (default=%d)\n")
        _T("      bottom=<int>        (下)         clip range to scan (default=%d)\n")
        _T("      left=<int>          (左)         clip range to scan (default=%d)\n")
        _T("      right=<int>         (右)         clip range to scan (default=%d)\n")
        _T("                                        left & right must be muitiple of 4\n")
        _T("      method_switch=<int> (切替点)     (default=%d, 0-256)\n")
        _T("      coeff_shift=<int>   (判定比)     (default=%d, 0-256)\n")
        _T("      thre_shift=<int>    (縞(シフト)) stripe(shift)thres (default=%d, 0-1024)\n")
        _T("      thre_deint=<int>    (縞(解除))   stripe(deint)thres (default=%d, 0-1024)\n")
        _T("      thre_motion_y=<int> (Y動き)      Y motion threshold (default=%d, 0-1024)\n")
        _T("      thre_motion_c=<int> (C動き)      C motion threshold (default=%d, 0-1024)\n")
        _T("      level=<int>         (解除Lv)     set deint level    (default=%d, 0-4\n")
        _T("      shift=<bool>  (フィールドシフト) enable field shift (default=%s)\n")
        _T("      drop=<bool>   (ドロップ)         enable frame drop  (default=%s)\n")
        _T("      smooth=<bool> (スムージング)     enable smoothing   (default=%s)\n")
        _T("      24fps=<bool>  (24fps化)          force 30fps->24fps (default=%s)\n")
        _T("      tune=<bool>   (調整モード)       show scan result   (default=%s)\n")
        _T("      rff=<bool>                       rff flag aware     (default=%s)\n")
        _T("      timecode=<bool>                  output timecode    (default=%s)\n")
        _T("      log=<bool>                       output log         (default=%s)\n"),
        FILTER_DEFAULT_AFS_CLIP_TB, FILTER_DEFAULT_AFS_CLIP_TB,
        FILTER_DEFAULT_AFS_CLIP_LR, FILTER_DEFAULT_AFS_CLIP_LR,
        FILTER_DEFAULT_AFS_METHOD_SWITCH, FILTER_DEFAULT_AFS_COEFF_SHIFT,
        FILTER_DEFAULT_AFS_THRE_SHIFT, FILTER_DEFAULT_AFS_THRE_DEINT,
        FILTER_DEFAULT_AFS_THRE_YMOTION, FILTER_DEFAULT_AFS_THRE_CMOTION,
        FILTER_DEFAULT_AFS_ANALYZE,
        FILTER_DEFAULT_AFS_SHIFT   ? _T("on") : _T("off"),
        FILTER_DEFAULT_AFS_DROP    ? _T("on") : _T("off"),
        FILTER_DEFAULT_AFS_SMOOTH  ? _T("on") : _T("off"),
        FILTER_DEFAULT_AFS_FORCE24 ? _T("on") : _T("off"),
        FILTER_DEFAULT_AFS_TUNE    ? _T("on") : _T("off"),
        FILTER_DEFAULT_AFS_RFF     ? _T("on") : _T("off"),
        FILTER_DEFAULT_AFS_TIMECODE ? _T("on") : _T("off"),
        FILTER_DEFAULT_AFS_LOG      ? _T("on") : _T("off"));
    str += strsprintf(_T("\n")
        _T("   --vpp-rff                    apply rff flag, with avhw reader only.\n"));
    str += strsprintf(_T("\n")
        _T("   --vpp-tweak [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     apply brightness, constrast, gamma, hue adjustment.\n")
        _T("    params\n")
        _T("      brightness=<float>        (default=%.1f, -1.0 - 1.0)\n")
        _T("      contrast=<float>          (default=%.1f, -2.0 - 2.0)\n")
        _T("      gamma=<float>             (default=%.1f,  0.1 - 10.0)\n")
        _T("      saturation=<float>        (default=%.1f,  0.0 - 3.0)\n")
        _T("      hue=<float>               (default=%.1f, -180 - 180)\n"),
        FILTER_DEFAULT_TWEAK_BRIGHTNESS,
        FILTER_DEFAULT_TWEAK_CONTRAST,
        FILTER_DEFAULT_TWEAK_GAMMA,
        FILTER_DEFAULT_TWEAK_SATURATION,
        FILTER_DEFAULT_TWEAK_HUE);
    str += strsprintf(_T("")
        _T("   --vpp-delogo <string>        set delogo file path\n")
        _T("   --vpp-delogo-select <string> set target logo name or auto select file\n")
        _T("                                 or logo index starting from 1.\n")
        _T("   --vpp-delogo-pos <int>:<int> set delogo pos offset\n")
        _T("   --vpp-delogo-depth <int>     set delogo depth [default:%d]\n")
        _T("   --vpp-delogo-y  <int>        set delogo y  param\n")
        _T("   --vpp-delogo-cb <int>        set delogo cb param\n")
        _T("   --vpp-delogo-cr <int>        set delogo cr param\n"),
        FILTER_DEFAULT_DELOGO_DEPTH);
    str += strsprintf(_T("")
        _T("   --vpp-perf-monitor           check duration of each filter.\n")
        _T("                                  may decrease overall transcode performance.\n"));
    str += strsprintf(_T("")
        _T("   --cuda-schedule <string>     set cuda schedule mode (default: sync).\n")
        _T("       auto  : let cuda driver to decide\n")
        _T("       spin  : CPU will spin when waiting GPU tasks,\n")
        _T("               will provide highest performance but with high CPU utilization.\n")
        _T("       yield : CPU will yield when waiting GPU tasks.\n")
        _T("       sync  : CPU will sleep when waiting GPU tasks, performance might\n")
        _T("                drop slightly, while CPU utilization will be lower,\n")
        _T("                especially on HW decode mode.\n"));
    str += strsprintf(_T("\n")
        _T("   --output-buf <int>           buffer size for output in MByte\n")
        _T("                                 default %d MB (0-%d)\n"),
        DEFAULT_OUTPUT_BUF, RGY_OUTPUT_BUF_MB_MAX
    );
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
#endif //#if ENABLE_AVCODEC_OUT_THREAD
        );
    str += strsprintf(_T("\n")
        _T("   --log <string>               set log file name\n")
        _T("   --log-level <string>         set log level\n")
        _T("                                  debug, info(default), warn, error\n")
        _T("   --log-framelist <string>     output frame info of avhw reader to path\n"));

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
#if 0 && ENABLE_NVML
        _T("                                 ved_load    ... gpu video decoder usage (%%)\n")
#endif
#if ENABLE_NVML
        _T("                                 ve_clock    ... gpu video engine clock\n")
#endif
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

static void show_help() {
    _ftprintf(stdout, _T("%s\n"), help().c_str());
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

#if ENABLE_CPP_REGEX
static vector<std::string> createOptionList() {
    vector<std::string> optionList;
    auto helpLines = split(tchar_to_string(help()), "\n");
    std::regex re1(R"(^\s{2,6}--([A-Za-z0-9][A-Za-z0-9-_]+)\s+.*)");
    std::regex re2(R"(^\s{0,3}-[A-Za-z0-9],--([A-Za-z0-9][A-Za-z0-9-_]+)\s+.*)");
    std::regex re3(R"(^\s{0,3}--\(no-\)([A-Za-z0-9][A-Za-z0-9-_]+)\s+.*)");
    for (const auto& line : helpLines) {
        std::smatch match;
        if (std::regex_match(line, match, re1) && match.size() == 2) {
            optionList.push_back(match[1]);
        } else if (std::regex_match(line, match, re2) && match.size() == 2) {
            optionList.push_back(match[1]);
        } else if (std::regex_match(line, match, re3) && match.size() == 2) {
            optionList.push_back(match[1]);
        }
    }
    return optionList;
}
#endif //#if ENABLE_CPP_REGEX

static void PrintHelp(const TCHAR *strAppName, const TCHAR *strErrorMessage, const TCHAR *strOptionName, const TCHAR *strErrorValue = nullptr) {
    UNREFERENCED_PARAMETER(strAppName);

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
#if (ENABLE_CPP_REGEX && ENABLE_DTL)
            if (strErrorValue) {
                //どのオプション名に近いか検証する
                auto optList = createOptionList();
                const auto invalid_opt = tchar_to_string(strErrorValue);
                //入力文字列を"-"で区切り、その組み合わせをすべて試す
                const auto invalid_opt_words = split(invalid_opt, "-", true);
                CombinationGenerator generator((int)invalid_opt_words.size());
                const auto combinationList = generator.generate();
                vector<std::pair<std::string, int>> editDistList;
                for (const auto& opt : optList) {
                    int nMinEditDist = INT_MAX;
                    for (const auto& combination : combinationList) {
                        std::string check_key;
                        for (auto i : combination) {
                            if (check_key.length() > 0) {
                                check_key += "-";
                            }
                            check_key += invalid_opt_words[i];
                        }
                        dtl::Diff<char, std::string> diff(check_key, opt);
                        diff.onOnlyEditDistance();
                        diff.compose();
                        nMinEditDist = (std::min)(nMinEditDist, (int)diff.getEditDistance());
                    }
                    editDistList.push_back(std::make_pair(opt, nMinEditDist));
                }
                std::sort(editDistList.begin(), editDistList.end(), [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) {
                    return b.second > a.second;
                });
                const int nMinEditDist = editDistList[0].second;
                _ftprintf(stderr, _T("Did you mean option(s) below?\n"));
                for (const auto& editDist : editDistList) {
                    if (editDist.second != nMinEditDist) {
                        break;
                    }
                    _ftprintf(stderr, _T("  --%s\n"), char_to_tstring(editDist.first).c_str());
                }
            }
#endif //#if ENABLE_DTL
        }
    } else {
        show_version();
        show_help();
    }
}

static void show_device_list() {
    if (!check_if_nvcuda_dll_available()) {
        _ftprintf(stdout, _T("CUDA not available.\n"));
        return;
    }

    NVEncoderGPUInfo gpuInfo(-1, false);
    auto gpuList = gpuInfo.getGPUList();
    if (0 == gpuList.size()) {
        _ftprintf(stdout, _T("No GPU found suitable for NVEnc Encoding.\n"));
        return;
    }

    for (const auto& gpu : gpuList) {
        _ftprintf(stdout, _T("DeviceId #%d: %s\n"), gpu.id, gpu.name.c_str());
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
                for (size_t i = _tcslen(cap.name); i <= max_length; i++) {
                    _ftprintf(stdout, _T(" "));
                }
                if (cap.isBool) {
                    _ftprintf(stdout, cap.value ? _T("yes\n") : _T("no\n"));
                } else {
                    _ftprintf(stdout, _T("%d\n"), cap.value);
                }
            }
            _ftprintf(stdout, _T("\n"));
        }
    }
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
    if (IS_OPTION("preset")) {
        i++;
        int value = get_value_from_name(strInput[i], list_nvenc_preset_names);
        if (value >= 0) {
            pParams->preset = value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            pParams->input.srcWidth  = a[0];
            pParams->input.srcHeight = a[1];
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
                PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        } else if (value < 0) {
            PrintHelp(strInput[0], _T("input-analyze requires non-negative value."), option_name);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (v == 0) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nVideoTrack = v;
        return 0;
    }
    if (IS_OPTION("video-streamid")) {
        i++;
        int v = 0;
        if (1 != _stscanf_s(strInput[i], _T("%i"), &v)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
                PrintHelp(strInput[0], _T("Invalid Value"), option_name);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (ret <= 2) {
            hh = 0;
        }
        if (ret <= 1) {
            mm = 0;
        }
        if (hh < 0 || mm < 0 || sec < 0) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return 1;
        }
        if (hh > 0 && mm >= 60) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return 1;
        }
        mm += hh * 60;
        if (mm > 0 && sec >= 60.0f) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
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
            PrintHelp(strInput[0], _T("Invalid track number"), option_name);
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
                PrintHelp(strInput[0], _T("Same output file name is used more than twice"), option_name);
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
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return 1;
        }
        return 0;
    }
#if ENABLE_AVSW_READER
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
                    PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
        if (i+1 < nArgNum) {
            const TCHAR *ptr = nullptr;
            const TCHAR *ptrDelim = nullptr;
            if (strInput[i+1][0] != _T('-')) {
                i++;
                ptrDelim = _tcschr(strInput[i], _T('?'));
                ptr = (ptrDelim == nullptr) ? strInput[i] : ptrDelim+1;
            }
            int trackId = 1;
            if (ptrDelim == nullptr) {
                trackId = argData->nParsedAudioEncode+1;
                int idx = getAudioTrackIdx(pParams, trackId);
                if (idx >= 0 && pParams->ppAudioSelectList[idx]->pAVAudioEncodeCodec != nullptr) {
                    trackId = getFreeAudioTrack(pParams);
                }
            } else {
                tstring temp = tstring(strInput[i]).substr(0, ptrDelim - strInput[i]);
                if (1 != _stscanf(temp.c_str(), _T("%d"), &trackId)) {
                    PrintHelp(strInput[0], _T("Invalid value"), option_name);
                    return 1;
                }
            }
            sAudioSelect *pAudioSelect = nullptr;
            int audioIdx = getAudioTrackIdx(pParams, trackId);
            if (audioIdx < 0) {
                pAudioSelect = (sAudioSelect *)calloc(1, sizeof(pAudioSelect[0]));
                pAudioSelect->nAudioSelect = trackId;
            } else {
                pAudioSelect = pParams->ppAudioSelectList[audioIdx];
            }
            pAudioSelect->pAVAudioEncodeCodec = _tcsdup((ptr) ? ptr : RGY_AVCODEC_AUTO);

            if (audioIdx < 0) {
                audioIdx = pParams->nAudioSelectCount;
                //新たに要素を追加
                pParams->ppAudioSelectList = (sAudioSelect **)realloc(pParams->ppAudioSelectList, sizeof(pParams->ppAudioSelectList[0]) * (pParams->nAudioSelectCount + 1));
                pParams->ppAudioSelectList[pParams->nAudioSelectCount] = pAudioSelect;
                pParams->nAudioSelectCount++;
            }
            argData->nParsedAudioEncode++;
        } else {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-bitrate"))) {
        if (i+1 < nArgNum) {
            i++;
            const TCHAR *ptr = _tcschr(strInput[i], _T('?'));
            int trackId = 1;
            if (ptr == nullptr) {
                trackId = argData->nParsedAudioBitrate+1;
                int idx = getAudioTrackIdx(pParams, trackId);
                if (idx >= 0 && pParams->ppAudioSelectList[idx]->nAVAudioEncodeBitrate > 0) {
                    trackId = getFreeAudioTrack(pParams);
                }
                ptr = strInput[i];
            } else {
                tstring temp = tstring(strInput[i]).substr(0, ptr - strInput[i]);
                if (1 != _stscanf(temp.c_str(), _T("%d"), &trackId)) {
                    PrintHelp(strInput[0], _T("Invalid value"), option_name);
                    return 1;
                }
                ptr++;
            }
            sAudioSelect *pAudioSelect = nullptr;
            int audioIdx = getAudioTrackIdx(pParams, trackId);
            if (audioIdx < 0) {
                pAudioSelect = (sAudioSelect *)calloc(1, sizeof(pAudioSelect[0]));
                pAudioSelect->nAudioSelect = trackId;
            } else {
                pAudioSelect = pParams->ppAudioSelectList[audioIdx];
            }
            int bitrate = 0;
            if (1 != _stscanf(ptr, _T("%d"), &bitrate)) {
                PrintHelp(strInput[0], _T("Invalid value"), option_name);
                return 1;
            }
            pAudioSelect->nAVAudioEncodeBitrate = bitrate;

            if (audioIdx < 0) {
                audioIdx = pParams->nAudioSelectCount;
                //新たに要素を追加
                pParams->ppAudioSelectList = (sAudioSelect **)realloc(pParams->ppAudioSelectList, sizeof(pParams->ppAudioSelectList[0]) * (pParams->nAudioSelectCount + 1));
                pParams->ppAudioSelectList[pParams->nAudioSelectCount] = pAudioSelect;
                pParams->nAudioSelectCount++;
            }
            argData->nParsedAudioBitrate++;
        } else {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-ignore-decode-error"))) {
        i++;
        uint32_t value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
        if (i+1 < nArgNum) {
            i++;
            const TCHAR *ptr = _tcschr(strInput[i], _T('?'));
            int trackId = 1;
            if (ptr == nullptr) {
                trackId = argData->nParsedAudioSamplerate+1;
                int idx = getAudioTrackIdx(pParams, trackId);
                if (idx >= 0 && pParams->ppAudioSelectList[idx]->nAudioSamplingRate > 0) {
                    trackId = getFreeAudioTrack(pParams);
                }
                ptr = strInput[i];
            } else {
                tstring temp = tstring(strInput[i]).substr(0, ptr - strInput[i]);
                if (1 != _stscanf(temp.c_str(), _T("%d"), &trackId)) {
                    PrintHelp(strInput[0], _T("Invalid value"), option_name);
                    return 1;
                }
                ptr++;
            }
            sAudioSelect *pAudioSelect = nullptr;
            int audioIdx = getAudioTrackIdx(pParams, trackId);
            if (audioIdx < 0) {
                pAudioSelect = (sAudioSelect *)calloc(1, sizeof(pAudioSelect[0]));
                pAudioSelect->nAudioSelect = trackId;
            } else {
                pAudioSelect = pParams->ppAudioSelectList[audioIdx];
            }
            int bitrate = 0;
            if (1 != _stscanf(ptr, _T("%d"), &bitrate)) {
                PrintHelp(strInput[0], _T("Invalid value"), option_name);
                return 1;
            }
            pAudioSelect->nAudioSamplingRate = bitrate;

            if (audioIdx < 0) {
                audioIdx = pParams->nAudioSelectCount;
                //新たに要素を追加
                pParams->ppAudioSelectList = (sAudioSelect **)realloc(pParams->ppAudioSelectList, sizeof(pParams->ppAudioSelectList[0]) * (pParams->nAudioSelectCount + 1));
                pParams->ppAudioSelectList[pParams->nAudioSelectCount] = pAudioSelect;
                pParams->nAudioSelectCount++;
            }
            argData->nParsedAudioSamplerate++;
        } else {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-resampler"))) {
        i++;
        int v = 0;
        if (PARSE_ERROR_FLAG != (v = get_value_from_chr(list_resampler, strInput[i]))) {
            pParams->nAudioResampler = v;
        } else if (1 == _stscanf_s(strInput[i], _T("%d"), &v) && 0 <= v && v < _countof(list_resampler) - 1) {
            pParams->nAudioResampler = v;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-stream"))) {
        if (!check_avcodec_dll()) {
            _ftprintf(stderr, _T("%s\n--audio-stream could not be used.\n"), error_mes_avcodec_dll_not_found().c_str());
            return 1;
        }
        int trackId = -1;
        const TCHAR *ptr = nullptr;
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            ptr = _tcschr(strInput[i], _T('?'));
            if (ptr != nullptr) {
                tstring temp = tstring(strInput[i]).substr(0, ptr - strInput[i]);
                if (1 != _stscanf(temp.c_str(), _T("%d"), &trackId)) {
                    PrintHelp(strInput[0], _T("Invalid value"), option_name);
                    return 1;
                }
                ptr++;
            } else {
                ptr = strInput[i];
            }
        }
        if (trackId < 0) {
            trackId = argData->nParsedAudioSplit+1;
            int idx = getAudioTrackIdx(pParams, trackId);
            if (idx >= 0 && bSplitChannelsEnabled(pParams->ppAudioSelectList[idx]->pnStreamChannelSelect)) {
                trackId = getFreeAudioTrack(pParams);
            }
        }
        sAudioSelect *pAudioSelect = nullptr;
        int audioIdx = getAudioTrackIdx(pParams, trackId);
        if (audioIdx < 0) {
            pAudioSelect = (sAudioSelect *)calloc(1, sizeof(pAudioSelect[0]));
            pAudioSelect->nAudioSelect = trackId;
        } else {
            pAudioSelect = pParams->ppAudioSelectList[audioIdx];
        }
        if (ptr == nullptr) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return 1;
        } else {
            auto streamSelectList = split(tchar_to_string(ptr), ",");
            if (streamSelectList.size() > _countof(pAudioSelect->pnStreamChannelSelect)) {
                PrintHelp(strInput[0], _T("Too much streams splitted"), option_name);
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
        if (audioIdx < 0) {
            audioIdx = pParams->nAudioSelectCount;
            //新たに要素を追加
            pParams->ppAudioSelectList = (sAudioSelect **)realloc(pParams->ppAudioSelectList, sizeof(pParams->ppAudioSelectList[0]) * (pParams->nAudioSelectCount + 1));
            pParams->ppAudioSelectList[pParams->nAudioSelectCount] = pAudioSelect;
            pParams->nAudioSelectCount++;
        }
        argData->nParsedAudioSplit++;
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("audio-filter"))) {
        if (i+1 < nArgNum) {
            const TCHAR *ptr = nullptr;
            const TCHAR *ptrDelim = nullptr;
            if (strInput[i+1][0] != _T('-')) {
                i++;
                ptrDelim = _tcschr(strInput[i], _T('?'));
                ptr = (ptrDelim == nullptr) ? strInput[i] : ptrDelim+1;
            } else {
                PrintHelp(strInput[0], _T("Invalid value"), option_name);
                return 1;
            }
            int trackId = 1;
            if (ptrDelim == nullptr) {
                trackId = argData->nParsedAudioFilter+1;
                int idx = getAudioTrackIdx(pParams, trackId);
                if (idx >= 0 && pParams->ppAudioSelectList[idx]->pAudioFilter != nullptr) {
                    trackId = getFreeAudioTrack(pParams);
                }
            } else {
                tstring temp = tstring(strInput[i]).substr(0, ptrDelim - strInput[i]);
                if (1 != _stscanf(temp.c_str(), _T("%d"), &trackId)) {
                    PrintHelp(strInput[0], _T("Invalid value"), option_name);
                    return 1;
                }
            }
            sAudioSelect *pAudioSelect = nullptr;
            int audioIdx = getAudioTrackIdx(pParams, trackId);
            if (audioIdx < 0) {
                pAudioSelect = (sAudioSelect *)calloc(1, sizeof(pAudioSelect[0]));
                pAudioSelect->nAudioSelect = trackId;
            } else {
                pAudioSelect = pParams->ppAudioSelectList[audioIdx];
            }
            if (pAudioSelect->pAudioFilter) {
                free(pAudioSelect->pAudioFilter);
            }
            pAudioSelect->pAudioFilter = _tcsdup(ptr);

            if (audioIdx < 0) {
                audioIdx = pParams->nAudioSelectCount;
                //新たに要素を追加
                pParams->ppAudioSelectList = (sAudioSelect **)realloc(pParams->ppAudioSelectList, sizeof(pParams->ppAudioSelectList[0]) * (pParams->nAudioSelectCount + 1));
                pParams->ppAudioSelectList[pParams->nAudioSelectCount] = pAudioSelect;
                pParams->nAudioSelectCount++;
            }
            argData->nParsedAudioFilter++;
        } else {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return 1;
        }
        return 0;
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
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
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
                    PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
    if (0 == _tcscmp(option_name, _T("avsync"))) {
        int value = 0;
        i++;
        if (PARSE_ERROR_FLAG != (value = get_value_from_chr(list_avsync, strInput[i]))) {
            pParams->nAVSyncMode = (RGYAVSync)value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        return 0;
    }
    if (0 == _tcscmp(option_name, _T("mux-option"))) {
        if (i+1 < nArgNum && strInput[i+1][0] != _T('-')) {
            i++;
            auto ptr = _tcschr(strInput[i], ':');
            if (ptr == nullptr) {
                PrintHelp(strInput[0], _T("invalid value"), option_name);
                return 1;
            } else {
                if (pParams->pMuxOpt == nullptr) {
                    pParams->pMuxOpt = new muxOptList();
                }
                pParams->pMuxOpt->push_back(std::make_pair<tstring, tstring>(tstring(strInput[i]).substr(0, ptr - strInput[i]), tstring(ptr+1)));
            }
        } else {
            PrintHelp(strInput[0], _T("invalid option"), option_name);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
    if (IS_OPTION("cbrhq")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->encConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR_HQ;
            pParams->encConfig.rcParams.averageBitRate = value * 1000;
            pParams->encConfig.rcParams.maxBitRate = value * 1000;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
        }
        if (IS_OPTION("qp-max")) {
            pParams->encConfig.rcParams.enableMaxQP = 1;
            ptrQP = &pParams->encConfig.rcParams.maxQP;
        }
        if (IS_OPTION("qp-min")) {
            pParams->encConfig.rcParams.enableMinQP = 1;
            ptrQP = &pParams->encConfig.rcParams.minQP;
        }
        if (ptrQP == nullptr) {
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
        if (   2 == _stscanf_s(strInput[i], _T("%d:%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d/%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d.%d"), &a[0], &a[1])
            || 2 == _stscanf_s(strInput[i], _T("%d,%d"), &a[0], &a[1])) {
            ptrQP->qpIntra  = a[0];
            ptrQP->qpInterP = a[1];
            ptrQP->qpInterB = a[1];
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
    if (IS_OPTION("lookahead")) {
        i++;
        int value = 0;
        if (1 == _stscanf_s(strInput[i], _T("%d"), &value)) {
            pParams->encConfig.rcParams.enableLookahead = 1;
            pParams->encConfig.rcParams.lookaheadDepth = (uint16_t)clamp(value, 0, 32);
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
                if (param_arg == _T("radius")) {
                    try {
                        pParams->vpp.unsharp.radius = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("weight")) {
                    try {
                        pParams->vpp.unsharp.weight = std::stof(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("threshold")) {
                    try {
                        pParams->vpp.unsharp.threshold = std::stof(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
                if (param_arg == _T("strength")) {
                    try {
                        pParams->vpp.edgelevel.strength = std::stof(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("threshold")) {
                    try {
                        pParams->vpp.edgelevel.threshold = std::stof(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("black")) {
                    try {
                        pParams->vpp.edgelevel.black = std::stof(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("white")) {
                    try {
                        pParams->vpp.edgelevel.white = std::stof(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
            }
        }
        return 0;
    }
    if (IS_OPTION("vpp-delogo")
        || IS_OPTION("vpp-delogo-file")) {
        i++;
        pParams->vpp.delogo.pFilePath = _tcsdup(strInput[i]);
        return 0;
    }
    if (IS_OPTION("vpp-delogo-select")) {
        i++;
        pParams->vpp.delogo.pSelect = _tcsdup(strInput[i]);
        return 0;
    }
    if (IS_OPTION("vpp-delogo-add")) {
        pParams->vpp.delogo.nMode = DELOGO_MODE_ADD;
        return 0;
    }
    if (IS_OPTION("vpp-delogo-pos")) {
        i++;
        int posOffsetX, posOffsetY;
        if (   2 != _stscanf_s(strInput[i], _T("%dx%d"), &posOffsetX, &posOffsetY)
            && 2 != _stscanf_s(strInput[i], _T("%d,%d"), &posOffsetX, &posOffsetY)
            && 2 != _stscanf_s(strInput[i], _T("%d/%d"), &posOffsetX, &posOffsetY)
            && 2 != _stscanf_s(strInput[i], _T("%d:%d"), &posOffsetX, &posOffsetY)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        pParams->vpp.delogo.nPosOffsetX = posOffsetX;
        pParams->vpp.delogo.nPosOffsetY = posOffsetY;
        return 0;
    }
    if (IS_OPTION("vpp-delogo-depth")) {
        i++;
        int depth;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &depth)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        pParams->vpp.delogo.nDepth = depth;
        return 0;
    }
    if (IS_OPTION("vpp-delogo-y")) {
        i++;
        int value;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        pParams->vpp.delogo.nYOffset = value;
        return 0;
    }
    if (IS_OPTION("vpp-delogo-cb")) {
        i++;
        int value;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        pParams->vpp.delogo.nCbOffset = value;
        return 0;
    }
    if (IS_OPTION("vpp-delogo-cr")) {
        i++;
        int value;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        pParams->vpp.delogo.nCrOffset = value;
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
                    if (param_arg == _T("radius")) {
                        try {
                            pParams->vpp.knn.radius = std::stoi(param_val);
                        } catch (...) {
                            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                            return -1;
                        }
                        continue;
                    }
                    if (param_arg == _T("strength")) {
                        try {
                            pParams->vpp.knn.strength = std::stof(param_val);
                        } catch (...) {
                            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                            return -1;
                        }
                        continue;
                    }
                    if (param_arg == _T("lerp")) {
                        try {
                            pParams->vpp.knn.lerpC = std::stof(param_val);
                        } catch (...) {
                            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                            return -1;
                        }
                        continue;
                    }
                    if (param_arg == _T("th_weight")) {
                        try {
                            pParams->vpp.knn.weight_threshold = std::stof(param_val);
                        } catch (...) {
                            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                            return -1;
                        }
                        continue;
                    }
                    if (param_arg == _T("th_lerp")) {
                        try {
                            pParams->vpp.knn.lerp_threshold = std::stof(param_val);
                        } catch (...) {
                            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                            return -1;
                        }
                        continue;
                    }
                    PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
                if (param_arg == _T("apply_count")) {
                    try {
                        pParams->vpp.pmd.applyCount = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("strength")) {
                    try {
                        pParams->vpp.pmd.strength = std::stof(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("threshold")) {
                    try {
                        pParams->vpp.pmd.threshold = std::stof(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("useexp")) {
                    try {
                        pParams->vpp.pmd.useExp = std::stoi(param_val) != 0;
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
                if (param_arg == _T("range")) {
                    try {
                        pParams->vpp.deband.range = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_y")) {
                    try {
                        pParams->vpp.deband.threY = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_cb")) {
                    try {
                        pParams->vpp.deband.threCb = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_cr")) {
                    try {
                        pParams->vpp.deband.threCr = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("dither")) {
                    try {
                        pParams->vpp.deband.ditherY = std::stoi(param_val);
                        pParams->vpp.deband.ditherC = pParams->vpp.deband.ditherY;
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("dither_y")) {
                    try {
                        pParams->vpp.deband.ditherY = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("dither_c")) {
                    try {
                        pParams->vpp.deband.ditherC = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("sample")) {
                    try {
                        pParams->vpp.deband.sample = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("seed")) {
                    try {
                        pParams->vpp.deband.seed = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
                PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
                PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
                        PrintHelp(strInput[0], _T("ini file does not exist."), option_name, strInput[i]);
                        return -1;
                    }
                }
                if (param_arg == _T("preset")) {
                    try {
                        int value = 0;
                        if (get_list_value(list_afs_preset, param_val.c_str(), &value)) {
                            NVEncFilterAfs::set_preset(&pParams->vpp.afs, value);
                        } else {
                            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                            return -1;
                        }
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
                if (param_arg == _T("top")) {
                    try {
                        pParams->vpp.afs.clip.top = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("bottom")) {
                    try {
                        pParams->vpp.afs.clip.bottom = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("left")) {
                    try {
                        pParams->vpp.afs.clip.left = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("right")) {
                    try {
                        pParams->vpp.afs.clip.right = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("method_switch")) {
                    try {
                        pParams->vpp.afs.method_switch = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("coeff_shift")) {
                    try {
                        pParams->vpp.afs.coeff_shift = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_shift")) {
                    try {
                        pParams->vpp.afs.thre_shift = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_deint")) {
                    try {
                        pParams->vpp.afs.thre_deint = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_motion_y")) {
                    try {
                        pParams->vpp.afs.thre_Ymotion = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("thre_motion_c")) {
                    try {
                        pParams->vpp.afs.thre_Cmotion = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("level")) {
                    try {
                        pParams->vpp.afs.analyze = std::stoi(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
                PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
                PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
                if (param_arg == _T("brightness")) {
                    try {
                        pParams->vpp.tweak.brightness = std::stof(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("contrast")) {
                    try {
                        pParams->vpp.tweak.contrast = std::stof(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("gamma")) {
                    try {
                        pParams->vpp.tweak.gamma = std::stof(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("saturation")) {
                    try {
                        pParams->vpp.tweak.saturation = std::stof(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                if (param_arg == _T("hue")) {
                    try {
                        pParams->vpp.tweak.hue = std::stof(param_val);
                    } catch (...) {
                        PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                        return -1;
                    }
                    continue;
                }
                PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
            } else {
                PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
                return -1;
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
        if (   0 == memcmp(&pParams->encConfig.profileGUID, &NV_ENC_H264_PROFILE_HIGH_444_GUID, sizeof(result_guid))
            || codecPrm[NV_ENC_HEVC].hevcConfig.tier == NV_ENC_TIER_HEVC_MAIN444) {
            pParams->yuv444 = TRUE;
        }
        if (codecPrm[NV_ENC_HEVC].hevcConfig.tier == NV_ENC_TIER_HEVC_MAIN10) {
            codecPrm[NV_ENC_HEVC].hevcConfig.pixelBitDepthMinus8 = 2;
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
        if (get_list_value(list_hevc_cu_size, strInput[i], &value)) {
            codecPrm[NV_ENC_HEVC].hevcConfig.minCUSize = (NV_ENC_HEVC_CUSIZE)value;
        } else {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        return 0;
    }
    if (IS_OPTION("max-procfps")) {
        i++;
        int value = 0;
        if (1 != _stscanf_s(strInput[i], _T("%d"), &value)) {
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return -1;
        }
        if (value < 0) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        if (value < 0) {
            PrintHelp(strInput[0], _T("Invalid value"), option_name);
            return 1;
        }
        pParams->nOutputBufSizeMB = (std::min)(value, RGY_OUTPUT_BUF_MB_MAX);
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
        pParams->nOutputThread = value;
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
        pParams->nAudioThread = value;
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
    if (0 == _tcscmp(option_name, _T("perf-monitor"))) {
        if (strInput[i+1][0] == _T('-') || _tcslen(strInput[i+1]) == 0) {
            pParams->nPerfMonitorSelect = (int)PERF_MONITOR_ALL;
        } else {
            i++;
            auto items = split(strInput[i], _T(","));
            for (const auto& item : items) {
                int value = 0;
                if (PARSE_ERROR_FLAG == (value = get_value_from_chr(list_pref_monitor, item.c_str()))) {
                    PrintHelp(item.c_str(), _T("Unknown value"), option_name);
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
            PrintHelp(strInput[0], _T("Unknown value"), option_name, strInput[i]);
            return 1;
        }
        pParams->nPerfMonitorInterval = std::max(50, v);
        return 0;
    }
    tstring mes = _T("Unknown option: --");
    mes += option_name;
    PrintHelp(strInput[0], (TCHAR *)mes.c_str(), NULL, strInput[i]);
    return -1;
}

int parse_cmd(InEncodeVideoParam *pParams, NV_ENC_CODEC_CONFIG *codecPrm, int nArgNum, const TCHAR **strInput) {

    if (nArgNum == 1) {
        show_version();
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
            show_version();
            show_help();
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
        auto sts = parse_one_option(option_name, strInput, i, nArgNum, pParams, codecPrm, &argsData);
        if (sts != 0) {
            return sts;
        }
    }

#undef IS_OPTION
    //オプションチェック
    if (0 == pParams->inputFilename.length()) {
        _ftprintf(stderr, _T("Input file is not specified.\n"));
        return -1;
    }
    if (0 == pParams->outputFilename.length()) {
        _ftprintf(stderr, _T("Output file is not specified.\n"));
        return -1;
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
            _ftprintf(stderr, _T("Unknown Output codec.\n"));
            return -1;
            break;
        }
    }
    pParams->encConfig.frameIntervalP = argsData.nBframes + 1;

    return 0;
}

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

    InEncodeVideoParam encPrm;
    NV_ENC_CODEC_CONFIG codecPrm[2] = { 0 };
    codecPrm[NV_ENC_H264] = NVEncCore::DefaultParamH264();
    codecPrm[NV_ENC_HEVC] = NVEncCore::DefaultParamHEVC();

    vector<const TCHAR *> argvCopy(argv, argv + argc);
    argvCopy.push_back(_T(""));
    if (parse_cmd(&encPrm, codecPrm, argc, argvCopy.data())) {
        return 1;
    }

#if defined(_WIN32) || defined(_WIN64)
    //set stdin to binary mode when using pipe input
    if (_tcscmp(encPrm.inputFilename.c_str(), _T("-")) == NULL) {
        if (_setmode(_fileno(stdin), _O_BINARY) == 1) {
            PrintHelp(argv[0], _T("failed to switch stdin to binary mode."), NULL);
            return 1;
        }
    }

    //set stdout to binary mode when using pipe output
    if (_tcscmp(encPrm.outputFilename.c_str(), _T("-")) == NULL) {
        if (_setmode(_fileno(stdout), _O_BINARY) == 1) {
            PrintHelp(argv[0], _T("failed to switch stdout to binary mode."), NULL);
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
#if ENABLE_AVSW_READER
    avformatNetworkDeinit();
#endif //#if ENABLE_AVCODEC_QSV_READER
    return ret;
}
