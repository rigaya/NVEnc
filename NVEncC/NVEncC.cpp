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
#include "NVEncCmd.h"
#include "rgy_util.h"

#if ENABLE_CPP_REGEX
#include <regex>
#endif //#if ENABLE_CPP_REGEX
#if ENABLE_DTL
#include <dtl/dtl.hpp>
#endif //#if ENABLE_DTL

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
        _T("   --check-profiles <string>    show profile names available for specified codec\n")
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
        _T("   --sub-copy [<int>[,...]]     copy subtitle to output file.\n")
        _T("                                 these could be only used with\n")
        _T("                                 avhw/avsw reader and avcodec muxer.\n")
        _T("                                 below are optional,\n")
        _T("                                  in [<int>?], specify track number to copy.\n")
        _T("   --caption2ass [<string>]     enable caption2ass during encode.\n")
        _T("                                  !! This feature requires Caption.dll !!\n")
        _T("                                 supported formats ... srt (default), ass\n")
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
        _T("   --tier <string>              set codec tier\n")
        _T("                                  HEVC : main, high\n")
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
        _T("-u,--preset <string>            set encoder preset\n")
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
        _T("                                  default: %d frames\n")
        _T("   --ref <int>                  set Ref frames / default %d frames\n")
        _T("   --weightp                    enable weighted prediction for P frame\n")
        _T("   --mv-precision <string>      set MV Precision / default: auto\n")
        _T("                                  auto,\n")
        _T("                                  Q-pel (High Quality),\n")
        _T("                                  half-pel,\n")
        _T("                                  full-pel (Low Quality, not recommended)\n")
        _T("   --slices <int>               number of slices, default 0 (auto)\n")
        _T("   --vbv-bufsize <int>          set vbv buffer size (kbit) / default: auto\n")
        _T("   --(no-)aq                    enable spatial adaptive quantization\n")
        _T("   --aq-temporal                [H264] enable temporal adaptive quantization\n")
        _T("   --aq-strength <int>          [H264] set aq strength (weak 1 - 15 strong)\n")
        _T("                                  default: 0 = auto\n")
        _T("   --bref-mode <string>         set B frame reference mode\n")
        _T("                                  - disabled (default)\n")
        _T("                                  - each\n")
        _T("                                  - middle\n")
        _T("   --direct <string>            [H264] set B Direct mode\n")
        _T("                                  auto(default), none, spatial, temporal\n")
        _T("   --(no-)adapt-transform       [H264] set adaptive transform mode (default=auto)\n"),
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
        _T("             leaving it auto will enhance video quality.\n"));

    str += PrintListOptions(_T("--videoformat <string>"), list_videoformat, 0);
    str += PrintListOptions(_T("--colormatrix <string>"), list_colormatrix, 0);
    str += PrintListOptions(_T("--colorprim <string>"), list_colorprim, 0);
    str += PrintListOptions(_T("--transfer <string>"), list_transfer, 0);
    str += strsprintf(_T("")
        _T("   --aud                        insert aud nal unit to ouput stream.\n")
        _T("   --pic-struct                 insert pic-timing SEI with pic_struct.\n")
        _T("   --chromaloc <int>            set chroma location flag [ 0 ... 5 ]\n")
        _T("                                  default: 0 = unspecified\n")
        _T("   --fullrange                  set fullrange\n")
        _T("   --max-cll <int>,<int>        set MaxCLL/MaxFall in nits. e.g. \"1000,300\"\n")
        _T("   --master-display <string>    set Mastering display data.\n")
        _T("   e.g. \"G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)\"\n")
        _T("   --dhdr10-info <string>       apply dynamic HDR10+ metadata from json file.\n"));

    str += strsprintf(_T("\n")
        _T("   --interlace <string>         set input as interlaced\n")
        _T("                                  tff, bff\n")
        _T("   --vpp-deinterlace <string>   set deinterlace mode / default: none\n")
        _T("                                  none, bob, adaptive (normal)\n")
        _T("                                  available only with avhw reader\n"));
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
        _T("   --vpp-nnedi [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     enable nnedi deinterlacer\n")
        _T("    params\n")
        _T("      field=<string>\n")
        _T("          auto (default)    Generate latter field from first field.\n")
        _T("          top               Generate bottom field using top field.\n")
        _T("          bottom            Generate top field using bottom field.\n")
        _T("      nns=<int>             Neurons of neural net (default: 32)\n")
        _T("                              16, 32, 64, 128, 256\n")
        _T("      nszie=<int>x<int>     Area size neural net uses to generate a pixel.\n")
        _T("                              8x6, 16x6, 32x6, 48x6, 8x4, 16x4, 32x4(default)\n")
        _T("      quality=<string>      quality settings\n")
        _T("                              fast (default), slow\n")
        _T("      prescreen=<string>    (default: new_block)\n")
        _T("          none              No pre-screening is done and all pixels will be\n")
        _T("                            generated by neural net.\n")
        _T("          original          Runs prescreener to determine which pixel to apply\n")
        _T("          new               neural net, other pixels will be generated from\n")
        _T("                            simple interpolation.\n")
        _T("          original_block    GPU optimized ver of original/new.\n")
        _T("          new_block\n")
        _T("      errortype=<string>    Select weight parameter for neural net.\n")
        _T("                              abs (default), square\n")
        _T("      prec=<string>         Select calculation precision.\n")
        _T("                              auto (default), fp16, fp32\n")
        _T("      weightfile=<string>   Set path of weight file. By default (not specified),\n")
        _T("                              internal weight params will be used.\n"));
    str += strsprintf(_T("\n")
        _T("   --vpp-yadif [<param1>=<value>]\n")
        _T("     enable yadif deinterlacer\n")
        _T("    params\n")
        _T("      mode=<string>\n")
        _T("          auto (default)    Generate latter field using first field.\n")
        _T("          tff               Generate bottom field using top field.\n")
        _T("          bff               Generate top field using bottom field.\n")
        _T("          bob               Generate one frame from each field.\n")
        _T("          bob_tff           Generate one frame from each field assuming tff.\n")
        _T("          bob_bff           Generate one frame from each field assuming bff.\n"));
    str += strsprintf(_T("\n")
        _T("   --vpp-rff                    apply rff flag, with avhw reader only.\n"));
#if ENABLE_NVRTC
    str += strsprintf(_T("\n")
        _T("   --vpp-colorspace [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     Converts colorspace of the video.\n")
        _T("    params\n")
        _T("      matrix=<from>:<to>\n")
        _T("        bt709, smpte170m, bt470bg, smpte240m, YCgCo, fcc, GBR,\n")
        _T("        bt2020nc, bt2020c\n")
        _T("      colorprim=<from>:<to>\n")
        _T("        bt709, smpte170m, bt470m, bt470bg, smpte240m, film, bt2020\n")
        _T("      transfer=<from>:<to>\n")
        _T("        bt709, smpte170m, bt470m, bt470bg, smpte240m, linear,\n")
        _T("        log100, log316, iec61966-2-4, iec61966-2-1,\n")
        _T("        bt2020-10, bt2020-12, smpte2084, arib-srd-b67\n")
        _T("      range=<from>:<to>\n")
        _T("        limited, full\n")
        _T("      hdr2sdr=<string>     Enables HDR10 to SDR.\n")
        _T("                             hable, mobius, reinhard, none\n")
        _T("      source_peak=<float>  (default: 1000.0)\n")
        _T("      ldr_nits=<float>  (default: 100.0)\n"));
#endif //#if ENABLE_NVRTC
    str += PrintListOptions(_T("--vpp-resize <string>"),     list_nppi_resize_help, 0);
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
    str += strsprintf(_T("\n")
        _T("   --vpp-pad <int>,<int>,<int>,<int>\n")
        _T("     add padding to left,top,right,bottom (in pixels)\n"));
    str += strsprintf(_T("\n")
        _T("   --vpp-select-every <int>[,offset=<int>]\n")
        _T("     select one frame per specified frames and create output.\n"));
    str += strsprintf(_T("\n")
        _T("   --vpp-subrbun [<param1>=<value>][,<param2>=<value>][...]\n")
        _T("     Burn in specified subtitle to the video.\n")
        _T("    params\n")
        _T("      track=<int>               subtitle track of the input file to burn in.\n")
        _T("      filename=<string>         subtitle file path to burn in.\n")
        _T("      charcode=<string>         subtitle charcter code.\n")
        _T("      shaping=<string>          rendering quality of text.\n"));
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

static void PrintHelp(tstring strAppName, tstring strErrorMessage, tstring strOptionName, tstring strErrorValue) {
    UNREFERENCED_PARAMETER(strAppName);

    if (strErrorMessage.length() > 0) {
        if (strOptionName.length() > 0) {
            if (strErrorValue.length() > 0) {
                _ftprintf(stderr, _T("Error: %s \"%s\" for \"--%s\"\n"), strErrorMessage.c_str(), strErrorValue.c_str(), strOptionName.c_str());
                if (0 == _tcsnccmp(strErrorValue.c_str(), _T("--"), _tcslen(_T("--")))
                    || (strErrorValue[0] == _T('-') && strErrorValue[2] == _T('\0') && cmd_short_opt_to_long(strErrorValue[1]) != nullptr)) {
                    _ftprintf(stderr, _T("       \"--%s\" requires value.\n\n"), strOptionName.c_str());
                }
            } else {
                _ftprintf(stderr, _T("Error: %s for --%s\n\n"), strErrorMessage.c_str(), strOptionName.c_str());
            }
        } else {
            _ftprintf(stderr, _T("Error: %s\n\n"), strErrorMessage.c_str());
#if (ENABLE_CPP_REGEX && ENABLE_DTL)
            if (strErrorValue.length() > 0) {
                //どのオプション名に近いか検証する
                auto optList = createOptionList();
                const auto invalid_opt = tchar_to_string(strErrorValue.c_str());
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
    nvFeature.createCacheAsync(deviceid, RGY_LOG_DEBUG);
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
    show_version();
    _ftprintf(stdout, _T("%s\n"), getEnviromentInfo(false).c_str());
}

static void show_nvenc_features(int deviceid) {
    NVEncFeature nvFeature;
    if (nvFeature.createCacheAsync(deviceid, RGY_LOG_INFO)) {
        _ftprintf(stdout, _T("error on checking features.\n"));
        return;
    }
    auto nvEncCaps = nvFeature.GetCachedNVEncCapability();

    show_version();
    _ftprintf(stdout, _T("\n%s\n"), getEnviromentInfo(false).c_str());
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
        show_hw(deviceid);
        return 1;
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


bool check_locale_is_ja() {
    const WORD LangID_ja_JP = MAKELANGID(LANG_JAPANESE, SUBLANG_JAPANESE_JAPAN);
    return GetUserDefaultLangID() == LangID_ja_JP;
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

    ParseCmdError err;
    InEncodeVideoParam encPrm;
    NV_ENC_CODEC_CONFIG codecPrm[2] = { 0 };
    codecPrm[NV_ENC_H264] = DefaultParamH264();
    codecPrm[NV_ENC_HEVC] = DefaultParamHEVC();

    vector<const TCHAR *> argvCopy(argv, argv + argc);
    argvCopy.push_back(_T(""));
    if (parse_cmd(&encPrm, codecPrm, argc, argvCopy.data(), err)) {
        PrintHelp(err.strAppName, err.strErrorMessage, err.strOptionName, err.strErrorValue);
        return 1;
    }
    //オプションチェック
    if (0 == encPrm.inputFilename.length()) {
        _ftprintf(stderr, _T("Input file is not specified.\n"));
        return -1;
    }
    if (0 == encPrm.outputFilename.length()) {
        _ftprintf(stderr, _T("Output file is not specified.\n"));
        return -1;
    }

#if defined(_WIN32) || defined(_WIN64)
    //set stdin to binary mode when using pipe input
    if (_tcscmp(encPrm.inputFilename.c_str(), _T("-")) == NULL) {
        if (_setmode(_fileno(stdin), _O_BINARY) == 1) {
            PrintHelp(argv[0], _T("failed to switch stdin to binary mode."), _T(""), _T(""));
            return 1;
        }
    }

    //set stdout to binary mode when using pipe output
    if (_tcscmp(encPrm.outputFilename.c_str(), _T("-")) == NULL) {
        if (_setmode(_fileno(stdout), _O_BINARY) == 1) {
            PrintHelp(argv[0], _T("failed to switch stdout to binary mode."), _T(""), _T(""));
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
