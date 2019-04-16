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
// ------------------------------------------------------------------------------------------

#include <numeric>
#include "rgy_version.h"

#if ENABLE_AVSW_READER && !FOR_AUO

#include "rgy_avutil.h"

extern "C" {
#include <libavutil/timestamp.h>
}

int64_t rational_rescale(int64_t v, rgy_rational<int> from, rgy_rational<int> to) {
    return av_rescale_q(v, av_make_q(from), av_make_q(to));
}

//必要なavcodecのdllがそろっているかを確認
bool check_avcodec_dll() {
#if defined(_WIN32) || defined(_WIN64)
    std::vector<HMODULE> hDllList;
    bool check = true;
    for (int i = 0; i < _countof(AVCODEC_DLL_NAME); i++) {
        HMODULE hDll = NULL;
        if (NULL == (hDll = LoadLibrary(AVCODEC_DLL_NAME[i]))) {
            check = false;
            break;
        }
        hDllList.push_back(hDll);
    }
    for (auto hDll : hDllList) {
        FreeLibrary(hDll);
    }
    return check;
#else
    return true;
#endif //#if defined(_WIN32) || defined(_WIN64)
}

//avcodecのdllが存在しない場合のエラーメッセージ
tstring error_mes_avcodec_dll_not_found() {
    tstring mes;
    mes += _T("avcodec: failed to load dlls.\n");
    mes += _T("please make sure ");
    for (int i = 0; i < _countof(AVCODEC_DLL_NAME); i++) {
        if (i) mes += _T(", ");
        if (i % 3 == 2) {
            mes += _T("\n");
        }
        mes += _T("\"") + tstring(AVCODEC_DLL_NAME[i]) + _T("\"");
    }
    mes += _T("\nis installed in your system.\n");
    return mes;
}

//avcodecのライセンスがLGPLであるかどうかを確認
bool checkAvcodecLicense() {
    auto check = [](const char *license) {
        std::string str(license);
        transform(str.begin(), str.end(), str.begin(), [](char in) -> char {return (char)tolower(in); });
        return std::string::npos != str.find("lgpl");
    };
    return (check(avutil_license()) && check(avcodec_license()) && check(avformat_license()));
}

//mfxFrameInfoから、AVFieldOrderを返す
AVFieldOrder picstrcut_rgy_to_avfieldorder(RGY_PICSTRUCT picstruct) {
    if (picstruct & RGY_PICSTRUCT_TFF) {
        return AV_FIELD_TT;
    }
    if (picstruct & RGY_PICSTRUCT_BFF) {
        return AV_FIELD_BB;
    }
    return AV_FIELD_PROGRESSIVE;
}

//avcodecのエラーを表示
tstring qsv_av_err2str(int ret) {
    char mes[256];
    av_make_error_string(mes, sizeof(mes), ret);
    return char_to_tstring(mes);
}

//コーデックの種類を表示
tstring get_media_type_string(AVCodecID codecId) {
    return char_to_tstring(av_get_media_type_string(avcodec_get_type(codecId))).c_str();
}

//avqsvでサポートされている動画コーデックを表示
tstring getHWDecSupportedCodecList() {
    tstring codecs;
    for (int i = 0; i < _countof(HW_DECODE_LIST); i++) {
        if (i == 0 || HW_DECODE_LIST[i-1].rgy_codec != HW_DECODE_LIST[i].rgy_codec) {
            if (i) codecs += _T(", ");
            codecs += CodecToStr(HW_DECODE_LIST[i].rgy_codec);
        }
    }
    return codecs;
}

//利用可能な音声エンコーダ/デコーダを表示
tstring getAVCodecs(RGYAVCodecType flag) {
    if (!check_avcodec_dll()) {
        return error_mes_avcodec_dll_not_found();
    }
    av_register_all();
    avcodec_register_all();

    struct avcodecName {
        uint32_t type;
        const char *name;
        const char *long_name;
    };

    vector<avcodecName> list;

    AVCodec *codec = nullptr;
    while (nullptr != (codec = av_codec_next(codec))) {
        if (codec->type == AVMEDIA_TYPE_AUDIO || codec->type == AVMEDIA_TYPE_SUBTITLE) {
            bool alreadyExists = false;
            for (uint32_t i = 0; i < list.size(); i++) {
                if (0 == strcmp(list[i].name, codec->name)) {
                    list[i].type |= codec->decode ? RGY_AVCODEC_DEC : 0x00;
                    list[i].type |= codec->encode2 ? RGY_AVCODEC_ENC : 0x00;
                    alreadyExists = true;
                    break;
                }
            }
            if (!alreadyExists) {
                uint32_t type = 0x00;
                type |= codec->decode ? RGY_AVCODEC_DEC : 0x00;
                type |= codec->encode2 ? RGY_AVCODEC_ENC : 0x00;
                list.push_back({ type, codec->name, codec->long_name });
            }
        }
    }

    std::sort(list.begin(), list.end(), [](const avcodecName& x, const avcodecName& y) {
        int i = 0;
        for (; x.name[i] && y.name[i]; i++) {
            if (x.name[i] != y.name[i]) {
                return x.name[i] < y.name[i];
            }
        }
        return x.name[i] < y.name[i];
    });
    uint32_t maxNameLength = 0;
    std::for_each(list.begin(), list.end(), [&maxNameLength](const avcodecName& format) { maxNameLength = (std::max)(maxNameLength, (uint32_t)strlen(format.name)); });
    maxNameLength = (std::min)(maxNameLength, 15u);

    uint32_t flag_dec = flag & RGY_AVCODEC_DEC;
    uint32_t flag_enc = flag & RGY_AVCODEC_ENC;
    int flagCount = popcnt32(flag);

    std::string codecstr = (flagCount > 1) ? "D-: Decode\n-E: Encode\n---------------------\n" : "";
    std::for_each(list.begin(), list.end(), [&codecstr, maxNameLength, flagCount, flag_dec, flag_enc](const avcodecName& format) {
        if (format.type & (flag_dec | flag_enc)) {
            if (flagCount > 1) {
                codecstr += (format.type & flag_dec) ? "D" : "-";
                codecstr += (format.type & flag_enc) ? "E" : "-";
                codecstr += " ";
            }
            codecstr += format.name;
            if (format.long_name) {
                for (uint32_t i = (uint32_t)strlen(format.name); i <= maxNameLength; i++)
                    codecstr += " ";
                codecstr += ": " + std::string(format.long_name);
            }
            codecstr += "\n";
        }
    });

    return char_to_tstring(codecstr);
}

std::vector<tstring> getAudioPofileList(const tstring& codec_name) {
    std::vector<tstring> profiles;
    auto codec_name_s = tchar_to_string(codec_name);
    auto codec = avcodec_find_encoder_by_name(codec_name_s.c_str());
    if (codec) {
        auto codecDesc = avcodec_descriptor_get(codec->id);
        if (codecDesc) {
            for (auto avprofile = codecDesc->profiles;
                avprofile != nullptr && avprofile->profile != FF_PROFILE_UNKNOWN;
                avprofile++) {
                profiles.push_back(char_to_tstring(avprofile->name));
            }
        }
        if (profiles.size() == 0) {
            profiles.push_back(_T("none"));
        }
    }
    return profiles;
}

//利用可能なフォーマットを表示
tstring getAVFormats(RGYAVFormatType flag) {
    if (!check_avcodec_dll()) {
        return error_mes_avcodec_dll_not_found();
    }
    av_register_all();
    avcodec_register_all();

    struct avformatName {
        uint32_t type;
        const char *name;
        const char *long_name;
    };

    vector<avformatName> list;

    std::string codecstr;
    AVInputFormat *iformat = nullptr;
    while (nullptr != (iformat = av_iformat_next(iformat))) {
        bool alreadyExists = false;
        for (uint32_t i = 0; i < list.size(); i++) {
            if (0 == strcmp(list[i].name, iformat->name)) {
                list[i].type |= RGY_AVFORMAT_DEMUX;
                alreadyExists = true;
                break;
            }
        }
        if (!alreadyExists) {
            list.push_back({ RGY_AVFORMAT_DEMUX, iformat->name, iformat->long_name });
        }
    }

    AVOutputFormat *oformat = nullptr;
    while (nullptr != (oformat = av_oformat_next(oformat))) {
        bool alreadyExists = false;
        for (uint32_t i = 0; i < list.size(); i++) {
            if (0 == strcmp(list[i].name, oformat->name)) {
                list[i].type |= RGY_AVFORMAT_MUX;
                alreadyExists = true;
                break;
            }
        }
        if (!alreadyExists) {
            list.push_back({ RGY_AVFORMAT_MUX, oformat->name, oformat->long_name });
        }
    }

    std::sort(list.begin(), list.end(), [](const avformatName& x, const avformatName& y) {
        int i = 0;
        for (; x.name[i] && y.name[i]; i++) {
            if (x.name[i] != y.name[i]) {
                return x.name[i] < y.name[i];
            }
        }
        return x.name[i] < y.name[i];
    });

    uint32_t maxNameLength = 0;
    std::for_each(list.begin(), list.end(), [&maxNameLength](const avformatName& format) { maxNameLength = (std::max)(maxNameLength, (uint32_t)strlen(format.name)); });
    maxNameLength = (std::min)(maxNameLength, 15u);

    uint32_t flag_demux = flag & RGY_AVFORMAT_DEMUX;
    uint32_t flag_mux = flag & RGY_AVFORMAT_MUX;
    int flagCount = popcnt32(flag);

    std::string formatstr = (flagCount > 1) ? "D-: Demux\n-M: Mux\n---------------------\n" : "";
    std::for_each(list.begin(), list.end(), [&formatstr, maxNameLength, flagCount, flag_demux, flag_mux](const avformatName& format) {
        if (format.type & (flag_demux | flag_mux)) {
            if (flagCount > 1) {
                formatstr += (format.type & flag_demux) ? "D" : "-";
                formatstr += (format.type & flag_mux) ? "M" : "-";
                formatstr += " ";
            }
            formatstr += format.name;
            if (format.long_name) {
                for (uint32_t i = (uint32_t)strlen(format.name); i <= maxNameLength; i++)
                    formatstr += " ";
                formatstr += ": " + std::string(format.long_name);
            }
            formatstr += "\n";
        }
    });

    return char_to_tstring(formatstr);
}

//利用可能なフィルターを表示
tstring getAVFilters() {
    if (!check_avcodec_dll()) {
        return error_mes_avcodec_dll_not_found();
    }
    av_register_all();
    avfilter_register_all();

    struct avfilterName {
        int type;
        const char *name;
        const char *long_name;
    };

    vector<avfilterName> list;
    {
        const AVFilter *filter = nullptr;
        while (nullptr != (filter = avfilter_next(filter))) {
            list.push_back({ filter->flags, filter->name, filter->description });
        }
    }

    std::sort(list.begin(), list.end(), [](const avfilterName& x, const avfilterName& y) {
        int i = 0;
        for (; x.name[i] && y.name[i]; i++) {
            if (x.name[i] != y.name[i]) {
                return x.name[i] < y.name[i];
            }
        }
        return x.name[i] < y.name[i];
    });

    const auto max_len = std::accumulate(list.begin(),  list.end(), (size_t)0, [](const size_t max_len, const avfilterName& filter) { return (std::max)(max_len, strlen(filter.name)); }) + 1;

    std::string mes = "all filters:\n";
    size_t len = 0;
    for (const auto& filter : list) {
        mes += filter.name;
        for (auto i = strlen(filter.name); i < max_len; i++) {
            mes += " ";
        }
        len += max_len;
        if (len >= 79 - max_len) {
            mes += "\n";
            len = 0;
        }
    }
    return char_to_tstring(mes);
}

std::string getChannelLayoutChar(int channels, uint64_t channel_layout) {
    char string[1024] = { 0 };
    av_get_channel_layout_string(string, _countof(string), channels, channel_layout);
    if (auto ptr = strstr(string, " channel")) {
        strcpy(ptr, "ch");
    }
    if (auto ptr = strstr(string, "channel")) {
        strcpy(ptr, "ch");
    }
    if (0 == _strnicmp(string, "stereo", strlen("stereo"))) {
        return "2ch";
    }
    return string;
}

tstring getChannelLayoutString(int channels, uint64_t channel_layout) {
    return char_to_tstring(getChannelLayoutChar(channels, channel_layout));
}

std::string getTimestampChar(int64_t ts, const AVRational& timebase) {
    char buf[AV_TS_MAX_STRING_SIZE];
    AVRational tb = timebase;
    return std::string(av_ts_make_time_string(buf, ts, &tb));
}

tstring getTimestampString(int64_t ts, const AVRational& timebase) {
    return char_to_tstring(getTimestampChar(ts, timebase));
}

uint32_t tagFromStr(std::string tagstr) {
    uint32_t tag = 0x00;
    for (size_t i = 0; i < std::max<size_t>(tagstr.length(), 4); i++) {
        tag |= tagstr[i] << (i*8);
    }
    return tag;
}

std::string tagToStr(uint32_t tag) {
    std::string str;
    for (int i = 0; i < 4; i++) {
        str.push_back((char)((tag >> (i*8)) & 0xff));
    }
    return str;
}

vector<std::string> getAVProtocolList(int bOutput) {
    vector<std::string> protocols;

    void *opaque = nullptr;
    const char *name = nullptr;
    while (nullptr != (name = avio_enum_protocols(&opaque, bOutput))) {
        std::string data = name;
        std::transform(data.begin(), data.end(), data.begin(), ::tolower);
        protocols.push_back(data);
    }
    return protocols;
}

tstring getAVProtocols() {
    if (!check_avcodec_dll()) {
        return error_mes_avcodec_dll_not_found();
    }
    av_register_all();
    avcodec_register_all();

    const auto inputProtocols  = getAVProtocolList(0);
    const auto outputProtocols = getAVProtocolList(1);

    auto max_len = std::accumulate(inputProtocols.begin(),  inputProtocols.end(), (size_t)0, [](const size_t max_len, const std::string& str) { return (std::max)(max_len, str.length()); });
    max_len      = std::accumulate(outputProtocols.begin(), outputProtocols.end(), max_len,  [](const size_t max_len, const std::string& str) { return (std::max)(max_len, str.length()); });
    max_len += 1;

    std::string mes = "input protocols:\n";
    size_t len = 0;
    for (const auto& protocols : inputProtocols) {
        mes += protocols;
        for (auto i = protocols.length(); i < max_len; i++) {
            mes += " ";
        }
        len += max_len;
        if (len >= 79 - max_len) {
            mes += "\n";
            len = 0;
        }
    }
    mes += "\n\noutput protocols:\n";
    len = 0;
    for (const auto& protocols : outputProtocols) {
        mes += protocols;
        for (auto i = protocols.length(); i < max_len; i++) {
            mes += " ";
        }
        len += max_len;
        if (len >= 79 - max_len) {
            mes += "\n";
            len = 0;
        }
    }
    return char_to_tstring(mes);
}

bool usingAVProtocols(std::string filename, int bOutput) {
    if (!check_avcodec_dll()) {
        return false;
    }
    const auto protocolList = getAVProtocolList(bOutput);
    const auto pos = filename.find_first_of(':');
    if (pos != std::string::npos) {
        std::string check = filename.substr(0, pos);
        std::transform(check.begin(), check.end(), check.begin(), tolower);
        if (std::find(protocolList.begin(), protocolList.end(), check) != protocolList.end()) {
            return true;
        }
    }
    return false;
}

tstring getAVVersions() {
    if (!check_avcodec_dll()) {
        return error_mes_avcodec_dll_not_found();
    }
    const uint32_t ver = avutil_version();
    auto ver2str = [](uint32_t ver) {
        return strsprintf("%3d.%3d.%4d", (ver >> 16) & 0xff, (ver >> 8) & 0xff, ver & 0xff);
    };
    std::string mes;
    mes  = std::string("ffmpeg     version: ") + std::string(av_version_info()) + "\n";
    mes += std::string("avutil     version: ") + ver2str(avutil_version()) + "\n";
    mes += std::string("avcodec    version: ") + ver2str(avcodec_version()) + "\n";
    mes += std::string("avformat   version: ") + ver2str(avformat_version()) + "\n";
    mes += std::string("avfilter   version: ") + ver2str(avfilter_version()) + "\n";
    mes += std::string("swresample version: ") + ver2str(swresample_version()) + "\n";
    return char_to_tstring(mes);
}

static const auto CSP_PIXFMT_RGY = make_array<std::pair<AVPixelFormat, RGY_CSP>>(
    std::make_pair(AV_PIX_FMT_YUV420P,     RGY_CSP_YV12),
    std::make_pair(AV_PIX_FMT_YUVJ420P,    RGY_CSP_YV12),
    std::make_pair(AV_PIX_FMT_NV12,        RGY_CSP_NV12),
    std::make_pair(AV_PIX_FMT_NV21,        RGY_CSP_NV12),
    std::make_pair(AV_PIX_FMT_YUV422P,     RGY_CSP_YUV422),
    std::make_pair(AV_PIX_FMT_YUVJ422P,    RGY_CSP_YUV422),
    std::make_pair(AV_PIX_FMT_YUYV422,     RGY_CSP_YUY2),
    std::make_pair(AV_PIX_FMT_UYVY422,     RGY_CSP_NA),
    std::make_pair(AV_PIX_FMT_NV16,        RGY_CSP_NV16),
    std::make_pair(AV_PIX_FMT_YUV444P,     RGY_CSP_YUV444),
    std::make_pair(AV_PIX_FMT_YUVJ444P,    RGY_CSP_YUV444),
    std::make_pair(AV_PIX_FMT_YUV420P16LE, RGY_CSP_YV12_16),
    std::make_pair(AV_PIX_FMT_YUV420P14LE, RGY_CSP_YV12_14),
    std::make_pair(AV_PIX_FMT_YUV420P12LE, RGY_CSP_YV12_12),
    std::make_pair(AV_PIX_FMT_YUV420P10LE, RGY_CSP_YV12_10),
    std::make_pair(AV_PIX_FMT_YUV420P9LE,  RGY_CSP_YV12_09),
    std::make_pair(AV_PIX_FMT_NV20LE,      RGY_CSP_NA),
    std::make_pair(AV_PIX_FMT_YUV422P16LE, RGY_CSP_YUV422_16),
    std::make_pair(AV_PIX_FMT_YUV422P14LE, RGY_CSP_YUV422_14),
    std::make_pair(AV_PIX_FMT_YUV422P12LE, RGY_CSP_YUV422_12),
    std::make_pair(AV_PIX_FMT_YUV422P10LE, RGY_CSP_YUV422_10),
    std::make_pair(AV_PIX_FMT_YUV444P16LE, RGY_CSP_YUV444_16),
    std::make_pair(AV_PIX_FMT_YUV444P14LE, RGY_CSP_YUV444_14),
    std::make_pair(AV_PIX_FMT_YUV444P12LE, RGY_CSP_YUV444_12),
    std::make_pair(AV_PIX_FMT_YUV444P10LE, RGY_CSP_YUV444_10),
    std::make_pair(AV_PIX_FMT_YUV444P9LE,  RGY_CSP_YUV444_09),
    std::make_pair(AV_PIX_FMT_RGB24,       RGY_CSP_BGR24),
    std::make_pair(AV_PIX_FMT_RGBA,        RGY_CSP_BGR32),
    std::make_pair(AV_PIX_FMT_BGR24,       RGY_CSP_RGB24),
    std::make_pair(AV_PIX_FMT_BGRA,        RGY_CSP_RGB32),
    std::make_pair(AV_PIX_FMT_GBRP,        RGY_CSP_GBR),
    std::make_pair(AV_PIX_FMT_GBRAP,       RGY_CSP_GBRA)
);

MAP_PAIR_0_1(csp, avpixfmt, AVPixelFormat, rgy, RGY_CSP, CSP_PIXFMT_RGY, AV_PIX_FMT_NONE, RGY_CSP_NA);

#endif //ENABLE_AVSW_READER
