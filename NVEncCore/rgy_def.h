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
// --------------------------------------------------------------------------------------------

#pragma once
#ifndef __RGY_DEF_H__
#define __RGY_DEF_H__

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <limits>
#include "rgy_tchar.h"
#include "rgy_version.h"
#include "convert_csp.h"

static const int MAX_FILENAME_LEN = 1024;

static const int RGY_OUTPUT_THREAD_AUTO = -1;
static const int RGY_AUDIO_THREAD_AUTO = -1;
static const int RGY_INPUT_THREAD_AUTO = -1;

static const int CHECK_PTS_MAX_INSERT_FRAMES = 18000;

static const int TRIM_MAX = std::numeric_limits<int>::max();
static const int TRIM_OVERREAD_FRAMES = 128;

static const int MAX_SPLIT_CHANNELS = 32;
static const char *RGY_CHANNEL_AUTO = "RGY_CHANNEL_AUTO";
static const int RGY_OUTPUT_BUF_MB_DEFAULT = 8;
static const int RGY_OUTPUT_BUF_MB_MAX = 128;

static const TCHAR *RGY_AVCODEC_AUTO = _T("auto");
static const TCHAR *RGY_AVCODEC_COPY = _T("copy");

typedef struct {
    int start, fin;
} sTrim;

typedef struct {
    std::vector<sTrim> list;
    int offset;
} sTrimParam;

enum {
    RGY_MUX_NONE     = 0x00,
    RGY_MUX_VIDEO    = 0x01,
    RGY_MUX_AUDIO    = 0x02,
    RGY_MUX_SUBTITLE = 0x04,
};

enum RGY_FRAMETYPE : uint32_t {
    RGY_FRAMETYPE_UNKNOWN = 0,

    RGY_FRAMETYPE_I       = 1<<0,
    RGY_FRAMETYPE_P       = 1<<1,
    RGY_FRAMETYPE_B       = 1<<2,

    RGY_FRAMETYPE_REF     = 1<<6,
    RGY_FRAMETYPE_IDR     = 1<<7,

    RGY_FRAMETYPE_xI      = 1<<8,
    RGY_FRAMETYPE_xP      = 1<<9,
    RGY_FRAMETYPE_xB      = 1<<10,

    RGY_FRAMETYPE_xREF    = 1<<14,
    RGY_FRAMETYPE_xIDR    = 1<<15
};

static RGY_FRAMETYPE operator|(RGY_FRAMETYPE a, RGY_FRAMETYPE b) {
    return (RGY_FRAMETYPE)((uint32_t)a | (uint32_t)b);
}

static RGY_FRAMETYPE operator|=(RGY_FRAMETYPE &a, RGY_FRAMETYPE b) {
    a = a | b;
    return a;
}

static RGY_FRAMETYPE operator&(RGY_FRAMETYPE a, RGY_FRAMETYPE b) {
    return (RGY_FRAMETYPE)((uint32_t)a & (uint32_t)b);
}

static RGY_FRAMETYPE operator&=(RGY_FRAMETYPE &a, RGY_FRAMETYPE b) {
    a = (RGY_FRAMETYPE)((uint32_t)a & (uint32_t)b);
    return a;
}

enum RGY_CODEC {
    RGY_CODEC_UNKNOWN = 0,
    RGY_CODEC_H264,
    RGY_CODEC_HEVC,
    RGY_CODEC_MPEG1,
    RGY_CODEC_MPEG2,
    RGY_CODEC_MPEG4,
    RGY_CODEC_VP8,
    RGY_CODEC_VP9,
    RGY_CODEC_VC1,
    RGY_CODEC_AV1,
    RGY_CODEC_VVC,
    RGY_CODEC_RAW,

    RGY_CODEC_NUM,
};

static tstring CodecToStr(RGY_CODEC codec) {
    switch (codec) {
    case RGY_CODEC_H264:  return _T("H.264/AVC");
    case RGY_CODEC_HEVC:  return _T("H.265/HEVC");
    case RGY_CODEC_MPEG2: return _T("MPEG2");
    case RGY_CODEC_MPEG1: return _T("MPEG1");
    case RGY_CODEC_VC1:   return _T("VC-1");
    case RGY_CODEC_MPEG4: return _T("MPEG4");
    case RGY_CODEC_VP8:   return _T("VP8");
    case RGY_CODEC_VP9:   return _T("VP9");
    case RGY_CODEC_AV1:   return _T("AV1");
    case RGY_CODEC_VVC:   return _T("VVC");
    case RGY_CODEC_RAW:   return _T("RAW");
    default: return _T("unknown");
    }
}

struct RGY_CODEC_DATA {
    RGY_CODEC codec;
    int codecProfile;

    RGY_CODEC_DATA() : codec(RGY_CODEC_UNKNOWN), codecProfile(0) {}
    RGY_CODEC_DATA(RGY_CODEC _codec, int profile) : codec(_codec), codecProfile(profile) {}

    bool operator<(const RGY_CODEC_DATA &right) const {
        return codec == right.codec ? codec < right.codec : codecProfile < right.codecProfile;
    }
    bool operator==(const RGY_CODEC_DATA &right) const {
        return codec == right.codec && codecProfile == right.codecProfile;
    }
};

enum RGY_INPUT_FMT {
    RGY_INPUT_FMT_AUTO = 0,
    RGY_INPUT_FMT_AUO = 0,
    RGY_INPUT_FMT_RAW,
    RGY_INPUT_FMT_Y4M,
    RGY_INPUT_FMT_AVI,
    RGY_INPUT_FMT_AVS,
    RGY_INPUT_FMT_VPY,
    RGY_INPUT_FMT_VPY_MT,
    RGY_INPUT_FMT_AVHW,
    RGY_INPUT_FMT_AVSW,
    RGY_INPUT_FMT_AVANY,
    RGY_INPUT_FMT_SM,
};

typedef struct CX_DESC {
    const TCHAR *desc;
    int value;
} CX_DESC;

typedef struct FEATURE_DESC {
    const TCHAR *desc;
    uint64_t value;
} FEATURE_DESC;

static int get_cx_desc_cmp(const wchar_t *str1, const wchar_t *str2) {
#if defined(_WIN32) || defined(_WIN64)
    return _wcsicmp(str1, str2);
#else
    return wcscasecmp(str1, str2);
#endif
}

static int get_cx_desc_cmp(const char *str1, const char *str2) {
#if defined(_WIN32) || defined(_WIN64)
    return _stricmp(str1, str2);
#else
    return strcasecmp(str1, str2);
#endif
}

template<typename T>
static const decltype(T::desc) get_chr_from_value(const T *list, decltype(T::value) v) {
    for (int i = 0; list[i].desc; i++)
        if (list[i].value == v)
            return list[i].desc;
    return _T("unknown");
}

template<typename T>
static int get_cx_index(const T *list, decltype(T::value) v) {
    for (int i = 0; list[i].desc; i++)
        if (list[i].value == v)
            return i;
    return 0;
}

template<typename T>
static int get_cx_index(const T *list, const decltype(T::desc) chr) {
    for (int i = 0; list[i].desc; i++)
        if (get_cx_desc_cmp(list[i].desc, chr) == 0)
            return i;
    return 0;
}

template<typename T>
static decltype(T::value) get_cx_value(const T *list, const decltype(T::desc) chr) {
    for (int i = 0; list[i].desc; i++)
        if (get_cx_desc_cmp(list[i].desc, chr) == 0)
            return list[i].value;
    return 0;
}

static int PARSE_ERROR_FLAG = std::numeric_limits<int>::min();
template<typename T>
static decltype(T::value) get_value_from_chr(const T *list, const decltype(T::desc) chr) {
    for (int i = 0; list[i].desc; i++)
        if (get_cx_desc_cmp(list[i].desc, chr) == 0)
            return list[i].value;
    return PARSE_ERROR_FLAG;
}

template<typename T>
static const decltype(T::desc) get_cx_desc(const T *list, decltype(T::value) v) {
    for (int i = 0; list[i].desc; i++)
        if (list[i].value == v)
            return list[i].desc;
    return nullptr;
}

template<typename T>
static bool get_list_value(const T *list, const decltype(T::desc) chr, decltype(T::value) *value) {
    for (int i = 0; list[i].desc; i++) {
        if (get_cx_desc_cmp(list[i].desc, chr) == 0) {
            *value = list[i].value;
            return true;
        }
    }
    return false;
};

static const CX_DESC list_rgy_codec[] = {
    { _T("h264"),  RGY_CODEC_H264 },
    { _T("avc"),   RGY_CODEC_H264 },
    { _T("h265"),  RGY_CODEC_HEVC },
    { _T("hevc"),  RGY_CODEC_HEVC },
    { _T("mpeg2"), RGY_CODEC_MPEG2 },
    { _T("mpeg1"), RGY_CODEC_MPEG1 },
    { _T("mpeg4"), RGY_CODEC_MPEG4 },
    { _T("vc1"),   RGY_CODEC_VC1 },
    { _T("vp8"),   RGY_CODEC_VP8 },
    { _T("vp9"),   RGY_CODEC_VP9 },
    { _T("av1"),   RGY_CODEC_AV1 },
    { _T("vvc"),   RGY_CODEC_VVC },
    { NULL, 0 }
};

enum {
    RGY_RESAMPLER_SWR,
    RGY_RESAMPLER_SOXR,
};

enum {
    DELOGO_MODE_REMOVE = 0,
    DELOGO_MODE_ADD,
    DELOGO_MODE_ADD_MULTI,
};

enum class RGYResizeResMode {
    Normal,
    PreserveOrgAspectDec,
    PreserveOrgAspectInc,
};

static const int COLOR_VALUE_AUTO = -1;
static const int COLOR_VALUE_AUTO_RESOLUTION = std::numeric_limits<int>::max();
static const TCHAR *COLOR_VALUE_AUTO_HD_NAME = _T("bt709");
static const TCHAR *COLOR_VALUE_AUTO_SD_NAME = _T("smpte170m");
static const int HD_HEIGHT_THRESHOLD = 720;

enum CspMatrix {
    RGY_MATRIX_AUTO        = COLOR_VALUE_AUTO,
    RGY_MATRIX_RGB         = 0,
    RGY_MATRIX_BT709       = 1,
    RGY_MATRIX_UNSPECIFIED = 2,
    RGY_MATRIX_FCC         = 4,
    RGY_MATRIX_BT470_BG    = 5,
    RGY_MATRIX_ST170_M     = 6,
    RGY_MATRIX_ST240_M     = 7,
    RGY_MATRIX_YCGCO       = 8,
    RGY_MATRIX_BT2020_NCL  = 9,
    RGY_MATRIX_BT2020_CL   = 10,
    RGY_MATRIX_DERIVED_NCL = 12,
    RGY_MATRIX_DERIVED_CL  = 13,
    RGY_MATRIX_ICTCP       = 14,
    RGY_MATRIX_2100_LMS,
};

static const std::array<CspMatrix, 14> CspMatrixList{
    RGY_MATRIX_RGB,
    RGY_MATRIX_BT709,
    RGY_MATRIX_UNSPECIFIED,
    RGY_MATRIX_FCC,
    RGY_MATRIX_BT470_BG,
    RGY_MATRIX_ST170_M,
    RGY_MATRIX_ST240_M,
    RGY_MATRIX_YCGCO,
    RGY_MATRIX_BT2020_NCL,
    RGY_MATRIX_BT2020_CL,
    RGY_MATRIX_DERIVED_NCL,
    RGY_MATRIX_DERIVED_CL,
    RGY_MATRIX_ICTCP,
    RGY_MATRIX_2100_LMS
};

const CX_DESC list_colormatrix[] = {
    { _T("undef"),       RGY_MATRIX_UNSPECIFIED  },
    { _T("auto"),        RGY_MATRIX_AUTO  },
    { _T("auto_res"),    COLOR_VALUE_AUTO_RESOLUTION },
    { _T("bt709"),       RGY_MATRIX_BT709  },
    { _T("smpte170m"),   RGY_MATRIX_ST170_M  },
    { _T("bt470bg"),     RGY_MATRIX_BT470_BG  },
    { _T("smpte240m"),   RGY_MATRIX_ST240_M  },
    { _T("YCgCo"),       RGY_MATRIX_YCGCO  },
    { _T("fcc"),         RGY_MATRIX_FCC  },
    { _T("GBR"),         RGY_MATRIX_RGB  },
    { _T("bt2020nc"),    RGY_MATRIX_BT2020_NCL  },
    { _T("bt2020c"),     RGY_MATRIX_BT2020_CL },
    { _T("derived-ncl"), RGY_MATRIX_DERIVED_NCL },
    { _T("derived-cl"),  RGY_MATRIX_DERIVED_CL },
    { _T("ictco"),       RGY_MATRIX_ICTCP },
    { _T("2100-lms"),    RGY_MATRIX_2100_LMS },
    { NULL, 0 }
};

enum CspTransfer {
    RGY_TRANSFER_AUTO         = COLOR_VALUE_AUTO,
    RGY_TRANSFER_UNKNOWN      = 0,
    RGY_TRANSFER_BT709        = 1,
    RGY_TRANSFER_UNSPECIFIED  = 2,
    RGY_TRANSFER_BT470_M      = 4,
    RGY_TRANSFER_BT470_BG     = 5,
    RGY_TRANSFER_BT601        = 6,  //BT709
    RGY_TRANSFER_ST240_M      = 7,
    RGY_TRANSFER_LINEAR       = 8,
    RGY_TRANSFER_LOG_100      = 9,
    RGY_TRANSFER_LOG_316      = 10,
    RGY_TRANSFER_IEC61966_2_4 = 11, //XVYCC
    RGY_TRANSFER_IEC61966_2_1 = 13, //SRGB
    RGY_TRANSFER_BT2020_10    = 14, //BT709
    RGY_TRANSFER_BT2020_12    = 15, //BT709
    RGY_TRANSFER_ST2084       = 16,
    RGY_TRANSFER_ARIB_B67     = 18
};

static const std::array<CspTransfer, 15> CspTransferList{
    RGY_TRANSFER_BT709,
    RGY_TRANSFER_UNSPECIFIED,
    RGY_TRANSFER_BT470_M,
    RGY_TRANSFER_BT470_BG,
    RGY_TRANSFER_BT601,  //BT709
    RGY_TRANSFER_ST240_M,
    RGY_TRANSFER_LINEAR,
    RGY_TRANSFER_LOG_100,
    RGY_TRANSFER_LOG_316,
    RGY_TRANSFER_IEC61966_2_4, //XVYCC
    RGY_TRANSFER_IEC61966_2_1, //SRGB
    RGY_TRANSFER_BT2020_10, //BT709
    RGY_TRANSFER_BT2020_12, //BT709
    RGY_TRANSFER_ST2084,
    RGY_TRANSFER_ARIB_B67
};

const CX_DESC list_transfer[] = {
    { _T("undef"),         RGY_TRANSFER_UNSPECIFIED  },
    { _T("unknown"),       RGY_TRANSFER_UNKNOWN },
    { _T("auto"),          RGY_TRANSFER_AUTO },
    { _T("auto_res"),      COLOR_VALUE_AUTO_RESOLUTION },
    { _T("bt709"),         RGY_TRANSFER_BT709  },
    { _T("smpte170m"),     RGY_TRANSFER_BT601  },
    { _T("bt470m"),        RGY_TRANSFER_BT470_M  },
    { _T("bt470bg"),       RGY_TRANSFER_BT470_BG  },
    { _T("smpte240m"),     RGY_TRANSFER_ST240_M  },
    { _T("linear"),        RGY_TRANSFER_LINEAR  },
    { _T("log100"),        RGY_TRANSFER_LOG_100  },
    { _T("log316"),        RGY_TRANSFER_LOG_316 },
    { _T("iec61966-2-4"),  RGY_TRANSFER_IEC61966_2_4 },
    { _T("bt1361e"),       12 },
    { _T("iec61966-2-1"),  RGY_TRANSFER_IEC61966_2_1 },
    { _T("bt2020-10"),     RGY_TRANSFER_BT2020_10 },
    { _T("bt2020-12"),     RGY_TRANSFER_BT2020_12 },
    { _T("smpte2084"),     RGY_TRANSFER_ST2084 },
    { _T("smpte428"),      17 },
    { _T("arib-std-b67"),  RGY_TRANSFER_ARIB_B67 },
    { NULL, 0 }
};

enum CspColorprim {
    RGY_PRIM_AUTO        = COLOR_VALUE_AUTO,
    RGY_PRIM_UNKNOWN     = 0,
    RGY_PRIM_BT709       = 1,
    RGY_PRIM_UNSPECIFIED = 2,
    RGY_PRIM_BT470_M     = 4,
    RGY_PRIM_BT470_BG    = 5,
    RGY_PRIM_ST170_M     = 6,
    RGY_PRIM_ST240_M     = 7,
    RGY_PRIM_FILM        = 8,
    RGY_PRIM_BT2020      = 9,
    RGY_PRIM_ST428, //XYZ
    RGY_PRIM_ST431_2, //DCI_P3
    RGY_PRIM_ST432_1, //DCI_P3_D65
    RGY_PRIM_EBU3213_E //JEDEC_P22
};

static const std::array<CspColorprim, 12> CspColorprimList{
    RGY_PRIM_BT709,
    RGY_PRIM_UNSPECIFIED,
    RGY_PRIM_BT470_M,
    RGY_PRIM_BT470_BG,
    RGY_PRIM_ST170_M,
    RGY_PRIM_ST240_M,
    RGY_PRIM_FILM,
    RGY_PRIM_BT2020,
    RGY_PRIM_ST428, //XYZ
    RGY_PRIM_ST431_2, //DCI_P3
    RGY_PRIM_ST432_1, //DCI_P3_D65
    RGY_PRIM_EBU3213_E //JEDEC_P22
};

const CX_DESC list_colorprim[] = {
    { _T("undef"),     RGY_PRIM_UNSPECIFIED  },
    { _T("unknown"),   RGY_PRIM_UNKNOWN      },
    { _T("auto"),      RGY_PRIM_AUTO      },
    { _T("auto_res"),  COLOR_VALUE_AUTO_RESOLUTION   },
    { _T("bt709"),     RGY_PRIM_BT709     },
    { _T("smpte170m"), RGY_PRIM_ST170_M   },
    { _T("bt470m"),    RGY_PRIM_BT470_M   },
    { _T("bt470bg"),   RGY_PRIM_BT470_BG  },
    { _T("smpte240m"), RGY_PRIM_ST240_M   },
    { _T("film"),      RGY_PRIM_FILM      },
    { _T("bt2020"),    RGY_PRIM_BT2020    },
    { _T("st428"),     RGY_PRIM_ST428     },
    { _T("st431-2"),   RGY_PRIM_ST431_2   },
    { _T("st432-1"),   RGY_PRIM_ST432_1   },
    { _T("ebu3213-e"), RGY_PRIM_EBU3213_E },
    { NULL, 0 }
};

const CX_DESC list_videoformat[] = {
    { _T("undef"),     5  },
    { _T("ntsc"),      2  },
    { _T("component"), 0  },
    { _T("pal"),       1  },
    { _T("secam"),     3  },
    { _T("mac"),       4  },
    { NULL, 0 }
};


enum RGYDOVIProfile {
    RGY_DOVI_PROFILE_UNSET = 0,
    RGY_DOVI_PROFILE_COPY  = -1,
    RGY_DOVI_PROFILE_50    = 50,
    RGY_DOVI_PROFILE_70    = 70,
    RGY_DOVI_PROFILE_81    = 81,
    RGY_DOVI_PROFILE_82    = 82,
    RGY_DOVI_PROFILE_84    = 84,
    RGY_DOVI_PROFILE_OTHER = 100,
};

const CX_DESC list_dovi_profile[] = {
    { _T("unset"), RGY_DOVI_PROFILE_UNSET },
    { _T("copy"),  RGY_DOVI_PROFILE_COPY },
    { _T("5.0"),   RGY_DOVI_PROFILE_50 },
    { _T("8.1"),   RGY_DOVI_PROFILE_81 },
    { _T("8.2"),   RGY_DOVI_PROFILE_82 },
    { _T("8.4"),   RGY_DOVI_PROFILE_84 },
    { NULL, 0 }
};

const CX_DESC list_dovi_profile_parse[] = {
    { _T("unset"), RGY_DOVI_PROFILE_UNSET },
    { _T("copy"),  RGY_DOVI_PROFILE_COPY },
    { _T("5.0"),   RGY_DOVI_PROFILE_50 },
    { _T("5"),     RGY_DOVI_PROFILE_50 },
    { _T("8.1"),   RGY_DOVI_PROFILE_81 },
    { _T("8.2"),   RGY_DOVI_PROFILE_82 },
    { _T("8.4"),   RGY_DOVI_PROFILE_84 },
    { NULL, 0 }
};

struct RGYDOVIRpuActiveAreaOffsets {
    bool enable;
    uint16_t left, top, right, bottom;

    RGYDOVIRpuActiveAreaOffsets() : enable(false), left(0), top(0), right(0), bottom(0) {};
    bool operator==(const RGYDOVIRpuActiveAreaOffsets &x) const {
        return enable == x.enable
            && left == x.left
            && top == x.top
            && right == x.right
            && bottom == x.bottom;
    }
    bool operator!=(const RGYDOVIRpuActiveAreaOffsets &x) const {
        return !(*this == x);
    }
};

class RGYDOVIRpuConvertParam {
public:
    bool convertProfile;
    bool removeMapping;
    RGYDOVIRpuActiveAreaOffsets activeAreaOffsets;
    RGYDOVIRpuConvertParam() : convertProfile(true), removeMapping(false), activeAreaOffsets() {};
    virtual ~RGYDOVIRpuConvertParam() {};
    bool operator==(const RGYDOVIRpuConvertParam &x) const {
        return convertProfile == x.convertProfile
            && removeMapping == x.removeMapping
            && activeAreaOffsets == x.activeAreaOffsets;
    }
    bool operator!=(const RGYDOVIRpuConvertParam &x) const {
        return !(*this == x);
    }
};

// 1st luma line > |X   X ...    |3 4 X ...     X が輝度ピクセル位置
//                 |             |1 2           1-6 are possible chroma positions
// 2nd luma line > |X   X ...    |5 6 X ...     bitstreamに入れるときは-1すること
enum CspChromaloc {
    RGY_CHROMALOC_AUTO = COLOR_VALUE_AUTO,
    RGY_CHROMALOC_UNSPECIFIED = 0,
    RGY_CHROMALOC_LEFT = 1,
    RGY_CHROMALOC_CENTER = 2,
    RGY_CHROMALOC_TOPLEFT = 3,
    RGY_CHROMALOC_TOP = 4,
    RGY_CHROMALOC_BOTTOMLEFT = 5,
    RGY_CHROMALOC_BOTTOM = 6,
};

const CX_DESC list_chromaloc[] = {
    { _T("undef"), RGY_CHROMALOC_UNSPECIFIED },
    { _T("0"),     RGY_CHROMALOC_LEFT },
    { _T("1"),     RGY_CHROMALOC_CENTER },
    { _T("2"),     RGY_CHROMALOC_TOPLEFT },
    { _T("3"),     RGY_CHROMALOC_TOP },
    { _T("4"),     RGY_CHROMALOC_BOTTOMLEFT },
    { _T("5"),     RGY_CHROMALOC_BOTTOM },
    { _T("auto"),  RGY_CHROMALOC_AUTO },
    { NULL, 0 }
};

const CX_DESC list_chromaloc_str[] = {
    { _T("undef"),      RGY_CHROMALOC_UNSPECIFIED },
    { _T("left"),       RGY_CHROMALOC_LEFT },
    { _T("center"),     RGY_CHROMALOC_CENTER },
    { _T("topleft"),    RGY_CHROMALOC_TOPLEFT },
    { _T("top"),        RGY_CHROMALOC_TOP },
    { _T("bottomleft"), RGY_CHROMALOC_BOTTOMLEFT },
    { _T("bottom"),     RGY_CHROMALOC_BOTTOM },
    { _T("auto"),       RGY_CHROMALOC_AUTO },
    { NULL, 0 }
};

enum CspColorRange {
    RGY_COLORRANGE_AUTO = COLOR_VALUE_AUTO,
    RGY_COLORRANGE_UNSPECIFIED = 0,
    RGY_COLORRANGE_LIMITED = 1,
    RGY_COLORRANGE_FULL = 2,
};

const CX_DESC list_colorrange[] = {
    { _T("undef"),   RGY_COLORRANGE_UNSPECIFIED },
    { _T("limited"), RGY_COLORRANGE_LIMITED },
    { _T("tv"),      RGY_COLORRANGE_LIMITED },
    { _T("full"),    RGY_COLORRANGE_FULL },
    { _T("pc"),      RGY_COLORRANGE_FULL },
    { _T("auto"),    RGY_COLORRANGE_AUTO },
    { NULL, 0 }
};

template<typename T>
void apply_auto_color_characteristic(T &value, const CX_DESC *list, int frame_height, T auto_val) {
    if (value == COLOR_VALUE_AUTO_RESOLUTION) {
        value = (T)get_cx_value(list, (frame_height >= HD_HEIGHT_THRESHOLD) ? COLOR_VALUE_AUTO_HD_NAME : COLOR_VALUE_AUTO_SD_NAME);
    } else if (value == COLOR_VALUE_AUTO) {
        value = auto_val;
    }
};

struct VideoVUIInfo {
    int descriptpresent; //colorprim, matrix, transfer is present
    CspColorprim colorprim;
    CspMatrix matrix;
    CspTransfer transfer;
    int format;
    CspColorRange colorrange;
    CspChromaloc chromaloc;

    VideoVUIInfo() :
        descriptpresent(0),
        colorprim((CspColorprim)get_cx_value(list_colorprim, _T("undef"))),
        matrix((CspMatrix)get_cx_value(list_colormatrix, _T("undef"))),
        transfer((CspTransfer)get_cx_value(list_transfer, _T("undef"))),
        format(get_cx_value(list_videoformat, _T("undef"))),
        colorrange((CspColorRange)get_cx_value(list_colorrange, _T("undef"))),
        chromaloc((CspChromaloc)get_cx_value(list_chromaloc_str, _T("undef"))) {

    }

    VideoVUIInfo(int descriptpresent_,
        CspColorprim colorprim_, 
        CspMatrix matrix_,
        CspTransfer transfer_,
        int format_,
        CspColorRange colorrange_,
        CspChromaloc chromaloc_) :
        descriptpresent(descriptpresent_),
        colorprim(colorprim_),
        matrix(matrix_),
        transfer(transfer_),
        format(format_),
        colorrange(colorrange_),
        chromaloc(chromaloc_) {}

    VideoVUIInfo to(CspMatrix csp_matrix) const {
        auto ret = *this;
        ret.matrix = csp_matrix;
        return ret;
    }
    VideoVUIInfo to(CspTransfer csp_transfer) const {
        auto ret = *this;
        ret.transfer = csp_transfer;
        return ret;
    }
    VideoVUIInfo to(CspColorprim prim) const {
        auto ret = *this;
        ret.colorprim = prim;
        return ret;
    }
    tstring print_main() const;
    tstring print_all(bool write_all = false) const;
    void apply_auto(const VideoVUIInfo &input, const int inputHeight) {
        apply_auto_color_characteristic(colorprim,  list_colorprim,   inputHeight, input.colorprim);
        apply_auto_color_characteristic(transfer,   list_transfer,    inputHeight, input.transfer);
        apply_auto_color_characteristic(matrix,     list_colormatrix, inputHeight, input.matrix);
        apply_auto_color_characteristic(colorrange, list_colorrange,  inputHeight, input.colorrange);
        apply_auto_color_characteristic(chromaloc,  list_chromaloc,   inputHeight, input.chromaloc);
    }
    void setDescriptPreset() {
        descriptpresent =
           get_cx_value(list_colormatrix, _T("undef")) != (int)matrix
        || get_cx_value(list_colorprim,   _T("undef")) != (int)colorprim
        || get_cx_value(list_transfer,    _T("undef")) != (int)transfer;
    }

    void setIfUnset(const VideoVUIInfo &x) {
        const auto defaultVUI = VideoVUIInfo();
        if (colorprim == defaultVUI.colorprim) {
            colorprim = x.colorprim;
        }
        if (matrix == defaultVUI.matrix) {
            matrix = x.matrix;
        }
        if (transfer == defaultVUI.transfer) {
            transfer = x.transfer;
        }
        if (format == defaultVUI.format) {
            format = x.format;
        }
        if (colorrange == defaultVUI.colorrange) {
            colorrange = x.colorrange;
        }
        if (chromaloc == defaultVUI.chromaloc) {
            chromaloc = x.chromaloc;
        }
        setDescriptPreset();
    }

    void setIfUnsetUnknwonAuto(const VideoVUIInfo &x) {
        const auto defaultVUI = VideoVUIInfo();
        if (   colorprim == RGY_PRIM_UNSPECIFIED
            || colorprim == RGY_PRIM_UNKNOWN
            || colorprim == RGY_PRIM_AUTO) {
            colorprim = x.colorprim;
        }
        if (   matrix == RGY_MATRIX_UNSPECIFIED
            || matrix == RGY_MATRIX_AUTO) {
            matrix = x.matrix;
        }
        if (   transfer == RGY_TRANSFER_UNKNOWN
            || transfer == RGY_TRANSFER_UNSPECIFIED
            || transfer == RGY_TRANSFER_AUTO) {
            transfer = x.transfer;
        }
        if (format == defaultVUI.format) {
            format = x.format;
        }
        if (colorrange == defaultVUI.colorrange) {
            colorrange = x.colorrange;
        }
        if (chromaloc == defaultVUI.chromaloc) {
            chromaloc = x.chromaloc;
        }
        setDescriptPreset();
    }

    bool operator==(const VideoVUIInfo &x) const {
        return descriptpresent == x.descriptpresent
            && colorprim == x.colorprim
            && matrix == x.matrix
            && transfer == x.transfer
            && format == x.format
            && colorrange == x.colorrange
            && chromaloc == x.chromaloc;
    }
    bool operator!=(const VideoVUIInfo &x) const {
        return !(*this == x);
    }
};

struct VideoInfo {
    //[ i    ] 入力モジュールに渡す際にセットする
    //[    i ] 入力モジュールによってセットされる
    //[ o    ] 出力モジュールに渡す際にセットする

    //[ i (i)] 種類 (RGY_INPUT_FMT_xxx)
    //  i      使用する入力モジュールの種類
    //     i   変更があれば
    RGY_INPUT_FMT type;

    //[(i) i ] 入力横解像度
    int srcWidth;

    //[(i) i ] 入力縦解像度
    int srcHeight;

    //[(i)(i)] 入力ピッチ 0なら入力横解像度に同じ
    uint32_t srcPitch;

                             //[      ] 出力解像度
    int dstWidth;

    //[      ] 出力解像度
    int dstHeight;

    //[      ] 出力解像度
    uint32_t dstPitch;

    //[    i ] 入力の取得した総フレーム数 (不明なら0)
    int frames;

    //[   (i)] ビット深度
    int bitdepth;

    //[   (i)] 入力の取得したフレームレート (分子)
    int fpsN;

    //[   (i)] 入力の取得したフレームレート (分母)
    int fpsD;

    //[ i    ] 入力時切り落とし
    sInputCrop crop;

    //[   (i)] 入力の取得したアスペクト比
    int sar[2];

    //[(i) i ] 入力色空間 (RGY_CSP_xxx)
    //  i      取得したい色空間をセット
    //     i   入力の取得する色空間
    RGY_CSP csp;

    //[(i)(i)] RGY_PICSTRUCT_xxx
    //  i      ユーザー指定の設定をセット
    //     i   入力の取得した値、あるいはそのまま
    RGY_PICSTRUCT picstruct;

    //[    i ] 入力コーデック (デコード時使用)
    //     i   HWデコード時セット
    RGY_CODEC codec;

    //[      ] 入力コーデックのヘッダー
    void *codecExtra;

    //[      ] 入力コーデックのヘッダーの大きさ
    uint32_t codecExtraSize;

    //[      ] 入力コーデックのレベル
    int codecLevel;

    //[      ] 入力コーデックのプロファイル
    int codecProfile;

    //[      ] 入力コーデックの遅延
    int videoDelay;

    //[      ] 入力コーデックのVUI情報
    VideoVUIInfo vui;

    VideoInfo() :
        type(RGY_INPUT_FMT_AUTO),
        srcWidth(0),
        srcHeight(0),
        srcPitch(0),
        dstWidth(0),
        dstHeight(0),
        dstPitch(0),
        frames(0),
        bitdepth(0),
        fpsN(0),
        fpsD(0),
        crop(),
        sar(),
        csp(RGY_CSP_NA),
        picstruct(RGY_PICSTRUCT_UNKNOWN),
        codec(RGY_CODEC_UNKNOWN),
        codecExtra(nullptr),
        codecExtraSize(0),
        codecLevel(0),
        codecProfile(0),
        videoDelay(0),
        vui() {};
    ~VideoInfo(){};
};

enum RGYAVSync : uint32_t {
    RGY_AVSYNC_AUTO       = 0x00,
    RGY_AVSYNC_FORCE_CFR  = 0x01,
    RGY_AVSYNC_VFR        = 0x02,
};

static RGYAVSync operator|(RGYAVSync a, RGYAVSync b) {
    return (RGYAVSync)((uint32_t)a | (uint32_t)b);
}

static RGYAVSync operator|=(RGYAVSync &a, RGYAVSync b) {
    a = a | b;
    return a;
}

static RGYAVSync operator&(RGYAVSync a, RGYAVSync b) {
    return (RGYAVSync)((uint32_t)a & (uint32_t)b);
}

static RGYAVSync operator&=(RGYAVSync &a, RGYAVSync b) {
    a = a & b;
    return a;
}

static RGYAVSync operator~(RGYAVSync a) {
    return (RGYAVSync)(~(uint32_t)a);
}

const CX_DESC list_empty[] = {
    { NULL, 0 }
};

static bool is_list_empty(const CX_DESC *list) {
    return list[0].desc == nullptr;
}

extern const CX_DESC list_log_level[];

const CX_DESC list_avsync[] = {
    { _T("auto"),     RGY_AVSYNC_AUTO  },
    { _T("cfr"),      RGY_AVSYNC_AUTO  },
    { _T("vfr"),      RGY_AVSYNC_VFR         },
    { _T("forcecfr"), RGY_AVSYNC_FORCE_CFR   },
    { NULL, 0 }
};

const CX_DESC list_resampler[] = {
    { _T("swr"),  RGY_RESAMPLER_SWR  },
    { _T("soxr"), RGY_RESAMPLER_SOXR },
    { NULL, 0 }
};

const CX_DESC list_interlaced[] = {
    { _T("progressive"), RGY_PICSTRUCT_FRAME     },
    { _T("tff"),         RGY_PICSTRUCT_FRAME_TFF },
    { _T("bff"),         RGY_PICSTRUCT_FRAME_BFF },
#if ENABLE_AUTO_PICSTRUCT
    { _T("auto"),        (int)RGY_PICSTRUCT_AUTO },
#endif //#if ENABLE_AUTO_PICSTRUCT
    { NULL, 0 }
};

const CX_DESC list_rgy_csp[] = {
    { _T("Invalid"),        RGY_CSP_NA },
    { _T("nv12"),           RGY_CSP_NV12 },
    { _T("yv12"),           RGY_CSP_YV12 },
    { _T("yuv420p"),        RGY_CSP_YV12 },
#if 0
    { _T("yuy2"),           RGY_CSP_YUY2 },
#endif
    { _T("yuv422p"),        RGY_CSP_YUV422 },
#if 0
    { _T("nv16"),           RGY_CSP_NV16 },
#endif
    { _T("yuv444p"),        RGY_CSP_YUV444 },
    { _T("yuv420p9le"),     RGY_CSP_YV12_09 },
    { _T("yuv420p10le"),    RGY_CSP_YV12_10 },
    { _T("yuv420p12le"),    RGY_CSP_YV12_12 },
    { _T("yuv420p14le"),    RGY_CSP_YV12_14 },
    { _T("yuv420p16le"),    RGY_CSP_YV12_16 },
    { _T("p010"),           RGY_CSP_P010 },
    { _T("yuv422p9le"),     RGY_CSP_YUV422_09 },
    { _T("yuv422p10le"),    RGY_CSP_YUV422_10 },
    { _T("yuv422p12le"),    RGY_CSP_YUV422_12 },
    { _T("yuv422p14le"),    RGY_CSP_YUV422_14 },
    { _T("yuv422p16le"),    RGY_CSP_YUV422_16 },
#if 0
    { _T("p210"),           RGY_CSP_P210 },
#endif
    { _T("yuv444p9le"),     RGY_CSP_YUV444_09 },
    { _T("yuv444p10le"),    RGY_CSP_YUV444_10 },
    { _T("yuv444p12le"),    RGY_CSP_YUV444_12 },
    { _T("yuv444p14le"),    RGY_CSP_YUV444_14 },
    { _T("yuv444p16le"),    RGY_CSP_YUV444_16 },
#if 0
    { _T("yuva444"),        RGY_CSP_YUVA444 },
    { _T("yuva444p16le"),   RGY_CSP_YUVA444_16 },
    { _T("rgb24r"),         RGY_CSP_RGB24R },
    { _T("rgb32r"),         RGY_CSP_RGB32R },
    { _T("rgb24"),          RGY_CSP_RGB24 },
    { _T("rgb32"),          RGY_CSP_RGB32 },
    { _T("bgr24"),          RGY_CSP_BGR24 },
    { _T("bgr32"),          RGY_CSP_BGR32 },
    { _T("rgb"),            RGY_CSP_RGB },
    { _T("rgba"),           RGY_CSP_RGBA },
    { _T("gbr"),            RGY_CSP_GBR },
    { _T("gbra"),           RGY_CSP_GBRA },
    { _T("yc48"),           RGY_CSP_YC48 },
    { _T("y8"),             RGY_CSP_Y8 },
    { _T("yc16"),           RGY_CSP_Y16 },
#endif
    { NULL, 0 }
};

typedef std::map<RGY_CODEC, std::vector<RGY_CSP>> CodecCsp;
typedef std::vector<std::pair<int, CodecCsp>> DeviceCodecCsp;
typedef std::vector<std::pair<tstring, tstring>> RGYOptList;

#endif //__RGY_DEF_H__
