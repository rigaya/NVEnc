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
#include "convert_csp.h"

typedef std::basic_string<TCHAR> tstring;

static const int MAX_FILENAME_LEN = 1024;

static const int RGY_OUTPUT_THREAD_AUTO = -1;
static const int RGY_AUDIO_THREAD_AUTO = -1;
static const int RGY_INPUT_THREAD_AUTO = -1;

static const int CHECK_PTS_MAX_INSERT_FRAMES = 8;

static const int TRIM_MAX = std::numeric_limits<int>::max();
static const int TRIM_OVERREAD_FRAMES = 128;

static const int MAX_SPLIT_CHANNELS = 32;
static const uint64_t RGY_CHANNEL_AUTO = std::numeric_limits<uint64_t>::max();
static const int RGY_OUTPUT_BUF_MB_MAX = 128;

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

static const TCHAR *get_chr_from_value(const CX_DESC *list, int v) {
    for (int i = 0; list[i].desc; i++)
        if (list[i].value == v)
            return list[i].desc;
    return _T("unknown");
}

static int get_cx_index(const CX_DESC *list, int v) {
    for (int i = 0; list[i].desc; i++)
        if (list[i].value == v)
            return i;
    return 0;
}

static int get_cx_index(const CX_DESC *list, const TCHAR *chr) {
    for (int i = 0; list[i].desc; i++)
        if (0 == _tcscmp(list[i].desc, chr))
            return i;
    return 0;
}

static int get_cx_value(const CX_DESC *list, const TCHAR *chr) {
    for (int i = 0; list[i].desc; i++)
        if (0 == _tcscmp(list[i].desc, chr))
            return list[i].value;
    return 0;
}

static int PARSE_ERROR_FLAG = std::numeric_limits<int>::min();
static int get_value_from_chr(const CX_DESC *list, const TCHAR *chr) {
    for (int i = 0; list[i].desc; i++)
        if (_tcsicmp(list[i].desc, chr) == 0)
            return list[i].value;
    return PARSE_ERROR_FLAG;
}

static const TCHAR *get_cx_desc(const CX_DESC *list, int v) {
    for (int i = 0; list[i].desc; i++)
        if (list[i].value == v)
            return list[i].desc;
    return nullptr;
}

static bool get_list_value(const CX_DESC *list, const TCHAR *chr, int *value) {
    for (int i = 0; list[i].desc; i++) {
        if (0 == _tcsicmp(list[i].desc, chr)) {
            *value = list[i].value;
            return true;
        }
    }
    return false;
};

enum {
    RGY_RESAMPLER_SWR,
    RGY_RESAMPLER_SOXR,
};

enum {
    DELOGO_MODE_REMOVE = 0,
    DELOGO_MODE_ADD,
};

const int COLOR_VALUE_AUTO = -1;
const int COLOR_VALUE_AUTO_RESOLUTION = std::numeric_limits<int>::max();
const int HD_HEIGHT_THRESHOLD = 720;
const int HD_INDEX = 3;
const int SD_INDEX = 4;

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
    { _T("auto"),     COLOR_VALUE_AUTO  },
    { _T("ntsc"),      2  },
    { _T("component"), 0  },
    { _T("pal"),       1  },
    { _T("secam"),     3  },
    { _T("mac"),       4  },
    { NULL, 0 }
};


// 1st luma line > |X   X ...    |3 4 X ...     X が輝度ピクセル位置
//                 |             |1 2           1-6 are possible chroma positions
// 2nd luma line > |X   X ...    |5 6 X ...
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
    { _T("0"), RGY_CHROMALOC_UNSPECIFIED },
    { _T("1"), RGY_CHROMALOC_LEFT },
    { _T("2"), RGY_CHROMALOC_CENTER },
    { _T("3"), RGY_CHROMALOC_TOPLEFT },
    { _T("4"), RGY_CHROMALOC_TOP },
    { _T("5"), RGY_CHROMALOC_BOTTOMLEFT },
    { _T("6"), RGY_CHROMALOC_BOTTOM },
    { _T("auto"),  COLOR_VALUE_AUTO },
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
    { _T("auto"),       COLOR_VALUE_AUTO },
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
        value = (T)list[(frame_height >= HD_HEIGHT_THRESHOLD) ? HD_INDEX : SD_INDEX].value;
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

    int codedWidth;     //[   (i)]
    int codedHeight;    //[   (i)]

                             //[      ] 出力解像度
    int dstWidth;

    //[      ] 出力解像度
    int dstHeight;

    //[      ] 出力解像度
    uint32_t dstPitch;

    //[    i ] 入力の取得した総フレーム数 (不明なら0)
    int frames;

    //[   (i)] 右shiftすべきビット数
    int shift;

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
};

enum RGYAVSync : uint32_t {
    RGY_AVSYNC_ASSUME_CFR = 0x00,
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

extern const CX_DESC list_log_level[];

const CX_DESC list_avsync[] = {
    { _T("cfr"),      RGY_AVSYNC_ASSUME_CFR   },
    { _T("vfr"),      RGY_AVSYNC_VFR       },
    { _T("forcecfr"), RGY_AVSYNC_FORCE_CFR },
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
    { _T("auto"),        (int)RGY_PICSTRUCT_AUTO },
    { NULL, 0 }
};

typedef std::map<RGY_CODEC, std::vector<RGY_CSP>> CodecCsp;
typedef std::vector<std::pair<int, CodecCsp>> DeviceCodecCsp;
typedef std::vector<std::pair<tstring, tstring>> muxOptList;

#endif //__RGY_DEF_H__
