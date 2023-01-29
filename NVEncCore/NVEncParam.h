// -----------------------------------------------------------------------------------------
// NVEnc by rigaya
// -----------------------------------------------------------------------------------------
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

#pragma once
#ifndef _NVENC_PARAM_H_
#define _NVENC_PARAM_H_

#include <limits.h>
#include <vector>
#include "rgy_osdep.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#pragma warning (disable: 4201)
#include "dynlink_cuviddec.h"
#include <npp.h>
#include "nvEncodeAPI.h"
#pragma warning (pop)
#include "rgy_tchar.h"
#include "rgy_util.h"
#include "rgy_simd.h"
#include "rgy_prm.h"
#include "convert_csp.h"

using std::vector;

static const TCHAR *FILTER_DEFAULT_CUSTOM_KERNEL_NAME = _T("kernel_filter");
static const int FILTER_DEFAULT_CUSTOM_THREAD_PER_BLOCK_X = 32;
static const int FILTER_DEFAULT_CUSTOM_THREAD_PER_BLOCK_Y = 8;
static const int FILTER_DEFAULT_CUSTOM_PIXEL_PER_THREAD_X = 1;
static const int FILTER_DEFAULT_CUSTOM_PIXEL_PER_THREAD_Y = 1;

static const int MAX_DECODE_FRAMES = 16;

static const int DEFAULT_GOP_LENGTH  = 0;
static const int DEFAULT_B_FRAMES    = 3;
static const int DEFAULT_REF_FRAMES  = 3;
static const int DEFAULT_NUM_SLICES  = 1;
static const int DEFAUTL_QP_I        = 20;
static const int DEFAULT_QP_P        = 23;
static const int DEFAULT_QP_B        = 25;
static const int DEFAULT_AVG_BITRATE = 7500000;
static const int DEFAULT_MAX_BITRATE = 17500000;
static const int DEFAULT_OUTPUT_BUF  = 8;
static const int DEFAULT_LOOKAHEAD   = 0;

static const int DEFAULT_CUDA_SCHEDULE = CU_CTX_SCHED_AUTO;

static const int PIPELINE_DEPTH = 4;
static const int MAX_FILTER_OUTPUT = 2;

enum {
    NV_ENC_AVCUVID_NATIVE = 0,
    NV_ENC_AVCUVID_CUDA,
};

typedef struct {
    GUID id;
    const TCHAR *desc;
    unsigned int value;
} guid_desc;

extern const GUID GUID_EMPTY; // 定義はNVEncUtil.cpp

const guid_desc h264_profile_names[] = {
    { NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID, _T("auto"),      0 },
    { NV_ENC_H264_PROFILE_BASELINE_GUID,    _T("baseline"), 66 },
    { NV_ENC_H264_PROFILE_MAIN_GUID,        _T("main"),     77 },
    { NV_ENC_H264_PROFILE_HIGH_GUID,        _T("high"),    100 },
    { NV_ENC_H264_PROFILE_HIGH_444_GUID,    _T("high444"), 244 },
    //{ NV_ENC_H264_PROFILE_STEREO_GUID,   _T("Stereo"),  128 }
};

enum {
    NV_ENC_PROFILE_HEVC_MAIN = 0,
    NV_ENC_PROFILE_HEVC_MAIN10 = 1,
    NV_ENC_PROFILE_HEVC_MAIN444 = 2
};

const guid_desc h265_profile_names[] = {
    //{ NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID, _T("auto"),                     0 },
    { NV_ENC_HEVC_PROFILE_MAIN_GUID,        _T("main"),    NV_ENC_PROFILE_HEVC_MAIN },
    { NV_ENC_HEVC_PROFILE_MAIN10_GUID,      _T("main10"),  NV_ENC_PROFILE_HEVC_MAIN10 },
    { NV_ENC_HEVC_PROFILE_FREXT_GUID,       _T("main444"), NV_ENC_PROFILE_HEVC_MAIN444 },
    //{ NV_ENC_HEVC_PROFILE_HIGH_GUID, _T("High"), NV_ENC_TIER_HEVC_HIGH },
};

const CX_DESC h265_tier_names[] = {
    { _T("main"),  NV_ENC_TIER_HEVC_MAIN },
    { _T("high"),  NV_ENC_TIER_HEVC_HIGH },
};

enum {
    NV_ENC_PROFILE_AV1_MAIN = 0,
    NV_ENC_PROFILE_AV1_HIGH = 1,
};

const guid_desc av1_profile_names[] = {
    //{ NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID, _T("auto"),                     0 },
    { NV_ENC_AV1_PROFILE_MAIN_GUID,      _T("main"),  NV_ENC_PROFILE_AV1_MAIN },
    { NV_ENC_AV1_PROFILE_HIGH_GUID,      _T("high"),  NV_ENC_PROFILE_AV1_HIGH },
    //{ NV_ENC_HEVC_PROFILE_HIGH_GUID, _T("High"), NV_ENC_TIER_HEVC_HIGH },
};

static const std::vector<guid_desc> get_codec_profile_list(const RGY_CODEC codec) {
    switch (codec) {
    case RGY_CODEC_H264: return make_vector(h264_profile_names);
    case RGY_CODEC_HEVC: return make_vector(h265_profile_names);
    case RGY_CODEC_AV1: return make_vector(av1_profile_names);
    default: return std::vector<guid_desc>();
    }
}

const CX_DESC av1_tier_names[] = {
    { _T("0"),  NV_ENC_TIER_AV1_0 },
    { _T("1"),  NV_ENC_TIER_AV1_1 },
};

enum {
    NVENC_PRESET_DEFAULT = 0,
    NVENC_PRESET_HP,
    NVENC_PRESET_P2,
    NVENC_PRESET_P3,
    NVENC_PRESET_P4,
    NVENC_PRESET_P5,
    NVENC_PRESET_P6,
    NVENC_PRESET_HQ,
    NVENC_PRESET_LL,
    NVENC_PRESET_LL_HP,
    NVENC_PRESET_LL_HQ,
    NVENC_PRESET_BD,
};

#pragma warning (push)
#pragma warning (disable: 4996)
RGY_DISABLE_WARNING_PUSH
RGY_DISABLE_WARNING_STR("-Wdeprecated-declarations")
const guid_desc list_nvenc_preset_names_ver9_2[] = {
    { NV_ENC_PRESET_DEFAULT_GUID,              _T("default"),                 NVENC_PRESET_DEFAULT },
    { NV_ENC_PRESET_HP_GUID,                   _T("performance"),             NVENC_PRESET_HP },
    { NV_ENC_PRESET_HQ_GUID,                   _T("quality"),                 NVENC_PRESET_HQ },
    { NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID,  _T("lowlatency"),              NVENC_PRESET_LL },
    { NV_ENC_PRESET_LOW_LATENCY_HP_GUID,       _T("lowlatency-performance"),  NVENC_PRESET_LL_HP },
    { NV_ENC_PRESET_LOW_LATENCY_HQ_GUID,       _T("lowlatency-quality"),      NVENC_PRESET_LL_HQ },
    //{ NV_ENC_PRESET_BD_GUID,                   _T("bluray"),                  NVENC_PRESET_BD },
};
RGY_DISABLE_WARNING_POP
#pragma warning (pop)

const guid_desc list_nvenc_preset_names_ver10[] = {
    { NV_ENC_PRESET_P1_GUID,                   _T("performance"),             NVENC_PRESET_HP },
    { NV_ENC_PRESET_P4_GUID,                   _T("default"),                 NVENC_PRESET_DEFAULT },
    { NV_ENC_PRESET_P7_GUID,                   _T("quality"),                 NVENC_PRESET_HQ },
    { NV_ENC_PRESET_P1_GUID,                   _T("P1"),                      NVENC_PRESET_HP },
    { NV_ENC_PRESET_P2_GUID,                   _T("P2"),                      NVENC_PRESET_P2 },
    { NV_ENC_PRESET_P3_GUID,                   _T("P3"),                      NVENC_PRESET_P3 },
    { NV_ENC_PRESET_P4_GUID,                   _T("P4"),                      NVENC_PRESET_P4 },
    { NV_ENC_PRESET_P5_GUID,                   _T("P5"),                      NVENC_PRESET_P5 },
    { NV_ENC_PRESET_P6_GUID,                   _T("P6"),                      NVENC_PRESET_P6 },
    { NV_ENC_PRESET_P7_GUID,                   _T("P7"),                      NVENC_PRESET_HQ },
};
static const int list_nvenc_preset_slower_than_default[] = {
    NVENC_PRESET_P5, NVENC_PRESET_P6, NVENC_PRESET_HQ, NVENC_PRESET_LL_HQ
};
static inline bool preset_slower_than_default(const int preset) {
    const auto end = list_nvenc_preset_slower_than_default + _countof(list_nvenc_preset_slower_than_default);
    return std::find(list_nvenc_preset_slower_than_default, end, preset) != end;
}

const guid_desc list_nvenc_codecs[] = {
    { NV_ENC_CODEC_H264_GUID, _T("H.264/AVC"),  RGY_CODEC_H264 },
    { NV_ENC_CODEC_HEVC_GUID, _T("H.265/HEVC"), RGY_CODEC_HEVC },
    { NV_ENC_CODEC_AV1_GUID,  _T("AV1"),        RGY_CODEC_AV1 },
};
const CX_DESC list_nvenc_multipass_mode[] = {
    { _T("none"),          NV_ENC_MULTI_PASS_DISABLED },
    { _T("2pass-quarter"), NV_ENC_TWO_PASS_QUARTER_RESOLUTION },
    { _T("2pass-full"),    NV_ENC_TWO_PASS_FULL_RESOLUTION },
    { NULL, 0 }
};

const CX_DESC list_nvenc_codecs_for_opt[] = {
    { _T("h264"), RGY_CODEC_H264 },
    { _T("avc"),  RGY_CODEC_H264 },
    { _T("hevc"), RGY_CODEC_HEVC },
    { _T("h265"), RGY_CODEC_HEVC },
    { _T("av1"),  RGY_CODEC_AV1  },
    { NULL, 0 }
};

const CX_DESC list_avc_level[] = {
    { _T("auto"), 0   },
    { _T("1"),    10  },
    { _T("1b"),   9   },
    { _T("1.1"),  11  },
    { _T("1.2"),  12  },
    { _T("1.3"),  13  },
    { _T("2"),    20  },
    { _T("2.1"),  21  },
    { _T("2.2"),  22  },
    { _T("3"),    30  },
    { _T("3.1"),  31  },
    { _T("3.2"),  32  },
    { _T("4"),    40  },
    { _T("4.1"),  41  },
    { _T("4.2"),  42  },
    { _T("5"),    50  },
    { _T("5.1"),  51  },
    { _T("5.2"),  52  },
    { _T("6"),    60  },
    { _T("6.1"),  61  },
    { _T("6.2"),  62  },
    { NULL, 0 }
};

const CX_DESC list_hevc_level[] = {
    { _T("auto"), 0   },
    { _T("1"),    NV_ENC_LEVEL_HEVC_1   },
    { _T("2"),    NV_ENC_LEVEL_HEVC_2   },
    { _T("2.1"),  NV_ENC_LEVEL_HEVC_21  },
    { _T("3"),    NV_ENC_LEVEL_HEVC_3   },
    { _T("3.1"),  NV_ENC_LEVEL_HEVC_31  },
    { _T("4"),    NV_ENC_LEVEL_HEVC_4   },
    { _T("4.1"),  NV_ENC_LEVEL_HEVC_41  },
    { _T("5"),    NV_ENC_LEVEL_HEVC_5   },
    { _T("5.1"),  NV_ENC_LEVEL_HEVC_51  },
    { _T("5.2"),  NV_ENC_LEVEL_HEVC_52  },
    { _T("6"),    NV_ENC_LEVEL_HEVC_6   },
    { _T("6.1"),  NV_ENC_LEVEL_HEVC_61  },
    { _T("6.2"),  NV_ENC_LEVEL_HEVC_62  },
    { NULL, 0 }
};

const CX_DESC list_av1_level[] = {
    { _T("auto"), NV_ENC_LEVEL_AV1_AUTOSELECT },
    { _T("2"),    NV_ENC_LEVEL_AV1_2   },
    { _T("2.1"),  NV_ENC_LEVEL_AV1_21  },
    { _T("2.2"),  NV_ENC_LEVEL_AV1_22  },
    { _T("2.3"),  NV_ENC_LEVEL_AV1_23  },
    { _T("3"),    NV_ENC_LEVEL_AV1_3   },
    { _T("3.1"),  NV_ENC_LEVEL_AV1_31  },
    { _T("3.2"),  NV_ENC_LEVEL_AV1_32  },
    { _T("3.3"),  NV_ENC_LEVEL_AV1_33  },
    { _T("4"),    NV_ENC_LEVEL_AV1_4   },
    { _T("4.1"),  NV_ENC_LEVEL_AV1_41  },
    { _T("4.2"),  NV_ENC_LEVEL_AV1_42  },
    { _T("4.3"),  NV_ENC_LEVEL_AV1_43  },
    { _T("5"),    NV_ENC_LEVEL_AV1_5   },
    { _T("5.1"),  NV_ENC_LEVEL_AV1_51  },
    { _T("5.2"),  NV_ENC_LEVEL_AV1_52  },
    { _T("5.3"),  NV_ENC_LEVEL_AV1_53  },
    { _T("6"),    NV_ENC_LEVEL_AV1_6   },
    { _T("6.1"),  NV_ENC_LEVEL_AV1_61  },
    { _T("6.2"),  NV_ENC_LEVEL_AV1_62  },
    { _T("6.3"),  NV_ENC_LEVEL_AV1_63  },
    { _T("7"),    NV_ENC_LEVEL_AV1_7   },
    { _T("7.1"),  NV_ENC_LEVEL_AV1_71  },
    { _T("7.2"),  NV_ENC_LEVEL_AV1_72  },
    { _T("7.3"),  NV_ENC_LEVEL_AV1_73  },
    { NULL, 0 }
};

static const CX_DESC *get_codec_level_list(RGY_CODEC codec) {
    switch (codec) {
    case RGY_CODEC_H264: return list_avc_level;
    case RGY_CODEC_HEVC: return list_hevc_level;
    case RGY_CODEC_AV1: return list_av1_level;
    default: return nullptr;
    }
}

const CX_DESC list_hevc_cu_size[] = {
    { _T("auto"), NV_ENC_HEVC_CUSIZE_AUTOSELECT },
    { _T("8"),    NV_ENC_HEVC_CUSIZE_8x8        },
    { _T("16"),   NV_ENC_HEVC_CUSIZE_16x16      },
    { _T("32"),   NV_ENC_HEVC_CUSIZE_32x32      },
    { _T("64"),   NV_ENC_HEVC_CUSIZE_64x64      },
    { NULL, 0 }
};

const CX_DESC list_mv_presicion[] = {
    { _T("auto"),     NV_ENC_MV_PRECISION_DEFAULT     },
    { _T("full-pel"), NV_ENC_MV_PRECISION_FULL_PEL    },
    { _T("half-pel"), NV_ENC_MV_PRECISION_HALF_PEL    },
    { _T("Q-pel"),    NV_ENC_MV_PRECISION_QUARTER_PEL },
    { NULL, 0 }
};

const CX_DESC list_nvenc_rc_method_en[] = {
    { _T("CQP"),                          NV_ENC_PARAMS_RC_CONSTQP   },
    { _T("CBR"),                          NV_ENC_PARAMS_RC_CBR       },
    { _T("CBRHQ"),                        NV_ENC_PARAMS_RC_CBR_HQ    },
    { _T("VBR"),                          NV_ENC_PARAMS_RC_VBR       },
    { _T("VBRHQ"),                        NV_ENC_PARAMS_RC_VBR_HQ    },
    { NULL, 0 }
};
const CX_DESC list_entropy_coding[] = {
    //{ _T("auto"),  NV_ENC_H264_ENTROPY_CODING_MODE_AUTOSELECT },
    { _T("cabac"), NV_ENC_H264_ENTROPY_CODING_MODE_CABAC      },
    { _T("cavlc"), NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC      },
    { NULL, 0 }
};

const CX_DESC list_bdirect[] = {
    { _T("auto"),     NV_ENC_H264_BDIRECT_MODE_AUTOSELECT },
    { _T("disabled"), NV_ENC_H264_BDIRECT_MODE_DISABLE    },
    { _T("temporal"), NV_ENC_H264_BDIRECT_MODE_TEMPORAL   },
    { _T("spatial"),  NV_ENC_H264_BDIRECT_MODE_SPATIAL    },
    { NULL, 0 }
};

static const int NV_ENC_BFRAME_REF_MODE_AUTO = -1;

const CX_DESC list_bref_mode[] = {
    { _T("auto"),     NV_ENC_BFRAME_REF_MODE_AUTO },
    { _T("disabled"), NV_ENC_BFRAME_REF_MODE_DISABLED },
    { _T("each"),     NV_ENC_BFRAME_REF_MODE_EACH },
    { _T("middle"),   NV_ENC_BFRAME_REF_MODE_MIDDLE },
    { NULL, 0 }
};

const CX_DESC list_fmo[] = {
    { _T("auto"),     NV_ENC_H264_FMO_AUTOSELECT },
    { _T("enabled"),  NV_ENC_H264_FMO_ENABLE     },
    { _T("disabled"), NV_ENC_H264_FMO_DISABLE    },
    { NULL, 0 }
};
const CX_DESC list_adapt_transform[] = {
    { _T("auto"),     NV_ENC_H264_ADAPTIVE_TRANSFORM_AUTOSELECT },
    { _T("disabled"), NV_ENC_H264_ADAPTIVE_TRANSFORM_DISABLE    },
    { _T("enabled"),  NV_ENC_H264_ADAPTIVE_TRANSFORM_ENABLE     },
    { NULL, 0 }
};
const CX_DESC list_bitdepth[] = {
    { _T("8bit"),    0 },
    { _T("10bit"),   2 },
    { NULL, 0 }
};

enum : uint32_t {
    NV_ENC_AQ_DISABLED = 0x00,
    NV_ENC_AQ_SPATIAL  = 0x01,
    NV_ENC_AQ_TEMPORAL = 0x02,
    NV_ENC_AQ_BOTH     = NV_ENC_AQ_SPATIAL | NV_ENC_AQ_TEMPORAL,
};
const CX_DESC list_aq[] = {
    { _T("disabled"), NV_ENC_AQ_DISABLED },
    { _T("spatial"),  NV_ENC_AQ_SPATIAL },
    { _T("temporal"), NV_ENC_AQ_TEMPORAL },
    { _T("both"),     NV_ENC_AQ_BOTH },
    { NULL, 0 }
};
#if 0
const CX_DESC list_preset[] = {
    { _T("fast"),    NV_ENC_PRESET_HP      },
    { _T("default"), NV_ENC_PRESET_DEFAULT },
    { _T("best"),    NV_ENC_PRESET_HQ      },
    { _T("bluray"),  NV_ENC_PRESET_BD      },
    { NULL, 0 }
};
const CX_DESC list_preset_ja[] = {
    { _T("高速"),   NV_ENC_PRESET_HP       },
    { _T("標準"),   NV_ENC_PRESET_DEFAULT  },
    { _T("高品質"), NV_ENC_PRESET_HQ       },
    { _T("Bluray"), NV_ENC_PRESET_BD       },
    { NULL, 0 }
};
#endif

const CX_DESC list_deinterlace[] = {
    { _T("none"),     cudaVideoDeinterlaceMode_Weave    },
    { _T("bob"),      cudaVideoDeinterlaceMode_Bob      },
    { _T("adaptive"), cudaVideoDeinterlaceMode_Adaptive },
    { _T("normal"),   cudaVideoDeinterlaceMode_Adaptive },
    { NULL, 0 }
};

const CX_DESC list_num_refs[] = {
    { _T("auto"),     NV_ENC_NUM_REF_FRAMES_AUTOSELECT },
    { _T("1"),        NV_ENC_NUM_REF_FRAMES_1          },
    { _T("2"),        NV_ENC_NUM_REF_FRAMES_2          },
    { _T("3"),        NV_ENC_NUM_REF_FRAMES_3          },
    { _T("4"),        NV_ENC_NUM_REF_FRAMES_4          },
    { _T("5"),        NV_ENC_NUM_REF_FRAMES_5          },
    { _T("6"),        NV_ENC_NUM_REF_FRAMES_6          },
    { _T("7"),        NV_ENC_NUM_REF_FRAMES_7          },
    { NULL, 0 }
};

const CX_DESC list_av1_refs_forward[] = {
    { _T("auto"),     NV_ENC_NUM_REF_FRAMES_AUTOSELECT },
    { _T("1"),        NV_ENC_NUM_REF_FRAMES_1          },
    { _T("2"),        NV_ENC_NUM_REF_FRAMES_2          },
    { _T("3"),        NV_ENC_NUM_REF_FRAMES_3          },
    { _T("4"),        NV_ENC_NUM_REF_FRAMES_4          },
    { NULL, 0 }
};

const CX_DESC list_av1_refs_backward[] = {
    { _T("auto"),     NV_ENC_NUM_REF_FRAMES_AUTOSELECT },
    { _T("1"),        NV_ENC_NUM_REF_FRAMES_1          },
    { _T("2"),        NV_ENC_NUM_REF_FRAMES_2          },
    { _T("3"),        NV_ENC_NUM_REF_FRAMES_3          },
    { NULL, 0 }
};

const CX_DESC list_av1_tiles[] = {
    { _T("auto"),     0 },
    { _T("1"),        1 },
    { _T("2"),        2 },
    { _T("4"),        4 },
    { _T("8"),        8 },
    { _T("16"),      16 },
    { _T("32"),      32 },
    { _T("64"),      64 },
    { NULL, 0 }
};

const CX_DESC list_part_size_av1[] = {
    { _T("auto"),  NV_ENC_AV1_PART_SIZE_AUTOSELECT },
    { _T("4"),     NV_ENC_AV1_PART_SIZE_4x4        },
    { _T("8"),     NV_ENC_AV1_PART_SIZE_8x8        },
    { _T("16"),    NV_ENC_AV1_PART_SIZE_16x16      },
    { _T("32"),    NV_ENC_AV1_PART_SIZE_32x32      },
    { _T("64"),    NV_ENC_AV1_PART_SIZE_64x64      },
    { NULL, 0 }
};

static const int DYNAMIC_PARAM_NOT_SELECTED = -1;

struct DynamicRCParam {
    int start;
    int end;
    NV_ENC_PARAMS_RC_MODE rc_mode;
    int avg_bitrate;
    int max_bitrate;
    int targetQuality;
    int targetQualityLSB;
    NV_ENC_QP qp;

    DynamicRCParam();
    tstring print() const;
    bool operator==(const DynamicRCParam &x) const;
    bool operator!=(const DynamicRCParam &x) const;
};
tstring printParams(const std::vector<DynamicRCParam> &dynamicRC);

const CX_DESC list_nppi_gauss[] = {
    { _T("disabled"), 0 },
    { _T("3"), NPP_MASK_SIZE_3_X_3 },
    { _T("5"), NPP_MASK_SIZE_5_X_5 },
    { _T("7"), NPP_MASK_SIZE_7_X_7 },
    { NULL, 0 }
};

const CX_DESC list_cuvid_mode[] = {
    { _T("native"), NV_ENC_AVCUVID_NATIVE },
    { _T("cuda"),   NV_ENC_AVCUVID_CUDA   },
    { NULL, 0 }
};

const CX_DESC list_cuda_schedule[] = {
    { _T("auto"),  CU_CTX_SCHED_AUTO },
    { _T("spin"),  CU_CTX_SCHED_SPIN },
    { _T("yield"), CU_CTX_SCHED_YIELD },
    { _T("sync"),  CU_CTX_SCHED_BLOCKING_SYNC },
    { NULL, 0 }
};

static const TCHAR *get_name_from_guid(GUID guid, const std::vector<guid_desc>& desc) {
    for (size_t i = 0; i < desc.size(); i++) {
        if (0 == memcmp(&desc[i].id, &guid, sizeof(GUID))) {
            return desc[i].desc;
        }
    }
    return _T("Unknown");
};

template<size_t count>
static const TCHAR *get_name_from_guid(GUID guid, const guid_desc (&desc)[count]) {
    for (size_t i = 0; i < count; i++) {
        if (0 == memcmp(&desc[i].id, &guid, sizeof(GUID))) {
            return desc[i].desc;
        }
    }
    return _T("Unknown");
};

template<size_t count>
static const TCHAR *get_name_from_value(int value, const guid_desc (&desc)[count]) {
    for (size_t i = 0; i < count; i++) {
        if ((int)desc[i].value == value) {
            return desc[i].desc;
        }
    }
    return _T("Unknown");
};

template<size_t count>
static int get_value_from_guid(GUID guid, const guid_desc (&desc)[count]) {
    for (size_t i = 0; i < count; i++) {
        if (0 == memcmp(&desc[i].id, &guid, sizeof(GUID))) {
            return desc[i].value;
        }
    }
    return 0;
};

template<size_t count>
static GUID get_guid_from_value(int value, const guid_desc (&desc)[count]) {
    for (size_t i = 0; i < count; i++) {
        if (desc[i].value == (uint32_t)value) {
            return desc[i].id;
        }
    }
    return GUID_EMPTY;
};

template<size_t count>
static GUID get_guid_from_name(const TCHAR *name, const guid_desc (&desc)[count]) {
    for (size_t i = 0; i < count; i++) {
        if (0 == _tcsicmp(name, desc[i].desc)) {
            return desc[i].id;
        }
    }
    return GUID_EMPTY;
};

template<size_t count>
static int get_value_from_name(const TCHAR *name, const guid_desc (&desc)[count]) {
    for (size_t i = 0; i < count; i++) {
        if (0 == _tcsicmp(name, desc[i].desc)) {
            return desc[i].value;
        }
    }
    return -1;
};

template<size_t count>
static int get_index_from_value(int value, const guid_desc (&desc)[count]) {
    for (size_t i = 0; i < count; i++) {
        if (desc[i].value == (uint32_t)value) {
            return i;
        }
    }
    return -1;
};

static inline bool is_interlaced(NV_ENC_PIC_STRUCT pic_struct) {
    return pic_struct != NV_ENC_PIC_STRUCT_FRAME;
}

const CX_DESC list_nvenc_caps_field_encoding[] = {
    { _T("no"),                 0 },
    { _T("field mode"),         1 },
    { _T("field + frame mode"), 2 },
    { NULL, 0 }
};

const CX_DESC list_nvenc_caps_bref_mode[] = {
    { _T("no"),                 0 },
    { _T("each"),               1 },
    { _T("only middle"),        2 },
    { _T("each + only middle"), 3 },
    { NULL, 0 }
};

const CX_DESC list_nvenc_caps_me_only[] = {
    { _T("no"),                 0 },
    { _T("I,P frames"),         1 },
    { _T("I,P,B frames"),       2 },
    { NULL, 0 }
};

typedef struct NVEncCap {
    int id;              //feature ID
    const TCHAR *name;   //feature名
    bool isBool;         //値がtrue/falseの値
    int value;           //featureの制限値
    const CX_DESC *desc; //説明
    const CX_DESC *desc_bit_flag; //説明
} NVEncCap;

//指定したIDのfeatureの値を取得する
static int get_value(int id, const std::vector<NVEncCap>& capList) {
    for (auto cap_info : capList) {
        if (cap_info.id == id)
            return cap_info.value;
    }
    return 0;
}

enum VppCustomInterface {
    VPP_CUSTOM_INTERFACE_PER_PLANE,
    VPP_CUSTOM_INTERFACE_PLANES,

    VPP_CUSTOM_INTERFACE_MAX,
};

const CX_DESC list_vpp_custom_interface[] = {
    { _T("per_plane"),    VPP_CUSTOM_INTERFACE_PER_PLANE },
    { _T("planes"),       VPP_CUSTOM_INTERFACE_PLANES },
    { NULL, 0 }
};

enum VppCustomInterlaceMode {
    VPP_CUSTOM_INTERLACE_UNSUPPORTED,
    VPP_CUSTOM_INTERLACE_PER_FIELD,
    VPP_CUSTOM_INTERLACE_FRAME,

    VPP_CUSTOM_INTERLACE_MAX,
};

const CX_DESC list_vpp_custom_interlace[] = {
    { _T("unsupported"), VPP_CUSTOM_INTERLACE_UNSUPPORTED },
    { _T("per_field"),   VPP_CUSTOM_INTERLACE_PER_FIELD },
    { _T("frame"),       VPP_CUSTOM_INTERLACE_FRAME },
    { NULL, 0 }
};

struct VppCustom {
    bool enable;
    tstring filter_name;
    tstring kernel_name;
    tstring kernel_path;
    std::string kernel;
    void *dev_params;
    std::string compile_options;
    VppCustomInterface kernel_interface;
    VppCustomInterlaceMode interlace;
    int threadPerBlockX;
    int threadPerBlockY;
    int pixelPerThreadX;
    int pixelPerThreadY;
    int dstWidth;
    int dstHeight;
    std::map<std::string, std::string> params;

    VppCustom();
    bool operator==(const VppCustom &x) const;
    bool operator!=(const VppCustom &x) const;
    tstring print() const;
};

struct VppParam {
    cudaVideoDeinterlaceMode deinterlace;
    NppiMaskSize             gaussMaskSize;

    VppParam();
};

struct InEncodeVideoParam {
    int deviceID;                 //使用するGPUのID
    int cudaSchedule;
    int sessionRetry;

    VideoInfo input;              //入力する動画の情報
    int preset;                   //出力プリセット
    int nHWDecType;               //
    int par[2];                   //使用されていません
    NV_ENC_CONFIG encConfig;      //エンコード設定
    std::vector<DynamicRCParam> dynamicRC;
    RGY_CODEC codec_rgy;          //出力コーデック
    int bluray;                   //bluray出力
    int yuv444;                   //YUV444出力
    int lossless;                 //ロスレス出力
    int losslessIgnoreInputCsp;
    int nWeightP;
    int chromaQPOffset;
    int brefMode;

    RGYParamCommon common;
    RGYParamInput inprm;
    RGYParamControl ctrl;
    RGYParamVpp vpp;
    VppParam vppnv;                 //vpp

    InEncodeVideoParam();

    void applyDOVIProfile();
};

NV_ENC_CONFIG DefaultParam();
NV_ENC_CODEC_CONFIG DefaultParamH264();
NV_ENC_CODEC_CONFIG DefaultParamHEVC();
NV_ENC_CODEC_CONFIG DefaultParamAV1();

static decltype(NV_ENC_CONFIG_H264::maxNumRefFrames)& numRefFrames(NV_ENC_CODEC_CONFIG& codec_config, const RGY_CODEC codec) {
    switch (codec) {
    case RGY_CODEC_H264: return codec_config.h264Config.maxNumRefFrames;
    case RGY_CODEC_HEVC: return codec_config.hevcConfig.maxNumRefFramesInDPB;
    case RGY_CODEC_AV1:  return codec_config.av1Config.maxNumRefFramesInDPB;
    default:
        abort();
    }
}

#define NV_CODEC_PARAM_ALL(x, other_return_val) \
static decltype(NV_ENC_CONFIG_H264::x) get_##x(const NV_ENC_CODEC_CONFIG& codec_config, const RGY_CODEC codec) { \
    switch (codec) { \
    case RGY_CODEC_HEVC: return codec_config.hevcConfig.x; \
    case RGY_CODEC_AV1:  return codec_config.av1Config.x; \
    case RGY_CODEC_H264: return codec_config.h264Config.x; \
    default: break; \
    } \
    return other_return_val; \
}; \
static void set_##x(NV_ENC_CODEC_CONFIG& codec_config, const RGY_CODEC codec, const decltype(NV_ENC_CONFIG_H264::x) value) { \
    switch (codec) { \
    case RGY_CODEC_HEVC: codec_config.hevcConfig.x = value; break; \
    case RGY_CODEC_AV1:  codec_config.av1Config.x = value; break; \
    case RGY_CODEC_H264: codec_config.h264Config.x = value; break; \
    default: break; \
    } \
};

#define NV_CODEC_PARAM_H264_HEVC(x, other_return_val) \
static decltype(NV_ENC_CONFIG_H264::x) get_##x(const NV_ENC_CODEC_CONFIG& codec_config, const RGY_CODEC codec) { \
    switch (codec) { \
    case RGY_CODEC_HEVC: return codec_config.hevcConfig.x; \
    case RGY_CODEC_H264: return codec_config.h264Config.x; \
    default: return other_return_val; \
    } \
}; \
static void set_##x(NV_ENC_CODEC_CONFIG& codec_config, const RGY_CODEC codec, const decltype(NV_ENC_CONFIG_H264::x) value) { \
    switch (codec) { \
    case RGY_CODEC_HEVC: codec_config.hevcConfig.x = value; break; \
    case RGY_CODEC_H264: codec_config.h264Config.x = value; break; \
    default: break; \
    } \
};

#define NV_CODEC_PARAM_HEVC_AV1(x, other_return_val) \
static decltype(NV_ENC_CONFIG_HEVC::x) get_##x(const NV_ENC_CODEC_CONFIG& codec_config, const RGY_CODEC codec) { \
    switch (codec) { \
    case RGY_CODEC_HEVC: return codec_config.hevcConfig.x; \
    case RGY_CODEC_AV1:  return codec_config.av1Config.x; \
    default: return other_return_val; \
    } \
}; \
static void set_##x(NV_ENC_CODEC_CONFIG& codec_config, const RGY_CODEC codec, const decltype(NV_ENC_CONFIG_HEVC::x) value) { \
    switch (codec) { \
    case RGY_CODEC_HEVC: codec_config.hevcConfig.x = value; break; \
    case RGY_CODEC_AV1:  codec_config.av1Config.x  = value; break; \
    default: break; \
    } \
};


#define NV_CODEC_PARAM_H264_HEVC_VUI(x, other_return_val) \
static decltype(NV_ENC_CONFIG_H264_VUI_PARAMETERS::x) get_##x(const NV_ENC_CODEC_CONFIG& codec_config, const RGY_CODEC codec) { \
    switch (codec) { \
    case RGY_CODEC_HEVC: return codec_config.hevcConfig.hevcVUIParameters.x; \
    case RGY_CODEC_H264: return codec_config.h264Config.h264VUIParameters.x; \
    default: return other_return_val; \
    } \
}; \
static void set_##x(NV_ENC_CODEC_CONFIG& codec_config, const RGY_CODEC codec, const int value) { \
    switch (codec) { \
    case RGY_CODEC_HEVC: codec_config.hevcConfig.hevcVUIParameters.x = (decltype(NV_ENC_CONFIG_H264_VUI_PARAMETERS::x))value; break; \
    case RGY_CODEC_H264: codec_config.h264Config.h264VUIParameters.x = (decltype(NV_ENC_CONFIG_HEVC_VUI_PARAMETERS::x))value; break; \
    default: break; \
    } \
};


NV_CODEC_PARAM_ALL(level, 0);
NV_CODEC_PARAM_HEVC_AV1(tier, 0);
NV_CODEC_PARAM_HEVC_AV1(pixelBitDepthMinus8, 0);
NV_CODEC_PARAM_ALL(idrPeriod, 300);
NV_CODEC_PARAM_ALL(useBFramesAsRef, NV_ENC_BFRAME_REF_MODE_DISABLED);
NV_CODEC_PARAM_H264_HEVC(enableLTR, false);
NV_CODEC_PARAM_H264_HEVC(ltrNumFrames, 0);
NV_CODEC_PARAM_H264_HEVC(sliceMode, 0);
NV_CODEC_PARAM_H264_HEVC(sliceModeData, 0);
NV_CODEC_PARAM_H264_HEVC(numRefL0, NV_ENC_NUM_REF_FRAMES_AUTOSELECT);
NV_CODEC_PARAM_H264_HEVC(numRefL1, NV_ENC_NUM_REF_FRAMES_AUTOSELECT);
NV_CODEC_PARAM_H264_HEVC(outputAUD, 0);
NV_CODEC_PARAM_H264_HEVC(outputPictureTimingSEI, 0);
NV_CODEC_PARAM_H264_HEVC(outputBufferingPeriodSEI, 0);
NV_CODEC_PARAM_H264_HEVC(repeatSPSPPS, 0);

static NV_ENC_VUI_COLOR_PRIMARIES get_colorprim(const NV_ENC_CODEC_CONFIG& codec_config, const RGY_CODEC codec) {
    switch (codec) {
    case RGY_CODEC_HEVC: return codec_config.hevcConfig.hevcVUIParameters.colourPrimaries;
    case RGY_CODEC_H264: return codec_config.h264Config.h264VUIParameters.colourPrimaries;
    case RGY_CODEC_AV1:  return codec_config.av1Config.colorPrimaries;
    default: return NV_ENC_VUI_COLOR_PRIMARIES_UNDEFINED;
    }
};
static void set_colorprim(NV_ENC_CODEC_CONFIG& codec_config, const RGY_CODEC codec, const int value) { \
    switch (codec) {
    case RGY_CODEC_HEVC: codec_config.hevcConfig.hevcVUIParameters.colourPrimaries = (NV_ENC_VUI_COLOR_PRIMARIES)value; break;
    case RGY_CODEC_H264: codec_config.h264Config.h264VUIParameters.colourPrimaries = (NV_ENC_VUI_COLOR_PRIMARIES)value; break;
    case RGY_CODEC_AV1:  codec_config.av1Config.colorPrimaries = (NV_ENC_VUI_COLOR_PRIMARIES)value; break;
    default: break;
    }
};
static NV_ENC_VUI_MATRIX_COEFFS get_colormatrix(const NV_ENC_CODEC_CONFIG& codec_config, const RGY_CODEC codec) {
    switch (codec) {
    case RGY_CODEC_HEVC: return codec_config.hevcConfig.hevcVUIParameters.colourMatrix;
    case RGY_CODEC_H264: return codec_config.h264Config.h264VUIParameters.colourMatrix;
    case RGY_CODEC_AV1:  return codec_config.av1Config.matrixCoefficients;
    default: return NV_ENC_VUI_MATRIX_COEFFS_UNSPECIFIED;
    }
};
static void set_colormatrix(NV_ENC_CODEC_CONFIG& codec_config, const RGY_CODEC codec, const int value) {
    switch (codec) {
    case RGY_CODEC_HEVC: codec_config.hevcConfig.hevcVUIParameters.colourMatrix = (NV_ENC_VUI_MATRIX_COEFFS)value; break;
    case RGY_CODEC_H264: codec_config.h264Config.h264VUIParameters.colourMatrix = (NV_ENC_VUI_MATRIX_COEFFS)value; break;
    case RGY_CODEC_AV1:  codec_config.av1Config.matrixCoefficients = (NV_ENC_VUI_MATRIX_COEFFS)value; break;
    default: break;
    }
};
static NV_ENC_VUI_TRANSFER_CHARACTERISTIC get_transfer(const NV_ENC_CODEC_CONFIG& codec_config, const RGY_CODEC codec) {
    switch (codec) {
    case RGY_CODEC_HEVC: return codec_config.hevcConfig.hevcVUIParameters.transferCharacteristics;
    case RGY_CODEC_H264: return codec_config.h264Config.h264VUIParameters.transferCharacteristics;
    case RGY_CODEC_AV1:  return codec_config.av1Config.transferCharacteristics;
    default: return NV_ENC_VUI_TRANSFER_CHARACTERISTIC_UNDEFINED;
    }
};
static void set_transfer(NV_ENC_CODEC_CONFIG& codec_config, const RGY_CODEC codec, const int value) {
    switch (codec) {
    case RGY_CODEC_HEVC: codec_config.hevcConfig.hevcVUIParameters.transferCharacteristics = (NV_ENC_VUI_TRANSFER_CHARACTERISTIC)value; break;
    case RGY_CODEC_H264: codec_config.h264Config.h264VUIParameters.transferCharacteristics = (NV_ENC_VUI_TRANSFER_CHARACTERISTIC)value; break;
    case RGY_CODEC_AV1:  codec_config.av1Config.transferCharacteristics = (NV_ENC_VUI_TRANSFER_CHARACTERISTIC)value; break;
    default: break;
    }
};

NV_CODEC_PARAM_H264_HEVC_VUI(videoFormat, NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED);
NV_CODEC_PARAM_H264_HEVC_VUI(overscanInfoPresentFlag, 0);
NV_CODEC_PARAM_H264_HEVC_VUI(overscanInfo, 0);
NV_CODEC_PARAM_H264_HEVC_VUI(videoSignalTypePresentFlag, 0);
NV_CODEC_PARAM_H264_HEVC_VUI(colourDescriptionPresentFlag, 0);
NV_CODEC_PARAM_H264_HEVC_VUI(chromaSampleLocationFlag, 0);
NV_CODEC_PARAM_H264_HEVC_VUI(chromaSampleLocationTop, 0);
NV_CODEC_PARAM_H264_HEVC_VUI(chromaSampleLocationBot, 0);

static uint32_t get_colorrange(const NV_ENC_CODEC_CONFIG& codec_config, const RGY_CODEC codec) {
    switch (codec) {
    case RGY_CODEC_HEVC: return codec_config.hevcConfig.hevcVUIParameters.videoFullRangeFlag;
    case RGY_CODEC_H264: return codec_config.h264Config.h264VUIParameters.videoFullRangeFlag;
    case RGY_CODEC_AV1:  return codec_config.av1Config.colorRange;
    default: return 0;
    }
};
static void set_colorrange(NV_ENC_CODEC_CONFIG& codec_config, const RGY_CODEC codec, const uint32_t value) {
    switch (codec) {
    case RGY_CODEC_HEVC: codec_config.hevcConfig.hevcVUIParameters.videoFullRangeFlag = value; break;
    case RGY_CODEC_H264: codec_config.h264Config.h264VUIParameters.videoFullRangeFlag = value; break;
    case RGY_CODEC_AV1:  codec_config.av1Config.colorRange = value; break;
    default: break;
    }
};

#endif //_NVENC_PARAM_H_
