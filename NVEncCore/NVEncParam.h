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
#include "NvHWEncoder.h"

using std::vector;

static const int   FILTER_DEFAULT_DELOGO_DEPTH = 128;
static const int   FILTER_DEFAULT_UNSHARP_RADIUS = 3;
static const float FILTER_DEFAULT_UNSHARP_WEIGHT = 0.5f;
static const float FILTER_DEFAULT_UNSHARP_THRESHOLD = 10.0f;
static const float FILTER_DEFAULT_EDGELEVEL_STRENGTH = 5.0f;
static const float FILTER_DEFAULT_EDGELEVEL_THRESHOLD = 20.0f;
static const float FILTER_DEFAULT_EDGELEVEL_BLACK = 0.0f;
static const float FILTER_DEFAULT_EDGELEVEL_WHITE = 0.0f;
static const int   FILTER_DEFAULT_KNN_RADIUS = 3;
static const float FILTER_DEFAULT_KNN_STRENGTH = 0.08f;
static const float FILTER_DEFAULT_KNN_LERPC = 0.20f;
static const float FILTER_DEFAULT_KNN_WEIGHT_THRESHOLD = 0.01f;
static const float FILTER_DEFAULT_KNN_LERPC_THRESHOLD = 0.80f;
static const float FILTER_DEFAULT_PMD_STRENGTH = 100.0f;
static const float FILTER_DEFAULT_PMD_THRESHOLD = 100.0f;
static const int   FILTER_DEFAULT_PMD_APPLY_COUNT = 2;
static const bool  FILTER_DEFAULT_PMD_USE_EXP = true;
static const int   FILTER_DEFAULT_SMOOTH_QUALITY = 3;
static const int   FILTER_DEFAULT_SMOOTH_QP = 12;
static const float FILTER_DEFAULT_SMOOTH_STRENGTH = 0.0f;
static const float FILTER_DEFAULT_SMOOTH_THRESHOLD = 0.0f;
static const int   FILTER_DEFAULT_SMOOTH_MODE = 0;
static const float FILTER_DEFAULT_SMOOTH_B_RATIO = 0.5f;
static const int   FILTER_DEFAULT_SMOOTH_MAX_QPTABLE_ERR = 10;
static const int   FILTER_DEFAULT_DEBAND_RANGE = 15;
static const int   FILTER_DEFAULT_DEBAND_THRE_Y = 15;
static const int   FILTER_DEFAULT_DEBAND_THRE_CB = 15;
static const int   FILTER_DEFAULT_DEBAND_THRE_CR = 15;
static const int   FILTER_DEFAULT_DEBAND_DITHER_Y = 15;
static const int   FILTER_DEFAULT_DEBAND_DITHER_C = 15;
static const int   FILTER_DEFAULT_DEBAND_MODE = 1;
static const int   FILTER_DEFAULT_DEBAND_SEED = 1234;
static const bool  FILTER_DEFAULT_DEBAND_BLUR_FIRST = false;
static const bool  FILTER_DEFAULT_DEBAND_RAND_EACH_FRAME = false;

static const int   FILTER_DEFAULT_DECIMATE_CYCLE = 5;
static const float FILTER_DEFAULT_DECIMATE_THRE_DUP = 1.1f;
static const float FILTER_DEFAULT_DECIMATE_THRE_SC = 15.0f;
static const int   FILTER_DEFAULT_DECIMATE_BLOCK_X = 32;
static const int   FILTER_DEFAULT_DECIMATE_BLOCK_Y = 32;
static const bool  FILTER_DEFAULT_DECIMATE_PREPROCESSED = false;
static const bool  FILTER_DEFAULT_DECIMATE_CHROMA = true;
static const bool  FILTER_DEFAULT_DECIMATE_LOG = false;

static const int   FILTER_DEFAULT_AFS_CLIP_TB = 16;
static const int   FILTER_DEFAULT_AFS_CLIP_LR = 32;
static const int   FILTER_DEFAULT_AFS_TB_ORDER = 0;
static const int   FILTER_DEFAULT_AFS_METHOD_SWITCH = 0;
static const int   FILTER_DEFAULT_AFS_COEFF_SHIFT = 192;
static const int   FILTER_DEFAULT_AFS_THRE_SHIFT = 128;
static const int   FILTER_DEFAULT_AFS_THRE_DEINT = 48;
static const int   FILTER_DEFAULT_AFS_THRE_YMOTION = 112;
static const int   FILTER_DEFAULT_AFS_THRE_CMOTION = 224;
static const int   FILTER_DEFAULT_AFS_ANALYZE = 3;
static const bool  FILTER_DEFAULT_AFS_SHIFT = true;
static const bool  FILTER_DEFAULT_AFS_DROP = false;
static const bool  FILTER_DEFAULT_AFS_SMOOTH = false;
static const bool  FILTER_DEFAULT_AFS_FORCE24 = false;
static const bool  FILTER_DEFAULT_AFS_TUNE = false;
static const bool  FILTER_DEFAULT_AFS_RFF = false;
static const bool  FILTER_DEFAULT_AFS_TIMECODE = false;
static const bool  FILTER_DEFAULT_AFS_LOG = false;

static const float FILTER_DEFAULT_TWEAK_BRIGHTNESS = 0.0f;
static const float FILTER_DEFAULT_TWEAK_CONTRAST = 1.0f;
static const float FILTER_DEFAULT_TWEAK_GAMMA = 1.0f;
static const float FILTER_DEFAULT_TWEAK_SATURATION = 1.0f;
static const float FILTER_DEFAULT_TWEAK_HUE = 0.0f;

static const double FILTER_DEFAULT_COLORSPACE_LDRNITS = 100.0;
static const double FILTER_DEFAULT_COLORSPACE_NOMINAL_SOURCE_PEAK = 100.0;
static const double FILTER_DEFAULT_COLORSPACE_HDR_SOURCE_PEAK = 1000.0;

static const double FILTER_DEFAULT_HDR2SDR_HABLE_A = 0.22;
static const double FILTER_DEFAULT_HDR2SDR_HABLE_B = 0.3;
static const double FILTER_DEFAULT_HDR2SDR_HABLE_C = 0.1;
static const double FILTER_DEFAULT_HDR2SDR_HABLE_D = 0.2;
static const double FILTER_DEFAULT_HDR2SDR_HABLE_E = 0.01;
static const double FILTER_DEFAULT_HDR2SDR_HABLE_F = 0.3;
static const double FILTER_DEFAULT_HDR2SDR_HABLE_W = 11.2;

static const double FILTER_DEFAULT_HDR2SDR_MOBIUS_TRANSITION = 0.3;
static const double FILTER_DEFAULT_HDR2SDR_MOBIUS_PEAK = 1.0;

static const double FILTER_DEFAULT_HDR2SDR_REINHARD_CONTRAST = 0.5;
static const double FILTER_DEFAULT_HDR2SDR_REINHARD_PEAK = 1.0;

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
static const int DEFAULT_LOOKAHEAD   = 16;

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
    { NV_ENC_PRESET_P1_GUID,                   _T("1"),                       NVENC_PRESET_HP },
    { NV_ENC_PRESET_P2_GUID,                   _T("2"),                       NVENC_PRESET_P2 },
    { NV_ENC_PRESET_P3_GUID,                   _T("3"),                       NVENC_PRESET_P3 },
    { NV_ENC_PRESET_P4_GUID,                   _T("4"),                       NVENC_PRESET_P4 },
    { NV_ENC_PRESET_P5_GUID,                   _T("5"),                       NVENC_PRESET_P5 },
    { NV_ENC_PRESET_P6_GUID,                   _T("6"),                       NVENC_PRESET_P6 },
    { NV_ENC_PRESET_P7_GUID,                   _T("7"),                       NVENC_PRESET_HQ },
};

const guid_desc list_nvenc_codecs[] = {
    { NV_ENC_CODEC_H264_GUID, _T("H.264/AVC"),  NV_ENC_H264 },
    { NV_ENC_CODEC_HEVC_GUID, _T("H.265/HEVC"), NV_ENC_HEVC },
};
const CX_DESC list_nvenc_multipass_mode[] = {
    { _T("none"),         NV_ENC_MULTI_PASS_DISABLED },
    { _T("2pass-quater"), NV_ENC_TWO_PASS_QUARTER_RESOLUTION },
    { _T("2pass-full"),   NV_ENC_TWO_PASS_FULL_RESOLUTION },
    { NULL, 0 }
};

const CX_DESC list_nvenc_codecs_for_opt[] = {
    { _T("h264"), NV_ENC_H264 },
    { _T("avc"),  NV_ENC_H264 },
    { _T("hevc"), NV_ENC_HEVC },
    { _T("h265"), NV_ENC_HEVC },
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

const CX_DESC list_mv_presicion_ja[] = {
    { _T("自動"),        NV_ENC_MV_PRECISION_DEFAULT     },
    { _T("1画素精度"),   NV_ENC_MV_PRECISION_FULL_PEL    },
    { _T("1/2画素精度"), NV_ENC_MV_PRECISION_HALF_PEL    },
    { _T("1/4画素精度"), NV_ENC_MV_PRECISION_QUARTER_PEL },
    { NULL, 0 }
};

const CX_DESC list_nvenc_rc_method[] = {
    { _T("CQP - 固定量子化量"),                     NV_ENC_PARAMS_RC_CONSTQP   },
    { _T("CBR - 固定ビットレート"),                 NV_ENC_PARAMS_RC_CBR       },
    //{ _T("CBR - 固定ビットレート (高品質)"),        NV_ENC_PARAMS_RC_CBR_HQ    },
    { _T("VBR - 可変ビットレート"),                 NV_ENC_PARAMS_RC_VBR       },
    //{ _T("VBR - 可変ビットレート (高品質)"),        NV_ENC_PARAMS_RC_VBR_HQ    },
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

const CX_DESC list_bref_mode[] = {
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

enum {
    NPPI_INTER_MAX = NPPI_INTER_LANCZOS3_ADVANCED,
    RESIZE_CUDA_TEXTURE_BILINEAR,
    RESIZE_CUDA_TEXTURE_NEAREST,
    RESIZE_CUDA_SPLINE16,
    RESIZE_CUDA_SPLINE36,
    RESIZE_CUDA_SPLINE64,
    RESIZE_CUDA_LANCZOS2,
    RESIZE_CUDA_LANCZOS3,
    RESIZE_CUDA_LANCZOS4,
};

const CX_DESC list_nppi_resize[] = {
    { _T("default"),       NPPI_INTER_UNDEFINED },
#if !defined(_M_IX86) || FOR_AUO
    { _T("nn"),            NPPI_INTER_NN },
    { _T("npp_linear"),    NPPI_INTER_LINEAR },
    { _T("cubic"),         NPPI_INTER_CUBIC },
    { _T("cubic_bspline"), NPPI_INTER_CUBIC2P_BSPLINE },
    { _T("cubic_catmull"), NPPI_INTER_CUBIC2P_CATMULLROM },
    { _T("cubic_b05c03"),  NPPI_INTER_CUBIC2P_B05C03 },
    { _T("super"),         NPPI_INTER_SUPER },
    { _T("lanczos"),       NPPI_INTER_LANCZOS },
#endif
    //{ _T("lanczons3"),     NPPI_INTER_LANCZOS3_ADVANCED },
    { _T("bilinear"),      RESIZE_CUDA_TEXTURE_BILINEAR },
    { _T("nearest"),       RESIZE_CUDA_TEXTURE_NEAREST },
    { _T("spline16"),      RESIZE_CUDA_SPLINE16 },
    { _T("spline36"),      RESIZE_CUDA_SPLINE36 },
    { _T("spline64"),      RESIZE_CUDA_SPLINE64 },
    { _T("lanczos2"),      RESIZE_CUDA_LANCZOS2 },
    { _T("lanczos3"),      RESIZE_CUDA_LANCZOS3 },
    { _T("lanczos4"),      RESIZE_CUDA_LANCZOS4 },
    { NULL, 0 }
};

const CX_DESC list_nppi_resize_help[] = {
    { _T("default"),       NPPI_INTER_UNDEFINED },
    { _T("nn"),            NPPI_INTER_NN },
    { _T("npp_linear"),    NPPI_INTER_LINEAR },
    { _T("cubic"),         NPPI_INTER_CUBIC },
    //{ _T("cubic_bspline"), NPPI_INTER_CUBIC2P_BSPLINE },
    //{ _T("cubic_catmull"), NPPI_INTER_CUBIC2P_CATMULLROM },
    //{ _T("cubic_b05c03"),  NPPI_INTER_CUBIC2P_B05C03 },
    { _T("super"),         NPPI_INTER_SUPER },
    { _T("lanczos"),       NPPI_INTER_LANCZOS },
    //{ _T("lanczons3"),     NPPI_INTER_LANCZOS3_ADVANCED },
    { _T("bilinear"),      RESIZE_CUDA_TEXTURE_BILINEAR },
    { _T("nearest"),       RESIZE_CUDA_TEXTURE_NEAREST },
    { _T("spline16"),      RESIZE_CUDA_SPLINE16 },
    { _T("spline36"),      RESIZE_CUDA_SPLINE36 },
    { _T("spline64"),      RESIZE_CUDA_SPLINE64 },
    { _T("lanczos2"),      RESIZE_CUDA_LANCZOS2 },
    { _T("lanczos3"),      RESIZE_CUDA_LANCZOS3 },
    { _T("lanczos4"),      RESIZE_CUDA_LANCZOS4 },
    { NULL, 0 }
};


enum VppFpPrecision {
    VPP_FP_PRECISION_UNKNOWN = -1,

    VPP_FP_PRECISION_AUTO = 0,
    VPP_FP_PRECISION_FP32,
    VPP_FP_PRECISION_FP16,

    VPP_FP_PRECISION_MAX,
};

const CX_DESC list_vpp_fp_prec[] = {
    { _T("auto"), VPP_FP_PRECISION_AUTO },
    { _T("fp32"), VPP_FP_PRECISION_FP32 },
    { _T("fp16"), VPP_FP_PRECISION_FP16 },
    { NULL, 0 }
};

const CX_DESC list_vpp_denoise[] = {
    { _T("none"),   0 },
    { _T("knn"),    1 },
    { _T("pmd"),    2 },
    { _T("smooth"), 3 },
    { NULL, 0 }
};

const CX_DESC list_vpp_detail_enahance[] = {
    { _T("none"),       0 },
    { _T("unsharp"),    1 },
    { _T("edgelevel"),  2 },
    { NULL, 0 }
};

const CX_DESC list_vpp_deband[] = {
    { _T("0 - 1点参照"),  0 },
    { _T("1 - 2点参照"),  1 },
    { _T("2 - 4点参照"),  2 },
    { NULL, 0 }
};

const CX_DESC list_vpp_rotate[] = {
    { _T("90"),   90 },
    { _T("180"), 180 },
    { _T("270"), 270 },
    { NULL, 0 }
};

enum HDR2SDRToneMap {
    HDR2SDR_DISABLED,
    HDR2SDR_HABLE,
    HDR2SDR_MOBIUS,
    HDR2SDR_REINHARD
};

const CX_DESC list_vpp_hdr2sdr[] = {
    { _T("none"),     HDR2SDR_DISABLED },
    { _T("hable"),    HDR2SDR_HABLE },
    { _T("mobius"),   HDR2SDR_MOBIUS },
    { _T("reinhard"), HDR2SDR_REINHARD },
    { NULL, 0 }
};

enum VppNnediField {
    VPP_NNEDI_FIELD_UNKNOWN = 0,
    VPP_NNEDI_FIELD_BOB_AUTO,
    VPP_NNEDI_FIELD_USE_AUTO,
    VPP_NNEDI_FIELD_USE_TOP,
    VPP_NNEDI_FIELD_USE_BOTTOM,
    VPP_NNEDI_FIELD_BOB_TOP_BOTTOM,
    VPP_NNEDI_FIELD_BOB_BOTTOM_TOP,

    VPP_NNEDI_FIELD_MAX,
};

const CX_DESC list_vpp_nnedi_field[] = {
    { _T("bob"),     VPP_NNEDI_FIELD_BOB_AUTO },
    { _T("auto"),    VPP_NNEDI_FIELD_USE_AUTO },
    { _T("top"),     VPP_NNEDI_FIELD_USE_TOP },
    { _T("bottom"),  VPP_NNEDI_FIELD_USE_BOTTOM },
    { _T("bob_tff"), VPP_NNEDI_FIELD_BOB_TOP_BOTTOM },
    { _T("bob_bff"), VPP_NNEDI_FIELD_BOB_BOTTOM_TOP },
    { NULL, 0 }
};

const CX_DESC list_vpp_nnedi_nns[] = {
    { _T("16"),   16 },
    { _T("32"),   32 },
    { _T("64"),   64 },
    { _T("128"), 128 },
    { _T("256"), 256 },
    { NULL, 0 }
};

enum VppNnediNSize {
    VPP_NNEDI_NSIZE_UNKNOWN = -1,

    VPP_NNEDI_NSIZE_8x6 = 0,
    VPP_NNEDI_NSIZE_16x6,
    VPP_NNEDI_NSIZE_32x6,
    VPP_NNEDI_NSIZE_48x6,
    VPP_NNEDI_NSIZE_8x4,
    VPP_NNEDI_NSIZE_16x4,
    VPP_NNEDI_NSIZE_32x4,

    VPP_NNEDI_NSIZE_MAX,
};

const CX_DESC list_vpp_nnedi_nsize[] = {
    { _T("8x6"),  VPP_NNEDI_NSIZE_8x6  },
    { _T("16x6"), VPP_NNEDI_NSIZE_16x6 },
    { _T("32x6"), VPP_NNEDI_NSIZE_32x6 },
    { _T("48x6"), VPP_NNEDI_NSIZE_48x6 },
    { _T("8x4"),  VPP_NNEDI_NSIZE_8x4  },
    { _T("16x4"), VPP_NNEDI_NSIZE_16x4 },
    { _T("32x4"), VPP_NNEDI_NSIZE_32x4 },
    { NULL, 0 }
};

enum VppNnediQuality {
    VPP_NNEDI_QUALITY_UNKNOWN = 0,
    VPP_NNEDI_QUALITY_FAST,
    VPP_NNEDI_QUALITY_SLOW,

    VPP_NNEDI_QUALITY_MAX,
};

const CX_DESC list_vpp_nnedi_quality[] = {
    { _T("fast"), VPP_NNEDI_QUALITY_FAST },
    { _T("slow"), VPP_NNEDI_QUALITY_SLOW },
    { NULL, 0 }
};

enum VppNnediPreScreen : uint32_t {
    VPP_NNEDI_PRE_SCREEN_NONE            = 0x00,
    VPP_NNEDI_PRE_SCREEN_ORIGINAL        = 0x01,
    VPP_NNEDI_PRE_SCREEN_NEW             = 0x02,
    VPP_NNEDI_PRE_SCREEN_MODE            = 0x07,
    VPP_NNEDI_PRE_SCREEN_BLOCK           = 0x10,
    VPP_NNEDI_PRE_SCREEN_ONLY            = 0x20,
    VPP_NNEDI_PRE_SCREEN_ORIGINAL_BLOCK  = VPP_NNEDI_PRE_SCREEN_ORIGINAL | VPP_NNEDI_PRE_SCREEN_BLOCK,
    VPP_NNEDI_PRE_SCREEN_NEW_BLOCK       = VPP_NNEDI_PRE_SCREEN_NEW      | VPP_NNEDI_PRE_SCREEN_BLOCK,
    VPP_NNEDI_PRE_SCREEN_ORIGINAL_ONLY   = VPP_NNEDI_PRE_SCREEN_ORIGINAL | VPP_NNEDI_PRE_SCREEN_ONLY,
    VPP_NNEDI_PRE_SCREEN_NEW_ONLY        = VPP_NNEDI_PRE_SCREEN_NEW      | VPP_NNEDI_PRE_SCREEN_ONLY,

    VPP_NNEDI_PRE_SCREEN_MAX,
};

static VppNnediPreScreen operator|(VppNnediPreScreen a, VppNnediPreScreen b) {
    return (VppNnediPreScreen)((uint32_t)a | (uint32_t)b);
}

static VppNnediPreScreen operator|=(VppNnediPreScreen& a, VppNnediPreScreen b) {
    a = a | b;
    return a;
}

static VppNnediPreScreen operator&(VppNnediPreScreen a, VppNnediPreScreen b) {
    return (VppNnediPreScreen)((uint32_t)a & (uint32_t)b);
}

static VppNnediPreScreen operator&=(VppNnediPreScreen& a, VppNnediPreScreen b) {
    a = (VppNnediPreScreen)((uint32_t)a & (uint32_t)b);
    return a;
}

const CX_DESC list_vpp_nnedi_pre_screen[] = {
    { _T("none"),           VPP_NNEDI_PRE_SCREEN_NONE },
    { _T("original"),       VPP_NNEDI_PRE_SCREEN_ORIGINAL },
    { _T("new"),            VPP_NNEDI_PRE_SCREEN_NEW },
    { _T("original_block"), VPP_NNEDI_PRE_SCREEN_ORIGINAL_BLOCK },
    { _T("new_block"),      VPP_NNEDI_PRE_SCREEN_NEW_BLOCK },
    { _T("original_only"),  VPP_NNEDI_PRE_SCREEN_ORIGINAL_ONLY },
    { _T("new_only"),       VPP_NNEDI_PRE_SCREEN_NEW_ONLY },
    { NULL, 0 }
};

enum VppNnediErrorType {
    VPP_NNEDI_ETYPE_ABS = 0,
    VPP_NNEDI_ETYPE_SQUARE,

    VPP_NNEDI_ETYPE_MAX,
};

const CX_DESC list_vpp_nnedi_error_type[] = {
    { _T("abs"),    VPP_NNEDI_ETYPE_ABS },
    { _T("square"), VPP_NNEDI_ETYPE_SQUARE },
    { NULL, 0 }
};

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

enum VppYadifMode : uint32_t {
    VPP_YADIF_MODE_UNKNOWN  = 0x00,

    VPP_YADIF_MODE_TFF      = 0x01,
    VPP_YADIF_MODE_BFF      = 0x02,
    VPP_YADIF_MODE_AUTO     = 0x04,
    VPP_YADIF_MODE_BOB      = 0x08,
    VPP_YADIF_MODE_BOB_TFF  = VPP_YADIF_MODE_BOB | VPP_YADIF_MODE_TFF,
    VPP_YADIF_MODE_BOB_BFF  = VPP_YADIF_MODE_BOB | VPP_YADIF_MODE_BFF,
    VPP_YADIF_MODE_BOB_AUTO = VPP_YADIF_MODE_BOB | VPP_YADIF_MODE_AUTO,

    VPP_YADIF_MODE_MAX = VPP_YADIF_MODE_BOB_AUTO + 1,
};

static VppYadifMode operator|(VppYadifMode a, VppYadifMode b) {
    return (VppYadifMode)((uint32_t)a | (uint32_t)b);
}

static VppYadifMode operator|=(VppYadifMode& a, VppYadifMode b) {
    a = a | b;
    return a;
}

static VppYadifMode operator&(VppYadifMode a, VppYadifMode b) {
    return (VppYadifMode)((uint32_t)a & (uint32_t)b);
}

static VppYadifMode operator&=(VppYadifMode& a, VppYadifMode b) {
    a = (VppYadifMode)((uint32_t)a & (uint32_t)b);
    return a;
}

const CX_DESC list_vpp_yadif_mode[] = {
    { _T("unknown"),  VPP_YADIF_MODE_UNKNOWN  },
    { _T("tff"),      VPP_YADIF_MODE_TFF      },
    { _T("bff"),      VPP_YADIF_MODE_BFF      },
    { _T("auto"),     VPP_YADIF_MODE_AUTO     },
    { _T("bob_tff"),  VPP_YADIF_MODE_BOB_TFF  },
    { _T("bob_bff"),  VPP_YADIF_MODE_BOB_BFF  },
    { _T("bob"),      VPP_YADIF_MODE_BOB_AUTO },
    { NULL, 0 }
};

const CX_DESC list_vpp_ass_shaping[] = {
    { _T("simple"),  0 },
    { _T("complex"), 1 },
    { NULL, 0 }
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
    return GUID{ 0 };
};

template<size_t count>
static GUID get_guid_from_name(const TCHAR *name, const guid_desc (&desc)[count]) {
    for (size_t i = 0; i < count; i++) {
        if (0 == _tcsicmp(name, desc[i].desc)) {
            return desc[i].id;
        }
    }
    return GUID{ 0 };
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

typedef struct NVEncCap {
    int id;            //feature ID
    const TCHAR *name; //feature名
    bool isBool;       //値がtrue/falseの値
    int value;         //featureの制限値
} NVEncCap;

//指定したIDのfeatureの値を取得する
static int get_value(int id, const std::vector<NVEncCap>& capList) {
    for (auto cap_info : capList) {
        if (cap_info.id == id)
            return cap_info.value;
    }
    return 0;
}

struct VppDelogo {
    bool enable;
    tstring logoFilePath;  //ロゴファイル名
    tstring logoSelect;    //ロゴの名前
    int posX, posY; //位置オフセット
    int depth;      //透明度深度
    int Y, Cb, Cr;  //(輝度・色差)オフセット
    int mode;
    bool autoFade;
    bool autoNR;
    int NRArea;
    int NRValue;
    bool log;

    VppDelogo();
    bool operator==(const VppDelogo& x) const;
    bool operator!=(const VppDelogo& x) const;
    tstring print() const;
};

struct VppUnsharp {
    bool  enable;
    int   radius;
    float weight;
    float threshold;

    VppUnsharp();
    bool operator==(const VppUnsharp& x) const;
    bool operator!=(const VppUnsharp& x) const;
    tstring print() const;
};

struct VppEdgelevel {
    bool  enable;
    float strength;
    float threshold;
    float black;
    float white;

    VppEdgelevel();
    bool operator==(const VppEdgelevel& x) const;
    bool operator!=(const VppEdgelevel& x) const;
    tstring print() const;
};

struct VppKnn {
    bool  enable;
    int   radius;
    float strength;
    float lerpC;
    float weight_threshold;
    float lerp_threshold;

    VppKnn();
    bool operator==(const VppKnn& x) const;
    bool operator!=(const VppKnn& x) const;
    tstring print() const;
};

struct VppPmd {
    bool  enable;
    float strength;
    float threshold;
    int   applyCount;
    bool  useExp;

    VppPmd();
    bool operator==(const VppPmd& x) const;
    bool operator!=(const VppPmd& x) const;
    tstring print() const;
};

struct VppDeband {
    bool enable;
    int range;
    int threY;
    int threCb;
    int threCr;
    int ditherY;
    int ditherC;
    int sample;
    int seed;
    bool blurFirst;
    bool randEachFrame;

    VppDeband();
    bool operator==(const VppDeband& x) const;
    bool operator!=(const VppDeband& x) const;
    tstring print() const;
};

const CX_DESC list_vpp_decimate_block[] = {
    { _T("4"),    4 },
    { _T("8"),    8 },
    { _T("16"),  16 },
    { _T("32"),  32 },
    { _T("64"),  64 },
    { NULL, 0 }
};

struct VppDecimate {
    bool enable;
    int cycle;
    float threDuplicate;
    float threSceneChange;
    int blockX;
    int blockY;
    bool preProcessed;
    bool chroma;
    bool log;

    VppDecimate();
    bool operator==(const VppDecimate &x) const;
    bool operator!=(const VppDecimate &x) const;
    tstring print() const;
};

struct VppSmooth {
    bool enable;
    int quality;
    int qp;
    VppFpPrecision prec;
    bool useQPTable;
    float strength;
    float threshold;
    float bratio;
    int maxQPTableErrCount;
    VppSmooth();
    bool operator==(const VppSmooth &x) const;
    bool operator!=(const VppSmooth &x) const;
    tstring print() const;
};

struct ColorspaceConv {
    VideoVUIInfo from, to;
    double sdr_source_peak;
    bool approx_gamma;
    bool scene_ref;

    ColorspaceConv();
    void set(const VideoVUIInfo& csp_from, const VideoVUIInfo &csp_to) {
        from = csp_from;
        to = csp_to;
    }
    bool operator==(const ColorspaceConv &x) const;
    bool operator!=(const ColorspaceConv &x) const;
};

struct TonemapHable {
    double a, b, c, d, e, f;

    TonemapHable();
    bool operator==(const TonemapHable &x) const;
    bool operator!=(const TonemapHable &x) const;
};

struct TonemapMobius {
    double transition, peak;

    TonemapMobius();
    bool operator==(const TonemapMobius &x) const;
    bool operator!=(const TonemapMobius &x) const;
};

struct TonemapReinhard {
    double contrast, peak;

    TonemapReinhard();
    bool operator==(const TonemapReinhard &x) const;
    bool operator!=(const TonemapReinhard &x) const;
};

struct HDR2SDRParams {
    HDR2SDRToneMap tonemap;
    TonemapHable hable;
    TonemapMobius mobius;
    TonemapReinhard reinhard;
    double ldr_nits;
    double hdr_source_peak;

    HDR2SDRParams();
    bool operator==(const HDR2SDRParams &x) const;
    bool operator!=(const HDR2SDRParams &x) const;
};

struct VppColorspace {
    bool enable;
    HDR2SDRParams hdr2sdr;
    vector<ColorspaceConv> convs;

    VppColorspace();
    bool operator==(const VppColorspace &x) const;
    bool operator!=(const VppColorspace &x) const;
};

struct VppTweak {
    bool  enable;
    float brightness; // -1.0 - 1.0 (0.0)
    float contrast;   // -2.0 - 2.0 (1.0)
    float gamma;      //  0.1 - 10.0 (1.0)
    float saturation; //  0.0 - 3.0 (1.0)
    float hue;        // -180 - 180 (0.0)

    VppTweak();
    bool operator==(const VppTweak& x) const;
    bool operator!=(const VppTweak& x) const;
    tstring print() const;
};

struct VppTransform {
    bool enable;
    bool transpose;
    bool flipX;
    bool flipY;

    VppTransform();
    int rotate() const;
    bool setRotate(int rotate);
    bool operator==(const VppTransform &x) const;
    bool operator!=(const VppTransform &x) const;
    tstring print() const;
};

struct VppSelectEvery {
    bool  enable;
    int   step;
    int   offset;

    VppSelectEvery();
    bool operator==(const VppSelectEvery& x) const;
    bool operator!=(const VppSelectEvery& x) const;
    tstring print() const;
};

struct VppSubburn {
    bool  enable;
    tstring filename;
    std::string charcode;
    int trackId;
    int assShaping;
    float scale;
    float transparency_offset;
    float brightness;
    float contrast;
    double ts_offset;
    bool vid_ts_offset;

    VppSubburn();
    bool operator==(const VppSubburn &x) const;
    bool operator!=(const VppSubburn &x) const;
    tstring print() const;
};

typedef struct {
    int top, bottom, left, right;
} AFS_SCAN_CLIP;

static inline AFS_SCAN_CLIP scan_clip(int top, int bottom, int left, int right) {
    AFS_SCAN_CLIP clip;
    clip.top = top;
    clip.bottom = bottom;
    clip.left = left;
    clip.right = right;
    return clip;
}

struct VppAfs {
    bool enable;
    int tb_order;
    AFS_SCAN_CLIP clip;    //上下左右
    int method_switch;     //切替点
    int coeff_shift;       //判定比
    int thre_shift;        //縞(ｼﾌﾄ)
    int thre_deint;        //縞(解除)
    int thre_Ymotion;      //Y動き
    int thre_Cmotion;      //C動き
    int analyze;           //解除Lv
    bool shift;            //フィールドシフト
    bool drop;             //間引き
    bool smooth;           //スムージング
    bool force24;          //24fps化
    bool tune;             //調整モード
    bool rff;              //rffフラグを認識して調整
    bool timecode;         //timecode出力
    bool log;              //log出力

    VppAfs();
    void set_preset(int preset);
    int read_afs_inifile(const TCHAR* inifile);
    bool operator==(const VppAfs& x) const;
    bool operator!=(const VppAfs& x) const;
    tstring print() const;

    void check();
};

struct VppYadif {
    bool enable;
    VppYadifMode mode;

    VppYadif();
    bool operator==(const VppYadif& x) const;
    bool operator!=(const VppYadif& x) const;
    tstring print() const;
};

struct VppPad {
    bool enable;
    int left, top, right, bottom;

    VppPad();
    bool operator==(const VppPad& x) const;
    bool operator!=(const VppPad& x) const;
    tstring print() const;
};

struct VppNnedi {
    bool              enable;
    VppNnediField     field;
    int               nns;
    VppNnediNSize     nsize;
    VppNnediQuality   quality;
    VppFpPrecision precision;
    VppNnediPreScreen pre_screen;
    VppNnediErrorType errortype;
    tstring           weightfile;

    bool isbob();
    VppNnedi();
    bool operator==(const VppNnedi& x) const;
    bool operator!=(const VppNnedi& x) const;
    tstring print() const;
};

enum {
    AFS_PRESET_DEFAULT = 0,
    AFS_PRESET_TRIPLE,        //動き重視
    AFS_PRESET_DOUBLE,        //二重化
    AFS_PRESET_ANIME,                     //映画/アニメ
    AFS_PRESET_CINEMA = AFS_PRESET_ANIME, //映画/アニメ
    AFS_PRESET_MIN_AFTERIMG,              //残像最小化
    AFS_PRESET_FORCE24_SD,                //24fps固定
    AFS_PRESET_FORCE24_HD,                //24fps固定 (HD)
    AFS_PRESET_FORCE30,                   //30fps固定
};

const CX_DESC list_afs_preset[] = {
    { _T("default"),      AFS_PRESET_DEFAULT },
    { _T("triple"),       AFS_PRESET_TRIPLE },
    { _T("double"),       AFS_PRESET_DOUBLE },
    { _T("anime/cinema"), AFS_PRESET_ANIME },
    { _T("anime"),        AFS_PRESET_ANIME },
    { _T("cinema"),       AFS_PRESET_CINEMA },
    { _T("min_afterimg"), AFS_PRESET_MIN_AFTERIMG },
    { _T("24fps"),        AFS_PRESET_FORCE24_HD },
    { _T("24fps_sd"),     AFS_PRESET_FORCE24_SD },
    { _T("30fps"),        AFS_PRESET_FORCE30 },
    { NULL, 0 }
};

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
    bool checkPerformance;
    cudaVideoDeinterlaceMode deinterlace;
    int                      resizeInterp;
    NppiMaskSize             gaussMaskSize;

    VppDelogo delogo;
    VppUnsharp unsharp;
    VppEdgelevel edgelevel;
    VppKnn knn;
    VppPmd pmd;
    VppSmooth smooth;
    VppDeband deband;
    VppAfs afs;
    VppNnedi nnedi;
    VppYadif yadif;
    VppTweak tweak;
    VppTransform transform;
    VppColorspace colorspace;
    VppPad pad;
    std::vector<VppSubburn> subburn;
    VppSelectEvery selectevery;
    VppDecimate decimate;
    bool rff;

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
    int codec;                    //出力コーデック
    int bluray;                   //bluray出力
    int yuv444;                   //YUV444出力
    int lossless;                 //ロスレス出力
    int nWeightP;

    RGYParamCommon common;
    RGYParamControl ctrl;
    VppParam vpp;                 //vpp
    bool ssim;
    bool psnr;

    InEncodeVideoParam();
};

NV_ENC_CONFIG DefaultParam();
NV_ENC_CODEC_CONFIG DefaultParamH264();
NV_ENC_CODEC_CONFIG DefaultParamHEVC();
