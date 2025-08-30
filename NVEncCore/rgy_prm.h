﻿// -----------------------------------------------------------------------------------------
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
#ifndef __RGY_PRM_H__
#define __RGY_PRM_H__

#include "rgy_def.h"
#include "rgy_log.h"
#include "rgy_util.h"
#include "rgy_thread_affinity.h"
#include "rgy_simd.h"
#include "rgy_hdr10plus.h"

static const int BITSTREAM_BUFFER_SIZE =  4 * 1024 * 1024;
static const int OUTPUT_BUF_SIZE       = 16 * 1024 * 1024;

static const int RGY_DEFAULT_PERF_MONITOR_INTERVAL = 500;
static const int DEFAULT_IGNORE_DECODE_ERROR = 10;
static const int DEFAULT_VIDEO_IGNORE_TIMESTAMP_ERROR = 10;

static const float DEFAULT_DUMMY_LOAD_PERCENT = 0.01f;

static const int RGY_AUDIO_QUALITY_DEFAULT = 0;

#if ENCODER_NVENC
#define ENABLE_VPP_FILTER_COLORSPACE   (ENABLE_NVRTC)
#else
#define ENABLE_VPP_FILTER_COLORSPACE   (ENCODER_QSV                    || ENCODER_VCEENC || ENCODER_MPP || CLFILTERS_AUF)
#endif
#define ENABLE_VPP_FILTER_AFS          (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP)
#define ENABLE_VPP_FILTER_NNEDI        (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP || CLFILTERS_AUF)
#define ENABLE_VPP_FILTER_YADIF        (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP)
#define ENABLE_VPP_FILTER_DECOMB       (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP)
#define ENABLE_VPP_FILTER_RFF          (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP)
#define ENABLE_VPP_FILTER_RFF_AVHW     (ENCODER_QSV   || ENCODER_NVENC                   || ENCODER_MPP)
#define ENABLE_VPP_FILTER_SELECT_EVERY (ENCODER_NVENC)
#define ENABLE_VPP_FILTER_DECIMATE     (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP)
#define ENABLE_VPP_FILTER_MPDECIMATE   (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP)
#define ENABLE_VPP_FILTER_PAD          (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP || CLFILTERS_AUF)
#define ENABLE_VPP_FILTER_NLMEANS      (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP || CLFILTERS_AUF)
#define ENABLE_VPP_FILTER_PMD          (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP || CLFILTERS_AUF)
#define ENABLE_VPP_FILTER_DENOISE_DCT  (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP || CLFILTERS_AUF)
#define ENABLE_VPP_FILTER_SMOOTH       (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP || CLFILTERS_AUF)
#define ENABLE_VPP_FILTER_FFT3D        (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP)
#define ENABLE_VPP_FILTER_CONVOLUTION3D (ENCODER_QSV  || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP)
#define ENABLE_VPP_FILTER_UNSHARP      (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP || CLFILTERS_AUF)
#define ENABLE_VPP_FILTER_WARPSHARP    (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP || CLFILTERS_AUF)
#define ENABLE_VPP_FILTER_EDGELEVEL    (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP || CLFILTERS_AUF)
#define ENABLE_VPP_FILTER_CURVES       (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP)
#define ENABLE_VPP_FILTER_TWEAK        (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP || CLFILTERS_AUF)
#define ENABLE_VPP_FILTER_OVERLAY      (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP)
#define ENABLE_VPP_FILTER_DEBAND       (ENCODER_QSV   || ENCODER_NVENC || ENCODER_VCEENC || ENCODER_MPP || CLFILTERS_AUF)
#define ENABLE_VPP_FILTER_LIBPLACEBO   (ENABLE_LIBPLACEBO && (ENCODER_QSV || ENCODER_NVENC || ENCODER_VCEENC || CLFILTERS_AUF))
#define ENABLE_VPP_FILTER_FRUC         (                 ENCODER_NVENC)
#define ENABLE_VPP_FILTER_DELOGO_MULTIADD  (             ENCODER_NVENC)
#define ENABLE_VPP_ORDER                   (CLFILTERS_AUF)

#define ENABLE_PARALLEL_ENC            (ENCODER_QSV   || ENCODER_NVENC || (ENCODER_VCEENC && ENABLE_D3D11))

enum class VppType : int {
    VPP_NONE,
#if ENCODER_QSV
    MFX_COLORSPACE,
    MFX_CROP,
    MFX_ROTATE,
    MFX_MIRROR,
    MFX_DEINTERLACE,
    MFX_IMAGE_STABILIZATION,
    MFX_MCTF,
    MFX_DENOISE,
    MFX_RESIZE,
    MFX_AISUPRERES,
    MFX_DETAIL_ENHANCE,
    MFX_FPS_CONV,
    MFX_PERC_ENC_PREFILTER,
    MFX_AI_FRAMEINTERP,
    MFX_COPY,
#endif //#if ENCODER_QSV
    MFX_MAX,
#if ENCODER_NVENC || CLFILTERS_AUF
    NVVFX_DENOISE,
    NVVFX_ARTIFACT_REDUCTION,
#endif
    NVVFX_MAX,
#if ENCODER_NVENC || CLFILTERS_AUF
    NGX_TRUEHDR,
#endif
    NGX_MAX,
#if ENCODER_NVENC
    NPP_GAUSS,
#endif
    NPP_MAX,
#if ENCODER_VCEENC
    AMF_CONVERTER,
    AMF_PREPROCESS,
    AMF_RESIZE,
    AMF_VQENHANCE,
    AMF_FRC,
#endif
    AMF_MAX,
#if ENCODER_MPP
    IEP_MIN = AMF_MAX,
    IEP_DEINTERLACE,
#endif
    IEP_MAX,
#if ENCODER_MPP
    RGA_MIN = IEP_MAX,
    RGA_CROP,
    RGA_CSPCONV,
    RGA_RESIZE,
#endif
    RGA_MAX,

    CL_MIN = RGA_MAX,

    CL_CROP,
    CL_COLORSPACE,
    CL_LIBPLACEBO_TONEMAP,
    CL_AFS,
    CL_NNEDI,
    CL_YADIF,
    CL_DECOMB,
    CL_DECIMATE,
    CL_MPDECIMATE,
    CL_RFF,
    CL_DELOGO,
    CL_SELECT_EVERY,
    CL_TRANSFORM,

    CL_CONVOLUTION3D,
    CL_DENOISE_KNN,
    CL_DENOISE_NLMEANS,
    CL_DENOISE_PMD,
    CL_DENOISE_DCT,
    CL_DENOISE_SMOOTH,
    CL_DENOISE_FFT3D,

    CL_LIBPLACEBO_SHADER,

    CL_RESIZE,

    CL_SUBBURN,

    CL_UNSHARP,
    CL_EDGELEVEL,
    CL_WARPSHARP,

    CL_CURVES,
    CL_TWEAK,

    CL_OVERLAY,

    CL_DEBAND,
    CL_LIBPLACEBO_DEBAND,

    CL_FRUC,

    CL_PAD,

    CL_MAX,
};

enum class VppFilterType { FILTER_NONE, FILTER_MFX, FILTER_NVVFX, FILTER_NGX, FILTER_NPP, FILTER_AMF, FILTER_IEP, FILTER_RGA, FILTER_OPENCL, FILTER_CUDA = FILTER_OPENCL };

static VppFilterType getVppFilterType(VppType vpptype) {
    if (vpptype == VppType::VPP_NONE) return VppFilterType::FILTER_NONE;
#if ENCODER_QSV
    if (vpptype < VppType::MFX_MAX) return VppFilterType::FILTER_MFX;
#endif // #if ENCODER_QSV
#if ENCODER_NVENC || CLFILTERS_AUF
    if (vpptype < VppType::NVVFX_MAX) return VppFilterType::FILTER_NVVFX;
#endif // #if ENCODER_NVENC || CLFILTERS_AUF
#if ENCODER_NVENC || CLFILTERS_AUF
    if (vpptype < VppType::NGX_MAX) return VppFilterType::FILTER_NGX;
#endif // #if ENCODER_NVENC || CLFILTERS_AUF
#if ENCODER_NVENC
    if (vpptype < VppType::NPP_MAX) return VppFilterType::FILTER_NPP;
#endif // #if ENCODER_NVENC
#if ENCODER_VCEENC
    if (vpptype < VppType::AMF_MAX) return VppFilterType::FILTER_AMF;
#endif
#if ENCODER_MPP
    if (vpptype < VppType::IEP_MAX) return VppFilterType::FILTER_IEP;
    if (vpptype < VppType::RGA_MAX) return VppFilterType::FILTER_RGA;
#endif
    if (vpptype < VppType::CL_MAX) return VppFilterType::FILTER_OPENCL;
    return VppFilterType::FILTER_NONE;
}

tstring vppfilter_type_to_str(VppType type);
VppType vppfilter_str_to_type(tstring str);
std::vector<CX_DESC> get_list_vpp_filter();

static const TCHAR* VMAF_DEFAULT_MODEL_VERSION = _T("vmaf_v0.6.1");

static const float FILTER_DEFAULT_LIBPLACEBO_RESAMPLE_RADIUS = -1.0f;
static const float FILTER_DEFAULT_LIBPLACEBO_RESAMPLE_CLAMP = 0.0f;
static const float FILTER_DEFAULT_LIBPLACEBO_RESAMPLE_TAPER = 0.0f;
static const float FILTER_DEFAULT_LIBPLACEBO_RESAMPLE_BLUR = 0.0f;
static const float FILTER_DEFAULT_LIBPLACEBO_RESAMPLE_ANTIRING = 0.0f;
static const int   FILTER_DEFAULT_LIBPLACEBO_RESAMPLE_CPLACE = 0;

static const int   FILTER_DEFAULT_LIBPLACEBO_DEBAND_ITERATIONS = 1;
static const float FILTER_DEFAULT_LIBPLACEBO_DEBAND_THRESHOLD = 4.0f;
static const float FILTER_DEFAULT_LIBPLACEBO_DEBAND_RADIUS = 16.0f;
static const float FILTER_DEFAULT_LIBPLACEBO_DEBAND_GRAINY = 6.0f;
static const float FILTER_DEFAULT_LIBPLACEBO_DEBAND_GRAINC = -1.0f;
static const int   FILTER_DEFAULT_LIBPLACEBO_DEBAND_DITHER = 1;
static const int   FILTER_DEFAULT_LIBPLACEBO_DEBAND_LUT_SIZE = 6;

static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_KNEES_ADAPTATION = 0.4f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_KNEES_MIN = 0.1f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_KNEES_MAX = 0.8f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_KNEES_DEFAULT = 0.4f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_KNEES_OFFSET = 1.0f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_SLOPE_TUNING = 1.5f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_SLOPE_OFFSET = 0.2f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_SPLINE_CONTRAST = 0.5f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_REINHARD_CONTRAST = 0.5f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_LINEAR_KNEE = 0.3f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_EXPOSURE = 1.0f;

static const int   FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_SRC_CSP = -1;
static const int   FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_DST_CSP = -1;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_SRC_MAX = -1.0f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_SRC_MIN = -1.0f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_DST_MAX = -1.0f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_DST_MIN = -1.0f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_NIT_MAX_HDR = 1000.0f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_NIT_MIN_HDR = 0.005f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_NIT_MAX_SDR = 203.0f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_NIT_MIN_SDR = 0.2023f;
static const bool  FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_DYNAMIC_PEAK_DETECTION = true;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_SMOOTH_PERIOD = 20.0f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_SCENE_THRESHOLD_LOW = 1.0f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_SCENE_THRESHOLD_HIGH = 3.0f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_PERCENTILE = 99.995f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_BLACK_CUTOFF = 1.0f;
static const int   FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_GAMUT_MAPPING = 1;
static const int   FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_TONEMAPPING_FUNCTION = 3;
static const int   FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_TONE_CONSTANTS = 1;
static const int   FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_METADATA = 0;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_CONTRAST_RECOVERY = 0.3f;
static const float FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_CONTRAST_SMOOTHNESS = 3.5f;
static const bool  FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_INVERSE_TONE_MAPPING = false;
static const bool  FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_VISUALIZE_LUT = false;
static const bool  FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_SHOW_CLIPPING = false;
static const int   FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_USE_DOVI = -1;
static const int   FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_LUT_TYPE = 3;
static const int   FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_DST_PL_TRANSFER = 0;
static const int   FILTER_DEFAULT_LIBPLACEBO_TONEMAPPING_DST_PL_COLORPRIM = 0;

static const double FILTER_DEFAULT_COLORSPACE_LDRNITS = 100.0;
static const double FILTER_DEFAULT_COLORSPACE_NOMINAL_SOURCE_PEAK = 100.0;
static const double FILTER_DEFAULT_COLORSPACE_HDR_SOURCE_PEAK = 1000.0;

static const double FILTER_DEFAULT_HDR2SDR_DESAT_BASE = 0.18;
static const double FILTER_DEFAULT_HDR2SDR_DESAT_STRENGTH = 0.75;
static const double FILTER_DEFAULT_HDR2SDR_DESAT_EXP = 1.5;

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

static const int   FILTER_DEFAULT_DELOGO_DEPTH = 128;

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
static const uint8_t FILTER_DEFAULT_AFS_TUNE = 0;
static const bool  FILTER_DEFAULT_AFS_RFF = true;
static const int   FILTER_DEFAULT_AFS_TIMECODE = 0;
static const bool  FILTER_DEFAULT_AFS_LOG = false;

static const bool  FILTER_DEFAULT_DECOMB_FULL = true;
static const int   FILTER_DEFAULT_DECOMB_THRESHOLD = 20;
static const int   FILTER_DEFAULT_DECOMB_DTHRESHOLD = 7;
static const bool  FILTER_DEFAULT_DECOMB_BLEND = false;

static const int   FILTER_DEFAULT_DECIMATE_CYCLE = 5;
static const int   FILTER_DEFAULT_DECIMATE_DROP = 1;
static const float FILTER_DEFAULT_DECIMATE_THRE_DUP = 1.1f;
static const float FILTER_DEFAULT_DECIMATE_THRE_SC = 15.0f;
static const int   FILTER_DEFAULT_DECIMATE_BLOCK_X = 32;
static const int   FILTER_DEFAULT_DECIMATE_BLOCK_Y = 32;
static const bool  FILTER_DEFAULT_DECIMATE_PREPROCESSED = false;
static const bool  FILTER_DEFAULT_DECIMATE_CHROMA = true;
static const bool  FILTER_DEFAULT_DECIMATE_LOG = false;

static const int   FILTER_DEFAULT_MPDECIMATE_HI = 768;
static const int   FILTER_DEFAULT_MPDECIMATE_LO = 320;
static const bool  FILTER_DEFAULT_MPDECIMATE_MAX = 0;
static const float FILTER_DEFAULT_MPDECIMATE_FRAC = 0.33f;
static const bool  FILTER_DEFAULT_MPDECIMATE_LOG = false;

static const int   FILTER_DEFAULT_CONVOLUTION3D_THRESH_Y_SPATIAL  = 3;
static const int   FILTER_DEFAULT_CONVOLUTION3D_THRESH_C_SPATIAL  = 4;
static const int   FILTER_DEFAULT_CONVOLUTION3D_THRESH_Y_TEMPORAL = 3;
static const int   FILTER_DEFAULT_CONVOLUTION3D_THRESH_C_TEMPORAL = 4;

static const int   FILTER_DEFAULT_KNN_RADIUS = 3;
static const float FILTER_DEFAULT_KNN_STRENGTH = 0.08f;
static const float FILTER_DEFAULT_KNN_LERPC = 0.20f;
static const float FILTER_DEFAULT_KNN_WEIGHT_THRESHOLD = 0.01f;
static const float FILTER_DEFAULT_KNN_LERPC_THRESHOLD = 0.80f;

static const float FILTER_DEFAULT_NLMEANS_FILTER_SIGMA = 0.005f;
static const int   FILTER_DEFAULT_NLMEANS_PATCH_SIZE = 5;
static const int   FILTER_DEFAULT_NLMEANS_SEARCH_SIZE = 11;
static const float FILTER_DEFAULT_NLMEANS_H = 0.05f;

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

static const float FILTER_DEFAULT_DENOISE_DCT_SIGMA = 4.0f;
static const int   FILTER_DEFAULT_DENOISE_DCT_STEP = 2;
static const int   FILTER_DEFAULT_DENOISE_DCT_BLOCK_SIZE = 8;

static const float FILTER_DEFAULT_DENOISE_FFT3D_SIGMA = 1.0f;
static const float FILTER_DEFAULT_DENOISE_FFT3D_AMOUNT = 1.0f;
static const int   FILTER_DEFAULT_DENOISE_FFT3D_BLOCK_SIZE = 32;
static const float FILTER_DEFAULT_DENOISE_FFT3D_OVERLAP  = 0.5;
static const float FILTER_DEFAULT_DENOISE_FFT3D_OVERLAP2 = 0.0;
static const int   FILTER_DEFAULT_DENOISE_FFT3D_METHOD = 0;
static const int   FILTER_DEFAULT_DENOISE_FFT3D_TEMPORAL = 1;

static const float FILTER_DEFAULT_TWEAK_BRIGHTNESS = 0.0f;
static const float FILTER_DEFAULT_TWEAK_CONTRAST = 1.0f;
static const float FILTER_DEFAULT_TWEAK_GAMMA = 1.0f;
static const float FILTER_DEFAULT_TWEAK_SATURATION = 1.0f;
static const float FILTER_DEFAULT_TWEAK_HUE = 0.0f;

static const float FILTER_DEFAULT_EDGELEVEL_STRENGTH = 5.0f;
static const float FILTER_DEFAULT_EDGELEVEL_THRESHOLD = 20.0f;
static const float FILTER_DEFAULT_EDGELEVEL_BLACK = 0.0f;
static const float FILTER_DEFAULT_EDGELEVEL_WHITE = 0.0f;

static const TCHAR *FILTER_DEFAULT_LIBPLACEBO_SHADER_RESAMPLER_NAME = _T("libplacebo-ewa-lanczos");
static const int   FILTER_DEFAULT_LIBPLACEBO_SHADER_COLORSYSTEM = 0;
static const int   FILTER_DEFAULT_LIBPLACEBO_SHADER_TRANSFER = 0;
static const int   FILTER_DEFAULT_LIBPLACEBO_SHADER_CHROMALOC = 0;
static const float FILTER_DEFAULT_LIBPLACEBO_SHADER_RADIUS = FILTER_DEFAULT_LIBPLACEBO_RESAMPLE_RADIUS;
static const float FILTER_DEFAULT_LIBPLACEBO_SHADER_CLAMP = FILTER_DEFAULT_LIBPLACEBO_RESAMPLE_CLAMP;
static const float FILTER_DEFAULT_LIBPLACEBO_SHADER_TAPER = FILTER_DEFAULT_LIBPLACEBO_RESAMPLE_TAPER;
static const float FILTER_DEFAULT_LIBPLACEBO_SHADER_BLUR = FILTER_DEFAULT_LIBPLACEBO_RESAMPLE_BLUR;
static const float FILTER_DEFAULT_LIBPLACEBO_SHADER_ANTIRING = FILTER_DEFAULT_LIBPLACEBO_RESAMPLE_ANTIRING;
static const bool  FILTER_DEFAULT_LIBPLACEBO_SHADER_LINEAR = false;

static const int   FILTER_DEFAULT_UNSHARP_RADIUS = 3;
static const float FILTER_DEFAULT_UNSHARP_WEIGHT = 0.5f;
static const float FILTER_DEFAULT_UNSHARP_THRESHOLD = 10.0f;

static const float FILTER_DEFAULT_WARPSHARP_THRESHOLD = 128.0f;
static const int   FILTER_DEFAULT_WARPSHARP_BLUR = 2;
static const int   FILTER_DEFAULT_WARPSHARP_TYPE = 0;
static const float FILTER_DEFAULT_WARPSHARP_DEPTH = 16.0f;
static const int   FILTER_DEFAULT_WARPSHARP_CHROMA = 0;

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

struct RGYQPSet {
    bool enable;
    int qpI, qpP, qpB;

    RGYQPSet();
    RGYQPSet(int i, int p, int b);
    RGYQPSet(int i, int p, int b, bool enable);
    int qp(int i) const;
    int& qp(int i);
    bool operator==(const RGYQPSet &x) const;
    bool operator!=(const RGYQPSet &x) const;
    int parse(const TCHAR *str);
    void applyQPMinMax(const int min, const int max);
};

enum class RGYHEVCBsf {
    INTERNAL,
    LIBAVCODEC
};

const CX_DESC list_hevc_bsf_mode[] = {
    { _T("internal"),   (int)RGYHEVCBsf::INTERNAL   },
    { _T("libavcodec"), (int)RGYHEVCBsf::LIBAVCODEC },
    { NULL, 0 }
};

const CX_DESC list_vpp_denoise[] = {
    { _T("none"),    0 },
#if ENCODER_QSV
    { _T("denoise"), 4 },
#endif
    { _T("knn"),     1 },
    { _T("nlmeans"), 9 },
    { _T("pmd"),     2 },
    { _T("denoise-dct"), 8 },
    { _T("smooth"),  3 },
    { _T("fft3d"), 10 },
    { _T("convolution3d"),  5 },
#if ENCODER_VCEENC
    { _T("preprocess"), 4 },
#endif
#if ENCODER_NVENC
    { _T("nvvfx-denoise"), 6 },
    { _T("nvvfx-artifact-reduction"), 7 },
#endif
    { NULL, 0 }
};

const CX_DESC list_vpp_detail_enahance[] = {
    { _T("none"),       0 },
#if ENCODER_QSV || ENCODER_VCEENC
    { _T("detail-enhance"), 4 },
#endif
    { _T("unsharp"),    1 },
    { _T("edgelevel"),  2 },
    { _T("warpsharp"),  3 },
    { NULL, 0 }
};

const CX_DESC list_vpp_deband_names[] = {
    { _T("none"),       0 },
    { _T("deband"),    1 },
    { _T("libplacebo-deband"), 2 },
    { NULL, 0 }
};

enum HDR2SDRToneMap {
    HDR2SDR_DISABLED,
    HDR2SDR_HABLE,
    HDR2SDR_MOBIUS,
    HDR2SDR_REINHARD,
    HDR2SDR_BT2390,
};

const CX_DESC list_vpp_hdr2sdr[] = {
    { _T("none"),     HDR2SDR_DISABLED },
    { _T("hable"),    HDR2SDR_HABLE },
    { _T("mobius"),   HDR2SDR_MOBIUS },
    { _T("reinhard"), HDR2SDR_REINHARD },
    { _T("bt2390"),   HDR2SDR_BT2390 },
    { NULL, 0 }
};

enum RGY_VPP_RESIZE_MODE {
    RGY_VPP_RESIZE_MODE_DEFAULT,
#if ENCODER_QSV
    RGY_VPP_RESIZE_MODE_MFX_LOWPOWER,
    RGY_VPP_RESIZE_MODE_MFX_QUALITY,
#endif
    RGY_VPP_RESIZE_MODE_UNKNOWN,
};

enum RGY_VPP_RESIZE_ALGO {
    RGY_VPP_RESIZE_AUTO,
    RGY_VPP_RESIZE_BILINEAR,
    RGY_VPP_RESIZE_BICUBIC,
#if ENCODER_NVENC || CUFILTERS
    RGY_VPP_RESIZE_NEAREST,
#endif
    RGY_VPP_RESIZE_SPLINE16,
    RGY_VPP_RESIZE_SPLINE36,
    RGY_VPP_RESIZE_SPLINE64,
    RGY_VPP_RESIZE_LANCZOS2,
    RGY_VPP_RESIZE_LANCZOS3,
    RGY_VPP_RESIZE_LANCZOS4,
    RGY_VPP_RESIZE_OPENCL_CUDA_MAX,
#if ENCODER_QSV
    RGY_VPP_RESIZE_MFX_NEAREST_NEIGHBOR,
    RGY_VPP_RESIZE_MFX_BILINEAR,
    RGY_VPP_RESIZE_MFX_ADVANCED,
    RGY_VPP_RESIZE_MFX_AI_SUPRERES,
    RGY_VPP_RESIZE_MFX_MAX,
#endif
#if (ENCODER_NVENC && (!defined(_M_IX86) || FOR_AUO)) || CUFILTERS || CLFILTERS_AUF
    RGY_VPP_RESIZE_NPPI_INTER_NN,        /**<  Nearest neighbor filtering. */
    RGY_VPP_RESIZE_NPPI_INTER_LINEAR,        /**<  Linear interpolation. */
    RGY_VPP_RESIZE_NPPI_INTER_CUBIC,        /**<  Cubic interpolation. */
    RGY_VPP_RESIZE_NPPI_INTER_CUBIC2P_BSPLINE,              /**<  Two-parameter cubic filter (B=1, C=0) */
    RGY_VPP_RESIZE_NPPI_INTER_CUBIC2P_CATMULLROM,           /**<  Two-parameter cubic filter (B=0, C=1/2) */
    RGY_VPP_RESIZE_NPPI_INTER_CUBIC2P_B05C03,               /**<  Two-parameter cubic filter (B=1/2, C=3/10) */
    RGY_VPP_RESIZE_NPPI_INTER_SUPER,        /**<  Super sampling. */
    RGY_VPP_RESIZE_NPPI_INTER_LANCZOS,       /**<  Lanczos filtering. */
    RGY_VPP_RESIZE_NPPI_INTER_LANCZOS3_ADVANCED,       /**<  Generic Lanczos filtering with order 3. */
    RGY_VPP_RESIZE_NPPI_SMOOTH_EDGE, /**<  Smooth edge filtering. */
    RGY_VPP_RESIZE_NPPI_MAX,
#endif
#if (ENCODER_NVENC && (!defined(_M_IX86) || FOR_AUO)) || CUFILTERS || CLFILTERS_AUF
    RGY_VPP_RESIZE_NVVFX_SUPER_RES,
    RGY_VPP_RESIZE_NVVFX_MAX,
#endif
#if (ENCODER_NVENC && (!defined(_M_IX86) || FOR_AUO)) || CUFILTERS || CLFILTERS_AUF
    RGY_VPP_RESIZE_NGX_VSR,
    RGY_VPP_RESIZE_NGX_MAX,
#endif
#if ((ENCODER_NVENC || ENCODER_QSV || ENCODER_VCEENC) && (ENABLE_VPP_FILTER_LIBPLACEBO || FOR_AUO)) || CUFILTERS || CLFILTERS_AUF
    RGY_VPP_RESIZE_LIBPLACEBO_SPLINE16,
    RGY_VPP_RESIZE_LIBPLACEBO_SPLINE36,
    RGY_VPP_RESIZE_LIBPLACEBO_SPLINE64,
    RGY_VPP_RESIZE_LIBPLACEBO_NEAREST,
    RGY_VPP_RESIZE_LIBPLACEBO_BILINEAR,
    RGY_VPP_RESIZE_LIBPLACEBO_GAUSSIAN,
    RGY_VPP_RESIZE_LIBPLACEBO_SINC,
    RGY_VPP_RESIZE_LIBPLACEBO_LANCZOS,
    RGY_VPP_RESIZE_LIBPLACEBO_GINSENG,
    RGY_VPP_RESIZE_LIBPLACEBO_EWA_JINC,
    RGY_VPP_RESIZE_LIBPLACEBO_EWA_LANCZOS,
    RGY_VPP_RESIZE_LIBPLACEBO_EWA_LANCZOSSHARP,
    RGY_VPP_RESIZE_LIBPLACEBO_EWA_LANCZOS4SHARPEST,
    RGY_VPP_RESIZE_LIBPLACEBO_EWA_GINSENG,
    RGY_VPP_RESIZE_LIBPLACEBO_EWA_HANN,
    RGY_VPP_RESIZE_LIBPLACEBO_EWA_HANNING,
    RGY_VPP_RESIZE_LIBPLACEBO_BICUBIC,
    RGY_VPP_RESIZE_LIBPLACEBO_TRIANGLE,
    RGY_VPP_RESIZE_LIBPLACEBO_HERMITE,
    RGY_VPP_RESIZE_LIBPLACEBO_CATMULL_ROM,
    RGY_VPP_RESIZE_LIBPLACEBO_MITCHELL,
    RGY_VPP_RESIZE_LIBPLACEBO_MITCHELL_CLAMP,
    RGY_VPP_RESIZE_LIBPLACEBO_ROBIDOUX,
    RGY_VPP_RESIZE_LIBPLACEBO_ROBIDOUXSHARP,
    RGY_VPP_RESIZE_LIBPLACEBO_EWA_ROBIDOUX,
    RGY_VPP_RESIZE_LIBPLACEBO_EWA_ROBIDOUXSHARP,
    RGY_VPP_RESIZE_LIBPLACEBO_MAX,
#endif
#if ENCODER_VCEENC
    RGY_VPP_RESIZE_AMF_BILINEAR,
    RGY_VPP_RESIZE_AMF_BICUBIC,
    RGY_VPP_RESIZE_AMF_FSR_10,
    RGY_VPP_RESIZE_AMF_FSR_11,
    RGY_VPP_RESIZE_AMF_POINT,
    RGY_VPP_RESIZE_AMF_MAX,
#endif
#if ENCODER_MPP
    RGY_VPP_RESIZE_RGA_NEAREST,
    RGY_VPP_RESIZE_RGA_BILINEAR,
    RGY_VPP_RESIZE_RGA_BICUBIC,
    RGY_VPP_RESIZE_RGA_MAX,
#endif
    RGY_VPP_RESIZE_UNKNOWN,
};

enum RGY_VPP_RESIZE_TYPE {
    RGY_VPP_RESIZE_TYPE_NONE,
    RGY_VPP_RESIZE_TYPE_AUTO,
    RGY_VPP_RESIZE_TYPE_OPENCL,
#if ENCODER_QSV
    RGY_VPP_RESIZE_TYPE_MFX,
#endif
#if ENCODER_NVENC && (!defined(_M_IX86) || FOR_AUO) || CUFILTERS || CLFILTERS_AUF
    RGY_VPP_RESIZE_TYPE_NPPI,
#endif
#if ENCODER_NVENC && (!defined(_M_IX86) || FOR_AUO) || CUFILTERS || CLFILTERS_AUF
    RGY_VPP_RESIZE_TYPE_NVVFX,
    RGY_VPP_RESIZE_TYPE_NGX,
#endif
#if ((ENCODER_NVENC || ENCODER_QSV || ENCODER_VCEENC) && (ENABLE_VPP_FILTER_LIBPLACEBO || FOR_AUO)) || CUFILTERS || CLFILTERS_AUF
    RGY_VPP_RESIZE_TYPE_LIBPLACEBO,
#endif
#if ENCODER_VCEENC
    RGY_VPP_RESIZE_TYPE_AMF,
#endif
#if ENCODER_MPP
    RGY_VPP_RESIZE_TYPE_RGA,
#endif
    RGY_VPP_RESIZE_TYPE_UNKNOWN,
};

RGY_VPP_RESIZE_TYPE getVppResizeType(RGY_VPP_RESIZE_ALGO resize);

static bool isQSVMFXResizeFiter(const RGY_VPP_RESIZE_ALGO interp) {
#if ENCODER_QSV && (!defined(_M_IX86) || FOR_AUO) || CUFILTERS || CLFILTERS_AUF
    return getVppResizeType(interp) == RGY_VPP_RESIZE_TYPE_MFX;
#else
    UNREFERENCED_PARAMETER(interp);
    return false;
#endif
}

static bool isNppResizeFiter(const RGY_VPP_RESIZE_ALGO interp) {
#if ENCODER_NVENC && (!defined(_M_IX86) || FOR_AUO) || CUFILTERS || CLFILTERS_AUF
    return getVppResizeType(interp) == RGY_VPP_RESIZE_TYPE_NPPI;
#else
    UNREFERENCED_PARAMETER(interp);
    return false;
#endif
}

static bool isNvvfxResizeFiter(const RGY_VPP_RESIZE_ALGO interp) {
#if ENCODER_NVENC && (!defined(_M_IX86) || FOR_AUO) || CUFILTERS || CLFILTERS_AUF
    return getVppResizeType(interp) == RGY_VPP_RESIZE_TYPE_NVVFX;
#else
    UNREFERENCED_PARAMETER(interp);
    return false;
#endif
}

static bool isNgxResizeFiter(const RGY_VPP_RESIZE_ALGO interp) {
#if ENCODER_NVENC && (!defined(_M_IX86) || FOR_AUO) || CUFILTERS || CLFILTERS_AUF
    return getVppResizeType(interp) == RGY_VPP_RESIZE_TYPE_NGX;
#else
    UNREFERENCED_PARAMETER(interp);
    return false;
#endif
}

static bool isLibplaceboResizeFiter(const RGY_VPP_RESIZE_ALGO interp) {
#if ((ENCODER_NVENC || ENCODER_QSV || ENCODER_VCEENC) && (ENABLE_VPP_FILTER_LIBPLACEBO || FOR_AUO)) || CUFILTERS || CLFILTERS_AUF
    return getVppResizeType(interp) == RGY_VPP_RESIZE_TYPE_LIBPLACEBO;
#else
    UNREFERENCED_PARAMETER(interp);
    return false;
#endif
}

const CX_DESC list_vpp_resize_mode[] = {
    { _T("auto"),     RGY_VPP_RESIZE_MODE_DEFAULT },
#if ENCODER_QSV
    { _T("lowpower"), RGY_VPP_RESIZE_MODE_MFX_LOWPOWER },
    { _T("quality"),  RGY_VPP_RESIZE_MODE_MFX_QUALITY },
#endif
    { NULL, 0 }
};

const CX_DESC list_vpp_resize[] = {
    { _T("auto"),     RGY_VPP_RESIZE_AUTO },
    { _T("bilinear"), RGY_VPP_RESIZE_BILINEAR },
    { _T("bicubic"),  RGY_VPP_RESIZE_BICUBIC },
#if ENCODER_NVENC
    { _T("nearest"),  RGY_VPP_RESIZE_NEAREST },
#endif
    { _T("spline16"), RGY_VPP_RESIZE_SPLINE16 },
    { _T("spline36"), RGY_VPP_RESIZE_SPLINE36 },
    { _T("spline64"), RGY_VPP_RESIZE_SPLINE64 },
    { _T("lanczos2"), RGY_VPP_RESIZE_LANCZOS2 },
    { _T("lanczos3"), RGY_VPP_RESIZE_LANCZOS3 },
    { _T("lanczos4"), RGY_VPP_RESIZE_LANCZOS4 },
#if ENCODER_QSV
  #if !FOR_AUO
    { _T("bilinear"), RGY_VPP_RESIZE_MFX_BILINEAR },
  #endif
    { _T("advanced"), RGY_VPP_RESIZE_MFX_ADVANCED },
    { _T("mfx_ai_superres"), RGY_VPP_RESIZE_MFX_AI_SUPRERES },
    { _T("simple"),   RGY_VPP_RESIZE_MFX_NEAREST_NEIGHBOR },
  #if !FOR_AUO
    { _T("fine"),     RGY_VPP_RESIZE_MFX_ADVANCED },
  #endif
#endif
#if ENCODER_NVENC && (!defined(_M_IX86) || FOR_AUO) || CUFILTERS || CLFILTERS_AUF
    { _T("nn"),            RGY_VPP_RESIZE_NPPI_INTER_NN },
    { _T("npp_linear"),    RGY_VPP_RESIZE_NPPI_INTER_LINEAR },
    { _T("cubic"),         RGY_VPP_RESIZE_NPPI_INTER_CUBIC },
    //下記値は無効(指定しても動作しない)
    //{ _T("cubic_bspline"), RGY_VPP_RESIZE_NPPI_INTER_CUBIC2P_BSPLINE },
    //{ _T("cubic_catmull"), RGY_VPP_RESIZE_NPPI_INTER_CUBIC2P_CATMULLROM },
    //{ _T("cubic_b05c03"),  RGY_VPP_RESIZE_NPPI_INTER_CUBIC2P_B05C03 },
    { _T("super"),         RGY_VPP_RESIZE_NPPI_INTER_SUPER },
    { _T("lanczos"),       RGY_VPP_RESIZE_NPPI_INTER_LANCZOS },
    //{ _T("smooth_edge"),   RGY_VPP_RESIZE_NPPI_SMOOTH_EDGE },
#endif
#if ENCODER_NVENC && (!defined(_M_IX86) || FOR_AUO) || CUFILTERS || CLFILTERS_AUF
    { _T("nvvfx-superres"),  RGY_VPP_RESIZE_NVVFX_SUPER_RES },
#endif
#if ENCODER_NVENC && (!defined(_M_IX86) || FOR_AUO) || CUFILTERS || CLFILTERS_AUF
    { _T("ngx-vsr"),      RGY_VPP_RESIZE_NGX_VSR },
#endif
#if ((ENCODER_NVENC || ENCODER_QSV || ENCODER_VCEENC) && (ENABLE_VPP_FILTER_LIBPLACEBO || FOR_AUO)) || CUFILTERS || CLFILTERS_AUF
    { _T("libplacebo-spline16"), RGY_VPP_RESIZE_LIBPLACEBO_SPLINE16 },
    { _T("libplacebo-spline36"), RGY_VPP_RESIZE_LIBPLACEBO_SPLINE36 },
    { _T("libplacebo-spline64"), RGY_VPP_RESIZE_LIBPLACEBO_SPLINE64 },
    { _T("libplacebo-nearest"),  RGY_VPP_RESIZE_LIBPLACEBO_NEAREST },
    { _T("libplacebo-bilinear"), RGY_VPP_RESIZE_LIBPLACEBO_BILINEAR },
    { _T("libplacebo-gaussian"), RGY_VPP_RESIZE_LIBPLACEBO_GAUSSIAN },
    { _T("libplacebo-sinc"),     RGY_VPP_RESIZE_LIBPLACEBO_SINC },
    { _T("libplacebo-lanczos"),  RGY_VPP_RESIZE_LIBPLACEBO_LANCZOS },
    { _T("libplacebo-ginseng"),  RGY_VPP_RESIZE_LIBPLACEBO_GINSENG },
    { _T("libplacebo-ewa-jinc"), RGY_VPP_RESIZE_LIBPLACEBO_EWA_JINC },
    { _T("libplacebo-ewa-lanczos"),          RGY_VPP_RESIZE_LIBPLACEBO_EWA_LANCZOS },
    { _T("libplacebo-ewa-lanczossharp"),     RGY_VPP_RESIZE_LIBPLACEBO_EWA_LANCZOSSHARP },
    { _T("libplacebo-ewa-lanczos4sharpest"), RGY_VPP_RESIZE_LIBPLACEBO_EWA_LANCZOS4SHARPEST },
    { _T("libplacebo-ewa-ginseng"),          RGY_VPP_RESIZE_LIBPLACEBO_EWA_GINSENG },
    { _T("libplacebo-ewa-hann"),             RGY_VPP_RESIZE_LIBPLACEBO_EWA_HANN },
    { _T("libplacebo-ewa-hanning"),          RGY_VPP_RESIZE_LIBPLACEBO_EWA_HANNING },
    { _T("libplacebo-bicubic"),              RGY_VPP_RESIZE_LIBPLACEBO_BICUBIC },
    { _T("libplacebo-triangle"),             RGY_VPP_RESIZE_LIBPLACEBO_TRIANGLE },
    { _T("libplacebo-hermite"),              RGY_VPP_RESIZE_LIBPLACEBO_HERMITE },
    { _T("libplacebo-catmull-rom"),          RGY_VPP_RESIZE_LIBPLACEBO_CATMULL_ROM },
    { _T("libplacebo-mitchell"),             RGY_VPP_RESIZE_LIBPLACEBO_MITCHELL },
    { _T("libplacebo-mitchell-clamp"),       RGY_VPP_RESIZE_LIBPLACEBO_MITCHELL_CLAMP },
    { _T("libplacebo-robidoux"),             RGY_VPP_RESIZE_LIBPLACEBO_ROBIDOUX },
    { _T("libplacebo-robidouxsharp"),        RGY_VPP_RESIZE_LIBPLACEBO_ROBIDOUXSHARP },
    { _T("libplacebo-ewa-robidoux"),         RGY_VPP_RESIZE_LIBPLACEBO_EWA_ROBIDOUX },
    { _T("libplacebo-ewa-robidouxsharp"),    RGY_VPP_RESIZE_LIBPLACEBO_EWA_ROBIDOUXSHARP },
#endif
#if ENCODER_VCEENC
    { _T("amf_bilinear"), RGY_VPP_RESIZE_AMF_BILINEAR },
    { _T("amf_bicubic"),  RGY_VPP_RESIZE_AMF_BICUBIC },
    { _T("amf_fsr"),      RGY_VPP_RESIZE_AMF_FSR_10 },
    { _T("amf_fsr_11"),   RGY_VPP_RESIZE_AMF_FSR_11 },
    { _T("amf_point"),    RGY_VPP_RESIZE_AMF_POINT },
    { _T("amf_max"),      RGY_VPP_RESIZE_AMF_MAX },
#endif
#if ENCODER_MPP
    { _T("rga_nearest"),  RGY_VPP_RESIZE_RGA_NEAREST },
    { _T("rga_bilinear"), RGY_VPP_RESIZE_RGA_BILINEAR },
    { _T("rga_bicubic"),  RGY_VPP_RESIZE_RGA_BICUBIC },
#endif
    { NULL, 0 }
};

const CX_DESC list_vpp_resize_help[] = {
    { _T("auto"),     RGY_VPP_RESIZE_AUTO },
    { _T("bilinear"), RGY_VPP_RESIZE_BILINEAR },
    { _T("bicubic"),  RGY_VPP_RESIZE_BICUBIC },
#if ENCODER_NVENC
    { _T("nearest"),  RGY_VPP_RESIZE_NEAREST },
#endif
    { _T("spline16"), RGY_VPP_RESIZE_SPLINE16 },
    { _T("spline36"), RGY_VPP_RESIZE_SPLINE36 },
    { _T("spline64"), RGY_VPP_RESIZE_SPLINE64 },
    { _T("lanczos2"), RGY_VPP_RESIZE_LANCZOS2 },
    { _T("lanczos3"), RGY_VPP_RESIZE_LANCZOS3 },
    { _T("lanczos4"), RGY_VPP_RESIZE_LANCZOS4 },
#if ENCODER_QSV
    { _T("bilinear"), RGY_VPP_RESIZE_MFX_BILINEAR },
    { _T("advanced"), RGY_VPP_RESIZE_MFX_ADVANCED },
    { _T("mfx_ai_superres"), RGY_VPP_RESIZE_MFX_AI_SUPRERES },
    { _T("simple"),   RGY_VPP_RESIZE_MFX_NEAREST_NEIGHBOR },
    { _T("fine"),     RGY_VPP_RESIZE_MFX_ADVANCED },
#endif
#if ENCODER_NVENC && (!defined(_M_IX86) || FOR_AUO) || CUFILTERS || CLFILTERS_AUF
    { _T("nn"),            RGY_VPP_RESIZE_NPPI_INTER_NN },
    { _T("npp_linear"),    RGY_VPP_RESIZE_NPPI_INTER_LINEAR },
    { _T("cubic"),         RGY_VPP_RESIZE_NPPI_INTER_CUBIC },
    //下記値は無効(指定しても動作しない)
    //{ _T("cubic_bspline"), RGY_VPP_RESIZE_NPPI_INTER_CUBIC2P_BSPLINE },
    //{ _T("cubic_catmull"), RGY_VPP_RESIZE_NPPI_INTER_CUBIC2P_CATMULLROM },
    //{ _T("cubic_b05c03"),  RGY_VPP_RESIZE_NPPI_INTER_CUBIC2P_B05C03 },
    { _T("super"),         RGY_VPP_RESIZE_NPPI_INTER_SUPER },
    { _T("lanczos"),       RGY_VPP_RESIZE_NPPI_INTER_LANCZOS },
    //{ _T("smooth_edge"),   RGY_VPP_RESIZE_NPPI_SMOOTH_EDGE },
#endif
#if ENCODER_NVENC && (!defined(_M_IX86) || FOR_AUO) || CUFILTERS || CLFILTERS_AUF
    { _T("nvvfx-superres"),  RGY_VPP_RESIZE_NVVFX_SUPER_RES },
#endif
#if ENCODER_NVENC && (!defined(_M_IX86) || FOR_AUO) || CUFILTERS || CLFILTERS_AUF
    { _T("ngx-vsr"),      RGY_VPP_RESIZE_NGX_VSR },
#endif
#if ((ENCODER_NVENC || ENCODER_QSV || ENCODER_VCEENC) && (ENABLE_VPP_FILTER_LIBPLACEBO || FOR_AUO)) || CUFILTERS || CLFILTERS_AUF
    { _T("libplacebo-spline16"), RGY_VPP_RESIZE_LIBPLACEBO_SPLINE16 },
    { _T("libplacebo-spline36"), RGY_VPP_RESIZE_LIBPLACEBO_SPLINE36 },
    { _T("libplacebo-spline64"), RGY_VPP_RESIZE_LIBPLACEBO_SPLINE64 },
    { _T("libplacebo-nearest"),  RGY_VPP_RESIZE_LIBPLACEBO_NEAREST },
    { _T("libplacebo-bilinear"), RGY_VPP_RESIZE_LIBPLACEBO_BILINEAR },
    { _T("libplacebo-gaussian"), RGY_VPP_RESIZE_LIBPLACEBO_GAUSSIAN },
    { _T("libplacebo-sinc"),     RGY_VPP_RESIZE_LIBPLACEBO_SINC },
    { _T("libplacebo-lanczos"),  RGY_VPP_RESIZE_LIBPLACEBO_LANCZOS },
    { _T("libplacebo-ginseng"),  RGY_VPP_RESIZE_LIBPLACEBO_GINSENG },
    { _T("libplacebo-ewa-jinc"), RGY_VPP_RESIZE_LIBPLACEBO_EWA_JINC },
    { _T("libplacebo-ewa-lanczos"),          RGY_VPP_RESIZE_LIBPLACEBO_EWA_LANCZOS },
    { _T("libplacebo-ewa-lanczossharp"),     RGY_VPP_RESIZE_LIBPLACEBO_EWA_LANCZOSSHARP },
    { _T("libplacebo-ewa-lanczos4sharpest"), RGY_VPP_RESIZE_LIBPLACEBO_EWA_LANCZOS4SHARPEST },
    { _T("libplacebo-ewa-ginseng"),          RGY_VPP_RESIZE_LIBPLACEBO_EWA_GINSENG },
    { _T("libplacebo-ewa-hann"),             RGY_VPP_RESIZE_LIBPLACEBO_EWA_HANN },
    { _T("libplacebo-ewa-hanning"),          RGY_VPP_RESIZE_LIBPLACEBO_EWA_HANNING },
    { _T("libplacebo-bicubic"),              RGY_VPP_RESIZE_LIBPLACEBO_BICUBIC },
    { _T("libplacebo-triangle"),             RGY_VPP_RESIZE_LIBPLACEBO_TRIANGLE },
    { _T("libplacebo-hermite"),              RGY_VPP_RESIZE_LIBPLACEBO_HERMITE },
    { _T("libplacebo-catmull-rom"),          RGY_VPP_RESIZE_LIBPLACEBO_CATMULL_ROM },
    { _T("libplacebo-mitchell"),             RGY_VPP_RESIZE_LIBPLACEBO_MITCHELL },
    { _T("libplacebo-mitchell-clamp"),       RGY_VPP_RESIZE_LIBPLACEBO_MITCHELL_CLAMP },
    { _T("libplacebo-robidoux"),             RGY_VPP_RESIZE_LIBPLACEBO_ROBIDOUX },
    { _T("libplacebo-robidouxsharp"),        RGY_VPP_RESIZE_LIBPLACEBO_ROBIDOUXSHARP },
    { _T("libplacebo-ewa-robidoux"),         RGY_VPP_RESIZE_LIBPLACEBO_EWA_ROBIDOUX },
    { _T("libplacebo-ewa-robidouxsharp"),    RGY_VPP_RESIZE_LIBPLACEBO_EWA_ROBIDOUXSHARP },
#endif
#if ENCODER_VCEENC
    { _T("amf_bilinear"), RGY_VPP_RESIZE_AMF_BILINEAR },
    { _T("amf_bicubic"),  RGY_VPP_RESIZE_AMF_BICUBIC },
    { _T("amf_fsr"),      RGY_VPP_RESIZE_AMF_FSR_10 },
#if !DONOTSHOW_AMF_POINT_FSR11
    { _T("amf_fsr_11"),   RGY_VPP_RESIZE_AMF_FSR_11 },
    { _T("amf_point"),    RGY_VPP_RESIZE_AMF_POINT },
#endif
#endif
#if ENCODER_MPP
    { _T("rga_nearest"),  RGY_VPP_RESIZE_RGA_NEAREST },
    { _T("rga_bilinear"), RGY_VPP_RESIZE_RGA_BILINEAR },
    { _T("rga_bicubic"),  RGY_VPP_RESIZE_RGA_BICUBIC },
#endif
    { NULL, 0 }
};

static const char *paramsResizeLibPlacebo[] = { "algo", "pl-radius", "pl-clamp", "pl-taper", "pl-blur", "pl-antiring"/*, "pl-cplace"*/ };
static const char *paramsResizeNVEnc[] = { "superres-mode", "superres-strength", "vsr-quality" };
static const char *paramsResizeQSVEnc[] = { "superres-mode", "superres-algo" };

const CX_DESC list_vpp_resize_res_mode[] = {
    { _T("normal"),   (int)RGYResizeResMode::Normal },
    { _T("decrease"), (int)RGYResizeResMode::PreserveOrgAspectDec },
    { _T("increase"), (int)RGYResizeResMode::PreserveOrgAspectInc },
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

const CX_DESC list_vpp_denoise_dct_block_size[] = {
    { _T("8"), 8 },
    { _T("16"), 16 },
    { NULL, 0 }
};

const CX_DESC list_vpp_denoise_dct_step[] = {
    { _T("1"), 1 },
    { _T("2"), 2 },
    { _T("4"), 4 },
    { _T("8"), 8 },
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

const CX_DESC list_vpp_rotate[] = {
    { _T("90"),   90 },
    { _T("180"), 180 },
    { _T("270"), 270 },
    { NULL, 0 }
};

const CX_DESC list_vpp_mirroring[] = {
    { _T("n"), 0   },
    { _T("h"), 1 /*horizontal*/ },
    { _T("v"), 2 /*vertical*/   },
    { NULL, 0 }
};

const CX_DESC list_vpp_ass_shaping[] = {
    { _T("simple"),  0 },
    { _T("complex"), 1 },
    { NULL, 0 }
};

const CX_DESC list_vpp_raduis[] = {
    { _T("1 - weak"),  1 },
    { _T("2"),  2 },
    { _T("3"),  3 },
    { _T("4"),  4 },
    { _T("5 - strong"),  5 },
    { NULL, 0 }
};

const CX_DESC list_vpp_apply_count[] = {
    { _T("1"),  1 },
    { _T("2"),  2 },
    { _T("3"),  3 },
    { _T("4"),  4 },
    { _T("5"),  5 },
    { _T("6"),  6 },
    { NULL, 0 }
};

const CX_DESC list_vpp_smooth_quality[] = {
    { _T("1 - fast"),  1 },
    { _T("2"),  2 },
    { _T("3"),  3 },
    { _T("4"),  4 },
    { _T("5"),  5 },
    { _T("6 - high quality"),  6 },
    { NULL, 0 }
};

const CX_DESC list_vpp_deband_sample[] = {
    { _T("0"),  0 },
    { _T("1"),  1 },
    { _T("2"),  2 },
    { NULL, 0 }
};

const CX_DESC list_vpp_1_to_10[] = {
    { _T("1"),  1 },
    { _T("2"),  2 },
    { _T("3"),  3 },
    { _T("4"),  4 },
    { _T("5"),  5 },
    { _T("6"),  6 },
    { _T("7"),  7 },
    { _T("8"),  8 },
    { _T("9"),  9 },
    { _T("10"),  10 },
    { NULL, 0 }
};

struct VppLibplaceboResample {
    bool enable;
    float radius;
    float clamp_;
    float taper;
    float blur;
    float antiring;
    int cplace;

    VppLibplaceboResample();
    bool operator==(const VppLibplaceboResample &x) const;
    bool operator!=(const VppLibplaceboResample &x) const;
    tstring print() const;
};

enum class VppLibplaceboDebandDitherMode {
    None,
    BlueNoise,
    OrderedLUT,
    OrderedFixed,
    WhiteNoise,
};

const CX_DESC list_vpp_libplacebo_deband_dither_mode[] = {
    { _T("none"),          (int)VppLibplaceboDebandDitherMode::None },
    { _T("blue_noise"),    (int)VppLibplaceboDebandDitherMode::BlueNoise },
    { _T("ordered_lut"),   (int)VppLibplaceboDebandDitherMode::OrderedLUT },
    { _T("ordered_fixed"), (int)VppLibplaceboDebandDitherMode::OrderedFixed },
    { _T("white_noise"),   (int)VppLibplaceboDebandDitherMode::WhiteNoise },
    { NULL, 0 }
};

const CX_DESC list_vpp_libplacebo_deband_lut_size[] = {
    { _T("2"),   1 },
    { _T("4"),   2 },
    { _T("8"),   3 },
    { _T("16"),  4 },
    { _T("32"),  5 },
    { _T("64"),  6 },
    { _T("128"), 7 },
    { _T("256"), 8 },
    { NULL, 0 }
};

struct VppLibplaceboDeband {
    bool enable;
    int iterations;
    float threshold;
    float radius;
    float grainY;
    float grainC;
    VppLibplaceboDebandDitherMode dither;
    int lut_size;

    VppLibplaceboDeband();
    bool operator==(const VppLibplaceboDeband &x) const;
    bool operator!=(const VppLibplaceboDeband &x) const;
    tstring print() const;
};

enum class VppLibplaceboColorsystem {
    UNKNOWN,
    BT_601,
    BT_709,
    SMPTE_240M,
    BT_2020_NC,
    BT_2020_C,
    BT_2100_PQ,
    BT_2100_HLG,
    DOLBYVISION,
    YCGCO,
    RGB,
    XYZ,
    COUNT
};

const CX_DESC list_vpp_libplacebo_colorsystem[] = {
    { _T("auto"),        (int)VppLibplaceboColorsystem::UNKNOWN },
    { _T("unknown"),     (int)VppLibplaceboColorsystem::UNKNOWN },
    { _T("bt601"),       (int)VppLibplaceboColorsystem::BT_601 },
    { _T("bt709"),       (int)VppLibplaceboColorsystem::BT_709 },
    { _T("smpte240m"),   (int)VppLibplaceboColorsystem::SMPTE_240M },
    { _T("bt2020nc"),    (int)VppLibplaceboColorsystem::BT_2020_NC },
    { _T("bt2020c"),     (int)VppLibplaceboColorsystem::BT_2020_C },
    { _T("bt2100pq"),    (int)VppLibplaceboColorsystem::BT_2100_PQ },
    { _T("bt2100hlg"),   (int)VppLibplaceboColorsystem::BT_2100_HLG },
    { _T("dolbyvision"), (int)VppLibplaceboColorsystem::DOLBYVISION },
    { _T("ycgco"),       (int)VppLibplaceboColorsystem::YCGCO },
    { _T("rgb"),         (int)VppLibplaceboColorsystem::RGB },
    { _T("xyz"),         (int)VppLibplaceboColorsystem::XYZ },
    { NULL, 0 }
};

enum class VppLibplaceboToneMappingCSP {
    Auto = -1,
    SDR,
    HDR10,
    HLG,
    DOVI,
    RGB,
};

enum class VppLibplaceboToneMappingGamutMapping {
    clip,
    perceptual,
    softclip,
    relative,
    saturation,
    absolute,
    desaturate,
    darken,
    highlight,
    linear,
};

enum class VppLibplaceboToneMappingFunction {
    clip,
    st2094_40,
    st2094_10,
    bt2390,
    bt2446a,
    spline,
    reinhard,
    mobius,
    hable,
    gamma,
    linear,
    linearlight,
};

enum class VppLibplaceboToneMappingMetadata {
    ANY,
    NONE,
    HDR10,
    HDR10PLUS,
    CIE_Y,
};

const CX_DESC list_vpp_libplacebo_tone_mapping_csp[] = {
    { _T("auto"),  (int)VppLibplaceboToneMappingCSP::Auto },
    { _T("sdr"),   (int)VppLibplaceboToneMappingCSP::SDR },
    { _T("hdr10"), (int)VppLibplaceboToneMappingCSP::HDR10 },
    { _T("hlg"),   (int)VppLibplaceboToneMappingCSP::HLG },
    { _T("dovi"),  (int)VppLibplaceboToneMappingCSP::DOVI },
    { _T("rgb"),   (int)VppLibplaceboToneMappingCSP::RGB },
    { NULL, 0 }
};

const CX_DESC list_vpp_libplacebo_tone_mapping_gamut_mapping[] = {
    { _T("clip"),           (int)VppLibplaceboToneMappingGamutMapping::clip },
    { _T("perceptual"),     (int)VppLibplaceboToneMappingGamutMapping::perceptual },
    { _T("softclip"),       (int)VppLibplaceboToneMappingGamutMapping::softclip },
    { _T("relative"),       (int)VppLibplaceboToneMappingGamutMapping::relative },
    { _T("saturation"),     (int)VppLibplaceboToneMappingGamutMapping::saturation },
    { _T("absolute"),       (int)VppLibplaceboToneMappingGamutMapping::absolute },
    { _T("desaturate"),     (int)VppLibplaceboToneMappingGamutMapping::desaturate },
    { _T("darken"),         (int)VppLibplaceboToneMappingGamutMapping::darken },
    { _T("highlight"),      (int)VppLibplaceboToneMappingGamutMapping::highlight },
    { _T("linear"),         (int)VppLibplaceboToneMappingGamutMapping::linear },
    { NULL, 0 }
};

const CX_DESC list_vpp_libplacebo_tone_mapping_function[] = {
    { _T("clip"),        (int)VppLibplaceboToneMappingFunction::clip },
    { _T("st2094-40"),   (int)VppLibplaceboToneMappingFunction::st2094_40 },
    { _T("st2094-10"),   (int)VppLibplaceboToneMappingFunction::st2094_10 },
    { _T("bt2390"),      (int)VppLibplaceboToneMappingFunction::bt2390 },
    { _T("bt2446a"),     (int)VppLibplaceboToneMappingFunction::bt2446a },
    { _T("spline"),      (int)VppLibplaceboToneMappingFunction::spline },
    { _T("reinhard"),    (int)VppLibplaceboToneMappingFunction::reinhard },
    { _T("mobius"),      (int)VppLibplaceboToneMappingFunction::mobius },
    { _T("hable"),       (int)VppLibplaceboToneMappingFunction::hable },
    { _T("gamma"),       (int)VppLibplaceboToneMappingFunction::gamma },
    { _T("linear"),      (int)VppLibplaceboToneMappingFunction::linear },
    { _T("linearlight"), (int)VppLibplaceboToneMappingFunction::linearlight },
    { NULL, 0 }
};

const CX_DESC list_vpp_libplacebo_tone_mapping_metadata[] = {
    { _T("any"),       (int)VppLibplaceboToneMappingMetadata::ANY },
    { _T("none"),      (int)VppLibplaceboToneMappingMetadata::NONE },
    { _T("hdr10"),     (int)VppLibplaceboToneMappingMetadata::HDR10 },
    { _T("hdr10plus"), (int)VppLibplaceboToneMappingMetadata::HDR10PLUS },
    { _T("cie_y"),     (int)VppLibplaceboToneMappingMetadata::CIE_Y },
    { NULL, 0 }
};

enum class VppLibplaceboToneMappingLUTType {
    Native,
    Normalized,
    Conversion,
};

const CX_DESC list_vpp_libplacebo_tone_mapping_lut_type[] = {
    { _T("native"),      (int)VppLibplaceboToneMappingLUTType::Native },
    { _T("normalized"),  (int)VppLibplaceboToneMappingLUTType::Normalized },
    { _T("conversion"),  (int)VppLibplaceboToneMappingLUTType::Conversion },
    { NULL, 0 }
};

struct VppLibplaceboToneMappingConstantsST2094 {
    float knee_adaptation;
    float knee_min;
    float knee_max;
    float knee_default;

    VppLibplaceboToneMappingConstantsST2094();
    bool operator==(const VppLibplaceboToneMappingConstantsST2094 &x) const;
    bool operator!=(const VppLibplaceboToneMappingConstantsST2094 &x) const;
    tstring print() const;
};

struct VppLibplaceboToneMappingConstantsBT2390 {
    float knee_offset;

    VppLibplaceboToneMappingConstantsBT2390();
    bool operator==(const VppLibplaceboToneMappingConstantsBT2390 &x) const;
    bool operator!=(const VppLibplaceboToneMappingConstantsBT2390 &x) const;
    tstring print() const;
};

struct VppLibplaceboToneMappingConstantsSpline {
    float slope_tuning;
    float slope_offset;
    float spline_contrast;

    VppLibplaceboToneMappingConstantsSpline();
    bool operator==(const VppLibplaceboToneMappingConstantsSpline &x) const;
    bool operator!=(const VppLibplaceboToneMappingConstantsSpline &x) const;
    tstring print() const;
};

struct VppLibplaceboToneMappingConstantsReinhard {
    float contrast;

    VppLibplaceboToneMappingConstantsReinhard();
    bool operator==(const VppLibplaceboToneMappingConstantsReinhard &x) const;
    bool operator!=(const VppLibplaceboToneMappingConstantsReinhard &x) const;
    tstring print() const;
};

struct VppLibplaceboToneMappingConstantsMobius {
    float linear_knee;

    VppLibplaceboToneMappingConstantsMobius();
    bool operator==(const VppLibplaceboToneMappingConstantsMobius &x) const;
    bool operator!=(const VppLibplaceboToneMappingConstantsMobius &x) const;
    tstring print() const;
};

struct VppLibplaceboToneMappingConstantsLinear {
    float exposure;

    VppLibplaceboToneMappingConstantsLinear();
    bool operator==(const VppLibplaceboToneMappingConstantsLinear &x) const;
    bool operator!=(const VppLibplaceboToneMappingConstantsLinear &x) const;
    tstring print() const;
};

enum class VppLibplaceboToneMappingTransfer {
    Unknown,
    BT1886,
    sRGB,
    Linear,
    Gamma18,
    Gamma20,
    Gamma22,
    Gamma24,
    Gamma26,
    Gamma28,
    ProPhoto,
    ST428,
    PQ,
    HLG,
    VLog,
    SLog1,
    SLog2,  
};

const CX_DESC list_vpp_libplacebo_tone_mapping_transfer[] = {
    { _T("unknown"),    (int)VppLibplaceboToneMappingTransfer::Unknown },
    { _T("srgb"),       (int)VppLibplaceboToneMappingTransfer::sRGB },
    { _T("bt1886"),     (int)VppLibplaceboToneMappingTransfer::BT1886 },
    { _T("linear"),     (int)VppLibplaceboToneMappingTransfer::Linear },
    { _T("gamma18"),    (int)VppLibplaceboToneMappingTransfer::Gamma18 },
    { _T("gamma20"),    (int)VppLibplaceboToneMappingTransfer::Gamma20 },
    { _T("gamma22"),    (int)VppLibplaceboToneMappingTransfer::Gamma22 },
    { _T("gamma24"),    (int)VppLibplaceboToneMappingTransfer::Gamma24 },
    { _T("gamma26"),    (int)VppLibplaceboToneMappingTransfer::Gamma26 },
    { _T("gamma28"),    (int)VppLibplaceboToneMappingTransfer::Gamma28 },
    { _T("prophoto"),   (int)VppLibplaceboToneMappingTransfer::ProPhoto },
    { _T("st428"),      (int)VppLibplaceboToneMappingTransfer::ST428 },
    { _T("pq"),         (int)VppLibplaceboToneMappingTransfer::PQ },
    { _T("hlg"),        (int)VppLibplaceboToneMappingTransfer::HLG },
    { _T("vlog"),       (int)VppLibplaceboToneMappingTransfer::VLog },
    { _T("slog1"),      (int)VppLibplaceboToneMappingTransfer::SLog1 },
    { _T("slog2"),      (int)VppLibplaceboToneMappingTransfer::SLog2 },
    { NULL, 0 }
};

enum class VppLibplaceboToneMappingColorprim {
    Unknown,
    BT601_525,
    BT601_625,
    BT709,
    BT470M,
    EBU_3213,
    BT2020,
    Apple,
    Adobe,
    ProPhoto,
    CIE_1931,
    DCI_P3,
    Display_P3,
    V_Gamut,
    S_Gamut,
    Film_C,
    ACES_AP0,
    ACES_AP1,
};

const CX_DESC list_vpp_libplacebo_tone_mapping_colorprim[] = {
    { _T("unknown"),   (int)VppLibplaceboToneMappingColorprim::Unknown },
    { _T("bt601_525"), (int)VppLibplaceboToneMappingColorprim::BT601_525 },
    { _T("bt601_625"), (int)VppLibplaceboToneMappingColorprim::BT601_625 },
    { _T("bt709"),     (int)VppLibplaceboToneMappingColorprim::BT709 },
    { _T("bt470m"),    (int)VppLibplaceboToneMappingColorprim::BT470M },
    { _T("ebu_3213"),  (int)VppLibplaceboToneMappingColorprim::EBU_3213 },
    { _T("bt2020"),    (int)VppLibplaceboToneMappingColorprim::BT2020 },
    { _T("apple"),     (int)VppLibplaceboToneMappingColorprim::Apple },
    { _T("adobe"),     (int)VppLibplaceboToneMappingColorprim::Adobe },
    { _T("prophoto"),  (int)VppLibplaceboToneMappingColorprim::ProPhoto },
    { _T("cie_1931"),  (int)VppLibplaceboToneMappingColorprim::CIE_1931 },
    { _T("dci_p3"),    (int)VppLibplaceboToneMappingColorprim::DCI_P3 },
    { _T("display_p3"),(int)VppLibplaceboToneMappingColorprim::Display_P3 },
    { _T("v_gamut"),   (int)VppLibplaceboToneMappingColorprim::V_Gamut },
    { _T("s_gamut"),   (int)VppLibplaceboToneMappingColorprim::S_Gamut },
    { _T("film_c"),    (int)VppLibplaceboToneMappingColorprim::Film_C },
    { _T("aces_ap0"),  (int)VppLibplaceboToneMappingColorprim::ACES_AP0 },
    { _T("aces_ap1"),  (int)VppLibplaceboToneMappingColorprim::ACES_AP1 },
    { NULL, 0 }
};

struct VppLibplaceboToneMappingConstants {
    VppLibplaceboToneMappingConstantsST2094 st2094;
    VppLibplaceboToneMappingConstantsBT2390 bt2390;
    VppLibplaceboToneMappingConstantsSpline spline;
    VppLibplaceboToneMappingConstantsReinhard reinhard;
    VppLibplaceboToneMappingConstantsMobius mobius;
    VppLibplaceboToneMappingConstantsLinear linear;

    VppLibplaceboToneMappingConstants();
    bool operator==(const VppLibplaceboToneMappingConstants &x) const;
    bool operator!=(const VppLibplaceboToneMappingConstants &x) const;
    tstring print(const VppLibplaceboToneMappingFunction tonemapping_function) const;
};

struct VppLibplaceboToneMapping {
    bool enable;
    VppLibplaceboToneMappingCSP src_csp;
    VppLibplaceboToneMappingCSP dst_csp;
    float src_max;
    float src_min;
    float dst_max;
    float dst_min;
    bool dynamic_peak_detection;
    float smooth_period;
    float scene_threshold_low;
    float scene_threshold_high;
    float percentile;
    float black_cutoff;
    VppLibplaceboToneMappingGamutMapping gamut_mapping;
    VppLibplaceboToneMappingFunction tonemapping_function;
    VppLibplaceboToneMappingConstants tone_constants;
    VppLibplaceboToneMappingMetadata metadata;
    float contrast_recovery;
    float contrast_smoothness;
    bool inverse_tone_mapping;
    bool visualize_lut;
    bool show_clipping;
    int use_dovi;
    tstring lut_path;
    VppLibplaceboToneMappingLUTType lut_type;
    VppLibplaceboToneMappingTransfer dst_pl_transfer;
    VppLibplaceboToneMappingColorprim dst_pl_colorprim;

    VppLibplaceboToneMapping();
    bool operator==(const VppLibplaceboToneMapping &x) const;
    bool operator!=(const VppLibplaceboToneMapping &x) const;
    tstring print() const;
};

struct VppLibplaceboShader {
    bool enable;
    tstring shader;
    int width;
    int height;
    std::vector<std::pair<tstring, tstring>> params;
    RGY_VPP_RESIZE_ALGO resize_algo;
    VppLibplaceboColorsystem colorsystem;
    VppLibplaceboToneMappingTransfer transfer;
    CspChromaloc chromaloc;
    float radius;
    float clamp_;
    float taper;
    float blur;
    float antiring;
    bool linear;

    VppLibplaceboShader();
    bool operator==(const VppLibplaceboShader &x) const;
    bool operator!=(const VppLibplaceboShader &x) const;
    tstring print() const;
};

struct ColorspaceConv {
    VideoVUIInfo from, to;
    double sdr_source_peak;
    bool approx_gamma;
    bool scene_ref;

    ColorspaceConv();
    ColorspaceConv(const VideoVUIInfo& csp_from, const VideoVUIInfo &csp_to) : ColorspaceConv() {
        from = csp_from;
        to = csp_to;
    }
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
    double desat_base;
    double desat_strength;
    double desat_exp;

    HDR2SDRParams();
    bool operator==(const HDR2SDRParams &x) const;
    bool operator!=(const HDR2SDRParams &x) const;
};

enum class LUT3DInterp {
    Nearest,
    Trilinear,
    Pyramid,
    Prism,
    Tetrahedral,
};

static const auto FILTER_DEFAULT_LUT3D_INTERP = LUT3DInterp::Tetrahedral;

const CX_DESC list_vpp_colorspace_lut3d_interp[] = {
    { _T("nearest"),     (int)LUT3DInterp::Nearest     },
    { _T("trilinear"),   (int)LUT3DInterp::Trilinear   },
    { _T("tetrahedral"), (int)LUT3DInterp::Tetrahedral },
    { _T("pyramid"),     (int)LUT3DInterp::Pyramid     },
    { _T("prism"),       (int)LUT3DInterp::Prism       },
    { NULL, 0 }
};

struct LUT3DParams {
    LUT3DInterp interp;
    tstring table_file;

    LUT3DParams();
    bool operator==(const LUT3DParams &x) const;
    bool operator!=(const LUT3DParams &x) const;
};

struct VppColorspace {
    bool enable;
    HDR2SDRParams hdr2sdr;
    LUT3DParams lut3d;
    std::vector<ColorspaceConv> convs;

    VppColorspace();
    bool operator==(const VppColorspace &x) const;
    bool operator!=(const VppColorspace &x) const;
};

struct VppRff {
    bool  enable;
    bool  log;

    VppRff();
    bool operator==(const VppRff& x) const;
    bool operator!=(const VppRff& x) const;
    tstring print() const;
};

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
    float multiaddDepthMin;
    float multiaddDepthMax;
    bool log;

    VppDelogo();
    bool operator==(const VppDelogo& x) const;
    bool operator!=(const VppDelogo& x) const;
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

enum AFS_TUNE_MODE {
    AFS_TUNE_MODE_NONE = 0,
    AFS_TUNE_MODE_FINAL,
    AFS_TUNE_MODE_ANALYZE_SHIFT_ALL,
    AFS_TUNE_MODE_ANALYZE_SHIFT_Y,
    AFS_TUNE_MODE_ANALYZE_SHIFT_U,
    AFS_TUNE_MODE_ANALYZE_SHIFT_V,
    AFS_TUNE_MODE_ANALYZE_NONSHIFT_ALL,
    AFS_TUNE_MODE_ANALYZE_NONSHIFT_Y,
    AFS_TUNE_MODE_ANALYZE_NONSHIFT_U,
    AFS_TUNE_MODE_ANALYZE_NONSHIFT_V,
};

const CX_DESC list_afs_tune_mode[] = {
    { _T("none"), AFS_TUNE_MODE_NONE },
    { _T("final"), AFS_TUNE_MODE_FINAL },
    { _T("analyze_shift_all"), AFS_TUNE_MODE_ANALYZE_SHIFT_ALL },
    { _T("analyze_shift_y"), AFS_TUNE_MODE_ANALYZE_SHIFT_Y },
    { _T("analyze_shift_u"), AFS_TUNE_MODE_ANALYZE_SHIFT_U },
    { _T("analyze_shift_v"), AFS_TUNE_MODE_ANALYZE_SHIFT_V },
    { _T("analyze_nonshift_all"), AFS_TUNE_MODE_ANALYZE_NONSHIFT_ALL },
    { _T("analyze_nonshift_y"), AFS_TUNE_MODE_ANALYZE_NONSHIFT_Y },
    { _T("analyze_nonshift_u"), AFS_TUNE_MODE_ANALYZE_NONSHIFT_U },
    { _T("analyze_nonshift_v"), AFS_TUNE_MODE_ANALYZE_NONSHIFT_V },
    { NULL, 0 }
};

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
    AFS_TUNE_MODE tune;    //調整モード
    bool rff;              //rffフラグを認識して調整
    int timecode;          //timecode出力
    bool log;              //log出力

    VppAfs();
    void set_preset(int preset);
    int read_afs_inifile(const TCHAR *inifile);
    bool operator==(const VppAfs &x) const;
    bool operator!=(const VppAfs &x) const;
    tstring print() const;

    void check();
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

struct VppYadif {
    bool enable;
    bool log;
    VppYadifMode mode;

    VppYadif();
    bool operator==(const VppYadif& x) const;
    bool operator!=(const VppYadif& x) const;
    tstring print() const;
};

struct VppDecomb {
    bool enable;
    bool full;
    int threshold;
    int dthreshold;
    bool blend;

    VppDecomb();
    bool operator==(const VppDecomb& x) const;
    bool operator!=(const VppDecomb& x) const;
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
    bool operator==(const VppNnedi &x) const;
    bool operator!=(const VppNnedi &x) const;
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
    int drop;
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

struct VppMpdecimate {
    bool enable;
    int lo, hi, max;
    float frac;
    bool log;

    VppMpdecimate();
    bool operator==(const VppMpdecimate& x) const;
    bool operator!=(const VppMpdecimate& x) const;
    tstring print() const;
};

struct VppPad {
    bool enable;
    int left, top, right, bottom;

    VppPad();
    bool operator==(const VppPad &x) const;
    bool operator!=(const VppPad &x) const;
    tstring print() const;
};

enum class VppConvolution3dMatrix {
    Standard,
    Simple,
};

const CX_DESC list_vpp_convolution3d_matrix[] = {
    { _T("standard"),  (int)VppConvolution3dMatrix::Standard },
    { _T("simple"),    (int)VppConvolution3dMatrix::Simple   },
    { NULL, 0 }
};

struct VppConvolution3d {
    bool enable;
    bool fast;
    VppConvolution3dMatrix matrix;
    int threshYspatial;
    int threshCspatial;
    int threshYtemporal;
    int threshCtemporal;

    VppConvolution3d();
    bool operator==(const VppConvolution3d &x) const;
    bool operator!=(const VppConvolution3d &x) const;
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
    bool operator==(const VppKnn &x) const;
    bool operator!=(const VppKnn &x) const;
    tstring print() const;
};

enum VppNLMeansFP16Opt {
    NoOpt,
    BlockDiff,
    All
};

const CX_DESC list_vpp_nlmeans_fp16[] = {
    { _T("none"),      (int)VppNLMeansFP16Opt::NoOpt     },
    { _T("blockdiff"), (int)VppNLMeansFP16Opt::BlockDiff },
    { _T("all"),       (int)VppNLMeansFP16Opt::All       },
    { NULL, 0 }
};

const CX_DESC list_vpp_nlmeans_block_size[] = {
    { _T("3"),   3 },
    { _T("5"),   5 },
    { _T("7"),   7 },
    { _T("9"),   9 },
    { _T("11"), 11 },
    { _T("13"), 13 },
    { _T("15"), 15 },
    { _T("17"), 17 },
    { _T("19"), 19 },
    { _T("21"), 21 },
    { NULL, 0 }
};

struct VppNLMeans {
    bool  enable;
    float sigma;
    int   patchSize;
    int   searchSize;
    float h;
    VppNLMeansFP16Opt fp16;
    bool sharedMem;

    VppNLMeans();
    bool operator==(const VppNLMeans &x) const;
    bool operator!=(const VppNLMeans &x) const;
    tstring print() const;
};

struct VppPmd {
    bool  enable;
    float strength;
    float threshold;
    int   applyCount;
    bool  useExp;

    VppPmd();
    bool operator==(const VppPmd &x) const;
    bool operator!=(const VppPmd &x) const;
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

struct VppDenoiseDct {
    bool enable;
    float sigma;
    int step;
    int block_size;
    VppDenoiseDct();
    bool operator==(const VppDenoiseDct &x) const;
    bool operator!=(const VppDenoiseDct &x) const;
    tstring print() const;
};

const CX_DESC list_vpp_fft3d_block_size[] = {
    { _T("8"),   8 },
    { _T("16"), 16 },
    { _T("32"), 32 },
    { _T("64"), 64 },
    { NULL, 0 }
};

struct VppDenoiseFFT3D {
    bool enable;
    float sigma;
    float amount;
    int block_size;
    float overlap;
    float overlap2;
    int method;
    int temporal;
    VppFpPrecision precision;
    VppDenoiseFFT3D();
    bool operator==(const VppDenoiseFFT3D &x) const;
    bool operator!=(const VppDenoiseFFT3D &x) const;
    tstring print() const;
};

struct VppSubburn {
    bool  enable;
    tstring filename;
    std::string charcode;
    tstring fontsdir;
    int trackId;
    int assShaping;
    float scale;
    float transparency_offset;
    float brightness;
    float contrast;
    double ts_offset;
    bool vid_ts_offset;
    bool forced_subs_only;

    VppSubburn();
    bool operator==(const VppSubburn &x) const;
    bool operator!=(const VppSubburn &x) const;
    tstring print() const;
};

struct VppUnsharp {
    bool  enable;
    int   radius;
    float weight;
    float threshold;

    VppUnsharp();
    bool operator==(const VppUnsharp &x) const;
    bool operator!=(const VppUnsharp &x) const;
    tstring print() const;
};

struct VppEdgelevel {
    bool  enable;
    float strength;
    float threshold;
    float black;
    float white;

    VppEdgelevel();
    bool operator==(const VppEdgelevel &x) const;
    bool operator!=(const VppEdgelevel &x) const;
    tstring print() const;
};

struct VppWarpsharp {
    bool enable;
    float threshold;
    int blur;
    int type;
    float depth;
    int chroma;

    VppWarpsharp();
    bool operator==(const VppWarpsharp& x) const;
    bool operator!=(const VppWarpsharp& x) const;
    tstring print() const;
};

struct VppTweakChannel {
    float offset;
    float gain;
    float gamma;

    VppTweakChannel();
    bool enabled() const;
    bool operator==(const VppTweakChannel &x) const;
    bool operator!=(const VppTweakChannel &x) const;
    tstring print(const bool print_gamma = true) const;
};

struct VppTweak {
    bool  enable;
    float brightness; // -1.0 - 1.0 (0.0)
    float contrast;   // -2.0 - 2.0 (1.0)
    float gamma;      //  0.1 - 10.0 (1.0)
    float saturation; //  0.0 - 3.0 (1.0)
    float hue;        // -180 - 180 (0.0)
    bool swapuv;

    VppTweakChannel y, cb, cr;
    VppTweakChannel r, g, b;

    VppTweak();
    bool operator==(const VppTweak &x) const;
    bool operator!=(const VppTweak &x) const;
    tstring print(const bool print_rgb = true, const bool print_header = true) const;
    bool yuv_filter_enabled() const;
    bool rgb_filter_enabled() const;
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


enum class VppCurvesPreset {
    NONE,
    COLOR_NEGATIVE,
    PROCESS,
    DARKER,
    LIGHTER,
    INCREASE_CONTRAST,
    LINEAR_CONTRAST,
    MEDIUM_CONTRAST,
    STRONG_CONTRAST,
    NEGATIVE,
    VINTAGE
};

const CX_DESC list_vpp_curves_preset[] = {
    { _T("none"),              (int)VppCurvesPreset::NONE },
    { _T("color_negative"),    (int)VppCurvesPreset::COLOR_NEGATIVE      },
    { _T("process"),           (int)VppCurvesPreset::PROCESS  },
    { _T("darker"),            (int)VppCurvesPreset::DARKER  },
    { _T("lighter"),           (int)VppCurvesPreset::LIGHTER  },
    { _T("increase_contrast"), (int)VppCurvesPreset::INCREASE_CONTRAST  },
    { _T("linear_contrast"),   (int)VppCurvesPreset::LINEAR_CONTRAST  },
    { _T("medium_contrast"),   (int)VppCurvesPreset::MEDIUM_CONTRAST  },
    { _T("strong_contrast"),   (int)VppCurvesPreset::STRONG_CONTRAST  },
    { _T("negative"),          (int)VppCurvesPreset::NEGATIVE  },
    { _T("vintage"),           (int)VppCurvesPreset::VINTAGE  },
    { NULL, 0 }
};

struct VppCurveParams {
    tstring r, g, b, m;

    VppCurveParams();
    VppCurveParams(const tstring& r_, const tstring& g_, const tstring& b_, const tstring& m_);
    bool operator==(const VppCurveParams &x) const;
    bool operator!=(const VppCurveParams &x) const;
};

struct VppCurves {
    bool enable;
    VppCurvesPreset preset;
    VppCurveParams prm;
    tstring all;

    VppCurves();
    bool operator==(const VppCurves &x) const;
    bool operator!=(const VppCurves &x) const;
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
    bool operator==(const VppDeband &x) const;
    bool operator!=(const VppDeband &x) const;
    tstring print() const;
};

enum class VppOverlayAlphaMode {
    Override,
    Mul,
    LumaKey,
};

const CX_DESC list_vpp_overlay_alpha_mode[] = {
    { _T("override"),  (int)VppOverlayAlphaMode::Override },
    { _T("mul"),       (int)VppOverlayAlphaMode::Mul      },
    { _T("lumakey"),   (int)VppOverlayAlphaMode::LumaKey  },
    { NULL, 0 }
};

struct VppOverlayAlphaKey {
    float threshold;
    float tolerance;
    float shoftness;

    VppOverlayAlphaKey();
    bool operator==(const VppOverlayAlphaKey &x) const;
    bool operator!=(const VppOverlayAlphaKey &x) const;
    tstring print() const;
};

struct VppOverlay {
    bool enable;
    tstring inputFile;
    int posX;
    int posY;
    int width;
    int height;
    float alpha; // 不透明度 透明(0.0 - 1.0)透明
    VppOverlayAlphaMode alphaMode;
    VppOverlayAlphaKey lumaKey;
    bool loop;

    VppOverlay();
    bool operator==(const VppOverlay &x) const;
    bool operator!=(const VppOverlay &x) const;
    tstring print() const;
};

enum class VppFrucMode {
    Disabled,
    NVOFFRUCx2,
    NVOFFRUCFps,
};

struct VppFruc {
    bool enable;
    VppFrucMode mode;
    rgy_rational<int> targetFps;

    VppFruc();
    bool operator==(const VppFruc &x) const;
    bool operator!=(const VppFruc &x) const;
    tstring print() const;
};

struct RGYParamVpp {
    std::vector<VppType> filterOrder;
    RGY_VPP_RESIZE_ALGO resize_algo;
    RGY_VPP_RESIZE_MODE resize_mode;
    VppLibplaceboResample resize_libplacebo;
    VppColorspace colorspace;
    VppLibplaceboToneMapping libplacebo_tonemapping;
    VppDelogo delogo;
    VppAfs afs;
    VppNnedi nnedi;
    VppYadif yadif;
    VppDecomb decomb;
    VppRff rff;
    VppSelectEvery selectevery;
    VppDecimate decimate;
    VppMpdecimate mpdecimate;
    VppPad pad;
    VppConvolution3d convolution3d;
    VppKnn knn;
    VppNLMeans nlmeans;
    VppPmd pmd;
    VppDenoiseDct dct;
    VppSmooth smooth;
    VppDenoiseFFT3D fft3d;
    std::vector<VppSubburn> subburn;
    std::vector<VppLibplaceboShader> libplacebo_shader;
    VppUnsharp unsharp;
    VppEdgelevel edgelevel;
    VppWarpsharp warpsharp;
    VppCurves curves;
    VppTweak tweak;
    VppTransform transform;
    VppDeband deband;
    VppLibplaceboDeband libplacebo_deband;
    std::vector<VppOverlay> overlay;
    VppFruc fruc;
    bool checkPerformance;

    RGYParamVpp();
    bool operator==(const RGYParamVpp& x) const;
    bool operator!=(const RGYParamVpp& x) const;
};


static const char *maxCLLSource = "copy";
static const char *masterDisplaySource = "copy";

static const TCHAR *RGY_METADATA_CLEAR = _T("clear");
static const TCHAR *RGY_METADATA_COPY = _T("copy");

static const int TRACK_SELECT_BY_LANG  = -1;
static const int TRACK_SELECT_BY_CODEC = -2;

struct AudioSelect {
    int      trackID;         //選択したトラックのリスト 1,2,...(1から連番で指定)
                              // 0 ... 全指定
                              // TRACK_SELECT_BY_LANG  ... langによる選択
                              // TRACK_SELECT_BY_CODEC ... selectCodecによる選択
    tstring  decCodecPrm;     //音声エンコードのデコーダのパラメータ
    tstring  encCodec;        //音声エンコードのコーデック
    tstring  encCodecPrm;     //音声エンコードのコーデックのパラメータ
    tstring  encCodecProfile; //音声エンコードのコーデックのプロファイル
    int      encBitrate;      //音声エンコードに選択した音声トラックのビットレート
    std::pair<bool, int> encQuality;      //音声エンコードに選択した音声トラックの品質 <値が設定されているかと値のペア>
    int      encSamplingRate;      //サンプリング周波数
    double   addDelayMs;           //追加する音声の遅延(millisecond)
    tstring  extractFilename;      //抽出する音声のファイル名のリスト
    tstring  extractFormat;        //抽出する音声ファイルのフォーマット
    tstring  filter;               //音声フィルタ
    std::array<std::string, MAX_SPLIT_CHANNELS> streamChannelSelect; //入力音声の使用するチャンネル
    std::array<std::string, MAX_SPLIT_CHANNELS> streamChannelOut;    //出力音声のチャンネル
    tstring  bsf;                  // 適用するbitstreamfilterの名前
    tstring  disposition;          // 指定のdisposition
    std::string lang;              // 言語選択
    std::string selectCodec;       // 対象コーデック
    std::vector<tstring> metadata;
    std::string resamplerPrm;

    AudioSelect();
    ~AudioSelect() {};
};

struct AudioSource {
    tstring filename;
    tstring format;
    RGYOptList inputOpt; //入力オプション
    std::map<int, AudioSelect> select;

    AudioSource();
    ~AudioSource() {};
};

struct SubtitleSelect {
    int trackID;         // 選択したトラックのリスト 1,2,...(1から連番で指定)
                         //  0 ... 全指定
                         //  TRACK_SELECT_BY_LANG ... langによる選択
                         //  TRACK_SELECT_BY_CODEC ... selectCodecによる選択
    tstring encCodec;
    tstring encCodecPrm;
    tstring decCodecPrm;
    bool asdata;
    tstring bsf;          // 適用するbitstreamfilterの名前
    tstring disposition;  // 指定のdisposition
    std::string lang;         // 言語選択
    std::string selectCodec;  // 対象コーデック
    std::vector<tstring> metadata;

    SubtitleSelect();
    ~SubtitleSelect() {};
};

struct SubSource {
    tstring filename;
    tstring format;
    RGYOptList inputOpt; //入力オプション
    std::map<int, SubtitleSelect> select;

    SubSource();
    ~SubSource() {};
};

struct DataSelect {
    int trackID;         // 選択したトラックのリスト 1,2,...(1から連番で指定)
                         //  0 ... 全指定
                         //  TRACK_SELECT_BY_LANG ... langによる選択
                         //  TRACK_SELECT_BY_CODEC ... selectCodecによる選択
    tstring encCodec;
    tstring disposition; // 指定のdisposition
    std::string lang;    // 言語選択
    std::string selectCodec; // 対象コーデック
    std::vector<tstring> metadata;

    DataSelect();
    ~DataSelect() {};
};

struct VMAFParam {
    bool enable;
    tstring model;
    int threads;
    int subsample;
    bool phone_model;
    bool enable_transform;

    VMAFParam();
    bool operator==(const VMAFParam &x) const;
    bool operator!=(const VMAFParam &x) const;
    tstring print() const;
};

struct RGYVideoQualityMetric {
    bool ssim;
    bool psnr;
    VMAFParam vmaf;

    RGYVideoQualityMetric();
    ~RGYVideoQualityMetric() {};
    bool enabled() const;
    tstring enabled_metric() const;
};

using AttachmentSelect = DataSelect;
using AttachmentSource = SubSource;

struct GPUAutoSelectMul {
    float cores;
    float gen;
    float gpu;
    float ve;

    GPUAutoSelectMul();
    bool operator==(const GPUAutoSelectMul &x) const;
    bool operator!=(const GPUAutoSelectMul &x) const;
};

struct RGYDebugLogFile {
    bool enable;
    tstring filename;

    RGYDebugLogFile();
    bool operator==(const RGYDebugLogFile &x) const;
    bool operator!=(const RGYDebugLogFile &x) const;
    tstring getFilename(const tstring& outputFilename, const tstring& defaultAppendix) const;
};

struct RGYParamInput {
    RGYResizeResMode resizeResMode;
    bool ignoreSAR;
    tstring avswDecoder; //avswデコーダの指定

    RGYParamInput();
    ~RGYParamInput();
};

struct RGYParamCommon {
    tstring inputFilename;        //入力ファイル名
    tstring outputFilename;       //出力ファイル名
    tstring muxOutputFormat;      //出力フォーマット
    VideoVUIInfo out_vui;
    RGYOptList inputOpt; //入力オプション
    std::string maxCll;
    std::string masterDisplay;
    CspTransfer atcSei;
    bool hdr10plusMetadataCopy;
    tstring dynamicHdr10plusJson;
    bool doviRpuMetadataCopy;
    tstring doviRpuFile;
    RGYDOVIRpuConvertParam doviRpuParams;
    RGYDOVIProfile doviProfile;
    std::string avVideoCodec;
    std::string videoCodecTag;
    tstring avcodec_videnc_prms;
    std::vector<tstring> videoMetadata;
    std::vector<tstring> formatMetadata;
    float seekRatio;               //指定された秒数分先頭を飛ばす
    float seekSec;               //指定された秒数分先頭を飛ばす
    float seekToSec;
    int nSubtitleSelectCount;
    SubtitleSelect **ppSubtitleSelectList;
    std::vector<SubSource> subSource;
    std::vector<AudioSource> audioSource;
    int nAudioSelectCount; //pAudioSelectの数
    AudioSelect **ppAudioSelectList;
    int        nDataSelectCount;
    DataSelect **ppDataSelectList;
    int        nAttachmentSelectCount;
    AttachmentSelect **ppAttachmentSelectList;
    std::vector<AttachmentSource> attachmentSource;
    int audioResampler;
    int inputRetry;
    double demuxAnalyzeSec;
    int64_t demuxProbesize;
    tstring inputPixFmtStr;
    int AVMuxTarget;                       //RGY_MUX_xxx
    int videoTrack;
    int videoStreamId;
    int nTrimCount;
    sTrim *pTrimList;
    bool copyChapter;
    bool keyOnChapter;
    bool chapterNoTrim;
    int audioIgnoreDecodeError;
    int videoIgnoreTimestampError;
    RGYOptList muxOpt;
    bool offsetVideoDtsAdvance;
    bool allowOtherNegativePts;
    bool disableMp4Opt;
    bool debugDirectAV1Out;
    bool debugRawOut;
    tstring outReplayFile;
    RGY_CODEC outReplayCodec;
    tstring chapterFile;
    tstring keyFile;
    TCHAR *AVInputFormat;
    RGYAVSync AVSyncMode;     //avsyncの方法 (NV_AVSYNC_xxx)
    bool timestampPassThrough; //timestampをそのまま出力する
    bool timecode;
    tstring timecodeFile;
    tstring tcfileIn;
    rgy_rational<int> timebase;
    RGYHEVCBsf hevcbsf;

    RGYVideoQualityMetric metric;

    RGYParamCommon();
    ~RGYParamCommon();
};

enum class RGYParamAvoidIdleClockMode {
    Disabled,
    Auto,
    Force
};

const CX_DESC list_avoid_idle_clock[] = {
    { _T("off"),  (int)RGYParamAvoidIdleClockMode::Disabled },
    { _T("auto"), (int)RGYParamAvoidIdleClockMode::Auto     },
    { _T("on"),   (int)RGYParamAvoidIdleClockMode::Force    },
    { NULL, 0 }
};

struct RGYParamAvoidIdleClock {
    RGYParamAvoidIdleClockMode mode;
    float loadPercent;

    RGYParamAvoidIdleClock();
    bool operator==(const RGYParamAvoidIdleClock &x) const;
    bool operator!=(const RGYParamAvoidIdleClock &x) const;
};

struct RGYParallelEncSendData;

enum class RGYParamParallelEncCache {
    Mem,
    File,
};

const CX_DESC list_parallel_enc_cache[] = {
    { _T("mem"),  (int)RGYParamParallelEncCache::Mem  },
    { _T("file"), (int)RGYParamParallelEncCache::File },
    { NULL, 0 }
};

struct RGYParamParallelEnc {
    int parallelCount; // 並列処理数
    int parallelId; // 親=-1, 子=0～
    int chunks; // 分割数
    RGYParamParallelEncCache cacheMode;
    bool delayChildSync; // 親-子間のデータやり取りを少し遅らせる
    RGYParallelEncSendData *sendData; // 並列処理時に親-子間のデータやり取り用
    RGYParamParallelEnc();
    bool operator==(const RGYParamParallelEnc &x) const;
    bool operator!=(const RGYParamParallelEnc &x) const;
    bool isParent() const { return (parallelCount > 1 || parallelCount == -1) && parallelId < 0; }
    bool isChild()  const { return parallelCount > 1 && parallelId >= 0; }
    bool isEnabled() const { return parallelCount > 1 || parallelCount == -1; }
};

enum class RGYParamInitVulkan {
    Disable,
    TargetVendor,
    All,
};

struct RGYParamLogOpt {
    bool addTime;
    bool addLogLevel;
    bool disableColor;

    RGYParamLogOpt();
    bool operator==(const RGYParamLogOpt &x) const;
    bool operator!=(const RGYParamLogOpt &x) const;
};

struct RGYParamControl {
    int threadCsp;
    RGY_SIMD simdCsp;
    tstring logfile;              //ログ出力先
    RGYParamLogLevel loglevel; //ログ出力レベル
    RGYParamLogOpt logOpt;
    RGYDebugLogFile logFramePosList;     //framePosList出力
    RGYDebugLogFile logPacketsList;
    RGYDebugLogFile logMuxVidTs;
    int threadOutput;
    int threadAudio;
    int threadInput;
    RGYParamThreads threadParams;
    int procSpeedLimit;      //処理速度制限 (0で制限なし)
    bool taskPerfMonitor;
    int64_t perfMonitorSelect;
    int64_t perfMonitorSelectMatplot;
    int     perfMonitorInterval;
    uint32_t parentProcessID;
    bool lowLatency;
    GPUAutoSelectMul gpuSelect;
    bool skipHWEncodeCheck;
    bool skipHWDecodeCheck;
    tstring avsdll;
    tstring vsdir;
    bool enableOpenCL;
    RGYParamInitVulkan enableVulkan;
    int openclBuildThreads;
    RGYParamAvoidIdleClock avoidIdleClock;
    bool processMonitorDevUsage;
    bool processMonitorDevUsageReset;

    int outputBufSizeMB;         //出力バッファサイズ

    RGYParamParallelEnc parallelEnc;

    RGYParamControl();
    ~RGYParamControl();
};

bool trim_active(const sTrimParam *pTrim);
std::pair<bool, int> frame_inside_range(int frame, const std::vector<sTrim> &trimList);
bool rearrange_trim_list(int frame, int offset, std::vector<sTrim> &trimList);
tstring print_metadata(const std::vector<tstring>& metadata);
bool metadata_copy(const std::vector<tstring> &metadata);
bool metadata_clear(const std::vector<tstring> &metadata);

const FEATURE_DESC list_simd[] = {
    { _T("auto"),     (uint64_t)RGY_SIMD::SIMD_ALL  },
    { _T("none"),     (uint64_t)RGY_SIMD::NONE },
    { _T("sse2"),     (uint64_t)RGY_SIMD::SSE2 },
    { _T("sse3"),     (uint64_t)(RGY_SIMD::SSE3| RGY_SIMD::SSE2) },
    { _T("ssse3"),    (uint64_t)(RGY_SIMD::SSSE3| RGY_SIMD::SSE3| RGY_SIMD::SSE2) },
    { _T("sse41"),    (uint64_t)(RGY_SIMD::SSE41| RGY_SIMD::SSSE3| RGY_SIMD::SSE3| RGY_SIMD::SSE2) },
    { _T("avx"),      (uint64_t)(RGY_SIMD::AVX  | RGY_SIMD::SSE42| RGY_SIMD::SSE41| RGY_SIMD::SSSE3| RGY_SIMD::SSE3| RGY_SIMD::SSE2) },
    { _T("avx2"),     (uint64_t)(RGY_SIMD::AVX2 | RGY_SIMD::AVX| RGY_SIMD::SSE42| RGY_SIMD::SSE41| RGY_SIMD::SSSE3| RGY_SIMD::SSE3| RGY_SIMD::SSE2) },
    { nullptr,        (uint64_t)RGY_SIMD::NONE }
};

template <uint32_t size>
static bool bSplitChannelsEnabled(const std::array<std::string, size>& streamChannels) {
    bool bEnabled = false;
    for (const auto& st : streamChannels) {
        bEnabled |= !st.empty();
    }
    return bEnabled;
}

unique_ptr<RGYHDR10Plus> initDynamicHDR10Plus(const tstring &dynamicHdr10plusJson, shared_ptr<RGYLog> log);

bool invalid_with_raw_out(const RGYParamCommon &prm, shared_ptr<RGYLog> log);

#endif //__RGY_PRM_H__
