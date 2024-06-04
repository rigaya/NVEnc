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

#include <set>
#include <iostream>
#include <iomanip>
#include "rgy_util.h"
#include "rgy_filesystem.h"
#include "rgy_version.h"
#if !CLFILTERS_AUF
#include "rgy_avutil.h"
#endif
#include "rgy_prm.h"
#include "rgy_err.h"
#include "rgy_perf_monitor.h"
#include "rgy_ini.h"
#if ENABLE_VPP_FILTER_AFS
#include "afs_stg.h"
#endif

static const auto VPPTYPE_TO_STR = make_array<std::pair<VppType, tstring>>(
    std::make_pair(VppType::VPP_NONE,                _T("none")),
#if ENCODER_QSV
    std::make_pair(VppType::MFX_COLORSPACE,          _T("mfx_colorspace")),
    std::make_pair(VppType::MFX_CROP,                _T("mfx_crop")),
    std::make_pair(VppType::MFX_ROTATE,              _T("mfx_rotate")),
    std::make_pair(VppType::MFX_MIRROR,              _T("mfx_mirror")),
    std::make_pair(VppType::MFX_DEINTERLACE,         _T("mfx_deinterlace")),
    std::make_pair(VppType::MFX_IMAGE_STABILIZATION, _T("mfx_image_stab")),
    std::make_pair(VppType::MFX_MCTF,                _T("mfx_mctf")),
    std::make_pair(VppType::MFX_DENOISE,             _T("mfx_denoise")),
    std::make_pair(VppType::MFX_RESIZE,              _T("mfx_resize")),
    std::make_pair(VppType::MFX_DETAIL_ENHANCE,      _T("mfx_detail_enhance")),
    std::make_pair(VppType::MFX_FPS_CONV,            _T("mfx_fps_conv")),
    std::make_pair(VppType::MFX_PERC_ENC_PREFILTER,  _T("mfx_perc_enc_prefilter")),
    std::make_pair(VppType::MFX_COPY,                _T("mfx_copy")),
#endif //#if ENCODER_QSV
#if ENCODER_NVENC || CLFILTERS_AUF
    std::make_pair(VppType::NVVFX_DENOISE,            _T("nvvfx_denoise")),
    std::make_pair(VppType::NVVFX_ARTIFACT_REDUCTION, _T("nvvfx_artifact_reduction")),
#endif
#if ENCODER_VCEENC
    std::make_pair(VppType::AMF_CONVERTER,           _T("amf_perc_enc_prefilter")),
    std::make_pair(VppType::AMF_PREPROCESS,          _T("amf_preprocess")),
    std::make_pair(VppType::AMF_RESIZE,              _T("amf_resize")),
    std::make_pair(VppType::AMF_VQENHANCE,           _T("amf_vqenhance")),
#endif //#if ENCODER_VCEENC
#if ENCODER_MPP
    std::make_pair(VppType::IEP_DEINTERLACE,         _T("iep_deinterlace")),
    std::make_pair(VppType::RGA_CROP,                _T("rga_crop")),
    std::make_pair(VppType::RGA_CSPCONV,             _T("rga_cspconv")),
    std::make_pair(VppType::RGA_RESIZE,              _T("rga_resize")),
#endif //#if ENCODER_VCEENC
    std::make_pair(VppType::CL_COLORSPACE,           _T("colorspace")),
    std::make_pair(VppType::CL_AFS,                  _T("afs")),
    std::make_pair(VppType::CL_NNEDI,                _T("nnedi")),
    std::make_pair(VppType::CL_YADIF,                _T("yadif")),
    std::make_pair(VppType::CL_DECOMB,               _T("decomb")),
    std::make_pair(VppType::CL_DECIMATE,             _T("decimate")),
    std::make_pair(VppType::CL_MPDECIMATE,           _T("mpdecimate")),
    std::make_pair(VppType::CL_RFF,                  _T("rff")),
    std::make_pair(VppType::CL_DELOGO,               _T("delogo")),
    std::make_pair(VppType::CL_TRANSFORM,            _T("transform")),
    std::make_pair(VppType::CL_CONVOLUTION3D,        _T("convolution3d")),
    std::make_pair(VppType::CL_DENOISE_KNN,          _T("knn")),
    std::make_pair(VppType::CL_DENOISE_NLMEANS,      _T("nlmeans")),
    std::make_pair(VppType::CL_DENOISE_PMD,          _T("pmd")),
    std::make_pair(VppType::CL_DENOISE_DCT,          _T("denoise-dct")),
    std::make_pair(VppType::CL_DENOISE_SMOOTH,       _T("smooth")),
    std::make_pair(VppType::CL_RESIZE,               _T("resize")),
    std::make_pair(VppType::CL_UNSHARP,              _T("unsharp")),
    std::make_pair(VppType::CL_EDGELEVEL,            _T("edgelevel")),
    std::make_pair(VppType::CL_WARPSHARP,            _T("warpsharp")),
    std::make_pair(VppType::CL_CURVES,               _T("curves")),
    std::make_pair(VppType::CL_TWEAK,                _T("tweak")),
    std::make_pair(VppType::CL_DEBAND,               _T("deband")),
    std::make_pair(VppType::CL_PAD,                  _T("pad"))
);
MAP_PAIR_0_1(vppfilter, type, VppType, str, tstring, VPPTYPE_TO_STR, VppType::VPP_NONE, _T("none"));

std::vector<CX_DESC> get_list_vpp_filter() {
    std::vector<CX_DESC> list_vpp_filter;
    list_vpp_filter.reserve(VPPTYPE_TO_STR.size()+1);
    for (const auto& vpp : VPPTYPE_TO_STR) {
        list_vpp_filter.push_back({ vpp.second.c_str(), (int)vpp.first});
    }
    list_vpp_filter.push_back({ nullptr, 0 });
    return list_vpp_filter;
}

RGYQPSet::RGYQPSet() :
    enable(true),
    qpI(0), qpP(0), qpB(0) {

};

RGYQPSet::RGYQPSet(int i, int p, int b) :
    enable(true),
    qpI(i), qpP(p), qpB(b) {

};

RGYQPSet::RGYQPSet(int i, int p, int b, bool enable_) :
    enable(enable_),
    qpI(i), qpP(p), qpB(b) {

};

int RGYQPSet::qp(int i) const {
    switch (i) {
    case 0: return qpI;
    case 1: return qpP;
    case 2: return qpB;
    default: return 0;
    }
}

int& RGYQPSet::qp(int i) {
    switch (i) {
    case 0: return qpI;
    case 1: return qpP;
    case 2: return qpB;
    default: return qpI;
    }
}

bool RGYQPSet::operator==(const RGYQPSet &x) const {
    return enable == x.enable
        && qpI == x.qpI
        && qpP == x.qpP
        && qpB == x.qpB;
}
bool RGYQPSet::operator!=(const RGYQPSet &x) const {
    return !(*this == x);
}

int RGYQPSet::parse(const TCHAR *str) {
    int a[4] = { 0 };
    if (   4 == _stscanf_s(str, _T("%d;%d:%d:%d"), &a[3], &a[0], &a[1], &a[2])
        || 4 == _stscanf_s(str, _T("%d;%d/%d/%d"), &a[3], &a[0], &a[1], &a[2])
        || 4 == _stscanf_s(str, _T("%d;%d.%d.%d"), &a[3], &a[0], &a[1], &a[2])
        || 4 == _stscanf_s(str, _T("%d;%d,%d,%d"), &a[3], &a[0], &a[1], &a[2])) {
        a[3] = a[3] ? 1 : 0;
    } else if (
           3 == _stscanf_s(str, _T("%d:%d:%d"), &a[0], &a[1], &a[2])
        || 3 == _stscanf_s(str, _T("%d/%d/%d"), &a[0], &a[1], &a[2])
        || 3 == _stscanf_s(str, _T("%d.%d.%d"), &a[0], &a[1], &a[2])
        || 3 == _stscanf_s(str, _T("%d,%d,%d"), &a[0], &a[1], &a[2])) {
        a[3] = 1;
    } else if (
           3 == _stscanf_s(str, _T("%d;%d:%d"), &a[3], &a[0], &a[1])
        || 3 == _stscanf_s(str, _T("%d;%d/%d"), &a[3], &a[0], &a[1])
        || 3 == _stscanf_s(str, _T("%d;%d.%d"), &a[3], &a[0], &a[1])
        || 3 == _stscanf_s(str, _T("%d;%d,%d"), &a[3], &a[0], &a[1])) {
        a[3] = a[3] ? 1 : 0;
        a[2] = a[1];
    } else if (
           2 == _stscanf_s(str, _T("%d:%d"), &a[0], &a[1])
        || 2 == _stscanf_s(str, _T("%d/%d"), &a[0], &a[1])
        || 2 == _stscanf_s(str, _T("%d.%d"), &a[0], &a[1])
        || 2 == _stscanf_s(str, _T("%d,%d"), &a[0], &a[1])) {
        a[3] = 1;
        a[2] = a[1];
    } else if (2 == _stscanf_s(str, _T("%d;%d"), &a[3], &a[0])) {
        a[3] = a[3] ? 1 : 0;
        a[1] = a[0];
        a[2] = a[0];
    } else if (1 == _stscanf_s(str, _T("%d"), &a[0])) {
        a[3] = 1;
        a[1] = a[0];
        a[2] = a[0];
    } else {
        return 1;
    }
    enable = a[3] != 0;
    qpI = a[0];
    qpP = a[1];
    qpB = a[2];
    return 0;
}

void RGYQPSet::applyQPMinMax(const int min, const int max) {
    qpI = clamp(qpI, min, max);
    qpP = clamp(qpP, min, max);
    qpB = clamp(qpB, min, max);
}

RGY_VPP_RESIZE_TYPE getVppResizeType(RGY_VPP_RESIZE_ALGO resize) {
    if (resize == RGY_VPP_RESIZE_AUTO) {
        return RGY_VPP_RESIZE_TYPE_AUTO;
    } else if (resize < RGY_VPP_RESIZE_OPENCL_CUDA_MAX) {
        return RGY_VPP_RESIZE_TYPE_OPENCL;
#if ENCODER_QSV
    } else if (resize < RGY_VPP_RESIZE_MFX_MAX) {
        return RGY_VPP_RESIZE_TYPE_MFX;
#endif
#if ENCODER_NVENC && (!defined(_M_IX86) || FOR_AUO)
    } else if (resize < RGY_VPP_RESIZE_NPPI_MAX) {
        return RGY_VPP_RESIZE_TYPE_NPPI;
#endif
#if ENCODER_NVENC && (!defined(_M_IX86) || FOR_AUO) || CUFILTERS || CLFILTERS_AUF
    } else if (resize < RGY_VPP_RESIZE_NVVFX_MAX) {
        return RGY_VPP_RESIZE_TYPE_NVVFX;
#endif
#if ENCODER_VCEENC
    } else if (resize < RGY_VPP_RESIZE_AMF_MAX) {
        return RGY_VPP_RESIZE_TYPE_AMF;
#endif
#if ENCODER_MPP
    } else if (resize < RGY_VPP_RESIZE_RGA_MAX) {
        return RGY_VPP_RESIZE_TYPE_RGA;
#endif
    } else {
        return RGY_VPP_RESIZE_TYPE_UNKNOWN;
    }
}

ColorspaceConv::ColorspaceConv() :
    from(),
    to(),
    sdr_source_peak(FILTER_DEFAULT_COLORSPACE_NOMINAL_SOURCE_PEAK),
    approx_gamma(false),
    scene_ref(false) {

}
bool ColorspaceConv::operator==(const ColorspaceConv &x) const {
    return from == x.from
        && to == x.to
        && sdr_source_peak == x.sdr_source_peak
        && approx_gamma == x.approx_gamma
        && scene_ref == x.scene_ref;
}
bool ColorspaceConv::operator!=(const ColorspaceConv &x) const {
    return !(*this == x);
}

TonemapHable::TonemapHable() :
    a(FILTER_DEFAULT_HDR2SDR_HABLE_A),
    b(FILTER_DEFAULT_HDR2SDR_HABLE_B),
    c(FILTER_DEFAULT_HDR2SDR_HABLE_C),
    d(FILTER_DEFAULT_HDR2SDR_HABLE_D),
    e(FILTER_DEFAULT_HDR2SDR_HABLE_E),
    f(FILTER_DEFAULT_HDR2SDR_HABLE_F) {}

bool TonemapHable::operator==(const TonemapHable &x) const {
    return a == x.a
        && b == x.b
        && c == x.c
        && d == x.d
        && e == x.e
        && f == x.f;
}
bool TonemapHable::operator!=(const TonemapHable &x) const {
    return !(*this == x);
}
TonemapMobius::TonemapMobius() :
    transition(FILTER_DEFAULT_HDR2SDR_MOBIUS_TRANSITION),
    peak(FILTER_DEFAULT_HDR2SDR_MOBIUS_PEAK) {
}
bool TonemapMobius::operator==(const TonemapMobius &x) const {
    return transition == x.transition
        && peak == x.peak;
}
bool TonemapMobius::operator!=(const TonemapMobius &x) const {
    return !(*this == x);
}
TonemapReinhard::TonemapReinhard() :
    contrast(FILTER_DEFAULT_HDR2SDR_REINHARD_CONTRAST),
    peak(FILTER_DEFAULT_HDR2SDR_REINHARD_PEAK) {
}
bool TonemapReinhard::operator==(const TonemapReinhard &x) const {
    return contrast == x.contrast
        &&peak == x.peak;
}
bool TonemapReinhard::operator!=(const TonemapReinhard &x) const {
    return !(*this == x);
}

HDR2SDRParams::HDR2SDRParams() :
    tonemap(HDR2SDR_DISABLED),
    hable(),
    mobius(),
    reinhard(),
    ldr_nits(FILTER_DEFAULT_COLORSPACE_LDRNITS),
    hdr_source_peak(FILTER_DEFAULT_COLORSPACE_HDR_SOURCE_PEAK),
    desat_base(FILTER_DEFAULT_HDR2SDR_DESAT_BASE),
    desat_strength(FILTER_DEFAULT_HDR2SDR_DESAT_STRENGTH),
    desat_exp(FILTER_DEFAULT_HDR2SDR_DESAT_EXP) {

}
bool HDR2SDRParams::operator==(const HDR2SDRParams &x) const {
    return tonemap == x.tonemap
        && hable == x.hable
        && mobius == x.mobius
        && reinhard == x.reinhard
        && ldr_nits == x.ldr_nits
        && hdr_source_peak == x.hdr_source_peak
        && desat_base == x.desat_base
        && desat_strength == x.desat_strength
        && desat_exp == x.desat_exp;
}
bool HDR2SDRParams::operator!=(const HDR2SDRParams &x) const {
    return !(*this == x);
}

LUT3DParams::LUT3DParams() :
    interp(FILTER_DEFAULT_LUT3D_INTERP),
    table_file() {

}
bool LUT3DParams::operator==(const LUT3DParams &x) const {
    return interp == x.interp
        && table_file == x.table_file;
}
bool LUT3DParams::operator!=(const LUT3DParams &x) const {
    return !(*this == x);
}

VppColorspace::VppColorspace() :
    enable(false),
    hdr2sdr(),
    lut3d(),
    convs() {

}

bool VppColorspace::operator==(const VppColorspace &x) const {
    if (enable != x.enable
        || x.hdr2sdr != this->hdr2sdr
        || x.lut3d != this->lut3d
        || x.convs.size() != this->convs.size()) {
        return false;
    }
    for (size_t i = 0; i < x.convs.size(); i++) {
        if (x.convs[i].from != this->convs[i].from
            || x.convs[i].to != this->convs[i].to) {
            return false;
        }
    }
    return true;
}
bool VppColorspace::operator!=(const VppColorspace &x) const {
    return !(*this == x);
}

VppRff::VppRff() :
    enable(false),
    log(false) {

}

bool VppRff::operator==(const VppRff &x) const {
    if (  enable != x.enable
        || log != x.log) {
        return false;
    }
    return true;
}
bool VppRff::operator!=(const VppRff &x) const {
    return !(*this == x);
}

tstring VppRff::print() const {
    return strsprintf(_T("rff: log %s"), (log) ? _T("on") : _T("off"));
}

VppDelogo::VppDelogo() :
    enable(false),
    logoFilePath(),
    logoSelect(),
    posX(0), posY(0),
    depth(FILTER_DEFAULT_DELOGO_DEPTH),
    Y(0), Cb(0), Cr(0),
    mode(DELOGO_MODE_REMOVE),
    autoFade(false),
    autoNR(false),
    NRArea(0),
    NRValue(0),
    multiaddDepthMin(0.0f),
    multiaddDepthMax(128.0f),
    log(false) {
}

bool VppDelogo::operator==(const VppDelogo& x) const {
    return enable == x.enable
        && logoFilePath == x.logoFilePath
        && logoSelect == x.logoSelect
        && posX == x.posX
        && posY == x.posY
        && depth == x.depth
        && Y == x.Y
        && Cb == x.Cb
        && Cr == x.Cr
        && mode == x.mode
        && autoFade == x.autoFade
        && autoNR == x.autoNR
        && NRArea == x.NRArea
        && NRValue == x.NRValue
        && multiaddDepthMin == x.multiaddDepthMin
        && multiaddDepthMax == x.multiaddDepthMax
        && log == x.log;
}
bool VppDelogo::operator!=(const VppDelogo& x) const {
    return !(*this == x);
}

tstring VppDelogo::print() const {
    tstring str = _T("");
    switch (mode) {
    case DELOGO_MODE_ADD:
        str += _T(", add");
        break;
    case DELOGO_MODE_ADD_MULTI:
        str += _T(", multi_add");
        break;
    case DELOGO_MODE_REMOVE:
    default:
        break;
    }
    if (posX || posY) {
        str += strsprintf(_T(", pos=%d:%d"), posX, posY);
    }
    if (depth != FILTER_DEFAULT_DELOGO_DEPTH) {
        str += strsprintf(_T(", dpth=%d"), depth);
    }
    if (Y || Cb || Cr) {
        str += strsprintf(_T(", YCbCr=%d:%d:%d"), Y, Cb, Cr);
    }
    if (autoFade) {
        str += _T(", auto_fade");
    }
    if (autoNR) {
        str += _T(", auto_nr");
    }
    if ((autoFade || autoNR) && log) {
        str += _T(", log");
    }
    if (NRValue) {
        str += strsprintf(_T(", nr_value=%d"), NRValue);
    }
    if (NRArea) {
        str += strsprintf(_T(", nr_area=%d"), NRArea);
    }
    if (mode == DELOGO_MODE_ADD_MULTI) {
        str += strsprintf(_T(", multi_add_depth=%.1f-%.1f"), multiaddDepthMin, multiaddDepthMax);
    }
    return str;
}

VppAfs::VppAfs() :
    enable(false),
    tb_order(FILTER_DEFAULT_AFS_TB_ORDER),
    clip(scan_clip(FILTER_DEFAULT_AFS_CLIP_TB, FILTER_DEFAULT_AFS_CLIP_TB, FILTER_DEFAULT_AFS_CLIP_LR, FILTER_DEFAULT_AFS_CLIP_LR)),
    method_switch(FILTER_DEFAULT_AFS_METHOD_SWITCH),
    coeff_shift(FILTER_DEFAULT_AFS_COEFF_SHIFT),
    thre_shift(FILTER_DEFAULT_AFS_THRE_SHIFT),
    thre_deint(FILTER_DEFAULT_AFS_THRE_DEINT),
    thre_Ymotion(FILTER_DEFAULT_AFS_THRE_YMOTION),
    thre_Cmotion(FILTER_DEFAULT_AFS_THRE_CMOTION),
    analyze(FILTER_DEFAULT_AFS_ANALYZE),
    shift(FILTER_DEFAULT_AFS_SHIFT),
    drop(FILTER_DEFAULT_AFS_DROP),
    smooth(FILTER_DEFAULT_AFS_SMOOTH),
    force24(FILTER_DEFAULT_AFS_FORCE24),
    tune(FILTER_DEFAULT_AFS_TUNE),
    rff(FILTER_DEFAULT_AFS_RFF),
    timecode(FILTER_DEFAULT_AFS_TIMECODE),
    log(FILTER_DEFAULT_AFS_LOG) {
    check();
}

bool VppAfs::operator==(const VppAfs &x) const {
    return enable == x.enable
        && tb_order == x.tb_order
        && clip.bottom == x.clip.bottom
        && clip.left == x.clip.left
        && clip.top == x.clip.top
        && clip.right == x.clip.right
        && method_switch == x.method_switch
        && coeff_shift == x.coeff_shift
        && thre_shift == x.thre_shift
        && thre_deint == x.thre_deint
        && thre_Ymotion == x.thre_Ymotion
        && thre_Cmotion == x.thre_Cmotion
        && analyze == x.analyze
        && shift == x.shift
        && drop == x.drop
        && smooth == x.smooth
        && force24 == x.force24
        && tune == x.tune
        && rff == x.rff
        && timecode == x.timecode
        && log == x.log;
}
bool VppAfs::operator!=(const VppAfs &x) const {
    return !(*this == x);
}

void VppAfs::check() {
    if (!shift) {
        method_switch = 0;
        coeff_shift = 0;
    }
    drop &= shift;
    smooth &= drop;
}

void VppAfs::set_preset(int preset) {
    switch (preset) {
    case AFS_PRESET_DEFAULT: //デフォルト
        method_switch = FILTER_DEFAULT_AFS_METHOD_SWITCH;
        coeff_shift   = FILTER_DEFAULT_AFS_COEFF_SHIFT;
        thre_shift    = FILTER_DEFAULT_AFS_THRE_SHIFT;
        thre_deint    = FILTER_DEFAULT_AFS_THRE_DEINT;
        thre_Ymotion  = FILTER_DEFAULT_AFS_THRE_YMOTION;
        thre_Cmotion  = FILTER_DEFAULT_AFS_THRE_CMOTION;
        analyze       = FILTER_DEFAULT_AFS_ANALYZE;
        shift         = FILTER_DEFAULT_AFS_SHIFT;
        drop          = FILTER_DEFAULT_AFS_DROP;
        smooth        = FILTER_DEFAULT_AFS_SMOOTH;
        force24       = FILTER_DEFAULT_AFS_FORCE24;
        tune          = FILTER_DEFAULT_AFS_TUNE;
        break;
    case AFS_PRESET_TRIPLE: //動き重視
        method_switch = 0;
        coeff_shift   = 192;
        thre_shift    = 128;
        thre_deint    = 48;
        thre_Ymotion  = 112;
        thre_Cmotion  = 224;
        analyze       = 1;
        shift         = false;
        drop          = false;
        smooth        = false;
        force24       = false;
        tune          = false;
        break;
    case AFS_PRESET_DOUBLE://二重化
        method_switch = 0;
        coeff_shift   = 192;
        thre_shift    = 128;
        thre_deint    = 48;
        thre_Ymotion  = 112;
        thre_Cmotion  = 224;
        analyze       = 2;
        shift         = true;
        drop          = true;
        smooth        = true;
        force24       = false;
        tune          = false;
        break;
    case AFS_PRESET_ANIME: //映画/アニメ
        method_switch = 64;
        coeff_shift   = 128;
        thre_shift    = 128;
        thre_deint    = 48;
        thre_Ymotion  = 112;
        thre_Cmotion  = 224;
        analyze       = 3;
        shift         = true;
        drop          = true;
        smooth        = true;
        force24       = false;
        tune          = false;
        break;
    case AFS_PRESET_MIN_AFTERIMG:      //残像最小化
        method_switch = 0;
        coeff_shift   = 192;
        thre_shift    = 128;
        thre_deint    = 48;
        thre_Ymotion  = 112;
        thre_Cmotion  = 224;
        analyze       = 4;
        shift         = true;
        drop          = true;
        smooth        = true;
        force24       = false;
        tune          = false;
        break;
    case AFS_PRESET_FORCE24_SD:        //24fps固定
        method_switch = 64;
        coeff_shift   = 128;
        thre_shift    = 128;
        thre_deint    = 48;
        thre_Ymotion  = 112;
        thre_Cmotion  = 224;
        analyze       = 3;
        shift         = true;
        drop          = true;
        smooth        = false;
        force24       = true;
        tune          = false;
        break;
    case AFS_PRESET_FORCE24_HD:        //24fps固定 (HD)
        method_switch = 92;
        coeff_shift   = 192;
        thre_shift    = 448;
        thre_deint    = 48;
        thre_Ymotion  = 112;
        thre_Cmotion  = 224;
        analyze       = 3;
        shift         = true;
        drop          = true;
        smooth        = true;
        force24       = true;
        tune          = false;
        break;
    case AFS_PRESET_FORCE30:           //30fps固定
        method_switch = 92;
        coeff_shift   = 192;
        thre_shift    = 448;
        thre_deint    = 48;
        thre_Ymotion  = 112;
        thre_Cmotion  = 224;
        analyze       = 3;
        shift         = false;
        drop          = false;
        smooth        = false;
        force24       = false;
        tune          = false;
        break;
    default:
        break;
    }
}

int VppAfs::read_afs_inifile(const TCHAR *inifile) {
    if (!rgy_file_exists(inifile)) {
        return 1;
    }
#if ENABLE_VPP_FILTER_AFS
    const auto filename = tchar_to_string(inifile);
    const auto section = AFS_STG_SECTION;

    clip.top      = GetPrivateProfileIntA(section, AFS_STG_UP, clip.top, filename.c_str());
    clip.bottom   = GetPrivateProfileIntA(section, AFS_STG_BOTTOM, clip.bottom, filename.c_str());
    clip.left     = GetPrivateProfileIntA(section, AFS_STG_LEFT, clip.left, filename.c_str());
    clip.right    = GetPrivateProfileIntA(section, AFS_STG_RIGHT, clip.right, filename.c_str());
    method_switch = GetPrivateProfileIntA(section, AFS_STG_METHOD_WATERSHED, method_switch, filename.c_str());
    coeff_shift   = GetPrivateProfileIntA(section, AFS_STG_COEFF_SHIFT, coeff_shift, filename.c_str());
    thre_shift    = GetPrivateProfileIntA(section, AFS_STG_THRE_SHIFT, thre_shift, filename.c_str());
    thre_deint    = GetPrivateProfileIntA(section, AFS_STG_THRE_DEINT, thre_deint, filename.c_str());
    thre_Ymotion  = GetPrivateProfileIntA(section, AFS_STG_THRE_Y_MOTION, thre_Ymotion, filename.c_str());
    thre_Cmotion  = GetPrivateProfileIntA(section, AFS_STG_THRE_C_MOTION, thre_Cmotion, filename.c_str());
    analyze       = GetPrivateProfileIntA(section, AFS_STG_MODE, analyze, filename.c_str());

    shift    = 0 != GetPrivateProfileIntA(section, AFS_STG_FIELD_SHIFT, shift, filename.c_str());
    drop     = 0 != GetPrivateProfileIntA(section, AFS_STG_DROP, drop, filename.c_str());
    smooth   = 0 != GetPrivateProfileIntA(section, AFS_STG_SMOOTH, smooth, filename.c_str());
    force24  = 0 != GetPrivateProfileIntA(section, AFS_STG_FORCE24, force24, filename.c_str());
    rff      = 0 != GetPrivateProfileIntA(section, AFS_STG_RFF, rff, filename.c_str());
    log      = 0 != GetPrivateProfileIntA(section, AFS_STG_LOG, log, filename.c_str());
    // GetPrivateProfileIntA(section, AFS_STG_DETECT_SC, fp->check[4], filename.c_str());
    tune     = 0 != GetPrivateProfileIntA(section, AFS_STG_TUNE_MODE, tune, filename.c_str());
    // GetPrivateProfileIntA(section, AFS_STG_LOG_SAVE, fp->check[6], filename.c_str());
    // GetPrivateProfileIntA(section, AFS_STG_TRACE_MODE, fp->check[7], filename.c_str());
    // GetPrivateProfileIntA(section, AFS_STG_REPLAY_MODE, fp->check[8], filename.c_str());
    // GetPrivateProfileIntA(section, AFS_STG_YUY2UPSAMPLE, fp->check[9], filename.c_str());
    // GetPrivateProfileIntA(section, AFS_STG_THROUGH_MODE, fp->check[10], filename.c_str());

    // GetPrivateProfileIntA(section, AFS_STG_PROC_MODE, g_afs.ex_data.proc_mode, filename.c_str());
    return 0;
#else
    return 1;
#endif
}

tstring VppAfs::print() const {
#define ON_OFF(b) ((b) ? _T("on") : _T("off"))
    return strsprintf(
        _T("afs: clip(T %d, B %d, L %d, R %d), switch %d, coeff_shift %d\n")
        _T("                    thre(shift %d, deint %d, Ymotion %d, Cmotion %d)\n")
        _T("                    level %d, shift %s, drop %s, smooth %s, force24 %s\n")
        _T("                    tune %s, tb_order %d(%s), rff %s, timecode %s, log %s"),
        clip.top, clip.bottom, clip.left, clip.right,
        method_switch, coeff_shift,
        thre_shift, thre_deint, thre_Ymotion, thre_Cmotion,
        analyze, ON_OFF(shift), ON_OFF(drop), ON_OFF(smooth), ON_OFF(force24),
        ON_OFF(tune), tb_order, tb_order ? _T("tff") : _T("bff"), ON_OFF(rff), ON_OFF(timecode), ON_OFF(log));
#undef ON_OFF
}

VppNnedi::VppNnedi() :
    enable(false),
    field(VPP_NNEDI_FIELD_USE_AUTO),
    nns(32),
    nsize(VPP_NNEDI_NSIZE_32x4),
    quality(VPP_NNEDI_QUALITY_FAST),
    precision(VPP_FP_PRECISION_AUTO),
    pre_screen(VPP_NNEDI_PRE_SCREEN_NEW_BLOCK),
    errortype(VPP_NNEDI_ETYPE_ABS),
    weightfile(_T("")) {

}

bool VppNnedi::isbob() {
    return field == VPP_NNEDI_FIELD_BOB_AUTO
        || field == VPP_NNEDI_FIELD_BOB_BOTTOM_TOP
        || field == VPP_NNEDI_FIELD_BOB_TOP_BOTTOM;
}

bool VppNnedi::operator==(const VppNnedi &x) const {
    return enable == x.enable
        && field == x.field
        && nns == x.nns
        && nsize == x.nsize
        && quality == x.quality
        && pre_screen == x.pre_screen
        && errortype == x.errortype
        && precision == x.precision
        && weightfile == x.weightfile;
}
bool VppNnedi::operator!=(const VppNnedi &x) const {
    return !(*this == x);
}

tstring VppNnedi::print() const {
    return strsprintf(
        _T("nnedi: field %s, nns %d, nsize %s, quality %s, prec %s\n")
        _T("                       pre_screen %s, errortype %s, weight \"%s\""),
        get_cx_desc(list_vpp_nnedi_field, field),
        nns,
        get_cx_desc(list_vpp_nnedi_nsize, nsize),
        get_cx_desc(list_vpp_nnedi_quality, quality),
        get_cx_desc(list_vpp_fp_prec, precision),
        get_cx_desc(list_vpp_nnedi_pre_screen, pre_screen),
        get_cx_desc(list_vpp_nnedi_error_type, errortype),
        ((weightfile.length()) ? weightfile.c_str() : _T("internal")));
}

VppYadif::VppYadif() :
    enable(false),
    mode(VPP_YADIF_MODE_AUTO) {

}

bool VppYadif::operator==(const VppYadif& x) const {
    return enable == x.enable
        && mode == x.mode;
}
bool VppYadif::operator!=(const VppYadif& x) const {
    return !(*this == x);
}

tstring VppYadif::print() const {
    return strsprintf(
        _T("yadif: mode %s"),
        get_cx_desc(list_vpp_yadif_mode, mode));
}

VppDecomb::VppDecomb() :
    enable(false),
    full(FILTER_DEFAULT_DECOMB_FULL),
    threshold(FILTER_DEFAULT_DECOMB_THRESHOLD),
    dthreshold(FILTER_DEFAULT_DECOMB_DTHRESHOLD),
    blend(FILTER_DEFAULT_DECOMB_BLEND) {

}

bool VppDecomb::operator==(const VppDecomb& x) const {
    return enable == x.enable
        && full == x.full
        && threshold == x.threshold
        && dthreshold == x.dthreshold
        && blend == x.blend;
}
bool VppDecomb::operator!=(const VppDecomb& x) const {
    return !(*this == x);
}

tstring VppDecomb::print() const {
    return strsprintf(
        _T("decomb: full %s, threshold %d, dthreshold %d, blend %s"),
        full ? _T("on") : _T("off"),
        threshold, dthreshold,
        blend ? _T("on") : _T("off"));
}

VppSelectEvery::VppSelectEvery() :
    enable(false),
    step(1),
    offset(0) {
}

bool VppSelectEvery::operator==(const VppSelectEvery& x) const {
    return enable == x.enable
        && step == x.step
        && offset == x.offset;
}
bool VppSelectEvery::operator!=(const VppSelectEvery& x) const {
    return !(*this == x);
}

tstring VppSelectEvery::print() const {
    return strsprintf(_T("selectevery %d (offset %d)"), step, offset);
}

VppDecimate::VppDecimate() :
    enable(false),
    cycle(FILTER_DEFAULT_DECIMATE_CYCLE),
    drop(FILTER_DEFAULT_DECIMATE_DROP),
    threDuplicate(FILTER_DEFAULT_DECIMATE_THRE_DUP),
    threSceneChange(FILTER_DEFAULT_DECIMATE_THRE_SC),
    blockX(FILTER_DEFAULT_DECIMATE_BLOCK_X),
    blockY(FILTER_DEFAULT_DECIMATE_BLOCK_Y),
    preProcessed(FILTER_DEFAULT_DECIMATE_PREPROCESSED),
    chroma(FILTER_DEFAULT_DECIMATE_CHROMA),
    log(FILTER_DEFAULT_DECIMATE_LOG) {

}

bool VppDecimate::operator==(const VppDecimate& x) const {
    return enable == x.enable
        && cycle == x.cycle
        && threDuplicate == x.threDuplicate
        && threSceneChange == x.threSceneChange
        && blockX == x.blockX
        && blockY == x.blockY
        && preProcessed == x.preProcessed
        && chroma == x.chroma
        && log == x.log;
}
bool VppDecimate::operator!=(const VppDecimate& x) const {
    return !(*this == x);
}

tstring VppDecimate::print() const {
    return strsprintf(_T("decimate: cycle %d, drop %d, threDup %.2f, threSC %.2f\n")
        _T("                         block %dx%d, chroma %s, log %s"),
        cycle, drop,
        threDuplicate, threSceneChange,
        blockX, blockY,
        /*preProcessed ? _T("on") : _T("off"),*/
        chroma ? _T("on") : _T("off"),
        log ? _T("on") : _T("off"));
}


VppMpdecimate::VppMpdecimate() :
    enable(false),
    lo(FILTER_DEFAULT_MPDECIMATE_LO),
    hi(FILTER_DEFAULT_MPDECIMATE_HI),
    max(FILTER_DEFAULT_MPDECIMATE_MAX),
    frac(FILTER_DEFAULT_MPDECIMATE_FRAC),
    log(FILTER_DEFAULT_MPDECIMATE_LOG) {

}

bool VppMpdecimate::operator==(const VppMpdecimate& x) const {
    return enable == x.enable
        && lo == x.lo
        && hi == x.hi
        && max == x.max
        && frac == x.frac
        && log == x.log;
}
bool VppMpdecimate::operator!=(const VppMpdecimate& x) const {
    return !(*this == x);
}

tstring VppMpdecimate::print() const {
    return strsprintf(_T("mpdecimate: hi %d, lo %d, frac %.2f, max %d, log %s"),
        hi, lo, frac, max,
        log ? _T("on") : _T("off"));
}

VppPad::VppPad() :
    enable(false),
    left(0),
    top(0),
    right(0),
    bottom(0) {

}

bool VppPad::operator==(const VppPad& x) const {
    return enable == x.enable
        && left == x.left
        && top == x.top
        && right == x.right
        && bottom == x.bottom;
}
bool VppPad::operator!=(const VppPad& x) const {
    return !(*this == x);
}

tstring VppPad::print() const {
    return strsprintf(_T("(right=%d, left=%d, top=%d, bottom=%d)"),
        right, left, top, bottom);
}

VppKnn::VppKnn() :
    enable(false),
    radius(FILTER_DEFAULT_KNN_RADIUS),
    strength(FILTER_DEFAULT_KNN_STRENGTH),
    lerpC(FILTER_DEFAULT_KNN_LERPC),
    weight_threshold(FILTER_DEFAULT_KNN_WEIGHT_THRESHOLD),
    lerp_threshold(FILTER_DEFAULT_KNN_LERPC_THRESHOLD) {
}

bool VppKnn::operator==(const VppKnn &x) const {
    return enable == x.enable
        && radius == x.radius
        && strength == x.strength
        && lerpC == x.lerpC
        && weight_threshold == x.weight_threshold
        && lerp_threshold == x.lerp_threshold;
}
bool VppKnn::operator!=(const VppKnn &x) const {
    return !(*this == x);
}

tstring VppKnn::print() const {
    return strsprintf(
        _T("denoise(knn): radius %d, strength %.2f, lerp %.2f\n")
        _T("                              th_weight %.2f, th_lerp %.2f"),
        radius, strength, lerpC,
        weight_threshold, lerp_threshold);
}

VppNLMeans::VppNLMeans() :
    enable(false),
    sigma(FILTER_DEFAULT_NLMEANS_FILTER_SIGMA),
    patchSize(FILTER_DEFAULT_NLMEANS_PATCH_SIZE),
    searchSize(FILTER_DEFAULT_NLMEANS_SEARCH_SIZE),
    h(FILTER_DEFAULT_NLMEANS_H),
    fp16(VppNLMeansFP16Opt::BlockDiff),
    sharedMem(true) {
}

bool VppNLMeans::operator==(const VppNLMeans &x) const {
    return enable == x.enable
        && sigma == x.sigma
        && patchSize == x.patchSize
        && searchSize == x.searchSize
        && h == x.h
        && fp16 == x.fp16
        && sharedMem == x.sharedMem;
}
bool VppNLMeans::operator!=(const VppNLMeans &x) const {
    return !(*this == x);
}

tstring VppNLMeans::print() const {
    return strsprintf(
        _T("denoise(nlmeans): sigma %.3f, h %.3f, patch %d, search %d, fp16 %s"),
        sigma, h, patchSize, searchSize, get_cx_desc(list_vpp_nlmeans_fp16, fp16));
}

VppPmd::VppPmd() :
    enable(false),
    strength(FILTER_DEFAULT_PMD_STRENGTH),
    threshold(FILTER_DEFAULT_PMD_THRESHOLD),
    applyCount(FILTER_DEFAULT_PMD_APPLY_COUNT),
    useExp(FILTER_DEFAULT_PMD_USE_EXP) {

}

bool VppPmd::operator==(const VppPmd& x) const {
    return enable == x.enable
        && strength == x.strength
        && threshold == x.threshold
        && applyCount == x.applyCount
        && useExp == x.useExp;
}
bool VppPmd::operator!=(const VppPmd& x) const {
    return !(*this == x);
}

tstring VppPmd::print() const {
    return strsprintf(_T("denoise(pmd): strength %d, threshold %d, apply %d, exp %d"),
        (int)strength, (int)threshold, applyCount, useExp);
}

VppSmooth::VppSmooth() :
    enable(false),
    quality(FILTER_DEFAULT_SMOOTH_QUALITY),
    qp(FILTER_DEFAULT_SMOOTH_QP),
    prec(VPP_FP_PRECISION_AUTO),
    useQPTable(false),
    strength(FILTER_DEFAULT_SMOOTH_STRENGTH),
    threshold(FILTER_DEFAULT_SMOOTH_THRESHOLD),
    bratio(FILTER_DEFAULT_SMOOTH_B_RATIO),
    maxQPTableErrCount(FILTER_DEFAULT_SMOOTH_MAX_QPTABLE_ERR) {

}

bool VppSmooth::operator==(const VppSmooth &x) const {
    return enable == x.enable
        && quality == x.quality
        && qp == x.qp
        && prec == x.prec
        && useQPTable == x.useQPTable
        && strength == x.strength
        && threshold == x.threshold
        && bratio == x.bratio
        && maxQPTableErrCount == x.maxQPTableErrCount;
}
bool VppSmooth::operator!=(const VppSmooth &x) const {
    return !(*this == x);
}

tstring VppSmooth::print() const {
    //return strsprintf(_T("smooth: quality %d, qp %d, threshold %.1f, strength %.1f, mode %d, use_bframe_qp %s"), quality, qp, threshold, strength, mode, use_bframe_qp ? _T("yes") : _T("no"));
    tstring str = strsprintf(_T("smooth: quality %d, qp %d, prec %s"), quality, qp, get_cx_desc(list_vpp_fp_prec, prec));
    if (useQPTable) {
        str += strsprintf(_T(", use QP table on"));
    }
    return str;
}

VppDenoiseDct::VppDenoiseDct() :
    enable(false),
    sigma(FILTER_DEFAULT_DENOISE_DCT_SIGMA),
    step(FILTER_DEFAULT_DENOISE_DCT_STEP),
    block_size(FILTER_DEFAULT_DENOISE_DCT_BLOCK_SIZE) {

}

bool VppDenoiseDct::operator==(const VppDenoiseDct &x) const {
    return enable == x.enable
        && sigma == x.sigma
        && step == x.step
        && block_size == x.block_size;
}
bool VppDenoiseDct::operator!=(const VppDenoiseDct &x) const {
    return !(*this == x);
}

tstring VppDenoiseDct::print() const {
    tstring str = strsprintf(_T("denoise-dct: sigma %.2f, step %d, block_size %d"), sigma, step, block_size);
    return str;
}

VppDenoiseFFT3D::VppDenoiseFFT3D() :
    enable(false),
    sigma(FILTER_DEFAULT_DENOISE_FFT3D_SIGMA),
    amount(FILTER_DEFAULT_DENOISE_FFT3D_AMOUNT),
    block_size(FILTER_DEFAULT_DENOISE_FFT3D_BLOCK_SIZE),
    overlap(FILTER_DEFAULT_DENOISE_FFT3D_OVERLAP),
    overlap2(FILTER_DEFAULT_DENOISE_FFT3D_OVERLAP2),
    method(FILTER_DEFAULT_DENOISE_FFT3D_METHOD),
    temporal(FILTER_DEFAULT_DENOISE_FFT3D_TEMPORAL),
    precision(VppFpPrecision::VPP_FP_PRECISION_AUTO) {

}

bool VppDenoiseFFT3D::operator==(const VppDenoiseFFT3D &x) const {
    return enable == x.enable
        && sigma == x.sigma
        && amount == x.amount
        && block_size == x.block_size
        && overlap == x.overlap
        && overlap2 == x.overlap2
        && method == x.method
        && temporal == x.temporal
        && precision == x.precision;
}
bool VppDenoiseFFT3D::operator!=(const VppDenoiseFFT3D &x) const {
    return !(*this == x);
}

tstring VppDenoiseFFT3D::print() const {
    tstring str = strsprintf(_T("denoise-fft3d: sigma %.2f, strength %.2f, block_size %d\n"
        "                         overlap %.2f:%.2f, method %d, temporal %d, precision %s"),
        sigma, amount, block_size, overlap, overlap2, method, temporal, get_cx_desc(list_vpp_fp_prec, precision));
    return str;
}

VppConvolution3d::VppConvolution3d() :
    enable(false),
    fast(false),
    matrix(VppConvolution3dMatrix::Standard),
    threshYspatial(FILTER_DEFAULT_CONVOLUTION3D_THRESH_Y_SPATIAL),
    threshCspatial(FILTER_DEFAULT_CONVOLUTION3D_THRESH_C_SPATIAL),
    threshYtemporal(FILTER_DEFAULT_CONVOLUTION3D_THRESH_Y_TEMPORAL),
    threshCtemporal(FILTER_DEFAULT_CONVOLUTION3D_THRESH_C_TEMPORAL) {

}

bool VppConvolution3d::operator==(const VppConvolution3d &x) const {
    return enable == x.enable
        && fast == x.fast
        && matrix == x.matrix
        && threshYspatial == x.threshYspatial
        && threshCspatial == x.threshCspatial
        && threshYtemporal == x.threshYtemporal
        && threshCtemporal == x.threshCtemporal;
}
bool VppConvolution3d::operator!=(const VppConvolution3d &x) const {
    return !(*this == x);
}

tstring VppConvolution3d::print() const {
    tstring str = strsprintf(_T("convolution3d: matrix %s, mode %s\n")
        _T("                       threshold spatial luma %d, chroma %d, temporal luma %d, chroma %d"),
        get_cx_desc(list_vpp_convolution3d_matrix, (int)matrix),
        fast ? _T("fast") : _T("normal"),
        threshYspatial, threshCspatial, threshYtemporal, threshCtemporal);
    return str;
}

VppSubburn::VppSubburn() :
    enable(false),
    filename(),
    charcode(),
    fontsdir(),
    trackId(0),
    assShaping(1),
    scale(0.0),
    transparency_offset(0.0),
    brightness(FILTER_DEFAULT_TWEAK_BRIGHTNESS),
    contrast(FILTER_DEFAULT_TWEAK_CONTRAST),
    ts_offset(0.0),
    vid_ts_offset(true),
    forced_subs_only(false) {
}

bool VppSubburn::operator==(const VppSubburn &x) const {
    return enable == x.enable
        && filename == x.filename
        && charcode == x.charcode
        && fontsdir == x.fontsdir
        && trackId == x.trackId
        && assShaping == x.assShaping
        && scale == x.scale
        && transparency_offset == x.transparency_offset
        && brightness == x.brightness
        && contrast == x.contrast
        && ts_offset == x.ts_offset
        && vid_ts_offset == x.vid_ts_offset
        && forced_subs_only == x.forced_subs_only;
}
bool VppSubburn::operator!=(const VppSubburn &x) const {
    return !(*this == x);
}

tstring VppSubburn::print() const {
    tstring str = strsprintf(_T("subburn: %s, scale x%.2f"),
        (filename.length() > 0)
        ? filename.c_str()
        : strsprintf(_T("track #%d"), trackId).c_str(),
        scale);
    if (transparency_offset != 0.0) {
        str += strsprintf(_T(", transparency %.2f"), transparency_offset);
    }
    if (brightness != FILTER_DEFAULT_TWEAK_BRIGHTNESS) {
        str += strsprintf(_T(", brightness %.2f"), brightness);
    }
    if (contrast != FILTER_DEFAULT_TWEAK_CONTRAST) {
        str += strsprintf(_T(", contrast %.2f"), contrast);
    }
    if (ts_offset != 0.0) {
        str += strsprintf(_T(", ts_offset %.2f"), ts_offset);
    }
    if (!vid_ts_offset) {
        str += _T(", vid_ts_offset off");
    }
    if (forced_subs_only) {
        str += _T(", forced_subs_only");
    }
    return str;
}

VppUnsharp::VppUnsharp() :
    enable(false),
    radius(FILTER_DEFAULT_UNSHARP_RADIUS),
    weight(FILTER_DEFAULT_UNSHARP_WEIGHT),
    threshold(FILTER_DEFAULT_UNSHARP_THRESHOLD) {

}

bool VppUnsharp::operator==(const VppUnsharp &x) const {
    return enable == x.enable
        && radius == x.radius
        && weight == x.weight
        && threshold == x.threshold;
}
bool VppUnsharp::operator!=(const VppUnsharp &x) const {
    return !(*this == x);
}

tstring VppUnsharp::print() const {
    return strsprintf(_T("unsharp: radius %d, weight %.1f, threshold %.1f"),
        radius, weight, threshold);
}

VppEdgelevel::VppEdgelevel() :
    enable(false),
    strength(FILTER_DEFAULT_EDGELEVEL_STRENGTH),
    threshold(FILTER_DEFAULT_EDGELEVEL_THRESHOLD),
    black(FILTER_DEFAULT_EDGELEVEL_BLACK),
    white(FILTER_DEFAULT_EDGELEVEL_WHITE) {
}

bool VppEdgelevel::operator==(const VppEdgelevel &x) const {
    return enable == x.enable
        && strength == x.strength
        && threshold == x.threshold
        && black == x.black
        && white == x.white;
}
bool VppEdgelevel::operator!=(const VppEdgelevel &x) const {
    return !(*this == x);
}

tstring VppEdgelevel::print() const {
    return strsprintf(_T("edgelevel: strength %.1f, threshold %.1f, black %.1f, white %.1f"),
        strength, threshold, black, white);
}

VppWarpsharp::VppWarpsharp() :
    enable(false),
    threshold(FILTER_DEFAULT_WARPSHARP_THRESHOLD),
    blur(FILTER_DEFAULT_WARPSHARP_BLUR),
    type(FILTER_DEFAULT_WARPSHARP_TYPE),
    depth(FILTER_DEFAULT_WARPSHARP_DEPTH),
    chroma(FILTER_DEFAULT_WARPSHARP_CHROMA) {
}

bool VppWarpsharp::operator==(const VppWarpsharp& x) const {
    return enable == x.enable
        && threshold == x.threshold
        && blur == x.blur
        && type == x.type
        && depth == x.depth
        && chroma == x.chroma;
}
bool VppWarpsharp::operator!=(const VppWarpsharp& x) const {
    return !(*this == x);
}

tstring VppWarpsharp::print() const {
    return strsprintf(_T("warpsharp: threshold %.1f, blur %d, type %d, depth %.1f, chroma %d"),
        threshold, blur, type, depth, chroma);
}

VppTweak::VppTweak() :
    enable(false),
    brightness(FILTER_DEFAULT_TWEAK_BRIGHTNESS),
    contrast(FILTER_DEFAULT_TWEAK_CONTRAST),
    gamma(FILTER_DEFAULT_TWEAK_GAMMA),
    saturation(FILTER_DEFAULT_TWEAK_SATURATION),
    hue(FILTER_DEFAULT_TWEAK_HUE),
    swapuv(false) {
}

bool VppTweak::operator==(const VppTweak &x) const {
    return enable == x.enable
        && brightness == x.brightness
        && contrast == x.contrast
        && gamma == x.gamma
        && saturation == x.saturation
        && hue == x.hue
        && swapuv == x.swapuv;
}
bool VppTweak::operator!=(const VppTweak &x) const {
    return !(*this == x);
}

tstring VppTweak::print() const {
    return strsprintf(_T("tweak: brightness %.2f, contrast %.2f, saturation %.2f, gamma %.2f, hue %.2f, swapuv %s"),
        brightness, contrast, saturation, gamma, hue, swapuv ? _T("on") : _T("off"));
}

VppCurveParams::VppCurveParams() : r(), g(), b(), m() {};
VppCurveParams::VppCurveParams(const tstring& r_, const tstring& g_, const tstring& b_, const tstring& m_) :
    r(r_), g(g_), b(b_), m(m_) {};

bool VppCurveParams::operator==(const VppCurveParams &x) const {
    return r == x.r
        && g == x.g
        && b == x.b
        && m == x.m;

}
bool VppCurveParams::operator!=(const VppCurveParams &x) const {
    return !(*this == x);
}

VppCurves::VppCurves() :
    enable(false),
    preset(VppCurvesPreset::NONE),
    prm(),
    all() {
}

bool VppCurves::operator==(const VppCurves &x) const {
    return enable == x.enable
        && preset == x.preset
        && prm == x.prm
        && all == x.all;
}
bool VppCurves::operator!=(const VppCurves &x) const {
    return !(*this == x);
}

tstring VppCurves::print() const {
    tstring str    = _T("curves: ");
    tstring indent = _T("                               ");
    if (preset != VppCurvesPreset::NONE) str += tstring(_T("preset ")) + get_cx_desc(list_vpp_curves_preset, (int)preset);
    if (prm.r.length() > 0) str += _T("\n") + indent + _T("r ") + prm.r;
    if (prm.g.length() > 0) str += _T("\n") + indent + _T("g ") + prm.g;
    if (prm.b.length() > 0) str += _T("\n") + indent + _T("b ") + prm.b;
    if (prm.m.length() > 0) str += _T("\n") + indent + _T("master ") + prm.m;
    if (all.length() > 0)   str += _T("\n") + indent + _T("all ") + all;
    return str;
}

VppTransform::VppTransform() :
    enable(false),
    transpose(false),
    flipX(false),
    flipY(false) {
}

int VppTransform::rotate() const {
    if (transpose) {
        if (!flipY && flipX) {
            return 270;
        } else if (flipY && !flipX) {
            return 90;
        }
    } else if (flipY && flipX) {
        return 180;
    }
    return 0;
}

bool VppTransform::setRotate(int rotate) {
    switch (rotate) {
    case 90:
        transpose = true;
        flipY = true;
        break;
    case 180:
        flipX = true;
        flipY = true;
        break;
    case 270:
        transpose = true;
        flipX = true;
        break;
    default:
        return false;
    }
    return true;
}

bool VppTransform::operator==(const VppTransform &x) const {
    return enable == x.enable
        && transpose == x.transpose
        && flipX == x.flipX
        && flipY == x.flipY;
}
bool VppTransform::operator!=(const VppTransform &x) const {
    return !(*this == x);
}

tstring VppTransform::print() const {
#define ON_OFF(b) ((b) ? _T("on") : _T("off"))
    const auto rotation = rotate();
    if (rotation) {
        return strsprintf(_T("rotate: %d"), rotation);
    } else {
        return strsprintf(_T("transform: transpose %s, flipX %s, flipY %s"),
            ON_OFF(transpose), ON_OFF(flipX), ON_OFF(flipY));
    }
#undef ON_OFF
}

VppOverlayAlphaKey::VppOverlayAlphaKey() :
    threshold(0.0f),
    tolerance(0.1f),
    shoftness(0.0f) {

}

bool VppOverlayAlphaKey::operator==(const VppOverlayAlphaKey &x) const {
    return threshold == x.threshold
        && tolerance == x.tolerance
        && shoftness == x.shoftness;
}
bool VppOverlayAlphaKey::operator!=(const VppOverlayAlphaKey &x) const {
    return !(*this == x);
}

tstring VppOverlayAlphaKey::print() const {
    return strsprintf(_T("threshold %.2f, tolerance %.2f, shoftness %.2f"),
        threshold, tolerance, shoftness);
}

VppOverlay::VppOverlay() :
    enable(false),
    inputFile(),
    posX(0),
    posY(0),
    width(0),
    height(0),
    alpha(0.0f),
    alphaMode(VppOverlayAlphaMode::Override),
    lumaKey(),
    loop(false) {

}

bool VppOverlay::operator==(const VppOverlay &x) const {
    return enable == x.enable
        && inputFile == x.inputFile
        && posX == x.posX
        && posY == x.posY
        && width == x.width
        && height == x.height
        && alpha == x.alpha
        && alphaMode == x.alphaMode
        && lumaKey == x.lumaKey
        && loop == x.loop;
}
bool VppOverlay::operator!=(const VppOverlay &x) const {
    return !(*this == x);
}

tstring VppOverlay::print() const {
    tstring alphaStr = _T("auto");
    if (alphaMode == VppOverlayAlphaMode::LumaKey) {
        alphaStr = (alpha > 0.0f) ? strsprintf(_T("%.2f "), alpha) : _T("");
        alphaStr += _T("lumakey ") + lumaKey.print();
    } else {
        if (alpha > 0.0f) {
            switch (alphaMode) {
            case VppOverlayAlphaMode::Override:
                alphaStr = strsprintf(_T("%.2f"), alpha);
                break;
            case VppOverlayAlphaMode::Mul:
                alphaStr = strsprintf(_T("*%.2f"), alpha);
                break;
            default:
                break;
            }
        }
    }
    return strsprintf(_T("overlay: %s\n")
        _T("                        pos (%d,%d), size %dx%d, loop %s\n")
        _T("                        alpha %s"),
        inputFile.c_str(),
        posX, posY,
        width, height,
        (loop) ? _T("on") : _T("off"),
        alphaStr.c_str());
}

VppDeband::VppDeband() :
    enable(false),
    range(FILTER_DEFAULT_DEBAND_RANGE),
    threY(FILTER_DEFAULT_DEBAND_THRE_Y),
    threCb(FILTER_DEFAULT_DEBAND_THRE_CB),
    threCr(FILTER_DEFAULT_DEBAND_THRE_CR),
    ditherY(FILTER_DEFAULT_DEBAND_DITHER_Y),
    ditherC(FILTER_DEFAULT_DEBAND_DITHER_C),
    sample(FILTER_DEFAULT_DEBAND_MODE),
    seed(FILTER_DEFAULT_DEBAND_SEED),
    blurFirst(FILTER_DEFAULT_DEBAND_BLUR_FIRST),
    randEachFrame(FILTER_DEFAULT_DEBAND_RAND_EACH_FRAME) {

}

bool VppDeband::operator==(const VppDeband &x) const {
    return enable == x.enable
        && range == x.range
        && threY == x.threY
        && threCb == x.threCb
        && threCr == x.threCr
        && ditherY == x.ditherY
        && ditherC == x.ditherC
        && sample == x.sample
        && seed == x.seed
        && blurFirst == x.blurFirst
        && randEachFrame == x.randEachFrame;
}
bool VppDeband::operator!=(const VppDeband &x) const {
    return !(*this == x);
}

tstring VppDeband::print() const {
    return strsprintf(_T("deband: mode %d, range %d, threY %d, threCb %d, threCr %d\n")
        _T("                       ditherY %d, ditherC %d, blurFirst %s, randEachFrame %s"),
        sample, range,
        threY, threCb, threCr,
        ditherY, ditherC,
        blurFirst ? _T("yes") : _T("no"),
        randEachFrame ? _T("yes") : _T("no"));
}

VppFruc::VppFruc() :
    enable(false),
    mode(VppFrucMode::Disabled),
    targetFps() {

}

bool VppFruc::operator==(const VppFruc &x) const {
    return enable == x.enable
        && mode == x.mode
        && targetFps == x.targetFps;
}
bool VppFruc::operator!=(const VppFruc &x) const {
    return !(*this == x);
}

tstring VppFruc::print() const {
    if (mode == VppFrucMode::NVOFFRUCx2) {
        return _T("nvof-fruc: double frames");
    } else if (mode == VppFrucMode::NVOFFRUCFps) {
        return strsprintf(_T("nvof-fruc: %.3f(%d/%d) fps"), targetFps.qdouble(), targetFps.n(), targetFps.d());
    } else {
        return _T("Unknown");
    }
}

RGYParamVpp::RGYParamVpp() :
    filterOrder(),
    resize_algo(RGY_VPP_RESIZE_AUTO),
    resize_mode(RGY_VPP_RESIZE_MODE_DEFAULT),
    colorspace(),
    delogo(),
    afs(),
    nnedi(),
    yadif(),
    decomb(),
    rff(),
    selectevery(),
    decimate(),
    mpdecimate(),
    pad(),
    convolution3d(),
    knn(),
    nlmeans(),
    pmd(),
    dct(),
    smooth(),
    fft3d(),
    subburn(),
    unsharp(),
    edgelevel(),
    warpsharp(),
    curves(),
    tweak(),
    transform(),
    deband(),
    overlay(),
    fruc(),
    checkPerformance(false) {

}

bool RGYParamVpp::operator==(const RGYParamVpp& x) const {
    return resize_algo == x.resize_algo
        && resize_mode == x.resize_mode
        && colorspace == x.colorspace
        && delogo == x.delogo
        && afs == x.afs
        && nnedi == x.nnedi
        && yadif == x.yadif
        && decomb == x.decomb
        && rff == x.rff
        && selectevery == x.selectevery
        && decimate == x.decimate
        && mpdecimate == x.mpdecimate
        && pad == x.pad
        && convolution3d == x.convolution3d
        && knn == x.knn
        && nlmeans == x.nlmeans
        && pmd == x.pmd
        && dct == x.dct
        && smooth == x.smooth
        && subburn == x.subburn
        && unsharp == x.unsharp
        && edgelevel == x.edgelevel
        && warpsharp == x.warpsharp
        && curves == x.curves
        && tweak == x.tweak
        && transform == x.transform
        && deband == x.deband
        && overlay == x.overlay
        && checkPerformance == x.checkPerformance;
}
bool RGYParamVpp::operator!=(const RGYParamVpp& x) const {
    return !(*this == x);
}


AudioSelect::AudioSelect() :
    trackID(0),
    decCodecPrm(),
    encCodec(),
    encCodecPrm(),
    encCodecProfile(),
    encBitrate(0),
    encQuality({ false, RGY_AUDIO_QUALITY_DEFAULT }),
    encSamplingRate(0),
    addDelayMs(0.0),
    extractFilename(),
    extractFormat(),
    filter(),
    streamChannelSelect(),
    streamChannelOut(),
    bsf(),
    disposition(),
    lang(),
    selectCodec(),
    metadata(),
    resamplerPrm() {
}

AudioSource::AudioSource() :
    filename(),
    format(),
    inputOpt(),
    select() {

}

SubtitleSelect::SubtitleSelect() :
    trackID(0),
    encCodec(),
    encCodecPrm(),
    decCodecPrm(),
    asdata(false),
    bsf(),
    disposition(),
    lang(),
    selectCodec(),
    metadata() {

}

SubSource::SubSource() :
    filename(),
    format(),
    inputOpt(),
    select() {

}

DataSelect::DataSelect() :
    trackID(0),
    encCodec(),
    disposition(),
    lang(),
    selectCodec(),
    metadata() {

}

VMAFParam::VMAFParam() :
    enable(false),
    model(VMAF_DEFAULT_MODEL_VERSION),
    threads(0),
    subsample(1),
    phone_model(false),
    enable_transform(false) {
};

bool VMAFParam::operator==(const VMAFParam &x) const {
    return enable == x.enable
        && model == x.model
        && threads == x.threads
        && subsample == x.subsample
        && phone_model == x.phone_model
        && enable_transform == x.enable_transform;
}
bool VMAFParam::operator!=(const VMAFParam &x) const {
    return !(*this == x);
}
tstring VMAFParam::print() const {
    auto str = strsprintf(_T("vmaf %s, threads %d, subsample %d"),
        model.c_str(), threads, subsample);
    if (phone_model) {
        str += _T(", phone_model");
    }
    if (enable_transform) {
        str += _T(", transform");
    }
    return str;
}

RGYVideoQualityMetric::RGYVideoQualityMetric() :
    ssim(false),
    psnr(false),
    vmaf() {

}
bool RGYVideoQualityMetric::enabled() const {
    return ssim || psnr || vmaf.enable;
}
tstring RGYVideoQualityMetric::enabled_metric() const {
    if (!enabled()) return _T("none");
    tstring str;
    if (ssim) str += _T(",ssim");
    if (psnr) str += _T(",psnr");
    if (vmaf.enable) str += _T(",vmaf");
    return (str.length() > 0) ? str.substr(1) : _T("unknown");
}

GPUAutoSelectMul::GPUAutoSelectMul() : cores(0.001f), gen(1.0f), gpu(1.0f), ve(1.0f) {}

bool GPUAutoSelectMul::operator==(const GPUAutoSelectMul &x) const {
    return cores == x.cores
        && gen == x.gen
        && gpu == x.gpu
        && ve == x.ve;
}
bool GPUAutoSelectMul::operator!=(const GPUAutoSelectMul &x) const {
    return !(*this == x);
}

RGYDebugLogFile::RGYDebugLogFile() : enable(false), filename() {}

bool RGYDebugLogFile::operator==(const RGYDebugLogFile &x) const {
    return enable == x.enable
        && filename == x.filename;
}
bool RGYDebugLogFile::operator!=(const RGYDebugLogFile &x) const {
    return !(*this == x);
}
tstring RGYDebugLogFile::getFilename(const tstring& outputFilename, const tstring& defaultAppendix) const {
    if (!enable) return tstring();
    if (filename.length() > 0) {
        return filename;
    }
    return outputFilename + defaultAppendix;
}

RGYParamInput::RGYParamInput() :
    resizeResMode(RGYResizeResMode::Normal),
    ignoreSAR(false),
    avswDecoder() {

}

RGYParamInput::~RGYParamInput() {};

RGYParamCommon::RGYParamCommon() :
    inputFilename(),
    outputFilename(),
    muxOutputFormat(),
    out_vui(),
    inputOpt(),
    maxCll(),
    masterDisplay(),
    atcSei(RGY_TRANSFER_UNKNOWN),
    hdr10plusMetadataCopy(false),
    dynamicHdr10plusJson(),
    doviRpuFile(),
    doviProfile(0),
    videoCodecTag(),
    videoMetadata(),
    formatMetadata(),
    seekSec(0.0f),               //指定された秒数分先頭を飛ばす
    seekToSec(0.0f),
    nSubtitleSelectCount(0),
    ppSubtitleSelectList(nullptr),
    subSource(),
    audioSource(),
    nAudioSelectCount(0), //pAudioSelectの数
    ppAudioSelectList(nullptr),
    nDataSelectCount(0),
    ppDataSelectList(nullptr),
    nAttachmentSelectCount(0),
    ppAttachmentSelectList(nullptr),
    attachmentSource(),
    audioResampler(RGY_RESAMPLER_SWR),
    inputRetry(0),
    demuxAnalyzeSec(-1),
    demuxProbesize(-1),
    AVMuxTarget(RGY_MUX_NONE),                       //RGY_MUX_xxx
    videoTrack(0),
    videoStreamId(0),
    nTrimCount(0),
    pTrimList(nullptr),
    copyChapter(false),
    keyOnChapter(false),
    chapterNoTrim(false),
    audioIgnoreDecodeError(DEFAULT_IGNORE_DECODE_ERROR),
    videoIgnoreTimestampError(DEFAULT_VIDEO_IGNORE_TIMESTAMP_ERROR),
    muxOpt(),
    allowOtherNegativePts(false),
    disableMp4Opt(false),
    debugDirectAV1Out(false),
    debugRawOut(false),
    outReplayFile(),
    outReplayCodec(RGY_CODEC_UNKNOWN),
    chapterFile(),
    AVInputFormat(nullptr),
    AVSyncMode(RGY_AVSYNC_AUTO),     //avsyncの方法 (RGY_AVSYNC_xxx)
    timestampPassThrough(false),
    timecode(false),
    timecodeFile(),
    tcfileIn(),
    timebase({ 0, 0 }),
    hevcbsf(RGYHEVCBsf::INTERNAL),
    metric() {

}

RGYParamAvoidIdleClock::RGYParamAvoidIdleClock() :
    mode(RGYParamAvoidIdleClockMode::Auto),
    loadPercent(DEFAULT_DUMMY_LOAD_PERCENT) {
};

bool RGYParamAvoidIdleClock::operator==(const RGYParamAvoidIdleClock &x) const {
    return mode == x.mode
        && loadPercent == x.loadPercent;
}
bool RGYParamAvoidIdleClock::operator!=(const RGYParamAvoidIdleClock &x) const {
    return !(*this == x);
}

RGYParamCommon::~RGYParamCommon() {};

RGYParamControl::RGYParamControl() :
    threadCsp(0),
    simdCsp(RGY_SIMD::SIMD_ALL),
    logfile(),              //ログ出力先
    loglevel(RGY_LOG_INFO),                 //ログ出力レベル
    logAddTime(false),
    logFramePosList(),     //framePosList出力
    logPacketsList(),
    logMuxVidTs(),
    threadOutput(RGY_OUTPUT_THREAD_AUTO),
    threadAudio(RGY_AUDIO_THREAD_AUTO),
    threadInput(RGY_INPUT_THREAD_AUTO),
    threadParams(),
    procSpeedLimit(0),      //処理速度制限 (0で制限なし)
    taskPerfMonitor(false),   //タスクの処理時間を計測する
    perfMonitorSelect(0),
    perfMonitorSelectMatplot(0),
    perfMonitorInterval(RGY_DEFAULT_PERF_MONITOR_INTERVAL),
    parentProcessID(0),
    lowLatency(false),
    gpuSelect(),
    skipHWEncodeCheck(false),
    skipHWDecodeCheck(false),
    avsdll(),
    vsdir(),
    enableOpenCL(true),
    avoidIdleClock(),
    outputBufSizeMB(RGY_OUTPUT_BUF_MB_DEFAULT) {

}
RGYParamControl::~RGYParamControl() {};


bool trim_active(const sTrimParam *pTrim) {
    if (pTrim == nullptr) {
        return false;
    }
    if (pTrim->list.size() == 0) {
        return false;
    }
    if (pTrim->list[0].start == 0 && pTrim->list[0].fin == TRIM_MAX) {
        return false;
    }
    return true;
}

//block index (空白がtrimで削除された領域)
//       #0       #0         #1         #1       #2    #2
//   |        |----------|         |----------|     |------
std::pair<bool, int> frame_inside_range(int frame, const std::vector<sTrim> &trimList) {
    int index = 0;
    if (trimList.size() == 0) {
        return std::make_pair(true, index);
    }
    if (frame < 0) {
        return std::make_pair(false, index);
    }
    for (; index < (int)trimList.size(); index++) {
        if (frame < trimList[index].start) {
            return std::make_pair(false, index);
        }
        if (frame <= trimList[index].fin) {
            return std::make_pair(true, index);
        }
    }
    return std::make_pair(false, index);
}

bool rearrange_trim_list(int frame, int offset, std::vector<sTrim> &trimList) {
    if (trimList.size() == 0)
        return true;
    if (frame < 0)
        return false;
    for (uint32_t i = 0; i < trimList.size(); i++) {
        if (trimList[i].start >= frame) {
            trimList[i].start = clamp(trimList[i].start + offset, 0, TRIM_MAX);
        }
        if (trimList[i].fin && trimList[i].fin >= frame) {
            trimList[i].fin = (int)clamp((int64_t)trimList[i].fin + offset, 0, (int64_t)TRIM_MAX);
        }
    }
    return false;
}

tstring print_metadata(const std::vector<tstring> &metadata) {
    tstring str;
    for (const auto &m : metadata) {
        str += _T(" \"") + m + _T("\"");
    }
    return str;
}

bool metadata_copy(const std::vector<tstring> &metadata) {
    return std::find(metadata.begin(), metadata.end(), RGY_METADATA_COPY) != metadata.end();
}

bool metadata_clear(const std::vector<tstring> &metadata) {
    return std::find(metadata.begin(), metadata.end(), RGY_METADATA_CLEAR) != metadata.end();
}

#if !FOR_AUO
unique_ptr<RGYHDR10Plus> initDynamicHDR10Plus(const tstring &dynamicHdr10plusJson, shared_ptr<RGYLog> log) {
    unique_ptr<RGYHDR10Plus> hdr10plus;
    if (!rgy_file_exists(dynamicHdr10plusJson)) {
        log->write(RGY_LOG_ERROR, RGY_LOGT_HDR10PLUS, _T("Cannot find the file specified : %s.\n"), dynamicHdr10plusJson.c_str());
    } else {
        hdr10plus = std::unique_ptr<RGYHDR10Plus>(new RGYHDR10Plus());
        auto ret = hdr10plus->init(dynamicHdr10plusJson);
        if (ret == RGY_ERR_NOT_FOUND) {
            log->write(RGY_LOG_ERROR, RGY_LOGT_HDR10PLUS, _T("Cannot find \"%s\" required for --dhdr10-info.\n"), RGYHDR10Plus::HDR10PLUS_GEN_EXE_NAME);
            hdr10plus.reset();
        } else if (ret != RGY_ERR_NONE) {
            log->write(RGY_LOG_ERROR, RGY_LOGT_HDR10PLUS, _T("Failed to initialize hdr10plus reader: %s.\n"), get_err_mes((RGY_ERR)ret));
            hdr10plus.reset();
        }
        log->write(RGY_LOG_DEBUG, RGY_LOGT_HDR10PLUS, _T("initialized hdr10plus reader: %s\n"), dynamicHdr10plusJson.c_str());
    }
    return hdr10plus;
}
#endif

bool invalid_with_raw_out(const RGYParamCommon &prm, shared_ptr<RGYLog> log) {
    bool error = false;
#define INVALID_WITH_RAW_OUT(check, option_name) { \
    if (check) { error = true; log->write(RGY_LOG_ERROR, RGY_LOGT_APP, _T("%s cannot be used with -c raw!\n"), _T(option_name)); } \
}

    INVALID_WITH_RAW_OUT(prm.maxCll.length() > 0, "--max-cll");
    INVALID_WITH_RAW_OUT(prm.masterDisplay.length() > 0, "--master-display");
    INVALID_WITH_RAW_OUT(prm.hdr10plusMetadataCopy, "--dhdr10-info copy");
    INVALID_WITH_RAW_OUT(prm.dynamicHdr10plusJson.length() > 0, "--dhdr10-info");
    INVALID_WITH_RAW_OUT(prm.doviRpuFile.length() > 0, "--dolby-vision-rpu");
    INVALID_WITH_RAW_OUT(prm.nAudioSelectCount > 0, "audio related options");
    INVALID_WITH_RAW_OUT(prm.audioSource.size() > 0, "--audio-source");
    INVALID_WITH_RAW_OUT(prm.nSubtitleSelectCount > 0, "subtitle related options");
    INVALID_WITH_RAW_OUT(prm.subSource.size() > 0, "--sub-source");
    INVALID_WITH_RAW_OUT(prm.nDataSelectCount > 0, "data related options");
    INVALID_WITH_RAW_OUT(prm.nAttachmentSelectCount > 0, "--attachment-copy");
    INVALID_WITH_RAW_OUT(prm.chapterFile.length() > 0, "--chapter");
    INVALID_WITH_RAW_OUT(prm.copyChapter, "--chapter-copy");
    INVALID_WITH_RAW_OUT(prm.formatMetadata.size() > 0, "--metadata");
    INVALID_WITH_RAW_OUT(prm.videoMetadata.size() > 0, "--video-metadata");
    INVALID_WITH_RAW_OUT(prm.muxOpt.size() > 0, "-m");
    INVALID_WITH_RAW_OUT(prm.keyFile.length() > 0, "--keyfile");
    INVALID_WITH_RAW_OUT(prm.timecodeFile.length() > 0, "--timecode");
    INVALID_WITH_RAW_OUT(prm.metric.ssim, "--ssim");
    INVALID_WITH_RAW_OUT(prm.metric.psnr, "--psnr");
    INVALID_WITH_RAW_OUT(prm.metric.vmaf.enable, "--vmaf");

    return error;
}
