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

#include "NVEncParam.h"
#include "afs_stg.h"
#include "rgy_ini.h"

using std::vector;

tstring get_codec_profile_name_from_guid(RGY_CODEC codec, const GUID& codecProfileGUID) {
    switch (codec) {
    case RGY_CODEC_H264: return get_name_from_guid(codecProfileGUID, h264_profile_names);
    case RGY_CODEC_HEVC: return get_name_from_guid(codecProfileGUID, h265_profile_names);
    default: return _T("Unknown codec.\n");
    }
}

tstring get_codec_level_name(RGY_CODEC codec, int level) {
    switch (codec) {
    case RGY_CODEC_H264: return get_chr_from_value(list_avc_level, level);
    case RGY_CODEC_HEVC: return get_chr_from_value(list_hevc_level, level);
    default: return _T("Unknown codec.\n");
    }
}

tstring printParams(const std::vector<DynamicRCParam> &dynamicRC) {
    TStringStream t;
    for (const auto& a : dynamicRC) {
        t << a.print() << std::endl;
    }
    return t.str();
};

DynamicRCParam::DynamicRCParam() : start(-1), end(-1), rc_mode(NV_ENC_PARAMS_RC_CONSTQP), avg_bitrate(-1), max_bitrate(0), targetQuality(-1), targetQualityLSB(-1), qp() {

}
tstring DynamicRCParam::print() const {
    TStringStream t;
    if (end == INT_MAX || end <= 0) {
        t << "frame=" << start << ":end";
    } else {
        t << "frame=" << start << ":" << end;
    }
    t << "," << get_chr_from_value(list_nvenc_rc_method_en, rc_mode) << "=";
    if (rc_mode == NV_ENC_PARAMS_RC_CONSTQP) {
        t << qp.qpIntra << ":" << qp.qpInterP << ":" << qp.qpInterB;
    } else {
        t << avg_bitrate / 1000;
        if (targetQuality >= 0) {
            double qual = targetQuality + targetQualityLSB / 256.0;
            t << ",vbr-quality=" << qual;
        }
    }
    if (max_bitrate != 0) {
        t << ",maxbitrate=" << max_bitrate / 1000;
    }
    return t.str();
}
bool DynamicRCParam::operator==(const DynamicRCParam &x) const {
    return start == x.start
        && end == x.end
        && rc_mode == x.rc_mode
        && avg_bitrate == x.avg_bitrate
        && max_bitrate == x.max_bitrate
        && targetQuality == x.targetQuality
        && targetQualityLSB == x.targetQualityLSB
        && memcmp(&qp, &x.qp, sizeof(qp)) == 0;
}
bool DynamicRCParam::operator!=(const DynamicRCParam &x) const {
    return !(*this == x);
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
    return str;
}

VppUnsharp::VppUnsharp() :
    enable(false),
    radius(FILTER_DEFAULT_UNSHARP_RADIUS),
    weight(FILTER_DEFAULT_UNSHARP_WEIGHT),
    threshold(FILTER_DEFAULT_UNSHARP_THRESHOLD) {

}

bool VppUnsharp::operator==(const VppUnsharp& x) const {
    return enable == x.enable
        && radius == x.radius
        && weight == x.weight
        && threshold == x.threshold;
}
bool VppUnsharp::operator!=(const VppUnsharp& x) const {
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

bool VppEdgelevel::operator==(const VppEdgelevel& x) const {
    return enable == x.enable
        && strength == x.strength
        && threshold == x.threshold
        && black == x.black
        && white == x.white;
}
bool VppEdgelevel::operator!=(const VppEdgelevel& x) const {
    return !(*this == x);
}

tstring VppEdgelevel::print() const {
    return strsprintf(_T("edgelevel: strength %.1f, threshold %.1f, black %.1f, white %.1f"),
        strength, threshold, black, white);
}

VppKnn::VppKnn() :
    enable(false),
    radius(FILTER_DEFAULT_KNN_RADIUS),
    strength(FILTER_DEFAULT_KNN_STRENGTH),
    lerpC(FILTER_DEFAULT_KNN_LERPC),
    weight_threshold(FILTER_DEFAULT_KNN_WEIGHT_THRESHOLD),
    lerp_threshold(FILTER_DEFAULT_KNN_LERPC_THRESHOLD) {
}

bool VppKnn::operator==(const VppKnn& x) const {
    return enable == x.enable
        && radius == x.radius
        && strength == x.strength
        && lerpC == x.lerpC
        && weight_threshold == x.weight_threshold
        && lerp_threshold == x.lerp_threshold;
}
bool VppKnn::operator!=(const VppKnn& x) const {
    return !(*this == x);
}

tstring VppKnn::print() const {
    return strsprintf(
        _T("denoise(knn): radius %d, strength %.2f, lerp %.2f\n")
        _T("                              th_weight %.2f, th_lerp %.2f"),
        radius, strength, lerpC,
        weight_threshold, lerp_threshold);
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

bool VppDeband::operator==(const VppDeband& x) const {
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
bool VppDeband::operator!=(const VppDeband& x) const {
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
        &&peak == x.peak;
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
    hdr_source_peak(FILTER_DEFAULT_COLORSPACE_HDR_SOURCE_PEAK) {

}
bool HDR2SDRParams::operator==(const HDR2SDRParams &x) const {
    return tonemap == x.tonemap
        && hable == x.hable
        && mobius == x.mobius
        && reinhard == x.reinhard;
}
bool HDR2SDRParams::operator!=(const HDR2SDRParams &x) const {
    return !(*this == x);
}

VppColorspace::VppColorspace() :
    enable(false),
    hdr2sdr(),
    convs() {

}

bool VppColorspace::operator==(const VppColorspace &x) const {
    if (enable != x.enable
        || x.hdr2sdr != this->hdr2sdr
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

VppTweak::VppTweak() :
    enable(false),
    brightness(FILTER_DEFAULT_TWEAK_BRIGHTNESS),
    contrast(FILTER_DEFAULT_TWEAK_CONTRAST),
    gamma(FILTER_DEFAULT_TWEAK_GAMMA),
    saturation(FILTER_DEFAULT_TWEAK_SATURATION),
    hue(FILTER_DEFAULT_TWEAK_HUE) {
}

bool VppTweak::operator==(const VppTweak& x) const {
    return enable == x.enable
        && brightness == x.brightness
        && contrast == x.contrast
        && gamma == x.gamma
        && saturation == x.saturation
        && hue == x.hue;
}
bool VppTweak::operator!=(const VppTweak& x) const {
    return !(*this == x);
}

tstring VppTweak::print() const {
    return strsprintf(_T("tweak: brightness %.2f, contrast %.2f, saturation %.2f, gamma %.2f, hue %.2f"),
        brightness, contrast, saturation, gamma, hue);
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

VppSubburn::VppSubburn() :
    enable(false),
    filename(),
    charcode(),
    trackId(0),
    assShaping(1),
    scale(0.0),
    transparency_offset(0.0),
    brightness(FILTER_DEFAULT_TWEAK_BRIGHTNESS),
    contrast(FILTER_DEFAULT_TWEAK_CONTRAST),
    ts_offset(0.0),
    vid_ts_offset(true) {
}

bool VppSubburn::operator==(const VppSubburn &x) const {
    return enable == x.enable
        && filename == x.filename
        && charcode == x.charcode
        && trackId == x.trackId
        && assShaping == x.assShaping
        && scale == x.scale
        && transparency_offset == x.transparency_offset
        && brightness == x.brightness
        && contrast == x.contrast
        && ts_offset == x.ts_offset
        && vid_ts_offset == x.vid_ts_offset;
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
    return str;
}

VppCustom::VppCustom() :
    enable(false),
    filter_name(),
    kernel_name(FILTER_DEFAULT_CUSTOM_KERNEL_NAME),
    kernel_path(),
    kernel(),
    compile_options(),
    kernel_interface(VPP_CUSTOM_INTERFACE_PER_PLANE),
    interlace(VPP_CUSTOM_INTERLACE_UNSUPPORTED),
    threadPerBlockX(FILTER_DEFAULT_CUSTOM_THREAD_PER_BLOCK_X),
    threadPerBlockY(FILTER_DEFAULT_CUSTOM_THREAD_PER_BLOCK_Y),
    pixelPerThreadX(FILTER_DEFAULT_CUSTOM_PIXEL_PER_THREAD_X),
    pixelPerThreadY(FILTER_DEFAULT_CUSTOM_PIXEL_PER_THREAD_Y),
    dstWidth(0),
    dstHeight(0),
    params() {

}

bool VppCustom::operator==(const VppCustom &x) const {
    return enable == x.enable
        && filter_name == x.filter_name
        && kernel_name == x.kernel_name
        && kernel_path == x.kernel_path
        && kernel == x.kernel
        && compile_options == x.compile_options
        && kernel_interface == x.kernel_interface
        && interlace == x.interlace
        && threadPerBlockX == x.threadPerBlockX
        && threadPerBlockY == x.threadPerBlockY
        && pixelPerThreadX == x.pixelPerThreadX
        && pixelPerThreadY == x.pixelPerThreadY
        && dstWidth == x.dstWidth
        && dstHeight == x.dstHeight
        && params == x.params;
}
bool VppCustom::operator!=(const VppCustom &x) const {
    return !(*this == x);
}

tstring VppCustom::print() const {
    return strsprintf(_T("%s: %s, interface %s, interlace %s\n")
        _T("                    thread/block (%d,%d), pixel/thread (%d,%d)\n"),
        filter_name.c_str(), kernel_path.c_str(),
        get_cx_desc(list_vpp_custom_interface, kernel_interface),
        get_cx_desc(list_vpp_custom_interlace, interlace),
        threadPerBlockX, threadPerBlockY,
        pixelPerThreadX, pixelPerThreadY);
}

VppParam::VppParam() :
    checkPerformance(false),
    deinterlace(cudaVideoDeinterlaceMode_Weave),
    resizeInterp(NPPI_INTER_UNDEFINED),
    gaussMaskSize((NppiMaskSize)0),
    delogo(),
    unsharp(),
    edgelevel(),
    knn(),
    pmd(),
    smooth(),
    deband(),
    afs(),
    nnedi(),
    yadif(),
    tweak(),
    transform(),
    colorspace(),
    pad(),
    subburn(),
    selectevery(),
    rff(false) {
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

bool VppAfs::operator==(const VppAfs& x) const {
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
bool VppAfs::operator!=(const VppAfs& x) const {
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

int VppAfs::read_afs_inifile(const TCHAR* inifile) {
    if (!PathFileExists(inifile)) {
        return 1;
    }
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

bool VppNnedi::operator==(const VppNnedi& x) const {
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
bool VppNnedi::operator!=(const VppNnedi& x) const {
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

NV_ENC_CODEC_CONFIG DefaultParamH264() {
    NV_ENC_CODEC_CONFIG config = { 0 };

    config.h264Config.level     = NV_ENC_LEVEL_AUTOSELECT;
    config.h264Config.idrPeriod = DEFAULT_GOP_LENGTH;

    config.h264Config.chromaFormatIDC            = 1;
    config.h264Config.disableDeblockingFilterIDC = 0;
    config.h264Config.disableSPSPPS              = 0;
    config.h264Config.sliceMode                  = 3;
    config.h264Config.sliceModeData              = DEFAULT_NUM_SLICES;
    config.h264Config.maxNumRefFrames            = DEFAULT_REF_FRAMES;
    config.h264Config.bdirectMode                = NV_ENC_H264_BDIRECT_MODE_AUTOSELECT;
    config.h264Config.adaptiveTransformMode      = NV_ENC_H264_ADAPTIVE_TRANSFORM_AUTOSELECT;
    config.h264Config.entropyCodingMode          = NV_ENC_H264_ENTROPY_CODING_MODE_CABAC;

    config.h264Config.h264VUIParameters.overscanInfo = 0;
    config.h264Config.h264VUIParameters.colourMatrix            = get_cx_value(list_colormatrix, _T("undef"));
    config.h264Config.h264VUIParameters.colourPrimaries         = get_cx_value(list_colorprim,   _T("undef"));
    config.h264Config.h264VUIParameters.transferCharacteristics = get_cx_value(list_transfer,    _T("undef"));
    config.h264Config.h264VUIParameters.videoFormat             = get_cx_value(list_videoformat, _T("undef"));

    return config;
}

NV_ENC_CODEC_CONFIG DefaultParamHEVC() {
    NV_ENC_CODEC_CONFIG config = { 0 };

    config.hevcConfig.level = NV_ENC_LEVEL_AUTOSELECT;
    config.hevcConfig.tier  = NV_ENC_TIER_HEVC_MAIN;
    config.hevcConfig.minCUSize = NV_ENC_HEVC_CUSIZE_AUTOSELECT;
    config.hevcConfig.maxCUSize = NV_ENC_HEVC_CUSIZE_AUTOSELECT;
    config.hevcConfig.sliceMode = 0;
    config.hevcConfig.sliceModeData = 0;
    config.hevcConfig.maxNumRefFramesInDPB = DEFAULT_REF_FRAMES;
    config.hevcConfig.chromaFormatIDC = 1;

    config.hevcConfig.hevcVUIParameters.overscanInfo = 0;
    config.hevcConfig.hevcVUIParameters.colourMatrix            = get_cx_value(list_colormatrix, _T("undef"));
    config.hevcConfig.hevcVUIParameters.colourPrimaries         = get_cx_value(list_colorprim,   _T("undef"));
    config.hevcConfig.hevcVUIParameters.transferCharacteristics = get_cx_value(list_transfer,    _T("undef"));
    config.hevcConfig.hevcVUIParameters.videoFormat             = get_cx_value(list_videoformat, _T("undef"));

    return config;
}

NV_ENC_CONFIG DefaultParam() {

    NV_ENC_CONFIG config = { 0 };
    SET_VER(config, NV_ENC_CONFIG);
    config.frameFieldMode                 = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
    config.profileGUID                    = NV_ENC_H264_PROFILE_HIGH_GUID;
    config.gopLength                      = DEFAULT_GOP_LENGTH;
    config.rcParams.rateControlMode       = NV_ENC_PARAMS_RC_CONSTQP;
    //config.encodeCodecConfig.h264Config.level;
    config.frameIntervalP                 = DEFAULT_B_FRAMES + 1;
    config.mvPrecision                    = NV_ENC_MV_PRECISION_DEFAULT;
    config.monoChromeEncoding             = 0;
    config.rcParams.version               = NV_ENC_RC_PARAMS_VER;
    config.rcParams.averageBitRate        = DEFAULT_AVG_BITRATE;
    config.rcParams.maxBitRate            = 0;
    config.rcParams.enableInitialRCQP     = 1;
    config.rcParams.initialRCQP.qpInterB  = DEFAULT_QP_B;
    config.rcParams.initialRCQP.qpInterP  = DEFAULT_QP_P;
    config.rcParams.initialRCQP.qpIntra   = DEFAUTL_QP_I;
    config.rcParams.maxQP.qpInterB        = 51;
    config.rcParams.maxQP.qpInterP        = 51;
    config.rcParams.maxQP.qpIntra         = 51;
    config.rcParams.constQP.qpInterB      = DEFAULT_QP_B;
    config.rcParams.constQP.qpInterP      = DEFAULT_QP_P;
    config.rcParams.constQP.qpIntra       = DEFAUTL_QP_I;
    config.rcParams.lookaheadDepth        = DEFAULT_LOOKAHEAD;
    config.rcParams.targetQuality         = 0; //auto
    config.rcParams.targetQualityLSB      = 0;

    config.rcParams.vbvBufferSize         = 0;
    config.rcParams.vbvInitialDelay       = 0;
    config.encodeCodecConfig              = DefaultParamH264();

    return config;
}

InEncodeVideoParam::InEncodeVideoParam() :
    deviceID(-1),
    cudaSchedule(DEFAULT_CUDA_SCHEDULE),
    gpuSelect(),
    sessionRetry(0),
    input(),
    preset(0),
    nHWDecType(0),
    par(),
    encConfig(),
    dynamicRC(),
    codec(NV_ENC_H264),
    bluray(0),                   //bluray出力
    yuv444(0),                   //YUV444出力
    lossless(0),                 //ロスレス出力
    common(),
    ctrl(),
    vpp(),
    ssim(false),
    psnr(false),
    nWeightP(0) {
    encConfig = DefaultParam();
    memset(&par, 0, sizeof(par));
    memset(&input, 0, sizeof(input));
    input.vui = VideoVUIInfo();
}

