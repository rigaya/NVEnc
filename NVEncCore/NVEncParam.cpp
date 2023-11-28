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
#include "rgy_bitstream.h"

using std::vector;

tstring get_codec_profile_name_from_guid(RGY_CODEC codec, const GUID& codecProfileGUID) {
    switch (codec) {
    case RGY_CODEC_H264: return get_name_from_guid(codecProfileGUID, h264_profile_names);
    case RGY_CODEC_HEVC: return get_name_from_guid(codecProfileGUID, h265_profile_names);
    case RGY_CODEC_AV1:  return get_name_from_guid(codecProfileGUID, av1_profile_names);
    default: return _T("Unknown codec.\n");
    }
}

tstring get_codec_level_name(RGY_CODEC codec, int level) {
    switch (codec) {
    case RGY_CODEC_H264: return get_chr_from_value(list_avc_level, level);
    case RGY_CODEC_HEVC: return get_chr_from_value(list_hevc_level, level);
    case RGY_CODEC_AV1:  return get_chr_from_value(list_av1_level, level);
    default: return _T("Unknown codec.\n");
    }
}

VppCustom::VppCustom() :
    enable(false),
    filter_name(),
    kernel_name(FILTER_DEFAULT_CUSTOM_KERNEL_NAME),
    kernel_path(),
    kernel(),
    dev_params(nullptr),
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
        && dev_params == x.dev_params
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

VppNvvfxDenoise::VppNvvfxDenoise() :
    enable(false),
    strength(FILTER_DEFAULT_NVVFX_DENOISE_STRENGTH) {

}

bool VppNvvfxDenoise::operator==(const VppNvvfxDenoise &x) const {
    return enable == x.enable
        && strength == x.strength;
}
bool VppNvvfxDenoise::operator!=(const VppNvvfxDenoise &x) const {
    return !(*this == x);
}

tstring VppNvvfxDenoise::print() const {
    return strsprintf(_T("nvvfx-denoise: strength %.0f"),
        strength);
}

VppNvvfxArtifactReduction::VppNvvfxArtifactReduction() :
    enable(false),
    mode(FILTER_DEFAULT_NVVFX_ARTIFACT_REDUCTION_MODE) {

}

bool VppNvvfxArtifactReduction::operator==(const VppNvvfxArtifactReduction &x) const {
    return enable == x.enable
        && mode == x.mode;
}
bool VppNvvfxArtifactReduction::operator!=(const VppNvvfxArtifactReduction &x) const {
    return !(*this == x);
}

tstring VppNvvfxArtifactReduction::print() const {
    return strsprintf(_T("nvvfx-artifact-reduction: mode %d (%s)"),
        mode, get_cx_desc(list_vpp_nvvfx_mode, mode));
}

VppNvvfxSuperRes::VppNvvfxSuperRes() :
    enable(false),
    mode(FILTER_DEFAULT_NVVFX_SUPER_RES_MODE),
    strength(FILTER_DEFAULT_NVVFX_SUPER_RES_STRENGTH) {

}

bool VppNvvfxSuperRes::operator==(const VppNvvfxSuperRes &x) const {
    return enable == x.enable
        && mode == x.mode
        && strength == x.strength;
}
bool VppNvvfxSuperRes::operator!=(const VppNvvfxSuperRes &x) const {
    return !(*this == x);
}

tstring VppNvvfxSuperRes::print() const {
    return strsprintf(_T("nvvfx-superres: mode: %d (%s), strength %.2f"),
        mode, get_cx_desc(list_vpp_nvvfx_mode, mode), strength);
}

VppNvvfxUpScaler::VppNvvfxUpScaler() :
    enable(false),
    strength(FILTER_DEFAULT_NVVFX_UPSCALER_STRENGTH) {

}

bool VppNvvfxUpScaler::operator==(const VppNvvfxUpScaler &x) const {
    return enable == x.enable
        && strength == x.strength;
}
bool VppNvvfxUpScaler::operator!=(const VppNvvfxUpScaler &x) const {
    return !(*this == x);
}

tstring VppNvvfxUpScaler::print() const {
    return strsprintf(_T("nvvfx-upscaler: strength %.2f"),
        strength);
}


VppParam::VppParam() :
    deinterlace(cudaVideoDeinterlaceMode_Weave),
    gaussMaskSize((NppiMaskSize)0),
    nvvfxDenoise(),
    nvvfxArtifactReduction(),
    nvvfxSuperRes(),
    nvvfxUpScaler(),
    nvvfxModelDir() {
}

tstring printParams(const std::vector<NVEncRCParam> &dynamicRC) {
    TStringStream t;
    for (const auto& a : dynamicRC) {
        t << a.print() << std::endl;
    }
    return t.str();
};

NVEncRCParam::NVEncRCParam() :
    start(-1),
    end(-1),
    rc_mode(NV_ENC_PARAMS_RC_VBR),
    avg_bitrate(0),
    max_bitrate(0),
    targetQuality(-1),
    targetQualityLSB(-1),
    qp() {

}
tstring NVEncRCParam::print() const {
    TStringStream t;
    if (start >= 0) {
        if (end == INT_MAX || end <= 0) {
            t << "frame=" << start << ":end";
        } else {
            t << "frame=" << start << ":" << end;
        }
        t << ",";
    }
    t << get_chr_from_value(list_nvenc_rc_method_en, rc_mode) << "=";
    if (rc_mode == NV_ENC_PARAMS_RC_CONSTQP) {
        t << qp.qpI << ":" << qp.qpP << ":" << qp.qpB;
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
bool NVEncRCParam::operator==(const NVEncRCParam &x) const {
    return start == x.start
        && end == x.end
        && rc_mode == x.rc_mode
        && avg_bitrate == x.avg_bitrate
        && max_bitrate == x.max_bitrate
        && targetQuality == x.targetQuality
        && targetQualityLSB == x.targetQualityLSB
        && qp == x.qp;
}
bool NVEncRCParam::operator!=(const NVEncRCParam &x) const {
    return !(*this == x);
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
    set_colormatrix(config, RGY_CODEC_H264, get_cx_value(list_colormatrix, _T("undef")));
    set_colorprim(  config, RGY_CODEC_H264, get_cx_value(list_colorprim,   _T("undef")));
    set_transfer(   config, RGY_CODEC_H264, get_cx_value(list_transfer,    _T("undef")));
    set_videoFormat(config, RGY_CODEC_H264, get_cx_value(list_videoformat, _T("undef")));

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
    set_colormatrix(config, RGY_CODEC_HEVC, get_cx_value(list_colormatrix, _T("undef")));
    set_colorprim(  config, RGY_CODEC_HEVC, get_cx_value(list_colorprim,   _T("undef")));
    set_transfer(   config, RGY_CODEC_HEVC, get_cx_value(list_transfer,    _T("undef")));
    set_videoFormat(config, RGY_CODEC_HEVC, get_cx_value(list_videoformat, _T("undef")));

    return config;
}

NV_ENC_CODEC_CONFIG DefaultParamAV1() {
    NV_ENC_CODEC_CONFIG config = { 0 };

    config.av1Config.level = NV_ENC_LEVEL_AV1_AUTOSELECT;
    config.av1Config.tier = NV_ENC_TIER_AV1_0;
    config.av1Config.minPartSize = NV_ENC_AV1_PART_SIZE_AUTOSELECT;
    config.av1Config.maxPartSize = NV_ENC_AV1_PART_SIZE_AUTOSELECT;
    config.av1Config.outputAnnexBFormat = 0;
    config.av1Config.disableSeqHdr = 0;
    config.av1Config.chromaFormatIDC = 1;
    config.av1Config.enableBitstreamPadding = 0;
    config.av1Config.enableCustomTileConfig = 0;
    config.av1Config.enableFilmGrainParams = 0;
    config.av1Config.inputPixelBitDepthMinus8 = 0;
    config.av1Config.pixelBitDepthMinus8 = 0;

    config.av1Config.idrPeriod = DEFAULT_GOP_LENGTH;

    config.av1Config.enableCustomTileConfig = 0;
    config.av1Config.numTileColumns = 0;
    config.av1Config.numTileRows = 0;

    config.av1Config.maxNumRefFramesInDPB = DEFAULT_REF_FRAMES;
    config.av1Config.maxTemporalLayersMinus1 = 0;
    config.av1Config.chromaFormatIDC = 1;

    config.av1Config.numFwdRefs = NV_ENC_NUM_REF_FRAMES_AUTOSELECT;
    config.av1Config.numBwdRefs = NV_ENC_NUM_REF_FRAMES_AUTOSELECT;
    
    set_colormatrix(config, RGY_CODEC_AV1, get_cx_value(list_colormatrix, _T("undef")));
    set_colorprim(  config, RGY_CODEC_AV1, get_cx_value(list_colorprim,   _T("undef")));
    set_transfer(   config, RGY_CODEC_AV1, get_cx_value(list_transfer,    _T("undef")));
    set_videoFormat(config, RGY_CODEC_AV1, get_cx_value(list_videoformat, _T("undef")));

    return config;
}

NV_ENC_CONFIG DefaultParam() {

    NV_ENC_CONFIG config = { 0 };
    config.frameFieldMode                 = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
    config.profileGUID                    = NV_ENC_H264_PROFILE_HIGH_GUID;
    config.gopLength                      = DEFAULT_GOP_LENGTH;
    config.rcParams.rateControlMode       = NV_ENC_PARAMS_RC_VBR;
    //config.encodeCodecConfig.h264Config.level;
    config.frameIntervalP                 = DEFAULT_B_FRAMES + 1;
    config.mvPrecision                    = NV_ENC_MV_PRECISION_DEFAULT;
    config.monoChromeEncoding             = 0;
    config.rcParams.version               = NV_ENC_RC_PARAMS_VER;
    config.rcParams.averageBitRate        = 0;
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
    config.rcParams.targetQuality         = 0;
    config.rcParams.targetQualityLSB      = 0;

    config.rcParams.vbvBufferSize         = 0;
    config.rcParams.vbvInitialDelay       = 0;
    config.encodeCodecConfig              = DefaultParamH264();

    return config;
}

InEncodeVideoParam::InEncodeVideoParam() :
    deviceID(-1),
    cudaSchedule(DEFAULT_CUDA_SCHEDULE),
    sessionRetry(0),
    disableNVML(0),
    input(),
    preset(0),
    nHWDecType(0),
    par(),
    rcParam(),
    gopLength(DEFAULT_GOP_LENGTH),
    bFrames(DEFAULT_B_FRAMES),
    mvPrecision(NV_ENC_MV_PRECISION_DEFAULT),
    qpInit(RGYQPSet(DEFAUTL_QP_I, DEFAULT_QP_P, DEFAULT_QP_B)),
    qpMin(RGYQPSet(0, 0, 0)),
    qpMax(RGYQPSet(255, 255, 255)),
    targetQuality(25),
    targetQualityLSB(0),
    vbvBufferSize(0),
    vbvInitialDelay(0),
    multipass(NV_ENC_MULTI_PASS_DISABLED),
    strictGOP(true),
    disableIadapt(false),
    disableBadapt(false),
    enableAQ(false),
    enableAQTemporal(false),
    nonrefP(false),
    enableLookahead(false),
    lookahead(DEFAULT_LOOKAHEAD),
    aqStrength(0),
    encConfig(),
    dynamicRC(),
    codec_rgy(RGY_CODEC_H264),
    bluray(0),                   //bluray出力
    outputDepth(8),
    yuv444(0),                   //YUV444出力
    lossless(0),                 //ロスレス出力
    losslessIgnoreInputCsp(0),
    nWeightP(0),
    chromaQPOffset(0),
    brefMode(NV_ENC_BFRAME_REF_MODE_AUTO),
    splitEncMode(NV_ENC_SPLIT_AUTO_MODE),
    common(),
    inprm(),
    ctrl(),
    vpp(),
    vppnv() {
    encConfig = DefaultParam();
    rcParam.qp = RGYQPSet(DEFAUTL_QP_I, DEFAULT_QP_P, DEFAULT_QP_B);
    rcParam.rc_mode = NV_ENC_PARAMS_RC_QVBR;
    rcParam.avg_bitrate = DEFAULT_AVG_BITRATE;
    rcParam.max_bitrate = DEFAULT_MAX_BITRATE;
    input.vui = VideoVUIInfo();
}

void InEncodeVideoParam::applyDOVIProfile() {
#if !FOR_AUO
    if (codec_rgy != RGY_CODEC_H264) {
        return;
    }
    if (common.doviProfile == 0) {
        return;
    }
    auto profile = getDOVIProfile(common.doviProfile);
    if (profile == nullptr) {
        return;
    }
    common.out_vui.setIfUnset(profile->vui);
    encConfig.encodeCodecConfig.hevcConfig.repeatSPSPPS = 1;
    if (profile->aud) {
        encConfig.encodeCodecConfig.hevcConfig.outputAUD = 1;
    }
    if (profile->HRDSEI) {
        encConfig.encodeCodecConfig.hevcConfig.outputBufferingPeriodSEI = 1;
        encConfig.encodeCodecConfig.hevcConfig.outputPictureTimingSEI = 1;
    }
    if (profile->profile == 50) {
        encConfig.rcParams.crQPIndexOffset = 3;
    }
    if (profile->profile == 81) {
        //hdr10sei
        //maxcll
    }
#endif //#if !FOR_AUO
}
