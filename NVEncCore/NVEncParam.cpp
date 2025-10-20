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

// -----------------------------------------------------------------------------
// Codec specific param structs default ctor
// -----------------------------------------------------------------------------
NVEncVideoParamH264::NVEncVideoParamH264() :
    profile(get_value_from_guid(NV_ENC_H264_PROFILE_HIGH_GUID, h264_profile_names)),
    level(NV_ENC_LEVEL_AUTOSELECT),
    bdirect(NV_ENC_H264_BDIRECT_MODE_AUTOSELECT),
    adaptTrans(NV_ENC_H264_ADAPTIVE_TRANSFORM_AUTOSELECT),
    entropy((NV_ENC_H264_ENTROPY_CODING_MODE)NV_ENC_H264_ENTROPY_CODING_MODE_CABAC),
    deblockIDC(0),
    hierarchicalPFrames(),
    hierarchicalBFrames() {
}

NVEncVideoParamHEVC::NVEncVideoParamHEVC() :
    profile(get_value_from_guid(NV_ENC_HEVC_PROFILE_MAIN_GUID, h265_profile_names)),
    level(NV_ENC_LEVEL_AUTOSELECT),
    tier(NV_ENC_TIER_HEVC_MAIN),
    cuMin(NV_ENC_HEVC_CUSIZE_AUTOSELECT),
    cuMax(NV_ENC_HEVC_CUSIZE_AUTOSELECT) {
}

NVEncVideoParamAV1::NVEncVideoParamAV1() :
    profile(get_value_from_guid(NV_ENC_AV1_PROFILE_MAIN_GUID, av1_profile_names)),
    level(NV_ENC_LEVEL_AV1_AUTOSELECT),
    tier(NV_ENC_TIER_AV1_0),
    partMin(NV_ENC_AV1_PART_SIZE_AUTOSELECT),
    partMax(NV_ENC_AV1_PART_SIZE_AUTOSELECT),
    tilesCols(0),
    tilesRows(0),
    fwdRefs(NV_ENC_NUM_REF_FRAMES_AUTOSELECT),
    bwdRefs(NV_ENC_NUM_REF_FRAMES_AUTOSELECT),
    annexB(),
    disableSeqHdr() {
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
    config.h264Config.idrPeriod = 0;

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
    config.hevcConfig.idrPeriod = 0;
    config.hevcConfig.tier  = NV_ENC_TIER_HEVC_MAIN;
    config.hevcConfig.minCUSize = NV_ENC_HEVC_CUSIZE_AUTOSELECT;
    config.hevcConfig.maxCUSize = NV_ENC_HEVC_CUSIZE_AUTOSELECT;
    config.hevcConfig.sliceMode = 0;
    config.hevcConfig.sliceModeData = 0;
    config.hevcConfig.maxNumRefFramesInDPB = DEFAULT_REF_FRAMES;
    config.hevcConfig.chromaFormatIDC = 1;
    config.hevcConfig.inputBitDepth = NV_ENC_BIT_DEPTH_8;
    config.hevcConfig.outputBitDepth = NV_ENC_BIT_DEPTH_8;

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
    config.av1Config.inputBitDepth = NV_ENC_BIT_DEPTH_8;
    config.av1Config.outputBitDepth = NV_ENC_BIT_DEPTH_8;

    config.av1Config.idrPeriod = 0;

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

InEncodeVideoParam::InEncodeVideoParam() :
    deviceID(-1),
    cudaSchedule(DEFAULT_CUDA_SCHEDULE),
    cudaStreamOpt(1),
    cudaMT(0),
    sessionRetry(0),
    disableNVML(0),
    disableDX11(false),
    input(),
    preset(0),
    nHWDecType(0),
    par(),
    rcParam(),
    gopLength(DEFAULT_GOP_LENGTH),
    bFrames(),
    mvPrecision(NV_ENC_MV_PRECISION_DEFAULT),
    qpInit(RGYQPSet(DEFAUTL_QP_I, DEFAULT_QP_P, DEFAULT_QP_B, false)),
    qpMin(RGYQPSet(0, 0, 0, true)),
    qpMax(RGYQPSet(255, 255, 255, false)),
    targetQuality(25),
    targetQualityLSB(0),
    vbvBufferSize(0),
    vbvInitialDelay(0),
    multipass(NV_ENC_MULTI_PASS_DISABLED),
    strictGOP(),
    disableIadapt(),
    disableBadapt(),
    enableAQ(),
    enableAQTemporal(),
    nonrefP(),
    unidirectB(false),
    enableLookahead(),
    lookahead(),
    lookaheadLevel(),
    aqStrength(),
    temporalFilterLevel(),
    tuningInfo(NV_ENC_TUNING_INFO_UNDEFINED),
    h264(),
    hevc(),
    av1(),
    dynamicRC(),
    codec_rgy(RGY_CODEC_H264),
    bluray(0),                   //bluray出力
    outputDepth(8),
    outputCsp(RGY_CSP_YV12),
    lossless(0),                 //ロスレス出力
    losslessIgnoreInputCsp(0),
    temporalSVC(false),
    temporalLayers(),
    alphaBitrateRatio(0),
    alphaChannelMode(0),
    nWeightP(),
    chromaQPOffset(0),
    brefMode(NV_ENC_BFRAME_REF_MODE_AUTO),
    splitEncMode(NV_ENC_SPLIT_AUTO_MODE),
    bitstreamPadding(false),
    maxRef(),
    refL0(NV_ENC_NUM_REF_FRAMES_AUTOSELECT),
    refL1(NV_ENC_NUM_REF_FRAMES_AUTOSELECT),
    slices(0),
    common(),
    inprm(),
    ctrl(),
    vpp(),
    vppnv() {
    rcParam.qp = RGYQPSet(DEFAUTL_QP_I, DEFAULT_QP_P, DEFAULT_QP_B);
    rcParam.rc_mode = NV_ENC_PARAMS_RC_QVBR;
    rcParam.avg_bitrate = DEFAULT_AVG_BITRATE;
    rcParam.max_bitrate = DEFAULT_MAX_BITRATE;
    input.vui = VideoVUIInfo();
}

void InEncodeVideoParam::applyDOVIProfile(const RGYDOVIProfile inputProfile) {
#if !FOR_AUO
    if (codec_rgy != RGY_CODEC_HEVC) {
        return;
    }
    auto targetDoviProfile = (common.doviProfile == RGY_DOVI_PROFILE_COPY) ? inputProfile : common.doviProfile;
    if (targetDoviProfile == 0) {
        return;
    }
    auto profile = getDOVIProfile(targetDoviProfile);
    if (profile == nullptr) {
        return;
    }
    common.out_vui.setIfUnset(profile->vui);
    repeatHeaders = true;
    if (profile->aud) {
        aud = true;
    }
    if (profile->HRDSEI) {
        bufferingPeriodSEI = true;
        picTimingSEI = true;
    }
    if (profile->profile == 50) {
        chromaQPOffset = 3;
    }
    if (profile->profile == 81) {
        //hdr10sei
        //maxcll
    }
#endif //#if !FOR_AUO
}
