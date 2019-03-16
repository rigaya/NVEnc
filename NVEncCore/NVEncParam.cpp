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

VppParam::VppParam() :
    bCheckPerformance(false),
    deinterlace(cudaVideoDeinterlaceMode_Weave),
    resizeInterp(NPPI_INTER_UNDEFINED),
    gaussMaskSize((NppiMaskSize)0),
    unsharp(),
    edgelevel(),
    delogo(),
    knn(),
    pmd(),
    deband(),
    afs(),
    nnedi(),
    tweak(),
    pad(),
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

VppNnedi::VppNnedi() :
    enable(false),
    field(VPP_NNEDI_FIELD_USE_AUTO),
    nns(32),
    nsize(VPP_NNEDI_NSIZE_32x4),
    quality(VPP_NNEDI_QUALITY_FAST),
    precision(VPP_NNEDI_PRECISION_AUTO),
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
    input(),
    inputFilename(),
    outputFilename(),
    sAVMuxOutputFormat(),
    preset(0),
    deviceID(-1),
    nHWDecType(0),
    par(),
    encConfig(),
    codec(0),
    bluray(0),                   //bluray出力
    yuv444(0),                   //YUV444出力
    lossless(0),                 //ロスレス出力
    sMaxCll(),
    sMasterDisplay(),
    videoCodecTag(),
    logfile(),              //ログ出力先
    loglevel(RGY_LOG_INFO),                 //ログ出力レベル
    nOutputBufSizeMB(DEFAULT_OUTPUT_BUF),         //出力バッファサイズ
    sFramePosListLog(),     //framePosList出力先
    fSeekSec(0.0f),               //指定された秒数分先頭を飛ばす
    nSubtitleSelectCount(0),
    pSubtitleSelect(nullptr),
    nAudioSourceCount(0),
    ppAudioSourceList(nullptr),
    nAudioSelectCount(0), //pAudioSelectの数
    ppAudioSelectList(nullptr),
    nAudioResampler(RGY_RESAMPLER_SWR),
    nAVDemuxAnalyzeSec(0),
    nAVMux(RGY_MUX_NONE),                       //RGY_MUX_xxx
    nVideoTrack(0),
    nVideoStreamId(0),
    nTrimCount(0),
    pTrimList(nullptr),
    bCopyChapter(false),
    keyOnChapter(false),
    caption2ass(FORMAT_INVALID),
    nOutputThread(RGY_OUTPUT_THREAD_AUTO),
    nAudioThread(RGY_INPUT_THREAD_AUTO),
    nInputThread(RGY_AUDIO_THREAD_AUTO),
    nAudioIgnoreDecodeError(DEFAULT_IGNORE_DECODE_ERROR),
    pMuxOpt(nullptr),
    sChapterFile(),
    pMuxVidTsLogFile(nullptr),
    pAVInputFormat(nullptr),
    nAVSyncMode(RGY_AVSYNC_ASSUME_CFR),     //avsyncの方法 (RGY_AVSYNC_xxx)
    nProcSpeedLimit(0),      //処理速度制限 (0で制限なし)
    vpp(),
    nWeightP(0),
    nPerfMonitorSelect(0),
    nPerfMonitorSelectMatplot(0),
    nPerfMonitorInterval(RGY_DEFAULT_PERF_MONITOR_INTERVAL),
    nCudaSchedule(DEFAULT_CUDA_SCHEDULE),
    sessionRetry(0),
    pPrivatePrm(nullptr) {
    encConfig = DefaultParam();
    memset(&par, 0, sizeof(par));
    memset(&input, 0, sizeof(input));
}

