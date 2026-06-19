// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc/VCEEnc by rigaya
// -----------------------------------------------------------------------------------------
//
// The MIT License
//
// Copyright (c) 2026 rigaya
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

#include "NVEncFilterRtgmc.h"
#include "NVEncFilterRtgmcCommon.h"
#include "rgy_cuda_util_kernel.h"
#pragma warning (push)
#pragma warning (disable: 4819)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#pragma warning (pop)

#include <algorithm>
#include <cmath>
#include <cstdlib>

static constexpr int RGY_RTGMC_MAX_OUT_FRAMES = 32;
static constexpr int RGY_RTGMC_MAX_RETURN_FRAMES = 2;
static constexpr int RGY_RTGMC_BORDER_PIXELS_Y = 4;
static constexpr int RGY_RTGMC_BORDER_BLOCK_X = 32;
static constexpr int RGY_RTGMC_BORDER_BLOCK_Y = 8;

namespace {
enum RtgmcFilterIndex : size_t {
    RTGMC_FILTER_BOB = 0,
    RTGMC_FILTER_SEARCH_PREFILTER,
    RTGMC_FILTER_ANALYZE,
    RTGMC_FILTER_NOISE,
    RTGMC_FILTER_EDI,
    RTGMC_FILTER_INPUTTYPE_BLEND,
    RTGMC_FILTER_TR1,
    RTGMC_FILTER_PRE_RETOUCH_SHIMMER_REPAIR,
    RTGMC_FILTER_PRE_RETOUCH_SOURCE_FIELD_RESTORE,
    RTGMC_FILTER_RETOUCH,
    RTGMC_FILTER_ADD_NOISE1,
    RTGMC_FILTER_TR2,
    RTGMC_FILTER_POST_TR2_SHIMMER_REPAIR,
    RTGMC_FILTER_POST_TR2_LIMIT,
    RTGMC_FILTER_FINAL_SOURCE_FIELD_RESTORE,
    RTGMC_FILTER_ADD_NOISE2,
};

static constexpr size_t RTGMC_MAX_STORED_EDI_REFS = 256;
static constexpr size_t RTGMC_MAX_STORED_POST_LIMIT_BASE_REFS = 256;
static constexpr size_t RTGMC_MAX_STORED_MATCH_CORRECTION_BASE_REFS = 256;

static bool rtgmcMatchCorrectionKernelMergeEnabled() {
    const char *env = std::getenv("QSVENC_RTGMC_KERNEL_MERGE_MATCH_CORRECTION");
    if (env == nullptr) {
        env = std::getenv("QSVENC_RTGMC_KERNEL_MERGE_SOURCE_REFINE");
    }
    return env == nullptr || env[0] != '0';
}

static int rtgmcNestedAnalyzeDelta(const VppRtgmc &rtgmc) {
    return std::max({ 1, std::min(rtgmc.analyze.delta, RGY_DEGRAIN_MAX_DELTA), std::min(rtgmc.tr1.delta, RGY_DEGRAIN_MAX_DELTA), std::min(rtgmc.tr2.delta, RGY_DEGRAIN_MAX_DELTA) });
}

static bool rtgmcInputTypeBlendEnabled(const VppRtgmc &rtgmc) {
    return (rtgmc.inputType == 2 || rtgmc.inputType == 3) && rtgmc.progSADMask > 0.0f;
}

static void eraseRtgmcInternalFrameData(RGYFrameInfo *frame) {
    if (!frame) {
        return;
    }
    rgy_degrain_erase_frame_data(frame->dataList);
    frame->dataList.erase(std::remove_if(frame->dataList.begin(), frame->dataList.end(), [](const std::shared_ptr<RGYFrameData> &data) {
        return std::dynamic_pointer_cast<RGYFrameDataRtgmcSearchLuma>(data) != nullptr
            || std::dynamic_pointer_cast<RGYFrameDataRtgmcEdi>(data) != nullptr
            || std::dynamic_pointer_cast<RGYFrameDataRtgmcComp>(data) != nullptr
            || std::dynamic_pointer_cast<RGYFrameDataRtgmcNoise>(data) != nullptr;
    }), frame->dataList.end());
}

static void propagateRtgmcInternalFrameData(RGYFrameInfo *dst, const RGYFrameInfo *src) {
    if (!dst || !src) {
        return;
    }
    if (rgy_degrain_get_frame_data(dst) == nullptr) {
        const auto degrain = rgy_degrain_get_frame_data(src);
        if (degrain) {
            dst->dataList.push_back(degrain);
        }
    }
    if (rtgmcGetAttachedEdi(dst) == nullptr) {
        const auto edi = rtgmcGetAttachedEdi(src);
        if (edi) {
            dst->dataList.push_back(edi);
        }
    }
    static const std::array<RGYRtgmcCompDirection, 2> directions = {
        RGYRtgmcCompDirection::Backward,
        RGYRtgmcCompDirection::Forward
    };
    for (const auto direction : directions) {
        if (rtgmcGetAttachedComp(dst, direction, 1) == nullptr) {
            const auto comp = rtgmcGetAttachedComp(src, direction, 1);
            if (comp) {
                dst->dataList.push_back(comp);
            }
        }
    }
    if (rtgmcGetAttachedNoise(dst) == nullptr) {
        const auto noise = rtgmcGetAttachedNoise(src);
        if (noise) {
            dst->dataList.push_back(noise);
        }
    }
}

static bool rtgmcSourceMatchCorrectionEdiSupported(VppRtgmcEdiMode mode) {
    switch (mode) {
    case VppRtgmcEdiMode::Bob:
    case VppRtgmcEdiMode::Yadif:
    case VppRtgmcEdiMode::cYadif:
    case VppRtgmcEdiMode::RepYadif:
    case VppRtgmcEdiMode::RepcYadif:
    case VppRtgmcEdiMode::NNEDI3:
        return true;
    default:
        return false;
    }
}

static bool rtgmcNoiseDenoiserSupported(VppRtgmcNoiseDenoiser denoiser) {
    return denoiser == VppRtgmcNoiseDenoiser::NLMeans
        || denoiser == VppRtgmcNoiseDenoiser::FFT3D;
}

static bool rtgmcNoiseDenoiserUsesNLMeans(VppRtgmcNoiseDenoiser denoiser) {
    return denoiser == VppRtgmcNoiseDenoiser::NLMeans;
}

static RGY_ERR rtgmcWaitEvents(cudaStream_t stream, const std::vector<RGYCudaEvent> &waitEvents) {
    for (const auto& waitEvent : waitEvents) {
        if (waitEvent() != nullptr) {
            const auto sts = err_to_rgy(cudaStreamWaitEvent(stream, waitEvent(), 0));
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
    }
    return RGY_ERR_NONE;
}

static RGY_ERR rtgmcRecordEvent(cudaStream_t stream, RGYCudaEvent *event) {
    if (!event) {
        return RGY_ERR_NONE;
    }
    auto cudaEvent = std::shared_ptr<cudaEvent_t>(new cudaEvent_t(), cudaevent_deleter());
    auto sts = err_to_rgy(cudaEventCreateWithFlags(cudaEvent.get(), cudaEventDisableTiming));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = err_to_rgy(cudaEventRecord(*cudaEvent, stream));
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    event->set(cudaEvent);
    return RGY_ERR_NONE;
}

static RGY_ERR rtgmcCopyFrameWithEvent(RGYFrameInfo *dst, const RGYFrameInfo *src, cudaStream_t stream,
    const std::vector<RGYCudaEvent> &waitEvents, RGYCudaEvent *event) {
    auto sts = rtgmcWaitEvents(stream, waitEvents);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = copyFrameAsync(dst, src, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return rtgmcRecordEvent(stream, event);
}

static RGY_ERR rtgmcRunFilterWithEvents(NVEncFilter *filter, RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &waitEvents, RGYCudaEvent *event) {
    if (!filter) {
        return RGY_ERR_NULL_PTR;
    }
    auto sts = rtgmcWaitEvents(stream, waitEvents);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = filter->filter(pInputFrame, ppOutputFrames, pOutputFrameNum, stream);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    return rtgmcRecordEvent(stream, event);
}
}

template<typename Type>
__global__ void kernel_rtgmc_border_edge(
    uint8_t *__restrict__ ptrDst,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ ptrSrc,
    const int srcPitch, const int srcWidth, const int srcHeight,
    const int borderRows) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < dstWidth && y < dstHeight) {
        int sy = y - borderRows;
        sy = min(max(sy, 0), srcHeight - 1);
        const Type *srcPix = (const Type *)(ptrSrc + sy * srcPitch + x * sizeof(Type));
        Type *dstPix = (Type *)(ptrDst + y * dstPitch + x * sizeof(Type));
        dstPix[0] = srcPix[0];
    }
}

template<typename Type>
__global__ void kernel_rtgmc_border_crop(
    uint8_t *__restrict__ ptrDst,
    const int dstPitch, const int dstWidth, const int dstHeight,
    const uint8_t *__restrict__ ptrSrc,
    const int srcPitch, const int srcWidth, const int srcHeight,
    const int cropRows) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < dstWidth && y < dstHeight) {
        const int sy = y + cropRows;
        const Type *srcPix = (const Type *)(ptrSrc + sy * srcPitch + x * sizeof(Type));
        Type *dstPix = (Type *)(ptrDst + y * dstPitch + x * sizeof(Type));
        dstPix[0] = srcPix[0];
    }
}

NVRtgmcSharedFramePool::NVRtgmcSharedFramePool(const char *allocStatsTag) :
    m_pool(),
    m_allocStatsTag(allocStatsTag) {
}

std::shared_ptr<CUFrameBuf> NVRtgmcSharedFramePool::acquire(const RGYFrameInfo *frame) {
    if (!frame) {
        return nullptr;
    }
    auto resetFrameState = [](RGYFrameInfo& frameInfo) {
        frameInfo.timestamp = 0;
        frameInfo.duration = 0;
        frameInfo.picstruct = RGY_PICSTRUCT_UNKNOWN;
        frameInfo.flags = RGY_FRAME_FLAG_NONE;
        frameInfo.inputFrameId = -1;
        frameInfo.dataList.clear();
    };
    auto matchesInfo = [frame](const std::shared_ptr<CUFrameBuf>& buf) {
        return buf
            && !cmpFrameInfoCspResolution(&buf->frame, frame)
            && RGY_CSP_BIT_DEPTH[buf->frame.csp] == RGY_CSP_BIT_DEPTH[frame->csp];
    };
    const bool knownSize = std::any_of(m_pool.begin(), m_pool.end(), matchesInfo);
    for (auto& buf : m_pool) {
        if (buf && buf.use_count() == 1 && matchesInfo(buf)) {
            resetFrameState(buf->frame);
            return buf;
        }
    }
    if (!knownSize) {
        const auto stale = std::find_if(m_pool.begin(), m_pool.end(), [&](const std::shared_ptr<CUFrameBuf>& buf) {
            return buf && buf.use_count() == 1 && !matchesInfo(buf);
        });
        if (stale != m_pool.end()) {
            m_pool.erase(stale);
        }
    }
    auto buf = std::make_shared<CUFrameBuf>(*frame);
    buf->releasePtr();
    RGYCudaAllocStatsTag allocStatsTagScope(m_allocStatsTag);
    if (buf->alloc() != RGY_ERR_NONE) {
        return nullptr;
    }
    resetFrameState(buf->frame);
    m_pool.push_back(buf);
    return buf;
}

void NVRtgmcSharedFramePool::clear() {
    m_pool.clear();
}

NVEncFilterRtgmc::NVEncFilterRtgmc() :
    NVEncFilter(),
    m_filters(),
    m_noiseFilter(nullptr),
    m_retouchCompFilters(),
    m_matchCorrectionPasses(),
    m_pendingEdiRefs(),
    m_pendingCompRefs(),
    m_pendingPostLimitBaseRefs(),
    m_pendingOutputFrames(),
    m_sourceCache(),
    m_sharedFramePool(std::make_shared<NVRtgmcSharedFramePool>()),
    m_ediSideDataFramePool(std::make_shared<NVRtgmcSharedFramePool>("RTGMC edi side-data frame pool")),
    m_borderFrame(),
    m_noiseDiffFilter(),
    m_pendingNoiseRefs(),
    m_inputFrame(),
    m_sourceCacheNext(0),
    m_outputBufferIndex(0),
    m_pendingSourceMatchFrameProps(),
    m_drainFilterIdx(0),
    m_draining(false),
    m_drainComplete(false),
    m_attachRetouchCompRefs(false),
    m_enablePostTR2Limit(false),
    m_sharedAnalysisMode(false),
    m_sharedData(),
    m_captureIntermediate(false),
    m_capturedIntermediates(),
    m_pendingIntermediateInputs(),
    m_debugResetAtFrame(0),
    m_nFrame(0) {
    m_name = _T("rtgmc");
    m_pathThrough = FILTER_PATHTHROUGH_NONE;
}

std::shared_ptr<CUFrameBuf> NVEncFilterRtgmc::getSharedFrameBuffer(const RGYFrameInfo *frame) {
    return m_sharedFramePool ? m_sharedFramePool->acquire(frame) : nullptr;
}

void NVEncFilterRtgmc::setSharedAnalysisData(const RtgmcSharedAnalysisData &data) {
    m_sharedData = data;
}

NVEncFilterRtgmc::RtgmcSharedAnalysisData NVEncFilterRtgmc::getSharedAnalysisData() {
    RtgmcSharedAnalysisData data;
    data.analyzeFilter = dynamic_cast<NVEncFilterDegrain *>(m_filters[RTGMC_FILTER_ANALYZE].get());
    data.pendingEdiRefs = &m_pendingEdiRefs;
    data.sourceCache = &m_sourceCache;
    data.pendingCompRefs = &m_pendingCompRefs;
    data.pendingNoiseRefs = &m_pendingNoiseRefs;
    data.sharedFramePool = m_sharedFramePool;
    return data;
}

void NVEncFilterRtgmc::enableIntermediateCapture(bool enable) {
    m_captureIntermediate = enable;
}

const std::vector<NVEncFilterRtgmc::RtgmcCapturedIntermediate>& NVEncFilterRtgmc::getCapturedIntermediates() const {
    return m_capturedIntermediates;
}

void NVEncFilterRtgmc::clearCapturedIntermediates() {
    m_capturedIntermediates.clear();
}

void NVEncFilterRtgmc::pushIntermediateInput(const RtgmcCapturedIntermediate &input) {
    auto queued = input;
    if (queued.frame && !queued.frameInfo.ptr[0]) {
        queued.frameInfo = queued.frame->frame;
        queued.frameInfo.dataList.push_back(std::make_shared<RGYFrameDataRtgmcFrameRef>(queued.frame));
    }
    m_pendingIntermediateInputs.push_back(queued);
}

NVEncFilterRtgmc::~NVEncFilterRtgmc() {
    close();
}

RGY_ERR NVEncFilterRtgmc::checkParam(const std::shared_ptr<NVEncFilterParamRtgmc> &prm) {
    if (!prm) {
        return RGY_ERR_NULL_PTR;
    }
    if (prm->frameIn.csp == RGY_CSP_NA) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid input colorspace.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->rtgmc.lossless < 0 || prm->rtgmc.lossless > 2) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc lossless must be 0 - 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->rtgmc.inputType < 0 || prm->rtgmc.inputType > 3) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc input_type must be 0 - 3.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->rtgmc.lossless > 0 && prm->rtgmc.inputType == 1) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc lossless modes cannot be used with input_type=1.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->rtgmc.progSADMask < 0.0f) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc prog_sad_mask must be 0.0 or larger.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->rtgmc.progSADMaskGamma <= 0.0f) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc prog_sad_mask_gamma must be larger than 0.0.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->rtgmc.mvSpatialRefine < -1) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc mv_spatial_refine must be -1 or greater.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (rtgmcInputTypeBlendEnabled(prm->rtgmc) && !RGY_HAS_RTGMC_MMASK_FILTER) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc input_type=2/3 with prog_sad_mask>0 requires rtgmc-mmask filter, but it is not available in this build.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->rtgmc.sourceMatch < 0 || prm->rtgmc.sourceMatch > 3) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc source_match must be 0 - 3.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->rtgmc.sourceMatch > 0 && !rtgmcSourceMatchCorrectionEdiSupported(prm->rtgmc.matchEdi.mode)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc source_match match_edi=%s is not yet supported. Supported: bob, yadif, cyadif, repyadif, repcyadif, nnedi3.\n"),
            get_cx_desc(list_vpp_rtgmc_edi_mode, (int)prm->rtgmc.matchEdi.mode));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->rtgmc.noise.noiseProcess < 0 || prm->rtgmc.noise.noiseProcess > 2) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc noise_process must be 0 - 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->rtgmc.noise.ezDenoise < 0.0f) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc ezdenoise must be 0 or larger.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->rtgmc.noise.sigma < 0.0f) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc sigma must be 0 or larger.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    if (prm->rtgmc.noise.noiseTR > 0) {
        AddMessage(RGY_LOG_WARN,
            _T("rtgmc noise_tr=%d is not yet supported, falling back to noise_tr=0.\n"),
            prm->rtgmc.noise.noiseTR);
        prm->rtgmc.noise.noiseTR = 0;
    }
    if (prm->rtgmc.noise.noiseProcess == 2 || prm->rtgmc.noise.ezKeepGrain > 0.0f
        || prm->rtgmc.noise.denoiseMC) {
        AddMessage(RGY_LOG_ERROR,
            _T("rtgmc noise_process=2, ezkeepgrain, and denoise_mc=true are not yet supported.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if ((prm->rtgmc.noise.grainRestore > 0.0f || prm->rtgmc.noise.noiseRestore > 0.0f) && prm->rtgmc.noise.noiseProcess != 1) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc grain_restore/noise_restore currently require noise_process=1.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->rtgmc.noise.grainRestore > 1.0f || prm->rtgmc.noise.noiseRestore > 1.0f) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc grain_restore/noise_restore above 1.0 are not yet supported.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (!rtgmcNoiseDenoiserSupported(prm->rtgmc.noise.denoiser)) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc denoiser=%s is unsupported; use nlmeans or fft3d.\n"),
            get_cx_desc(list_vpp_rtgmc_noise_denoiser, (int)prm->rtgmc.noise.denoiser));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->rtgmc.noise.noiseDeint == VppRtgmcNoiseDeint::Generate) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc noise_deint=generate is not yet supported.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    if (prm->rtgmc.matchTR1 < 0 || prm->rtgmc.matchTR1 > 2 || prm->rtgmc.matchTR2 < 0 || prm->rtgmc.matchTR2 > 2) {
        AddMessage(RGY_LOG_ERROR, _T("rtgmc match_tr1/match_tr2 must be 0 - 2.\n"));
        return RGY_ERR_INVALID_PARAM;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmc::initRetouchCompFilters(const std::shared_ptr<NVEncFilterParamRtgmc> &prm, const RGYFrameInfo &frameInfo, const rgy_rational<int> &baseFps) {
    m_attachRetouchCompRefs =
        prm->rtgmc.retouch.slrad > 0 &&
        (prm->rtgmc.retouch.slmode == 2 || prm->rtgmc.retouch.slmode == 4);
    for (auto &filter : m_retouchCompFilters) {
        filter.reset();
    }
    m_pendingEdiRefs.clear();
    m_pendingCompRefs.clear();
    if (!m_attachRetouchCompRefs) {
        return RGY_ERR_NONE;
    }

    static const std::array<VppDegrainMode, 2> modes = {
        VppDegrainMode::MotionBack,
        VppDegrainMode::MotionForw
    };
    for (size_t i = 0; i < m_retouchCompFilters.size(); i++) {
        auto filter = std::make_unique<NVEncFilterDegrain>();
        auto param = std::make_shared<NVEncFilterParamDegrain>();
        param->frameIn = frameInfo;
        param->frameOut = frameInfo;
        param->baseFps = baseFps;
        param->bOutOverwrite = false;
        param->degrain = prm->rtgmc.tr1;
        if (param->degrain.overlap != 0 && param->degrain.overlap * 2 != param->degrain.blksize) {
            AddMessage(RGY_LOG_WARN,
                _T("retouch helper overlap=%d is adjusted to %d because the current Degrain backend supports overlap=0 or blksize/2.\n"),
                param->degrain.overlap, param->degrain.blksize / 2);
            param->degrain.overlap = param->degrain.blksize / 2;
        }
        param->degrain.delta = 1;
        param->degrain.mode = modes[i];
        param->degrain.stage = VppDegrainStage::TR1;
        param->zeroCopyCache = true;
        auto sts = filter->init(param, m_pLog);
        if (sts != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to initialize retouch helper %s: %s.\n"),
                get_cx_desc(list_vpp_degrain_mode, (int)modes[i]), get_err_mes(sts));
            return sts;
        }
        m_retouchCompFilters[i] = std::move(filter);
    }
    AddMessage(RGY_LOG_DEBUG,
        _T("retouch slmode=%d helper path is enabled: EDI side-data and motionBack/motionForw delta=1 are cached for direct retouch handoff.\n"),
        prm->rtgmc.retouch.slmode);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmc::initSourceMatchCorrectionFilters(const std::shared_ptr<NVEncFilterParamRtgmc> &prm, const RGYFrameInfo &sourceFrameIn,
    const RGYFrameInfo &frameInfo, const rgy_rational<int> &sourceBaseFps, const rgy_rational<int> &sourceTimebase,
    const rgy_rational<int> &baseFps) {
    for (auto &stage : m_matchCorrectionPasses) {
        stage = RtgmcMatchCorrectionPass();
    }
    if (prm->rtgmc.sourceMatch <= 0) {
        return RGY_ERR_NONE;
    }
    const bool combineCorrectionKernels = rtgmcMatchCorrectionKernelMergeEnabled();
    const bool matchEdiProvidesChroma = prm->rtgmc.matchEdi.mode != VppRtgmcEdiMode::NNEDI3
        || prm->rtgmc.matchEdi.chromaEdi == VppRtgmcChromaEdiMode::NNEDI3;
    const bool processSourceMatchChroma = prm->rtgmc.searchPrefilter.chromaMotion && matchEdiProvidesChroma;

    auto initOne = [&](NVEncFilter *filter, const std::shared_ptr<NVEncFilterParam> &param) {
        param->frameIn = frameInfo;
        param->frameOut = frameInfo;
        param->baseFps = baseFps;
        param->bOutOverwrite = false;
        return filter->init(param, m_pLog);
    };
    for (int stageIdx = 0; stageIdx < prm->rtgmc.sourceMatch; stageIdx++) {
        auto &pass = m_matchCorrectionPasses[stageIdx];
        const int matchTR = (stageIdx == 0) ? prm->rtgmc.matchTR1 : 0;
        pass.fusedCorrectionBuild = combineCorrectionKernels;
        pass.fusedCorrectionApply = combineCorrectionKernels && matchTR == 0 && processSourceMatchChroma;
        pass.interpolator = std::make_unique<NVEncFilterRtgmcEdi>();
        {
            auto param = std::make_shared<NVEncFilterParamRtgmcEdi>();
            param->mode = prm->rtgmc.matchEdi.mode;
            param->chromaEdi = prm->rtgmc.matchEdi.chromaEdi;
            param->nnsize = prm->rtgmc.matchEdi.nnsize;
            param->nneurons = prm->rtgmc.matchEdi.nneurons;
            param->ediqual = prm->rtgmc.matchEdi.ediqual;
            param->order = prm->rtgmc.bob.order;
            param->sourceFrameIn = sourceFrameIn;
            param->sourceBaseFps = sourceBaseFps;
            param->sourceTimebase = sourceTimebase;
            auto sts = initOne(pass.interpolator.get(), param);
            if (sts != RGY_ERR_NONE) return sts;
        }
        pass.correctionBuild = std::make_unique<NVEncFilterRtgmcPrimitive>();
        {
            auto param = std::make_shared<NVEncFilterParamRtgmcPrimitive>();
            if (pass.fusedCorrectionApply) {
                param->op = RGYRtgmcPrimitiveOp::MakeDiffRemoveGrain20AddDiff;
            } else if (pass.fusedCorrectionBuild) {
                param->op = RGYRtgmcPrimitiveOp::MakeDiffRemoveGrain20;
            } else {
                param->op = RGYRtgmcPrimitiveOp::MakeDiff;
            }
            param->processChroma = processSourceMatchChroma;
            param->planes = processSourceMatchChroma ? 0x07 : 0x01;
            auto sts = initOne(pass.correctionBuild.get(), param);
            if (sts != RGY_ERR_NONE) return sts;
        }
        if (!pass.fusedCorrectionBuild) {
            pass.correctionSpatialPrepass = std::make_unique<NVEncFilterRtgmcPrimitive>();
            auto param = std::make_shared<NVEncFilterParamRtgmcPrimitive>();
            param->op = RGYRtgmcPrimitiveOp::RemoveGrain;
            param->mode = 20;
            param->processChroma = processSourceMatchChroma;
            param->planes = processSourceMatchChroma ? 0x07 : 0x01;
            auto sts = initOne(pass.correctionSpatialPrepass.get(), param);
            if (sts != RGY_ERR_NONE) return sts;
        }
        if (matchTR > 0) {
            pass.correctionTemporalFilter = std::make_unique<NVEncFilterDegrain>();
            auto param = std::make_shared<NVEncFilterParamDegrain>();
            param->degrain = (stageIdx == 0) ? prm->rtgmc.tr1 : prm->rtgmc.tr2;
            param->degrain.chroma = processSourceMatchChroma;
            if (param->degrain.overlap != 0 && param->degrain.overlap * 2 != param->degrain.blksize) {
                AddMessage(RGY_LOG_WARN,
                    _T("source-match correction overlap=%d is adjusted to %d because the current Degrain backend supports overlap=0 or blksize/2.\n"),
                    param->degrain.overlap, param->degrain.blksize / 2);
                param->degrain.overlap = param->degrain.blksize / 2;
            }
            param->degrain.delta = matchTR;
            param->degrain.mode = VppDegrainMode::Degrain;
            param->degrain.stage = (stageIdx == 0) ? VppDegrainStage::TR1 : VppDegrainStage::TR2;
            param->zeroCopyCache = true;
            auto sts = initOne(pass.correctionTemporalFilter.get(), param);
            if (sts != RGY_ERR_NONE) return sts;
        }
        if (!pass.fusedCorrectionApply) {
            pass.correctionApply = std::make_unique<NVEncFilterRtgmcPrimitive>();
            auto param = std::make_shared<NVEncFilterParamRtgmcPrimitive>();
            param->op = RGYRtgmcPrimitiveOp::AddDiff;
            param->processChroma = processSourceMatchChroma;
            param->planes = processSourceMatchChroma ? 0x07 : 0x01;
            auto sts = initOne(pass.correctionApply.get(), param);
            if (sts != RGY_ERR_NONE) return sts;
        }
        if (stageIdx == 0 && prm->rtgmc.matchEnhance > 0.0f) {
            pass.correctionEnhance = std::make_unique<NVEncFilterRtgmcRetouch>();
            auto param = std::make_shared<NVEncFilterParamRtgmcRetouch>();
            param->rtgmc_retouch = prm->rtgmc.retouch;
            param->rtgmc_retouch.sharpness = prm->rtgmc.matchEnhance;
            param->rtgmc_retouch.limit = 0.0f;
            param->rtgmc_retouch.smode = 2;
            param->rtgmc_retouch.slmode = 0;
            param->rtgmc_retouch.slrad = 0;
            param->rtgmc_retouch.sovs = 0;
            param->rtgmc_retouch.svthin = 0.0f;
            param->rtgmc_retouch.sbb = 0;
            param->processChroma = processSourceMatchChroma;
            auto sts = initOne(pass.correctionEnhance.get(), param);
            if (sts != RGY_ERR_NONE) return sts;
        }
    }
    AddMessage(RGY_LOG_DEBUG, _T("source-match correction is enabled: source_match=%d, match_tr1=%d, match_tr2=%d, match_edi=%s, match_enhance=%.3f.\n"),
        prm->rtgmc.sourceMatch, prm->rtgmc.matchTR1, prm->rtgmc.matchTR2,
        get_cx_desc(list_vpp_rtgmc_edi_mode, (int)prm->rtgmc.matchEdi.mode), prm->rtgmc.matchEnhance);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmc::attachEdiReference(RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!frame || !frame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    const auto attached = rtgmcGetAttachedEdi(frame);
    if (attached) {
        storeEdiReference(frame, attached, event ? *event : RGYCudaEvent());
        return RGY_ERR_NONE;
    }
    auto sharedFrame = m_ediSideDataFramePool ? m_ediSideDataFramePool->acquire(frame) : nullptr;
    if (!sharedFrame) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate edi side-data frame.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    auto err = rtgmcCopyFrameWithEvent(&sharedFrame->frame, frame, stream, wait_events, event);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy edi side-data frame: %s.\n"), get_err_mes(err));
        return err;
    }
    copyFramePropWithoutRes(&sharedFrame->frame, frame);
    auto frameData = std::make_shared<RGYFrameDataRtgmcEdi>(sharedFrame);
    frame->dataList.push_back(frameData);
    storeEdiReference(frame, frameData, event ? *event : RGYCudaEvent());
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmc::updateCompReferenceStore(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    if (!m_attachRetouchCompRefs || !frame || !frame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    NVEncFilterDegrain *analyze = nullptr;
    if (m_sharedAnalysisMode && m_sharedData.analyzeFilter) {
        analyze = m_sharedData.analyzeFilter;
    } else {
        analyze = dynamic_cast<NVEncFilterDegrain *>(m_filters[RTGMC_FILTER_ANALYZE].get());
    }
    if (!analyze) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid analyze filter instance for retouch helper.\n"));
        return RGY_ERR_INVALID_CALL;
    }

    std::array<RGYDegrainCompensateInlineParams, 3> backwardParams = {};
    std::array<RGYDegrainCompensateInlineParams, 3> forwardParams = {};
    bool backwardReady = false;
    bool forwardReady = false;
    RGYFrameInfo backwardIdentity = {};
    RGYFrameInfo forwardIdentity = {};

    for (size_t i = 0; i < m_retouchCompFilters.size(); i++) {
        auto filter = m_retouchCompFilters[i].get();
        if (!filter) {
            continue;
        }
        filter->setDirectAnalyzeResultSet(analyze->analyzeResultSet());

        auto sts = filter->feedFrameOnly(frame, stream, wait_events, nullptr);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        if (!filter->outputReady()) {
            continue;
        }

        auto &params = (i == 0) ? backwardParams : forwardParams;
        auto &identity = (i == 0) ? backwardIdentity : forwardIdentity;
        auto &ready = (i == 0) ? backwardReady : forwardReady;

        sts = filter->buildCompensateInlineParams(params, &identity, stream);
        if (sts == RGY_ERR_NONE) {
            ready = true;
        } else if (sts != RGY_ERR_MORE_DATA) {
            return sts;
        }
    }

    if (backwardReady || forwardReady) {
        const RGYFrameInfo *identityFrame = backwardReady ? &backwardIdentity : &forwardIdentity;
        auto compRef = findStoredCompReference(identityFrame);
        if (!compRef) {
            storeCompReference(identityFrame, nullptr, nullptr, RGYCudaEvent(), RGYCudaEvent());
            compRef = findStoredCompReference(identityFrame);
        }
        if (compRef) {
            compRef->hasInlineParams = true;
            compRef->backwardInlineParams = backwardParams;
            compRef->forwardInlineParams = forwardParams;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmc::drainCompReferenceStore(cudaStream_t stream) {
    if (!m_attachRetouchCompRefs) {
        return RGY_ERR_NONE;
    }
    NVEncFilterDegrain *analyze = nullptr;
    if (m_sharedAnalysisMode && m_sharedData.analyzeFilter) {
        analyze = m_sharedData.analyzeFilter;
    } else {
        analyze = dynamic_cast<NVEncFilterDegrain *>(m_filters[RTGMC_FILTER_ANALYZE].get());
    }
    if (!analyze) {
        AddMessage(RGY_LOG_ERROR, _T("Invalid analyze filter instance while draining retouch helpers.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    bool progress = true;
    while (progress) {
        progress = false;

        std::array<RGYDegrainCompensateInlineParams, 3> backwardParams = {};
        std::array<RGYDegrainCompensateInlineParams, 3> forwardParams = {};
        bool backwardReady = false;
        bool forwardReady = false;
        RGYFrameInfo backwardIdentity = {};
        RGYFrameInfo forwardIdentity = {};

        for (size_t i = 0; i < m_retouchCompFilters.size(); i++) {
            auto &filter = m_retouchCompFilters[i];
            if (!filter) {
                continue;
            }
            filter->setDirectAnalyzeResultSet(analyze->analyzeResultSet());

            if (!filter->drainReady()) {
                continue;
            }

            auto &params = (i == 0) ? backwardParams : forwardParams;
            auto &identity = (i == 0) ? backwardIdentity : forwardIdentity;
            auto &ready = (i == 0) ? backwardReady : forwardReady;

            auto sts = filter->drainBuildInlineParams(params, &identity, stream);
            if (sts == RGY_ERR_NONE) {
                ready = true;
                progress = true;
            } else if (sts != RGY_ERR_MORE_DATA) {
                return sts;
            }
        }

        if (backwardReady || forwardReady) {
            const RGYFrameInfo *identityFrame = backwardReady ? &backwardIdentity : &forwardIdentity;
            auto compRef = findStoredCompReference(identityFrame);
            if (!compRef) {
                storeCompReference(identityFrame, nullptr, nullptr, RGYCudaEvent(), RGYCudaEvent());
                compRef = findStoredCompReference(identityFrame);
            }
            if (compRef) {
                compRef->hasInlineParams = true;
                compRef->backwardInlineParams = backwardParams;
                compRef->forwardInlineParams = forwardParams;
            }
        }
    }
    return RGY_ERR_NONE;
}

void NVEncFilterRtgmc::attachStoredCompReferences(RGYFrameInfo *frame, std::vector<RGYCudaEvent> *wait_events) {
    if (!frame) {
        return;
    }
    RtgmcPendingCompRef *pending = nullptr;
    if (m_sharedAnalysisMode && m_sharedData.pendingCompRefs) {
        auto it = std::find_if(m_sharedData.pendingCompRefs->begin(), m_sharedData.pendingCompRefs->end(), [frame](const RtgmcPendingCompRef &entry) {
            return entry.key.matches(frame) || entry.key.matchesFrameIdentity(frame);
        });
        pending = (it != m_sharedData.pendingCompRefs->end()) ? &(*it) : nullptr;
    } else if (m_attachRetouchCompRefs) {
        pending = findStoredCompReference(frame);
    }
    if (!pending) {
        return;
    }
    if (pending->backward && rtgmcGetAttachedComp(frame, RGYRtgmcCompDirection::Backward, 1) == nullptr) {
        frame->dataList.push_back(pending->backward);
        if (wait_events && pending->backwardEvent() != nullptr) {
            wait_events->push_back(pending->backwardEvent);
        }
    }
    if (pending->forward && rtgmcGetAttachedComp(frame, RGYRtgmcCompDirection::Forward, 1) == nullptr) {
        frame->dataList.push_back(pending->forward);
        if (wait_events && pending->forwardEvent() != nullptr) {
            wait_events->push_back(pending->forwardEvent);
        }
    }
}

RGY_ERR NVEncFilterRtgmc::cacheSourceFrame(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    if (!frame || !frame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    if (m_sharedAnalysisMode) {
        return RGY_ERR_NONE;
    }
    auto &entry = m_sourceCache[m_sourceCacheNext];
    if (!entry.frame || cmpFrameInfoCspResolution(&entry.frame->frame, frame)) {
        entry.frame = std::make_unique<CUFrameBuf>(*frame);
        entry.frame->releasePtr();
        RGYCudaAllocStatsTag allocStatsTagScope("RTGMC source cache");
        if (!entry.frame || entry.frame->alloc() != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to allocate source cache frame.\n"));
            return RGY_ERR_MEMORY_ALLOC;
        }
    }
    RGYCudaEvent copyEvent;
    auto err = rtgmcCopyFrameWithEvent(&entry.frame->frame, frame, stream, wait_events, &copyEvent);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to cachesource frame: %s.\n"), get_err_mes(err));
        return err;
    }
    copyFramePropWithoutRes(&entry.frame->frame, frame);
    entry.key = RtgmcFrameKey(frame);
    entry.event = copyEvent;
    m_sourceCacheNext = (m_sourceCacheNext + 1) % (int)m_sourceCache.size();
    return RGY_ERR_NONE;
}

const RGYFrameInfo *NVEncFilterRtgmc::findCachedSourceFrame(const RGYFrameInfo *frame, std::vector<RGYCudaEvent> *wait_events) {
    if (!frame) {
        return nullptr;
    }
    const auto &cache = m_sharedAnalysisMode && m_sharedData.sourceCache ? *m_sharedData.sourceCache : m_sourceCache;
    auto cached = std::find_if(cache.begin(), cache.end(), [frame](const RtgmcSourceCacheFrame &entry) {
        return entry.frame && entry.key.inputFrameId == frame->inputFrameId;
    });
    if (cached == cache.end()) {
        return nullptr;
    }
    if (wait_events && cached->event() != nullptr) {
        wait_events->push_back(cached->event);
    }
    return &cached->frame->frame;
}

int NVEncFilterRtgmc::sourceFieldForFrame(const RGYFrameInfo *frame) const {
    if (!frame) {
        return 0;
    }
    const auto rtgmcParam = std::dynamic_pointer_cast<NVEncFilterParamRtgmc>(m_param);
    const auto &cache = (m_sharedAnalysisMode && m_sharedData.sourceCache) ? *m_sharedData.sourceCache : m_sourceCache;
    auto cached = std::find_if(cache.begin(), cache.end(), [frame](const RtgmcSourceCacheFrame &entry) {
        return entry.frame && entry.key.inputFrameId == frame->inputFrameId;
    });
    bool tff = true;
    if (rtgmcParam) {
        if (rtgmcParam->rtgmc.bob.order == VppRtgmcBobOrder::BFF) {
            tff = false;
        } else if (rtgmcParam->rtgmc.bob.order == VppRtgmcBobOrder::TFF) {
            tff = true;
        } else if (cached != cache.end() && cached->frame) {
            tff = (cached->frame->frame.picstruct & RGY_PICSTRUCT_BFF) == 0;
        } else {
            tff = (frame->picstruct & RGY_PICSTRUCT_BFF) == 0;
        }
    }
    if (cached == cache.end() || cached->key.duration <= 0) {
        return tff ? 0 : 1;
    }
    const auto halfDuration = (cached->key.duration + 1) / 2;
    return (frame->timestamp >= cached->key.timestamp + halfDuration)
        ? (tff ? 1 : 0)
        : (tff ? 0 : 1);
}

void NVEncFilterRtgmc::storeEdiReference(const RGYFrameInfo *frame, const std::shared_ptr<RGYFrameDataRtgmcEdi> &edi, const RGYCudaEvent &event) {
    if (!frame || !edi) {
        return;
    }
    const auto key = RtgmcFrameKey(frame);
    auto pending = std::find_if(m_pendingEdiRefs.begin(), m_pendingEdiRefs.end(), [&key](const RtgmcPendingEdiRef &entry) {
        return entry.key.inputFrameId == key.inputFrameId
            && entry.key.timestamp == key.timestamp
            && entry.key.duration == key.duration;
    });
    if (pending == m_pendingEdiRefs.end()) {
        m_pendingEdiRefs.push_back(RtgmcPendingEdiRef{ key, edi, event });
    } else {
        pending->edi = edi;
        pending->event = event;
    }
    while (m_pendingEdiRefs.size() > RTGMC_MAX_STORED_EDI_REFS) {
        m_pendingEdiRefs.pop_front();
    }
}

NVEncFilterRtgmc::RtgmcPendingEdiRef *NVEncFilterRtgmc::findStoredEdiReference(const RGYFrameInfo *frame) {
    if (!frame) {
        return nullptr;
    }
    auto pending = std::find_if(m_pendingEdiRefs.begin(), m_pendingEdiRefs.end(), [frame](const RtgmcPendingEdiRef &entry) {
        return entry.key.matches(frame);
    });
    if (pending != m_pendingEdiRefs.end()) {
        return &(*pending);
    }
    pending = std::find_if(m_pendingEdiRefs.begin(), m_pendingEdiRefs.end(), [frame](const RtgmcPendingEdiRef &entry) {
        return entry.key.matchesFrameIdentity(frame);
    });
    return (pending != m_pendingEdiRefs.end()) ? &(*pending) : nullptr;
}

void NVEncFilterRtgmc::clearStoredEdiReference(const RGYFrameInfo *frame) {
    if (!frame) {
        return;
    }
    m_pendingEdiRefs.erase(std::remove_if(m_pendingEdiRefs.begin(), m_pendingEdiRefs.end(), [frame](const RtgmcPendingEdiRef &entry) {
        return entry.key.matches(frame) || entry.key.matchesFrameIdentity(frame);
    }), m_pendingEdiRefs.end());
}

void NVEncFilterRtgmc::storeCompReference(const RGYFrameInfo *frame, const std::shared_ptr<RGYFrameData> &backward, const std::shared_ptr<RGYFrameData> &forward,
    const RGYCudaEvent &backwardEvent, const RGYCudaEvent &forwardEvent) {
    if (!frame) {
        return;
    }
    const auto key = RtgmcFrameKey(frame);
    auto pending = std::find_if(m_pendingCompRefs.begin(), m_pendingCompRefs.end(), [frame](const RtgmcPendingCompRef &entry) {
        return entry.key.matches(frame) || entry.key.matchesFrameIdentity(frame);
    });
    if (pending == m_pendingCompRefs.end()) {
        m_pendingCompRefs.push_back(RtgmcPendingCompRef());
        pending = std::prev(m_pendingCompRefs.end());
        pending->key = key;
    }
    if (backward) {
        pending->backward = backward;
        pending->backwardEvent = backwardEvent;
    }
    if (forward) {
        pending->forward = forward;
        pending->forwardEvent = forwardEvent;
    }
}

NVEncFilterRtgmc::RtgmcPendingCompRef *NVEncFilterRtgmc::findStoredCompReference(const RGYFrameInfo *frame) {
    if (!frame) {
        return nullptr;
    }
    auto pending = std::find_if(m_pendingCompRefs.begin(), m_pendingCompRefs.end(), [frame](const RtgmcPendingCompRef &entry) {
        return entry.key.matches(frame) || entry.key.matchesFrameIdentity(frame);
    });
    return (pending != m_pendingCompRefs.end()) ? &(*pending) : nullptr;
}

void NVEncFilterRtgmc::clearStoredCompReferences(const RGYFrameInfo *frame) {
    if (!frame) {
        return;
    }
    m_pendingCompRefs.erase(std::remove_if(m_pendingCompRefs.begin(), m_pendingCompRefs.end(), [frame](const RtgmcPendingCompRef &entry) {
        return entry.key.matches(frame) || entry.key.matchesFrameIdentity(frame);
    }), m_pendingCompRefs.end());
}

RGY_ERR NVEncFilterRtgmc::storePostLimitBaseReference(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    if (!m_enablePostTR2Limit || !frame || !frame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    auto sharedFrame = getSharedFrameBuffer(frame);
    if (!sharedFrame) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocatepost-TR2 limit base frame.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    RGYCudaEvent copyEvent;
    auto err = rtgmcCopyFrameWithEvent(&sharedFrame->frame, frame, stream, wait_events, &copyEvent);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copypost-TR2 limit base frame: %s.\n"), get_err_mes(err));
        return err;
    }
    copyFramePropWithoutRes(&sharedFrame->frame, frame);
    const auto key = RtgmcFrameKey(frame);
    auto pending = std::find_if(m_pendingPostLimitBaseRefs.begin(), m_pendingPostLimitBaseRefs.end(), [frame](const RtgmcPendingFrameRef &entry) {
        return entry.key.matches(frame) || entry.key.matchesFrameIdentity(frame);
    });
    if (pending == m_pendingPostLimitBaseRefs.end()) {
        m_pendingPostLimitBaseRefs.push_back(RtgmcPendingFrameRef());
        pending = std::prev(m_pendingPostLimitBaseRefs.end());
        pending->key = key;
    }
    pending->frame = std::move(sharedFrame);
    pending->event = copyEvent;
    while (m_pendingPostLimitBaseRefs.size() > RTGMC_MAX_STORED_POST_LIMIT_BASE_REFS) {
        m_pendingPostLimitBaseRefs.pop_front();
    }
    return RGY_ERR_NONE;
}

NVEncFilterRtgmc::RtgmcPendingFrameRef *NVEncFilterRtgmc::findPostLimitBaseReference(const RGYFrameInfo *frame) {
    if (!frame) {
        return nullptr;
    }
    auto pending = std::find_if(m_pendingPostLimitBaseRefs.begin(), m_pendingPostLimitBaseRefs.end(), [frame](const RtgmcPendingFrameRef &entry) {
        return entry.key.matches(frame) || entry.key.matchesFrameIdentity(frame);
    });
    return (pending != m_pendingPostLimitBaseRefs.end()) ? &(*pending) : nullptr;
}

void NVEncFilterRtgmc::clearPostLimitBaseReference(const RGYFrameInfo *frame) {
    if (!frame) {
        return;
    }
    m_pendingPostLimitBaseRefs.erase(std::remove_if(m_pendingPostLimitBaseRefs.begin(), m_pendingPostLimitBaseRefs.end(), [frame](const RtgmcPendingFrameRef &entry) {
        return entry.key.matches(frame) || entry.key.matchesFrameIdentity(frame);
    }), m_pendingPostLimitBaseRefs.end());
}

RGY_ERR NVEncFilterRtgmc::storeMatchCorrectionBaseReference(int stage, const RGYFrameInfo *keyFrame, const RGYFrameInfo *baseFrame,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    if (stage < 0 || stage >= (int)m_matchCorrectionPasses.size() || !keyFrame || !keyFrame->ptr[0] || !baseFrame || !baseFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    auto sharedFrame = getSharedFrameBuffer(baseFrame);
    if (!sharedFrame) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocate source-match correction base frame.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    RGYCudaEvent copyEvent;
    auto err = rtgmcCopyFrameWithEvent(&sharedFrame->frame, baseFrame, stream, wait_events, &copyEvent);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copy source-match correction base frame: %s.\n"), get_err_mes(err));
        return err;
    }
    copyFramePropWithoutRes(&sharedFrame->frame, baseFrame);
    auto &baseRefs = m_matchCorrectionPasses[stage].composeBaseRefs;
    const auto key = RtgmcFrameKey(keyFrame);
    auto pending = std::find_if(baseRefs.begin(), baseRefs.end(), [keyFrame](const RtgmcPendingFrameRef &entry) {
        return entry.key.matches(keyFrame) || entry.key.matchesFrameIdentity(keyFrame);
    });
    if (pending == baseRefs.end()) {
        baseRefs.push_back(RtgmcPendingFrameRef());
        pending = std::prev(baseRefs.end());
        pending->key = key;
    }
    pending->frame = std::move(sharedFrame);
    pending->event = copyEvent;
    while (baseRefs.size() > RTGMC_MAX_STORED_MATCH_CORRECTION_BASE_REFS) {
        baseRefs.pop_front();
    }
    return RGY_ERR_NONE;
}

NVEncFilterRtgmc::RtgmcPendingFrameRef *NVEncFilterRtgmc::findMatchCorrectionBaseReference(int stage, const RGYFrameInfo *frame) {
    if (stage < 0 || stage >= (int)m_matchCorrectionPasses.size() || !frame) {
        return nullptr;
    }
    auto &baseRefs = m_matchCorrectionPasses[stage].composeBaseRefs;
    auto pending = std::find_if(baseRefs.begin(), baseRefs.end(), [frame](const RtgmcPendingFrameRef &entry) {
        return entry.key.matches(frame) || entry.key.matchesFrameIdentity(frame);
    });
    return (pending != baseRefs.end()) ? &(*pending) : nullptr;
}

void NVEncFilterRtgmc::clearMatchCorrectionBaseReference(int stage, const RGYFrameInfo *frame) {
    if (stage < 0 || stage >= (int)m_matchCorrectionPasses.size() || !frame) {
        return;
    }
    auto &baseRefs = m_matchCorrectionPasses[stage].composeBaseRefs;
    baseRefs.erase(std::remove_if(baseRefs.begin(), baseRefs.end(), [frame](const RtgmcPendingFrameRef &entry) {
        return entry.key.matches(frame) || entry.key.matchesFrameIdentity(frame);
    }), baseRefs.end());
}

void NVEncFilterRtgmc::enqueueSourceMatchFrameProp(const RGYFrameInfo *frame) {
    if (frame) {
        m_pendingSourceMatchFrameProps.emplace_back(frame);
    }
}

RGY_ERR NVEncFilterRtgmc::applySourceMatchFrameProp(RGYFrameInfo *frame) {
    if (!frame) {
        return RGY_ERR_INVALID_PARAM;
    }
    if (m_pendingSourceMatchFrameProps.empty()) {
        AddMessage(RGY_LOG_ERROR, _T("source match output frame property stream is empty.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    const auto key = m_pendingSourceMatchFrameProps.front();
    m_pendingSourceMatchFrameProps.pop_front();
    frame->inputFrameId = key.inputFrameId;
    frame->timestamp = key.timestamp;
    frame->duration = key.duration;
    return RGY_ERR_NONE;
}

void NVEncFilterRtgmc::attachStoredEdiReference(RGYFrameInfo *frame, std::vector<RGYCudaEvent> *wait_events) {
    if (!frame || rtgmcGetAttachedEdi(frame) != nullptr) {
        return;
    }
    RtgmcPendingEdiRef *pending = nullptr;
    if (m_sharedAnalysisMode && m_sharedData.pendingEdiRefs) {
        auto it = std::find_if(m_sharedData.pendingEdiRefs->begin(), m_sharedData.pendingEdiRefs->end(), [frame](const RtgmcPendingEdiRef &entry) {
            return entry.key.matches(frame) || entry.key.matchesFrameIdentity(frame);
        });
        pending = (it != m_sharedData.pendingEdiRefs->end()) ? &(*it) : nullptr;
    } else {
        pending = findStoredEdiReference(frame);
    }
    if (!pending || !pending->edi) {
        return;
    }
    frame->dataList.push_back(pending->edi);
    if (wait_events && pending->event() != nullptr) {
        wait_events->push_back(pending->event);
    }
}

RGY_ERR NVEncFilterRtgmc::initBorderFrame(const RGYFrameInfo &frameInfo) {
    RGYFrameInfo borderFrameInfo = frameInfo;
    borderFrameInfo.height += RGY_RTGMC_BORDER_PIXELS_Y * 2;
    m_borderFrame = std::make_unique<CUFrameBuf>(borderFrameInfo);
    m_borderFrame->releasePtr();
    if (!m_borderFrame || m_borderFrame->alloc() != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocateborder frame.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    return buildBorderCopyProgram(borderFrameInfo);
}

RGY_ERR NVEncFilterRtgmc::buildBorderCopyProgram(const RGYFrameInfo &frameInfo) {
    UNREFERENCED_PARAMETER(frameInfo);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmc::runBorderCopy(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame, bool crop,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    RGYCudaEvent prevEvent;
    std::vector<RGYCudaEvent> planeWaitEvents = wait_events;
    const int planes = RGY_CSP_PLANES[pInputFrame->csp];
    const int bitDepth = RGY_CSP_BIT_DEPTH[pInputFrame->csp];
    for (int iplane = 0; iplane < planes; iplane++) {
        const auto planeIn = getPlane(pInputFrame, (RGY_PLANE)iplane);
        auto planeOut = getPlane(pOutputFrame, (RGY_PLANE)iplane);
        const int edgeRows = crop ? (planeIn.height - planeOut.height) / 2 : (planeOut.height - planeIn.height) / 2;
        const bool validPlaneSize = planeOut.width == planeIn.width
            && edgeRows >= 0
            && ((crop && planeIn.height == planeOut.height + edgeRows * 2)
                || (!crop && planeOut.height == planeIn.height + edgeRows * 2));
        if (!validPlaneSize) {
            AddMessage(RGY_LOG_ERROR, _T("invalidborder/crop plane size: in %dx%d, out %dx%d.\n"),
                planeIn.width, planeIn.height, planeOut.width, planeOut.height);
            return RGY_ERR_INVALID_PARAM;
        }
        RGYCudaEvent planeEvent;
        auto err = rtgmcWaitEvents(stream, planeWaitEvents);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error waiting for %s(%s): %s.\n"),
                crop ? _T("kernel_rtgmc_border_crop") : _T("kernel_rtgmc_border_edge"),
                RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(err));
            return err;
        }
        const dim3 blockSize(RGY_RTGMC_BORDER_BLOCK_X, RGY_RTGMC_BORDER_BLOCK_Y);
        const dim3 gridSize(divCeil(planeOut.width, blockSize.x), divCeil(planeOut.height, blockSize.y));
#define LAUNCH_RTGMC_BORDER_KERNEL(kernel, ...) \
        do { \
            if (bitDepth <= 8) { \
                kernel<uint8_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__); \
            } else { \
                kernel<uint16_t><<<gridSize, blockSize, 0, stream>>>(__VA_ARGS__); \
            } \
        } while (0)
        if (crop) {
            LAUNCH_RTGMC_BORDER_KERNEL(kernel_rtgmc_border_crop,
                planeOut.ptr[0], planeOut.pitch[0], planeOut.width, planeOut.height,
                planeIn.ptr[0], planeIn.pitch[0], planeIn.width, planeIn.height,
                edgeRows);
        } else {
            LAUNCH_RTGMC_BORDER_KERNEL(kernel_rtgmc_border_edge,
                planeOut.ptr[0], planeOut.pitch[0], planeOut.width, planeOut.height,
                planeIn.ptr[0], planeIn.pitch[0], planeIn.width, planeIn.height,
                edgeRows);
        }
#undef LAUNCH_RTGMC_BORDER_KERNEL
        err = err_to_rgy(cudaGetLastError());
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("error at %s(%s): %s.\n"),
                crop ? _T("kernel_rtgmc_border_crop") : _T("kernel_rtgmc_border_edge"),
                RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(err));
            return err;
        }
        err = rtgmcRecordEvent(stream, &planeEvent);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to record %s event(%s): %s.\n"),
                crop ? _T("kernel_rtgmc_border_crop") : _T("kernel_rtgmc_border_edge"),
                RGY_CSP_NAMES[pInputFrame->csp], get_err_mes(err));
            return err;
        }
        prevEvent = planeEvent;
        planeWaitEvents.clear();
        if (prevEvent() != nullptr) {
            planeWaitEvents.push_back(prevEvent);
        }
    }
    if (event && prevEvent() != nullptr) {
        *event = prevEvent;
    }
    copyFramePropWithoutRes(pOutputFrame, pInputFrame);
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmc::addBorderToInput(const RGYFrameInfo *frame, cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    if (!m_borderFrame) {
        AddMessage(RGY_LOG_ERROR, _T("border frame is not initialized.\n"));
        return RGY_ERR_INVALID_CALL;
    }
    return runBorderCopy(&m_borderFrame->frame, frame, false, stream, wait_events, event);
}

RGY_ERR NVEncFilterRtgmc::copyFinalOutput(RGYFrameInfo *pOutputFrame, const RGYFrameInfo *pInputFrame,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    const auto rtgmcParam = std::dynamic_pointer_cast<NVEncFilterParamRtgmc>(m_param);
    if (rtgmcParam && rtgmcParam->rtgmc.border) {
        return runBorderCopy(pOutputFrame, pInputFrame, true, stream, wait_events, event);
    }
    return rtgmcCopyFrameWithEvent(pOutputFrame, pInputFrame, stream, wait_events, event);
}

bool NVEncFilterRtgmc::noiseRestoreEnabled() const {
    const auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmc>(m_param);
    return prm
        && prm->rtgmc.noise.noiseProcess == 1
        && (prm->rtgmc.noise.grainRestore > 0.0f || prm->rtgmc.noise.noiseRestore > 0.0f);
}

RGY_ERR NVEncFilterRtgmc::storeNoiseReference(const RGYFrameInfo *baseFrame, RGYFrameInfo *denoisedFrame,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events) {
    if (!noiseRestoreEnabled() || !m_noiseDiffFilter || !baseFrame || !denoisedFrame || !baseFrame->ptr[0] || !denoisedFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    int diffOutNum = 0;
    RGYFrameInfo *diffOutFrames[RGY_RTGMC_MAX_OUT_FRAMES] = { 0 };
    RGYCudaEvent diffEvent;
    auto sts = m_noiseDiffFilter->run_filter(baseFrame, denoisedFrame, diffOutFrames, &diffOutNum, stream, wait_events, &diffEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    if (diffOutNum <= 0 || !diffOutFrames[0] || !diffOutFrames[0]->ptr[0]) {
        return RGY_ERR_NONE;
    }
    auto sharedFrame = getSharedFrameBuffer(diffOutFrames[0]);
    if (!sharedFrame) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocatenoise reference frame.\n"));
        return RGY_ERR_MEMORY_ALLOC;
    }
    std::vector<RGYCudaEvent> copyWaitEvents;
    if (diffEvent() != nullptr) {
        copyWaitEvents.push_back(diffEvent);
    }
    RGYCudaEvent copyEvent;
    auto err = rtgmcCopyFrameWithEvent(&sharedFrame->frame, diffOutFrames[0], stream, copyWaitEvents, &copyEvent);
    if (err != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to copynoise reference frame: %s.\n"), get_err_mes(err));
        return err;
    }
    copyFramePropWithoutRes(&sharedFrame->frame, denoisedFrame);
    denoisedFrame->dataList.push_back(std::make_shared<RGYFrameDataRtgmcNoise>(sharedFrame, copyEvent));
    auto pending = std::find_if(m_pendingNoiseRefs.begin(), m_pendingNoiseRefs.end(), [denoisedFrame](const RtgmcPendingFrameRef &entry) {
        return entry.key.matches(denoisedFrame) || entry.key.matchesFrameIdentity(denoisedFrame);
    });
    if (pending == m_pendingNoiseRefs.end()) {
        m_pendingNoiseRefs.push_back(RtgmcPendingFrameRef());
        pending = std::prev(m_pendingNoiseRefs.end());
        pending->key = RtgmcFrameKey(denoisedFrame);
    }
    pending->frame = sharedFrame;
    pending->event = copyEvent;
    while (m_pendingNoiseRefs.size() > RTGMC_MAX_STORED_EDI_REFS) {
        m_pendingNoiseRefs.pop_front();
    }
    return RGY_ERR_NONE;
}

NVEncFilterRtgmc::RtgmcPendingFrameRef *NVEncFilterRtgmc::findNoiseReference(const RGYFrameInfo *frame) {
    if (!frame) {
        return nullptr;
    }
    auto &noiseRefs = (m_sharedAnalysisMode && m_sharedData.pendingNoiseRefs) ? *m_sharedData.pendingNoiseRefs : m_pendingNoiseRefs;
    auto pending = std::find_if(noiseRefs.begin(), noiseRefs.end(), [frame](const RtgmcPendingFrameRef &entry) {
        return entry.key.matches(frame) || entry.key.matchesFrameIdentity(frame);
    });
    return (pending != noiseRefs.end()) ? &(*pending) : nullptr;
}

void NVEncFilterRtgmc::clearNoiseReference(const RGYFrameInfo *frame) {
    if (!frame) {
        return;
    }
    m_pendingNoiseRefs.erase(std::remove_if(m_pendingNoiseRefs.begin(), m_pendingNoiseRefs.end(), [frame](const RtgmcPendingFrameRef &entry) {
        return entry.key.matches(frame) || entry.key.matchesFrameIdentity(frame);
    }), m_pendingNoiseRefs.end());
}

RGY_ERR NVEncFilterRtgmc::initFilters(const std::shared_ptr<NVEncFilterParamRtgmc> &prm) {
    RGYFrameInfo currentFrame = prm->frameIn;
    m_noiseFilter = nullptr;
    m_noiseDiffFilter.reset();
    if (prm->rtgmc.border) {
        auto sts = initBorderFrame(currentFrame);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        currentFrame = m_borderFrame->frame;
    }
    auto currentFps = prm->baseFps;
    const int nestedAnalyzeDelta = rtgmcNestedAnalyzeDelta(prm->rtgmc);
    m_enablePostTR2Limit = prm->rtgmc.retouch.slmode == 3 || prm->rtgmc.retouch.slmode == 4;

    if (prm->rtgmc.analyze.delta != nestedAnalyzeDelta) {
        AddMessage(RGY_LOG_WARN,
            _T("nested analyze delta=%d is adjusted to delta=%d to match the currently implemented shared MV/SAD references.\n"),
            prm->rtgmc.analyze.delta, nestedAnalyzeDelta);
    }
    auto initOne = [&](std::unique_ptr<NVEncFilter> filter, std::shared_ptr<NVEncFilterParam> childParam) {
        childParam->frameIn = currentFrame;
        childParam->frameOut = currentFrame;
        childParam->baseFps = currentFps;
        childParam->bOutOverwrite = false;
        auto sts = filter->init(childParam, m_pLog);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        currentFrame = childParam->frameOut;
        currentFps = childParam->baseFps;
        m_filters.push_back(std::move(filter));
        return RGY_ERR_NONE;
    };
    auto initBypass = [&]() {
        m_filters.push_back(nullptr);
        return RGY_ERR_NONE;
    };
    auto rtgDegrainRuntimeParam = [&](VppDegrain degrain, const TCHAR *stage) {
        if (degrain.overlap != 0 && degrain.overlap * 2 != degrain.blksize) {
            AddMessage(RGY_LOG_WARN,
                _T("%s overlap=%d is adjusted to %d because the current Degrain backend supports overlap=0 or blksize/2.\n"),
                stage, degrain.overlap, degrain.blksize / 2);
            degrain.overlap = degrain.blksize / 2;
        }
        return degrain;
    };

    if (m_sharedAnalysisMode) {
        // shared stages: BOB/SEARCH_PREFILTER/ANALYZE/NOISE/EDI/INPUTTYPE_BLEND are all bypassed
        for (int i = RTGMC_FILTER_BOB; i <= RTGMC_FILTER_INPUTTYPE_BLEND; i++) {
            auto sts = initBypass();
            if (sts != RGY_ERR_NONE) return sts;
        }
    } else {
        {
            auto filter = std::make_unique<NVEncFilterRtgmcBob>();
            auto param = std::make_shared<NVEncFilterParamRtgmcBob>();
            param->order = (prm->rtgmc.bob.order == VppRtgmcBobOrder::TFF) ? RGYRtgmcBobFieldOrder::TFF
                : (prm->rtgmc.bob.order == VppRtgmcBobOrder::BFF) ? RGYRtgmcBobFieldOrder::BFF
                : RGYRtgmcBobFieldOrder::Auto;
            param->timebase = prm->timebase;
            auto sts = initOne(std::move(filter), param);
            if (sts != RGY_ERR_NONE) return sts;
        }
        {
            auto filter = std::make_unique<NVEncFilterRtgmcSearchPrefilter>();
            auto param = std::make_shared<NVEncFilterParamRtgmcSearchPrefilter>();
            param->tr0 = prm->rtgmc.searchPrefilter.tr0;
            param->rep0Thin = prm->rtgmc.searchPrefilter.rep0Thin;
            param->rep0Pad = prm->rtgmc.searchPrefilter.rep0Pad;
            param->searchRefine = prm->rtgmc.searchPrefilter.searchRefine;
            param->tvRange = prm->rtgmc.searchPrefilter.tvRange;
            param->chromaMotion = prm->rtgmc.searchPrefilter.chromaMotion;
            // Search-luma side data is kept only inside the nested filter chain.
            param->attachSearchLuma = true;
            auto sts = initOne(std::move(filter), param);
            if (sts != RGY_ERR_NONE) return sts;
        }
        {
            auto filter = std::make_unique<NVEncFilterDegrain>();
            auto param = std::make_shared<NVEncFilterParamDegrain>();
            param->degrain = rtgDegrainRuntimeParam(prm->rtgmc.analyze, _T("analyze"));
            param->degrain.delta = nestedAnalyzeDelta;
            param->degrain.mode = VppDegrainMode::Analyze;
            param->degrain.stage = VppDegrainStage::TR1;
            // TR1/TR2 have their own temporal delay, so a single latest direct result
            // cannot identify the frame being emitted by those filters. Keep the
            // analysis payload internal to the nested chain and erase it at final output.
            param->attachAnalysisData = true;
            param->zeroCopyCache = true;
            auto sts = initOne(std::move(filter), param);
            if (sts != RGY_ERR_NONE) return sts;
        }
        {
            std::unique_ptr<NVEncFilter> filter;
            if (prm->rtgmc.noise.noiseProcess == 1 && rtgmcNoiseDenoiserUsesNLMeans(prm->rtgmc.noise.denoiser)) {
                auto noise = std::make_unique<NVEncFilterDenoiseNLMeans>();
                auto noiseParam = std::make_shared<NVEncFilterParamDenoiseNLMeans>();
                noiseParam->frameIn = currentFrame;
                noiseParam->frameOut = currentFrame;
                noiseParam->baseFps = currentFps;
                noiseParam->bOutOverwrite = false;
                noiseParam->nlmeans.sigma = prm->rtgmc.noise.sigma / 255.0f;
                noiseParam->nlmeans.processChroma = prm->rtgmc.noise.chromaNoise;
                auto sts = noise->init(noiseParam, m_pLog);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                m_noiseFilter = noise.get();
                filter = std::move(noise);
                currentFrame = noiseParam->frameOut;
                currentFps = noiseParam->baseFps;
            } else if (prm->rtgmc.noise.noiseProcess == 1) {
                AddMessage(RGY_LOG_DEBUG, _T("denoiser=%s is mapped to QSVEnc fft3d.\n"),
                    get_cx_desc(list_vpp_rtgmc_noise_denoiser, (int)prm->rtgmc.noise.denoiser));
                auto noise = std::make_unique<NVEncFilterDenoiseFFT3D>();
                auto noiseParam = std::make_shared<NVEncFilterParamDenoiseFFT3D>();
                noiseParam->frameIn = currentFrame;
                noiseParam->frameOut = currentFrame;
                noiseParam->baseFps = currentFps;
                noiseParam->bOutOverwrite = false;
                noiseParam->fft3d.enable = true;
                noiseParam->fft3d.sigma = prm->rtgmc.noise.sigma;
                noiseParam->fft3d.amount = 1.0f;
                noiseParam->fft3d.block_size = FILTER_DEFAULT_DENOISE_FFT3D_BLOCK_SIZE;
                noiseParam->fft3d.overlap = FILTER_DEFAULT_DENOISE_FFT3D_OVERLAP;
                noiseParam->fft3d.overlap2 = FILTER_DEFAULT_DENOISE_FFT3D_OVERLAP2;
                noiseParam->fft3d.method = FILTER_DEFAULT_DENOISE_FFT3D_METHOD;
                noiseParam->fft3d.temporal = 0;
                noiseParam->fft3d.precision = VppFpPrecision::VPP_FP_PRECISION_AUTO;
                noiseParam->processChroma = prm->rtgmc.noise.chromaNoise;
                auto sts = noise->init(noiseParam, m_pLog);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                m_noiseFilter = noise.get();
                filter = std::move(noise);
                currentFrame = noiseParam->frameOut;
                currentFps = noiseParam->baseFps;
            } else {
                m_noiseFilter = nullptr;
                auto sts = initBypass();
                if (sts != RGY_ERR_NONE) return sts;
            }
            if (filter) {
                m_filters.push_back(std::move(filter));
            }
        }
        if (prm->rtgmc.noise.grainRestore > 0.0f || prm->rtgmc.noise.noiseRestore > 0.0f) {
            m_noiseDiffFilter = std::make_unique<NVEncFilterRtgmcPrimitive>();
            auto param = std::make_shared<NVEncFilterParamRtgmcPrimitive>();
            param->frameIn = currentFrame;
            param->frameOut = currentFrame;
            param->baseFps = currentFps;
            param->bOutOverwrite = false;
            param->op = RGYRtgmcPrimitiveOp::MakeDiff;
            param->processChroma = prm->rtgmc.noise.chromaNoise;
            param->planes = prm->rtgmc.noise.chromaNoise ? 0x07 : 0x01;
            auto sts = m_noiseDiffFilter->init(param, m_pLog);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        {
            auto filter = std::make_unique<NVEncFilterRtgmcEdi>();
            auto param = std::make_shared<NVEncFilterParamRtgmcEdi>();
            param->mode = prm->rtgmc.edi.mode;
            param->chromaEdi = prm->rtgmc.edi.chromaEdi;
            param->nnsize = prm->rtgmc.edi.nnsize;
            param->nneurons = prm->rtgmc.edi.nneurons;
            param->ediqual = prm->rtgmc.edi.ediqual;
            param->order = prm->rtgmc.bob.order;
            param->sourceFrameIn = currentFrame;
            param->sourceBaseFps = prm->baseFps;
            param->sourceTimebase = prm->timebase;
            auto sts = initOne(std::move(filter), param);
            if (sts != RGY_ERR_NONE) return sts;
            sts = initRetouchCompFilters(prm, currentFrame, currentFps);
            if (sts != RGY_ERR_NONE) return sts;
        }
        if (rtgmcInputTypeBlendEnabled(prm->rtgmc)) {
#if RGY_HAS_RTGMC_MMASK_FILTER
            auto filter = std::make_unique<NVEncFilterRtgmcMMask>();
            auto param = std::make_shared<NVEncFilterParamRtgmcMMask>();
            param->kind = 1;
            param->ml = prm->rtgmc.progSADMask;
            param->gamma = prm->rtgmc.progSADMaskGamma;
            param->time = 100;
            auto sts = initOne(std::move(filter), param);
            if (sts != RGY_ERR_NONE) return sts;
            AddMessage(RGY_LOG_DEBUG, _T("InputTypeBlend is enabled: input_type=%d, prog_sad_mask=%.3f, gamma=%.3f.\n"),
                prm->rtgmc.inputType, prm->rtgmc.progSADMask, prm->rtgmc.progSADMaskGamma);
#else
            return RGY_ERR_UNSUPPORTED;
#endif
        } else {
            auto sts = initBypass();
            if (sts != RGY_ERR_NONE) return sts;
        }
    }
    const RGYFrameInfo rtgmcSourceFrameIn = currentFrame;
    {
        auto filter = std::make_unique<NVEncFilterDegrain>();
        auto param = std::make_shared<NVEncFilterParamDegrain>();
        param->degrain = rtgDegrainRuntimeParam(prm->rtgmc.tr1, _T("tr1"));
        param->degrain.mode = VppDegrainMode::Degrain;
        param->degrain.stage = VppDegrainStage::TR1;
        param->zeroCopyCache = true;
        auto sts = initOne(std::move(filter), param);
        if (sts != RGY_ERR_NONE) return sts;
    }
    {
        auto filter = std::make_unique<NVEncFilterRtgmcShimmerRepair>();
        auto param = std::make_shared<NVEncFilterParamRtgmcShimmerRepair>();
        param->stage = RGYRtgmcShimmerRepairStage::PreRetouch;
        param->repairThin = prm->rtgmc.rep1.repThin;
        param->repairPad = prm->rtgmc.rep1.repPad;
        param->processChroma = prm->rtgmc.rep1.repChroma;
        auto sts = initOne(std::move(filter), param);
        if (sts != RGY_ERR_NONE) return sts;
    }
    {
        auto filter = std::make_unique<NVEncFilterRtgmcLossless>();
        auto param = std::make_shared<NVEncFilterParamRtgmcLossless>();
        param->level = 2;
        param->inputType = (prm->rtgmc.lossless == 2) ? prm->rtgmc.inputType : 0;
        param->sourceField = 0;
        auto sts = initOne(std::move(filter), param);
        if (sts != RGY_ERR_NONE) return sts;
    }
    {
        auto filter = std::make_unique<NVEncFilterRtgmcRetouch>();
        auto param = std::make_shared<NVEncFilterParamRtgmcRetouch>();
        param->rtgmc_retouch = prm->rtgmc.retouch;
        param->skipPostTR2LimitModes = true;
        param->rtgmc_retouch.tr1 = prm->rtgmc.tr1.delta;
        param->rtgmc_retouch.tr2 = prm->rtgmc.tr2.delta;
        auto sts = initOne(std::move(filter), param);
        if (sts != RGY_ERR_NONE) return sts;
    }
    {
        RGY_ERR sts = RGY_ERR_NONE;
        if (prm->rtgmc.noise.grainRestore > 0.0f) {
            auto filter = std::make_unique<NVEncFilterRtgmcPrimitive>();
            auto param = std::make_shared<NVEncFilterParamRtgmcPrimitive>();
            param->op = RGYRtgmcPrimitiveOp::AddWeightedDiff;
            param->weight = clamp(prm->rtgmc.noise.grainRestore, 0.0f, 1.0f);
            param->processChroma = prm->rtgmc.noise.chromaNoise;
            param->planes = prm->rtgmc.noise.chromaNoise ? 0x07 : 0x01;
            sts = initOne(std::move(filter), param);
        } else {
            sts = initBypass();
        }
        if (sts != RGY_ERR_NONE) return sts;
    }
    {
        if (prm->rtgmc.tr2.delta <= 0) {
            auto sts = initBypass();
            if (sts != RGY_ERR_NONE) return sts;
            AddMessage(RGY_LOG_DEBUG, _T("TR2 Degrain stage is skipped because tr2 delta=%d.\n"), prm->rtgmc.tr2.delta);
        } else {
            auto filter = std::make_unique<NVEncFilterDegrain>();
            auto param = std::make_shared<NVEncFilterParamDegrain>();
            param->degrain = rtgDegrainRuntimeParam(prm->rtgmc.tr2, _T("tr2"));
            param->degrain.mode = VppDegrainMode::Degrain;
            param->degrain.stage = VppDegrainStage::TR2;
            param->zeroCopyCache = true;
            auto sts = initOne(std::move(filter), param);
            if (sts != RGY_ERR_NONE) return sts;
        }
    }
    {
        auto filter = std::make_unique<NVEncFilterRtgmcShimmerRepair>();
        auto param = std::make_shared<NVEncFilterParamRtgmcShimmerRepair>();
        param->stage = RGYRtgmcShimmerRepairStage::PostTR2;
        param->repairThin = prm->rtgmc.rep2.repThin;
        param->repairPad = prm->rtgmc.rep2.repPad;
        param->processChroma = prm->rtgmc.rep2.repChroma;
        auto sts = initOne(std::move(filter), param);
        if (sts != RGY_ERR_NONE) return sts;
    }
    {
        auto filter = std::make_unique<NVEncFilterRtgmcRetouch>();
        auto param = std::make_shared<NVEncFilterParamRtgmcRetouch>();
        param->rtgmc_retouch = prm->rtgmc.retouch;
        param->rtgmc_retouch.smode = 0;
        param->rtgmc_retouch.sharpness = 0.0f;
        param->rtgmc_retouch.svthin = 0.0f;
        param->rtgmc_retouch.sbb = 0;
        param->rtgmc_retouch.slmode = m_enablePostTR2Limit ? prm->rtgmc.retouch.slmode : 0;
        param->rtgmc_retouch.tr1 = prm->rtgmc.tr1.delta;
        param->rtgmc_retouch.tr2 = prm->rtgmc.tr2.delta;
        auto sts = m_enablePostTR2Limit ? initOne(std::move(filter), param) : initBypass();
        if (sts != RGY_ERR_NONE) return sts;
        if (m_enablePostTR2Limit) {
            AddMessage(RGY_LOG_DEBUG, _T("post-TR2 retouch limit stage is enabled for slmode=%d.\n"), prm->rtgmc.retouch.slmode);
        }
    }
    {
        auto filter = std::make_unique<NVEncFilterRtgmcLossless>();
        auto param = std::make_shared<NVEncFilterParamRtgmcLossless>();
        param->level = 1;
        param->inputType = (prm->rtgmc.lossless == 1) ? prm->rtgmc.inputType : 0;
        param->sourceField = 0;
        auto sts = initOne(std::move(filter), param);
        if (sts != RGY_ERR_NONE) return sts;
    }
    {
        RGY_ERR sts = RGY_ERR_NONE;
        if (prm->rtgmc.noise.noiseRestore > 0.0f) {
            auto filter = std::make_unique<NVEncFilterRtgmcPrimitive>();
            auto param = std::make_shared<NVEncFilterParamRtgmcPrimitive>();
            param->op = RGYRtgmcPrimitiveOp::AddWeightedDiff;
            param->weight = clamp(prm->rtgmc.noise.noiseRestore, 0.0f, 1.0f);
            param->processChroma = prm->rtgmc.noise.chromaNoise;
            param->planes = prm->rtgmc.noise.chromaNoise ? 0x07 : 0x01;
            sts = initOne(std::move(filter), param);
        } else {
            sts = initBypass();
        }
        if (sts != RGY_ERR_NONE) return sts;
    }

    {
        auto sts = initSourceMatchCorrectionFilters(prm, rtgmcSourceFrameIn, currentFrame, prm->baseFps, prm->timebase, currentFps);
        if (sts != RGY_ERR_NONE) return sts;
    }

    prm->frameOut = currentFrame;
    if (prm->rtgmc.border) {
        prm->frameOut.height = prm->frameIn.height;
    }
    prm->baseFps = currentFps;
    auto sts = AllocFrameBuf(prm->frameOut, RGY_RTGMC_MAX_OUT_FRAMES);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to allocateoutput buffer: %s.\n"), get_err_mes(sts));
        return sts;
    }
    for (int i = 0; i < RGY_CSP_PLANES[m_frameBuf[0]->frame.csp]; i++) {
        prm->frameOut.pitch[i] = m_frameBuf[0]->frame.pitch[i];
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmc::init(shared_ptr<NVEncFilterParam> pParam, shared_ptr<RGYLog> pPrintMes) {
    m_pLog = pPrintMes;
    auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmc>(pParam);
    auto sts = checkParam(prm);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    m_sharedAnalysisMode = prm->sharedAnalysisMode;
    close();
    m_param = prm;
    sts = initFilters(prm);
    if (sts != RGY_ERR_NONE) {
        close();
        return sts;
    }
    setFilterInfo(prm->print());
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmc::runSourceMatchCorrectionPass(int stageIdx, RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    *pOutputFrameNum = 0;
    if (stageIdx < 0 || stageIdx >= (int)m_matchCorrectionPasses.size()) {
        return RGY_ERR_INVALID_PARAM;
    }
    auto &pass = m_matchCorrectionPasses[stageIdx];
    if (!pass.correctionBuild) {
        if (pInputFrame && pInputFrame->ptr[0]) {
            ppOutputFrames[(*pOutputFrameNum)++] = pInputFrame;
        }
        return RGY_ERR_NONE;
    }

    auto runAddDiff = [&](RGYFrameInfo *diffFrame, const RGYFrameInfo *fallbackBaseFrame, const RGYFrameInfo *propFrame, const std::vector<RGYCudaEvent> &addWaitEvents,
        RGYFrameInfo **addOutFrames, int *addOutFrameNum, RGYCudaEvent *addEvent) {
        auto baseRef = findMatchCorrectionBaseReference(stageIdx, diffFrame);
        const RGYFrameInfo *baseFrame = (baseRef && baseRef->frame) ? &baseRef->frame->frame : fallbackBaseFrame;
        if (!baseFrame || !baseFrame->ptr[0]) {
            addOutFrames[(*addOutFrameNum)++] = diffFrame;
            return RGY_ERR_NONE;
        }
        auto waits = addWaitEvents;
        if (baseRef && baseRef->event() != nullptr) {
            waits.push_back(baseRef->event);
        }
        int outNum = 0;
        RGYFrameInfo *outFrames[RGY_RTGMC_MAX_OUT_FRAMES] = { 0 };
        auto sts = pass.correctionApply->run_filter(baseFrame, diffFrame, outFrames, &outNum, stream, waits, addEvent);
        clearMatchCorrectionBaseReference(stageIdx, diffFrame);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        for (int i = 0; i < outNum; i++) {
            copyFramePropWithoutRes(outFrames[i], propFrame ? propFrame : diffFrame);
            addOutFrames[(*addOutFrameNum)++] = outFrames[i];
        }
        return RGY_ERR_NONE;
    };

    if (!pInputFrame || !pInputFrame->ptr[0]) {
        if (!pass.correctionTemporalFilter) {
            return RGY_ERR_NONE;
        }
        NVEncFilterDegrain *analyze = nullptr;
        if (m_sharedAnalysisMode && m_sharedData.analyzeFilter) {
            analyze = m_sharedData.analyzeFilter;
        } else {
            analyze = dynamic_cast<NVEncFilterDegrain *>(m_filters[RTGMC_FILTER_ANALYZE].get());
        }
        if (analyze) {
            pass.correctionTemporalFilter->setDirectAnalyzeResultSet(analyze->analyzeResultSet());
        }
        int diffOutNum = 0;
        RGYFrameInfo *diffOutFrames[RGY_RTGMC_MAX_OUT_FRAMES] = { 0 };
        RGYFrameInfo drainFrame;
        RGYCudaEvent diffEvent;
        auto sts = rtgmcRunFilterWithEvents(pass.correctionTemporalFilter.get(), &drainFrame, diffOutFrames, &diffOutNum, stream, {}, &diffEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        std::vector<RGYCudaEvent> addWaitEvents;
        if (diffEvent() != nullptr) {
            addWaitEvents.push_back(diffEvent);
        }
        for (int i = 0; i < diffOutNum; i++) {
            sts = runAddDiff(diffOutFrames[i], nullptr, nullptr, addWaitEvents, ppOutputFrames, pOutputFrameNum, event);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        return RGY_ERR_NONE;
    }

    auto sourceWaitEvents = wait_events;
    const RGYFrameInfo *sourceFrame = findCachedSourceFrame(pInputFrame, &sourceWaitEvents);
    if (!sourceFrame) {
        AddMessage(RGY_LOG_ERROR, _T("source-match correction stage %d source frame is missing for inputFrameId=%d.\n"), stageIdx + 1, pInputFrame->inputFrameId);
        return RGY_ERR_INVALID_CALL;
    }

    int sourceOutNum = 0;
    RGYFrameInfo *sourceOutFrames[RGY_RTGMC_MAX_OUT_FRAMES] = { 0 };
    RGYCudaEvent sourceEvent;
    auto sts = rtgmcWaitEvents(stream, sourceWaitEvents);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    sts = pass.interpolator->run_filter(pInputFrame, sourceFrame, sourceFrame, sourceOutFrames, &sourceOutNum, stream);
    if (sts == RGY_ERR_NONE) {
        sts = rtgmcRecordEvent(stream, &sourceEvent);
    }
    if (sts != RGY_ERR_NONE || sourceOutNum <= 0) {
        return sts;
    }
    std::vector<RGYCudaEvent> diffWaitEvents;
    if (sourceEvent() != nullptr) {
        diffWaitEvents.push_back(sourceEvent);
    }

    int diffOutNum = 0;
    RGYFrameInfo *diffOutFrames[RGY_RTGMC_MAX_OUT_FRAMES] = { 0 };
    RGYCudaEvent diffEvent;
    sts = pass.correctionBuild->run_filter(sourceOutFrames[0], pInputFrame, diffOutFrames, &diffOutNum, stream, diffWaitEvents, &diffEvent);
    if (sts != RGY_ERR_NONE || diffOutNum <= 0) {
        return sts;
    }
    int adjustedOutNum = 0;
    RGYFrameInfo *adjustedOutFrames[RGY_RTGMC_MAX_OUT_FRAMES] = { 0 };
    RGYCudaEvent adjustedEvent;
    if (pass.correctionSpatialPrepass) {
        std::vector<RGYCudaEvent> adjustWaitEvents;
        if (diffEvent() != nullptr) {
            adjustWaitEvents.push_back(diffEvent);
        }
        sts = rtgmcRunFilterWithEvents(pass.correctionSpatialPrepass.get(), diffOutFrames[0], adjustedOutFrames, &adjustedOutNum, stream, adjustWaitEvents, &adjustedEvent);
        if (sts != RGY_ERR_NONE || adjustedOutNum <= 0) {
            return sts;
        }
    } else {
        adjustedOutNum = diffOutNum;
        for (int i = 0; i < adjustedOutNum; i++) {
            adjustedOutFrames[i] = diffOutFrames[i];
        }
        adjustedEvent = diffEvent;
    }

    std::vector<RGYCudaEvent> addWaitEvents;
    if (adjustedEvent() != nullptr) {
        addWaitEvents.push_back(adjustedEvent);
    }
    if (pass.fusedCorrectionApply) {
        for (int i = 0; i < adjustedOutNum; i++) {
            copyFramePropWithoutRes(adjustedOutFrames[i], pInputFrame);
            ppOutputFrames[(*pOutputFrameNum)++] = adjustedOutFrames[i];
        }
        if (event && adjustedEvent() != nullptr) {
            *event = adjustedEvent;
        }
    } else if (pass.correctionTemporalFilter) {
        NVEncFilterDegrain *analyze = nullptr;
        if (m_sharedAnalysisMode && m_sharedData.analyzeFilter) {
            analyze = m_sharedData.analyzeFilter;
        } else {
            analyze = dynamic_cast<NVEncFilterDegrain *>(m_filters[RTGMC_FILTER_ANALYZE].get());
        }
        if (analyze) {
            pass.correctionTemporalFilter->setDirectAnalyzeResultSet(analyze->analyzeResultSet());
        }
        sts = storeMatchCorrectionBaseReference(stageIdx, adjustedOutFrames[0], pInputFrame, stream, wait_events);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        int degrainOutNum = 0;
        RGYFrameInfo *degrainOutFrames[RGY_RTGMC_MAX_OUT_FRAMES] = { 0 };
        RGYCudaEvent degrainEvent;
        sts = rtgmcRunFilterWithEvents(pass.correctionTemporalFilter.get(), adjustedOutFrames[0], degrainOutFrames, &degrainOutNum, stream, addWaitEvents, &degrainEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        addWaitEvents.clear();
        if (degrainEvent() != nullptr) {
            addWaitEvents.push_back(degrainEvent);
        }
        for (int i = 0; i < degrainOutNum; i++) {
            sts = runAddDiff(degrainOutFrames[i], pInputFrame, degrainOutFrames[i], addWaitEvents, ppOutputFrames, pOutputFrameNum, event);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
    } else {
        sts = storeMatchCorrectionBaseReference(stageIdx, adjustedOutFrames[0], pInputFrame, stream, wait_events);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = runAddDiff(adjustedOutFrames[0], pInputFrame, pInputFrame, addWaitEvents, ppOutputFrames, pOutputFrameNum, event);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
    }
    if (pass.correctionEnhance && *pOutputFrameNum > 0) {
        int enhanceOutNum = 0;
        RGYFrameInfo *enhanceOutFrames[RGY_RTGMC_MAX_OUT_FRAMES] = { 0 };
        RGYCudaEvent enhanceEvent;
        auto enhanceWaitEvents = (event && (*event)() != nullptr) ? std::vector<RGYCudaEvent>{ *event } : std::vector<RGYCudaEvent>();
        sts = rtgmcRunFilterWithEvents(pass.correctionEnhance.get(), ppOutputFrames[0], enhanceOutFrames, &enhanceOutNum, stream, enhanceWaitEvents, &enhanceEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        *pOutputFrameNum = 0;
        for (int i = 0; i < enhanceOutNum; i++) {
            ppOutputFrames[(*pOutputFrameNum)++] = enhanceOutFrames[i];
        }
        if (event && enhanceEvent() != nullptr) {
            *event = enhanceEvent;
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmc::runSourceMatchCorrectionPasses(int firstStage, int lastStage, RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    *pOutputFrameNum = 0;
    RGYFrameInfo *stageInFrames[RGY_RTGMC_MAX_OUT_FRAMES] = { pInputFrame };
    int stageInNum = (pInputFrame && pInputFrame->ptr[0]) ? 1 : 0;
    std::vector<RGYCudaEvent> stageWaitEvents = wait_events;
    RGYCudaEvent stageEvent;
    for (int stageIdx = firstStage; stageIdx <= lastStage; stageIdx++) {
        RGYFrameInfo *stageOutFrames[RGY_RTGMC_MAX_OUT_FRAMES] = { 0 };
        int stageOutNum = 0;
        if (stageInNum == 0) {
            auto sts = runSourceMatchCorrectionPass(stageIdx, nullptr, stageOutFrames, &stageOutNum, stream, {}, &stageEvent);
            if (sts != RGY_ERR_NONE) return sts;
        } else {
            for (int i = 0; i < stageInNum; i++) {
                int oneOutNum = 0;
                RGYFrameInfo *oneOutFrames[RGY_RTGMC_MAX_OUT_FRAMES] = { 0 };
                auto sts = runSourceMatchCorrectionPass(stageIdx, stageInFrames[i], oneOutFrames, &oneOutNum, stream, stageWaitEvents, &stageEvent);
                if (sts != RGY_ERR_NONE) return sts;
                for (int j = 0; j < oneOutNum; j++) {
                    if (stageOutNum >= RGY_RTGMC_MAX_OUT_FRAMES) {
                        AddMessage(RGY_LOG_ERROR, _T("too many source-match correction output frames.\n"));
                        return RGY_ERR_UNKNOWN;
                    }
                    stageOutFrames[stageOutNum++] = oneOutFrames[j];
                }
            }
        }
        stageInNum = stageOutNum;
        for (int i = 0; i < stageOutNum; i++) {
            stageInFrames[i] = stageOutFrames[i];
        }
        stageWaitEvents.clear();
        if (stageEvent() != nullptr) {
            stageWaitEvents.push_back(stageEvent);
        }
    }
    *pOutputFrameNum = stageInNum;
    for (int i = 0; i < stageInNum; i++) {
        ppOutputFrames[i] = stageInFrames[i];
    }
    if (event && stageEvent() != nullptr) {
        *event = stageEvent;
    }
    return RGY_ERR_NONE;
}

static std::vector<RGYCudaEvent> rtgmcPropagateWaitEvents(const std::vector<RGYCudaEvent> &wait_events, const RGYCudaEvent &event) {
    if (event() != nullptr) {
        return { event };
    }
    return wait_events;
}

static RGY_ERR rtgmcBypassFilter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    *pOutputFrameNum = 0;
    ppOutputFrames[0] = nullptr;
    if (!pInputFrame || !pInputFrame->ptr[0]) {
        return RGY_ERR_NONE;
    }
    ppOutputFrames[(*pOutputFrameNum)++] = pInputFrame;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmc::runNestedFilter(size_t filterIdx, RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    const bool hasInputFrame = pInputFrame && pInputFrame->ptr[0];
    if (filterIdx < m_filters.size() && !m_filters[filterIdx]) {
        return rtgmcBypassFilter(pInputFrame, ppOutputFrames, pOutputFrameNum);
    }
    if (filterIdx == RTGMC_FILTER_EDI) {
        auto edi = dynamic_cast<NVEncFilterRtgmcEdi *>(m_filters[filterIdx].get());
        if (!edi) {
            AddMessage(RGY_LOG_ERROR, _T("InvalidEDI filter instance.\n"));
            return RGY_ERR_INVALID_CALL;
        }
        auto ediWaitEvents = wait_events;
        const RGYFrameInfo *pSourceInputFrame = hasInputFrame ? findCachedSourceFrame(pInputFrame, &ediWaitEvents) : nullptr;
        auto sts = rtgmcWaitEvents(stream, ediWaitEvents);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        sts = edi->run_filter(pInputFrame, pInputFrame, pSourceInputFrame, ppOutputFrames, pOutputFrameNum, stream);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        return rtgmcRecordEvent(stream, event);
    }

    if (filterIdx == RTGMC_FILTER_NOISE) {
        if (m_noiseFilter) {
            return rtgmcRunFilterWithEvents(m_noiseFilter, pInputFrame, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
        }
        return rtgmcRunFilterWithEvents(m_filters[filterIdx].get(), pInputFrame, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
    }

    if (filterIdx == RTGMC_FILTER_ADD_NOISE1 || filterIdx == RTGMC_FILTER_ADD_NOISE2) {
        if (!hasInputFrame) {
            *pOutputFrameNum = 0;
            return RGY_ERR_NONE;
        }
        auto addNoise = dynamic_cast<NVEncFilterRtgmcPrimitive *>(m_filters[filterIdx].get());
        const auto addNoiseParam = addNoise ? dynamic_cast<const NVEncFilterParamRtgmcPrimitive *>(addNoise->GetFilterParam()) : nullptr;
        if (!addNoise || !addNoiseParam) {
            AddMessage(RGY_LOG_ERROR, _T("Invalidadd-noise filter instance.\n"));
            return RGY_ERR_INVALID_CALL;
        }
        if (addNoiseParam->op != RGYRtgmcPrimitiveOp::AddWeightedDiff) {
            return rtgmcRunFilterWithEvents(m_filters[filterIdx].get(), pInputFrame, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
        }
        auto addWaitEvents = wait_events;
        const RGYFrameInfo *noiseFrame = nullptr;
        const auto attachedNoise = rtgmcGetAttachedNoise(pInputFrame);
        if (attachedNoise && attachedNoise->frame() && attachedNoise->frame()->ptr[0]) {
            noiseFrame = attachedNoise->frame();
            if (attachedNoise->event()() != nullptr) {
                addWaitEvents.push_back(attachedNoise->event());
            }
        } else {
            auto noiseRef = findNoiseReference(pInputFrame);
            if (noiseRef && noiseRef->frame && noiseRef->frame->frame.ptr[0]) {
                noiseFrame = &noiseRef->frame->frame;
                if (noiseRef->event() != nullptr) {
                    addWaitEvents.push_back(noiseRef->event);
                }
            }
        }
        if (!noiseFrame) {
            AddMessage(RGY_LOG_DEBUG, _T("noise restore reference is missing; passing the frame through.\n"));
            *pOutputFrameNum = 0;
            ppOutputFrames[(*pOutputFrameNum)++] = pInputFrame;
            return RGY_ERR_NONE;
        }
        return addNoise->run_filter(pInputFrame, noiseFrame, ppOutputFrames, pOutputFrameNum, stream, addWaitEvents, event);
    }

    if (filterIdx == RTGMC_FILTER_INPUTTYPE_BLEND) {
        const auto rtgmcParam = std::dynamic_pointer_cast<NVEncFilterParamRtgmc>(m_param);
        if (!rtgmcParam || !rtgmcInputTypeBlendEnabled(rtgmcParam->rtgmc)) {
            return rtgmcRunFilterWithEvents(m_filters[filterIdx].get(), pInputFrame, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
        }
        if (!hasInputFrame) {
            return rtgmcRunFilterWithEvents(m_filters[filterIdx].get(), pInputFrame, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
        }
#if RGY_HAS_RTGMC_MMASK_FILTER
        auto inputTypeBlend = dynamic_cast<NVEncFilterRtgmcMMask *>(m_filters[filterIdx].get());
        auto analyze = dynamic_cast<NVEncFilterDegrain *>(m_filters[RTGMC_FILTER_ANALYZE].get());
        if (!inputTypeBlend || !analyze) {
            AddMessage(RGY_LOG_ERROR, _T("InvalidInputTypeBlend filter instance.\n"));
            return RGY_ERR_INVALID_CALL;
        }
        auto blendWaitEvents = wait_events;
        const RGYFrameInfo *pSourceInputFrame = findCachedSourceFrame(pInputFrame, &blendWaitEvents);
        if (!pSourceInputFrame) {
            AddMessage(RGY_LOG_ERROR, _T("InputTypeBlend source frame is missing for inputFrameId=%d.\n"), pInputFrame->inputFrameId);
            return RGY_ERR_INVALID_CALL;
        }
        NVEncFilterDegrain *analyzeForBlend = analyze;
        if (m_sharedAnalysisMode && m_sharedData.analyzeFilter) {
            analyzeForBlend = m_sharedData.analyzeFilter;
        }
        auto analyzeResult = rgy_degrain_get_analyze_result(pInputFrame);
        if (!analyzeResult.valid()) {
            analyzeResult = analyzeForBlend->analyzeResult();
        }
        if (analyzeResult.valid()
            && (analyzeResult.inputFrameId != pInputFrame->inputFrameId || analyzeResult.timestamp != pInputFrame->timestamp)) {
            AddMessage(RGY_LOG_ERROR, _T("InputTypeBlend MV/SAD result does not match the frame identity.\n"));
            return RGY_ERR_INVALID_CALL;
        }
        return inputTypeBlend->run_filter(pSourceInputFrame, pInputFrame, analyzeResult,
            ppOutputFrames, pOutputFrameNum, stream, blendWaitEvents, event);
#else
        AddMessage(RGY_LOG_ERROR, _T("rtgmc-mmask filter is not available in this build.\n"));
        return RGY_ERR_UNSUPPORTED;
#endif
    }

    if (filterIdx == RTGMC_FILTER_TR1 || filterIdx == RTGMC_FILTER_TR2) {
        if (filterIdx == RTGMC_FILTER_TR2) {
            const auto rtgmcParam = std::dynamic_pointer_cast<NVEncFilterParamRtgmc>(m_param);
            if (rtgmcParam && rtgmcParam->rtgmc.tr2.delta <= 0) {
                return rtgmcRunFilterWithEvents(m_filters[filterIdx].get(), pInputFrame, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
            }
        }
        NVEncFilterDegrain *analyze = nullptr;
        if (m_sharedAnalysisMode && m_sharedData.analyzeFilter) {
            analyze = m_sharedData.analyzeFilter;
        } else {
            analyze = dynamic_cast<NVEncFilterDegrain *>(m_filters[RTGMC_FILTER_ANALYZE].get());
        }
        auto degrain = dynamic_cast<NVEncFilterDegrain *>(m_filters[filterIdx].get());
        if (!analyze || !degrain) {
            AddMessage(RGY_LOG_ERROR, _T("Invaliddegrain filter instance.\n"));
            return RGY_ERR_INVALID_CALL;
        }
        if (!degrain->setDirectAnalyzeResultSet(analyze->analyzeResultSet())) {
            AddMessage(RGY_LOG_DEBUG, _T("direct MV/SAD result is not ready; degrain will use its fallback path.\n"));
        }
        return rtgmcRunFilterWithEvents(m_filters[filterIdx].get(), pInputFrame, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
    }

    if (filterIdx == RTGMC_FILTER_PRE_RETOUCH_SHIMMER_REPAIR || filterIdx == RTGMC_FILTER_POST_TR2_SHIMMER_REPAIR) {
        if (!hasInputFrame) {
            return rtgmcRunFilterWithEvents(m_filters[filterIdx].get(), pInputFrame, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
        }
        auto repair = dynamic_cast<NVEncFilterRtgmcShimmerRepair *>(m_filters[filterIdx].get());
        if (!repair) {
            AddMessage(RGY_LOG_ERROR, _T("Invalid shimmer repair filter instance.\n"));
            return RGY_ERR_INVALID_CALL;
        }
        auto repairWaitEvents = wait_events;
        attachStoredEdiReference(pInputFrame, &repairWaitEvents);
        const auto edi = rtgmcGetAttachedEdi(pInputFrame);
        if (!edi || !edi->frame()) {
            AddMessage(RGY_LOG_WARN, _T("rtgmc shimmer repair stage %d EDI reference is missing; falling back to single-input repair path.\n"),
                filterIdx == RTGMC_FILTER_PRE_RETOUCH_SHIMMER_REPAIR ? 1 : 2);
            return rtgmcRunFilterWithEvents(m_filters[filterIdx].get(), pInputFrame, ppOutputFrames, pOutputFrameNum, stream, repairWaitEvents, event);
        }
        return repair->run_filter(pInputFrame, edi->frame(), ppOutputFrames, pOutputFrameNum, stream, repairWaitEvents, event);
    }

    if (filterIdx == RTGMC_FILTER_FINAL_SOURCE_FIELD_RESTORE || filterIdx == RTGMC_FILTER_PRE_RETOUCH_SOURCE_FIELD_RESTORE) {
        const auto rtgmcParam = std::dynamic_pointer_cast<NVEncFilterParamRtgmc>(m_param);
        const int activeLevel = rtgmcParam ? rtgmcParam->rtgmc.lossless : 0;
        const int filterLevel = (filterIdx == RTGMC_FILTER_FINAL_SOURCE_FIELD_RESTORE) ? 1 : 2;
        if (activeLevel != filterLevel) {
            *pOutputFrameNum = 0;
            if (hasInputFrame) {
                ppOutputFrames[(*pOutputFrameNum)++] = pInputFrame;
            }
            return RGY_ERR_NONE;
        }
        if (!hasInputFrame) {
            return rtgmcRunFilterWithEvents(m_filters[filterIdx].get(), pInputFrame, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
        }
        auto lossless = dynamic_cast<NVEncFilterRtgmcLossless *>(m_filters[filterIdx].get());
        if (!lossless) {
            AddMessage(RGY_LOG_ERROR, _T("Invalidlossless filter instance.\n"));
            return RGY_ERR_INVALID_CALL;
        }
        auto losslessWaitEvents = wait_events;
        const RGYFrameInfo *pSourceInputFrame = findCachedSourceFrame(pInputFrame, &losslessWaitEvents);
        if (!pSourceInputFrame) {
            AddMessage(RGY_LOG_ERROR, _T("Lossless=%d source frame is missing for inputFrameId=%d.\n"),
                filterLevel, pInputFrame->inputFrameId);
            return RGY_ERR_INVALID_CALL;
        }
        return lossless->run_filter(pInputFrame, pSourceInputFrame, sourceFieldForFrame(pInputFrame),
            ppOutputFrames, pOutputFrameNum, stream, losslessWaitEvents, event);
    }

    if (filterIdx == RTGMC_FILTER_RETOUCH || filterIdx == RTGMC_FILTER_POST_TR2_LIMIT) {
        if (!hasInputFrame) {
            return rtgmcRunFilterWithEvents(m_filters[filterIdx].get(), pInputFrame, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
        }
        auto retouch = dynamic_cast<NVEncFilterRtgmcRetouch *>(m_filters[filterIdx].get());
        if (!retouch) {
            AddMessage(RGY_LOG_ERROR, _T("Invalidretouch filter instance.\n"));
            return RGY_ERR_INVALID_CALL;
        }
        RGYRtgmcRetouchTemporalLimitFrames temporalLimit;
        auto repairWaitEvents = wait_events;
        const auto rtgmcParam = std::dynamic_pointer_cast<NVEncFilterParamRtgmc>(m_param);
        const int slmode = rtgmcParam ? rtgmcParam->rtgmc.retouch.slmode : 0;
        if (filterIdx == RTGMC_FILTER_POST_TR2_LIMIT) {
            const auto baseRef = findPostLimitBaseReference(pInputFrame);
            if (baseRef && baseRef->frame) {
                retouch->setSpatialLimitBaseFrame(&baseRef->frame->frame);
                if (baseRef->event() != nullptr) {
                    repairWaitEvents.push_back(baseRef->event);
                }
            } else if (m_enablePostTR2Limit) {
                retouch->clearSpatialLimitBaseFrame();
                AddMessage(RGY_LOG_WARN, _T("post-TR2 slmode=%d base frame is missing; using current frame as limit base.\n"),
                    slmode);
            }
        }
        if (filterIdx == RTGMC_FILTER_RETOUCH || slmode == 3 || slmode == 4) {
            attachStoredEdiReference(pInputFrame, &repairWaitEvents);
            const auto edi = rtgmcGetAttachedEdi(pInputFrame);

            bool usedInlineComp = false;
            if (filterIdx == RTGMC_FILTER_RETOUCH) {
                auto compRef = findStoredCompReference(pInputFrame);
                if (compRef && compRef->hasInlineParams && edi && edi->frame()) {
                    std::array<RGYDegrainCompensateInlineParams, 3> combinedParams = compRef->backwardInlineParams;
                    for (int p = 0; p < 3; p++) {
                        combinedParams[p].refForw = compRef->forwardInlineParams[p].refBack;
                        combinedParams[p].refDirForw = compRef->forwardInlineParams[p].refDirBack;
                    }
                    const bool inlineCompChroma = rtgmcParam ? rtgmcParam->rtgmc.tr1.chroma : true;
                    retouch->setTemporalLimitInlineComp(edi->frame(), combinedParams, inlineCompChroma);
                    usedInlineComp = true;
                }
            }

            if (!usedInlineComp) {
                attachStoredCompReferences(pInputFrame, &repairWaitEvents);
                const auto motionBack = rtgmcGetAttachedComp(pInputFrame, RGYRtgmcCompDirection::Backward, 1);
                const auto motionForw = rtgmcGetAttachedComp(pInputFrame, RGYRtgmcCompDirection::Forward, 1);
                if (filterIdx == RTGMC_FILTER_POST_TR2_LIMIT && slmode == 3 && edi && edi->frame()) {
                    retouch->setSpatialLimitBaseFrame(edi->frame());
                }
                temporalLimit.ref = edi ? edi->frame() : nullptr;
                temporalLimit.motionBack = motionBack ? motionBack->frame() : temporalLimit.ref;
                temporalLimit.motionForw = motionForw ? motionForw->frame() : temporalLimit.ref;
                if (temporalLimit.valid()) {
                    retouch->setTemporalLimitFrames(temporalLimit);
                } else {
                    retouch->clearTemporalLimitFrames();
                    if (m_attachRetouchCompRefs) {
                        AddMessage(RGY_LOG_WARN, _T("retouch slmode=%d temporal references are incomplete; using retouch fallback for this frame.\n"),
                            slmode);
                    }
                }
            }
        }
        auto sts = rtgmcRunFilterWithEvents(retouch, pInputFrame, ppOutputFrames, pOutputFrameNum, stream, repairWaitEvents, event);
        retouch->clearTemporalLimitFrames();
        retouch->clearSpatialLimitBaseFrame();
        return sts;
    }

    return rtgmcRunFilterWithEvents(m_filters[filterIdx].get(), pInputFrame, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
}

RGY_ERR NVEncFilterRtgmc::runThrough(size_t filterIdx, RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event, bool storePending) {
    if (filterIdx >= m_filters.size()) {
        if (m_pendingOutputFrames.size() >= m_frameBuf.size()) {
            AddMessage(RGY_LOG_ERROR, _T("Too many pending output frames fromchain.\n"));
            return RGY_ERR_UNKNOWN;
        }
        if (!storePending && *pOutputFrameNum >= RGY_RTGMC_MAX_OUT_FRAMES) {
            AddMessage(RGY_LOG_ERROR, _T("Too many output frames fromchain.\n"));
            return RGY_ERR_UNKNOWN;
        }
        const int outputIndex = m_outputBufferIndex;
        m_outputBufferIndex = (m_outputBufferIndex + 1) % (int)m_frameBuf.size();
        auto pOut = &m_frameBuf[outputIndex]->frame;
        auto err = copyFinalOutput(pOut, pInputFrame, stream, wait_events, event);
        if (err != RGY_ERR_NONE) {
            AddMessage(RGY_LOG_ERROR, _T("failed to copyfinal output: %s.\n"), get_err_mes(err));
            return err;
        }
        copyFramePropWithoutRes(pOut, pInputFrame);
        eraseRtgmcInternalFrameData(pOut);
        clearStoredEdiReference(pOut);
        clearStoredCompReferences(pOut);
        clearPostLimitBaseReference(pOut);
        clearNoiseReference(pOut);
        for (int stage = 0; stage < (int)m_matchCorrectionPasses.size(); stage++) {
            clearMatchCorrectionBaseReference(stage, pOut);
        }
        const auto rtgmcParam = std::dynamic_pointer_cast<NVEncFilterParamRtgmc>(m_param);
        if (rtgmcParam && rtgmcParam->rtgmc.sourceMatch > 0) {
            err = applySourceMatchFrameProp(pOut);
            if (err != RGY_ERR_NONE) {
                return err;
            }
        }
        if (storePending) {
            m_pendingOutputFrames.push_back(outputIndex);
        } else {
            ppOutputFrames[(*pOutputFrameNum)++] = pOut;
        }
        return RGY_ERR_NONE;
    }

    int childOutFrameNum = 0;
    RGYFrameInfo *childOutFrames[RGY_RTGMC_MAX_OUT_FRAMES] = { 0 };
    RGYCudaEvent childEvent;
    auto sts = runNestedFilter(filterIdx, pInputFrame, childOutFrames, &childOutFrameNum, stream, wait_events, &childEvent);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    auto childWaitEvents = rtgmcPropagateWaitEvents(wait_events, childEvent);
    for (int i = 0; i < childOutFrameNum; i++) {
        propagateRtgmcInternalFrameData(childOutFrames[i], pInputFrame);
        auto nextWaitEvents = childWaitEvents;
        attachStoredEdiReference(childOutFrames[i], &nextWaitEvents);
        if (filterIdx == RTGMC_FILTER_NOISE) {
            auto noiseWaitEvents = wait_events;
            if (childEvent() != nullptr) {
                noiseWaitEvents.push_back(childEvent);
            }
            sts = storeNoiseReference(pInputFrame, childOutFrames[i], stream, noiseWaitEvents);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        if (filterIdx == RTGMC_FILTER_EDI) {
            // Rep1/Rep2 and retouch do not yet expose a stable explicit ref-input API.
            // Keep the edi reference on the frame itself so those workers can switch to it
            // (or to a future direct API) without touching the nested chain again.
            RGYCudaEvent ediRefEvent;
            sts = attachEdiReference(childOutFrames[i], stream, childWaitEvents, &ediRefEvent);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            if (ediRefEvent() != nullptr) {
                nextWaitEvents.push_back(ediRefEvent);
            }
            sts = updateCompReferenceStore(childOutFrames[i], stream, childWaitEvents);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        } else if (filterIdx == RTGMC_FILTER_TR1) {
            attachStoredCompReferences(childOutFrames[i], &nextWaitEvents);
        }
        if (filterIdx == RTGMC_FILTER_RETOUCH) {
            sts = storePostLimitBaseReference(childOutFrames[i], stream, childWaitEvents);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        const auto rtgmcParam = std::dynamic_pointer_cast<NVEncFilterParamRtgmc>(m_param);
        if (rtgmcParam && rtgmcParam->rtgmc.sourceMatch >= 1 && filterIdx == RTGMC_FILTER_PRE_RETOUCH_SHIMMER_REPAIR) {
            int smOutNum = 0;
            RGYFrameInfo *smOutFrames[RGY_RTGMC_MAX_OUT_FRAMES] = { 0 };
            RGYCudaEvent smEvent;
            sts = runSourceMatchCorrectionPasses(0, rtgmcParam->rtgmc.sourceMatch - 1, childOutFrames[i], smOutFrames, &smOutNum, stream, nextWaitEvents, &smEvent);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            auto smWaitEvents = rtgmcPropagateWaitEvents(nextWaitEvents, smEvent);
            for (int j = 0; j < smOutNum; j++) {
                enqueueSourceMatchFrameProp(smOutFrames[j]);
                if (filterIdx == RTGMC_FILTER_INPUTTYPE_BLEND && m_captureIntermediate) {
                    auto buf = getSharedFrameBuffer(smOutFrames[j]);
                    if (buf) {
                        copyFrameAsync(&buf->frame, smOutFrames[j], stream);
                        copyFramePropWithoutRes(&buf->frame, smOutFrames[j]);
                        RGYCudaEvent captureEvent;
                        rtgmcRecordEvent(stream, &captureEvent);
                        auto frameInfo = buf->frame;
                        frameInfo.dataList.push_back(std::make_shared<RGYFrameDataRtgmcFrameRef>(buf));
                        m_capturedIntermediates.push_back({buf, frameInfo, captureEvent});
                    }
                }
                sts = runThrough(filterIdx + 1, smOutFrames[j], ppOutputFrames, pOutputFrameNum, stream, smWaitEvents, event, storePending);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }
        } else {
            if (filterIdx == RTGMC_FILTER_INPUTTYPE_BLEND && m_captureIntermediate) {
                auto buf = getSharedFrameBuffer(childOutFrames[i]);
                if (buf) {
                    copyFrameAsync(&buf->frame, childOutFrames[i], stream);
                    copyFramePropWithoutRes(&buf->frame, childOutFrames[i]);
                    RGYCudaEvent captureEvent;
                    rtgmcRecordEvent(stream, &captureEvent);
                    auto frameInfo = buf->frame;
                    frameInfo.dataList.push_back(std::make_shared<RGYFrameDataRtgmcFrameRef>(buf));
                    m_capturedIntermediates.push_back({buf, frameInfo, captureEvent});
                }
            }
            sts = runThrough(filterIdx + 1, childOutFrames[i], ppOutputFrames, pOutputFrameNum, stream, nextWaitEvents, event, storePending);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmc::drainFrom(size_t filterIdx, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, RGYCudaEvent *event) {
    if (m_drainComplete) {
        return RGY_ERR_NONE;
    }
    if (m_drainFilterIdx < filterIdx) {
        m_drainFilterIdx = filterIdx;
    }

    auto processDrainOutputs = [this, ppOutputFrames, pOutputFrameNum, &stream, event](
        const size_t currentFilterIdx,
        RGYFrameInfo **childOutFrames,
        const int childOutFrameNum,
        const RGYFrameInfo *drainFrame,
        const std::vector<RGYCudaEvent>& childWaitEvents) {
        auto sts = RGY_ERR_NONE;
        for (int i = 0; i < childOutFrameNum; i++) {
            propagateRtgmcInternalFrameData(childOutFrames[i], drainFrame);
            auto nextWaitEvents = childWaitEvents;
            attachStoredEdiReference(childOutFrames[i], &nextWaitEvents);
            if (currentFilterIdx == RTGMC_FILTER_NOISE) {
                sts = storeNoiseReference(drainFrame, childOutFrames[i], stream, childWaitEvents);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }
            if (currentFilterIdx == RTGMC_FILTER_EDI) {
                RGYCudaEvent ediRefEvent;
                sts = attachEdiReference(childOutFrames[i], stream, childWaitEvents, &ediRefEvent);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                if (ediRefEvent() != nullptr) {
                    nextWaitEvents.push_back(ediRefEvent);
                }
                sts = updateCompReferenceStore(childOutFrames[i], stream, childWaitEvents);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            } else if (currentFilterIdx == RTGMC_FILTER_TR1) {
                attachStoredCompReferences(childOutFrames[i], &nextWaitEvents);
            }
            if (currentFilterIdx == RTGMC_FILTER_RETOUCH) {
                sts = storePostLimitBaseReference(childOutFrames[i], stream, childWaitEvents);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }
            const auto rtgmcParam = std::dynamic_pointer_cast<NVEncFilterParamRtgmc>(m_param);
            if (rtgmcParam && rtgmcParam->rtgmc.sourceMatch >= 1 && currentFilterIdx == RTGMC_FILTER_PRE_RETOUCH_SHIMMER_REPAIR) {
                int smOutNum = 0;
                RGYFrameInfo *smOutFrames[RGY_RTGMC_MAX_OUT_FRAMES] = { 0 };
                RGYCudaEvent smEvent;
                sts = runSourceMatchCorrectionPasses(0, rtgmcParam->rtgmc.sourceMatch - 1, childOutFrames[i], smOutFrames, &smOutNum, stream, nextWaitEvents, &smEvent);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
                auto smWaitEvents = rtgmcPropagateWaitEvents(nextWaitEvents, smEvent);
                for (int j = 0; j < smOutNum; j++) {
                    enqueueSourceMatchFrameProp(smOutFrames[j]);
                    sts = runThrough(currentFilterIdx + 1, smOutFrames[j], ppOutputFrames, pOutputFrameNum, stream, smWaitEvents, event, true);
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                }
            } else {
                sts = runThrough(currentFilterIdx + 1, childOutFrames[i], ppOutputFrames, pOutputFrameNum, stream, nextWaitEvents, event, true);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }
        }
        return sts;
    };

    while (m_drainFilterIdx < m_filters.size()) {
        int childOutFrameNum = 0;
        RGYFrameInfo *childOutFrames[RGY_RTGMC_MAX_OUT_FRAMES] = { 0 };
        RGYFrameInfo drainFrame;
        RGYCudaEvent childEvent;
        auto sts = runNestedFilter(m_drainFilterIdx, &drainFrame, childOutFrames, &childOutFrameNum, stream, {}, &childEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        auto childWaitEvents = rtgmcPropagateWaitEvents({}, childEvent);
        if (childOutFrameNum > 0) {
            return processDrainOutputs(m_drainFilterIdx, childOutFrames, childOutFrameNum, &drainFrame, childWaitEvents);
        }
        if (m_drainFilterIdx == RTGMC_FILTER_EDI) {
            sts = drainCompReferenceStore(stream);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        const auto rtgmcParam = std::dynamic_pointer_cast<NVEncFilterParamRtgmc>(m_param);
        if (rtgmcParam && rtgmcParam->rtgmc.sourceMatch >= 1 && m_drainFilterIdx == RTGMC_FILTER_PRE_RETOUCH_SHIMMER_REPAIR) {
            int smOutNum = 0;
            RGYFrameInfo *smOutFrames[RGY_RTGMC_MAX_OUT_FRAMES] = { 0 };
            RGYCudaEvent smEvent;
            sts = runSourceMatchCorrectionPasses(0, rtgmcParam->rtgmc.sourceMatch - 1, nullptr, smOutFrames, &smOutNum, stream, {}, &smEvent);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
            if (smOutNum > 0) {
                auto smWaitEvents = rtgmcPropagateWaitEvents({}, smEvent);
                for (int i = 0; i < smOutNum; i++) {
                    enqueueSourceMatchFrameProp(smOutFrames[i]);
                    sts = runThrough(m_drainFilterIdx + 1, smOutFrames[i], ppOutputFrames, pOutputFrameNum, stream, smWaitEvents, event, true);
                    if (sts != RGY_ERR_NONE) {
                        return sts;
                    }
                }
                return RGY_ERR_NONE;
            }
        }
        m_drainFilterIdx++;
    }
    m_drainComplete = true;
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmc::returnPendingFrames(RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum) {
    *pOutputFrameNum = 0;
    while (!m_pendingOutputFrames.empty() && *pOutputFrameNum < RGY_RTGMC_MAX_RETURN_FRAMES) {
        const int outputIndex = m_pendingOutputFrames.front();
        m_pendingOutputFrames.pop_front();
        ppOutputFrames[(*pOutputFrameNum)++] = &m_frameBuf[outputIndex]->frame;
    }
    if (*pOutputFrameNum <= 0) {
        ppOutputFrames[0] = nullptr;
    }
    return RGY_ERR_NONE;
}

RGY_ERR NVEncFilterRtgmc::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream) {
    return run_filter(pInputFrame, ppOutputFrames, pOutputFrameNum, stream, {}, nullptr);
}

RGY_ERR NVEncFilterRtgmc::filter(RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    return run_filter(pInputFrame, ppOutputFrames, pOutputFrameNum, stream, wait_events, event);
}

RGY_ERR NVEncFilterRtgmc::run_filter(const RGYFrameInfo *pInputFrame, RGYFrameInfo **ppOutputFrames, int *pOutputFrameNum,
    cudaStream_t stream, const std::vector<RGYCudaEvent> &wait_events, RGYCudaEvent *event) {
    *pOutputFrameNum = 0;
    if (pInputFrame == nullptr || pInputFrame->ptr[0] == nullptr) {
        m_draining = true;
        if (!m_pendingOutputFrames.empty()) {
            return returnPendingFrames(ppOutputFrames, pOutputFrameNum);
        }
        if (m_sharedAnalysisMode) {
            while (!m_pendingIntermediateInputs.empty()) {
                auto intermediate = std::move(m_pendingIntermediateInputs.front());
                m_pendingIntermediateInputs.pop_front();
                std::vector<RGYCudaEvent> intWaitEvents;
                if (intermediate.event() != nullptr) {
                    intWaitEvents.push_back(intermediate.event);
                }
                auto sts = runThrough(RTGMC_FILTER_TR1, &intermediate.frameInfo, ppOutputFrames, pOutputFrameNum, stream, intWaitEvents, event, false);
                if (sts != RGY_ERR_NONE) {
                    return sts;
                }
            }
        }
        const size_t drainStartIdx = m_sharedAnalysisMode ? RTGMC_FILTER_TR1 : 0;
        while (m_pendingOutputFrames.empty() && !m_drainComplete) {
            auto sts = drainFrom(drainStartIdx, ppOutputFrames, pOutputFrameNum, stream, event);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        return returnPendingFrames(ppOutputFrames, pOutputFrameNum);
    }
    const auto rtgmcParam = std::dynamic_pointer_cast<NVEncFilterParamRtgmc>(m_param);
    const RGYFrameInfo *chainInputFrame = pInputFrame;
    std::vector<RGYCudaEvent> chainWaitEvents = wait_events;
    RGYCudaEvent borderEvent;
    if (rtgmcParam && rtgmcParam->rtgmc.border) {
        auto sts = addBorderToInput(pInputFrame, stream, wait_events, &borderEvent);
        if (sts != RGY_ERR_NONE) {
            return sts;
        }
        chainInputFrame = &m_borderFrame->frame;
        chainWaitEvents.clear();
        if (borderEvent() != nullptr) {
            chainWaitEvents.push_back(borderEvent);
        }
    }
    // Debug hook: NVENC_RTGMC_DEBUG_RESET_AT
    if (m_debugResetAtFrame == 0) {
        // First valid input frame: parse env var once
        const char *envVal = std::getenv("NVENC_RTGMC_DEBUG_RESET_AT");
        if (envVal && envVal[0] != '\0') {
            m_debugResetAtFrame = std::atoi(envVal);
            if (m_debugResetAtFrame <= 0) m_debugResetAtFrame = -1; // disable
        } else {
            m_debugResetAtFrame = -1; // env not set -> disable
        }
    }
    if (m_debugResetAtFrame > 0 && m_nFrame == m_debugResetAtFrame) {
        fprintf(stderr, "[NVEncFilterRtgmc] resetTemporalState triggered at frame %d\n", m_nFrame);
        resetTemporalState();
        m_debugResetAtFrame = -1; // one-shot: resetTemporalState() clears m_nFrame, so disarm to avoid re-firing
    }
    m_nFrame++;

    m_drainFilterIdx = 0;
    m_draining = false;
    m_drainComplete = false;
    auto sts = cacheSourceFrame(chainInputFrame, stream, chainWaitEvents);
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    m_inputFrame = *chainInputFrame;
    if (m_sharedAnalysisMode) {
        *pOutputFrameNum = 0;
        while (!m_pendingIntermediateInputs.empty()) {
            auto intermediate = std::move(m_pendingIntermediateInputs.front());
            m_pendingIntermediateInputs.pop_front();
            std::vector<RGYCudaEvent> intWaitEvents = chainWaitEvents;
            if (intermediate.event() != nullptr) {
                intWaitEvents.push_back(intermediate.event);
            }
            sts = runThrough(RTGMC_FILTER_TR1, &intermediate.frameInfo, ppOutputFrames, pOutputFrameNum, stream, intWaitEvents, event, false);
            if (sts != RGY_ERR_NONE) {
                return sts;
            }
        }
        return RGY_ERR_NONE;
    }
    return runThrough(0, &m_inputFrame, ppOutputFrames, pOutputFrameNum, stream, chainWaitEvents, event, false);
}

bool NVEncFilterRtgmc::draining() const {
    return m_draining;
}

bool NVEncFilterRtgmc::drainComplete() const {
    return m_draining && m_drainComplete && m_pendingOutputFrames.empty();
}

int NVEncFilterRtgmc::requiredPrimingSourceFrames() const {
    const auto prm = std::dynamic_pointer_cast<NVEncFilterParamRtgmc>(m_param);
    if (!prm) {
        return 8;
    }
    const auto& rtgmc = prm->rtgmc;
    const int temporalRadius = std::max({
        std::max(0, rtgmc.searchPrefilter.tr0),
        std::max(0, rtgmc.analyze.delta),
        std::max(0, rtgmc.tr1.delta),
        std::max(0, rtgmc.tr2.delta),
        std::max(0, rtgmc.matchTR1),
        std::max(0, rtgmc.matchTR2)
    });
    const int sourceMatchMargin = std::max(0, rtgmc.sourceMatch) * std::max({ 1, std::max(0, rtgmc.matchTR1), std::max(0, rtgmc.matchTR2) });
    return std::max(8, temporalRadius + sourceMatchMargin + 4);
}

void NVEncFilterRtgmc::resetTemporalState() {
    // Reset time-dependent state only. Filter objects, GPU buffers, and kernel programs are preserved.

    // 1. Nested RTGMC sub-filters
    for (auto &filter : m_filters) {
        if (filter) filter->resetTemporalState();
    }
    for (auto &filter : m_retouchCompFilters) {
        if (filter) filter->resetTemporalState();
    }
    for (auto &stage : m_matchCorrectionPasses) {
        if (stage.interpolator) stage.interpolator->resetTemporalState();
        if (stage.correctionBuild) stage.correctionBuild->resetTemporalState();
        if (stage.correctionSpatialPrepass) stage.correctionSpatialPrepass->resetTemporalState();
        if (stage.correctionTemporalFilter) stage.correctionTemporalFilter->resetTemporalState();
        if (stage.correctionApply) stage.correctionApply->resetTemporalState();
        if (stage.correctionEnhance) stage.correctionEnhance->resetTemporalState();
        stage.composeBaseRefs.clear();
    }
    if (m_noiseDiffFilter) m_noiseDiffFilter->resetTemporalState();

    // 2. Pending reference queues (EDI, comp, post-limit, noise)
    m_pendingEdiRefs.clear();
    m_pendingCompRefs.clear();
    m_pendingPostLimitBaseRefs.clear();
    m_pendingNoiseRefs.clear();
    m_pendingOutputFrames.clear();
    m_pendingSourceMatchFrameProps.clear();
    if (m_ediSideDataFramePool) {
        m_ediSideDataFramePool->clear();
    }

    // 3. Source cache: clear key/event metadata but keep the GPU frame buffers so they
    //    can be re-used; set a fresh sourceCacheNext to overwrite from index 0.
    for (auto &frame : m_sourceCache) {
        frame.key = RtgmcFrameKey();
        frame.event.reset();
        // frame.frame (GPU buffer) is preserved intentionally
    }
    m_sourceCacheNext = 0;

    // 4. Output / drain state
    m_outputBufferIndex = 0;
    m_drainFilterIdx = 0;
    m_draining = false;
    m_drainComplete = false;

    // 5. Intermediate capture queue (debug only — safe to discard)
    m_capturedIntermediates.clear();
    m_pendingIntermediateInputs.clear();

    // 6. Last input frame identity
    m_inputFrame = RGYFrameInfo();
    // 7. Input frame counter (used by debug hook)
    m_nFrame = 0;
}

void NVEncFilterRtgmc::close() {
    m_filters.clear();
    m_noiseFilter = nullptr;
    for (auto &filter : m_retouchCompFilters) {
        filter.reset();
    }
    for (auto &stage : m_matchCorrectionPasses) {
        stage = RtgmcMatchCorrectionPass();
    }
    m_pendingEdiRefs.clear();
    m_pendingCompRefs.clear();
    m_pendingPostLimitBaseRefs.clear();
    m_noiseDiffFilter.reset();
    m_pendingNoiseRefs.clear();
    m_pendingOutputFrames.clear();
    m_pendingSourceMatchFrameProps.clear();
    if (m_sharedFramePool) {
        m_sharedFramePool->clear();
    }
    if (m_ediSideDataFramePool) {
        m_ediSideDataFramePool->clear();
    }
    for (auto &frame : m_sourceCache) {
        frame.key = RtgmcFrameKey();
        frame.frame.reset();
        frame.event.reset();
    }
    m_borderFrame.reset();
    m_frameBuf.clear();
    m_sourceCacheNext = 0;
    m_outputBufferIndex = 0;
    m_drainFilterIdx = 0;
    m_draining = false;
    m_drainComplete = false;
    m_attachRetouchCompRefs = false;
    m_enablePostTR2Limit = false;
    m_capturedIntermediates.clear();
    m_pendingIntermediateInputs.clear();
    m_captureIntermediate = false;
    m_nFrame = 0;
    m_debugResetAtFrame = 0;
}
